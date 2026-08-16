#!/usr/bin/env python3
"""Find public companies with comparable earnings behaviour.

``search_comparable`` uses web research to discover possible peers and then
applies a deterministic, evidence-aware ranking.  The live implementation uses
the OpenAI Responses API with web search; callers can inject a researcher for
tests or for another data provider.

The result is research, not investment advice.  In particular, a company's
presence on a public "competitors" page is treated only as a discovery signal.
Revenue, cost/COGS and EPS similarity must be supported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import posixpath
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "comparable.xlsx"
DEFAULT_MODEL = "gpt-5.6-sol"
METRIC_WEIGHTS = {"revenue": 0.30, "costs": 0.20, "eps": 0.25}
OTHER_WEIGHTS = {"macro_similarity_score": 0.15, "business_similarity_score": 0.07,
                 "scale_similarity_score": 0.03}
SOURCE_TYPES = {
    "issuer",
    "regulator",
    "institutional_dataset",
    "reputable_news",
    "peer_list",
    "other",
}
OFFICIAL_SOURCE_TYPES = {"issuer", "regulator"}
APPLIES_TO = {"target", "candidate", "both"}
DIRECT_ASSESSMENTS = {"useful", "limited", "rejected"}
METRIC_ALIASES = {
    "sales": "revenue",
    "total revenue": "revenue",
    "net sales": "revenue",
    "revenue": "revenue",
    "cost": "costs",
    "cogs": "costs",
    "cost of goods sold": "costs",
    "cost of sales": "costs",
    "gross margin": "costs",
    "costs": "costs",
    "eps": "eps",
    "diluted eps": "eps",
    "adjusted eps": "eps",
}
INVALID_SHEET_CHARACTERS = re.compile(r"[\\/*?:\[\]]")
SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ElementTree.register_namespace("", SPREADSHEET_NS)
ElementTree.register_namespace("r", RELATIONSHIP_NS)


def _bounded_score(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number from 0 to 100")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError(f"{field} must be a finite number from 0 to 100")
    return number


def _domain(url: str) -> str:
    return urlparse(url).netloc.casefold().removeprefix("www.")


def _clean_source(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    url = str(value.get("url") or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    source_type = str(value.get("source_type") or "other").strip().casefold()
    if source_type not in SOURCE_TYPES:
        source_type = "other"
    applies_to = str(value.get("applies_to") or "both").strip().casefold()
    if applies_to not in APPLIES_TO:
        applies_to = "both"
    return {
        "name": str(value.get("name") or _domain(url)).strip(),
        "url": url,
        "published_date": str(value.get("published_date") or "").strip(),
        "source_type": source_type,
        "applies_to": applies_to,
        "claim": str(value.get("claim") or "").strip(),
    }


def _clean_sources(value: object, *, allow_peer_lists: bool = True) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        source = _clean_source(item)
        if source is None or (not allow_peer_lists and source["source_type"] == "peer_list"):
            continue
        key = (source["url"], source["applies_to"], source["claim"])
        if key not in seen:
            result.append(source)
            seen.add(key)
    return result


def _canonical_metric(value: object) -> str | None:
    key = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    return METRIC_ALIASES.get(key)


def _metric_evidence(sources: Sequence[Mapping[str, str]]) -> dict[str, object]:
    target = any(source["applies_to"] in {"target", "both"} for source in sources)
    candidate = any(source["applies_to"] in {"candidate", "both"} for source in sources)
    official_target = any(
        source["source_type"] in OFFICIAL_SOURCE_TYPES
        and source["applies_to"] in {"target", "both"}
        for source in sources
    )
    official_candidate = any(
        source["source_type"] in OFFICIAL_SOURCE_TYPES
        and source["applies_to"] in {"candidate", "both"}
        for source in sources
    )
    return {
        "covers_both_companies": target and candidate,
        "official_for_both": official_target and official_candidate,
        "independent_domains": len({_domain(source["url"]) for source in sources}),
    }


def _clean_metric(value: object) -> tuple[str, dict[str, object]] | None:
    if not isinstance(value, Mapping):
        return None
    metric = _canonical_metric(value.get("metric"))
    if metric is None:
        return None
    sources = _clean_sources(value.get("sources"), allow_peer_lists=False)
    evidence = _metric_evidence(sources)
    raw_score = _bounded_score(value.get("similarity_score"), f"{metric}.similarity_score")
    # A score without evidence for each company is a hypothesis, not a validated
    # earnings comparison.  Retain it for auditability but do not rank on it.
    ranking_score = raw_score if evidence["covers_both_companies"] else 0.0
    return metric, {
        "target_metric": str(value.get("target_metric") or "").strip(),
        "candidate_metric": str(value.get("candidate_metric") or "").strip(),
        "similarity_score": round(raw_score, 2),
        "ranking_score": round(ranking_score, 2),
        "comparison": str(value.get("comparison") or "").strip(),
        "period_compared": str(value.get("period_compared") or "").strip(),
        "evidence": evidence,
        "sources": sources,
    }


def _clean_direct_sources(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        source = _clean_source(item.get("source"))
        if source is None:
            continue
        source["source_type"] = "peer_list"
        methodology = str(item.get("methodology") or "").strip()
        assessment = str(item.get("assessment") or "limited").strip().casefold()
        if assessment not in DIRECT_ASSESSMENTS:
            assessment = "limited"
        # A direct list with no disclosed or researched methodology is never
        # promoted to "useful", even if the upstream researcher called it so.
        if assessment == "useful" and not methodology:
            assessment = "limited"
        result.append(
            {
                "source": source,
                "methodology": methodology,
                "assessment": assessment,
                "validation_notes": str(item.get("validation_notes") or "").strip(),
            }
        )
    return result


def _confidence(metrics: Mapping[str, Mapping[str, object]]) -> tuple[str, dict[str, object]]:
    covered = sum(bool(item["evidence"]["covers_both_companies"]) for item in metrics.values())
    official = sum(bool(item["evidence"]["official_for_both"]) for item in metrics.values())
    domains = {
        _domain(source["url"])
        for item in metrics.values()
        for source in item["sources"]  # type: ignore[union-attr]
    }
    if covered == 3 and official >= 2 and len(domains) >= 2:
        label = "high"
    elif covered >= 2 and len(domains) >= 2:
        label = "medium"
    else:
        label = "low"
    return label, {
        "metrics_with_both_company_evidence": covered,
        "metrics_with_official_evidence_for_both": official,
        "independent_financial_source_domains": len(domains),
    }


def normalize_candidate(candidate: Mapping[str, object], target_company: str) -> dict[str, object]:
    """Validate one researched candidate and calculate its ranking locally."""
    company = str(candidate.get("company") or "").strip()
    if not company:
        raise ValueError("Comparable candidate is missing a company name")
    if re.sub(r"\W+", "", company).casefold() == re.sub(r"\W+", "", target_company).casefold():
        raise ValueError("The target company cannot be its own comparable")

    metrics: dict[str, dict[str, object]] = {}
    raw_metrics = candidate.get("metrics")
    if isinstance(raw_metrics, Sequence) and not isinstance(raw_metrics, (str, bytes)):
        for raw_metric in raw_metrics:
            cleaned = _clean_metric(raw_metric)
            if cleaned is not None:
                metrics[cleaned[0]] = cleaned[1]

    component_scores = {
        field: _bounded_score(candidate.get(field), field) for field in OTHER_WEIGHTS
    }
    score = sum(
        float(metrics.get(metric, {}).get("ranking_score", 0)) * weight
        for metric, weight in METRIC_WEIGHTS.items()
    )
    score += sum(component_scores[field] * weight for field, weight in OTHER_WEIGHTS.items())
    confidence, evidence_summary = _confidence(metrics)

    relationships = candidate.get("relationship_types")
    if not isinstance(relationships, Sequence) or isinstance(relationships, (str, bytes)):
        relationships = []
    drivers = candidate.get("shared_drivers")
    if not isinstance(drivers, Sequence) or isinstance(drivers, (str, bytes)):
        drivers = []
    caveats = candidate.get("caveats")
    if not isinstance(caveats, Sequence) or isinstance(caveats, (str, bytes)):
        caveats = []

    return {
        "company": company,
        "ticker": str(candidate.get("ticker") or "").strip(),
        "exchange": str(candidate.get("exchange") or "").strip(),
        "industry": str(candidate.get("industry") or "").strip(),
        "score": round(score, 2),
        "confidence": confidence,
        "component_scores": {key: round(value, 2) for key, value in component_scores.items()},
        "evidence_summary": evidence_summary,
        "relationship_types": [str(item).strip() for item in relationships if str(item).strip()],
        "shared_drivers": [str(item).strip() for item in drivers if str(item).strip()],
        "why_comparable": str(candidate.get("why_comparable") or "").strip(),
        "metrics": metrics,
        "direct_comparable_sources": _clean_direct_sources(
            candidate.get("direct_comparable_sources")
        ),
        "caveats": [str(item).strip() for item in caveats if str(item).strip()],
    }


def _source_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "url": {"type": "string"},
            "published_date": {"type": "string"},
            "source_type": {"type": "string", "enum": sorted(SOURCE_TYPES)},
            "applies_to": {"type": "string", "enum": sorted(APPLIES_TO)},
            "claim": {"type": "string"},
        },
        "required": ["name", "url", "published_date", "source_type", "applies_to", "claim"],
    }


def _research_schema() -> dict[str, object]:
    source = _source_schema()
    metric = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metric": {"type": "string", "enum": ["revenue", "costs", "eps"]},
            "target_metric": {"type": "string"},
            "candidate_metric": {"type": "string"},
            "similarity_score": {"type": "number"},
            "comparison": {"type": "string"},
            "period_compared": {"type": "string"},
            "sources": {"type": "array", "items": source},
        },
        "required": [
            "metric", "target_metric", "candidate_metric", "similarity_score",
            "comparison", "period_compared", "sources",
        ],
    }
    direct_source = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source": source,
            "methodology": {"type": "string"},
            "assessment": {"type": "string", "enum": sorted(DIRECT_ASSESSMENTS)},
            "validation_notes": {"type": "string"},
        },
        "required": ["source", "methodology", "assessment", "validation_notes"],
    }
    candidate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "company": {"type": "string"},
            "ticker": {"type": "string"},
            "exchange": {"type": "string"},
            "industry": {"type": "string"},
            "business_similarity_score": {"type": "number"},
            "macro_similarity_score": {"type": "number"},
            "scale_similarity_score": {"type": "number"},
            "relationship_types": {"type": "array", "items": {"type": "string"}},
            "shared_drivers": {"type": "array", "items": {"type": "string"}},
            "why_comparable": {"type": "string"},
            "metrics": {"type": "array", "items": metric},
            "direct_comparable_sources": {"type": "array", "items": direct_source},
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "company", "ticker", "exchange", "industry", "business_similarity_score",
            "macro_similarity_score", "scale_similarity_score", "relationship_types",
            "shared_drivers", "why_comparable", "metrics", "direct_comparable_sources",
            "caveats",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"candidates": {"type": "array", "items": candidate}},
        "required": ["candidates"],
    }


def _research_prompt(company_name: str, candidate_count: int) -> str:
    return f"""Find and validate earnings comparables for the public company {company_name!r}.

Return up to {candidate_count} strongest candidates after researching a broader initial universe.
The purpose is earnings forecasting, not valuation or stock-price correlation.

Candidate discovery:
- Look for public competitor/peer pages or peer APIs when available. Research and state each list's
  selection methodology, freshness and limitations. Such lists are discovery sources only.
- Include same-industry peers and cross-sector companies exposed to genuinely similar macro drivers,
  demand cycles, input costs, customer budgets, regulation, themes or industry trends.

Financial validation:
- Compare at least the latest 8 reported quarters when available, and use up to 5 fiscal years for
  cyclicality. Align fiscal periods and distinguish quarterly from annual data.
- Separately compare (1) sales/total revenue direction and growth pattern, (2) COGS/cost of sales,
  gross-margin or input-cost behaviour, and (3) diluted EPS and adjusted EPS direction/growth.
- Score similarity from 0 to 100 for each metric. Explain the observed pattern; do not score merely
  because both values increased. Use like-for-like definitions, and flag differing adjusted-EPS
  definitions, currencies, fiscal calendars, acquisitions and segment mix.
- Prefer issuer earnings releases and regulator filings. SEC companyfacts/frames are suitable for
  standardized US-GAAP facts. Use reputable news or institutional datasets as corroboration.
- Every metric's sources must cover both the target and candidate; provide separate sources and set
  applies_to accordingly. Never use a peer-list page as financial evidence.
- business_similarity_score, macro_similarity_score and scale_similarity_score are 0-100. Scale is
  based on revenue/earnings scale, not market capitalization alone.
- Do not include the target itself, private companies, ETFs or candidates supported only by a generic
  peer list. Be candid when COGS or adjusted EPS is not comparable.

Direct-source assessment guidance:
- "useful" means methodology is disclosed and reasonably relevant, but still needs validation.
- "limited" means it is usable only to seed candidates (for example same exchange/industry/market cap).
- "rejected" means opaque, stale, circular, price-focused or otherwise unsuitable.
"""


def _extract_response_text(response: Mapping[str, object]) -> str:
    output = response.get("output")
    if isinstance(output, Sequence):
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, Sequence):
                continue
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "output_text":
                    return str(part.get("text") or "")
    raise RuntimeError("OpenAI response did not contain output_text")


def research_with_openai(
    company_name: str,
    candidate_count: int,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> list[Mapping[str, object]]:
    """Run structured comparable-company research with live web search."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for live comparable-company research")
    payload = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "input": _research_prompt(company_name, candidate_count),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "comparable_company_research",
                "strict": True,
                "schema": _research_schema(),
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            response = json.load(handle)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"OpenAI research request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI research request failed: {exc.reason}") from exc
    parsed = json.loads(_extract_response_text(response))
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list) or any(not isinstance(item, Mapping) for item in candidates):
        raise RuntimeError("OpenAI research output did not contain a valid candidates list")
    return candidates


def _call_researcher(
    researcher: object, company_name: str, candidate_count: int
) -> list[Mapping[str, object]]:
    if hasattr(researcher, "research"):
        result = researcher.research(company_name, candidate_count)  # type: ignore[attr-defined]
    elif callable(researcher):
        result = researcher(company_name, candidate_count)
    else:
        raise TypeError("researcher must be callable or provide a research() method")
    if isinstance(result, Mapping):
        result = result.get("candidates")
    if not isinstance(result, list) or any(not isinstance(item, Mapping) for item in result):
        raise ValueError("researcher must return a candidate list or {'candidates': [...]} mapping")
    return result


def research_comparables(
    company_name: str,
    *,
    limit: int = 8,
    min_score: float = 35.0,
    include_low_confidence: bool = False,
    researcher: object | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> list[dict[str, object]]:
    """Research evidence-backed comparables ordered from strongest to weakest.

    Args:
        company_name: Public company name (a ticker may be included to disambiguate).
        limit: Maximum number of returned comparables, from 1 to 20.
        min_score: Deterministic weighted-score floor, from 0 to 100.
        include_low_confidence: Retain candidates with fewer than two evidenced metrics.
        researcher: Optional callable ``(company_name, candidate_count) -> candidates`` or
            object with a matching ``research`` method.  Omit for live web research.
        api_key/model/timeout: Settings for the live OpenAI research path.

    Scores weight revenue 30%, cost/COGS 20%, EPS 25%, shared macro drivers 15%,
    business overlap 7% and earnings scale 3%.  Unsupported metric scores contribute zero.
    """
    company_name = str(company_name).strip()
    if not company_name:
        raise ValueError("company_name cannot be empty")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer from 1 to 20")
    min_score = _bounded_score(min_score, "min_score")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer")

    candidate_count = min(30, max(10, limit * 2))
    if researcher is None:
        researcher = lambda name, count: research_with_openai(
            name, count, api_key=api_key, model=model, timeout=timeout
        )
    raw_candidates = _call_researcher(researcher, company_name, candidate_count)

    candidates = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_candidates:
        candidate = normalize_candidate(raw, company_name)
        identity = (str(candidate["company"]).casefold(), str(candidate["ticker"]).casefold())
        if identity in seen or float(candidate["score"]) < min_score:
            continue
        if not include_low_confidence and candidate["confidence"] == "low":
            continue
        seen.add(identity)
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            float(item["score"]),
            {"high": 2, "medium": 1, "low": 0}[str(item["confidence"])],
            str(item["company"]).casefold(),
        ),
        reverse=True,
    )
    selected = candidates[:limit]
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    return selected


def safe_sheet_name(company_name: str) -> str:
    """Return a legal Excel tab name derived from the original company name."""
    name = INVALID_SHEET_CHARACTERS.sub(" ", company_name)
    name = "".join(character for character in name if ord(character) >= 32)
    name = re.sub(r"\s+", " ", name).strip(" '")
    return (name or "Company")[:31].rstrip()


def _worksheet_xml(company_names: Sequence[str]) -> bytes:
    rows = []
    for row_number, company_name in enumerate(company_names, start=1):
        value = str(company_name).strip()
        if not value:
            continue
        preserve = ' xml:space="preserve"' if value != value.strip() else ""
        rows.append(
            f'<row r="{row_number}" ht="18" customHeight="1">'
            f'<c r="A{row_number}" t="inlineStr"><is><t{preserve}>{escape(value)}</t></is></c>'
            "</row>"
        )
    last_row = max(len(rows), 1)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{SPREADSHEET_NS}">'
        f'<dimension ref="A1:A{last_row}"/>'
        '<sheetViews><sheetView showGridLines="0" workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="18"/>'
        '<cols><col min="1" max="1" width="42" customWidth="1"/></cols>'
        f'<sheetData>{"".join(rows)}</sheetData>'
        '<pageMargins left="0.5" right="0.5" top="0.5" bottom="0.5" '
        'header="0.2" footer="0.2"/>'
        '</worksheet>'
    ).encode("utf-8")


def _new_workbook_parts(sheet_name: str, company_names: Sequence[str]) -> dict[str, bytes]:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{CONTENT_TYPES_NS}">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    root_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PACKAGE_RELATIONSHIP_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    workbook = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{SPREADSHEET_NS}" xmlns:r="{RELATIONSHIP_NS}">
  <bookViews><workbookView/></bookViews>
  <sheets><sheet name={quoteattr(sheet_name)} sheetId="1" r:id="rId1"/></sheets>
  <calcPr calcId="191029"/>
</workbook>'''
    workbook_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PACKAGE_RELATIONSHIP_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="{SPREADSHEET_NS}">
  <fonts count="1"><font><sz val="10"/><name val="Aptos"/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>comparable.py</dc:creator><dc:title>Comparable companies</dc:title>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
</cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>comparable.py</Application></Properties>'''
    return {
        "[Content_Types].xml": content_types.encode("utf-8"),
        "_rels/.rels": root_rels.encode("utf-8"),
        "xl/workbook.xml": workbook.encode("utf-8"),
        "xl/_rels/workbook.xml.rels": workbook_rels.encode("utf-8"),
        "xl/styles.xml": styles.encode("utf-8"),
        "xl/worksheets/sheet1.xml": _worksheet_xml(company_names),
        "docProps/core.xml": core.encode("utf-8"),
        "docProps/app.xml": app.encode("utf-8"),
    }


def _relationship_member(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join("xl", target))


def _serialize_xml(element: ElementTree.Element, default_namespace: str) -> bytes:
    """Serialize OOXML with the default namespace expected by Excel readers."""
    ElementTree.register_namespace("", default_namespace)
    ElementTree.register_namespace("r", RELATIONSHIP_NS)
    rendered = ElementTree.tostring(element, encoding="utf-8", xml_declaration=True)
    ElementTree.register_namespace("", SPREADSHEET_NS)
    return rendered


def _updated_workbook_parts(
    output_path: Path, sheet_name: str, company_names: Sequence[str]
) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(output_path) as archive:
            parts = {item.filename: archive.read(item.filename) for item in archive.infolist()}
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Existing output is not a readable XLSX workbook: {output_path}") from exc

    required = {"[Content_Types].xml", "xl/workbook.xml", "xl/_rels/workbook.xml.rels"}
    if not required <= parts.keys():
        raise ValueError(f"Existing output is missing required XLSX components: {output_path}")

    workbook = ElementTree.fromstring(parts["xl/workbook.xml"])
    relationships = ElementTree.fromstring(parts["xl/_rels/workbook.xml.rels"])
    sheets = workbook.find(f"{{{SPREADSHEET_NS}}}sheets")
    if sheets is None:
        raise ValueError(f"Existing output has no worksheet collection: {output_path}")
    relationship_by_id = {
        item.get("Id", ""): item
        for item in relationships.findall(f"{{{PACKAGE_RELATIONSHIP_NS}}}Relationship")
    }

    existing = next(
        (item for item in sheets if str(item.get("name") or "").casefold() == sheet_name.casefold()),
        None,
    )
    if existing is not None:
        relationship_id = existing.get(f"{{{RELATIONSHIP_NS}}}id", "")
        relationship = relationship_by_id.get(relationship_id)
        if relationship is None:
            raise ValueError(f"Could not resolve worksheet {sheet_name!r} in {output_path}")
        worksheet_member = _relationship_member(str(relationship.get("Target") or ""))
    else:
        sheet_ids = [int(item.get("sheetId", "0")) for item in sheets]
        relationship_numbers = [
            int(match.group(1))
            for item in relationships
            if (match := re.fullmatch(r"rId(\d+)", str(item.get("Id") or "")))
        ]
        worksheet_numbers = [
            int(match.group(1))
            for name in parts
            if (match := re.fullmatch(r"xl/worksheets/sheet(\d+)\.xml", name))
        ]
        new_sheet_id = max(sheet_ids, default=0) + 1
        new_relationship_id = f"rId{max(relationship_numbers, default=0) + 1}"
        new_worksheet_number = max(worksheet_numbers, default=0) + 1
        worksheet_member = f"xl/worksheets/sheet{new_worksheet_number}.xml"
        sheet = ElementTree.SubElement(
            sheets,
            f"{{{SPREADSHEET_NS}}}sheet",
            {"name": sheet_name, "sheetId": str(new_sheet_id)},
        )
        sheet.set(f"{{{RELATIONSHIP_NS}}}id", new_relationship_id)
        ElementTree.SubElement(
            relationships,
            f"{{{PACKAGE_RELATIONSHIP_NS}}}Relationship",
            {
                "Id": new_relationship_id,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
                "Target": f"worksheets/sheet{new_worksheet_number}.xml",
            },
        )
        content_types = ElementTree.fromstring(parts["[Content_Types].xml"])
        ElementTree.SubElement(
            content_types,
            f"{{{CONTENT_TYPES_NS}}}Override",
            {
                "PartName": f"/xl/worksheets/sheet{new_worksheet_number}.xml",
                "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
            },
        )
        parts["[Content_Types].xml"] = _serialize_xml(content_types, CONTENT_TYPES_NS)

    parts["xl/workbook.xml"] = _serialize_xml(workbook, SPREADSHEET_NS)
    parts["xl/_rels/workbook.xml.rels"] = _serialize_xml(
        relationships, PACKAGE_RELATIONSHIP_NS
    )
    parts[worksheet_member] = _worksheet_xml(company_names)
    return parts


def save_comparables(
    company_name: str,
    candidates: Sequence[Mapping[str, object]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save comparable names to the company's tab, preserving every other tab."""
    company_name = str(company_name).strip()
    if not company_name:
        raise ValueError("company_name cannot be empty")
    output = Path(output_path)
    if output.suffix.casefold() != ".xlsx":
        output = output.with_suffix(".xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)
    comparable_names = [
        str(candidate.get("company") or "").strip()
        for candidate in candidates
        if str(candidate.get("company") or "").strip()
    ]
    sheet_name = safe_sheet_name(company_name)
    parts = (
        _updated_workbook_parts(output, sheet_name, comparable_names)
        if output.exists()
        else _new_workbook_parts(sheet_name, comparable_names)
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".xlsx", dir=output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in parts.items():
                archive.writestr(name, content)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def search_comparable(
    company_name: str,
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int = 8,
    min_score: float = 35.0,
    include_low_confidence: bool = False,
    researcher: object | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> Path:
    """Research comparables and update ``data/comparable.xlsx``.

    The original company is used as the tab name.  Comparable company names are
    written to column A, one per row.  Existing tabs are preserved; rerunning the
    same company replaces that company's tab contents.
    """
    candidates = research_comparables(
        company_name,
        limit=limit,
        min_score=min_score,
        include_low_confidence=include_low_confidence,
        researcher=researcher,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
    return save_comparables(company_name, candidates, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("company", help="Public company name, optionally including its ticker")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=35.0)
    parser.add_argument("--include-low-confidence", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Shared XLSX output; defaults to data/comparable.xlsx",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = search_comparable(
        args.company,
        output_path=args.output,
        limit=args.limit,
        min_score=args.min_score,
        include_low_confidence=args.include_low_confidence,
        model=args.model,
        timeout=args.timeout,
    )
    print(f"Updated comparable-company workbook: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
