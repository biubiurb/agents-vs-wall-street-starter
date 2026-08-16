#!/usr/bin/env python3
"""Research earnings-relevant news and save it as an auditable text file.

The live path uses the OpenAI Responses API with web search.  Tests and other
callers can inject a deterministic ``researcher`` callable instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODEL = "gpt-5.6-sol"
DIVIDER = "=" * 88
SOURCE_TYPES = {
    "official_company",
    "government_regulator",
    "major_newswire",
    "national_financial_press",
    "national_general_news",
    "established_broadcaster",
    "established_trade_press",
    "local_or_other",
}
SOURCE_PRIORITY = {
    "official_company": 0,
    "government_regulator": 0,
    "major_newswire": 1,
    "national_financial_press": 1,
    "national_general_news": 2,
    "established_broadcaster": 2,
    "established_trade_press": 3,
    "local_or_other": 4,
}
CATEGORIES = {"company", "industry", "macro"}


def _parse_date(value: str | date, field: str) -> date:
    """Parse an ISO calendar date without accepting ambiguous date formats."""
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        text = str(value).strip()
        try:
            parsed = date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO date in YYYY-MM-DD format") from exc
        if text != parsed.isoformat():
            raise ValueError(f"{field} must be an ISO date in YYYY-MM-DD format")
    return parsed


def _clean_publication_time(value: object) -> tuple[str, date]:
    """Validate a sourced date/timestamp and normalize exact timestamps to UTC."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("published_at is required")
    try:
        published_date = date.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError as exc:
            raise ValueError("published_at must be an ISO date or timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("An exact published_at timestamp must include a timezone")
        parsed = parsed.astimezone(timezone.utc).replace(microsecond=0)
        return parsed.isoformat().replace("+00:00", "Z"), parsed.date()
    if text != published_date.isoformat():
        raise ValueError("published_at must be an ISO date or timestamp")
    return text, published_date


def _clean_url(value: object) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be a valid public HTTP(S) URL")
    return url


def _clean_string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result = []
    for item in value:
        cleaned = re.sub(r"\s+", " ", str(item)).strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def normalize_news_piece(
    piece: Mapping[str, object], start_date: date, end_date: date
) -> dict[str, object] | None:
    """Validate one research result; return ``None`` when it is outside scope."""
    headline = re.sub(r"\s+", " ", str(piece.get("headline") or "")).strip()
    source_name = re.sub(r"\s+", " ", str(piece.get("source_name") or "")).strip()
    if not headline or not source_name:
        raise ValueError("Every news piece needs a headline and source_name")

    published_at, published_date = _clean_publication_time(piece.get("published_at"))
    if not start_date <= published_date <= end_date:
        return None

    impact_score = piece.get("impact_score")
    if isinstance(impact_score, bool) or not isinstance(impact_score, (int, float)):
        raise ValueError("impact_score must be a number from 1 to 5")
    impact_score = float(impact_score)
    if not math.isfinite(impact_score) or not 1 <= impact_score <= 5:
        raise ValueError("impact_score must be a finite number from 1 to 5")

    source_type = str(piece.get("source_type") or "local_or_other").strip().casefold()
    if source_type not in SOURCE_TYPES:
        source_type = "local_or_other"
    category = str(piece.get("category") or "company").strip().casefold()
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of: {', '.join(sorted(CATEGORIES))}")

    statement_lines = _clean_string_list(piece.get("financial_statement_lines"))
    impact_rationale = re.sub(
        r"\s+", " ", str(piece.get("impact_rationale") or "")
    ).strip()
    if not statement_lines or not impact_rationale:
        raise ValueError(
            "Every news piece needs financial_statement_lines and an impact_rationale"
        )

    body_available = bool(piece.get("body_available"))
    content = re.sub(r"\s+", " ", str(piece.get("content") or "")).strip()
    if not content:
        content = headline
        body_available = False

    return {
        "headline": headline,
        "published_at": published_at,
        "source_name": source_name,
        "source_url": _clean_url(piece.get("source_url")),
        "source_type": source_type,
        "category": category,
        "impact_score": impact_score,
        "financial_statement_lines": statement_lines,
        "impact_rationale": impact_rationale,
        "body_available": body_available,
        "content": content,
    }


def _research_schema() -> dict[str, object]:
    piece = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline": {"type": "string"},
            "published_at": {
                "type": "string",
                "pattern": (
                    r"^\d{4}-\d{2}-\d{2}"
                    r"(?:T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2}))?$"
                ),
            },
            "source_name": {"type": "string"},
            "source_url": {"type": "string"},
            "source_type": {"type": "string", "enum": sorted(SOURCE_TYPES)},
            "category": {"type": "string", "enum": sorted(CATEGORIES)},
            "impact_score": {"type": "number"},
            "financial_statement_lines": {
                "type": "array",
                "items": {"type": "string"},
            },
            "impact_rationale": {"type": "string"},
            "body_available": {"type": "boolean"},
            "content": {"type": "string"},
        },
        "required": [
            "headline",
            "published_at",
            "source_name",
            "source_url",
            "source_type",
            "category",
            "impact_score",
            "financial_statement_lines",
            "impact_rationale",
            "body_available",
            "content",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"news": {"type": "array", "items": piece}},
        "required": ["news"],
    }


def _research_prompt(company_name: str, start_date: date, end_date: date, limit: int) -> str:
    return f"""Research financially material news for {company_name!r} published from
{start_date.isoformat()} through {end_date.isoformat()}, inclusive. Return at most {limit} pieces.

Search three lanes deliberately:
1. Company-specific events: demand, pricing, contracts, products, customers, suppliers,
   operations, labour, litigation, regulation, tax, capital allocation, financing, acquisitions
   and divestitures.
2. Industry events that can materially alter the company's volumes, price, mix, input costs or
   competitive position.
3. Macro events with a credible company-specific earnings channel, such as central-bank rate moves,
   tariffs, commodity/energy shocks, fiscal policy, recession indicators or currency moves.

Inclusion rules:
- Include only items likely to have a meaningful (impact score 3-5), non-speculative effect on at
  least one financial-statement line: revenue/sales, COGS, operating expense, depreciation,
  impairment, interest, tax, working capital, capex, debt, cash flow, EPS or another named line.
- Do not include routine market recaps, stock-price moves, analyst rating changes, generic opinion,
  duplicate syndications, low-substance SEO pages, or events whose earnings link is merely possible.
- Prefer primary company/regulator releases, Reuters/AP and other major newswires, established
  financial/national press and respected trade publications. Avoid small unattributed aggregators.
- The publication must fall inside the requested date range. Use the exact ISO 8601 timestamp with
  its timezone when the source exposes one; otherwise use YYYY-MM-DD. Never invent a time.
- A paywall is not grounds for exclusion. If only a headline is available, set body_available false
  and put the headline in content. Otherwise content should be a concise factual summary, not a
  reproduction of the article.
- Describe the concrete earnings transmission mechanism in impact_rationale and name every likely
  financial-statement line. Score 3 for material but indirect, 4 for clearly material, and 5 only
  for unusually consequential events. Do not state a quantified effect unless a source supports it.
- Use category company, industry or macro. Classify source_type conservatively.
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
    start_date: date,
    end_date: date,
    max_results: int,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> list[Mapping[str, object]]:
    """Run structured news research using live web search."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for live news research")
    payload = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "input": _research_prompt(company_name, start_date, end_date, max_results),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "earnings_relevant_news",
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
        raise RuntimeError(f"OpenAI news research request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI news research request failed: {exc.reason}") from exc

    parsed = json.loads(_extract_response_text(response))
    news = parsed.get("news")
    if not isinstance(news, list) or any(not isinstance(item, Mapping) for item in news):
        raise RuntimeError("OpenAI research output did not contain a valid news list")
    return news


def _call_researcher(
    researcher: object, company_name: str, start_date: date, end_date: date
) -> list[Mapping[str, object]]:
    if hasattr(researcher, "research"):
        result = researcher.research(  # type: ignore[attr-defined]
            company_name, start_date, end_date
        )
    elif callable(researcher):
        result = researcher(company_name, start_date, end_date)
    else:
        raise TypeError("researcher must be callable or provide a research() method")
    if isinstance(result, Mapping):
        result = result.get("news")
    if not isinstance(result, list) or any(not isinstance(item, Mapping) for item in result):
        raise ValueError("researcher must return a news list or {'news': [...]} mapping")
    return result


def _identity(piece: Mapping[str, object]) -> tuple[str, str]:
    headline = re.sub(r"[^a-z0-9]+", "", str(piece["headline"]).casefold())
    return headline, str(piece["source_url"]).casefold()


def _rank_news(
    raw_news: Sequence[Mapping[str, object]],
    start_date: date,
    end_date: date,
    min_impact: float,
    max_results: int,
) -> list[dict[str, object]]:
    candidates = []
    for raw in raw_news:
        normalized = normalize_news_piece(raw, start_date, end_date)
        if normalized is not None and float(normalized["impact_score"]) >= min_impact:
            candidates.append(normalized)
    candidates.sort(
        key=lambda item: (
            SOURCE_PRIORITY[str(item["source_type"])],
            -float(item["impact_score"]),
            str(item["published_at"]),
            str(item["headline"]).casefold(),
        )
    )

    selected = []
    seen_headlines: set[str] = set()
    seen_urls: set[str] = set()
    for piece in candidates:
        headline_key, url_key = _identity(piece)
        if headline_key in seen_headlines or url_key in seen_urls:
            continue
        selected.append(piece)
        seen_headlines.add(headline_key)
        seen_urls.add(url_key)
        if len(selected) == max_results:
            break
    return selected


def render_news_file(
    company_name: str,
    start_date: date,
    end_date: date,
    news: Sequence[Mapping[str, object]],
) -> str:
    """Render news pieces with visible, plain-text audit metadata and dividers."""
    lines = [
        f"EARNINGS-RELEVANT NEWS: {company_name}",
        f"DATE RANGE: {start_date.isoformat()} to {end_date.isoformat()} (inclusive)",
        "",
    ]
    if not news:
        lines.append("No qualifying earnings-relevant news was found.")
        return "\n".join(lines).rstrip() + "\n"

    for index, piece in enumerate(news, start=1):
        if index > 1:
            lines.extend(["", DIVIDER, ""])
        score = float(piece["impact_score"])
        score_text = str(int(score)) if score.is_integer() else f"{score:g}"
        availability = (
            "Summary from accessible article" if piece["body_available"] else "Headline only"
        )
        lines.extend(
            [
                f"RELEASE TIME: {piece['published_at']}",
                f"NEWS SOURCE: {piece['source_name']} ({piece['source_type']})",
                f"SOURCE URL: {piece['source_url']}",
                "",
                f"HEADLINE: {piece['headline']}",
                f"CATEGORY: {str(piece['category']).upper()}",
                f"IMPACT SCORE: {score_text}/5",
                "FINANCIAL STATEMENT LINE(S): "
                + ", ".join(str(item) for item in piece["financial_statement_lines"]),
                f"WHY IT COULD AFFECT EARNINGS: {piece['impact_rationale']}",
                f"CONTENT ({availability}): {piece['content']}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _safe_company_filename(company_name: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", company_name, flags=re.UNICODE).strip("._-")
    return value or "company"


def get_news(
    company_name: str,
    start_date: str | date,
    end_date: str | date,
    *,
    output_path: str | Path | None = None,
    min_impact: float = 3.0,
    max_results: int = 40,
    researcher: object | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> Path:
    """Find material company/industry/macro news and save it to a text file.

    ``researcher`` may be a callable taking ``(company_name, start_date, end_date)``
    or an object with a matching ``research`` method.  Omitting it enables live
    OpenAI web research and requires ``OPENAI_API_KEY``.
    """
    company_name = re.sub(r"\s+", " ", str(company_name)).strip()
    if not company_name:
        raise ValueError("company_name cannot be empty")
    first = _parse_date(start_date, "start_date")
    last = _parse_date(end_date, "end_date")
    if first > last:
        raise ValueError("start_date cannot be after end_date")
    if isinstance(min_impact, bool) or not isinstance(min_impact, (int, float)):
        raise ValueError("min_impact must be a number from 1 to 5")
    min_impact = float(min_impact)
    if not math.isfinite(min_impact) or not 1 <= min_impact <= 5:
        raise ValueError("min_impact must be a finite number from 1 to 5")
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 100
    ):
        raise ValueError("max_results must be an integer from 1 to 100")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer")

    if researcher is None:
        researcher = lambda name, start, end: research_with_openai(
            name,
            start,
            end,
            max_results,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
    raw_news = _call_researcher(researcher, company_name, first, last)
    selected = _rank_news(raw_news, first, last, min_impact, max_results)

    if output_path is None:
        filename = (
            f"news_{_safe_company_filename(company_name)}_"
            f"{first.isoformat()}_to_{last.isoformat()}.txt"
        )
        output = DEFAULT_DATA_DIR / filename
    else:
        output = Path(output_path)
        if output.suffix.casefold() != ".txt":
            output = output.with_suffix(".txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_news_file(company_name, first, last, selected), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True, help="Company name, optionally with ticker")
    parser.add_argument("--start", required=True, help="First publication date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Last publication date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, help="Output .txt path; defaults to data/")
    parser.add_argument("--min-impact", type=float, default=3.0, help="Minimum impact score (1-5)")
    parser.add_argument("--max-results", type=int, default=40, help="Maximum pieces (1-100)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model with web search")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = get_news(
            args.company,
            args.start,
            args.end,
            output_path=args.output,
            min_impact=args.min_impact,
            max_results=args.max_results,
            model=args.model,
            timeout=args.timeout,
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote earnings-relevant news: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
