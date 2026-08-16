#!/usr/bin/env python3
"""Collect reporting-period documents and online forward guidance in one text file.

The live researcher uses the OpenAI Responses API with web search. Tests and
other callers can inject a deterministic ``researcher`` callable instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Union
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "challenge" / "companies.json"
DEFAULT_DOCUMENT_ROOT = REPO_ROOT / "challenge" / "offline-data"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODEL = "gpt-5.6-terra"
DIVIDER = "=" * 96


@dataclass(frozen=True, order=True)
class Quarter:
    """A normalized quarter label."""

    year: int
    quarter: int

    def __post_init__(self) -> None:
        if self.year < 1900 or self.year > 2200 or self.quarter not in range(1, 5):
            raise ValueError(f"Invalid quarter: Q{self.quarter} {self.year}")

    @classmethod
    def parse(cls, value: str | "Quarter") -> "Quarter":
        if isinstance(value, cls):
            return value
        text = re.sub(r"[\s_-]+", "", str(value)).upper()
        match = re.fullmatch(r"(?:FY)?(\d{4})Q([1-4])", text)
        if match:
            return cls(int(match.group(1)), int(match.group(2)))
        match = re.fullmatch(r"Q([1-4])(?:FY)?(\d{4})", text)
        if match:
            return cls(int(match.group(2)), int(match.group(1)))
        match = re.fullmatch(r"([1-4])Q(?:FY)?(\d{4})", text)
        if match:
            return cls(int(match.group(2)), int(match.group(1)))
        raise ValueError(
            f"Invalid quarter '{value}'. Use a format such as 'Q1 2026' or 'FY2026Q1'."
        )

    def __str__(self) -> str:
        return f"Q{self.quarter} {self.year}"


@dataclass(frozen=True, order=True)
class Half:
    """A normalized half-year label."""

    year: int
    half: int

    def __post_init__(self) -> None:
        if self.year < 1900 or self.year > 2200 or self.half not in range(1, 3):
            raise ValueError(f"Invalid half: H{self.half} {self.year}")

    @classmethod
    def parse(cls, value: str | "Half") -> "Half":
        if isinstance(value, cls):
            return value
        text = re.sub(r"[\s_-]+", "", str(value)).upper()
        match = re.fullmatch(r"(?:FY)?(\d{4})H([1-2])", text)
        if match:
            return cls(int(match.group(1)), int(match.group(2)))
        match = re.fullmatch(r"H([1-2])(?:FY)?(\d{4})", text)
        if match:
            return cls(int(match.group(2)), int(match.group(1)))
        match = re.fullmatch(r"([1-2])H(?:FY)?(\d{4})", text)
        if match:
            return cls(int(match.group(2)), int(match.group(1)))
        raise ValueError(
            f"Invalid half '{value}'. Use a format such as 'H1 2026' or 'FY2026H1'."
        )

    def __str__(self) -> str:
        return f"H{self.half} {self.year}"


@dataclass(frozen=True, order=True)
class FiscalYear:
    """A normalized full fiscal-year label."""

    year: int

    def __post_init__(self) -> None:
        if self.year < 1900 or self.year > 2200:
            raise ValueError(f"Invalid fiscal year: FY{self.year}")

    @classmethod
    def parse(cls, value: str | "FiscalYear") -> "FiscalYear":
        if isinstance(value, cls):
            return value
        text = re.sub(r"[\s_-]+", "", str(value)).upper()
        match = re.fullmatch(r"FY(\d{4})", text) or re.fullmatch(r"(\d{4})(?:FY)?", text)
        if not match:
            raise ValueError(f"Invalid fiscal year '{value}'. Use a format such as 'FY2026'.")
        return cls(int(match.group(1)))

    def __str__(self) -> str:
        return f"FY{self.year}"


ReportingPeriod = Union[Quarter, Half, FiscalYear]


def parse_reporting_period(value: str | ReportingPeriod) -> ReportingPeriod:
    """Parse a quarter, half-year, or full fiscal-year label."""
    if isinstance(value, (Quarter, Half, FiscalYear)):
        return value
    text = re.sub(r"[\s_-]+", "", str(value)).upper()
    if re.fullmatch(r"(?:FY)?\d{4}(?:FY)?", text):
        return FiscalYear.parse(value)
    if "H" in text:
        return Half.parse(value)
    return Quarter.parse(value)


@dataclass(frozen=True)
class LocalDocument:
    path: Path
    published_at: str
    source_url: str
    document_type: str
    period: str
    title: str
    body: str


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse the JSON-compatible YAML frontmatter used by the offline corpus."""
    if not text.startswith("---\n"):
        return {}, text
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return {}, text

    metadata: dict[str, object] = {}
    for line in text[4:marker].splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        raw_value = raw_value.strip()
        try:
            metadata[key.strip()] = json.loads(raw_value)
        except json.JSONDecodeError:
            metadata[key.strip()] = raw_value
    return metadata, text[marker + 5 :]


def load_company(config_path: Path, selector: str) -> dict[str, object]:
    """Resolve an exact company name or ticker from the challenge config."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    companies = payload.get("companies", [])
    wanted = re.sub(r"\s+", " ", selector).strip().casefold()
    matches = []
    for company in companies:
        ticker = str(company.get("ticker") or "")
        identifiers = {
            str(company.get("company") or "").casefold(),
            ticker.casefold(),
            ticker.rsplit(":", 1)[-1].casefold(),
        }
        if wanted in identifiers:
            matches.append(company)
    if len(matches) != 1:
        available = ", ".join(str(item.get("company")) for item in companies)
        raise ValueError(f"Company '{selector}' was not found. Available companies: {available}")
    return matches[0]


def _document_identity(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(4096)
    metadata, _ = parse_frontmatter(prefix)
    return str(metadata.get("company") or ""), str(metadata.get("ticker") or "")


def find_document_directory(root: Path, company: Mapping[str, object]) -> Path:
    """Find a corpus directory using document metadata instead of folder spelling."""
    expected_name = str(company.get("company") or "").casefold()
    expected_ticker = str(company.get("ticker") or "").casefold()
    if not root.is_dir():
        raise FileNotFoundError(f"Offline document root does not exist: {root}")
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        sample = next(
            (
                path
                for path in directory.rglob("*.md")
                if path.name not in {"INDEX.md", "README.md"}
            ),
            None,
        )
        if sample is None:
            continue
        name, ticker = _document_identity(sample)
        if name.casefold() == expected_name or ticker.casefold() == expected_ticker:
            return directory
    raise ValueError(f"No offline documents found for {company.get('company')} under {root}")


def period_matches(period: object, quarter: str | ReportingPeriod) -> bool:
    """Return whether a metadata period contains the requested quarter/half and year."""
    requested = parse_reporting_period(quarter)
    text = str(period or "").upper()
    if isinstance(requested, FiscalYear):
        return bool(re.search(rf"(?<![A-Z0-9])FY\s*{requested.year}(?!\d)", text))
    kind = "Q" if isinstance(requested, Quarter) else "H"
    number = requested.quarter if isinstance(requested, Quarter) else requested.half
    patterns = (
        rf"(?<![A-Z0-9]){kind}{number}\s*(?:FY\s*)?{requested.year}(?!\d)",
        rf"(?<!\d)(?:FY\s*)?{requested.year}\s*{kind}{number}(?![A-Z0-9])",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def find_quarter_documents(
    directory: Path,
    company: Mapping[str, object],
    quarter: str | ReportingPeriod,
) -> list[LocalDocument]:
    """Load every company corpus document whose period names the quarter."""
    expected_name = str(company.get("company") or "").casefold()
    expected_ticker = str(company.get("ticker") or "").casefold()
    documents = []
    for path in sorted(directory.rglob("*.md")):
        if path.name in {"INDEX.md", "README.md"}:
            continue
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        same_company = (
            str(metadata.get("company") or "").casefold() == expected_name
            or str(metadata.get("ticker") or "").casefold() == expected_ticker
        )
        if not same_company or not period_matches(metadata.get("period"), quarter):
            continue
        title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        documents.append(
            LocalDocument(
                path=path,
                published_at=str(metadata.get("published_at") or "Unknown"),
                source_url=str(metadata.get("source_url") or ""),
                document_type=str(metadata.get("document_type") or "Document")
                .replace("_", " ")
                .title(),
                period=str(metadata.get("period") or "Unknown"),
                title=title_match.group(1).strip() if title_match else path.stem,
                body=body.strip(),
            )
        )
    return sorted(documents, key=lambda item: (item.published_at, item.path.as_posix()))


def _clean_publication_time(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("published_at is required for every online source")
    try:
        parsed_date = date.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError as exc:
            raise ValueError("published_at must be an ISO date or timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("An exact published_at timestamp must include a timezone")
        return (
            parsed.astimezone(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if text != parsed_date.isoformat():
        raise ValueError("published_at must be an ISO date or timestamp")
    return text


def _clean_url(value: object) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be a valid public HTTP(S) URL")
    return url


def _clean_text(value: object, field: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    if not cleaned:
        raise ValueError(f"{field} is required for every online source")
    return cleaned


def normalize_online_piece(piece: Mapping[str, object]) -> dict[str, object]:
    """Validate one online source and its metric-level forward guidance."""
    raw_metrics = piece.get("metrics")
    if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
        raise ValueError("metrics must be a list for every online source")
    metrics = []
    for raw in raw_metrics:
        if not isinstance(raw, Mapping):
            raise ValueError("Every guidance metric must be an object")
        metrics.append(
            {
                "metric": _clean_text(raw.get("metric"), "metric"),
                "guidance": _clean_text(raw.get("guidance"), "guidance"),
                "target_period": _clean_text(raw.get("target_period"), "target_period"),
                "basis": _clean_text(raw.get("basis"), "basis"),
            }
        )
    if not metrics:
        raise ValueError("Every online source must contain at least one guidance metric")
    return {
        "headline": _clean_text(piece.get("headline"), "headline"),
        "published_at": _clean_publication_time(piece.get("published_at")),
        "source_name": _clean_text(piece.get("source_name"), "source_name"),
        "source_url": _clean_url(piece.get("source_url")),
        "summary": _clean_text(piece.get("summary"), "summary"),
        "metrics": metrics,
    }


def _research_schema() -> dict[str, object]:
    metric = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metric": {"type": "string"},
            "guidance": {"type": "string"},
            "target_period": {"type": "string"},
            "basis": {"type": "string"},
        },
        "required": ["metric", "guidance", "target_period", "basis"],
    }
    piece = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline": {"type": "string"},
            "published_at": {"type": "string"},
            "source_name": {"type": "string"},
            "source_url": {"type": "string"},
            "summary": {"type": "string"},
            "metrics": {"type": "array", "items": metric},
        },
        "required": [
            "headline",
            "published_at",
            "source_name",
            "source_url",
            "summary",
            "metrics",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"guidance": {"type": "array", "items": piece}},
        "required": ["guidance"],
    }


def _research_prompt(company_name: str, quarter: ReportingPeriod, max_results: int) -> str:
    return f"""Search the web for earnings releases, earnings-call coverage, filings and reliable
news about {company_name!r}'s {quarter} earnings that report management's forward guidance.
Return at most {max_results} distinct source articles or releases.

Include only sources explicitly tied to the requested period's earnings release or call and only
when they state forward-looking management guidance for at least one metric. Guidance may cover the
next quarter, the fiscal year or another future period. Metrics include revenue/sales, organic or
comparable growth, EPS, margins, expenses, tax, capex, cash flow, segment measures and other
quantified or qualitative outlook measures. Do not mistake reported results, analyst estimates,
price targets or unsourced predictions for company guidance.

Prefer issuer investor-relations releases, regulatory filings and transcripts, followed by Reuters,
Bloomberg and established financial publications. Preserve explicit ranges, units, currencies,
growth bases and GAAP/non-GAAP labels in the guidance field. Name the target period and describe the
basis (for example reported, organic, adjusted, constant-currency, or qualitative). Use a concise
factual summary; do not reproduce an article. Use the source's exact ISO 8601 publication timestamp
with timezone when available, otherwise YYYY-MM-DD. Use the canonical public article URL and do not
invent values, publication times or sources. Return an empty guidance list if none can be verified.
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
    quarter: ReportingPeriod,
    max_results: int,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> list[Mapping[str, object]]:
    """Research reporting-period-specific forward guidance using live web search."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for live guidance research")
    payload = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "input": _research_prompt(company_name, quarter, max_results),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "period_forward_guidance",
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
        raise RuntimeError(f"OpenAI guidance research request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI guidance research request failed: {exc.reason}") from exc

    parsed = json.loads(_extract_response_text(response))
    guidance = parsed.get("guidance")
    if not isinstance(guidance, list) or any(not isinstance(item, Mapping) for item in guidance):
        raise RuntimeError("OpenAI research output did not contain a valid guidance list")
    return guidance


def _call_researcher(
    researcher: object, company_name: str, quarter: ReportingPeriod
) -> list[Mapping[str, object]]:
    if hasattr(researcher, "research"):
        result = researcher.research(company_name, quarter)  # type: ignore[attr-defined]
    elif callable(researcher):
        result = researcher(company_name, quarter)
    else:
        raise TypeError("researcher must be callable or provide a research() method")
    if isinstance(result, Mapping):
        result = result.get("guidance")
    if not isinstance(result, list) or any(not isinstance(item, Mapping) for item in result):
        raise ValueError("researcher must return a guidance list or {'guidance': [...]} mapping")
    return result


def _deduplicate_online_pieces(
    raw_pieces: Sequence[Mapping[str, object]], max_results: int
) -> list[dict[str, object]]:
    pieces = [normalize_online_piece(piece) for piece in raw_pieces]
    pieces.sort(key=lambda item: (str(item["published_at"]), str(item["source_name"])))
    selected = []
    seen_urls: set[str] = set()
    for piece in pieces:
        url = str(piece["source_url"]).casefold()
        if url in seen_urls:
            continue
        selected.append(piece)
        seen_urls.add(url)
        if len(selected) == max_results:
            break
    return selected


def _safe_company_filename(company_name: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", company_name, flags=re.UNICODE).strip("._-")
    return value or "company"


def render_guidance_file(
    company_name: str,
    quarter: ReportingPeriod,
    local_documents: Sequence[LocalDocument],
    online_pieces: Sequence[Mapping[str, object]],
    *,
    document_root: Path = DEFAULT_DOCUMENT_ROOT,
) -> str:
    """Render all local and online pieces with release metadata and dividers."""
    lines = [
        f"FORWARD GUIDANCE RESEARCH: {company_name}",
        f"EARNINGS PERIOD: {quarter}",
        f"LOCAL DOCUMENTS FOUND: {len(local_documents)}",
        f"ONLINE SOURCES FOUND: {len(online_pieces)}",
    ]

    if not local_documents and not online_pieces:
        lines.extend(["", DIVIDER, "", "No matching local documents or online guidance was found."])
        return "\n".join(lines).rstrip() + "\n"

    for document in local_documents:
        try:
            relative = document.path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            try:
                relative = document.path.relative_to(document_root.parent.parent).as_posix()
            except ValueError:
                relative = document.path.as_posix()
        local_header = [
            "",
            DIVIDER,
            "",
            f"RELEASE DATE: {document.published_at}",
            f"SOURCE: Offline corpus — {relative}",
        ]
        if document.source_url:
            local_header.append(f"SOURCE URL: {document.source_url}")
        lines.extend(
            local_header
            + [
                "SOURCE KIND: OFFLINE DOCUMENT",
                f"DOCUMENT TYPE: {document.document_type}",
                f"REPORTING PERIOD: {document.period}",
                "",
                f"TITLE: {document.title}",
                "",
                document.body or "[Document body is empty]",
            ]
        )

    for piece in online_pieces:
        lines.extend(
            [
                "",
                DIVIDER,
                "",
                f"RELEASE DATE: {piece['published_at']}",
                f"SOURCE: {piece['source_name']}",
                f"SOURCE URL: {piece['source_url']}",
                "SOURCE KIND: ONLINE FORWARD-GUIDANCE REPORT",
                "",
                f"HEADLINE: {piece['headline']}",
                f"SUMMARY: {piece['summary']}",
                "",
                "FORWARD GUIDANCE:",
            ]
        )
        for metric in piece["metrics"]:  # type: ignore[index]
            lines.extend(
                [
                    f"- METRIC: {metric['metric']}",
                    f"  GUIDANCE: {metric['guidance']}",
                    f"  TARGET PERIOD: {metric['target_period']}",
                    f"  BASIS: {metric['basis']}",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def get_guidance(
    company_name: str,
    quarter: str,
    *,
    output_path: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    document_root: str | Path = DEFAULT_DOCUMENT_ROOT,
    researcher: object | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    max_results: int = 30,
    timeout: int = 300,
) -> Path:
    """Collect local period documents and online forward guidance into a text file.

    ``researcher`` may be a callable taking ``(company_name, reporting_period)`` or an
    object with a matching ``research`` method. Omitting it enables live OpenAI
    web research and requires ``OPENAI_API_KEY``.
    """
    selector = re.sub(r"\s+", " ", str(company_name)).strip()
    if not selector:
        raise ValueError("company_name cannot be empty")
    requested = parse_reporting_period(quarter)
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 100:
        raise ValueError("max_results must be an integer from 1 to 100")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer")

    root = Path(document_root)
    try:
        company = load_company(Path(config_path), selector)
    except ValueError:
        # Non-challenge companies have no local corpus, but online collection remains valid.
        company = {"company": selector, "ticker": ""}
        local_documents = []
    else:
        directory = find_document_directory(root, company)
        local_documents = find_quarter_documents(directory, company, requested)
    canonical_name = str(company["company"])

    if researcher is None:
        researcher = lambda name, period: research_with_openai(
            name,
            period,
            max_results,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
    raw_online = _call_researcher(researcher, canonical_name, requested)
    online_pieces = _deduplicate_online_pieces(raw_online, max_results)

    if output_path is None:
        if isinstance(requested, FiscalYear):
            period_slug = f"FY{requested.year}"
        else:
            kind = "Q" if isinstance(requested, Quarter) else "H"
            number = requested.quarter if isinstance(requested, Quarter) else requested.half
            period_slug = f"{kind}{number}_{requested.year}"
        filename = f"guidance_{_safe_company_filename(canonical_name)}_{period_slug}.txt"
        output = DEFAULT_DATA_DIR / filename
    else:
        output = Path(output_path)
        if output.suffix.casefold() != ".txt":
            output = output.with_suffix(".txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_guidance_file(
            canonical_name,
            requested,
            local_documents,
            online_pieces,
            document_root=root,
        ),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True, help="Exact company name or ticker")
    parser.add_argument(
        "--quarter",
        "--period",
        dest="quarter",
        required=True,
        help="Reporting period, for example Q1 2026, H1 2026, or FY2026",
    )
    parser.add_argument("--output", type=Path, help="Output .txt path; defaults to data/")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--document-root", type=Path, default=DEFAULT_DOCUMENT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model with web search")
    parser.add_argument("--max-results", type=int, default=30, help="Maximum online sources (1-100)")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = get_guidance(
            args.company,
            args.quarter,
            output_path=args.output,
            config_path=args.config,
            document_root=args.document_root,
            model=args.model,
            max_results=args.max_results,
            timeout=args.timeout,
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote guidance research: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
