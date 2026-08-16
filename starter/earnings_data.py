#!/usr/bin/env python3
"""Research actual earnings and pre-earnings consensus, then write an audited XLSX.

The live researcher uses the OpenAI Responses API and its web-search tool.  Tests
and other callers can inject a deterministic ``researcher`` callable instead.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODEL = "gpt-5.6-sol"
SOURCE_CLASSES = {
    "official",
    "bloomberg",
    "validated_unofficial",
    "single_source",
    "not_found",
}
INVALID_SHEET_CHARACTERS = re.compile(r"[\\/*?:\[\]]")


@dataclass(frozen=True, order=True)
class Quarter:
    """A calendar-style quarter label used to bound the requested research."""

    year: int
    quarter: int

    def __post_init__(self) -> None:
        if self.year < 1900 or self.year > 2200 or self.quarter not in range(1, 5):
            raise ValueError(f"Invalid quarter: Q{self.quarter} {self.year}")

    @classmethod
    def parse(cls, value: str) -> "Quarter":
        text = re.sub(r"\s+", "", str(value)).upper()
        match = re.fullmatch(r"(?:FY)?(\d{4})Q([1-4])", text)
        if not match:
            match = re.fullmatch(r"Q([1-4])(?:FY)?(\d{4})", text)
            if match:
                return cls(int(match.group(2)), int(match.group(1)))
        if not match:
            raise ValueError(
                f"Invalid quarter '{value}'. Use formats such as 'Q1 2020' or '2020Q1'."
            )
        return cls(int(match.group(1)), int(match.group(2)))

    def next(self) -> "Quarter":
        return Quarter(self.year + (self.quarter == 4), 1 if self.quarter == 4 else self.quarter + 1)

    def __str__(self) -> str:
        return f"Q{self.quarter} {self.year}"


def quarter_range(start: str | Quarter, end: str | Quarter) -> list[Quarter]:
    """Return an inclusive sequence of quarters."""
    first = start if isinstance(start, Quarter) else Quarter.parse(start)
    last = end if isinstance(end, Quarter) else Quarter.parse(end)
    if first > last:
        raise ValueError(f"Start quarter {first} is after end quarter {last}")
    result = []
    current = first
    while current <= last:
        result.append(current)
        current = current.next()
    return result


def _as_list(value: str | Sequence[str], label: str) -> list[str]:
    values = [value] if isinstance(value, str) else list(value)
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    if not cleaned:
        raise ValueError(f"At least one {label} is required")
    return cleaned


def normalize_metrics(
    metrics: str | Sequence[str] | Mapping[str, str],
) -> dict[str, str]:
    """Normalize metric input to ``{label: units}`` while preserving order."""
    if isinstance(metrics, Mapping):
        normalized = {str(key).strip(): str(value).strip() for key, value in metrics.items()}
    else:
        normalized = {item: "" for item in _as_list(metrics, "metric")}
    if not normalized or any(not label for label in normalized):
        raise ValueError("Metric names cannot be empty")
    return normalized


def safe_sheet_name(company: str, used: set[str] | None = None) -> str:
    """Create a legal, unique Excel worksheet name."""
    used = used if used is not None else set()
    base = INVALID_SHEET_CHARACTERS.sub(" ", company).strip(" '") or "Company"
    base = re.sub(r"\s+", " ", base)[:31]
    candidate = base
    number = 2
    while candidate.casefold() in {item.casefold() for item in used}:
        suffix = f" ({number})"
        candidate = base[: 31 - len(suffix)].rstrip() + suffix
        number += 1
    used.add(candidate)
    return candidate


def _source_name(source: Mapping[str, object]) -> str:
    return str(source.get("name") or source.get("document_title") or source.get("url") or "Unknown")


def _source_domain(source: Mapping[str, object]) -> str:
    return urlparse(str(source.get("url") or "")).netloc.casefold().removeprefix("www.")


def _source_text(sources: Sequence[Mapping[str, object]]) -> str:
    parts = []
    for source in sources:
        name = _source_name(source)
        url = str(source.get("url") or "").strip()
        title = str(source.get("document_title") or "").strip()
        detail = name
        if title and title.casefold() != name.casefold():
            detail += f" — {title}"
        if url:
            detail += f" ({url})"
        if detail not in parts:
            parts.append(detail)
    return "; ".join(parts)


def _short_source_marker(sources: Sequence[Mapping[str, object]]) -> str:
    names = []
    for source in sources:
        name = _source_name(source)
        if name not in names:
            names.append(name)
    return ", ".join(names[:3]) or "unverified"


def _normalize_sources(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    sources = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        source = {
            "name": str(item.get("name") or "").strip(),
            "url": str(item.get("url") or "").strip(),
            "document_title": str(item.get("document_title") or "").strip(),
            "published_date": str(item.get("published_date") or "").strip(),
        }
        if source["name"] or source["url"]:
            sources.append(source)
    return sources


def _clean_number(value: object, field: str) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number or null, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    return value


def _clean_release_time(value: object) -> str | None:
    """Validate an exact timestamp, convert it to UTC, and emit ISO 8601 with ``Z``."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError(
            "earnings_release_time_utc must be an exact ISO 8601 timestamp with a timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("earnings_release_time_utc must include a timezone")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_source_class(source_class: object, sources: list[dict[str, str]]) -> str:
    result = str(source_class or "not_found").strip().casefold()
    if result not in SOURCE_CLASSES:
        result = "single_source" if sources else "not_found"
    if result == "validated_unofficial":
        domains = {_source_domain(source) for source in sources if _source_domain(source)}
        # Two pages from one publisher are not independent validation.
        if len(domains) < 2:
            result = "single_source"
    if result != "not_found" and not sources:
        result = "single_source"
    return result


def _confidence(*source_classes: str) -> str:
    classes = set(source_classes) - {"not_found"}
    if not classes:
        return "Not found"
    if classes <= {"official", "bloomberg"}:
        return "High"
    if "single_source" in classes:
        return "Low"
    return "Medium"


def normalize_record(record: Mapping[str, object], company: str) -> dict[str, object]:
    """Validate a research result and apply independent-source rules."""
    quarter = str(Quarter.parse(str(record.get("quarter") or "")))
    metric = str(record.get("metric") or "").strip()
    if not metric:
        raise ValueError("Research record is missing a metric")
    actual_sources = _normalize_sources(record.get("actual_sources"))
    consensus_sources = _normalize_sources(record.get("consensus_sources"))
    release_time_sources = _normalize_sources(record.get("release_time_sources"))
    actual_class = _validate_source_class(record.get("actual_source_class"), actual_sources)
    consensus_class = _validate_source_class(record.get("consensus_source_class"), consensus_sources)
    release_time_class = _validate_source_class(
        record.get("release_time_source_class"), release_time_sources
    )
    actual_value = _clean_number(record.get("actual_value"), "actual_value")
    consensus_value = _clean_number(record.get("consensus_value"), "consensus_value")
    release_time = _clean_release_time(record.get("earnings_release_time_utc"))
    if actual_value is None:
        actual_class = "not_found"
    if consensus_value is None:
        consensus_class = "not_found"
    if release_time is None:
        release_time_class = "not_found"
    return {
        "company": company,
        "quarter": quarter,
        "earnings_release_time_utc": release_time,
        "release_time_source_class": release_time_class,
        "release_time_sources": release_time_sources,
        "metric": metric,
        "units": str(record.get("units") or "").strip(),
        "actual_value": actual_value,
        "actual_source_class": actual_class,
        "actual_sources": actual_sources,
        "consensus_value": consensus_value,
        "consensus_source_class": consensus_class,
        "consensus_sources": consensus_sources,
        "consensus_as_of": str(record.get("consensus_as_of") or "").strip(),
        "confidence": _confidence(actual_class, consensus_class, release_time_class),
        "notes": str(record.get("notes") or "").strip(),
    }


def _research_schema() -> dict[str, object]:
    source = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "url": {"type": "string"},
            "document_title": {"type": "string"},
            "published_date": {"type": "string"},
        },
        "required": ["name", "url", "document_title", "published_date"],
    }
    record = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "quarter": {"type": "string"},
            "earnings_release_time_utc": {"type": ["string", "null"]},
            "release_time_source_class": {"type": "string", "enum": sorted(SOURCE_CLASSES)},
            "release_time_sources": {"type": "array", "items": source},
            "metric": {"type": "string"},
            "units": {"type": "string"},
            "actual_value": {"type": ["number", "null"]},
            "actual_source_class": {"type": "string", "enum": sorted(SOURCE_CLASSES)},
            "actual_sources": {"type": "array", "items": source},
            "consensus_value": {"type": ["number", "null"]},
            "consensus_source_class": {"type": "string", "enum": sorted(SOURCE_CLASSES)},
            "consensus_sources": {"type": "array", "items": source},
            "consensus_as_of": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": [
            "quarter",
            "earnings_release_time_utc",
            "release_time_source_class",
            "release_time_sources",
            "metric",
            "units",
            "actual_value",
            "actual_source_class",
            "actual_sources",
            "consensus_value",
            "consensus_source_class",
            "consensus_sources",
            "consensus_as_of",
            "notes",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"records": {"type": "array", "items": record}},
        "required": ["records"],
    }


def _research_prompt(
    company: str,
    metrics: Mapping[str, str],
    start: Quarter,
    end: Quarter,
    include_next_consensus: bool = True,
) -> str:
    metric_text = ", ".join(
        f"{label} ({units})" if units else label for label, units in metrics.items()
    )
    next_instruction = (
        f"Also search for consensus for {end.next()}, but include that next quarter only if a "
        "consensus has already been published."
        if include_next_consensus
        else "Do not return periods outside this requested range."
    )
    return f"""Research quarterly earnings for {company}.

Requested periods: {start} through {end}, inclusive.
Requested metrics: {metric_text}.
{next_instruction} Treat the labels as the company's fiscal quarters when the company uses a
non-calendar fiscal year, and explain ambiguity in notes.

For every requested metric and quarter, find:
1. The exact time the earnings release was issued, expressed as an ISO 8601 UTC timestamp ending in
   Z (for example, 2026-05-19T10:00:00Z). This is the results publication time, not the earnings-call
   time. Convert a sourced local time to UTC. If only a date or an approximate label such as "before
   market open" is available, return null rather than inventing a time. For the next quarter, include
   a scheduled release timestamp only when an exact time has been published.
2. The reported actual value.
3. The Wall Street consensus/estimate that was available BEFORE that earnings release. Never use a
   consensus revised after the result as the pre-earnings estimate. Record its as-of/publication date.

Evidence rules, in descending priority:
- Actuals: issuer earnings releases/reports and regulatory filings (10-Q/10-K, SEC, Companies House,
  or the relevant national regulator) are primary. Prefer the source containing the exact metric.
- Consensus: prefer Bloomberg consensus. Next prefer FactSet, LSEG/Refinitiv, Visible Alpha, S&P
  Capital IQ, or a reputable newswire explicitly attributing one of those datasets.
- If no official/Bloomberg figure is accessible, validate it with at least two independent publishers.
  Two pages from the same publisher are one source. Do not infer, calculate, or silently substitute a
  differently defined GAAP/non-GAAP metric.
- Use source_class 'official' only for issuer/regulator documents; 'bloomberg' only for Bloomberg;
  'validated_unofficial' only for two independent publishers; 'single_source' when only one source
  supports the figure; and 'not_found' when there is no defensible value.
- Values must be numbers in the units field. Do not put currency symbols or source text in values.
- Apply the same evidence/source-class rules to release_time_sources. Prefer a timestamp on the
  issuer release or regulatory filing; otherwise validate the exact time independently where possible.
- Copy each requested metric label exactly into the metric field and format quarters as "Qn YYYY".
- Return one record for every requested quarter/metric combination, using null for unavailable data.
"""


def _extract_response_text(response: Mapping[str, object]) -> str:
    for output in response.get("output", []):
        if not isinstance(output, Mapping) or output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                return str(content.get("text") or "")
    raise RuntimeError("OpenAI response did not contain output_text")


def research_with_openai(
    company: str,
    metrics: Mapping[str, str],
    start: Quarter,
    end: Quarter,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    include_next_consensus: bool = True,
) -> list[dict[str, object]]:
    """Research one company using Responses API web search and structured output."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for live earnings research")
    payload = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "input": _research_prompt(company, metrics, start, end, include_next_consensus),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "earnings_research",
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
    records = parsed.get("records")
    if not isinstance(records, list):
        raise RuntimeError("OpenAI research output did not contain a records list")
    return records


def _research_live_batched(
    company: str,
    metrics: Mapping[str, str],
    start: Quarter,
    end: Quarter,
    *,
    api_key: str | None,
    model: str,
    batch_size: int = 4,
) -> list[dict[str, object]]:
    """Research in bounded windows so long historical ranges receive adequate searches."""
    quarters = quarter_range(start, end)
    records: list[dict[str, object]] = []
    for offset in range(0, len(quarters), batch_size):
        batch = quarters[offset : offset + batch_size]
        is_last = offset + batch_size >= len(quarters)
        records.extend(
            research_with_openai(
                company,
                metrics,
                batch[0],
                batch[-1],
                api_key=api_key,
                model=model,
                include_next_consensus=is_last,
            )
        )
    return records


def _call_researcher(
    researcher: object,
    company: str,
    metrics: Mapping[str, str],
    start: Quarter,
    end: Quarter,
) -> list[Mapping[str, object]]:
    if hasattr(researcher, "research"):
        result = researcher.research(company, metrics, start, end)  # type: ignore[attr-defined]
    elif callable(researcher):
        result = researcher(company, metrics, start, end)
    else:
        raise TypeError("researcher must be callable or provide a research() method")
    if isinstance(result, Mapping):
        result = result.get("records")
    if not isinstance(result, list) or any(not isinstance(item, Mapping) for item in result):
        raise ValueError("researcher must return a list of record mappings")
    return result


def _complete_records(
    raw_records: Sequence[Mapping[str, object]],
    company: str,
    metrics: Mapping[str, str],
    quarters: Sequence[Quarter],
) -> list[dict[str, object]]:
    normalized = [normalize_record(record, company) for record in raw_records]
    wanted_metrics = {label.casefold(): (label, units) for label, units in metrics.items()}
    allowed_quarters = {str(quarter) for quarter in quarters}
    next_quarter = str(quarters[-1].next())
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for record in normalized:
        metric_key = str(record["metric"]).casefold()
        quarter = str(record["quarter"])
        if metric_key not in wanted_metrics:
            continue
        # The next quarter is estimate-only and is included only when a consensus exists.
        if quarter == next_quarter:
            if record["consensus_value"] is None:
                continue
            record["actual_value"] = None
            record["actual_source_class"] = "not_found"
            record["actual_sources"] = []
        elif quarter not in allowed_quarters:
            continue
        label, default_units = wanted_metrics[metric_key]
        record["metric"] = label
        record["units"] = record["units"] or default_units
        by_key[(quarter, metric_key)] = record

    result = []
    for quarter in quarters:
        for metric_key, (label, units) in wanted_metrics.items():
            record = by_key.get((str(quarter), metric_key))
            if record is None:
                record = normalize_record(
                    {
                        "quarter": str(quarter),
                        "earnings_release_time_utc": None,
                        "release_time_source_class": "not_found",
                        "release_time_sources": [],
                        "metric": label,
                        "units": units,
                        "actual_value": None,
                        "actual_source_class": "not_found",
                        "actual_sources": [],
                        "consensus_value": None,
                        "consensus_source_class": "not_found",
                        "consensus_sources": [],
                        "consensus_as_of": "",
                        "notes": "No defensible value returned by the researcher.",
                    },
                    company,
                )
            result.append(record)
    for metric_key in wanted_metrics:
        record = by_key.get((next_quarter, metric_key))
        if record is not None:
            result.append(record)

    # Release time is quarter-level data. Use the strongest sourced candidate and repeat it
    # consistently on every metric row for that quarter.
    release_rank = {
        "not_found": 0,
        "single_source": 1,
        "validated_unofficial": 2,
        "bloomberg": 3,
        "official": 4,
    }
    release_by_quarter: dict[str, dict[str, object]] = {}
    for record in normalized:
        if record["earnings_release_time_utc"] is None:
            continue
        quarter = str(record["quarter"])
        candidate = release_by_quarter.get(quarter)
        if candidate is None or release_rank[str(record["release_time_source_class"])] > release_rank[
            str(candidate["release_time_source_class"])
        ]:
            release_by_quarter[quarter] = record
    for record in result:
        release = release_by_quarter.get(str(record["quarter"]))
        if release is not None:
            record["earnings_release_time_utc"] = release["earnings_release_time_utc"]
            record["release_time_source_class"] = release["release_time_source_class"]
            record["release_time_sources"] = release["release_time_sources"]
            record["confidence"] = _confidence(
                str(record["actual_source_class"]),
                str(record["consensus_source_class"]),
                str(record["release_time_source_class"]),
            )
    return result


def _display_value(value: object, source_class: str, sources: Sequence[Mapping[str, object]]) -> object:
    if value is None:
        return "Not found"
    if source_class == "official":
        return value
    return f"{value:g} [Source: {_short_source_marker(sources)}]"


def _display_release_time(
    value: object, source_class: str, sources: Sequence[Mapping[str, object]]
) -> str:
    if value is None:
        return "Not found"
    if source_class == "official":
        return str(value)
    return f"{value} [Source: {_short_source_marker(sources)}]"


HEADERS = [
    "Quarter",
    "Earnings release time (UTC)",
    "Release-time source class",
    "Release-time source(s)",
    "Metric",
    "Units",
    "Reported actual",
    "Actual source class",
    "Actual source(s)",
    "Pre-earnings consensus",
    "Consensus source class",
    "Consensus source(s)",
    "Consensus as of",
    "Confidence",
    "Validation notes",
]


def rows_for_workbook(records: Sequence[Mapping[str, object]]) -> list[list[object]]:
    rows = [HEADERS]
    for record in records:
        rows.append(
            [
                record["quarter"],
                _display_release_time(
                    record["earnings_release_time_utc"],
                    str(record["release_time_source_class"]),
                    record["release_time_sources"],  # type: ignore[arg-type]
                ),
                record["release_time_source_class"],
                _source_text(record["release_time_sources"]),  # type: ignore[arg-type]
                record["metric"],
                record["units"],
                _display_value(
                    record["actual_value"],
                    str(record["actual_source_class"]),
                    record["actual_sources"],  # type: ignore[arg-type]
                ),
                record["actual_source_class"],
                _source_text(record["actual_sources"]),  # type: ignore[arg-type]
                _display_value(
                    record["consensus_value"],
                    str(record["consensus_source_class"]),
                    record["consensus_sources"],  # type: ignore[arg-type]
                ),
                record["consensus_source_class"],
                _source_text(record["consensus_sources"]),  # type: ignore[arg-type]
                record["consensus_as_of"],
                record["confidence"],
                record["notes"],
            ]
        )
    return rows


def _column_name(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _cell_xml(
    row: int, column: int, value: object, header: bool = False, styled: bool = True
) -> str:
    reference = f"{_column_name(column)}{row}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        style = (1 if header else 3) if styled else 0
        return f'<c r="{reference}" s="{style}"><v>{value}</v></c>'
    style = (1 if header else 2) if styled else 0
    text = escape(str(value))
    preserve = ' xml:space="preserve"' if str(value).strip() != str(value) else ""
    return f'<c r="{reference}" s="{style}" t="inlineStr"><is><t{preserve}>{text}</t></is></c>'


def _sheet_xml(rows: Sequence[Sequence[object]], *, styled: bool = True) -> str:
    widths = [12, 30, 23, 55, 28, 14, 24, 20, 55, 28, 22, 55, 18, 14, 55]
    column_count = max([len(widths), *(len(row) for row in rows)])
    widths.extend([24] * (column_count - len(widths)))
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    xml_rows = []
    for row_number, values in enumerate(rows, start=1):
        cells = "".join(
            _cell_xml(row_number, column, value, header=row_number == 1, styled=styled)
            for column, value in enumerate(values, start=1)
        )
        height = ' ht="30" customHeight="1"' if row_number == 1 else ""
        xml_rows.append(f'<row r="{row_number}"{height}>{cells}</row>')
    last_column = _column_name(column_count)
    last_cell = f"{last_column}{max(len(rows), 1)}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_cell}"/>'
        '<sheetViews><sheetView showGridLines="0" workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/>'
        f'<cols>{columns}</cols><sheetData>{"".join(xml_rows)}</sheetData>'
        f'<autoFilter ref="A1:{last_column}{max(len(rows), 1)}"/>'
        '<pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>'
        '</worksheet>'
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="#,##0.00##;[Red](#,##0.00##);-"/></numFmts>
  <fonts count="2"><font><sz val="10"/><name val="Aptos"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos Display"/></font></fonts>
  <fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill></fills>
  <borders count="2"><border/><border><bottom style="thin"><color rgb="FFB4C6E7"/></bottom></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="top"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def write_xlsx(sheets: Sequence[tuple[str, Sequence[Sequence[object]]]], output_path: Path) -> Path:
    """Write the small tabular workbook used by :func:`get_earnings`."""
    if not sheets:
        raise ValueError("At least one worksheet is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f"{sheet_overrides}</Types>"
    )
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    workbook_sheets = "".join(
        f'<sheet name={quoteattr(name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _) in enumerate(sheets, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{workbook_sheets}</sheets><calcPr calcId="191029"/></workbook>'
    )
    sheet_rels = "".join(
        f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{sheet_rels}<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:creator>earnings_data.py</dc:creator><dc:title>Earnings actuals and consensus</dc:title><dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created></cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>earnings_data.py</Application></Properties>"""
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", _styles_xml())
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows))
    return output_path


MAIN_XML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPE_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
WORKSHEET_REL_TYPE = f"{OFFICE_REL_NS}/worksheet"
WORKSHEET_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
)


def _shared_strings(files: Mapping[str, bytes]) -> list[str]:
    payload = files.get("xl/sharedStrings.xml")
    if payload is None:
        return []
    root = ElementTree.fromstring(payload)
    return ["".join(node.text or "" for node in item.iter(f"{{{MAIN_XML_NS}}}t")) for item in root]


def _cell_column(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference.upper())
    if not match:
        raise ValueError(f"Invalid worksheet cell reference: {reference}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - 64
    return result


def _read_sheet_rows(payload: bytes, shared_strings: Sequence[str]) -> list[list[object]]:
    root = ElementTree.fromstring(payload)
    rows: list[list[object]] = []
    for row in root.findall(f".//{{{MAIN_XML_NS}}}sheetData/{{{MAIN_XML_NS}}}row"):
        values: list[object] = []
        for cell in row.findall(f"{{{MAIN_XML_NS}}}c"):
            column = _cell_column(str(cell.get("r") or "A"))
            while len(values) < column:
                values.append("")
            cell_type = cell.get("t")
            if cell_type == "inlineStr":
                value: object = "".join(
                    node.text or "" for node in cell.iter(f"{{{MAIN_XML_NS}}}t")
                )
            else:
                value_node = cell.find(f"{{{MAIN_XML_NS}}}v")
                raw = value_node.text if value_node is not None and value_node.text is not None else ""
                if cell_type == "s" and raw:
                    value = shared_strings[int(raw)]
                elif cell_type in {"str", "e"} or not raw:
                    value = raw
                elif cell_type == "b":
                    value = raw == "1"
                else:
                    number = float(raw)
                    value = int(number) if number.is_integer() else number
            values[column - 1] = value
        while values and values[-1] == "":
            values.pop()
        rows.append(values)
    return rows


def _is_empty_workbook_value(value: object) -> bool:
    text = str(value or "").strip()
    return text in {
        "",
        "Not found",
        "not_found",
        "No defensible value returned by the researcher.",
    }


def _merge_sheet_rows(
    existing_rows: Sequence[Sequence[object]], new_rows: Sequence[Sequence[object]]
) -> list[list[object]]:
    """Append new quarter/metric rows and fill only empty cells in existing rows."""
    if not existing_rows:
        return [list(row) for row in new_rows]
    if not new_rows:
        return [list(row) for row in existing_rows]
    existing_headers = [str(value) for value in existing_rows[0]]
    new_headers = [str(value) for value in new_rows[0]]
    if "Quarter" not in existing_headers or "Metric" not in existing_headers:
        raise ValueError(
            "Existing company tab is not an earnings-data table (Quarter/Metric headers missing)"
        )
    headers = list(new_headers)
    headers.extend(header for header in existing_headers if header not in headers)

    def align(row: Sequence[object], source_headers: Sequence[str]) -> list[object]:
        mapped = {header: row[index] if index < len(row) else "" for index, header in enumerate(source_headers)}
        return [mapped.get(header, "") for header in headers]

    merged = [headers]
    positions: dict[tuple[str, str], int] = {}
    for row in existing_rows[1:]:
        aligned = align(row, existing_headers)
        merged.append(aligned)
        mapped = dict(zip(headers, aligned))
        key = (str(mapped.get("Quarter", "")).casefold(), str(mapped.get("Metric", "")).casefold())
        if all(key) and key not in positions:
            positions[key] = len(merged) - 1

    for row in new_rows[1:]:
        incoming = align(row, new_headers)
        mapped = dict(zip(headers, incoming))
        key = (str(mapped.get("Quarter", "")).casefold(), str(mapped.get("Metric", "")).casefold())
        existing_index = positions.get(key)
        if existing_index is None:
            merged.append(incoming)
            if all(key):
                positions[key] = len(merged) - 1
            continue
        current = merged[existing_index]
        for index, value in enumerate(incoming):
            if _is_empty_workbook_value(current[index]) and not _is_empty_workbook_value(value):
                current[index] = value
    return merged


def _has_compatible_styles(files: Mapping[str, bytes]) -> bool:
    payload = files.get("xl/styles.xml")
    if payload is None:
        return False
    try:
        root = ElementTree.fromstring(payload)
        cell_xfs = root.find(f"{{{MAIN_XML_NS}}}cellXfs")
        return cell_xfs is not None and len(cell_xfs) >= 4
    except ElementTree.ParseError:
        return False


def update_xlsx(
    sheets: Sequence[tuple[str, Sequence[Sequence[object]]]], output_path: Path
) -> Path:
    """Incrementally add/merge company sheets while preserving other workbook parts."""
    try:
        with zipfile.ZipFile(output_path, "r") as archive:
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Existing output is not a readable XLSX workbook: {output_path}") from exc

    required = {"xl/workbook.xml", "xl/_rels/workbook.xml.rels", "[Content_Types].xml"}
    missing = required - files.keys()
    if missing:
        raise ValueError(f"Existing workbook is missing required part(s): {', '.join(sorted(missing))}")

    workbook_root = ElementTree.fromstring(files["xl/workbook.xml"])
    relationships_root = ElementTree.fromstring(files["xl/_rels/workbook.xml.rels"])
    content_types_root = ElementTree.fromstring(files["[Content_Types].xml"])
    shared = _shared_strings(files)
    styled = _has_compatible_styles(files)

    relationships = {
        relation.get("Id"): relation
        for relation in relationships_root.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    sheet_parent = workbook_root.find(f"{{{MAIN_XML_NS}}}sheets")
    if sheet_parent is None:
        raise ValueError("Existing workbook has no worksheets collection")

    sheet_details: dict[str, tuple[ElementTree.Element, str]] = {}
    for sheet in sheet_parent.findall(f"{{{MAIN_XML_NS}}}sheet"):
        relation_id = sheet.get(f"{{{OFFICE_REL_NS}}}id")
        relation = relationships.get(relation_id)
        if relation is None or relation.get("Type") != WORKSHEET_REL_TYPE:
            continue
        target = str(relation.get("Target") or "")
        path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(
            posixpath.join("xl", target)
        )
        sheet_details[str(sheet.get("name") or "").casefold()] = (sheet, path)

    used_paths = set(files)
    for requested_name, new_rows in sheets:
        detail = sheet_details.get(requested_name.casefold())
        if detail is not None:
            _, sheet_path = detail
            if sheet_path not in files:
                raise ValueError(f"Existing worksheet part is missing: {sheet_path}")
            existing_rows = _read_sheet_rows(files[sheet_path], shared)
            merged_rows = _merge_sheet_rows(existing_rows, new_rows)
            files[sheet_path] = _sheet_xml(merged_rows, styled=styled).encode("utf-8")
            continue

        sheet_number = 1
        while f"xl/worksheets/sheet{sheet_number}.xml" in used_paths:
            sheet_number += 1
        sheet_path = f"xl/worksheets/sheet{sheet_number}.xml"
        used_paths.add(sheet_path)
        relation_number = 1
        while f"rId{relation_number}" in relationships:
            relation_number += 1
        relation_id = f"rId{relation_number}"
        relation = ElementTree.SubElement(
            relationships_root,
            f"{{{PACKAGE_REL_NS}}}Relationship",
            {
                "Id": relation_id,
                "Type": WORKSHEET_REL_TYPE,
                "Target": f"worksheets/sheet{sheet_number}.xml",
            },
        )
        relationships[relation_id] = relation
        existing_ids = [int(sheet.get("sheetId") or 0) for sheet in sheet_parent]
        sheet = ElementTree.SubElement(
            sheet_parent,
            f"{{{MAIN_XML_NS}}}sheet",
            {
                "name": requested_name,
                "sheetId": str(max(existing_ids, default=0) + 1),
                f"{{{OFFICE_REL_NS}}}id": relation_id,
            },
        )
        sheet_details[requested_name.casefold()] = (sheet, sheet_path)
        ElementTree.SubElement(
            content_types_root,
            f"{{{CONTENT_TYPE_NS}}}Override",
            {"PartName": f"/{sheet_path}", "ContentType": WORKSHEET_CONTENT_TYPE},
        )
        files[sheet_path] = _sheet_xml(new_rows, styled=styled).encode("utf-8")

    files["xl/workbook.xml"] = ElementTree.tostring(
        workbook_root, encoding="utf-8", xml_declaration=True
    )
    files["xl/_rels/workbook.xml.rels"] = ElementTree.tostring(
        relationships_root, encoding="utf-8", xml_declaration=True
    )
    files["[Content_Types].xml"] = ElementTree.tostring(
        content_types_root, encoding="utf-8", xml_declaration=True
    )

    original_mode = output_path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}-", suffix=".xlsx", dir=output_path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)
        temporary_path.chmod(original_mode)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def get_earnings(
    companies: str | Sequence[str],
    metrics: str | Sequence[str] | Mapping[str, str],
    start_quarter: str,
    end_quarter: str,
    *,
    output_path: str | Path | None = None,
    researcher: object | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Research earnings actuals/consensus and save one company per Excel tab.

    ``metrics`` may be a label, a sequence of labels, or ``{label: units}``.
    ``researcher`` is an optional callable taking ``(company, metrics, start, end)``;
    omitting it enables live OpenAI web research and requires ``OPENAI_API_KEY``.
    """
    company_names = _as_list(companies, "company")
    metric_map = normalize_metrics(metrics)
    quarters = quarter_range(start_quarter, end_quarter)
    if researcher is None:
        researcher = lambda company, requested, start, end: _research_live_batched(
            company, requested, start, end, api_key=api_key, model=model
        )
    workbook_sheets = []
    used_names: set[str] = set()
    for company in company_names:
        raw = _call_researcher(researcher, company, metric_map, quarters[0], quarters[-1])
        records = _complete_records(raw, company, metric_map, quarters)
        workbook_sheets.append((safe_sheet_name(company, used_names), rows_for_workbook(records)))
    if output_path is None:
        output = DEFAULT_DATA_DIR / "earnings_data.xlsx"
    else:
        output = Path(output_path)
        if output.suffix.casefold() != ".xlsx":
            output = output.with_suffix(".xlsx")
    if output.exists():
        return update_xlsx(workbook_sheets, output)
    return write_xlsx(workbook_sheets, output)


def _parse_metric_arguments(values: Sequence[str]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for value in values:
        label, separator, units = value.partition("|")
        if not label.strip():
            raise ValueError("Metric names cannot be empty")
        metrics[label.strip()] = units.strip() if separator else ""
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", action="append", required=True, help="Company name; repeatable")
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Metric, optionally followed by |units; repeatable (for example Revenue|USDm)",
    )
    parser.add_argument("--start", required=True, help="First quarter, for example 'Q1 2020'")
    parser.add_argument("--end", required=True, help="Last quarter, for example 'Q2 2026'")
    parser.add_argument(
        "--output", type=Path, help="Output .xlsx path; defaults to data/earnings_data.xlsx"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model with web search")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = get_earnings(
            args.company,
            _parse_metric_arguments(args.metric),
            args.start,
            args.end,
            output_path=args.output,
            model=args.model,
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote earnings data: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
