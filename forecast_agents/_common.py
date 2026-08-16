"""Shared file-resolution and OpenAI Responses API helpers."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_CONFIG = REPO_ROOT / "challenge" / "companies.json"
DEFAULT_MODEL = "gpt-5.6-sol"


def clean_company_name(company_name: str) -> str:
    """Normalize whitespace and resolve known ticker aliases to canonical names."""
    selector = re.sub(r"\s+", " ", str(company_name)).strip()
    if not selector:
        raise ValueError("company_name cannot be empty")
    if not DEFAULT_CONFIG.is_file():
        return selector

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    wanted = selector.casefold()
    for company in payload.get("companies", []):
        ticker = str(company.get("ticker") or "")
        aliases = {
            str(company.get("company") or "").casefold(),
            ticker.casefold(),
            ticker.rsplit(":", 1)[-1].casefold(),
        }
        if wanted in aliases:
            return str(company["company"])
    return selector


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._-")
    return cleaned or "company"


def normalize_period(period: object) -> tuple[str, str]:
    """Return one canonical display period and cache-file suffix.

    Accepted inputs include fiscal quarters, half years, and full years in common orders such as
    ``FY2026Q2``, ``Q2 FY2026``, ``2026H1``, ``1H2026`` and ``FY 2026``.
    """
    text = re.sub(r"[\s_-]+", "", str(period)).upper()
    match = re.fullmatch(r"(?:FY)?(\d{4})Q([1-4])", text)
    if not match:
        match = re.fullmatch(r"Q([1-4])(?:FY)?(\d{4})", text)
        if match:
            number, year = match.groups()
            return f"Q{number} {year}", f"Q{number}_{year}"
    else:
        year, number = match.groups()
        return f"Q{number} {year}", f"Q{number}_{year}"

    match = re.fullmatch(r"([1-4])Q(?:FY)?(\d{4})", text)
    if match:
        number, year = match.groups()
        return f"Q{number} {year}", f"Q{number}_{year}"

    match = re.fullmatch(r"(?:FY)?(\d{4})H([1-2])", text)
    if not match:
        match = re.fullmatch(r"H([1-2])(?:FY)?(\d{4})", text)
        if match:
            number, year = match.groups()
            return f"H{number} {year}", f"H{number}_{year}"
    else:
        year, number = match.groups()
        return f"H{number} {year}", f"H{number}_{year}"

    match = re.fullmatch(r"([1-2])H(?:FY)?(\d{4})", text)
    if match:
        number, year = match.groups()
        return f"H{number} {year}", f"H{number}_{year}"

    match = re.fullmatch(r"FY(\d{4})", text)
    if not match:
        match = re.fullmatch(r"(\d{4})(?:FY)?", text)
    if match:
        year = match.group(1)
        return f"FY{year}", f"FY{year}"

    raise ValueError(
        f"Invalid reporting period '{period}'. Use FY2026Q1, FY2026H1, or FY2026."
    )


def normalize_quarter(quarter: str) -> tuple[str, str]:
    """Backward-compatible alias for :func:`normalize_period`."""
    return normalize_period(quarter)


def parse_iso_date(value: object, field: str) -> date:
    """Normalize an ISO date or timestamp to a calendar date.

    External research sometimes returns a full ISO timestamp where a date was requested. Accepting
    it here prevents otherwise valid report metadata from failing at the next pipeline boundary.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    try:
        parsed_date = date.fromisoformat(text)
    except ValueError:
        try:
            parsed_datetime = datetime.fromisoformat(
                text[:-1] + "+00:00" if text.endswith("Z") else text
            )
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO date or timestamp") from exc
        return parsed_datetime.date()
    if text != parsed_date.isoformat():
        raise ValueError(f"{field} must be an ISO date or timestamp")
    return parsed_date


def normalize_metrics(
    metrics: Sequence[str | Mapping[str, object]] | Mapping[str, object],
) -> list[dict[str, str]]:
    """Normalize every metric input to ``{'label': ..., 'unit': ...}``.

    Sequence items may be ``Label|Unit`` strings or mappings using either singular or plural unit
    keys. A top-level ``{label: unit}`` mapping is also accepted.
    """
    if isinstance(metrics, Mapping):
        raw_metrics: Sequence[str | Mapping[str, object]] = [
            {"label": label, "unit": unit} for label, unit in metrics.items()
        ]
    else:
        raw_metrics = metrics
    if (
        isinstance(raw_metrics, (str, bytes))
        or not isinstance(raw_metrics, Sequence)
        or not raw_metrics
    ):
        raise ValueError("metrics must be a non-empty sequence or label-to-unit mapping")

    normalized = []
    seen = set()
    for raw in raw_metrics:
        if isinstance(raw, Mapping):
            label = str(raw.get("label") or raw.get("metric") or "")
            unit = str(raw.get("unit") or raw.get("units") or "")
        else:
            label, separator, unit = str(raw).partition("|")
            if not separator:
                unit = ""
        label = re.sub(r"\s+", " ", label).strip()
        unit = re.sub(r"\s+", " ", unit).strip()
        if not label or not unit:
            raise ValueError("each metric needs a label and unit (for example 'Revenue|USDm')")
        key = label.casefold()
        if key in seen:
            raise ValueError(f"duplicate metric: {label}")
        seen.add(key)
        normalized.append({"label": label, "unit": unit})
    return normalized


def resolve_data_file(
    expected_name: str,
    *,
    data_dir: str | Path,
    explicit_path: str | Path | None,
    kind: str,
) -> Path:
    """Resolve an explicitly supplied path or an exact collector filename."""
    path = Path(explicit_path) if explicit_path is not None else Path(data_dir) / expected_name
    if not path.is_file():
        folder = path.parent
        available = []
        if folder.is_dir():
            available = sorted(candidate.name for candidate in folder.glob(f"{kind}_*.txt"))
        hint = f" Available {kind} files: {', '.join(available)}" if available else ""
        raise FileNotFoundError(
            f"Expected {kind} data file was not found: {path}. "
            f"Run starter/{kind}.py first or pass an explicit path.{hint}"
        )
    return path


def extract_response_text(response: Mapping[str, object]) -> str:
    """Extract the first assistant output_text item from a Responses API response."""
    if response.get("error"):
        raise RuntimeError(f"OpenAI returned an error: {response['error']}")
    output = response.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        refusals = []
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "output_text":
                    return str(part.get("text") or "")
                if part.get("type") == "refusal":
                    refusals.append(str(part.get("refusal") or "Model refused the request"))
        if refusals:
            raise RuntimeError("OpenAI refused the analysis: " + " ".join(refusals))
    raise RuntimeError("OpenAI response did not contain output_text")


def request_structured_output(
    prompt: str,
    schema: Mapping[str, object],
    schema_name: str,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    web_search: bool = False,
) -> dict[str, object]:
    """Call the Responses API and parse a strict JSON-schema response."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for live LLM analysis")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer")

    payload: dict[str, object] = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    if web_search:
        payload["tools"] = [{"type": "web_search"}]

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
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"OpenAI analysis request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI analysis request failed: {exc.reason}") from exc

    parsed = json.loads(extract_response_text(response))
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI structured output was not a JSON object")
    return parsed


def normalize_mapping_result(result: object, field: str) -> dict[str, object]:
    """Normalize an injected analyzer's mapping or JSON-string result."""
    if isinstance(result, str):
        result = json.loads(result)
    if not isinstance(result, Mapping):
        raise ValueError(f"analyzer must return a mapping containing '{field}'")
    return dict(result)
