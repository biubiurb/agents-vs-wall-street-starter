#!/usr/bin/env python3
"""End-to-end earnings agent: collect evidence, estimate changes, and forecast values."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Union

from ._common import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL,
    clean_company_name,
    normalize_metrics,
    normalize_period,
    parse_iso_date,
    request_structured_output,
    safe_filename,
)
from .comparable_analysis import DEFAULT_SEASON_LOOKBACK_DAYS, comparable_analysis
from .guidance_analysis import guidance_analysis
from .news_analysis import news_analysis


MetricInput = Union[str, Mapping[str, object]]
MetricCollection = Union[Sequence[MetricInput], Mapping[str, object]]


def configured_metrics(company_name: str) -> list[dict[str, str]]:
    """Load the challenge metrics for a known company or ticker."""
    canonical = clean_company_name(company_name)
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for company in payload.get("companies", []):
        if str(company.get("company")) == canonical:
            return [
                {"label": str(item["label"]), "unit": str(item["units"])}
                for item in company.get("metrics", [])
            ]
    raise ValueError(
        f"No configured metrics for {company_name}; supply one or more --metric 'Label|Unit'."
    )


def _normalize_metrics(metrics: MetricCollection) -> list[dict[str, str]]:
    """Backward-compatible local name for the shared metric normalizer."""
    return normalize_metrics(metrics)


def _parse_iso_date(value: object, field: str) -> date:
    """Backward-compatible local name for the shared ISO date parser."""
    return parse_iso_date(value, field)


def _report_schema(
    metrics: Sequence[Mapping[str, str]], target_period: str | None = None
) -> dict[str, object]:
    actual = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metric": {"type": "string", "enum": [item["label"] for item in metrics]},
            "unit": {"type": "string", "enum": sorted({item["unit"] for item in metrics})},
            "value": {"type": ["number", "null"]},
            "source_url": {"type": "string"},
        },
        "required": ["metric", "unit", "value", "source_url"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_report_date": {"type": "string"},
            "target_period": {
                "type": "string",
                **({"enum": [normalize_period(target_period)[0]]} if target_period else {}),
            },
            "previous_report_date": {"type": "string"},
            "previous_period": {"type": "string"},
            "previous_actuals": {"type": "array", "items": actual},
        },
        "required": [
            "target_report_date",
            "target_period",
            "previous_report_date",
            "previous_period",
            "previous_actuals",
        ],
    }


def _report_prompt(
    company_name: str, target_period: str, metrics: Sequence[Mapping[str, str]]
) -> str:
    metric_text = "\n".join(f"- {item['label']} ({item['unit']})" for item in metrics)
    return f"""Research the reporting schedule and immediately preceding earnings report for
{company_name}. The forecast target is {target_period}.

Return the target period in canonical form, the announced or best verified expected report date,
the actual date of the immediately preceding earnings report, and that preceding report's fiscal
period. Dates must be YYYY-MM-DD. Then extract the reported actual for every requested metric from
the preceding report:
{metric_text}

Copy each metric label and unit exactly. Use the issuer earnings release, filing, or investor
presentation wherever possible and provide the direct source URL. Do not substitute a differently
defined GAAP/non-GAAP, quarterly/half-year, currency, or percentage measure. Return null rather than
guessing an unavailable value. The previous report must chronologically precede the target report.
"""


def _call_report_researcher(
    researcher: object,
    company_name: str,
    target_period: str,
    metrics: Sequence[Mapping[str, str]],
) -> object:
    if hasattr(researcher, "research"):
        return researcher.research(  # type: ignore[attr-defined]
            company_name, target_period, metrics
        )
    if callable(researcher):
        return researcher(company_name, target_period, metrics)
    raise TypeError("report_researcher must be callable or provide a research() method")


def _validate_report_context(
    payload: Mapping[str, object], metrics: Sequence[Mapping[str, str]], target_period: str
) -> dict[str, object]:
    expected_target, _ = normalize_period(target_period)
    reported_target, _ = normalize_period(payload.get("target_period") or expected_target)
    if reported_target != expected_target:
        raise ValueError(
            f"report research returned target period {reported_target}, expected {expected_target}"
        )
    target_date = _parse_iso_date(payload.get("target_report_date"), "target_report_date")
    previous_date = _parse_iso_date(payload.get("previous_report_date"), "previous_report_date")
    if previous_date >= target_date:
        raise ValueError("previous_report_date must be before target_report_date")
    previous_period, _ = normalize_period(str(payload.get("previous_period") or ""))

    raw_actuals: object = payload.get("previous_actuals")
    if isinstance(raw_actuals, Mapping):
        raw_evidence = payload.get("previous_actual_evidence")
        if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
            raw_evidence = []
        evidence_by_metric = {
            str(item.get("metric")): item
            for item in raw_evidence
            if isinstance(item, Mapping)
        }
        raw_actuals = [
            {
                "metric": metric,
                "unit": next(
                    (item["unit"] for item in metrics if item["label"] == metric), ""
                ),
                "value": value,
                "source_url": str(evidence_by_metric.get(str(metric), {}).get("source_url") or ""),
            }
            for metric, value in raw_actuals.items()
        ]
    if not isinstance(raw_actuals, Sequence) or isinstance(raw_actuals, (str, bytes)):
        raise ValueError("report research must contain a previous_actuals list")
    expected = {item["label"]: item["unit"] for item in metrics}
    actual_values: dict[str, float] = {}
    actuals = []
    for raw in raw_actuals:
        if not isinstance(raw, Mapping):
            raise ValueError("every previous actual must be a mapping")
        metric = str(raw.get("metric") or "").strip()
        if metric not in expected or metric in actual_values:
            raise ValueError(f"unexpected or duplicate previous actual: {metric}")
        if str(raw.get("unit") or "").strip() != expected[metric]:
            raise ValueError(f"previous actual used the wrong unit for {metric}")
        value = raw.get("value")
        if value is None:
            raise ValueError(f"no defensible previous actual was found for {metric}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"previous actual for {metric} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"previous actual for {metric} must be finite")
        actual_values[metric] = number
        actuals.append(
            {
                "metric": metric,
                "unit": expected[metric],
                "value": number,
                "source_url": str(raw.get("source_url") or "").strip(),
            }
        )
    missing = [metric for metric in expected if metric not in actual_values]
    if missing:
        raise ValueError("report research omitted previous actuals: " + ", ".join(missing))
    return {
        "target_report_date": target_date.isoformat(),
        "target_period": expected_target,
        "previous_report_date": previous_date.isoformat(),
        "previous_period": previous_period,
        "previous_actuals": actuals,
    }


def _load_or_fetch_report_context(
    company_name: str,
    target_period: str,
    metrics: Sequence[Mapping[str, str]],
    path: Path,
    *,
    report_researcher: object | None,
    api_key: str | None,
    model: str,
    timeout: int,
) -> dict[str, object]:
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif report_researcher is None:
        raw = request_structured_output(
            _report_prompt(company_name, target_period, metrics),
            _report_schema(metrics, target_period),
            "earnings_report_context",
            api_key=api_key,
            model=model,
            timeout=timeout,
            web_search=True,
        )
    else:
        raw = _call_report_researcher(report_researcher, company_name, target_period, metrics)
    if not isinstance(raw, Mapping):
        raise ValueError("report researcher must return a mapping")
    context = _validate_report_context(raw, metrics, target_period)
    if not path.is_file() or dict(raw) != context:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return context


def _ensure_collected(
    path: Path,
    collector: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> Path:
    if path.is_file():
        return path
    result = collector(*args, output_path=path, **kwargs)
    resolved = Path(result) if result is not None else path
    if not resolved.is_file():
        raise FileNotFoundError(f"collector did not create expected data file: {resolved}")
    return resolved


def run_analysis(
    company_name: str,
    previous_quarter: str,
    start_date: str,
    end_date: str,
    metrics: MetricCollection,
    *,
    target_period: str | None = None,
    comparable_summary: str | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    guidance_path: str | Path | None = None,
    news_path: str | Path | None = None,
    details_output_path: str | Path | None = None,
    offline_data_dir: str | Path = DEFAULT_CONFIG.parent / "offline-data",
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> dict[str, dict[str, float | str]]:
    """Backward-compatible two-stage analysis returning guidance-relative adjustments."""
    guidance_summary = guidance_analysis(
        company_name,
        previous_quarter,
        data_dir=data_dir,
        guidance_path=guidance_path,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
    return news_analysis(
        company_name,
        start_date,
        end_date,
        metrics,
        target_period=target_period,
        guidance_summary=guidance_summary,
        comparable_summary=comparable_summary,
        data_dir=data_dir,
        news_path=news_path,
        offline_data_dir=offline_data_dir,
        details_output_path=details_output_path,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )


def forecast_company(
    company_name: str,
    target_period: str,
    metrics: MetricCollection,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    report_path: str | Path | None = None,
    guidance_path: str | Path | None = None,
    news_path: str | Path | None = None,
    comparable_summary: str | None = None,
    comparable_companies: Sequence[str] | None = None,
    comparable_season_lookback_days: int = DEFAULT_SEASON_LOOKBACK_DAYS,
    as_of_date: str | date | None = None,
    details_output_path: str | Path | None = None,
    offline_data_dir: str | Path = DEFAULT_CONFIG.parent / "offline-data",
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    report_researcher: object | None = None,
    guidance_collector: Callable[..., object] | None = None,
    news_collector: Callable[..., object] | None = None,
    guidance_researcher: object | None = None,
    news_researcher: object | None = None,
    guidance_analyzer: object | None = None,
    guidance_fallback_analyzer: object | None = None,
    comparable_analyzer: object | None = None,
    event_analyzer: object | None = None,
    impact_analyzer: object | None = None,
) -> dict[str, dict[str, float | str]]:
    """Run the complete forecast pipeline and return absolute metric predictions."""
    canonical = clean_company_name(company_name)
    display_target, target_slug = normalize_period(target_period)
    normalized_metrics = _normalize_metrics(metrics)
    root = Path(data_dir)
    report_file = (
        Path(report_path)
        if report_path is not None
        else root / f"earnings_{safe_filename(canonical)}_{target_slug}.json"
    )
    context = _load_or_fetch_report_context(
        canonical,
        display_target,
        normalized_metrics,
        report_file,
        report_researcher=report_researcher,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )

    previous_period = str(context["previous_period"])
    _, previous_slug = normalize_period(previous_period)
    previous_report_date = _parse_iso_date(context["previous_report_date"], "previous_report_date")
    target_report_date = _parse_iso_date(context["target_report_date"], "target_report_date")
    requested_cutoff = (
        _parse_iso_date(as_of_date, "as_of_date") if as_of_date is not None else date.today()
    )
    cutoff = min(requested_cutoff, target_report_date - timedelta(days=1))
    news_start = previous_report_date + timedelta(days=1)
    if cutoff < news_start:
        raise ValueError("information cutoff must be after the previous report date")

    if guidance_collector is None or news_collector is None:
        from starter.guidance import get_guidance
        from starter.news import get_news

        guidance_collector = guidance_collector or get_guidance
        news_collector = news_collector or get_news

    guidance_file = (
        Path(guidance_path)
        if guidance_path is not None
        else root / f"guidance_{safe_filename(canonical)}_{previous_slug}.txt"
    )
    guidance_file = _ensure_collected(
        guidance_file,
        guidance_collector,
        canonical,
        previous_period,
        researcher=guidance_researcher,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
    news_file = (
        Path(news_path)
        if news_path is not None
        else root
        / (
            f"news_{safe_filename(canonical)}_{news_start.isoformat()}_to_"
            f"{cutoff.isoformat()}.txt"
        )
    )
    news_file = _ensure_collected(
        news_file,
        news_collector,
        canonical,
        news_start.isoformat(),
        cutoff.isoformat(),
        researcher=news_researcher,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )

    guidance_summary = guidance_analysis(
        canonical,
        previous_period,
        data_dir=root,
        guidance_path=guidance_file,
        api_key=api_key,
        model=model,
        timeout=timeout,
        analyzer=guidance_analyzer,
        fallback_analyzer=guidance_fallback_analyzer,
    )
    if comparable_summary is None:
        comparable_summary = comparable_analysis(
            canonical,
            display_target,
            target_report_date,
            normalized_metrics,
            information_cutoff=cutoff,
            season_lookback_days=comparable_season_lookback_days,
            comparable_companies=comparable_companies,
            api_key=api_key,
            model=model,
            timeout=timeout,
            analyzer=comparable_analyzer,
        )
    elif not str(comparable_summary).strip():
        raise ValueError("comparable_summary cannot be empty")
    raw_previous_actuals = context["previous_actuals"]
    if not isinstance(raw_previous_actuals, Sequence) or isinstance(
        raw_previous_actuals, (str, bytes)
    ):
        raise ValueError("cached report context has an invalid previous_actuals format")
    previous_actuals = {
        str(item["metric"]): float(item["value"])
        for item in raw_previous_actuals
        if isinstance(item, Mapping)
    }
    changes = news_analysis(
        canonical,
        news_start,
        cutoff,
        normalized_metrics,
        target_period=display_target,
        guidance_summary=guidance_summary,
        comparable_summary=comparable_summary,
        previous_actuals=previous_actuals,
        data_dir=root,
        news_path=news_file,
        offline_data_dir=offline_data_dir,
        details_output_path=details_output_path,
        api_key=api_key,
        model=model,
        timeout=timeout,
        event_analyzer=event_analyzer,
        impact_analyzer=impact_analyzer,
    )
    predictions = {}
    expected_units = {item["label"]: item["unit"] for item in normalized_metrics}
    for metric, impact in changes.items():
        change = impact.get("predicted_change")
        if isinstance(change, bool) or not isinstance(change, (int, float)):
            raise ValueError(f"predicted change for {metric} must be numeric")
        unit = str(impact.get("unit") or "").strip()
        if unit != expected_units.get(metric):
            raise ValueError(f"predicted change used the wrong unit for {metric}")
        predictions[metric] = {
            "predicted_value": float(previous_actuals[metric]) + float(change),
            "unit": unit,
        }

    if details_output_path is not None:
        audit_path = Path(details_output_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        audit.update(
            {
                "target_period": display_target,
                "target_report_date": target_report_date.isoformat(),
                "previous_period": previous_period,
                "previous_report_date": previous_report_date.isoformat(),
                "report_context_file": str(report_file),
                "guidance_file": str(guidance_file),
                "news_file": str(news_file),
                "predicted_changes": changes,
                "predicted_values": predictions,
            }
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return predictions


def forecast_earnings(
    requests: Sequence[Mapping[str, object]], **kwargs: object
) -> dict[str, dict[str, dict[str, float | str]]]:
    """Forecast multiple companies from request mappings with company/period/metrics keys."""
    if not requests:
        raise ValueError("at least one forecast request is required")
    results = {}
    for request in requests:
        company = str(request.get("company") or request.get("company_name") or "").strip()
        period = str(request.get("period") or request.get("target_period") or "").strip()
        if not company or not period:
            raise ValueError("each request needs company and period")
        metrics = request.get("metrics")
        if metrics is None:
            metrics = configured_metrics(company)
        if not isinstance(metrics, (Sequence, Mapping)) or isinstance(
            metrics, (str, bytes)
        ):
            raise ValueError("each request needs company, period and metrics with explicit units")
        canonical = clean_company_name(company)
        if canonical in results:
            raise ValueError(f"duplicate company forecast request: {canonical}")
        results[canonical] = forecast_company(
            company, period, metrics, **kwargs  # type: ignore[arg-type]
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True, help="Company name or configured ticker")
    parser.add_argument(
        "--period", required=True, help="Target period, e.g. FY2026Q2, FY2026H1, or FY2026"
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Target metric as 'Label|Unit'; repeat it. Defaults to challenge config.",
    )
    parser.add_argument("--as-of", help="Information cutoff, YYYY-MM-DD; defaults to today")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-file", type=Path, help="Cached report context JSON")
    parser.add_argument("--guidance-file", type=Path)
    parser.add_argument("--news-file", type=Path)
    parser.add_argument(
        "--offline-data-dir",
        type=Path,
        default=DEFAULT_CONFIG.parent / "offline-data",
        help="Local company corpus used for cutoff-safe calculation context",
    )
    parser.add_argument("--details-output", type=Path, help="Optional detailed audit JSON")
    parser.add_argument("--output", type=Path, help="Optional predicted-values JSON output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = args.metric or configured_metrics(args.company)
        result = forecast_company(
            args.company,
            args.period,
            metrics,
            data_dir=args.data_dir,
            report_path=args.report_file,
            guidance_path=args.guidance_file,
            news_path=args.news_file,
            as_of_date=args.as_of,
            offline_data_dir=args.offline_data_dir,
            details_output_path=args.details_output,
            model=args.model,
            timeout=args.timeout,
        )
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
