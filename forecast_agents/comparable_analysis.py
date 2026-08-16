#!/usr/bin/env python3
"""Summarize same-season comparable earnings released before a target company reports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, timedelta

try:
    from ._common import (
        DEFAULT_MODEL,
        clean_company_name,
        normalize_mapping_result,
        normalize_metrics,
        normalize_period,
        parse_iso_date,
        request_structured_output,
    )
except ImportError:  # Allow direct imports during local experiments.
    from _common import (  # type: ignore[no-redef]
        DEFAULT_MODEL,
        clean_company_name,
        normalize_mapping_result,
        normalize_metrics,
        normalize_period,
        parse_iso_date,
        request_structured_output,
    )


DEFAULT_SEASON_LOOKBACK_DAYS = 45


def _comparable_schema(metrics: Sequence[Mapping[str, str]]) -> dict[str, object]:
    earnings_impact = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "factor": {"type": "string"},
            "scope": {"type": "string", "enum": ["macro", "industry"]},
            "direction": {
                "type": "string",
                "enum": ["positive", "negative", "mixed", "neutral"],
            },
            "comparable_impact": {"type": "string"},
            "target_read_through": {"type": "string"},
            "source_urls": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "factor",
            "scope",
            "direction",
            "comparable_impact",
            "target_read_through",
            "source_urls",
        ],
    }
    comparable = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "company": {"type": "string"},
            "ticker": {"type": "string"},
            "report_date": {"type": "string"},
            "reported_period": {"type": "string"},
            "why_comparable": {"type": "string"},
            "shared_exposures": {"type": "array", "items": {"type": "string"}},
            "earnings_source_urls": {"type": "array", "items": {"type": "string"}},
            "earnings_impacts": {"type": "array", "items": earnings_impact},
        },
        "required": [
            "company",
            "ticker",
            "report_date",
            "reported_period",
            "why_comparable",
            "shared_exposures",
            "earnings_source_urls",
            "earnings_impacts",
        ],
    }
    adjustment = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metric": {"type": "string", "enum": [item["label"] for item in metrics]},
            "unit": {"type": "string", "enum": sorted({item["unit"] for item in metrics})},
            "projected_change": {"type": "number"},
            "direction": {
                "type": "string",
                "enum": ["increase", "decrease", "neutral"],
            },
            "calculation": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "comparable_companies": {"type": "array", "items": {"type": "string"}},
            "evidence_urls": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "metric",
            "unit",
            "projected_change",
            "direction",
            "calculation",
            "assumptions",
            "comparable_companies",
            "evidence_urls",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "comparables": {"type": "array", "items": comparable},
            "metric_adjustments": {"type": "array", "items": adjustment},
        },
        "required": ["comparables", "metric_adjustments"],
    }


def _comparable_prompt(
    company_name: str,
    target_period: str,
    target_report_date: date,
    season_start: date,
    cutoff: date,
    metrics: Sequence[Mapping[str, str]],
    comparable_companies: Sequence[str] | None,
) -> str:
    metric_text = "\n".join(f"- {item['label']} ({item['unit']})" for item in metrics)
    if comparable_companies is None:
        peer_instruction = """No preselected peer list was supplied. Identify only direct public
earnings comparables with clearly shared demand, pricing, input-cost or macro exposures. A broad
sector label or a website competitor list alone is not enough."""
    else:
        peer_instruction = (
            "Use only the following prevalidated peer candidates; omit any that do not meet every "
            "date and evidence rule:\n- " + "\n- ".join(comparable_companies)
        )
    return f"""You are a comparable-earnings read-through analyst for {company_name}'s
{target_period} results, expected on {target_report_date.isoformat()}.

{peer_instruction}

STRICT ELIGIBILITY WINDOW: Include a company only if it released earnings from
{season_start.isoformat()} through {cutoff.isoformat()}, inclusive. This window defines "the same
earnings season" for this analysis. Its report must precede {company_name}'s report. Verify each
release date and reported fiscal period from the issuer's earnings release, filing, investor page
or transcript. Exclude older-quarter commentary, companies that have not yet reported, releases
after the information cutoff, and news that is not part of a qualifying earnings release or call.

For every qualifying comparable:
- Explain the direct comparability and shared exposures.
- Extract only macro or industry factors mentioned in that earnings release or call that could also
  affect {company_name}. Exclude peer-specific execution, acquisitions, accounting items and other
  idiosyncratic factors unless a genuinely shared transmission exists.
- State factually how each factor affected the comparable (revenue, volume, price, mix, margin,
  costs, profit, EPS, cash flow or guidance), with quantities when reported, and cite URLs.
- Explain the read-through to {company_name} without assuming identical exposure or copying the
  peer's percentage change.

Then estimate a central, signed comparable-only add-on for every target metric below:
{metric_text}

The add-on is a preliminary incremental signal, not an absolute forecast. Use the exact listed unit;
for percent metrics use percentage points. Return zero when there is no defensible mapping. Show the
calculation and all scaling, timing and exposure assumptions. Use only qualifying comparables in the
adjustment and its evidence. Do not use stock-price reactions, analyst estimates, target-company
news, later results or facts published after {cutoff.isoformat()}. Return empty comparables and
metric_adjustments lists if none qualify.
"""


def _call_analyzer(
    analyzer: object,
    company_name: str,
    target_period: str,
    target_report_date: date,
    season_start: date,
    cutoff: date,
    metrics: Sequence[Mapping[str, str]],
    comparable_companies: Sequence[str] | None,
) -> object:
    if hasattr(analyzer, "analyze_comparables"):
        function = analyzer.analyze_comparables  # type: ignore[attr-defined]
    elif callable(analyzer):
        function = analyzer
    else:
        raise TypeError("analyzer must be callable or provide analyze_comparables()")
    return function(
        company_name,
        target_period,
        target_report_date,
        season_start,
        cutoff,
        metrics,
        comparable_companies,
    )


def _validate_payload(
    payload: Mapping[str, object],
    company_name: str,
    season_start: date,
    cutoff: date,
    metrics: Sequence[Mapping[str, str]],
    eligible_companies: Sequence[str] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_comparables = payload.get("comparables")
    raw_adjustments = payload.get("metric_adjustments")
    if not isinstance(raw_comparables, Sequence) or isinstance(raw_comparables, (str, bytes)):
        raise ValueError("comparable analyzer output must contain a comparables list")
    if not isinstance(raw_adjustments, Sequence) or isinstance(raw_adjustments, (str, bytes)):
        raise ValueError("comparable analyzer output must contain a metric_adjustments list")
    if not raw_comparables:
        if raw_adjustments:
            raise ValueError("metric adjustments are not allowed without qualifying comparables")
        return [], []

    allowed = (
        {str(item).strip().casefold() for item in eligible_companies}
        if eligible_companies is not None
        else None
    )
    comparables = []
    seen_companies = set()
    for raw in raw_comparables:
        if not isinstance(raw, Mapping):
            raise ValueError("every comparable must be a mapping")
        peer = str(raw.get("company") or "").strip()
        if not peer or peer.casefold() == company_name.casefold() or peer.casefold() in seen_companies:
            raise ValueError("every comparable must be unique and different from the target company")
        if allowed is not None and peer.casefold() not in allowed:
            raise ValueError(f"comparable analyzer used a company outside the eligible list: {peer}")
        report_date = parse_iso_date(raw.get("report_date"), f"report_date for {peer}")
        if not season_start <= report_date <= cutoff:
            raise ValueError(
                f"{peer} reported on {report_date}, outside the eligible comparable window"
            )
        if not str(raw.get("reported_period") or "").strip():
            raise ValueError(f"{peer} is missing its reported fiscal period")
        impacts = raw.get("earnings_impacts")
        sources = raw.get("earnings_source_urls")
        if not isinstance(impacts, Sequence) or isinstance(impacts, (str, bytes)) or not impacts:
            raise ValueError(f"{peer} needs at least one transferable earnings impact")
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
            raise ValueError(f"{peer} needs earnings release evidence")
        normalized = dict(raw)
        normalized["report_date"] = report_date.isoformat()
        comparables.append(normalized)
        seen_companies.add(peer.casefold())

    expected = {item["label"]: item["unit"] for item in metrics}
    adjustments = []
    seen_metrics = set()
    for raw in raw_adjustments:
        if not isinstance(raw, Mapping):
            raise ValueError("every comparable metric adjustment must be a mapping")
        metric = str(raw.get("metric") or "").strip()
        if metric not in expected or metric in seen_metrics:
            raise ValueError(f"unexpected or duplicate comparable metric adjustment: {metric}")
        if str(raw.get("unit") or "").strip() != expected[metric]:
            raise ValueError(f"comparable adjustment used the wrong unit for {metric}")
        value = raw.get("projected_change")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"comparable projected_change for {metric} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"comparable projected_change for {metric} must be finite")
        direction = str(raw.get("direction") or "")
        expected_direction = "increase" if number > 0 else "decrease" if number < 0 else "neutral"
        if direction != expected_direction:
            raise ValueError(f"comparable adjustment direction does not match its value for {metric}")
        referenced = raw.get("comparable_companies")
        if not isinstance(referenced, Sequence) or isinstance(referenced, (str, bytes)):
            raise ValueError(f"comparable_companies for {metric} must be a list")
        unknown = [str(item) for item in referenced if str(item).strip().casefold() not in seen_companies]
        if unknown:
            raise ValueError(f"comparable adjustment for {metric} cites unknown companies: {unknown}")
        normalized = dict(raw)
        normalized["projected_change"] = number
        adjustments.append(normalized)
        seen_metrics.add(metric)
    missing = [metric for metric in expected if metric not in seen_metrics]
    if missing:
        raise ValueError("comparable analysis omitted metric adjustments: " + ", ".join(missing))
    return comparables, adjustments


def _render_summary(
    company_name: str,
    target_period: str,
    season_start: date,
    cutoff: date,
    comparables: Sequence[Mapping[str, object]],
    adjustments: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        f"COMPARABLE EARNINGS ANALYSIS: {company_name} — {target_period}",
        f"STRICT REPORT WINDOW: {season_start.isoformat()} to {cutoff.isoformat()}",
    ]
    for comparable in comparables:
        lines.extend(
            [
                "",
                f"COMPARABLE: {comparable['company']} ({comparable['ticker']})",
                f"REPORT DATE: {comparable['report_date']}",
                f"REPORTED PERIOD: {comparable['reported_period']}",
                f"WHY COMPARABLE: {comparable['why_comparable']}",
                "SHARED EXPOSURES:",
            ]
        )
        lines.extend(f"- {item}" for item in comparable.get("shared_exposures", []))
        lines.append("TRANSFERABLE EARNINGS POINTS:")
        for impact in comparable.get("earnings_impacts", []):
            if not isinstance(impact, Mapping):
                continue
            lines.extend(
                [
                    f"- {impact['factor']} [{str(impact['scope']).upper()}, "
                    f"{str(impact['direction']).upper()}]",
                    f"  IMPACT ON COMPARABLE: {impact['comparable_impact']}",
                    f"  READ-THROUGH TO TARGET: {impact['target_read_through']}",
                ]
            )
            lines.extend(f"  SOURCE: {url}" for url in impact.get("source_urls", []))
        lines.extend(f"EARNINGS SOURCE: {url}" for url in comparable.get("earnings_source_urls", []))

    lines.extend(["", "PRELIMINARY COMPARABLE-ONLY METRIC ADD-ONS:"])
    for adjustment in adjustments:
        lines.extend(
            [
                f"- {adjustment['metric']}: {float(adjustment['projected_change']):g} "
                f"{adjustment['unit']} ({str(adjustment['direction']).upper()})",
                f"  CALCULATION: {adjustment['calculation']}",
            ]
        )
        lines.extend(f"  ASSUMPTION: {item}" for item in adjustment.get("assumptions", []))
        lines.extend(f"  SOURCE: {url}" for url in adjustment.get("evidence_urls", []))
    return "\n".join(lines) + "\n"


def comparable_analysis(
    company_name: str,
    target_period: str,
    target_report_date: str | date,
    metrics: Sequence[str | Mapping[str, object]] | Mapping[str, object],
    *,
    information_cutoff: str | date | None = None,
    season_lookback_days: int = DEFAULT_SEASON_LOOKBACK_DAYS,
    comparable_companies: Sequence[str] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    analyzer: object | None = None,
) -> str | None:
    """Return comparable earnings points and preliminary signed metric add-ons.

    Only earnings released inside the deterministic window ending before the target report are
    accepted. ``None`` means no qualifying comparable with a transferable macro/industry signal was
    found. A supplied ``comparable_companies`` list is an allow-list, not merely a suggestion.
    """
    canonical = clean_company_name(company_name)
    normalized_period = normalize_period(target_period)[0]
    report_date = parse_iso_date(target_report_date, "target_report_date")
    if isinstance(season_lookback_days, bool) or not isinstance(season_lookback_days, int):
        raise ValueError("season_lookback_days must be a positive integer")
    if season_lookback_days < 1:
        raise ValueError("season_lookback_days must be a positive integer")
    season_start = report_date - timedelta(days=season_lookback_days)
    supplied_cutoff = (
        parse_iso_date(information_cutoff, "information_cutoff")
        if information_cutoff is not None
        else report_date - timedelta(days=1)
    )
    cutoff = min(supplied_cutoff, report_date - timedelta(days=1))
    if cutoff < season_start:
        return None
    normalized_metrics = normalize_metrics(metrics)
    eligible = None
    if comparable_companies is not None:
        if isinstance(comparable_companies, (str, bytes)):
            raise ValueError("comparable_companies must be a sequence of company names")
        eligible = []
        seen = set()
        for item in comparable_companies:
            peer = str(item).strip()
            if not peer or peer.casefold() == canonical.casefold() or peer.casefold() in seen:
                continue
            eligible.append(peer)
            seen.add(peer.casefold())
        if not eligible:
            return None

    if analyzer is None:
        payload = request_structured_output(
            _comparable_prompt(
                canonical,
                normalized_period,
                report_date,
                season_start,
                cutoff,
                normalized_metrics,
                eligible,
            ),
            _comparable_schema(normalized_metrics),
            "same_season_comparable_earnings",
            api_key=api_key,
            model=model,
            timeout=timeout,
            web_search=True,
        )
    else:
        payload = normalize_mapping_result(
            _call_analyzer(
                analyzer,
                canonical,
                normalized_period,
                report_date,
                season_start,
                cutoff,
                normalized_metrics,
                eligible,
            ),
            "comparables",
        )
    comparables, adjustments = _validate_payload(
        payload,
        canonical,
        season_start,
        cutoff,
        normalized_metrics,
        eligible,
    )
    if not comparables:
        return None
    return _render_summary(
        canonical, normalized_period, season_start, cutoff, comparables, adjustments
    )


if __name__ == "__main__":
    raise SystemExit("Use `python3 -m forecast_agents.main ...` or import comparable_analysis().")
