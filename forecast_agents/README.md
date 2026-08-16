# Forecast analysis agents

This package runs the full forecast pipeline. It researches the target and previous report dates,
extracts the prior reported actuals, collects guidance and news when their cache files are missing,
analyzes same-season comparable earnings released just before the target company, estimates each
total metric change versus the prior report, and applies that change to the prior actual. It uses
strict structured outputs, cutoff-safe documents from `challenge/offline-data`, and web search for
evidence and calculation inputs.

EPS is not forecast as an independent change. For each requested earnings-per-share metric, the
impact stage must build a target-period earnings bridge from revenue and margin (or operating
income), through non-operating items and tax, to net income and diluted shares. Python recomputes EPS
from that bridge, checks any linked requested revenue or operating-income metric, and rejects a
direct EPS estimate that does not reconcile. When an absolute revenue, sales, or net-fees metric is
requested, it must be linked to EPS through a gross-margin or operating-margin route.

Guidance extraction uses two passes. It first analyzes the collected guidance file. If that produces
no guidance, it searches the company's official investor-relations and newsroom pages, verified
company or executive social media, direct management interviews, and reputable news coverage. It
returns quantitative guidance when management stated a number, qualitative guidance tied to the
relevant metric when management only gave rationale or direction, and `None` only when neither pass
finds any management outlook.

Comparable analysis uses a strict 45-day window ending at the information cutoff and never later
than the day before the target report. It verifies each peer's earnings release date and fiscal
period, keeps only transferable macro or industry factors from qualifying releases or calls,
describes how those factors affected each peer, and produces preliminary signed metric add-ons. The
final impact stage reconciles those add-ons with guidance and target-company news and keeps only the
incremental read-through, preventing the same factor from being counted twice.

Set `OPENAI_API_KEY` and run the main agent from the repository root:

```bash
python3 -m forecast_agents.main \
  --company HD \
  --period FY2026Q2 \
  --as-of 2026-08-16 \
  --details-output data/HD-forecast-audit.json \
  --output data/HD-forecast.json
```

For a non-configured company, repeat `--metric 'Metric label|Unit'`. For configured challenge
companies, omitting `--metric` loads the labels and units from `challenge/companies.json`.

The command prints absolute predicted values with their units:

```json
{
  "Net sales": {
    "predicted_value": 45250.0,
    "unit": "USDm"
  },
  "Adjusted diluted EPS": {
    "predicted_value": 4.61,
    "unit": "USD / share"
  },
  "Comparable sales, total company": {
    "predicted_value": 1.1,
    "unit": "%"
  }
}
```

Quarter, half-year, and full-year labels are accepted in common fiscal formats and normalized to one
canonical form. The information cutoff defaults to today and is capped at the day before the target
report. Report context is cached as JSON; legacy cache shapes are normalized on read. Guidance and
news are cached as text, so repeated runs reuse collected evidence. The optional details file
contains prior actuals, predicted changes, consolidated events, transmission paths, assumptions,
calculations, the comparable-earnings summary, and URLs. For EPS metrics it also contains the full
earnings bridge and recomputed operating income, pretax income, net income, and EPS.

The impact stage adapts to the guidance result: quantitative guidance supplies the numerical
baseline; qualitative guidance is combined with news and converted into an explicitly assumed
metric effect; `None` switches the model to news-only forecasting and forbids a synthetic guidance
contribution.

When guidance covers the full fiscal year but the target is an intermediate period, the impact agent
builds an explicit period bridge using reported year-to-date actuals, management phasing, seasonality
and segment mix. Annual or recurring news effects are applied to FY guidance before phasing; discrete
period-specific items are added after FY guidance is phased; mixed events are split between the two
routes.

Python use:

```python
from forecast_agents.main import forecast_company

predictions = forecast_company(
    "HD", "FY2026Q2",
    {"Net sales": "USDm", "Adjusted diluted EPS": "USD / share"},
    as_of_date="2026-08-16",
)
```
