# Forecast analysis agents

This package runs the full forecast pipeline. It researches the target and previous report dates,
extracts the prior reported actuals, collects guidance and news when their cache files are missing,
estimates each total metric change versus the prior report, and applies that change to the prior
actual. It uses strict structured outputs, cutoff-safe documents from `challenge/offline-data`, and
web search for evidence and calculation inputs.

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
calculations and URLs.

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
