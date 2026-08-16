# Historical-document search helper

This small Python script searches the supplied Markdown corpus and writes a cited research note. It does not calculate forecasts, select final historical values or edit the submission workbooks.

Python 3.10 or later is sufficient. There are no external dependencies.

## Run it

Search for the three challenge metrics for Home Depot:

```bash
python3 starter/search.py --company HD
```

Supply one or more narrower searches:

```bash
python3 starter/search.py \
  --company HD \
  --query "net sales" \
  --query "adjusted diluted EPS" \
  --query "comparable sales"
```

The default output is `research/HD.md`. Each result includes the source document, publication date, reporting period, excerpt and any numbers found in that excerpt.

The supported company selectors are:

- `HD` — Home Depot
- `ADI` — Analog Devices
- `HAS` — Hays plc
- `DE` — Deere & Company

The results are leads, not verified financial history. Read the cited document before using a figure and keep reported, adjusted, quarterly and annual values separate.

## Build an earnings-history workbook

`earnings_data.py` researches reported results and the consensus published before each earnings
release. It prioritizes issuer/regulatory documents for actuals and Bloomberg or named institutional
datasets for consensus. Unofficial values carry their source name directly in the value cell, with
full URLs in adjacent audit columns. Each quarter also includes the exact earnings-release timestamp,
normalized to ISO 8601 UTC; approximate times are left as `Not found`. Live research requires
`OPENAI_API_KEY`.

```bash
python3 starter/earnings_data.py \
  --company "Home Depot" \
  --metric "Revenue|USDm" \
  --metric "Adjusted diluted EPS|USD / share" \
  --start "Q1 2024" \
  --end "Q2 2026"
```

The default output is `data/earnings_data.xlsx`, with one tab per company. Repeated runs update that
workbook incrementally: a missing company gets a new tab, while an existing company tab gains missing
columns and new quarter/metric rows. Existing populated cells are preserved and newly available data
fills blank or `Not found` cells. The Python API is
`get_earnings(companies, metrics, start_quarter, end_quarter, ...)`; pass a researcher callable to run
against another data service or a deterministic fixture instead of live web research.

## Use it with Codex or Claude Code

You can ask either harness:

> Run the historical-document search helper for Home Depot, open the resulting research note and help me check the evidence for each challenge metric.

The same script works with another configured company. It identifies the document folder from the company metadata rather than hardcoding the four folder names.

## Test it

```bash
python3 -m unittest discover -s starter -p 'test_*.py'
```
