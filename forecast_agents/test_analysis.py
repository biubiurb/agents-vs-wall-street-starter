import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from forecast_agents._common import normalize_metrics, normalize_period, parse_iso_date
from forecast_agents.comparable_analysis import comparable_analysis
from forecast_agents.guidance_analysis import _guidance_prompt, guidance_analysis
from forecast_agents.main import _validate_report_context, forecast_company
from forecast_agents.news_analysis import (
    _calculate_derived_eps,
    _impact_prompt,
    _load_offline_context,
    _validate_impacts,
    news_analysis,
)


class ForecastAnalysisTests(unittest.TestCase):
    @staticmethod
    def _comparable_payload(report_date="2026-08-10", projected_change=2.0):
        direction = (
            "increase" if projected_change > 0 else "decrease" if projected_change < 0 else "neutral"
        )
        return {
            "comparables": [
                {
                    "company": "Peer Co",
                    "ticker": "PEER",
                    "report_date": report_date,
                    "reported_period": "Q2 2026",
                    "why_comparable": "Both sell into the same industrial end markets.",
                    "shared_exposures": ["Industrial production", "Steel input costs"],
                    "earnings_source_urls": ["https://peer.example/q2"],
                    "earnings_impacts": [
                        {
                            "factor": "Industrial demand",
                            "scope": "industry",
                            "direction": "positive",
                            "comparable_impact": "Peer volume rose 4% and lifted revenue.",
                            "target_read_through": "Shared end markets support target volume.",
                            "source_urls": ["https://peer.example/q2"],
                        }
                    ],
                }
            ],
            "metric_adjustments": [
                {
                    "metric": "Revenue",
                    "unit": "USDm",
                    "projected_change": projected_change,
                    "direction": direction,
                    "calculation": "USD100m baseline x 4% peer growth x 50% exposure = USD2m",
                    "assumptions": ["Target exposure is half the peer's"],
                    "comparable_companies": ["Peer Co"],
                    "evidence_urls": ["https://peer.example/q2"],
                }
            ],
        }

    def test_comparable_analysis_summarizes_only_same_season_prior_reports(self):
        calls = []

        def analyzer(company, period, report_date, season_start, cutoff, metrics, peers):
            calls.append((company, period, report_date, season_start, cutoff, metrics, peers))
            return self._comparable_payload()

        summary = comparable_analysis(
            "Example Co",
            "FY2026Q2",
            "2026-08-20",
            ["Revenue|USDm"],
            information_cutoff="2026-08-19",
            comparable_companies=["Peer Co"],
            analyzer=analyzer,
        )

        self.assertEqual(calls[0][3], date(2026, 7, 6))
        self.assertEqual(calls[0][4], date(2026, 8, 19))
        self.assertIn("COMPARABLE: Peer Co (PEER)", summary)
        self.assertIn("IMPACT ON COMPARABLE: Peer volume rose 4%", summary)
        self.assertIn("Revenue: 2 USDm", summary)

    def test_comparable_analysis_rejects_reports_outside_strict_window(self):
        with self.assertRaisesRegex(ValueError, "outside the eligible comparable window"):
            comparable_analysis(
                "Example Co",
                "FY2026Q2",
                "2026-08-20",
                ["Revenue|USDm"],
                analyzer=lambda *_: self._comparable_payload(report_date="2026-06-01"),
            )

    def test_shared_boundary_normalizers_produce_canonical_contracts(self):
        self.assertEqual(normalize_period("Q3 FY2026"), ("Q3 2026", "Q3_2026"))
        self.assertEqual(normalize_period("1H2026"), ("H1 2026", "H1_2026"))
        self.assertEqual(normalize_period("FY 2026"), ("FY2026", "FY2026"))
        self.assertEqual(
            normalize_metrics({"Revenue": "USDm", "Adjusted EPS": "USD / share"}),
            [
                {"label": "Revenue", "unit": "USDm"},
                {"label": "Adjusted EPS", "unit": "USD / share"},
            ],
        )
        self.assertEqual(
            parse_iso_date("2026-08-20T07:00:00-04:00", "report_date"),
            date(2026, 8, 20),
        )

    def test_report_context_has_the_same_shape_before_and_after_caching(self):
        metrics = [{"label": "Revenue", "unit": "USDm"}]
        researched = {
            "target_period": "FY2026Q2",
            "target_report_date": "2026-08-20T11:00:00Z",
            "previous_report_date": "2026-05-20",
            "previous_period": "Q1 FY2026",
            "previous_actuals": [
                {
                    "metric": "Revenue",
                    "unit": "USDm",
                    "value": 100,
                    "source_url": "https://example.com/q1",
                }
            ],
        }

        normalized = _validate_report_context(researched, metrics, "Q2 2026")
        reloaded = _validate_report_context(normalized, metrics, "FY2026Q2")

        self.assertEqual(reloaded, normalized)
        self.assertEqual(normalized["target_report_date"], "2026-08-20")
        self.assertEqual(normalized["previous_period"], "Q1 2026")

    def test_context_aware_impact_analyzer_receives_normalized_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            news_file = data / "news_Example_Co_2026-05-21_to_2026-08-19.txt"
            news_file.write_text("No material news.", encoding="utf-8")

            def impact_analyzer(
                company, metrics, guidance, events, *, target_period, previous_actuals
            ):
                self.assertEqual(target_period, "Q2 2026")
                self.assertEqual(metrics, [{"label": "Revenue", "unit": "USDm"}])
                self.assertEqual(previous_actuals, {"Revenue": 100.0})
                return {
                    "metric_impacts": [
                        {
                            "metric": "Revenue",
                            "unit": "USDm",
                            "projected_change": 5.0,
                            "direction": "increase",
                            "metric_total_calculation": "105 - 100",
                            "event_contributions": [
                                {
                                    "event_id": "GUIDANCE",
                                    "affected_line": "Revenue",
                                    "transmission_path": "Guidance bridge",
                                    "assumptions": [],
                                    "calculation": "105 - 100",
                                    "projected_change": 5.0,
                                    "evidence_urls": [],
                                }
                            ],
                        }
                    ]
                }

            result = news_analysis(
                "Example Co",
                "2026-05-21",
                "2026-08-19",
                {"Revenue": "USDm"},
                target_period="FY2026Q2",
                guidance_summary="Revenue guidance is 105.",
                previous_actuals={"Revenue": 100.0},
                news_path=news_file,
                offline_data_dir=data / "offline",
                event_analyzer=lambda *_: {"events": []},
                impact_analyzer=impact_analyzer,
            )

        self.assertEqual(result["Revenue"]["predicted_change"], 5.0)

    def test_guidance_prompt_preserves_exact_timeframes(self):
        prompt = _guidance_prompt("Example Co", "Q2 2026", "Example source")

        self.assertIn("Pay particular attention to the projection's timeframe", prompt)
        self.assertIn("Do not assume that guidance", prompt)
        self.assertIn("different timeframes", prompt)
        self.assertIn("timeframe_scope", prompt)

    def test_impact_prompt_requires_period_alignment_and_event_specific_routing(self):
        prompt = _impact_prompt(
            "Example Co",
            date.fromisoformat("2026-05-01"),
            date.fromisoformat("2026-08-01"),
            [{"label": "Revenue", "unit": "USDm"}],
            "FY2026Q3",
            "FY revenue guidance is USD1,000m.",
            [],
            "No offline context.",
        )

        self.assertIn("Target reporting period: FY2026Q3", prompt)
        self.assertIn("ANNUAL/RUN-RATE ROUTE", prompt)
        self.assertIn("DISCRETE/TIMED ROUTE", prompt)
        self.assertIn("MIXED ROUTE", prompt)
        self.assertIn("Do not divide by four or two mechanically", prompt)
        self.assertIn("EPS IS A DERIVED OUTPUT", prompt)
        self.assertIn("Do not copy direct EPS guidance", prompt)

    def test_eps_bridge_derives_eps_from_the_income_statement(self):
        derived = _calculate_derived_eps(
            {
                "route": "gross_margin",
                "revenue": 1000.0,
                "gross_margin_percent": 40.0,
                "operating_expenses": 300.0,
                "other_operating_income": 0.0,
                "operating_margin_percent": None,
                "operating_income": 100.0,
                "net_nonoperating_expense": 20.0,
                "pretax_adjustments": 0.0,
                "tax_rate_percent": 25.0,
                "net_income_adjustments": 0.0,
                "diluted_shares": 100.0,
                "eps_unit_multiplier": 1.0,
            }
        )

        self.assertAlmostEqual(derived["operating_income"], 100.0)
        self.assertAlmostEqual(derived["pretax_income"], 80.0)
        self.assertAlmostEqual(derived["net_income"], 60.0)
        self.assertAlmostEqual(derived["derived_eps"], 0.60)

        pence = _calculate_derived_eps(
            {
                "route": "operating_income",
                "revenue": None,
                "gross_margin_percent": None,
                "operating_expenses": None,
                "other_operating_income": 0.0,
                "operating_margin_percent": None,
                "operating_income": 20.1,
                "net_nonoperating_expense": 6.7,
                "pretax_adjustments": 0.0,
                "tax_rate_percent": 44.8,
                "net_income_adjustments": 0.0,
                "diluted_shares": 1595.7,
                "eps_unit_multiplier": 100.0,
            }
        )
        self.assertAlmostEqual(pence["derived_eps"], 0.46355, places=5)

    def test_eps_change_is_recomputed_and_linked_to_requested_revenue(self):
        bridge = {
            "route": "operating_margin",
            "linked_revenue_metric": "Revenue",
            "linked_operating_income_metric": "",
            "revenue": 1000.0,
            "gross_margin_percent": None,
            "operating_expenses": None,
            "other_operating_income": 0.0,
            "operating_margin_percent": 10.0,
            "operating_income": 100.0,
            "net_nonoperating_expense": 20.0,
            "pretax_adjustments": 0.0,
            "tax_rate_percent": 25.0,
            "net_income_adjustments": 0.0,
            "diluted_shares": 100.0,
            "eps_unit_multiplier": 1.0,
            "calculation": "((1000 x 10%) - 20) x 75% / 100 = 0.60",
            "assumptions": [],
            "evidence_urls": [],
        }
        payload = {
            "metric_impacts": [
                {
                    "metric": "Revenue",
                    "unit": "USDm",
                    "projected_change": 100.0,
                    "direction": "increase",
                    "metric_total_calculation": "1000 - 900",
                    "event_contributions": [
                        {"event_id": "GUIDANCE", "projected_change": 100.0}
                    ],
                    "earnings_bridge": None,
                },
                {
                    "metric": "Adjusted EPS",
                    "unit": "USD / share",
                    "projected_change": 0.10,
                    "direction": "increase",
                    "metric_total_calculation": "Derived EPS 0.60 - prior EPS 0.50",
                    "event_contributions": [
                        {"event_id": "GUIDANCE", "projected_change": 0.10}
                    ],
                    "earnings_bridge": bridge,
                },
            ]
        }

        values, details = _validate_impacts(
            payload,
            [
                {"label": "Revenue", "unit": "USDm"},
                {"label": "Adjusted EPS", "unit": "USD / share"},
            ],
            {"GUIDANCE"},
            {"Revenue": 900.0, "Adjusted EPS": 0.50},
        )

        self.assertAlmostEqual(values["Adjusted EPS"]["predicted_change"], 0.10)
        eps_detail = next(item for item in details if item["metric"] == "Adjusted EPS")
        self.assertAlmostEqual(
            eps_detail["derived_eps_calculation"]["derived_eps"], 0.60
        )

        unlinked = json.loads(json.dumps(payload))
        unlinked["metric_impacts"][1]["earnings_bridge"]["linked_revenue_metric"] = ""
        with self.assertRaisesRegex(ValueError, "must link a requested revenue-level metric"):
            _validate_impacts(
                unlinked,
                [
                    {"label": "Revenue", "unit": "USDm"},
                    {"label": "Adjusted EPS", "unit": "USD / share"},
                ],
                {"GUIDANCE"},
                {"Revenue": 900.0, "Adjusted EPS": 0.50},
            )

    def test_eps_bridge_rejects_an_unreconciled_prediction(self):
        with self.assertRaisesRegex(ValueError, "does not reconcile with earnings_bridge"):
            _validate_impacts(
                {
                    "metric_impacts": [
                        {
                            "metric": "Adjusted EPS",
                            "unit": "USD / share",
                            "projected_change": 0.30,
                            "direction": "increase",
                            "metric_total_calculation": "Unreconciled direct guidance",
                            "event_contributions": [
                                {"event_id": "GUIDANCE", "projected_change": 0.30}
                            ],
                            "earnings_bridge": {
                                "route": "operating_income",
                                "linked_revenue_metric": "",
                                "linked_operating_income_metric": "",
                                "revenue": None,
                                "gross_margin_percent": None,
                                "operating_expenses": None,
                                "other_operating_income": 0.0,
                                "operating_margin_percent": None,
                                "operating_income": 100.0,
                                "net_nonoperating_expense": 20.0,
                                "pretax_adjustments": 0.0,
                                "tax_rate_percent": 25.0,
                                "net_income_adjustments": 0.0,
                                "diluted_shares": 100.0,
                                "eps_unit_multiplier": 1.0,
                            },
                        }
                    ]
                },
                [{"label": "Adjusted EPS", "unit": "USD / share"}],
                {"GUIDANCE"},
                {"Adjusted EPS": 0.50},
            )

    def test_main_agent_fetches_missing_inputs_uses_cache_and_returns_absolute_forecast(self):
        calls = {"report": 0, "guidance": 0, "news": 0, "comparables": 0}

        def report_researcher(company, period, metrics):
            calls["report"] += 1
            return {
                "target_report_date": "2026-08-20",
                "previous_report_date": "2026-05-20",
                "previous_period": "FY2026Q1",
                "previous_actuals": [
                    {
                        "metric": "Revenue",
                        "unit": "USDm",
                        "value": 100.0,
                        "source_url": "https://example.com/q1",
                    }
                ],
            }

        def guidance_collector(company, period, *, output_path, **kwargs):
            calls["guidance"] += 1
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text("Revenue guidance is 115.", encoding="utf-8")
            return output_path

        def news_collector(company, start, end, *, output_path, **kwargs):
            calls["news"] += 1
            Path(output_path).write_text("No material post-guidance news.", encoding="utf-8")
            return output_path

        guidance_analyzer = lambda *_: {
            "guidance": [
                {
                    "metric": "Revenue",
                    "guidance_type": "quantitative",
                    "projection": "115",
                    "projection_value": 115,
                    "unit": "USDm",
                    "target_period": "Q2 2026",
                    "timeframe_scope": "quarter",
                    "basis": "Reported",
                    "rationale_drivers": ["Volume growth"],
                }
            ]
        }
        event_analyzer = lambda *_: {"events": []}

        def comparable_analyzer(*_):
            calls["comparables"] += 1
            return self._comparable_payload()

        impact_analyzer = lambda *_: {
            "metric_impacts": [
                {
                    "metric": "Revenue",
                    "unit": "USDm",
                    "projected_change": 15.0,
                    "direction": "increase",
                    "metric_total_calculation": "115 - 100 = 15",
                    "event_contributions": [
                        {
                            "event_id": "GUIDANCE",
                            "affected_line": "Revenue",
                            "transmission_path": "Management revenue outlook",
                            "assumptions": [],
                            "calculation": "115 - 100",
                            "projected_change": 15.0,
                            "evidence_urls": ["https://example.com/q1"],
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            kwargs = {
                "data_dir": data,
                "as_of_date": "2026-08-19",
                "offline_data_dir": data / "offline",
                "report_researcher": report_researcher,
                "guidance_collector": guidance_collector,
                "news_collector": news_collector,
                "guidance_analyzer": guidance_analyzer,
                "comparable_analyzer": comparable_analyzer,
                "event_analyzer": event_analyzer,
                "impact_analyzer": impact_analyzer,
            }
            first = forecast_company("Example Co", "FY2026Q2", ["Revenue|USDm"], **kwargs)
            second = forecast_company("Example Co", "FY2026Q2", ["Revenue|USDm"], **kwargs)

        self.assertEqual(first, {"Revenue": {"predicted_value": 115.0, "unit": "USDm"}})
        self.assertEqual(second, first)
        self.assertEqual(calls, {"report": 1, "guidance": 1, "news": 1, "comparables": 2})

    def test_news_analysis_reconciles_comparable_summary_with_guidance_and_news(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            news_file = data / "news_Example_Co_2026-05-21_to_2026-08-19.txt"
            news_file.write_text("No separate target-company news.", encoding="utf-8")
            comparable_summary = "PRELIMINARY COMPARABLE-ONLY METRIC ADD-ONS:\n- Revenue: 2 USDm"

            def impact(company, metrics, guidance, events, *, comparable_summary):
                self.assertIn("Revenue: 2 USDm", comparable_summary)
                return {
                    "metric_impacts": [
                        {
                            "metric": "Revenue",
                            "unit": "USDm",
                            "projected_change": 7.0,
                            "direction": "increase",
                            "metric_total_calculation": "5 guidance + 2 comparable",
                            "event_contributions": [
                                {
                                    "event_id": "GUIDANCE",
                                    "affected_line": "Revenue",
                                    "transmission_path": "Management guidance",
                                    "assumptions": [],
                                    "calculation": "105 - 100",
                                    "projected_change": 5.0,
                                    "evidence_urls": [],
                                },
                                {
                                    "event_id": "COMPARABLES",
                                    "affected_line": "Revenue",
                                    "transmission_path": "Shared industry demand",
                                    "assumptions": ["Incremental to guidance"],
                                    "calculation": "Peer read-through = 2",
                                    "projected_change": 2.0,
                                    "evidence_urls": ["https://peer.example/q2"],
                                },
                            ],
                        }
                    ]
                }

            result = news_analysis(
                "Example Co",
                "2026-05-21",
                "2026-08-19",
                ["Revenue|USDm"],
                guidance_summary="Revenue guidance is 105.",
                comparable_summary=comparable_summary,
                previous_actuals={"Revenue": 100.0},
                news_path=news_file,
                offline_data_dir=data / "offline",
                event_analyzer=lambda *_: {"events": []},
                impact_analyzer=impact,
            )

        self.assertEqual(result["Revenue"]["predicted_change"], 7.0)

    def test_offline_context_respects_information_cutoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "offline-data" / "example"
            root.mkdir(parents=True)
            for name, published, marker in (
                ("prior.md", "2026-05-30", "SAFE BASELINE"),
                ("future.md", "2026-06-02", "FUTURE RESULT"),
            ):
                (root / name).write_text(
                    "---\n"
                    'company: "Example Co"\n'
                    f'published_at: "{published}"\n'
                    "---\n"
                    f"{marker}\n",
                    encoding="utf-8",
                )

            context, paths = _load_offline_context(
                "Example Co", date.fromisoformat("2026-05-31"), root.parent
            )

        self.assertIn("SAFE BASELINE", context)
        self.assertNotIn("FUTURE RESULT", context)
        self.assertEqual(len(paths), 1)

    def test_guidance_analysis_reads_collector_file_and_renders_multiple_drivers(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            source = data / "guidance_Example_Co_Q1_2026.txt"
            source.write_text("Revenue outlook: 5% to 7%, led by price and volume.", encoding="utf-8")

            def fake_analyzer(company, quarter, text):
                self.assertEqual((company, quarter), ("Example Co", "Q1 2026"))
                self.assertIn("Revenue outlook", text)
                return {
                    "guidance": [
                        {
                            "metric": "Revenue growth",
                            "guidance_type": "quantitative",
                            "projection": "5% to 7%",
                            "projection_value": None,
                            "unit": "%",
                            "target_period": "FY 2026",
                            "timeframe_scope": "full_year",
                            "basis": "Reported year-over-year",
                            "rationale_drivers": ["Pricing tailwind", "Volume recovery"],
                        }
                    ]
                }

            result = guidance_analysis(
                "Example Co", "FY2026Q1", data_dir=data, analyzer=fake_analyzer
            )

        self.assertIn("METRIC: Revenue growth", result)
        self.assertIn("PROJECTION/GUIDANCE: 5% to 7%", result)
        self.assertIn("TIMEFRAME/TARGET PERIOD: FY 2026", result)
        self.assertIn("TIMEFRAME SCOPE: Full Year", result)
        self.assertIn("- Pricing tailwind", result)
        self.assertIn("- Volume recovery", result)

    def test_guidance_analysis_falls_back_to_qualitative_web_guidance(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            source = data / "guidance_Example_Co_Q1_2026.txt"
            source.write_text("No numerical outlook in the collected file.", encoding="utf-8")
            calls = []

            def fallback(company, period):
                calls.append((company, period))
                return {
                    "guidance": [
                        {
                            "metric": "Operating margin",
                            "guidance_type": "qualitative",
                            "projection": None,
                            "projection_value": None,
                            "unit": None,
                            "target_period": "FY 2026",
                            "timeframe_scope": "full_year",
                            "basis": "Management commentary",
                            "rationale_drivers": [
                                "Management expects labour inflation to pressure margins."
                            ],
                        }
                    ]
                }

            result = guidance_analysis(
                "Example Co",
                "FY2026Q1",
                data_dir=data,
                analyzer=lambda *_: {"guidance": []},
                fallback_analyzer=fallback,
            )

        self.assertEqual(calls, [("Example Co", "Q1 2026")])
        self.assertIsNotNone(result)
        self.assertIn("METRIC: Operating margin", result)
        self.assertIn("GUIDANCE TYPE: Qualitative", result)
        self.assertIn("NUMERICAL GUIDANCE: None stated", result)
        self.assertNotIn("UNIT:", result)

    def test_guidance_analysis_returns_none_when_both_passes_find_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "guidance_Example_Co_Q1_2026.txt").write_text(
                "No outlook.", encoding="utf-8"
            )
            result = guidance_analysis(
                "Example Co",
                "FY2026Q1",
                data_dir=data,
                analyzer=lambda *_: {"guidance": []},
                fallback_analyzer=lambda *_: {"guidance": []},
            )

        self.assertIsNone(result)

    def test_news_prompt_supports_none_and_qualitative_guidance(self):
        common = (
            "Example Co",
            date.fromisoformat("2026-05-01"),
            date.fromisoformat("2026-08-01"),
            [{"label": "Revenue", "unit": "USDm"}],
            "FY2026Q3",
        )
        no_guidance = _impact_prompt(
            *common,
            None,
            [],
            "No offline context.",
            {"Revenue": 100.0},
        )
        qualitative = _impact_prompt(
            *common,
            "METRIC: Revenue\nGUIDANCE TYPE: Qualitative\n"
            "MAIN RATIONALE/DRIVERS:\n- Demand is improving",
            [],
            "No offline context.",
            {"Revenue": 100.0},
        )

        self.assertIn("GUIDANCE MODE: NONE", no_guidance)
        self.assertIn("forecast using", no_guidance)
        self.assertIn("qualifying comparable earnings", no_guidance)
        self.assertIn("GUIDANCE contribution is not permitted", no_guidance)
        self.assertIn("GUIDANCE MODE: QUALITATIVE ONLY", qualitative)
        self.assertIn("related metric and rationale", qualitative)

    def test_news_analysis_uses_news_only_when_guidance_is_none(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            news_file = data / "news_Example_Co_2026-05-01_to_2026-05-31.txt"
            news_file.write_text("A new contract was announced.", encoding="utf-8")
            event = {
                "event_id": "E1",
                "title": "New customer contract",
                "first_reported_at": "2026-05-10",
                "last_reported_at": "2026-05-10",
                "summary": "The company won incremental business.",
                "source_headlines": ["New customer contract"],
                "source_urls": ["https://example.com/contract"],
                "financial_statement_lines": ["Revenue"],
            }

            def impact(company, metrics, guidance, events):
                self.assertIsNone(guidance)
                return {
                    "metric_impacts": [
                        {
                            "metric": "Revenue",
                            "unit": "USDm",
                            "projected_change": 10.0,
                            "direction": "increase",
                            "metric_total_calculation": "Contract revenue = USD10m",
                            "event_contributions": [
                                {
                                    "event_id": "E1",
                                    "affected_line": "Revenue",
                                    "transmission_path": "Contract increases sales.",
                                    "assumptions": [],
                                    "calculation": "USD10m recognized in period",
                                    "projected_change": 10.0,
                                    "evidence_urls": ["https://example.com/contract"],
                                }
                            ],
                        }
                    ]
                }

            result = news_analysis(
                "Example Co",
                "2026-05-01",
                "2026-05-31",
                ["Revenue|USDm"],
                guidance_summary=None,
                previous_actuals={"Revenue": 100.0},
                news_path=news_file,
                offline_data_dir=data / "offline",
                event_analyzer=lambda *_: {"events": [event]},
                impact_analyzer=impact,
            )

        self.assertEqual(
            result,
            {"Revenue": {"predicted_change": 10.0, "unit": "USDm"}},
        )

    def test_news_analysis_consolidates_events_and_returns_numeric_dictionary(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            news_file = data / "news_Example_Co_2026-05-01_to_2026-05-31.txt"
            news_file.write_text(
                "RELEASE TIME: 2026-05-10\nHEADLINE: Plant outage\n\n"
                "RELEASE TIME: 2026-05-11\nHEADLINE: Plant outage update\n",
                encoding="utf-8",
            )

            def fake_events(company, start, end, text):
                self.assertEqual(text.count("Plant outage"), 2)
                return {
                    "events": [
                        {
                            "event_id": "E1",
                            "title": "Plant outage",
                            "first_reported_at": "2026-05-10",
                            "last_reported_at": "2026-05-11",
                            "summary": "Two reports describe one outage.",
                            "source_headlines": ["Plant outage", "Plant outage update"],
                            "source_urls": ["https://example.com/1", "https://example.com/2"],
                            "financial_statement_lines": ["Revenue", "COGS"],
                        }
                    ]
                }

            def fake_impacts(company, metrics, guidance, events):
                self.assertEqual(len(events), 1)
                self.assertIn("Revenue guidance", guidance)
                return {
                    "metric_impacts": [
                        {
                            "metric": "Revenue",
                            "unit": "USDm",
                            "projected_change": -25.0,
                            "direction": "decrease",
                            "metric_total_calculation": "Lost volume of USD25m.",
                            "event_contributions": [
                                {
                                    "event_id": "E1",
                                    "affected_line": "Revenue",
                                    "transmission_path": "Lost production reduces sales.",
                                    "assumptions": ["No inventory offset"],
                                    "calculation": "-USD25m sales",
                                    "projected_change": -25.0,
                                    "evidence_urls": ["https://example.com/1"],
                                }
                            ],
                        },
                        {
                            "metric": "Adjusted EPS",
                            "unit": "USD / share",
                            "projected_change": -0.08,
                            "direction": "decrease",
                            "metric_total_calculation": "After-tax profit divided by shares.",
                            "event_contributions": [
                                {
                                    "event_id": "E1",
                                    "affected_line": "Adjusted net income",
                                    "transmission_path": "Lower sales reduce after-tax profit.",
                                    "assumptions": ["Stable margin and share count"],
                                    "calculation": "After-tax loss / diluted shares = -0.08",
                                    "projected_change": -0.08,
                                    "evidence_urls": ["https://example.com/1"],
                                }
                            ],
                        },
                    ]
                }

            audit = data / "audit.json"
            result = news_analysis(
                "Example Co",
                "2026-05-01",
                "2026-05-31",
                ["Revenue|USDm", "Adjusted EPS|USD / share"],
                guidance_summary="Revenue guidance assumes normal operations.",
                data_dir=data,
                offline_data_dir=data / "offline-data",
                event_analyzer=fake_events,
                impact_analyzer=fake_impacts,
                details_output_path=audit,
            )

            audit_payload = json.loads(audit.read_text(encoding="utf-8"))

        self.assertEqual(
            result,
            {
                "Revenue": {"predicted_change": -25.0, "unit": "USDm"},
                "Adjusted EPS": {
                    "predicted_change": -0.08,
                    "unit": "USD / share",
                },
            },
        )
        self.assertEqual(len(audit_payload["events"]), 1)

    def test_news_analysis_requires_every_requested_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "news_Example_Co_2026-05-01_to_2026-05-31.txt").write_text(
                "No qualifying news.", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "omitted metrics"):
                news_analysis(
                    "Example Co",
                    "2026-05-01",
                    "2026-05-31",
                    ["Revenue|USDm"],
                    guidance_summary="Flat revenue guidance.",
                    data_dir=data,
                    offline_data_dir=data / "offline-data",
                    event_analyzer=lambda *_: {"events": []},
                    impact_analyzer=lambda *_: {"metric_impacts": []},
                )


if __name__ == "__main__":
    unittest.main()
