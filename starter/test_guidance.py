import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import guidance


def online_piece(
    *,
    headline="Example Co raises full-year revenue outlook",
    published_at="2026-05-12T09:30:00-04:00",
    source_name="Example Co Investor Relations",
    source_url="https://investor.example.com/q1-results",
):
    return {
        "headline": headline,
        "published_at": published_at,
        "source_name": source_name,
        "source_url": source_url,
        "summary": "Management updated its outlook when it released first-quarter earnings.",
        "metrics": [
            {
                "metric": "Revenue growth",
                "guidance": "6% to 8%",
                "target_period": "FY 2026",
                "basis": "Reported year-over-year growth",
            }
        ],
    }


class GuidanceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        config = root / "companies.json"
        config.write_text(
            json.dumps(
                {
                    "companies": [
                        {"company": "Example Co", "ticker": "EX", "metrics": []}
                    ]
                }
            ),
            encoding="utf-8",
        )
        documents = root / "offline-data" / "example-co"
        documents.mkdir(parents=True)
        return config, documents

    def _write_document(
        self,
        path: Path,
        *,
        period: str,
        published_at: str,
        body: str,
        source_url=None,
    ) -> None:
        path.write_text(
            "---\n"
            'company: "Example Co"\n'
            'ticker: "EX"\n'
            f'published_at: "{published_at}"\n'
            'document_type: "CALL_TRANSCRIPT"\n'
            f'period: "{period}"\n'
            f"source_url: {json.dumps(source_url)}\n"
            "---\n"
            f"{body}\n",
            encoding="utf-8",
        )

    def test_period_parser_accepts_common_formats_and_matches_complete_periods(self):
        self.assertEqual(str(guidance.Quarter.parse("FY2026Q1")), "Q1 2026")
        self.assertEqual(str(guidance.Quarter.parse("Q1 2026")), "Q1 2026")
        self.assertEqual(str(guidance.parse_reporting_period("1H2026")), "H1 2026")
        self.assertEqual(str(guidance.parse_reporting_period("FY 2026")), "FY2026")
        self.assertTrue(guidance.period_matches("Q4 2025, Q1 2026", "Q1 2026"))
        self.assertFalse(guidance.period_matches("Q1 2025", "Q1 2026"))
        self.assertTrue(guidance.period_matches("FY 2025, Outlook FY 2026", "FY2026"))

    def test_find_quarter_documents_returns_all_and_only_exact_company_period(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, documents = self._fixture(root)
            self._write_document(
                documents / "first.md",
                period="Q1 2026",
                published_at="2026-05-12",
                body="# Q1 release\n\nRevenue guidance was increased.",
            )
            self._write_document(
                documents / "second.md",
                period="Q1 2026, Outlook FY 2026",
                published_at="2026-05-13",
                body="# Q1 call\n\nThe full-year outlook was discussed.",
            )
            self._write_document(
                documents / "wrong-quarter.md",
                period="Q1 2025",
                published_at="2025-05-13",
                body="# Old call\n\nOld guidance.",
            )
            company = guidance.load_company(config, "EX")
            found = guidance.find_quarter_documents(documents, company, "Q1 2026")

        self.assertEqual([item.path.name for item in found], ["first.md", "second.md"])
        self.assertIn("Revenue guidance was increased", found[0].body)

    def test_get_guidance_combines_full_local_documents_and_online_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, documents = self._fixture(root)
            self._write_document(
                documents / "release.md",
                period="Q1 2026",
                published_at="2026-05-10",
                body="# First-quarter release\n\nLOCAL FULL DOCUMENT SENTINEL.",
            )
            output = root / "data" / "combined.txt"

            def fake_researcher(company, quarter):
                self.assertEqual(company, "Example Co")
                self.assertEqual(str(quarter), "Q1 2026")
                return {"guidance": [online_piece()]}

            result = guidance.get_guidance(
                "EX",
                "Q1 2026",
                output_path=output,
                config_path=config,
                document_root=root / "offline-data",
                researcher=fake_researcher,
            )
            text = result.read_text(encoding="utf-8")

        self.assertEqual(result, output)
        self.assertIn("LOCAL FULL DOCUMENT SENTINEL", text)
        self.assertIn("RELEASE DATE: 2026-05-10", text)
        self.assertIn("SOURCE: Offline corpus", text)
        self.assertIn("RELEASE DATE: 2026-05-12T13:30:00Z", text)
        self.assertIn("SOURCE: Example Co Investor Relations", text)
        self.assertIn("METRIC: Revenue growth", text)
        self.assertGreaterEqual(text.count(guidance.DIVIDER), 2)

    def test_default_filename_uses_company_quarter_and_shared_data_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, documents = self._fixture(root)
            self._write_document(
                documents / "release.md",
                period="Q2 2026",
                published_at="2026-08-12",
                body="# Q2 release\n\nResults.",
            )
            with mock.patch.object(guidance, "DEFAULT_DATA_DIR", root / "data"):
                result = guidance.get_guidance(
                    "Example Co",
                    "2026Q2",
                    config_path=config,
                    document_root=root / "offline-data",
                    researcher=lambda *_: [],
                )

        self.assertEqual(result.name, "guidance_Example_Co_Q2_2026.txt")

    def test_nonconfigured_company_and_full_year_use_online_only_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = self._fixture(root)
            with mock.patch.object(guidance, "DEFAULT_DATA_DIR", root / "data"):
                result = guidance.get_guidance(
                    "Outside Co",
                    "FY 2026",
                    config_path=config,
                    document_root=root / "offline-data",
                    researcher=lambda company, period: [],
                )

        self.assertEqual(result.name, "guidance_Outside_Co_FY2026.txt")

    def test_online_source_requires_guidance_metrics_and_valid_url(self):
        missing_metrics = {**online_piece(), "metrics": []}
        with self.assertRaisesRegex(ValueError, "at least one guidance metric"):
            guidance.normalize_online_piece(missing_metrics)
        bad_url = {**online_piece(), "source_url": "not-a-url"}
        with self.assertRaisesRegex(ValueError, "valid public HTTP"):
            guidance.normalize_online_piece(bad_url)


if __name__ == "__main__":
    unittest.main()
