import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import earnings_data


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def workbook_values(path, sheet_number=1):
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read(f"xl/worksheets/sheet{sheet_number}.xml"))
    values = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        output = []
        for cell in row.findall("x:c", NS):
            inline = cell.find("x:is/x:t", NS)
            number = cell.find("x:v", NS)
            output.append(inline.text if inline is not None else float(number.text))
        values.append(output)
    return values


def workbook_sheet_names(path):
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    return [sheet.get("name") for sheet in root.findall("x:sheets/x:sheet", NS)]


class EarningsDataTests(unittest.TestCase):
    def test_quarter_parser_accepts_common_formats_and_range_is_inclusive(self):
        self.assertEqual(str(earnings_data.Quarter.parse("FY2020Q4")), "Q4 2020")
        self.assertEqual(str(earnings_data.Quarter.parse("Q1 2021")), "Q1 2021")
        self.assertEqual(
            [str(item) for item in earnings_data.quarter_range("Q4 2020", "Q2 2021")],
            ["Q4 2020", "Q1 2021", "Q2 2021"],
        )
        with self.assertRaises(ValueError):
            earnings_data.quarter_range("Q2 2021", "Q1 2021")

    def test_unofficial_validation_requires_two_independent_domains(self):
        record = earnings_data.normalize_record(
            {
                "quarter": "Q1 2026",
                "metric": "Revenue",
                "actual_value": 100,
                "actual_source_class": "validated_unofficial",
                "actual_sources": [
                    {"name": "Example", "url": "https://example.com/a"},
                    {"name": "Example", "url": "https://example.com/b"},
                ],
                "consensus_value": None,
                "consensus_source_class": "not_found",
                "consensus_sources": [],
            },
            "Example Co",
        )
        self.assertEqual(record["actual_source_class"], "single_source")
        self.assertEqual(record["confidence"], "Low")

    def test_get_earnings_writes_company_tab_and_marks_nonofficial_value_cell(self):
        def fake_researcher(company, metrics, start, end):
            self.assertEqual(company, "Example/Co")
            self.assertEqual(metrics, {"Revenue": "USDm"})
            return [
                {
                    "quarter": "Q1 2026",
                    "earnings_release_time_utc": "2026-04-20T07:00:00-04:00",
                    "release_time_source_class": "official",
                    "release_time_sources": [
                        {
                            "name": "Example IR",
                            "url": "https://investor.example.com/q1",
                            "document_title": "Q1 results",
                            "published_date": "2026-04-20",
                        }
                    ],
                    "metric": "Revenue",
                    "units": "USDm",
                    "actual_value": 125.5,
                    "actual_source_class": "official",
                    "actual_sources": [
                        {
                            "name": "Example IR",
                            "url": "https://investor.example.com/q1",
                            "document_title": "Q1 results",
                            "published_date": "2026-04-20",
                        }
                    ],
                    "consensus_value": 123.0,
                    "consensus_source_class": "validated_unofficial",
                    "consensus_sources": [
                        {"name": "Reuters", "url": "https://reuters.com/a"},
                        {"name": "MarketWatch", "url": "https://marketwatch.com/b"},
                    ],
                    "consensus_as_of": "2026-04-19",
                    "notes": "Consensus predates the release.",
                },
                {
                    "quarter": "Q2 2026",
                    "earnings_release_time_utc": "2026-07-21T11:00:00Z",
                    "release_time_source_class": "bloomberg",
                    "release_time_sources": [
                        {"name": "Bloomberg", "url": "https://bloomberg.com/calendar"}
                    ],
                    "metric": "Revenue",
                    "units": "USDm",
                    "actual_value": None,
                    "actual_source_class": "not_found",
                    "actual_sources": [],
                    "consensus_value": 130.0,
                    "consensus_source_class": "bloomberg",
                    "consensus_sources": [
                        {"name": "Bloomberg", "url": "https://bloomberg.com/example"}
                    ],
                    "consensus_as_of": "2026-07-15",
                    "notes": "Next-quarter consensus is already available.",
                },
            ]

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "data" / "earnings.xlsx"
            result = earnings_data.get_earnings(
                "Example/Co",
                {"Revenue": "USDm"},
                "Q1 2026",
                "Q1 2026",
                output_path=output,
                researcher=fake_researcher,
            )
            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                workbook = archive.read("xl/workbook.xml").decode()
            self.assertIn('name="Example Co"', workbook)
            rows = workbook_values(output)
            self.assertEqual(rows[0][1], "Earnings release time (UTC)")
            self.assertEqual(rows[1][1], "2026-04-20T11:00:00Z")
            self.assertEqual(rows[1][6], 125.5)  # Official actual stays numeric.
            self.assertEqual(rows[1][9], "123 [Source: Reuters, MarketWatch]")
            self.assertEqual(rows[2][0], "Q2 2026")
            self.assertEqual(
                rows[2][1], "2026-07-21T11:00:00Z [Source: Bloomberg]"
            )
            self.assertEqual(rows[2][9], "130 [Source: Bloomberg]")

    def test_release_time_requires_timezone_and_normalizes_to_utc(self):
        common = {
            "quarter": "Q1 2026",
            "metric": "Revenue",
            "actual_value": None,
            "actual_source_class": "not_found",
            "actual_sources": [],
            "consensus_value": None,
            "consensus_source_class": "not_found",
            "consensus_sources": [],
            "release_time_source_class": "official",
            "release_time_sources": [{"name": "Issuer", "url": "https://issuer.example/q1"}],
        }
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            earnings_data.normalize_record(
                {**common, "earnings_release_time_utc": "2026-04-20T07:00:00"},
                "Example Co",
            )
        record = earnings_data.normalize_record(
            {**common, "earnings_release_time_utc": "2026-04-20T07:00:00-04:00"},
            "Example Co",
        )
        self.assertEqual(record["earnings_release_time_utc"], "2026-04-20T11:00:00Z")

    def test_missing_research_rows_are_written_as_not_found(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "missing.xlsx"
            earnings_data.get_earnings(
                "Empty Co",
                ["Revenue", "Interest expense"],
                "Q4 2025",
                "Q1 2026",
                output_path=output,
                researcher=lambda *_: [],
            )
            rows = workbook_values(output)
            self.assertEqual(len(rows), 5)  # header + 2 quarters x 2 metrics
            self.assertTrue(all(row[1] == "Not found" for row in rows[1:]))
            self.assertTrue(all(row[6] == "Not found" for row in rows[1:]))
            self.assertTrue(all(row[9] == "Not found" for row in rows[1:]))

    def test_existing_workbook_adds_company_tab_and_appends_company_rows(self):
        def record(company, metric, actual):
            return {
                "quarter": "Q1 2026",
                "metric": metric,
                "units": "USDm",
                "actual_value": actual,
                "actual_source_class": "official",
                "actual_sources": [{"name": f"{company} IR", "url": "https://issuer.example/q1"}],
                "consensus_value": None,
                "consensus_source_class": "not_found",
                "consensus_sources": [],
            }

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "earnings.xlsx"
            earnings_data.get_earnings(
                "Alpha Co",
                ["Revenue"],
                "Q1 2026",
                "Q1 2026",
                output_path=output,
                researcher=lambda company, *_: [record(company, "Revenue", 100)],
            )
            earnings_data.get_earnings(
                "Beta Co",
                ["Revenue"],
                "Q1 2026",
                "Q1 2026",
                output_path=output,
                researcher=lambda company, *_: [record(company, "Revenue", 200)],
            )
            earnings_data.get_earnings(
                "Alpha Co",
                ["Interest expense"],
                "Q1 2026",
                "Q1 2026",
                output_path=output,
                researcher=lambda company, *_: [record(company, "Interest expense", 5)],
            )

            self.assertEqual(workbook_sheet_names(output), ["Alpha Co", "Beta Co"])
            alpha_rows = workbook_values(output, 1)
            beta_rows = workbook_values(output, 2)
            self.assertEqual([row[4] for row in alpha_rows[1:]], ["Revenue", "Interest expense"])
            self.assertEqual(alpha_rows[1][6], 100.0)
            self.assertEqual(alpha_rows[2][6], 5.0)
            self.assertEqual(beta_rows[1][6], 200.0)

    def test_merge_adds_new_columns_without_replacing_populated_cells(self):
        existing = [["Quarter", "Metric", "Reported actual"], ["Q1 2026", "Revenue", 100]]
        incoming = [
            earnings_data.HEADERS,
            [
                "Q1 2026",
                "2026-04-20T11:00:00Z",
                "official",
                "Issuer (https://issuer.example/q1)",
                "Revenue",
                "USDm",
                999,
                "official",
                "Issuer",
                "Not found",
                "not_found",
                "",
                "",
                "High",
                "",
            ],
        ]
        merged = earnings_data._merge_sheet_rows(existing, incoming)
        self.assertEqual(merged[0], earnings_data.HEADERS)
        self.assertEqual(merged[1][1], "2026-04-20T11:00:00Z")
        self.assertEqual(merged[1][6], 100)  # Existing populated value is preserved.


if __name__ == "__main__":
    unittest.main()
