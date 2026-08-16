import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import news


def piece(
    headline="Company wins a material customer contract",
    *,
    published_at="2026-05-12T09:30:00-04:00",
    source_name="Reuters",
    source_url="https://www.reuters.com/example-contract",
    source_type="major_newswire",
    category="company",
    impact_score=4,
    body_available=True,
    content="The multi-year contract is expected to increase sales from next quarter.",
):
    return {
        "headline": headline,
        "published_at": published_at,
        "source_name": source_name,
        "source_url": source_url,
        "source_type": source_type,
        "category": category,
        "impact_score": impact_score,
        "financial_statement_lines": ["Revenue", "Operating profit"],
        "impact_rationale": "Incremental contract sales should raise revenue and operating profit.",
        "body_available": body_available,
        "content": content,
    }


class NewsTests(unittest.TestCase):
    def test_validates_date_range_and_timestamp_timezone(self):
        with self.assertRaisesRegex(ValueError, "start_date cannot be after"):
            news.get_news(
                "Example Co",
                "2026-05-13",
                "2026-05-12",
                researcher=lambda *_: [],
            )
        bad = piece(published_at="2026-05-12T09:30:00")
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            news.get_news(
                "Example Co",
                "2026-05-01",
                "2026-05-31",
                researcher=lambda *_: [bad],
            )

    def test_get_news_writes_source_and_release_time_above_each_piece(self):
        def fake_researcher(company, start, end):
            self.assertEqual(company, "Example/Co")
            self.assertEqual(start, date(2026, 5, 1))
            self.assertEqual(end, date(2026, 5, 31))
            return [piece()]

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "news-output.txt"
            result = news.get_news(
                "Example/Co",
                "2026-05-01",
                "2026-05-31",
                output_path=output,
                researcher=fake_researcher,
            )

            self.assertEqual(result, output)
            text = output.read_text(encoding="utf-8")
            release_position = text.index("RELEASE TIME: 2026-05-12T13:30:00Z")
            source_position = text.index("NEWS SOURCE: Reuters")
            headline_position = text.index("HEADLINE: Company wins")
            self.assertLess(release_position, headline_position)
            self.assertLess(source_position, headline_position)
            self.assertIn("FINANCIAL STATEMENT LINE(S): Revenue, Operating profit", text)

    def test_filters_low_impact_and_out_of_range_and_prioritizes_legitimate_sources(self):
        official = piece(
            "Regulator approves company's new product",
            published_at="2026-05-20",
            source_name="Government regulator",
            source_url="https://regulator.example/decision",
            source_type="government_regulator",
            impact_score=3,
        )
        other = piece(
            "Input shortage raises industry costs",
            published_at="2026-05-05",
            source_name="Regional Business Daily",
            source_url="https://regional.example/shortage",
            source_type="local_or_other",
            category="industry",
            impact_score=5,
        )
        too_small = piece(
            "Company opens a routine small office",
            source_url="https://www.reuters.com/small-office",
            impact_score=2,
        )
        outside = piece(
            "Central bank changes interest rates",
            published_at="2026-06-01",
            source_url="https://www.reuters.com/rates",
            category="macro",
            impact_score=5,
        )

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "ranked.txt"
            news.get_news(
                "Example Co",
                "2026-05-01",
                "2026-05-31",
                output_path=output,
                researcher=lambda *_: [other, too_small, outside, official],
            )
            text = output.read_text(encoding="utf-8")

        self.assertLess(text.index(official["headline"]), text.index(other["headline"]))
        self.assertNotIn(too_small["headline"], text)
        self.assertNotIn(outside["headline"], text)
        self.assertIn(news.DIVIDER, text)

    def test_keeps_paywalled_headline_and_deduplicates_syndication(self):
        paywalled = piece(
            "Company warns tariff costs will reduce margins",
            body_available=False,
            content="",
        )
        duplicate = {**paywalled, "source_url": "https://example.com/syndicated-copy"}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "paywall.txt"
            news.get_news(
                "Example Co",
                "2026-05-01",
                "2026-05-31",
                output_path=output,
                researcher=lambda *_: {"news": [duplicate, paywalled]},
            )
            text = output.read_text(encoding="utf-8")

        self.assertEqual(text.count("HEADLINE: Company warns"), 1)
        self.assertIn("CONTENT (Headline only): Company warns tariff costs", text)

    def test_default_filename_uses_company_and_date_range_in_shared_data_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(news, "DEFAULT_DATA_DIR", Path(temp)):
                result = news.get_news(
                    "Example/Co",
                    "2026-05-01",
                    "2026-05-31",
                    researcher=lambda *_: [],
                )
        self.assertEqual(result.name, "news_Example_Co_2026-05-01_to_2026-05-31.txt")


if __name__ == "__main__":
    unittest.main()
