import posixpath
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import comparable


SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def workbook_values(path):
    with zipfile.ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.get("Id"): item.get("Target")
            for item in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        }
        result = {}
        for sheet in workbook.findall(f".//{{{SHEET_NS}}}sheet"):
            relationship_id = sheet.get(f"{{{REL_NS}}}id")
            target = targets[relationship_id]
            member = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath(posixpath.join("xl", target))
            )
            worksheet = ElementTree.fromstring(archive.read(member))
            result[sheet.get("name")] = [
                "".join(node.text or "" for node in cell.findall(f".//{{{SHEET_NS}}}t"))
                for cell in worksheet.findall(f".//{{{SHEET_NS}}}c")
                if str(cell.get("r") or "").startswith("A")
            ]
        return result


def source(company, source_type="issuer"):
    return {
        "name": f"{company} results",
        "url": f"https://{company}.example/results",
        "published_date": "2026-05-01",
        "source_type": source_type,
        "applies_to": company,
        "claim": "Quarterly financial history",
    }


def metric(name, score, sources=None):
    return {
        "metric": name,
        "target_metric": name,
        "candidate_metric": name,
        "similarity_score": score,
        "comparison": "Similar direction and growth pattern over eight quarters.",
        "period_compared": "Q2 2024 through Q1 2026",
        "sources": sources if sources is not None else [source("target"), source("candidate")],
    }


def candidate(company="Peer Inc", ticker="PEER", scores=(80, 70, 90)):
    return {
        "company": company,
        "ticker": ticker,
        "exchange": "NYSE",
        "industry": "Example industry",
        "business_similarity_score": 90,
        "macro_similarity_score": 85,
        "scale_similarity_score": 60,
        "relationship_types": ["same industry", "shared input costs"],
        "shared_drivers": ["housing activity"],
        "why_comparable": "Demand and margins respond to the same cycle.",
        "metrics": [
            metric("revenue", scores[0]),
            metric("costs", scores[1]),
            metric("eps", scores[2]),
        ],
        "direct_comparable_sources": [],
        "caveats": ["Different fiscal calendar"],
    }


class ComparableTests(unittest.TestCase):
    def test_search_ranks_with_local_weights_and_returns_audit_detail(self):
        stronger = candidate()
        weaker = candidate("Other Corp", "OTHER", (50, 50, 50))

        result = comparable.research_comparables(
            "Target Co",
            limit=2,
            researcher=lambda company, count: [weaker, stronger],
        )

        self.assertEqual([item["ticker"] for item in result], ["PEER", "OTHER"])
        self.assertEqual(result[0]["score"], 81.35)
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["confidence"], "high")
        self.assertEqual(result[0]["evidence_summary"]["metrics_with_both_company_evidence"], 3)

    def test_unsupported_metric_score_contributes_zero(self):
        item = candidate()
        item["metrics"][0]["sources"] = [source("candidate")]

        normalized = comparable.normalize_candidate(item, "Target Co")

        self.assertEqual(normalized["metrics"]["revenue"]["ranking_score"], 0)
        self.assertEqual(normalized["score"], 57.35)

    def test_peer_list_cannot_be_used_as_financial_evidence(self):
        item = candidate()
        peer_page = source("both", "peer_list")
        peer_page["applies_to"] = "both"
        item["metrics"] = [metric("revenue", 100, [peer_page])]

        normalized = comparable.normalize_candidate(item, "Target Co")

        self.assertEqual(normalized["metrics"]["revenue"]["sources"], [])
        self.assertEqual(normalized["metrics"]["revenue"]["ranking_score"], 0)

    def test_direct_source_without_methodology_is_downgraded(self):
        item = candidate()
        direct = source("peerlist", "peer_list")
        direct["applies_to"] = "both"
        item["direct_comparable_sources"] = [
            {
                "source": direct,
                "methodology": "",
                "assessment": "useful",
                "validation_notes": "No disclosed selection criteria.",
            }
        ]

        normalized = comparable.normalize_candidate(item, "Target Co")

        self.assertEqual(normalized["direct_comparable_sources"][0]["assessment"], "limited")

    def test_low_confidence_candidates_are_excluded_by_default(self):
        item = candidate()
        item["metrics"] = [metric("revenue", 95)]

        default = comparable.research_comparables(
            "Target Co", researcher=lambda company, count: [item]
        )
        included = comparable.research_comparables(
            "Target Co",
            include_low_confidence=True,
            researcher=lambda company, count: [item],
        )

        self.assertEqual(default, [])
        self.assertEqual(len(included), 1)

    def test_rejects_target_as_its_own_peer(self):
        with self.assertRaisesRegex(ValueError, "own comparable"):
            comparable.normalize_candidate(candidate("Target Co"), "Target Co")

    def test_validates_public_inputs(self):
        with self.assertRaisesRegex(ValueError, "company_name"):
            comparable.research_comparables("", researcher=lambda company, count: [])
        with self.assertRaisesRegex(ValueError, "limit"):
            comparable.research_comparables("Target", limit=0, researcher=lambda company, count: [])

    def test_search_saves_one_company_per_row_and_adds_new_company_tabs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "data" / "comparable.xlsx"
            first = candidate()
            second = candidate("Other Corp", "OTHER", (50, 50, 50))

            returned = comparable.search_comparable(
                "Target Co",
                output_path=output,
                limit=2,
                researcher=lambda company, count: [second, first],
            )
            comparable.search_comparable(
                "Second Target",
                output_path=output,
                researcher=lambda company, count: [candidate("Second Peer", "SECOND")],
            )

            self.assertEqual(returned, output)
            self.assertEqual(
                workbook_values(output),
                {
                    "Target Co": ["Peer Inc", "Other Corp"],
                    "Second Target": ["Second Peer"],
                },
            )

    def test_rerunning_a_company_replaces_only_its_existing_tab(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "comparable.xlsx"
            comparable.search_comparable(
                "Target Co", output_path=output, researcher=lambda company, count: [candidate()]
            )
            comparable.search_comparable(
                "Other Target",
                output_path=output,
                researcher=lambda company, count: [candidate("Other Peer", "OP")],
            )
            comparable.search_comparable(
                "Target Co",
                output_path=output,
                researcher=lambda company, count: [candidate("Replacement Peer", "RP")],
            )

            self.assertEqual(
                workbook_values(output),
                {"Target Co": ["Replacement Peer"], "Other Target": ["Other Peer"]},
            )


if __name__ == "__main__":
    unittest.main()
