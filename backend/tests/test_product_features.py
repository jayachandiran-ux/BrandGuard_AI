import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import main
from rag import KnowledgeBase


class TestEmbeddings:
    terms = ("northstar", "designed", "portland", "seattle", "warranty", "moon")

    @classmethod
    def vector(cls, text: str) -> np.ndarray:
        lowered = text.casefold()
        return np.asarray(
            [0.05, *(1.0 if term in lowered else 0.0 for term in cls.terms)],
            dtype=np.float32,
        )

    @classmethod
    def documents(cls, chunks: list[dict]) -> np.ndarray:
        return np.vstack([cls.vector(chunk["text"]) for chunk in chunks])

    @classmethod
    def query(cls, text: str) -> np.ndarray:
        return cls.vector(text).reshape(1, -1)


class ProductFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.old_data_file = main.DATA_FILE
        self.old_knowledge_base = main.knowledge_base
        main.DATA_FILE = self.root / "data.json"
        main.knowledge_base = KnowledgeBase(
            root=self.root / "rag_data",
            document_embedder=TestEmbeddings.documents,
            query_embedder=TestEmbeddings.query,
            min_score=0.6,
        )
        self.addCleanup(setattr, main, "DATA_FILE", self.old_data_file)
        self.addCleanup(setattr, main, "knowledge_base", self.old_knowledge_base)
        self.gemini_patch = patch.object(main, "gemini_review", return_value=None)
        self.gemini_patch.start()
        self.addCleanup(self.gemini_patch.stop)
        self.client = TestClient(main.app)
        self.guidelines = {
            "brand_name": "Northstar",
            "voice_tone": "Warm, clear, practical; never a hype machine.",
            "preferred_terms": ["members"],
            "prohibited_terms": ["guaranteed"],
            "approved_facts": ["Northstar products are designed in Portland."],
            "instructions": 'Always include "made for members".',
            "is_sample": False,
        }
        saved = self.client.put("/api/guidelines", json=self.guidelines)
        self.assertEqual(saved.status_code, 200, saved.text)

    def create_review(self, title: str, content: str, previous_review_id: str | None = None) -> dict:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            response = self.client.post(
                "/api/reviews",
                json={
                    "title": title,
                    "content_type": "Blog",
                    "content": content,
                    "previous_review_id": previous_review_id,
                },
            )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_claims_are_supported_contradicted_or_unverified_against_approved_facts(self) -> None:
        review = self.create_review(
            "Claim states",
            "Northstar products are designed in Portland. "
            "Northstar products are not designed in Portland. "
            "Northstar ships to the moon.",
        )

        claims = {item["claim"]: item for item in review["claim_assessments"]}
        self.assertEqual(len(claims), 3)
        self.assertEqual(
            claims["Northstar products are designed in Portland."]["classification"],
            "Supported",
        )
        self.assertEqual(
            claims["Northstar products are not designed in Portland."]["classification"],
            "Contradicted",
        )
        unsupported = claims["Northstar ships to the moon."]
        self.assertEqual(unsupported["classification"], "Unverified")
        self.assertIn("insufficient", unsupported["explanation"].casefold())
        self.assertEqual(unsupported["method"], "rule-based")

    def test_claim_extractor_finds_common_factual_predicates(self) -> None:
        content = (
            "Crafted in Portland from recycled steel. "
            "The bottle keeps drinks cold for 24 hours. "
            "Our cotton uses 80% recycled fibers. "
            "We make every bottle locally. "
            "The warranty will cover repairs for one year."
        )
        claims = main.extract_factual_claims(content)
        self.assertEqual(len(claims), 5)
        lowered_claims = [claim.casefold() for claim in claims]
        self.assertTrue(any("crafted in portland" in claim for claim in lowered_claims))
        self.assertTrue(any("keeps drinks cold" in claim for claim in lowered_claims))
        self.assertTrue(any("80% recycled fibers" in claim for claim in lowered_claims))
        self.assertTrue(any("make every bottle locally" in claim for claim in lowered_claims))
        self.assertTrue(any("will cover repairs" in claim for claim in lowered_claims))

    def test_preferred_terms_are_taken_from_current_guidelines(self) -> None:
        custom = main.Guidelines(
            brand_name="Harbor",
            preferred_terms=["patrons"],
            prohibited_terms=[],
            approved_facts=[],
        )
        issues = main.rule_based_issues("Harbor welcomes every guest.", custom)
        preferred = [issue for issue in issues if issue.category == "Preferred terminology"]
        self.assertEqual(len(preferred), 1)
        self.assertIn("patrons", preferred[0].explanation)
        self.assertNotIn("members", preferred[0].explanation)

    def test_rule_based_tone_flags_follow_saved_tone_guidance(self) -> None:
        measured = main.Guidelines(brand_name="Northstar", voice_tone="Calm and measured.")
        energetic = main.Guidelines(brand_name="Northstar", voice_tone="Energetic and celebratory.")

        measured_issues = main.rule_based_issues("Buy now and discover the new collection.", measured)
        energetic_issues = main.rule_based_issues("Buy now and discover the new collection.", energetic)

        self.assertTrue(any(issue.category == "Brand voice and tone" for issue in measured_issues))
        self.assertFalse(any(issue.category == "Brand voice and tone" for issue in energetic_issues))

    def test_compliance_score_uses_documented_severity_penalties(self) -> None:
        issues = [
            main.Issue(category="Prohibited language", severity="High", explanation="x", suggested_correction="y"),
            main.Issue(category="Brand voice and tone", severity="Medium", explanation="x", suggested_correction="y"),
            main.Issue(category="Accessibility and clarity", severity="Low", explanation="x", suggested_correction="y"),
        ]
        self.assertEqual(main.calculate_score(issues), 66)

    def test_retired_gemini_model_configuration_uses_supported_fallback(self) -> None:
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}):
            self.assertEqual(main.configured_gemini_model(), "gemini-3.6-flash")
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.8-flash"}):
            self.assertEqual(main.configured_gemini_model(), "gemini-3.8-flash")

    def test_suggested_revision_cannot_remove_or_add_detected_claims_or_numbers(self) -> None:
        original = "Northstar products are designed in Portland. The bottles use 80% recycled steel."
        self.assertTrue(main.validate_suggested_revision(
            original,
            "Northstar products are designed in Portland. The bottles use 80% recycled steel. Shop today.",
        ))
        self.assertFalse(main.validate_suggested_revision(
            original,
            "Northstar products are designed in Seattle. The bottles use 95% recycled steel.",
        ))

    def test_api_rejects_gemini_revision_that_changes_facts(self) -> None:
        original = "Northstar products are designed in Portland. The bottles use 80% recycled steel."
        ai_result = main.GeminiReview(
            issues=[],
            revised_content="Northstar products are designed in Seattle. The bottles use 95% recycled steel.",
        )
        with patch.object(main, "gemini_review", return_value=ai_result):
            response = self.client.post(
                "/api/reviews",
                json={
                    "title": "Unsafe generated rewrite",
                    "content_type": "Blog",
                    "content": original,
                },
            )
        self.assertEqual(response.status_code, 201, response.text)
        review = response.json()
        self.assertEqual(review["revised_content"], original)
        self.assertIn("changed extracted factual claims", review["revision_notice"])

    def test_rule_based_revision_does_not_delete_copy_to_remove_prohibited_terms(self) -> None:
        content = "Our product is guaranteed to keep drinks cold."
        issues = main.rule_based_issues(content, main.Guidelines(
            brand_name="Northstar",
            prohibited_terms=["guaranteed"],
        ))
        self.assertEqual(main.revised_copy(content, issues), content)

    def test_uploaded_pdf_passage_is_retrieved_and_cited_for_claim(self) -> None:
        main.knowledge_base.ingest_pages(
            "northstar-guide.pdf",
            "b" * 64,
            [{
                "page_number": 4,
                "text": "Northstar products are designed in Portland.",
            }],
            4,
        )
        review = self.create_review(
            "PDF evidence",
            "Northstar products are designed in Portland.",
        )

        claim = review["claim_assessments"][0]
        self.assertEqual(claim["classification"], "Supported")
        document_sources = [item for item in claim["evidence"] if item["source_type"] == "document"]
        self.assertEqual(len(document_sources), 1)
        self.assertEqual(document_sources[0]["filename"], "northstar-guide.pdf")
        self.assertEqual(document_sources[0]["page_number"], 4)
        self.assertIn("designed in Portland", document_sources[0]["excerpt"])

    def test_gemini_claim_assessment_is_labeled_and_fabricated_citations_are_removed(self) -> None:
        ai_result = main.GeminiReview.model_validate({
            "issues": [],
            "revised_content": "Northstar products are designed in Portland.",
            "claims": [{
                "claim": "Northstar products are designed in Portland.",
                "classification": "Supported",
                "explanation": "The supplied approved fact directly matches the statement.",
                "evidence_ids": ["approved_fact_0", "invented-evidence"],
            }],
        })
        with patch.object(main, "gemini_review", return_value=ai_result):
            response = self.client.post(
                "/api/reviews",
                json={
                    "title": "AI assessment",
                    "content_type": "Blog",
                    "content": "Northstar products are designed in Portland.",
                },
            )
        self.assertEqual(response.status_code, 201, response.text)
        claim = response.json()["claim_assessments"][0]
        self.assertEqual(claim["method"], "Gemini AI")
        self.assertEqual(claim["classification"], "Supported")
        self.assertEqual([item["source_id"] for item in claim["evidence"]], ["approved_fact_0"])

    def test_insufficient_evidence_is_never_reported_as_false(self) -> None:
        self.guidelines["approved_facts"] = []
        saved = self.client.put("/api/guidelines", json=self.guidelines)
        self.assertEqual(saved.status_code, 200, saved.text)
        review = self.create_review("Unverified claim", "Northstar ships to the moon.")

        claim = review["claim_assessments"][0]
        self.assertEqual(claim["classification"], "Unverified")
        self.assertEqual(claim["evidence"], [])
        self.assertIn("not proof", claim["explanation"].casefold())

    def test_drift_reports_guideline_evidence_and_selected_approved_copy_differences(self) -> None:
        approved = self.create_review("Approved baseline", "Members enjoy thoughtful design.")
        decision = self.client.post(f"/api/reviews/{approved['id']}/approve")
        self.assertEqual(decision.status_code, 200, decision.text)

        response = self.client.post(
            "/api/drift",
            json={
                "content": "Guaranteed products are revolutionary. Buy now.",
                "approved_review_id": approved["id"],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        categories = {item["category"] for item in result["findings"]}
        self.assertEqual(
            categories,
            {
                "Prohibited language",
                "Preferred terminology",
                "Brand messaging consistency",
                "Brand voice and tone",
            },
        )
        self.assertEqual(result["score"], 56)
        self.assertIn("subtract 20", result["score_method"])
        self.assertTrue(all(item["evidence"] for item in result["findings"]))
        self.assertEqual(result["approved_content_title"], "Approved baseline")
        self.assertTrue(result["approved_content_differences"])

    def test_versions_persist_content_findings_and_exact_approval_status(self) -> None:
        original = self.create_review("Launch copy", "Our products are guaranteed.")
        self.assertTrue(any(issue["category"] == "Prohibited language" for issue in original["issues"]))
        revision = self.create_review(
            "Launch copy revised",
            "Members enjoy practical design.",
            previous_review_id=original["id"],
        )

        self.assertEqual(revision["version_group_id"], original["version_group_id"])
        self.assertEqual(revision["version_number"], 2)
        self.assertEqual(revision["previous_review_id"], original["id"])
        self.assertTrue(revision["content_differences"])
        self.assertTrue(any(change["change"] == "modified" for change in revision["content_differences"]))
        self.assertFalse(any(issue["category"] == "Prohibited language" for issue in revision["issues"]))
        self.assertEqual(revision["content"], "Members enjoy practical design.")

        approved = self.client.post(f"/api/reviews/{revision['id']}/approve")
        self.assertEqual(approved.status_code, 200, approved.text)
        persisted = {item["id"]: item for item in self.client.get("/api/reviews").json()}
        self.assertEqual(persisted[original["id"]]["status"], "Pending")
        self.assertEqual(persisted[revision["id"]]["status"], "Approved")
        self.assertEqual(persisted[revision["id"]]["content"], "Members enjoy practical design.")
        self.assertEqual(persisted[revision["id"]]["version_number"], 2)


if __name__ == "__main__":
    unittest.main()
