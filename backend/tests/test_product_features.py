import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
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

    def test_brand_assistant_sends_general_questions_to_gemini_without_workspace_evidence(self) -> None:
        captured_prompt = {}
        client_options = {}

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                captured_prompt.update(json.loads(kwargs["contents"]))
                return type("Response", (), {
                    "text": json.dumps({
                        "answer": "Machine learning lets computers learn patterns from examples.",
                        "evidence_ids": [],
                    }),
                })()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                client_options.update(kwargs)
                self.models = FakeModels()

        with patch.object(
            main,
            "get_guidelines",
            return_value=main.Guidelines(brand_name="Northstar"),
        ):
            with patch.object(main, "list_reviews", return_value=[]):
                retrieval = patch.object(
                    main.knowledge_base,
                    "retrieve",
                    return_value={"status": "empty", "sources": [], "error": None},
                )
                with retrieval as retrieve:
                    with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
                        with patch("google.genai.Client", FakeClient):
                            response = self.client.post(
                                "/api/assistant",
                                json={"question": "Explain machine learning in simple terms."},
                            )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(
            result["answer"],
            "Machine learning lets computers learn patterns from examples.",
        )
        self.assertFalse(result["evidence_available"])
        self.assertEqual(result["sources"], [])
        self.assertFalse(captured_prompt["workspace_question"])
        self.assertEqual(captured_prompt["evidence"], [])
        self.assertEqual(client_options["http_options"].timeout, 15000)
        retrieve.assert_not_called()

    def test_brand_assistant_retrieves_saved_guidance_for_brand_questions(self) -> None:
        captured_prompt = {}

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                captured_prompt.update(json.loads(kwargs["contents"]))
                return type("Response", (), {
                    "text": json.dumps({
                        "answer": "Use “members” as the preferred term.",
                        "evidence_ids": ["guideline_preferred_terms"],
                    }),
                })()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "empty", "sources": [], "error": None},
            ) as retrieve:
                with patch("google.genai.Client", FakeClient):
                    response = self.client.post(
                        "/api/assistant",
                        json={"question": "What are our preferred brand terms?"},
                    )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["evidence_available"])
        self.assertEqual(result["sources"][0]["source_type"], "guideline")
        self.assertIn("members", result["sources"][0]["excerpt"])
        self.assertTrue(captured_prompt["workspace_question"])
        retrieve.assert_called_once_with("What are our preferred brand terms?")

    def test_brand_assistant_uses_saved_voice_guidance_for_rewriting(self) -> None:
        captured_prompt = {}

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                captured_prompt.update(json.loads(kwargs["contents"]))
                return type("Response", (), {
                    "text": json.dumps({
                        "answer": "Here is a clearer version in the saved warm, practical voice.",
                        "evidence_ids": ["guideline_voice"],
                    }),
                })()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "empty", "sources": [], "error": None},
            ) as retrieve:
                with patch("google.genai.Client", FakeClient):
                    response = self.client.post(
                        "/api/assistant",
                        json={"question": "Rewrite this copy to be clearer: The product is good."},
                    )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["evidence_available"])
        self.assertEqual(response.json()["sources"][0]["source_type"], "guideline")
        self.assertTrue(captured_prompt["writing_request"])
        self.assertFalse(captured_prompt["workspace_question"])
        retrieve.assert_called_once()

    def test_brand_assistant_includes_recent_turns_for_follow_up_questions(self) -> None:
        captured_prompt = {}

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                captured_prompt.update(json.loads(kwargs["contents"]))
                return type("Response", (), {
                    "text": json.dumps({
                        "answer": "In short, it learns patterns from examples.",
                        "evidence_ids": [],
                    }),
                })()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch("google.genai.Client", FakeClient):
                response = self.client.post(
                    "/api/assistant",
                    json={
                        "question": "Can you make that shorter?",
                        "history": [
                            {"role": "user", "text": "Explain machine learning simply."},
                            {
                                "role": "assistant",
                                "text": "Machine learning finds patterns in examples and uses them to make predictions.",
                            },
                        ],
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            captured_prompt["conversation_history"],
            [
                {"role": "user", "text": "Explain machine learning simply."},
                {
                    "role": "assistant",
                    "text": "Machine learning finds patterns in examples and uses them to make predictions.",
                },
            ],
        )
        self.assertEqual(response.json()["answer"], "In short, it learns patterns from examples.")

    def test_brand_assistant_uses_workspace_context_for_brand_follow_ups(self) -> None:
        captured_prompt = {}

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                captured_prompt.update(json.loads(kwargs["contents"]))
                return type("Response", (), {
                    "text": json.dumps({
                        "answer": "Use the preferred term naturally when it accurately describes the audience.",
                        "evidence_ids": ["guideline_preferred_terms"],
                    }),
                })()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        question = "What are our preferred brand terms?"
        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "empty", "sources": [], "error": None},
            ) as retrieve:
                with patch("google.genai.Client", FakeClient):
                    response = self.client.post(
                        "/api/assistant",
                        json={
                            "question": "How should we use them?",
                            "history": [{"role": "user", "text": question}],
                        },
                    )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["evidence_available"])
        self.assertEqual(response.json()["sources"][0]["source_type"], "guideline")
        self.assertTrue(captured_prompt["workspace_question"])
        retrieve.assert_called_once()
        self.assertIn(question, retrieve.call_args.args[0])

    def test_brand_assistant_refuses_to_guess_when_brand_evidence_is_missing(self) -> None:
        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                return type("Response", (), {
                    "text": json.dumps({
                        "answer": "The preferred term is unavailable.",
                        "evidence_ids": [],
                    }),
                })()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch.object(
                main,
                "get_guidelines",
                return_value=main.Guidelines(brand_name="Northstar"),
            ):
                with patch.object(
                    main.knowledge_base,
                    "retrieve",
                    return_value={"status": "empty", "sources": [], "error": None},
                ):
                    with patch("google.genai.Client", FakeClient):
                        response = self.client.post(
                            "/api/assistant",
                            json={"question": "What are our preferred brand terms?"},
                        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result["evidence_available"])
        self.assertEqual(result["sources"], [])
        self.assertIn("couldn't verify", result["answer"].casefold())

    def test_brand_assistant_reports_no_saved_reviews_for_today(self) -> None:
        with patch.object(
            main,
            "get_guidelines",
            return_value=main.Guidelines(brand_name="Northstar"),
        ):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "empty", "sources": [], "error": None},
            ):
                response = self.client.post(
                    "/api/assistant",
                    json={
                        "question": "What are today's review contents?",
                        "local_today": datetime.now(timezone.utc).date().isoformat(),
                        "timezone_offset_minutes": 0,
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["answer"], "There are no saved reviews yet.")
        self.assertFalse(response.json()["evidence_available"])

    def test_brand_assistant_distinguishes_no_reviews_today_from_empty_history(self) -> None:
        review = self.create_review("Previous campaign", "Yesterday's reviewed content.")
        store = main.load_store()
        item = next(item for item in store["reviews"] if item["id"] == review["id"])
        item["created_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        main.save_store(store)

        with patch.object(
            main,
            "get_guidelines",
            return_value=main.Guidelines(brand_name="Northstar"),
        ):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "empty", "sources": [], "error": None},
            ):
                response = self.client.post(
                    "/api/assistant",
                    json={
                        "question": "What are today's review contents?",
                        "local_today": datetime.now(timezone.utc).date().isoformat(),
                        "timezone_offset_minutes": 0,
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("No reviews have been saved today", response.json()["answer"])
        self.assertFalse(response.json()["evidence_available"])

    def test_brand_assistant_uses_actual_reviews_for_todays_history_question(self) -> None:
        old_review = self.create_review("Previous campaign", "Old reviewed content.")
        local_today = (datetime.now(timezone.utc) + timedelta(minutes=330)).date()
        local_midnight_utc = (
            datetime.combine(local_today, datetime.min.time(), timezone.utc)
            - timedelta(minutes=330)
        )
        store = main.load_store()
        old_item = next(item for item in store["reviews"] if item["id"] == old_review["id"])
        old_item["created_at"] = (local_midnight_utc - timedelta(minutes=15)).isoformat()
        main.save_store(store)
        today_review = self.create_review("Today's campaign", "Today's approved campaign copy.")
        store = main.load_store()
        today_item = next(item for item in store["reviews"] if item["id"] == today_review["id"])
        today_item["created_at"] = (local_midnight_utc + timedelta(minutes=15)).isoformat()
        main.save_store(store)
        captured_prompt = {}

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                prompt = json.loads(kwargs["contents"])
                captured_prompt.update(prompt)
                response = {
                    "answer": "Today's campaign was reviewed. Its saved content is shown in the cited review.",
                    "evidence_ids": [f"review_content_{today_review['id']}"],
                }
                return type("Response", (), {"text": json.dumps(response)})()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch.object(
                main,
                "get_guidelines",
                return_value=main.Guidelines(brand_name="Northstar"),
            ):
                with patch.object(
                    main.knowledge_base,
                    "retrieve",
                    return_value={"status": "empty", "sources": [], "error": None},
                ):
                    with patch("google.genai.Client", FakeClient):
                        response = self.client.post(
                            "/api/assistant",
                            json={
                                "question": "What are today's review contents?",
                                "local_today": local_today.isoformat(),
                                "timezone_offset_minutes": -330,
                            },
                        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["evidence_available"])
        self.assertIn("Today's campaign was reviewed", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["source_type"], "review_content")
        self.assertEqual(result["sources"][0]["excerpt"], "Today's approved campaign copy.")
        prompt_sources = captured_prompt["evidence"]
        review_content_sources = [
            source for source in prompt_sources if source["source_type"] == "review_content"
        ]
        self.assertEqual(len(review_content_sources), 1)
        self.assertIn("Today's campaign", review_content_sources[0]["title"])
        self.assertNotIn("Previous campaign", review_content_sources[0]["title"])

    def test_brand_assistant_cites_retrieved_document_and_selected_review_findings(self) -> None:
        review = self.create_review("Campaign draft", "Our products are guaranteed.")
        self.assertTrue(review["issues"])
        retrieved_source = {
            "chunk_id": "guide-chunk-1",
            "document_id": "guide-1",
            "filename": "brand-guide.pdf",
            "page_number": 4,
            "excerpt": "Avoid unsupported guarantees in public marketing.",
            "similarity": 0.91,
        }
        captured_prompt = {}

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                prompt = json.loads(kwargs["contents"])
                captured_prompt.update(prompt)
                response = {
                    "answer": "The reviewed copy uses prohibited wording; the guide says to avoid unsupported guarantees.",
                    "evidence_ids": ["review_issue_0", "guide-chunk-1", "unknown-source"],
                }
                return type("Response", (), {"text": json.dumps(response)})()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "retrieved", "sources": [retrieved_source], "error": None},
            ):
                with patch("google.genai.Client", FakeClient):
                    response = self.client.post(
                        "/api/assistant",
                        json={
                            "question": "Why was this campaign flagged?",
                            "review_id": review["id"],
                        },
                    )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["evidence_available"])
        self.assertEqual(
            {source["source_id"] for source in result["sources"]},
            {"review_issue_0", "guide-chunk-1"},
        )
        document = next(source for source in result["sources"] if source["source_type"] == "document")
        self.assertEqual(document["filename"], "brand-guide.pdf")
        self.assertEqual(document["page_number"], 4)
        self.assertIn("review_issue_0", {item["source_id"] for item in captured_prompt["evidence"]})
        self.assertNotIn("unknown-source", {source["source_id"] for source in result["sources"]})

    def test_brand_assistant_handles_gemini_failure_without_exposing_key(self) -> None:
        secret = "assistant-test-key"

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                raise RuntimeError(f"temporary generation error: {secret}")

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        output = StringIO()
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "empty", "sources": [], "error": None},
            ):
                with patch("google.genai.Client", FakeClient):
                    with redirect_stdout(output):
                        response = self.client.post(
                            "/api/assistant",
                            json={"question": "What terminology should this brand use?"},
                        )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, output.getvalue())
        self.assertIn("try again", response.json()["detail"].casefold())


if __name__ == "__main__":
    unittest.main()
