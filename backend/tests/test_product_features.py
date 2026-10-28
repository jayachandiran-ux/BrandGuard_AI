import json
import logging
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
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
        self.assertEqual(client_options["http_options"].retry_options.attempts, 2)
        retrieve.assert_not_called()

    def test_brand_assistant_retrieves_saved_guidance_for_brand_questions(self) -> None:
        captured_prompt = {}
        client_options = {}

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
                client_options.update(kwargs)
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

    def test_brand_assistant_answers_project_review_metrics_from_saved_data(self) -> None:
        first = self.create_review("Most findings", "Our guaranteed product is the best.")
        pending = self.create_review("Pending campaign", "Our guaranteed product helps.")
        latest = self.create_review("Old rejected", "Our guaranteed product works.")
        local_today = (datetime.now(timezone.utc) + timedelta(minutes=330)).date()
        local_timezone = timezone(timedelta(minutes=330))
        local_midnight_utc = (
            datetime.combine(local_today, datetime.min.time(), timezone.utc)
            - timedelta(minutes=330)
        )
        store = main.load_store()
        records = {review["id"]: review for review in store["reviews"]}
        issue = records[first["id"]]["issues"][0]
        records[first["id"]].update({
            "status": "Approved",
            "score": 42,
            "created_at": (local_midnight_utc + timedelta(minutes=30)).isoformat(),
            "issues": [issue, issue],
        })
        records[pending["id"]].update({
            "status": "Pending",
            "score": 75,
            "created_at": (local_midnight_utc + timedelta(minutes=60)).isoformat(),
            "issues": [issue],
        })
        records[latest["id"]].update({
            "status": "Rejected",
            "score": 93,
            "created_at": (local_midnight_utc - timedelta(days=1)).isoformat(),
            "issues": [],
        })
        main.save_store(store)

        generations = []

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                prompt = json.loads(kwargs["contents"])
                generations.append(prompt)
                source_ids = [source["source_id"] for source in prompt["evidence"]]
                response = {
                    "answer": "Here is the saved BrandGuard workspace information.",
                    "evidence_ids": source_ids,
                }
                return type("Response", (), {"text": json.dumps(response)})()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        questions = [
            "What is the content review status?",
            "How many reviews were completed today?",
            "Show pending reviews.",
            "Summarise the latest review.",
            "What is the average compliance score?",
            "Which content has the most brand issues?",
            "What features are currently implemented?",
            "What are the known problems in our application?",
            "What should we improve next?",
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "empty", "sources": [], "error": None},
            ):
                with patch("google.genai.Client", FakeClient):
                    for question in questions:
                        with self.subTest(question=question):
                            response = self.client.post(
                                "/api/assistant",
                                json={
                                    "question": question,
                                    "local_today": local_today.isoformat(),
                                    "timezone_offset_minutes": -330,
                                },
                            )
                            self.assertEqual(response.status_code, 200, response.text)
                            result = response.json()
                            self.assertTrue(result["evidence_available"])
                            self.assertTrue(result["sources"])

        review_prompts = generations[:6]
        self.assertEqual(len(review_prompts), 6)
        summary = json.loads(next(
            source["excerpt"]
            for source in review_prompts[0]["evidence"]
            if source["source_id"] == "review_workspace_summary"
        ))
        self.assertEqual(summary["total_saved_reviews"], 3)
        self.assertEqual(summary["reviews_created_today"], 2)
        self.assertIsNone(summary["reviews_completed"])
        self.assertIn("no completed status", summary["completion_count_unavailable_reason"])
        self.assertEqual(summary["today_status_counts"], {
            "Pending": 1,
            "Approved": 1,
            "Rejected": 0,
        })
        self.assertEqual(summary["all_time_status_counts"], {
            "Pending": 1,
            "Approved": 1,
            "Rejected": 1,
        })
        self.assertEqual(summary["average_compliance_score"], 70)
        self.assertEqual(summary["total_findings"], 3)
        self.assertEqual(summary["review_with_most_findings"], {
            "title": "Most findings",
            "finding_count": 2,
        })
        self.assertIn("no separate completion timestamp", " ".join(summary["data_notes"]))

        pending_sources = review_prompts[2]["evidence"]
        pending_reviews = [
            source for source in pending_sources
            if source["source_type"] == "review_content"
        ]
        self.assertEqual(len(pending_reviews), 1)
        self.assertIn("Pending campaign", pending_reviews[0]["title"])
        latest_sources = review_prompts[3]["evidence"]
        latest_review = next(
            source for source in latest_sources
            if source["source_type"] == "review_content"
        )
        self.assertIn("Pending campaign", latest_review["title"])
        for project_prompt in generations[6:]:
            project_doc = next(
                source for source in project_prompt["evidence"]
                if source["source_id"] == "project_maturity_readme"
            )
            self.assertIn("Partially working", project_doc["excerpt"])

    def test_brand_assistant_handles_empty_review_data_with_real_zero_summary(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            response = self.client.post(
                "/api/assistant",
                json={"question": "Show pending reviews."},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["answer"], "There are no saved reviews yet.")
        self.assertFalse(response.json()["evidence_available"])

    def test_brand_assistant_does_not_turn_review_store_failure_into_zero(self) -> None:
        with patch(
            "main.list_reviews",
            side_effect=HTTPException(status_code=500, detail="Could not read local data store."),
        ):
            response = self.client.post(
                "/api/assistant",
                json={"question": "Show pending reviews."},
            )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Could not read local data store", response.json()["detail"])

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

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
            with patch.object(
                main.knowledge_base,
                "retrieve",
                return_value={"status": "empty", "sources": [], "error": None},
            ):
                with patch("google.genai.Client", FakeClient):
                    with self.assertLogs("main", level=logging.ERROR) as captured:
                        response = self.client.post(
                            "/api/assistant",
                            json={"question": "Explain AI in simple terms."},
                        )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, "\n".join(captured.output))
        self.assertIn("try again", response.json()["detail"].casefold())

    def test_brand_assistant_maps_gemini_provider_errors_safely(self) -> None:
        from google.genai import errors

        cases = [
            (
                errors.ServerError(
                    503,
                    {"error": {"message": "temporary capacity issue", "status": "UNAVAILABLE"}},
                ),
                503,
                "provider_unavailable",
                "temporarily unavailable",
            ),
            (
                errors.ClientError(
                    429,
                    {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
                ),
                503,
                "quota_or_rate_limit",
                "quota",
            ),
            (
                errors.ClientError(
                    401,
                    {"error": {"message": "invalid credential", "status": "UNAUTHENTICATED"}},
                ),
                503,
                "authentication_or_permission",
                "credentials",
            ),
            (
                errors.ClientError(
                    404,
                    {"error": {"message": "unknown model", "status": "NOT_FOUND"}},
                ),
                503,
                "model_unavailable",
                "GEMINI_MODEL",
            ),
        ]
        for error, expected_status, expected_category, detail_marker in cases:
            with self.subTest(category=expected_category):
                status, category, detail = main.assistant_generation_error(error)
                self.assertEqual(status, expected_status)
                self.assertEqual(category, expected_category)
                self.assertIn(detail_marker.casefold(), detail.casefold())

    def test_brand_assistant_returns_saved_workspace_metrics_during_gemini_quota_failure(self) -> None:
        from google.genai import errors

        review = self.create_review("Quota test review", "Our guaranteed product helps.")
        stored = main.load_store()
        item = next(candidate for candidate in stored["reviews"] if candidate["id"] == review["id"])
        item["score"] = 73
        main.save_store(stored)

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                raise errors.ClientError(
                    429,
                    {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
                )

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "assistant-test-key"}):
            with patch("google.genai.Client", FakeClient):
                response = self.client.post(
                    "/api/assistant",
                    json={"question": "What is the average compliance score?"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn("average compliance score: 73.0/100", result["answer"].casefold())
        self.assertIn("quota", result["answer"].casefold())
        self.assertTrue(result["evidence_available"])
        self.assertTrue(any(
            source["source_id"] == "review_workspace_summary"
            for source in result["sources"]
        ))


if __name__ == "__main__":
    unittest.main()
