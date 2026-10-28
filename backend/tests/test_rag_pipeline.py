import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pymupdf
from fastapi.testclient import TestClient

import main
from rag import (
    EMBEDDING_MODEL,
    KnowledgeBase,
    KnowledgeBaseError,
    embed_documents,
    generate_embeddings,
    make_chunks,
)


class ControlledEmbeddings:
    terms = ("voice", "clear", "warm", "warranty", "portland", "recipes")

    @classmethod
    def vector(cls, text: str) -> np.ndarray:
        lowered = text.lower()
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


def create_pdf(*pages: str) -> bytes:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


class RagPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.knowledge_base = KnowledgeBase(
            root=self.root / "rag_data",
            document_embedder=ControlledEmbeddings.documents,
            query_embedder=ControlledEmbeddings.query,
            min_score=0.6,
        )
        self.old_knowledge_base = main.knowledge_base
        self.old_data_file = main.DATA_FILE
        main.knowledge_base = self.knowledge_base
        main.DATA_FILE = self.root / "data.json"
        self.addCleanup(setattr, main, "knowledge_base", self.old_knowledge_base)
        self.addCleanup(setattr, main, "DATA_FILE", self.old_data_file)
        self.client = TestClient(main.app)

    def test_pdf_extraction_chunking_and_page_metadata(self) -> None:
        pages, page_count = main.extract_pdf_pages(
            create_pdf(
                "Use a warm, clear brand voice. Keep claims specific.",
                "Warranty terms must match approved company facts.",
            )
        )
        chunks = make_chunks(
            "doc-1",
            "brand-guide.pdf",
            pages,
            chunk_size=32,
            overlap=6,
        )

        self.assertEqual(page_count, 2)
        self.assertEqual([page["page_number"] for page in pages], [1, 2])
        self.assertTrue(chunks)
        self.assertTrue(all(chunk["text"].strip() for chunk in chunks))
        self.assertEqual({chunk["page_number"] for chunk in chunks}, {1, 2})
        self.assertTrue(all(chunk["filename"] == "brand-guide.pdf" for chunk in chunks))
        self.assertTrue(all(chunk["chunk_id"].startswith("doc-1-p") for chunk in chunks))

    def test_upload_retrieval_and_persistence_across_reload(self) -> None:
        pdf = create_pdf(
            "Use a warm, clear brand voice in every message.",
            "Warranty terms must match approved company facts.",
        )
        uploaded = self.client.post(
            "/api/documents",
            files={
                "file": (
                    "brand-guide.pdf",
                    pdf,
                    "application/pdf",
                )
            },
        )
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        source_document = uploaded.json()
        self.assertEqual(source_document["filename"], "brand-guide.pdf")
        self.assertEqual(source_document["page_count"], 2)
        self.assertGreater(source_document["extracted_characters"], 0)
        self.assertGreater(source_document["chunk_count"], 0)
        self.assertEqual(source_document["ingestion_status"], "Indexed")
        duplicate = self.client.post(
            "/api/documents",
            files={"file": ("renamed-copy.pdf", pdf, "application/pdf")},
        )
        self.assertEqual(duplicate.status_code, 200)
        self.assertTrue(duplicate.json()["duplicate"])
        self.assertEqual(duplicate.json()["page_count"], 2)
        self.assertEqual(len(self.knowledge_base.list_documents()["documents"]), 1)
        retrieved = self.knowledge_base.retrieve("Write in a clear warm voice")
        self.assertEqual(retrieved["status"], "retrieved")
        self.assertEqual(retrieved["sources"][0]["filename"], "brand-guide.pdf")
        self.assertEqual(retrieved["sources"][0]["page_number"], 1)

        unrelated = self.knowledge_base.retrieve("Share banana recipes")
        self.assertEqual(unrelated["status"], "no_relevant_sources")

        reloaded = KnowledgeBase(
            root=self.root / "rag_data",
            document_embedder=ControlledEmbeddings.documents,
            query_embedder=ControlledEmbeddings.query,
            min_score=0.6,
        )
        after_restart = reloaded.retrieve("warm and clear voice")
        self.assertEqual(after_restart["status"], "retrieved")
        self.assertEqual(after_restart["sources"][0]["chunk_id"], retrieved["sources"][0]["chunk_id"])
        self.assertEqual(reloaded.list_documents()["documents"][0]["document_id"], source_document["document_id"])
        restart_script = """
import json
import sys
import numpy as np
from rag import KnowledgeBase

def vector(text):
    lowered = text.lower()
    terms = ("voice", "clear", "warm", "warranty", "portland", "recipes")
    return np.asarray([0.05, *(1.0 if term in lowered else 0.0 for term in terms)], dtype=np.float32)

def embed_documents(chunks):
    return np.vstack([vector(chunk["text"]) for chunk in chunks])

def embed_query(text):
    return vector(text).reshape(1, -1)

kb = KnowledgeBase(root=sys.argv[1], document_embedder=embed_documents, query_embedder=embed_query, min_score=0.6)
print(json.dumps(kb.retrieve("warm and clear voice")))
"""
        restarted_process = subprocess.run(
            [sys.executable, "-c", restart_script, str(self.root / "rag_data")],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(restarted_process.returncode, 0, restarted_process.stderr)
        restarted_result = json.loads(restarted_process.stdout)
        self.assertEqual(restarted_result["status"], "retrieved")
        self.assertEqual(restarted_result["sources"][0]["chunk_id"], retrieved["sources"][0]["chunk_id"])

    def test_embedding_wrapper_uses_sdk_model_and_normalizes_output(self) -> None:
        calls: list[dict] = []

        class FakeModels:
            def embed_content(self, **kwargs: object) -> object:
                calls.append(kwargs)
                embedding = type("Embedding", (), {"values": [3.0, 4.0]})()
                return type("Response", (), {"embeddings": [embedding]})()

        client = type("Client", (), {"models": FakeModels()})()
        with patch("rag._client", return_value=client):
            vectors = generate_embeddings(["a safe test string"])

        self.assertEqual(calls[0]["model"], EMBEDDING_MODEL)
        contents = calls[0]["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].parts[0].text, "a safe test string")
        np.testing.assert_allclose(vectors, np.asarray([[0.6, 0.8]], dtype=np.float32))

    def test_embedding_wrapper_preserves_batch_mapping_and_filters_empty_text(self) -> None:
        calls: list[list[str]] = []

        class FakeModels:
            def embed_content(self, *, model: str, contents: list[object]) -> object:
                texts = [content.parts[0].text for content in contents]
                calls.append(texts)
                embeddings = [
                    type("Embedding", (), {"values": [float(index + 1), 1.0]})()
                    for index, _ in enumerate(texts)
                ]
                return type("Response", (), {"embeddings": embeddings})()

        client = type("Client", (), {"models": FakeModels()})()
        texts = [f"chunk {index}" for index in range(33)]
        texts.insert(4, "  ")
        with patch("rag._client", return_value=client):
            vectors = generate_embeddings(texts)

        self.assertEqual([len(batch) for batch in calls], [32, 1])
        self.assertEqual(calls[0][4], "chunk 4")
        self.assertEqual(calls[1], ["chunk 32"])
        self.assertEqual(vectors.shape, (33, 2))
        np.testing.assert_allclose(
            np.linalg.norm(vectors, axis=1),
            np.ones(33),
            rtol=2e-7,
            atol=2e-7,
        )

    def test_embedding_wrapper_rejects_missing_batch_vectors(self) -> None:
        class FakeModels:
            def embed_content(self, **kwargs: object) -> object:
                embedding = type("Embedding", (), {"values": [1.0, 0.0]})()
                return type("Response", (), {"embeddings": [embedding]})()

        client = type("Client", (), {"models": FakeModels()})()
        with patch("rag._client", return_value=client):
            with self.assertRaisesRegex(
                KnowledgeBaseError,
                "returned 1 vectors for 2 texts in the batch starting at item 1",
            ):
                generate_embeddings(["first text", "second text"])

    def test_document_embeddings_skip_empty_chunks_without_reordering(self) -> None:
        submitted: list[list[str]] = []

        class FakeModels:
            def embed_content(self, *, model: str, contents: list[object]) -> object:
                texts = [content.parts[0].text for content in contents]
                submitted.append(texts)
                embeddings = [
                    type("Embedding", (), {"values": [float(index + 1), 1.0]})()
                    for index, _ in enumerate(texts)
                ]
                return type("Response", (), {"embeddings": embeddings})()

        client = type("Client", (), {"models": FakeModels()})()
        chunks = [
            {"filename": "guide.pdf", "page_number": 1, "text": "first passage"},
            {"filename": "guide.pdf", "page_number": 1, "text": "   "},
            {"filename": "guide.pdf", "page_number": 2, "text": "second passage"},
        ]
        with patch("rag._client", return_value=client):
            vectors = embed_documents(chunks)

        self.assertEqual(len(submitted[0]), 2)
        self.assertIn("first passage", submitted[0][0])
        self.assertIn("second passage", submitted[0][1])
        self.assertEqual(vectors.shape, (2, 2))

    def test_gemini_review_errors_redact_api_key_from_logs(self) -> None:
        api_key = "test-key-never-log"
        output = StringIO()
        guidelines = main.Guidelines(brand_name="Test")
        retrieval = {"status": "empty", "sources": []}
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}), \
                patch("google.genai.Client", side_effect=RuntimeError(f"request failed: {api_key}")), \
                redirect_stdout(output):
            result = main.gemini_review("test content", guidelines, retrieval, [])

        self.assertIsNone(result)
        self.assertNotIn(api_key, output.getvalue())
        self.assertIn("[redacted]", output.getvalue())

    def test_review_supplies_retrieved_excerpt_and_filters_fabricated_ids(self) -> None:
        self.knowledge_base.ingest_pages(
            "brand-guide.pdf",
            "a" * 64,
            [{"page_number": 3, "text": "Use a warm, clear voice in every message."}],
            3,
        )
        source = self.knowledge_base.retrieve("warm clear voice")["sources"][0]
        captured_prompt: dict = {}

        class FakeModels:
            def generate_content(self, *, model: str, contents: str, config: object) -> object:
                captured_prompt.update(json.loads(contents))
                response = {
                    "issues": [
                        {
                            "category": "Brand voice and tone",
                            "severity": "Medium",
                            "problematic_text": "Use confusing jargon",
                            "explanation": "The wording conflicts with the supplied voice guidance.",
                            "suggested_correction": "Use clear language.",
                            "evidence": "Use a warm, clear voice in every message.",
                            "source_chunk_ids": [source["chunk_id"], "invented-source-id"],
                        }
                    ],
                    "revised_content": "Use clear language.",
                }
                return type("Response", (), {"text": json.dumps(response)})()

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.models = FakeModels()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-not-a-real-secret"}):
            with patch("google.genai.Client", FakeClient):
                response = self.client.post(
                    "/api/reviews",
                    json={
                        "title": "Voice check",
                        "content_type": "Blog",
                        "content": "Use confusing jargon in a warm, clear voice.",
                    },
                )

        self.assertEqual(response.status_code, 201, response.text)
        review = response.json()
        self.assertEqual(review["engine"], "Gemini RAG")
        self.assertTrue(review["context_used_by_model"])
        self.assertEqual(captured_prompt["retrieved_sources"][0]["excerpt"], source["excerpt"])
        self.assertEqual(captured_prompt["retrieved_sources"][0]["filename"], "brand-guide.pdf")
        self.assertEqual(captured_prompt["retrieved_sources"][0]["page_number"], 3)
        self.assertEqual(review["sources"], [source])
        self.assertEqual(review["issues"][0]["source_chunk_ids"], [source["chunk_id"]])

    def test_empty_knowledge_base_and_missing_credentials_keep_fallback_working(self) -> None:
        main.knowledge_base = KnowledgeBase(root=self.root / "no_key_rag")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            review = self.client.post(
                "/api/reviews",
                json={
                    "title": "No PDF sources",
                    "content_type": "Social media",
                    "content": "A calm message.",
                },
            )
            upload = self.client.post(
                "/api/documents",
                files={
                    "file": (
                        "guide.pdf",
                        create_pdf("Use clear language."),
                        "application/pdf",
                    )
                },
            )

        self.assertEqual(review.status_code, 201, review.text)
        self.assertEqual(review.json()["retrieval_status"], "empty")
        self.assertEqual(review.json()["engine"], "Rule-based fallback")
        self.assertFalse(review.json()["context_used_by_model"])
        self.assertEqual(upload.status_code, 503)
        self.assertIn("GEMINI_API_KEY", upload.json()["detail"])

    def test_malformed_and_non_text_pdfs_are_rejected(self) -> None:
        malformed = self.client.post(
            "/api/documents",
            files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
        )
        image_only = self.client.post(
            "/api/documents",
            files={"file": ("image.pdf", create_pdf(""), "application/pdf")},
        )
        self.assertEqual(malformed.status_code, 415)
        self.assertIn("valid PDF signature", malformed.json()["detail"])
        self.assertEqual(image_only.status_code, 422)
        self.assertIn("OCR may be required", image_only.json()["detail"])
        self.assertIn("OCR is not implemented", image_only.json()["detail"])

    def test_empty_oversized_and_unsupported_uploads_return_specific_errors(self) -> None:
        empty = self.client.post(
            "/api/documents",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        unsupported = self.client.post(
            "/api/documents",
            files={"file": ("guidelines.txt", b"not a pdf", "text/plain")},
        )
        oversized = self.client.post(
            "/api/documents",
            files={
                "file": (
                    "large.pdf",
                    b"%PDF-" + b"x" * main.MAX_PDF_BYTES,
                    "application/pdf",
                )
            },
        )

        self.assertEqual(empty.status_code, 422)
        self.assertIn("PDF is empty", empty.json()["detail"])
        self.assertEqual(unsupported.status_code, 415)
        self.assertIn("PDF file", unsupported.json()["detail"])
        self.assertEqual(oversized.status_code, 413)
        self.assertIn("15 MB", oversized.json()["detail"])

    def test_embedding_failure_does_not_replace_previously_indexed_documents(self) -> None:
        first = self.client.post(
            "/api/documents",
            files={"file": ("first.pdf", create_pdf("Use a warm, clear brand voice."), "application/pdf")},
        )
        self.assertEqual(first.status_code, 201, first.text)

        def fail_embedding(_chunks: list[dict]) -> np.ndarray:
            raise RuntimeError("simulated provider outage")

        self.knowledge_base.document_embedder = fail_embedding
        second = self.client.post(
            "/api/documents",
            files={"file": ("second.pdf", create_pdf("A separate guideline passage."), "application/pdf")},
        )

        self.assertEqual(second.status_code, 503)
        self.assertIn("Document embedding generation failed", second.json()["detail"])
        self.assertEqual(
            [item["filename"] for item in self.knowledge_base.list_documents()["documents"]],
            ["first.pdf"],
        )

    def test_upload_route_accepts_browser_cors_origin(self) -> None:
        response = self.client.options(
            "/api/documents",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://127.0.0.1:5173",
        )


if __name__ == "__main__":
    unittest.main()
