import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.30


class KnowledgeBaseError(RuntimeError):
    pass


class EmbeddingUnavailableError(KnowledgeBaseError):
    pass


def _client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EmbeddingUnavailableError("Set GEMINI_API_KEY in backend/.env to generate document embeddings.")
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30000))


def redact_api_key(message: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    return message.replace(api_key, "[redacted]") if api_key else message


def _normalise_vectors(vectors: list[list[float]]) -> np.ndarray:
    if not vectors:
        raise KnowledgeBaseError("The embedding service returned no vectors.")
    if any(not vector for vector in vectors):
        raise KnowledgeBaseError("The embedding service returned an empty vector.")
    if len({len(vector) for vector in vectors}) != 1:
        raise KnowledgeBaseError("The embedding service returned vectors with inconsistent dimensions.")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise KnowledgeBaseError("The embedding service returned invalid vectors.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise KnowledgeBaseError("The embedding service returned a zero-length vector.")
    faiss.normalize_L2(matrix)
    return matrix


def generate_embeddings(texts: list[str]) -> np.ndarray:
    if any(not isinstance(text, str) for text in texts):
        raise KnowledgeBaseError("Embedding inputs must be text.")
    texts = [text for text in texts if text.strip()]
    if not texts:
        raise KnowledgeBaseError("Cannot embed an empty text batch.")
    client = _client()
    from google.genai import types

    vectors: list[list[float]] = []
    batch_size = 32
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        contents = [
            types.UserContent(parts=[types.Part.from_text(text=text)])
            for text in batch
        ]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise KnowledgeBaseError(
                f"The embedding service returned {len(embeddings)} vectors for "
                f"{len(batch)} texts in the batch starting at item {offset + 1}."
            )
        for embedding in embeddings:
            if not embedding.values:
                raise KnowledgeBaseError("The embedding service returned an empty vector.")
            vectors.append(embedding.values)
    return _normalise_vectors(vectors)


def embed_documents(chunks: list[dict]) -> np.ndarray:
    nonempty_chunks = [
        chunk for chunk in chunks
        if isinstance(chunk.get("text"), str) and chunk["text"].strip()
    ]
    formatted = [
        f"title: {chunk['filename']} | text: {chunk['text']}"
        for chunk in nonempty_chunks
    ]
    return generate_embeddings(formatted)


def embed_query(query: str) -> np.ndarray:
    formatted = f"task: search result | query: {query}"
    return generate_embeddings([formatted])


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size.")
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = max(
                cleaned.rfind(". ", start, end),
                cleaned.rfind("; ", start, end),
                cleaned.rfind(", ", start, end),
                cleaned.rfind(" ", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        excerpt = cleaned[start:end].strip()
        if excerpt:
            chunks.append(excerpt)
        if end >= len(cleaned):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and cleaned[next_start].isalnum() and cleaned[next_start - 1].isalnum():
            next_start += 1
        start = next_start
    return chunks


def make_chunks(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    result: list[dict] = []
    for page in pages:
        page_number = int(page["page_number"])
        for chunk_number, text in enumerate(
            chunk_text(page.get("text", ""), chunk_size=chunk_size, overlap=overlap)
        ):
            chunk_id = f"{document_id}-p{page_number}-c{chunk_number}"
            result.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "text": text,
            })
    return result


class KnowledgeBase:
    def __init__(
        self,
        root: str | Path | None = None,
        document_embedder: Callable[[list[dict]], np.ndarray] = embed_documents,
        query_embedder: Callable[[str], np.ndarray] = embed_query,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        self.root = Path(root or os.getenv("BRANDGUARD_RAG_DIR", Path(__file__).resolve().parent / "rag_data"))
        self.document_embedder = document_embedder
        self.query_embedder = query_embedder
        self.embedding_model = embedding_model
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", chunk_size))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", chunk_overlap))
        self.top_k = int(os.getenv("RAG_TOP_K", top_k))
        self.min_score = float(os.getenv("RAG_MIN_SCORE", min_score))
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("RAG_CHUNK_SIZE must be positive and RAG_CHUNK_OVERLAP smaller than it.")
        if self.top_k < 1 or not -1 <= self.min_score <= 1:
            raise ValueError("RAG_TOP_K must be positive and RAG_MIN_SCORE between -1 and 1.")
        self._lock = threading.RLock()

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _read_state(self) -> tuple[dict, str | None]:
        if not self._manifest_path().exists():
            return {"documents": [], "chunks": []}, None
        try:
            manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
            generation = manifest["generation"]
            if not re.fullmatch(r"[a-f0-9]{32}", generation):
                raise ValueError("Invalid snapshot identifier.")
            state_path = self.root / f"state-{generation}.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state.get("documents"), list) or not isinstance(state.get("chunks"), list):
                raise ValueError("Invalid knowledge-base metadata.")
            return state, generation
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise KnowledgeBaseError(f"Knowledge-base metadata could not be loaded: {exc}") from exc

    def _read_index(self, generation: str | None, chunk_count: int) -> faiss.Index | None:
        if chunk_count == 0:
            return None
        if not generation:
            raise KnowledgeBaseError("Knowledge-base metadata exists without a FAISS index.")
        index_path = self.root / f"index-{generation}.faiss"
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise KnowledgeBaseError(f"FAISS index could not be loaded: {exc}") from exc
        if index.ntotal != chunk_count or index.d != self._metadata_dimension(generation):
            raise KnowledgeBaseError("FAISS index and chunk metadata do not match.")
        return index

    def _metadata_dimension(self, generation: str) -> int:
        try:
            manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
            if manifest.get("generation") != generation:
                raise ValueError("Snapshot changed.")
            return int(manifest["dimension"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise KnowledgeBaseError(f"Knowledge-base manifest is invalid: {exc}") from exc

    def _save_snapshot(self, state: dict, index: faiss.Index | None) -> None:
        if index is None or index.ntotal != len(state["chunks"]):
            raise KnowledgeBaseError("Cannot persist a mismatched knowledge-base snapshot.")
        generation = uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        index_path = self.root / f"index-{generation}.faiss"
        state_path = self.root / f"state-{generation}.json"
        index_tmp = index_path.with_suffix(".tmp")
        state_tmp = state_path.with_suffix(".tmp")
        manifest_tmp = self._manifest_path().with_suffix(".tmp")
        try:
            faiss.write_index(index, str(index_tmp))
            state_tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            index_tmp.replace(index_path)
            state_tmp.replace(state_path)
            manifest_tmp.write_text(
                json.dumps({"generation": generation, "dimension": index.d, "embedding_model": self.embedding_model}),
                encoding="utf-8",
            )
            manifest_tmp.replace(self._manifest_path())
        except OSError as exc:
            raise KnowledgeBaseError(f"Could not persist the FAISS index: {exc}") from exc

    def _build_index(self, chunks: list[dict]) -> faiss.Index:
        try:
            vectors = self.document_embedder(chunks)
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Document embedding generation failed: {redact_api_key(str(exc))}"
            ) from exc
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise KnowledgeBaseError("Document embedding count does not match the chunk count.")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise KnowledgeBaseError("Document embeddings contain invalid values.")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise KnowledgeBaseError("Document embeddings contain zero-length vectors.")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def _recover_index(self, state: dict) -> faiss.Index:
        if not state["chunks"]:
            raise KnowledgeBaseError("There are no stored chunks to rebuild the index from.")
        index = self._build_index(state["chunks"])
        self._save_snapshot(state, index)
        return index

    def ingest_pages(
        self,
        filename: str,
        file_sha256: str,
        pages: list[dict],
        page_count: int,
    ) -> tuple[dict, bool]:
        with self._lock:
            state, generation = self._read_state()
            for existing in state["documents"]:
                if existing["sha256"] == file_sha256:
                    return self._public_document(existing), True

            document_id = uuid.uuid5(uuid.NAMESPACE_URL, file_sha256).hex
            new_chunks = make_chunks(
                document_id,
                filename,
                pages,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            if not new_chunks:
                raise KnowledgeBaseError("No non-empty text chunks were extracted from the PDF.")

            all_chunks = [*state["chunks"], *new_chunks]
            state["chunks"] = all_chunks
            document = {
                "document_id": document_id,
                "filename": filename,
                "sha256": file_sha256,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "page_count": page_count,
                "chunk_count": len(new_chunks),
                "extracted_pages": pages,
            }
            state["documents"].append(document)

            try:
                current_manifest = {}
                if generation:
                    current_manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
                if (
                    generation
                    and current_manifest.get("embedding_model") == self.embedding_model
                    and len(state["chunks"]) > len(new_chunks)
                ):
                    try:
                        index = self._read_index(generation, len(state["chunks"]) - len(new_chunks))
                    except KnowledgeBaseError:
                        index = self._build_index(state["chunks"])
                    else:
                        try:
                            vectors = self.document_embedder(new_chunks)
                        except EmbeddingUnavailableError:
                            raise
                        except Exception as exc:
                            raise KnowledgeBaseError(
                                f"Document embedding generation failed: {redact_api_key(str(exc))}"
                            ) from exc
                        if vectors.ndim != 2 or vectors.shape[0] != len(new_chunks) or vectors.shape[1] != index.d:
                            raise KnowledgeBaseError("New embeddings do not match the existing FAISS index.")
                        vectors = np.asarray(vectors, dtype=np.float32)
                        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
                            raise KnowledgeBaseError("New document embeddings contain invalid vectors.")
                        faiss.normalize_L2(vectors)
                        index.add(vectors)
                else:
                    index = self._build_index(state["chunks"])
                self._save_snapshot(state, index)
            except Exception:
                raise
            return self._public_document(document), False

    @staticmethod
    def _public_document(document: dict) -> dict:
        return {
            "document_id": document["document_id"],
            "filename": document["filename"],
            "uploaded_at": document["uploaded_at"],
            "page_count": document["page_count"],
            "chunk_count": document["chunk_count"],
            "extracted_characters": sum(
                len(page.get("text", "")) for page in document.get("extracted_pages", [])
            ),
            "status": "Indexed",
            "ingestion_status": "Indexed",
        }

    def list_documents(self) -> dict:
        with self._lock:
            try:
                state, generation = self._read_state()
                if state["chunks"]:
                    try:
                        self._read_index(generation, len(state["chunks"]))
                    except KnowledgeBaseError as exc:
                        return {
                            "status": "failed",
                            "documents": [self._public_document(item) for item in state["documents"]],
                            "chunk_count": len(state["chunks"]),
                            "error": str(exc),
                        }
                return {
                    "status": "ready" if state["documents"] else "empty",
                    "documents": [self._public_document(item) for item in state["documents"]],
                    "chunk_count": len(state["chunks"]),
                }
            except KnowledgeBaseError as exc:
                return {"status": "failed", "documents": [], "chunk_count": 0, "error": str(exc)}

    def retrieve(self, query: str, top_k: int | None = None) -> dict:
        with self._lock:
            try:
                state, generation = self._read_state()
                if not state["chunks"]:
                    return {"status": "empty", "sources": [], "error": None}
                if not os.getenv("GEMINI_API_KEY") and self.query_embedder is embed_query:
                    return {
                        "status": "unavailable",
                        "sources": [],
                        "error": "GEMINI_API_KEY is required to create query embeddings.",
                    }
                manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
                try:
                    if manifest.get("embedding_model") != self.embedding_model:
                        index = self._recover_index(state)
                    else:
                        index = self._read_index(generation, len(state["chunks"]))
                except KnowledgeBaseError:
                    index = self._recover_index(state)
                query_vector = np.asarray(self.query_embedder(query), dtype=np.float32)
                if query_vector.ndim == 1:
                    query_vector = query_vector.reshape(1, -1)
                if query_vector.shape != (1, index.d) or not np.isfinite(query_vector).all():
                    raise KnowledgeBaseError("Query embedding does not match the stored FAISS index.")
                faiss.normalize_L2(query_vector)
                count = min(max(1, int(top_k or self.top_k)), len(state["chunks"]))
                scores, positions = index.search(query_vector, count)
                sources = []
                for score, position in zip(scores[0], positions[0]):
                    if position < 0 or float(score) < self.min_score:
                        continue
                    chunk = state["chunks"][int(position)]
                    sources.append({
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page_number": chunk["page_number"],
                        "excerpt": chunk["text"],
                        "similarity": round(float(score), 4),
                    })
                return {
                    "status": "retrieved" if sources else "no_relevant_sources",
                    "sources": sources,
                    "error": None,
                }
            except EmbeddingUnavailableError as exc:
                return {"status": "unavailable", "sources": [], "error": redact_api_key(str(exc))}
            except Exception as exc:
                return {"status": "failed", "sources": [], "error": redact_api_key(str(exc))}
