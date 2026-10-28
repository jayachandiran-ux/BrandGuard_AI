# BrandGuard AI

BrandGuard AI is a manager-only brand governance workspace created by **Team SYNAPSE** for **CODEX 48**. It reviews marketing content against saved brand guidance, reports evidence and findings, tracks review decisions, and provides basic workspace-level reporting. It does not include login, registration, client accounts, or a client portal.

## Core features and current maturity

| Feature | What it does | Current status and limitations |
|---|---|---|
| Brand Compliance Scoring | Produces a score from review findings and their severity. | Implemented and tested. It is a transparent rule-based penalty score, not a statistical or objective measure of brand quality. |
| Factual Claim Detection and Verification | Extracts factual-sounding sentences and compares them with approved facts and retrieved PDF passages. Classifications include Supported, Contradicted, and Unverified. | Partially working. Extraction uses sentence patterns and can miss claims; evidence checks do not replace external fact-checking. Missing evidence is not treated as falsity. |
| Tone and Brand Voice Analysis | Checks content against saved tone guidance and highlights configured terminology and language differences. | Partially working. Tone rules are narrow heuristics, not comprehensive sentiment analysis or objective tone measurement. |
| RAG-Based Brand Guideline Verification | Retrieves page-aware passages from uploaded guideline PDFs to provide evidence for reviews. | Implemented and covered by backend tests for extraction, embeddings, retrieval, citations, and persistence. PDF indexing requires Gemini credentials and network/API availability. |
| Suggested Content Revision | Shows revised copy when the review engine can provide a safe suggestion. | Partially working. Revisions are checked against extracted claims and numeric details; rule-based fallback preserves the original copy and asks the manager to make changes manually. The checks cannot guarantee semantic equivalence. |
| Brand DNA Drift and Analytics | Compares draft content with current guidelines and a selected approved review; Reports summarizes saved review data. | Partially working. Drift scoring is based on documented text rules, not a learned brand model or full longitudinal trend analysis. Reports is a basic frontend summary of saved reviews. |

## Technology stack

- **Frontend:** React, Vite, JavaScript, CSS
- **Backend:** Python, FastAPI, Pydantic
- **AI:** Google Gemini via `google-genai`
- **PDF and retrieval:** PyMuPDF, NumPy, FAISS
- **Persistence:** Local JSON workspace data and a persisted local FAISS index
- **Testing/build:** Python `unittest`, Oxlint, Vite production build

## Architecture

```text
frontend/
  src/                 React manager workspace and feature pages
  package.json         Frontend scripts and dependencies

backend/
  main.py              FastAPI routes, review logic, scoring, claims, versions, drift
  rag.py               PDF extraction, chunking, Gemini embeddings, FAISS retrieval
  tests/               Backend feature and RAG pipeline tests
  requirements.txt     Python dependencies
  .env.example         Safe local configuration template
```

The browser calls the FastAPI service on port `8000`. The API persists manager guidelines and review/version decisions in `backend/data.json`. Uploaded PDF text chunks, metadata, and the FAISS index are stored under `backend/rag_data/`. These are local workspace data and are intentionally excluded from Git. The JSON persistence layer is intended for a local single-manager MVP, not concurrent multi-user or production deployment.

## Requirements

- Python 3.10 or newer
- Node.js and npm
- A Gemini API key for Gemini review and PDF embedding/indexing

## Installation and configuration

From the project root, open two PowerShell terminals.

### Backend

```powershell
Set-Location .\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend\.env` locally and replace the placeholder with your Gemini API key:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Keep the real key in `backend\.env` only. Do not put it in frontend configuration, source code, issue reports, or commits. The backend loads this file locally; `.env` files are ignored by Git. Gemini model access and capacity are controlled by Google and can change or be temporarily unavailable. When generation is unavailable, the review endpoint identifies its rule-based fallback. PDF embedding requires working Gemini access.

### Frontend

```powershell
Set-Location .\frontend
npm install
```

## Run the application

Start the backend in the first terminal:

```powershell
Set-Location .\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in the second terminal:

```powershell
Set-Location .\frontend
npm run dev
```

Open <http://127.0.0.1:5173>. Backend health check: <http://127.0.0.1:8000/health>.

## Run tests and production build

```powershell
Set-Location .\backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

Set-Location ..\frontend
npm run lint
npm run build
```

## Current limitations

- Claim extraction is rule/pattern based and does not discover every factual assertion.
- Claim classifications depend on saved approved facts and retrieved evidence; they are not independent verification of real-world truth.
- Tone and brand voice rules are deliberately limited heuristics.
- Suggested-revision validation compares detected claims and numbers but cannot prove complete semantic equivalence.
- Brand DNA drift compares a draft with current saved guidance and an optional selected approved review. Its score is an explicitly rule-based consistency score, not an AI-derived truth or quality score.
- Reports aggregate reviews available in the local workspace; they are not a separate analytics service or a full historical trend engine.
- The local JSON store and FAISS index are not a production multi-user database. Back up `backend/data.json` and `backend/rag_data/` locally if you need to preserve workspace data; do not commit them.
- Gemini calls require valid credentials, network access, model availability, and sufficient API quota.

## Team

**Team SYNAPSE** — **CODEX 48**
