# LexShield AI 🏛️
### AI-Powered Indian Legal Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Status-Week%201%20Complete-green" />
  <img src="https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-blue" />
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-orange" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-teal" />
  <img src="https://img.shields.io/badge/OCR-Tesseract-red" />
  <img src="https://img.shields.io/badge/Build-4%20Weeks-purple" />
</p>

---

## The Problem

India has **50+ million pending court cases**. Most citizens cannot afford a lawyer. Legal documents are written in language ordinary people do not understand. When someone's landlord illegally withholds their deposit, their employer steals their wages, or they face a false case — they have nowhere to turn.

Existing tools fail in every way that matters:
- Search engines return PDFs no one can read
- Generic AI chatbots hallucinate section numbers and punishments
- No tool understands scanned Indian legal documents
- No tool is aware of jurisdiction-specific state laws
- No multilingual support for regional languages

---

## What LexShield AI Does

LexShield AI makes Indian law accessible to every citizen regardless of language, location, or legal knowledge.

- **Document Intelligence** — Upload a court order, rental agreement, or legal notice as an image or PDF. LexShield extracts the text using OCR and explains the legal implications in plain language.
- **Legal Q&A** — Ask any legal question in natural language. Get a grounded answer citing real sections from IPC, BNS, CrPC, Consumer Protection Act, Payment of Wages Act, and Kerala Rent Control Act.
- **Zero Hallucination** — Every answer is grounded strictly in retrieved legal sections. If the knowledge base does not have it, LexShield says so explicitly.
- **Citation on Every Answer** — Every claim is backed by the exact source law and section, with a relevance score.

---

## Week 1 — What Is Built

> **Status: Complete ✅**

| Component | Description | Status |
|---|---|---|
| FastAPI Backend | 5 endpoints, Swagger UI, health check | ✅ |
| CV Pipeline | OpenCV preprocessing + Tesseract OCR (eng + mal + hin) | ✅ |
| Legal Knowledge Base | 4,148 chunks from IPC, BNS, CrPC, Consumer Protection Act, Payment of Wages Act, Kerala Rent Control Act, SC Judgments | ✅ |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU-safe) | ✅ |
| Vector Store | ChromaDB persistent local store with cosine similarity search | ✅ |
| RAG Pipeline | Query → embed → retrieve → filter → prompt → LLM → cited answer | ✅ |
| LLM Integration | Groq LLaMA 3.3 70B, temperature 0.1, anti-hallucination prompt | ✅ |
| Orchestrator | Routes document uploads vs text queries through correct pipeline | ✅ |
| Test Suite | pytest covering all components | ✅ |

---

## Architecture (Week 1)

```
User Query ──────────────────────────────────────────────────────────┐
                                                                      │
                              ┌─────────────┐                         │
                              │ Orchestrator│◄────────────────────────┘
                              └──────┬──────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                                             │
              ▼                                             ▼
   ┌─────────────────────┐                      ┌─────────────────────┐
   │   CV Pipeline       │                      │   RAG Pipeline      │
   │  OpenCV + Tesseract │                      │                     │
   │  Image/PDF → Text   │                      │  Query Preprocessor │
   └──────────┬──────────┘                      │  ChromaDB Search    │
              │                                 │  Prompt Builder     │
              │ extracted text                  │  Groq LLaMA 3.3 70B │
              └─────────────────────────────────►  Citation Engine    │
                                                └──────────┬──────────┘
                                                           │
                                                           ▼
                                                  Grounded Answer
                                                  + Citations
                                                  + Relevance Scores
```

---

## Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Backend | FastAPI + Uvicorn | Fast, async, automatic Swagger UI |
| LLM | Groq (LLaMA 3.3 70B) | Free tier, fast LPU inference, strong legal reasoning |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | CPU-safe, 90MB, 384-dim, no GPU needed |
| Vector DB | ChromaDB (PersistentClient) | Free, local, cosine similarity, metadata filtering |
| OCR | OpenCV + Tesseract | Free, multilingual (eng+mal+hin), handles scanned docs |
| PDF Text | PyMuPDF (fitz) | Direct text layer extraction for digital PDFs |
| PDF→Image | pdf2image + Poppler | Page-by-page conversion for scanned PDFs |
| Data | indiacode.nic.in + HuggingFace | Official government source + academic SC judgment datasets |
| Testing | pytest | Full component and integration test coverage |
| Container | Docker Compose | ChromaDB + API containerized |

---

## Legal Knowledge Base

| Source | Type | Chunks |
|---|---|---|
| IPC 1860 | Statute | included |
| BNS 2023 (replaced IPC) | Statute | included |
| CrPC 1973 | Statute | included |
| Consumer Protection Act 2019 | Statute | included |
| Payment of Wages Act 1936 | Statute | included |
| Kerala Rent Control Act 1965 | Statute | included |
| BNS Handbook 2024 | Reference | included |
| Supreme Court Judgments | Judgment | 3,000 chunks |
| **Total** | | **4,148 chunks** |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Live status of all services — ChromaDB, LLM, embedder |
| POST | `/api/v1/legal/query` | Legal Q&A — natural language question → cited answer |
| POST | `/api/v1/document/analyze` | OCR — image or PDF → extracted text |
| POST | `/api/v1/orchestrate/query` | Text query via orchestrator |
| POST | `/api/v1/orchestrate/document` | Document upload via orchestrator → OCR → legal analysis |

---

## Local Setup

### Prerequisites
- Python 3.11+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) with `eng`, `mal`, `hin` language packs
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (Windows only, for pdf2image)
- Free [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/lexshield-ai.git
cd lexshield-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key_here
INDIAN_KANOON_API_KEY=your_key_when_available
```

### Build the Knowledge Base (One-Time)

```bash
# Download legal datasets from HuggingFace
python data/download_datasets.py

# Process and chunk all legal documents
python data/preprocessor.py

# Embed chunks and store in ChromaDB (~45 minutes on CPU)
python rag/ingest.py
```

### Run the API

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Run Tests

```bash
pytest tests/ -v
```

---

## Project Roadmap

| Week | Focus | Status |
|---|---|---|
| Week 1 | Infrastructure, CV Pipeline, Embeddings, Basic RAG | ✅ Complete |
| Week 2 | Hybrid BM25 + Semantic Search, NLP/NER, Query Rewriting | 🔄 In Progress |
| Week 3 | Multi-Agent System (LangGraph), Legal Drafting Agent, Multilingual | ⏳ Planned |
| Week 4 | MLOps (RAGAS eval, MLflow), GCP Cloud Run Deployment | ⏳ Planned |

---

## Known Limitations (Week 1)

| Limitation | Planned Fix |
|---|---|
| Pure semantic search struggles with bare section number queries ("216 IPC") | Week 2: Hybrid BM25 + semantic search |
| No conversation memory — each query is stateless | Week 2: LangGraph stateful agents |
| English only in RAG pipeline | Week 2: IndicBERT multilingual agent |
| No document risk scoring yet | Week 2: XGBoost risk classifier |

---

## Built By

**Anantha Krishnan K**
CS Graduate — Hansraj College, University of Delhi
Aspiring AI/ML Engineer

[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) • [GitHub](https://github.com/YOUR_USERNAME)

---

*LexShield AI is a portfolio project. It provides legal information, not legal advice. For specific legal situations, consult a qualified lawyer.*
