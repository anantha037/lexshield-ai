"""
LexShield AI — Main FastAPI Application
========================================
Entry point. Mounts all routers and middleware.

Run:
  uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ── Create DB tables on startup (safe to run every time — skips if exist) ──────
from models.database import create_tables
create_tables()

app = FastAPI(
    title="LexShield AI",
    description="AI-Powered Indian Legal Intelligence Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
from api.auth        import router as auth_router
from api.document    import router as document_router
from api.legal       import router as legal_router
from api.orchestator import router as orchestrator_router
from api.classify    import router as classify_router
from api.master      import router as master_router

app.include_router(auth_router)
app.include_router(master_router)
app.include_router(classify_router)
app.include_router(document_router)
app.include_router(legal_router)
app.include_router(orchestrator_router)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """
    Returns status of all core services.
    Checks: ChromaDB connection, LLM reachability, embedding model.
    """
    status = {
        "service":  "LexShield AI",
        "version":  "1.0.0",
        "chromadb": "unknown",
        "llm":      "unknown",
        "embedder": "unknown",
    }

    try:
        from rag.vectorstore import vectorstore
        count = vectorstore.count()
        status["chromadb"] = f"ok — {count} chunks indexed"
    except Exception as e:
        status["chromadb"] = f"error: {e}"

    try:
        from rag.embedder import embedder
        _ = embedder.embed_single("test")
        status["embedder"] = f"ok — {embedder.model_name}"
    except Exception as e:
        status["embedder"] = f"error: {e}"

    try:
        from rag.llm import llm
        _ = llm.generate("Reply with the single word: ok", max_tokens=5)
        status["llm"] = f"ok — {llm.model}"
    except Exception as e:
        status["llm"] = f"error: {e}"

    all_ok = all("ok" in str(v) for k, v in status.items() if k not in ("service", "version"))
    status["overall"] = "healthy" if all_ok else "degraded"

    return status