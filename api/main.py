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

app = FastAPI(
    title="LexShield AI",
    description="AI-Powered Indian Legal Intelligence Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from api.document     import router as document_router
from api.legal        import router as legal_router
from api.orchestator  import router as orchestrator_router

app.include_router(document_router)
app.include_router(legal_router)
app.include_router(orchestrator_router)



# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """
    Returns status of all core services.
    Checks: ChromaDB connection, LLM reachability, embedding model.
    """
    status = {
        "service": "LexShield AI",
        "version": "1.0.0",
        "chromadb": "unknown",
        "llm":      "unknown",
        "embedder": "unknown",
    }

    # Check ChromaDB
    try:
        from rag.vectorstore import vectorstore
        count = vectorstore.count()
        status["chromadb"] = f"ok — {count} chunks indexed"
    except Exception as e:
        status["chromadb"] = f"error: {e}"

    # Check embedder
    try:
        from rag.embedder import embedder
        _ = embedder.embed_single("test")
        status["embedder"] = f"ok — {embedder.model_name}"
    except Exception as e:
        status["embedder"] = f"error: {e}"

    # Check LLM (Groq)
    try:
        from rag.llm import llm
        _ = llm.generate("Reply with the single word: ok", max_tokens=5)
        status["llm"] = f"ok — {llm.model}"
    except Exception as e:
        status["llm"] = f"error: {e}"

    all_ok = all("ok" in str(v) for k, v in status.items() if k != "service" and k != "version")
    status["overall"] = "healthy" if all_ok else "degraded"

    return status