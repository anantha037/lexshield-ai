"""
LexShield AI — Main FastAPI Application  (Session 6 — Final)
=============================================================
Entry point. Mounts all routers and middleware.

Changes in Session 6:
  - CORS origins now read from ALLOWED_ORIGINS env var (comma-separated)
    Fallback: http://localhost:3000,http://localhost:5173
  - Auth router already included; no changes needed there
  - All previous sessions' behaviour preserved

Run:
  uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load .env FIRST — before any other import
# ═══════════════════════════════════════════════════════════════════════════════
# dotenv must run before LangGraph imports so LangSmith tracing is active
# from the first graph compilation, not just from the first request.

from dotenv import load_dotenv
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: LangSmith observability setup
# ═══════════════════════════════════════════════════════════════════════════════
# Must be set in os.environ (not just .env file) before LangGraph is imported.
# LangGraph reads these at import time to configure the LangSmith callback handler.
#
# How it works:
#   - LANGCHAIN_TRACING_V2=true  -> enables automatic trace instrumentation
#   - LANGCHAIN_API_KEY          -> authenticates to LangSmith
#   - LANGCHAIN_PROJECT          -> groups traces under "lexshield-ai" project
#   - LANGCHAIN_ENDPOINT         -> LangSmith API endpoint (default is correct)
#
# Once active, EVERY graph.invoke() call creates a trace showing:
#   • classify_intent_node -> route_by_intent -> [node] -> END
#   • Each Groq LLM call with prompt, completion, token counts, latency
#   • ChromaDB retrieval with query vector and returned chunks
#   • Total wall-clock time per node and end-to-end

_LANGSMITH_KEYS = {
    "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "false"),
    "LANGCHAIN_API_KEY":    os.getenv("LANGCHAIN_API_KEY",    ""),
    "LANGCHAIN_PROJECT":    os.getenv("LANGCHAIN_PROJECT",    "lexshield-ai"),
    "LANGCHAIN_ENDPOINT":   os.getenv("LANGCHAIN_ENDPOINT",   "https://api.smith.langchain.com"),
}

for _key, _val in _LANGSMITH_KEYS.items():
    os.environ[_key] = _val

_tracing_enabled = _LANGSMITH_KEYS["LANGCHAIN_TRACING_V2"].lower() == "true"
_api_key_present = bool(_LANGSMITH_KEYS["LANGCHAIN_API_KEY"])

import logging

# ── NEW CODE: Configure the root logger ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# ───────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

if _tracing_enabled and _api_key_present:
    try:
        from langsmith import Client
        _ = Client()
        print(
            f"[LexShield] LangSmith tracing ENABLED — "
            f"project='{_LANGSMITH_KEYS['LANGCHAIN_PROJECT']}' | "
            f"dashboard: https://smith.langchain.com/projects/lexshield-ai"
        )
    except Exception as e:
        logger.warning(f"[LexShield] Failed to initialize LangSmith tracing: {e}")
elif _tracing_enabled and not _api_key_present:
    logger.warning(
        "[LexShield] WARNING: LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY "
        "is not set. Tracing will fail. Continuing without observability."
    )
else:
    print("[LexShield] LangSmith tracing DISABLED (set LANGCHAIN_TRACING_V2=true to enable)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Database tables
# ═══════════════════════════════════════════════════════════════════════════════
# NOTE: PostgreSQL table initialization (users, sessions) is handled by
# _init_auth_tables() inside api/auth.py, which runs automatically when that
# module is imported below (Step 5).  The legacy create_tables() call that
# pointed at models/database.py (SQLAlchemy / SQLite) has been removed.


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: FastAPI app
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title       = "LexShield AI",
    description = (
        "Agentic Indian Legal Intelligence Platform — "
        "Capstone project by Anantha Krishnan K, CS Graduate, "
        "Hansraj College, University of Delhi."
    ),
    version     = "1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Read allowed origins from .env: ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
# Falls back to both common dev ports if env var is absent.
_raw_origins   = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://localhost:5173,https://lexshield-ai-prod.web.app,https://lexshield.co.in,https://www.lexshield.co.in"
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _allowed_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

print(f"[LexShield] CORS allowed origins: {_allowed_origins}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Routers
# ═══════════════════════════════════════════════════════════════════════════════

from api.auth        import router as auth_router
from api.document    import router as document_router
from api.legal       import router as legal_router
# from api.orchestator import router as orchestrator_router
from api.classify    import router as classify_router
from api.master      import router as master_router

app.include_router(auth_router)
app.include_router(master_router)
app.include_router(classify_router)
app.include_router(document_router)
app.include_router(legal_router)
# app.include_router(orchestrator_router)


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    """
    Returns status of all core services.
    Checks: ChromaDB, LLM (Groq), Embedding model, LangSmith tracing.

    curl -s http://localhost:8000/health | python -m json.tool
    """
    logger.debug("Health check requested")
    status = {
        "service":  "LexShield AI",
        "version":  "1.0.0",
        "chromadb": "unknown",
        "llm":      "unknown",
        "embedder": "unknown",
        "tracing":  (
            f"enabled — project={_LANGSMITH_KEYS['LANGCHAIN_PROJECT']}"
            if (_tracing_enabled and _api_key_present)
            else "disabled"
        ),
        "cors_origins": _allowed_origins,
    }

    try:
        from rag.vectorstore import vectorstore
        count = vectorstore.count()
        status["chromadb"] = f"ok — {count} chunks indexed"
    except Exception as e:
        logger.exception("Health check failed for ChromaDB")
        status["chromadb"] = f"error: {e}"

    try:
        from rag.embedder import embedder
        _ = embedder.embed_single("test")
        status["embedder"] = f"ok — {embedder.model_name}"
    except Exception as e:
        logger.exception("Health check failed for Embedder")
        status["embedder"] = f"error: {e}"

    try:
        from rag.llm import llm
        _ = llm.generate("Reply with the single word: ok", max_tokens=5)
        status["llm"] = f"ok — {llm.model}"
    except Exception as e:
        logger.exception("Health check failed for LLM")
        status["llm"] = f"error: {e}"

    all_ok = all(
        "ok" in str(v)
        for k, v in status.items()
        if k not in ("service", "version", "tracing", "cors_origins")
    )
    status["overall"] = "healthy" if all_ok else "degraded"

    return status