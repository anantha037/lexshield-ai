"""
LexShield AI — Master Orchestrator API
========================================
Endpoints:
  POST   /api/v1/master/query               — text query, intent-routed
  POST   /api/v1/master/document            — pre-extracted document text
  GET    /api/v1/master/session/{id}/history — conversation history
  DELETE /api/v1/master/session/{id}         — clear session
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from agents.orchestrator import master_orchestrator
from agents.memory import session_memory

router = APIRouter(prefix="/api/v1/master", tags=["Master Orchestrator"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class DocumentRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    filename: Optional[str] = None

class OrchestratorResponse(BaseModel):
    session_id: str
    intent: str
    confidence: float
    answer: str
    sources_consulted: int = 0
    synthesis_note: str = ""
    grounding_warning: str = ""
    rewritten_queries: list = []
    reranker_used: bool = False
    mode: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/query", response_model=OrchestratorResponse)
def master_query(request: QueryRequest):
    """
    Main query endpoint. Classifies intent and routes to correct agent.

    Body:
      query:      user query text (required)
      session_id: existing session ID (optional — creates new if absent)
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    result = master_orchestrator.handle_query(
        query=request.query.strip(),
        session_id=request.session_id,
    )

    return OrchestratorResponse(
        session_id=result.session_id,
        intent=result.intent,
        confidence=result.confidence,
        answer=result.answer,
        sources_consulted=result.sources_consulted,
        synthesis_note=result.synthesis_note,
        grounding_warning=result.grounding_warning,
        rewritten_queries=result.rewritten_queries,
        reranker_used=result.reranker_used,
        mode=result.mode,
    )


@router.post("/document", response_model=OrchestratorResponse)
def master_document(request: DocumentRequest):
    """
    Document analysis endpoint. Accepts pre-extracted text.
    Routes as document_analysis intent through the orchestrator.

    Body:
      text:       extracted document text (required)
      session_id: existing session ID (optional)
      filename:   original filename for context (optional)
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    result = master_orchestrator.handle_document(
        extracted_text=request.text.strip(),
        session_id=request.session_id,
        filename=request.filename,
    )

    return OrchestratorResponse(
        session_id=result.session_id,
        intent=result.intent,
        confidence=result.confidence,
        answer=result.answer,
        sources_consulted=result.sources_consulted,
        synthesis_note=result.synthesis_note,
        grounding_warning=result.grounding_warning,
        rewritten_queries=result.rewritten_queries,
        reranker_used=result.reranker_used,
        mode=result.mode,
    )


@router.get("/session/{session_id}/history")
def get_session_history(session_id: str):
    """
    Return full conversation history for a session.
    """
    if not session_memory.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    history = session_memory.get_history(session_id)
    return {
        "session_id": session_id,
        "turn_count": len(history),
        "history": history,
    }


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """
    Delete a session and its full conversation history.
    """
    deleted = session_memory.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"session_id": session_id, "deleted": True}