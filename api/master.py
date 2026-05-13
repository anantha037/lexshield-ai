"""
LexShield AI — Master Orchestrator API
========================================
Returns LexShieldResponse structured output on every endpoint.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from agents.orchestrator import master_orchestrator
from agents.memory       import session_memory

router = APIRouter(prefix="/api/v1/master", tags=["Master Orchestrator"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query:      str
    session_id: Optional[str] = None

class DocumentRequest(BaseModel):
    text:       str
    session_id: Optional[str] = None
    filename:   Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RiskInfo(BaseModel):
    score:   float
    level:   str
    factors: List[str]

class CitationInfo(BaseModel):
    source_number:   int
    source:          str
    section:         str  = ""
    section_title:   str  = ""
    preview:         str  = ""
    relevance_score: Optional[float] = None
    era:             str  = ""

class StructuredResponse(BaseModel):
    # Answer
    answer_text:  str
    summary:      str
    key_clauses:  List[str]
    suggestions:  List[str]
    risk:         RiskInfo
    citations:    List[CitationInfo]
    draft:        str

    # Routing
    intent:       str
    session_id:   str
    confidence:   float
    mode:         str

    # RAG metadata
    sources_consulted: int
    synthesis_note:    str
    grounding_warning: str
    rewritten_queries: List[str]
    reranker_used:     bool


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/query", response_model=StructuredResponse)
def master_query(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    resp = master_orchestrator.handle_query(
        query      = request.query.strip(),
        session_id = request.session_id,
    )

    return StructuredResponse(
        answer_text       = resp.answer_text,
        summary           = resp.summary,
        key_clauses       = resp.key_clauses,
        suggestions       = resp.suggestions,
        risk              = RiskInfo(
            score   = resp.risk_score,
            level   = resp.risk_level,
            factors = resp.risk_factors,
        ),
        citations         = [
            CitationInfo(
                source_number   = c.source_number,
                source          = c.source,
                section         = c.section,
                section_title   = c.section_title,
                preview         = c.preview,
                relevance_score = c.relevance_score,
                era             = c.era,
            ) for c in resp.citations
        ],
        draft             = resp.draft,
        intent            = resp.intent,
        session_id        = resp.session_id,
        confidence        = resp.confidence,
        mode              = resp.mode,
        sources_consulted = resp.sources_consulted,
        synthesis_note    = resp.synthesis_note,
        grounding_warning = resp.grounding_warning,
        rewritten_queries = resp.rewritten_queries,
        reranker_used     = resp.reranker_used,
    )


@router.post("/document", response_model=StructuredResponse)
def master_document(request: DocumentRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    resp = master_orchestrator.handle_document(
        extracted_text = request.text.strip(),
        session_id     = request.session_id,
        filename       = request.filename,
    )

    return StructuredResponse(
        answer_text       = resp.answer_text,
        summary           = resp.summary,
        key_clauses       = resp.key_clauses,
        suggestions       = resp.suggestions,
        risk              = RiskInfo(
            score   = resp.risk_score,
            level   = resp.risk_level,
            factors = resp.risk_factors,
        ),
        citations         = [
            CitationInfo(
                source_number   = c.source_number,
                source          = c.source,
                section         = c.section,
                section_title   = c.section_title,
                preview         = c.preview,
                relevance_score = c.relevance_score,
                era             = c.era,
            ) for c in resp.citations
        ],
        draft             = resp.draft,
        intent            = resp.intent,
        session_id        = resp.session_id,
        confidence        = resp.confidence,
        mode              = resp.mode,
        sources_consulted = resp.sources_consulted,
        synthesis_note    = resp.synthesis_note,
        grounding_warning = resp.grounding_warning,
        rewritten_queries = resp.rewritten_queries,
        reranker_used     = resp.reranker_used,
    )


@router.get("/session/{session_id}/history")
def get_session_history(session_id: str):
    if not session_memory.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    history = session_memory.get_history(session_id)
    return {"session_id": session_id, "turn_count": len(history), "history": history}


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    deleted = session_memory.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"session_id": session_id, "deleted": True}