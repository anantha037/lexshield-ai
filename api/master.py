"""
LexShield AI — Master Orchestrator API  (Session 6 — Final)
=============================================================
Returns LexShieldResponse structured output on every endpoint.

Endpoints
---------
POST   /api/v1/master/query              — JSON body { query, session_id, language? }
POST   /api/v1/master/document           — multipart/form-data file + session_id
GET    /api/v1/master/session/{sid}/history  — session turn history (all turns)
DELETE /api/v1/master/session/{sid}      — delete session
GET    /api/v1/master/sessions           — NEW: list sessions for authenticated user

Auth (optional)
---------------
All /query and /document endpoints accept an optional Authorization: Bearer <token>.
If a valid token is present, the session is linked to the user in SQLite.
If no token, anonymous session (existing behaviour unchanged).

GET /api/v1/master/sessions requires a valid Bearer token.
"""

import io
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from pydantic import BaseModel

from agents.orchestrator import master_orchestrator
from agents.memory       import session_memory
from api.auth            import get_current_user, get_optional_user

router = APIRouter(prefix="/api/v1/master", tags=["Master Orchestrator"])

MAX_FILE_SIZE_MB   = 10
SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".txt", ".docx"}


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query:      str
    session_id: Optional[str] = None
    language:   Optional[str] = None   # e.g. "ml", "hi" — passed for multilingual routing
    run_rag:    Optional[bool] = True   # accepted but handled by graph node selection


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
    section:         str             = ""
    section_title:   str             = ""
    preview:         str             = ""
    relevance_score: Optional[float] = None
    era:             str             = ""

class StructuredResponse(BaseModel):
    answer_text:       str
    summary:           str
    key_clauses:       List[str]
    suggestions:       List[str]
    risk:              RiskInfo
    citations:         List[CitationInfo]
    draft:             str
    intent:            str
    session_id:        str
    confidence:        float
    mode:              str
    sources_consulted: int
    synthesis_note:    str
    grounding_warning: str
    rewritten_queries: List[str]
    reranker_used:     bool


class SessionSummary(BaseModel):
    session_id:    str
    created_at:    float
    last_active:   float
    turn_count:    int
    first_message: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_bytes)
    elif suffix == ".txt":
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="ignore")
    elif suffix == ".docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise HTTPException(status_code=500, detail="python-docx not installed")
    elif suffix in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        return _extract_image(file_bytes)
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix}. "
                   f"Supported: PDF, DOCX, TXT, JPG, PNG, TIFF, BMP",
        )


def _extract_pdf(file_bytes: bytes) -> str:
    """PyMuPDF first (fast); falls back to CV OCR for scanned PDFs."""
    try:
        import fitz
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > 100:
            return text
    except Exception:
        pass
    try:
        from cv.pipeline import extract_text_from_pdf_bytes
        return extract_text_from_pdf_bytes(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {exc}")


def _extract_image(file_bytes: bytes) -> str:
    try:
        import cv2
        import numpy as np
        from cv.pipeline import preprocess_image, extract_text_from_image
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        return extract_text_from_image(preprocess_image(image))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image OCR failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_structured_response(resp) -> StructuredResponse:
    """Map orchestrator LexShieldResponse → API StructuredResponse."""
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
            )
            for c in resp.citations
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


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/query", response_model=StructuredResponse)
def master_query(
    request:      QueryRequest,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Answer a legal query via the multi-agent LangGraph orchestrator.
    Body: { query, session_id?, language?, run_rag? }

    session_id is returned in the response — pass it back on subsequent
    requests to continue the conversation.  State persists across restarts
    via the SqliteSaver checkpointer.

    If a valid Bearer token is included, the session is linked to the user.

    curl -s -X POST http://localhost:8000/api/v1/master/query \\
      -H "Content-Type: application/json" \\
      -d '{"query":"What is Section 138 NI Act?","session_id":null}' | python -m json.tool
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    resp = master_orchestrator.handle_query(
        query      = request.query.strip(),
        session_id = request.session_id,
    )

    # Link session to user if authenticated
    if current_user:
        session_memory.link_session_to_user(resp.session_id, current_user["id"])

    return _build_structured_response(resp)


@router.post("/document", response_model=StructuredResponse)
async def master_document(
    file:         UploadFile      = File(...),
    session_id:   Optional[str]   = Form(None),
    current_user: Optional[dict]  = Depends(get_optional_user),
):
    """
    Analyse an uploaded legal document via the multi-agent orchestrator.
    Accepts: PDF, DOCX, TXT, JPG, PNG, TIFF, BMP  (max 10 MB)
    Form fields: file (required), session_id (optional)

    curl -s -X POST http://localhost:8000/api/v1/master/document \\
      -F "file=@rental_agreement.pdf" \\
      -F "session_id=" | python -m json.tool
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_bytes = await file.read()
    size_mb    = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )

    extracted_text = _extract_text(file_bytes, file.filename)

    if not extracted_text or len(extracted_text.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract meaningful text from the document. "
                "File may be blank, password-protected, or heavily scanned."
            ),
        )

    resp = master_orchestrator.handle_document(
        extracted_text = extracted_text.strip(),
        session_id     = session_id,
        filename       = file.filename,
    )

    # Link session to user if authenticated
    if current_user:
        session_memory.link_session_to_user(resp.session_id, current_user["id"])

    return _build_structured_response(resp)


@router.get("/session/{session_id}/history")
def get_session_history(session_id: str):
    """
    Return FULL conversation history for a session (all turns, no cap).
    Reads from SQLite turns table — persists across server restarts.
    Used by frontend to restore complete chat history when user reopens session.

    curl -s http://localhost:8000/api/v1/master/session/<session_id>/history | python -m json.tool
    """
    if not session_memory.session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )
    history = session_memory.get_history(session_id)
    return {
        "session_id": session_id,
        "turn_count": len(history),
        "history":    history,
    }


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """
    Delete a session and all its turns from SQLite.
    Also clears any LangGraph checkpoint state for this thread_id.

    curl -s -X DELETE http://localhost:8000/api/v1/master/session/<session_id>
    """
    deleted = session_memory.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )
    return {"session_id": session_id, "deleted": True}


@router.get("/sessions", response_model=List[SessionSummary])
def get_user_sessions(current_user: dict = Depends(get_current_user)):
    """
    Return all chat sessions belonging to the authenticated user.
    Ordered by most-recently-active first.
    Requires: Authorization: Bearer <token>

    Each item includes:
      - session_id
      - created_at (Unix timestamp)
      - last_active (Unix timestamp of most recent turn)
      - turn_count
      - first_message (first user message, truncated to 60 chars)

    curl -s http://localhost:8000/api/v1/master/sessions \\
      -H "Authorization: Bearer <your_token>" | python -m json.tool
    """
    sessions = session_memory.get_user_sessions(current_user["id"])
    return [SessionSummary(**s) for s in sessions]