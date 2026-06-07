"""
LexShield AI — Master Orchestrator API  (Session 6 — Final)
=============================================================
Returns LexShieldResponse structured output on every endpoint.

Endpoints
---------
POST   /api/v1/master/query              — JSON body { query, session_id, language? }
POST   /api/v1/master/document           — multipart/form-data file + session_id
GET    /api/v1/master/session/{sid}/history  — session turn history (all turns)
GET    /api/v1/master/session/new        -- BUG FIX 4: guaranteed fresh session_id
DELETE /api/v1/master/session/{sid}      -- delete session
GET    /api/v1/master/sessions           -- list sessions (?type=chat|document|draft|all)

Auth (optional)
---------------
All /query and /document endpoints accept an optional Authorization: Bearer <token>.
If a valid token is present, the session is linked to the user in SQLite.
If no token, anonymous session (existing behaviour unchanged).

GET /api/v1/master/sessions requires a valid Bearer token.
"""

import io
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

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

class CaseLawItem(BaseModel):
    title:    str = ""
    court:    str = ""
    date:     str = ""
    citation: str = ""
    headline: str = ""
    url:      str = ""
    summary:  str = ""

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
    case_law_results:  List[CaseLawItem] = []
    debug_scratchpad:  Optional[dict] = None
    citation_status:   str = "unverified"
    validation_status: str = "not_applicable"
    scope_status:      str = "in_scope"
    scope_message:     Optional[str] = None


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
            logger.exception("python-docx not installed")
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
        logger.exception("PDF extraction failed")
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
        logger.exception("Image OCR failed")
        raise HTTPException(status_code=500, detail=f"Image OCR failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_structured_response(resp, debug_scratchpad=None) -> StructuredResponse:
    """Map orchestrator LexShieldResponse -> API StructuredResponse."""
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
        citation_status   = getattr(resp, "citation_status", "unverified"),
        validation_status = getattr(resp, "validation_status", "not_applicable"),
        scope_status      = getattr(resp, "scope_status", "in_scope"),
        scope_message     = getattr(resp, "scope_message", None),
        sources_consulted = resp.sources_consulted,
        synthesis_note    = resp.synthesis_note,
        grounding_warning = resp.grounding_warning,
        rewritten_queries = resp.rewritten_queries,
        reranker_used     = resp.reranker_used,
        case_law_results  = [
            CaseLawItem(**c) for c in (resp.case_law_results or [])
        ],
        debug_scratchpad  = debug_scratchpad,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/query", response_model=StructuredResponse)
def master_query(
    request:      QueryRequest,
    req:          Request = None,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Answer a legal query via the multi-agent LangGraph orchestrator.
    Body: { query, session_id?, language?, run_rag? }

    session_id is returned in the response — pass it back on subsequent
    requests to continue the conversation.  State persists across restarts
    via the SqliteSaver checkpointer.

    If a valid Bearer token is included, the session is linked to the user.

    Append ?debug=true to include the agent scratchpad in the response.

    curl -s -X POST http://localhost:8000/api/v1/master/query?debug=true \\
      -H "Content-Type: application/json" \\
      -d '{"query":"What is Section 138 NI Act?","session_id":null}' | python -m json.tool
    """
    logger.info(f"Master query request: {request.query}, session_id: {request.session_id}")
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    resp = master_orchestrator.handle_query(
        query      = request.query.strip(),
        session_id = request.session_id,
        language   = request.language,
    )

    # Link session to user if authenticated
    if current_user:
        logger.debug(f"Linking session {resp.session_id} to user {current_user['id']}")
        session_memory.link_session_to_user(resp.session_id, current_user["id"])

    debug = req.query_params.get("debug", "").lower() == "true" if req else False
    scratchpad = resp.debug_scratchpad if debug else None
    return _build_structured_response(resp, debug_scratchpad=scratchpad)


@router.post("/document", response_model=StructuredResponse)
async def master_document(
    req:          Request,
    file:         UploadFile      = File(...),
    session_id:   Optional[str]   = Form(None),
    current_user: Optional[dict]  = Depends(get_optional_user),
):
    """
    Analyse an uploaded legal document via the multi-agent orchestrator.
    Accepts: PDF, DOCX, TXT, JPG, PNG, TIFF, BMP  (max 10 MB)
    Form fields: file (required), session_id (optional)

    Append ?debug=true to include the agent scratchpad in the response.

    curl -s -X POST http://localhost:8000/api/v1/master/document?debug=true \\
      -F "file=@rental_agreement.pdf" \\
      -F "session_id=" | python -m json.tool
    """
    logger.info(f"Master document request for file: {file.filename}, session_id: {session_id}")
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
        logger.debug(f"Linking session {resp.session_id} to user {current_user['id']}")
        session_memory.link_session_to_user(resp.session_id, current_user["id"])

    debug = req.query_params.get("debug", "").lower() == "true"
    scratchpad = resp.debug_scratchpad if debug else None
    return _build_structured_response(resp, debug_scratchpad=scratchpad)


@router.get("/session/{session_id}/history")
def get_session_history(session_id: str):
    """
    Return FULL conversation history for a session (all turns, no cap).
    Reads from SQLite turns table — persists across server restarts.
    Used by frontend to restore complete chat history when user reopens session.

    curl -s http://localhost:8000/api/v1/master/session/<session_id>/history | python -m json.tool
    """
    logger.info(f"Fetching session history for: {session_id}")
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
    logger.info(f"Deleting session: {session_id}")
    deleted = session_memory.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )
    return {"session_id": session_id, "deleted": True}


@router.get("/sessions", response_model=List[SessionSummary])
def get_user_sessions(request: Request, current_user: dict = Depends(get_current_user)):
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
    logger.info(f"Fetching user sessions for user: {current_user['id']}")
    sessions = session_memory.get_user_sessions(current_user["id"])

    # BUG FIX 5: filter by ?type= query parameter
    session_type = request.query_params.get("type", "all")
    if session_type == "chat":
        sessions = [
            s for s in sessions
            if not (s.get('first_message') or '').startswith('Document:')
            and not (s.get('first_message') or '').lower().startswith('draft')
            and 'drafting' not in (s.get('first_message') or '').lower()
            and 'i need help drafting' not in (s.get('first_message') or '').lower()
        ]
    elif session_type == "document":
        sessions = [s for s in sessions if (s.get('first_message') or '').startswith('Document:')]
    elif session_type == "draft":
        sessions = [
            s for s in sessions
            if (s.get('first_message') or '').lower().startswith('draft')
            or 'drafting' in (s.get('first_message') or '').lower()
            or 'complaint' in (s.get('first_message') or '').lower()
            or (s.get('first_message') or '').startswith('I need help drafting')
        ]

    # Add session_type field to each dict (BUG FIX 5)
    def _infer_type(fm: str) -> str:
        if fm.startswith('Document:'):
            return 'document'
        fm_l = fm.lower()
        if fm_l.startswith('draft') or 'drafting' in fm_l or 'complaint' in fm_l or fm_l.startswith('i need help drafting'):
            return 'draft'
        return 'chat'

    for s in sessions:
        s['session_type'] = _infer_type(s.get('first_message') or '')

    return [SessionSummary(**{k: v for k, v in s.items() if k in SessionSummary.__fields__}) for s in sessions]


@router.get("/session/new")
def new_session_id():
    """
    BUG FIX 4: Return a guaranteed-fresh session_id.
    Frontend calls this before each 'Ask about this' click from RightsView
    to guarantee the resulting chat starts in a new session, not an existing one.

    Returns: {"session_id": "LX-XXXXXXXX"}
    """
    fresh_id = "LX-" + uuid.uuid4().hex[:8].upper()
    logger.info(f"Generated new session id: {fresh_id}")
    return {"session_id": fresh_id}