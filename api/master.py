"""
LexShield AI — Master Orchestrator API
========================================
Returns LexShieldResponse structured output on every endpoint.

/query    — JSON body { query, session_id }
/document — multipart/form-data file upload (PDF, DOCX, TXT, images)
"""

import io
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from agents.orchestrator import master_orchestrator
from agents.memory       import session_memory

router = APIRouter(prefix="/api/v1/master", tags=["Master Orchestrator"])

MAX_FILE_SIZE_MB  = 10
SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".txt", ".docx"}


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query:      str
    session_id: Optional[str] = None
    run_rag:    Optional[bool] = True   # accepted but handled by orchestrator


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
    section:         str            = ""
    section_title:   str            = ""
    preview:         str            = ""
    relevance_score: Optional[float] = None
    era:             str            = ""

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


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from uploaded file based on extension."""
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
            detail=f"Unsupported file type: {suffix}. Supported: PDF, DOCX, TXT, JPG, PNG"
        )


def _extract_pdf(file_bytes: bytes) -> str:
    """Try PyMuPDF first, fall back to OCR for scanned PDFs."""
    try:
        import fitz
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > 100:
            return text
    except Exception:
        pass
    # Fallback to CV OCR pipeline
    try:
        from cv.pipeline import extract_text_from_pdf_bytes
        return extract_text_from_pdf_bytes(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image OCR failed: {e}")


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


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/query", response_model=StructuredResponse)
def master_query(request: QueryRequest):
    """
    Answer a legal query via the multi-agent orchestrator.
    Body: { query, session_id?, run_rag? }
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    resp = master_orchestrator.handle_query(
        query      = request.query.strip(),
        session_id = request.session_id,
    )
    return _build_structured_response(resp)


@router.post("/document", response_model=StructuredResponse)
async def master_document(
    file:       UploadFile      = File(...),
    session_id: Optional[str]   = Form(None),
):
    """
    Analyse an uploaded legal document via the multi-agent orchestrator.
    Accepts: PDF, DOCX, TXT, JPG, PNG, TIFF, BMP (max 10 MB)
    Form fields: file (required), session_id (optional)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_bytes = await file.read()

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB"
        )

    # Extract text from the uploaded file
    extracted_text = _extract_text(file_bytes, file.filename)

    if not extracted_text or len(extracted_text.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Could not extract meaningful text from the document. "
                   "File may be blank, password-protected, or heavily scanned."
        )

    resp = master_orchestrator.handle_document(
        extracted_text = extracted_text.strip(),
        session_id     = session_id,
        filename       = file.filename,
    )
    return _build_structured_response(resp)


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