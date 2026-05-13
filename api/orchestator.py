"""
LexShield AI — Legacy Orchestrator Router
==========================================
Kept for backward compatibility.
Both endpoints now delegate to MasterOrchestrator (Week 3).

POST /api/v1/orchestrate/query    → master_orchestrator.handle_query()
POST /api/v1/orchestrate/document → master_orchestrator.handle_document()
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/v1", tags=["Orchestrator (Legacy)"])

ALLOWED_DOC_TYPES = {
    "application/pdf",
    "image/jpeg", "image/jpg",
    "image/png", "image/tiff", "image/bmp",
}


# ── Request model ─────────────────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    query:      str
    session_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/orchestrate/query")
async def orchestrate_text_query(request: TextQueryRequest):
    """
    Legacy text query endpoint.
    Delegates to MasterOrchestrator — returns full LexShieldResponse.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    from agents.orchestrator import master_orchestrator

    resp = master_orchestrator.handle_query(
        query      = request.query.strip(),
        session_id = request.session_id,
    )
    return resp.to_dict()


@router.post("/orchestrate/document")
async def orchestrate_document_upload(
    file:       UploadFile = File(...),
    question:   str        = Form(default=""),
    session_id: str        = Form(default=""),
):
    """
    Legacy document upload endpoint.
    Extracts text via CV pipeline then delegates to MasterOrchestrator.
    """
    if file.content_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, JPEG, PNG, TIFF, BMP",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # CV extraction
    try:
        from cv.pipeline import extract_text_from_pdf_bytes, extract_text_from_image, preprocess_image
        import cv2, numpy as np

        if file.content_type == "application/pdf":
            extracted_text = extract_text_from_pdf_bytes(file_bytes)
        else:
            np_array       = np.frombuffer(file_bytes, np.uint8)
            image          = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image.")
            extracted_text = extract_text_from_image(preprocess_image(image))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {e}")

    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from this file.")

    # Build query
    doc_text = extracted_text[:3000]
    if question.strip():
        full_query = f"Document text:\n{doc_text}\n\nQuestion: {question.strip()}"
    else:
        full_query = f"Analyze and summarize this legal document:\n{doc_text}"

    from agents.orchestrator import master_orchestrator

    resp = master_orchestrator.handle_document(
        extracted_text = full_query,
        session_id     = session_id or None,
        filename       = file.filename,
    )
    return resp.to_dict()