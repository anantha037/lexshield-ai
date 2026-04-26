"""
LexShield Orchestrator
=======================
Routes incoming requests to the correct pipeline.
 
Two flows:
  1. text_query    → RAG pipeline directly
  2. document_upload → CV pipeline (OCR) → RAG pipeline on extracted text
 
This is intentionally simple in Week 1.
Week 2 replaces this with a LangGraph multi-agent orchestrator.
 
Endpoint: POST /api/v1/orchestrate
"""
 
import tempfile
import os
from pathlib import Path
 
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
 
router = APIRouter(prefix="/api/v1", tags=["Orchestrator"])
 
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/tiff", "image/bmp"}
ALLOWED_DOC_TYPES   = {"application/pdf"} | ALLOWED_IMAGE_TYPES
 
 
# ── Request / Response models ─────────────────────────────────────────────────
 
class TextQueryRequest(BaseModel):
    query:           str
    doc_type_filter: Optional[str] = None  # 'statute' | 'judgment' | None
 
 
class OrchestratorResponse(BaseModel):
    flow:            str            # 'text_query' | 'document_upload'
    query_used:      str            # the actual query sent to RAG
    answer:          str
    citations:       list[dict]
    context_used:    bool
    extracted_text:  Optional[str] = None  # only populated for document flow
    warning:         Optional[str] = None
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _run_rag(query: str, doc_type_filter: Optional[str] = None) -> dict:
    """Run the RAG pipeline and return a plain dict."""
    from rag.pipeline import rag_pipeline
    result = rag_pipeline.answer(query, doc_type_filter=doc_type_filter)
    return {
        "answer":      result.answer,
        "citations":   result.citations,
        "context_used": result.context_used,
        "warning":     result.warning,
    }
 
 
def _extract_text_from_upload(file_bytes: bytes, content_type: str) -> str:
    """
    Routes uploaded file bytes through the CV pipeline.
    Returns extracted text string.
    """
    import cv2
    import numpy as np
    from cv.pipeline import preprocess_image, extract_text_from_image, extract_text_from_pdf_bytes
 
    if content_type == "application/pdf":
        return extract_text_from_pdf_bytes(file_bytes)
 
    # Image
    np_array = np.frombuffer(file_bytes, np.uint8)
    image    = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image — file may be corrupted.")
    preprocessed = preprocess_image(image)
    return extract_text_from_image(preprocessed)
 
 
# ── Endpoints ─────────────────────────────────────────────────────────────────
 
@router.post("/orchestrate/query", response_model=OrchestratorResponse)
async def orchestrate_text_query(request: TextQueryRequest):
    """
    Flow 1: Text query → RAG pipeline → cited answer.
 
    Example:
      POST /api/v1/orchestrate/query
      {"query": "What are my rights if my landlord won't return my deposit?"}
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
 
    rag_result = _run_rag(request.query, request.doc_type_filter)
 
    return OrchestratorResponse(
        flow         = "text_query",
        query_used   = request.query,
        **rag_result,
    )
 
 
@router.post("/orchestrate/document", response_model=OrchestratorResponse)
async def orchestrate_document_upload(
    file:     UploadFile = File(...),
    question: str        = Form(
        default="",
        description="Optional: specific question about this document. "
                    "If empty, LexShield will analyse the document and summarise key legal points."
    ),
):
    """
    Flow 2: Document upload → OCR → RAG pipeline.
 
    Accepts a PDF or image. Extracts text using the CV pipeline.
    Then either:
      - Answers the user's specific question about the document, OR
      - Summarises the document's key legal points (if no question given)
 
    Example:
      POST /api/v1/orchestrate/document
      file: rental_agreement.jpg
      question: "Does this agreement violate the Kerala Rent Control Act?"
    """
    if file.content_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: PDF, JPEG, PNG, TIFF, BMP"
        )
 
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
 
    # ── Step 1: Extract text via CV pipeline ──────────────────────────────────
    try:
        extracted_text = _extract_text_from_upload(file_bytes, file.content_type)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {e}")
 
    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from this file. "
                   "If it is a scanned image, ensure it is clear and high-resolution."
        )
 
    # ── Step 2: Build RAG query ───────────────────────────────────────────────
    if question.strip():
        # User has a specific question about the document
        rag_query = (
            f"The following is the text of a legal document:\n\n"
            f"{extracted_text[:3000]}\n\n"   # cap at 3000 chars to stay within context
            f"Based on this document, answer the following question: {question}"
        )
    else:
        # No question — summarise legal points
        rag_query = (
            f"The following is the text of a legal document:\n\n"
            f"{extracted_text[:3000]}\n\n"
            f"Identify and explain the key legal clauses, rights, and obligations "
            f"mentioned in this document. Flag any clauses that may be unfair or "
            f"illegal under Indian law."
        )
 
    # ── Step 3: Run RAG ───────────────────────────────────────────────────────
    rag_result = _run_rag(rag_query)
 
    return OrchestratorResponse(
        flow           = "document_upload",
        query_used     = question.strip() or "Document analysis — key legal points",
        extracted_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
        **rag_result,
    )