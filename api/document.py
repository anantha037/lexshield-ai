"""
LexShield Document Intelligence Endpoint
=========================================
Accepts image or PDF uploads and returns extracted text.
(Pure OCR — no RAG. For document + RAG combined, use orchestrator.)
"""

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from cv.pipeline import (
    preprocess_image,
    extract_text_from_image,
    extract_text_from_pdf_bytes,
)

router = APIRouter(prefix="/api/v1/document", tags=["Document Intelligence"])

ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/bmp",
}


@router.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Accepts a PDF or image file.
    Returns extracted text and basic metadata.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: PDF, JPEG, PNG, TIFF, BMP",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        if file.content_type == "application/pdf":
            extracted_text = extract_text_from_pdf_bytes(file_bytes)
            file_type = "pdf"
        else:
            np_array = np.frombuffer(file_bytes, np.uint8)
            image    = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=422, detail="Could not decode image.")
            preprocessed   = preprocess_image(image)
            extracted_text = extract_text_from_image(preprocessed)
            file_type      = "image"

        return JSONResponse(content={
            "filename":        file.filename,
            "file_type":       file_type,
            "character_count": len(extracted_text),
            "extracted_text":  extracted_text,
            "success":         bool(extracted_text.strip()),
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")