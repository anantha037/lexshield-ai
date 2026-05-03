"""
LexShield AI — Document Analysis API  (Week 2, Day 4 update)
=============================================================
POST /api/v1/document/analyze

Week 1: returned OCR text only
Week 2 Day 4: returns OCR text + structured NER entities

Request: multipart/form-data with file field
Response (structured JSON):
  {
    "filename":    "rental_agreement.pdf",
    "text":        "THIS RENTAL AGREEMENT...",
    "word_count":  342,
    "entities": {
      "persons":       ["Rajesh Kumar", "Priya Sharma"],
      "organizations": ["Acme Enterprises Pvt Ltd"],
      "dates":         ["1st January 2024", "31st December 2024"],
      "locations":     ["Thiruvananthapuram", "Kerala"],
      "monetary":      ["₹15,000", "Rs. 5,000"],
      "ipc_sections":  [],
      "case_numbers":  [],
      "acts":          ["Kerala Buildings (Lease and Rent Control) Act"],
      "entity_counts": { "persons": 2, ... }
    },
    "ocr_used":    false,
    "page_count":  2
  }
"""

import os
import io
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/document", tags=["document"])

# Supported file types
SUPPORTED_TYPES = {
    "application/pdf":  ".pdf",
    "image/jpeg":       ".jpg",
    "image/jpg":        ".jpg",
    "image/png":        ".png",
    "image/tiff":       ".tiff",
    "image/bmp":        ".bmp",
    "text/plain":       ".txt",
}

MAX_FILE_SIZE_MB = 10


# ── Response models ───────────────────────────────────────────────────────────

class EntityCounts(BaseModel):
    persons:       int = 0
    organizations: int = 0
    dates:         int = 0
    locations:     int = 0
    monetary:      int = 0
    ipc_sections:  int = 0
    case_numbers:  int = 0
    acts:          int = 0


class EntitiesModel(BaseModel):
    persons:       list[str] = []
    organizations: list[str] = []
    dates:         list[str] = []
    locations:     list[str] = []
    monetary:      list[str] = []
    ipc_sections:  list[str] = []
    case_numbers:  list[str] = []
    acts:          list[str] = []
    entity_counts: EntityCounts = EntityCounts()


class DocumentAnalysisResponse(BaseModel):
    filename:   str
    text:       str
    word_count: int
    entities:   EntitiesModel
    ocr_used:   bool
    page_count: int
    warning:    Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int, bool]:
    """
    Extract text from PDF.
    Tries PyMuPDF first (fast, preserves text layer).
    Falls back to OCR if text layer is empty or minimal.
    Returns (text, page_count, ocr_used).
    """
    try:
        import fitz
        doc       = fitz.open(stream=file_bytes, filetype="pdf")
        pages     = [page.get_text("text") for page in doc]
        page_count = len(pages)
        text      = "\n".join(pages)
        doc.close()

        # If text layer has content, use it
        if len(text.strip().split()) > 20:
            return text, page_count, False

        # Fallback to OCR for scanned PDFs
        return _ocr_pdf(file_bytes, page_count), page_count, True

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")


def _ocr_pdf(file_bytes: bytes, page_count: int) -> str:
    """OCR a scanned PDF using pdf2image + cv/pipeline."""
    try:
        from pdf2image import convert_from_bytes
        from cv.pipeline import preprocess_image, extract_text_from_image
        import numpy as np

        images = convert_from_bytes(file_bytes, dpi=200)
        texts  = []
        for img in images[:20]:   # cap at 20 pages for RAM safety
            img_array = np.array(img)
            processed = preprocess_image(img_array)
            texts.append(extract_text_from_image(processed))
        return "\n".join(texts)
    except Exception as e:
        return f"[OCR failed: {e}]"


def _extract_text_from_image(file_bytes: bytes, suffix: str) -> tuple[str, int, bool]:
    """OCR a single image file."""
    try:
        from cv.pipeline import preprocess_image, extract_text_from_image
        import numpy as np
        from PIL import Image

        img       = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(img)
        processed = preprocess_image(img_array)
        text      = extract_text_from_image(processed)
        return text, 1, True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image OCR failed: {e}")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze an uploaded legal document.

    Pipeline:
      1. Detect file type
      2. Extract text (PyMuPDF for digital PDFs, OCR for scanned/images)
      3. Run NER pipeline (spaCy + regex, OpenNyAI if available)
      4. Return structured response with text + entities

    Supported formats: PDF, JPEG, PNG, TIFF, BMP, TXT
    Max file size: 10 MB
    """
    # File size check
    file_bytes = await file.read()
    size_mb    = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )

    filename     = file.filename or "uploaded_file"
    content_type = file.content_type or ""
    suffix       = Path(filename).suffix.lower()
    warning      = None

    # ── Extract text ──────────────────────────────────────────────────────────
    if suffix == ".pdf" or content_type == "application/pdf":
        text, page_count, ocr_used = _extract_text_from_pdf(file_bytes)

    elif suffix == ".txt" or content_type == "text/plain":
        text       = file_bytes.decode("utf-8", errors="replace")
        page_count = 1
        ocr_used   = False

    elif suffix in (".jpg", ".jpeg", ".png", ".tiff", ".bmp") or content_type.startswith("image/"):
        text, page_count, ocr_used = _extract_text_from_image(file_bytes, suffix)

    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix or content_type}. "
                   f"Supported: PDF, JPEG, PNG, TIFF, BMP, TXT",
        )

    if not text or len(text.strip()) < 10:
        warning = "Very little text extracted. Document may be blank or heavily image-based."
        text    = text or ""

    # ── NER ───────────────────────────────────────────────────────────────────
    from nlp.ner_pipeline import extract_entities
    entity_result = extract_entities(text)
    entity_dict   = entity_result.to_dict()

    # Build pydantic model
    entities = EntitiesModel(
        persons       = entity_dict.get("persons",       []),
        organizations = entity_dict.get("organizations", []),
        dates         = entity_dict.get("dates",         []),
        locations     = entity_dict.get("locations",     []),
        monetary      = entity_dict.get("monetary",      []),
        ipc_sections  = entity_dict.get("ipc_sections",  []),
        case_numbers  = entity_dict.get("case_numbers",  []),
        acts          = entity_dict.get("acts",          []),
        entity_counts = EntityCounts(**entity_dict.get("entity_counts", {})),
    )

    return DocumentAnalysisResponse(
        filename   = filename,
        text       = text[:5000],   # cap text in response to 5000 chars
        word_count = len(text.split()),
        entities   = entities,
        ocr_used   = ocr_used,
        page_count = page_count,
        warning    = warning,
    )