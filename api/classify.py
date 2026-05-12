# api/classify.py - UPDATED: accepts file uploads + text

"""
Classifier API Router — /api/v1/classify
Accepts: raw text, PDF upload, image upload, or Word doc upload
"""

import io
import tempfile
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["classifier"])

# Supported upload types
SUPPORTED_TYPES = {
    "application/pdf":                                                  "pdf",
    "image/jpeg":                                                       "image",
    "image/png":                                                        "image",
    "image/tiff":                                                       "image",
    "image/bmp":                                                        "image",
    "text/plain":                                                       "text",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf", ".jpg": "image", ".jpeg": "image",
    ".png": "image", ".tiff": "image", ".bmp": "image",
    ".txt": "text", ".docx": "docx",
}


# ── Schemas ───────────────────────────────────────────────────────────────────

class ClassifyTextRequest(BaseModel):
    text: str
    include_ner: bool = False
    use_llm_risk: bool = False


class ClassifyResponse(BaseModel):
    label:          int
    label_name:     str
    confidence:     float
    uncertain:      bool
    all_scores:     dict
    mode:           Optional[str]  = None
    warning:        Optional[str]  = None
    entities:       Optional[dict] = None
    risk:           Optional[dict] = None
    # File upload fields
    extracted_text: Optional[str]  = None
    file_type:      Optional[str]  = None
    char_count:     Optional[int]  = None


# ── Text extraction helpers ───────────────────────────────────────────────────

def _extract_from_pdf_bytes(data: bytes) -> str:
    """Try PyMuPDF first (digital PDF), fall back to OCR (scanned)."""
    text = ""
    try:
        import fitz   # PyMuPDF
        doc  = fitz.open(stream=data, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
        text = text.strip()
    except Exception:
        pass

    # If PyMuPDF got nothing meaningful, use OCR
    if len(text) < 100:
        from cv.pipeline import extract_text_from_pdf_bytes
        text = extract_text_from_pdf_bytes(data)

    return text


def _extract_from_image_bytes(data: bytes, filename: str) -> str:
    """Save to temp file, run CV pipeline."""
    import cv2
    import numpy as np
    from cv.pipeline import preprocess_image, extract_text_from_image

    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    preprocessed = preprocess_image(image)
    return extract_text_from_image(preprocessed)


def _extract_from_docx_bytes(data: bytes) -> str:
    """Extract text from Word document bytes."""
    try:
        import docx
        doc  = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="python-docx not installed. Run: pip install python-docx"
        )


def _extract_from_txt_bytes(data: bytes) -> str:
    """Decode plain text file."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _detect_file_type(filename: str, content_type: str) -> str:
    """Detect file type from extension or content-type."""
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]
    if content_type in SUPPORTED_TYPES:
        return SUPPORTED_TYPES[content_type]
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type: {ext or content_type}. "
               f"Supported: PDF, JPG, PNG, TIFF, BMP, TXT, DOCX"
    )


def _run_classification(
    text:         str,
    include_ner:  bool = False,
    use_llm_risk: bool = False,
) -> dict:
    """Core classification logic — shared by text and file endpoints."""
    if not text or len(text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Extracted text too short (< 20 chars). "
                   "Check if the document is scanned/empty."
        )

    from models.classifier import classifier
    result = classifier.predict(text)

    # NER
    entities = {}
    if include_ner:
        try:
            from nlp.ner_pipeline import run_ner
            entities          = run_ner(text)
            result["entities"] = entities
        except Exception as e:
            result["entities"] = {"warning": f"NER failed: {e}"}

    # Risk scoring
    try:
        from models.risk_scorer import risk_scorer
        risk           = risk_scorer.score(
            text     = text,
            doc_type = result.get("label_name", "unknown"),
            entities = entities,
            use_llm  = use_llm_risk,
        )
        result["risk"] = risk.to_dict()
    except Exception as e:
        result["risk"] = {"warning": f"Risk scoring failed: {e}"}

    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/classify", response_model=ClassifyResponse)
async def classify_text(req: ClassifyTextRequest):
    """
    Classify from raw text.
    
    Body:
        text (str): Document text
        include_ner (bool): Run NER (slower, default False)
        use_llm_risk (bool): Use Groq for risk scoring (slower, default False)
    """
    try:
        result = _run_classification(req.text, req.include_ner, req.use_llm_risk)
        result["char_count"] = len(req.text)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify/file", response_model=ClassifyResponse)
async def classify_file(
    file:         UploadFile = File(...),
    include_ner:  bool       = Form(False),
    use_llm_risk: bool       = Form(False),
    return_text:  bool       = Form(False),   # include extracted text in response
):
    """
    Classify from uploaded file.
    Accepts: PDF, JPG, PNG, TIFF, BMP, TXT, DOCX

    Form fields:
        file (required): The document file
        include_ner (bool): Run NER pipeline (default False)
        use_llm_risk (bool): Use LLM for risk assessment (default False)
        return_text (bool): Return extracted text in response (default False)

    Example curl:
        curl -X POST http://localhost:8000/api/v1/classify/file \\
          -F "file=@my_fir.pdf" \\
          -F "include_ner=true"
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read file bytes
    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Detect type
    file_type = _detect_file_type(file.filename, file.content_type or "")

    # Extract text
    try:
        if file_type == "pdf":
            text = _extract_from_pdf_bytes(data)
        elif file_type == "image":
            text = _extract_from_image_bytes(data, file.filename)
        elif file_type == "docx":
            text = _extract_from_docx_bytes(data)
        elif file_type == "text":
            text = _extract_from_txt_bytes(data)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction failed for {file_type}: {e}"
        )

    if not text or len(text.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract meaningful text from {file.filename}. "
                   f"File may be empty, password-protected, or heavily corrupted."
        )

    # Classify
    try:
        result            = _run_classification(text, include_ner, use_llm_risk)
        result["file_type"]  = file_type
        result["char_count"] = len(text)
        if return_text:
            result["extracted_text"] = text[:5000]   # cap at 5000 chars
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Utility endpoints ─────────────────────────────────────────────────────────

@router.get("/classify/categories")
async def get_categories():
    from models.classifier import CATEGORIES
    return {"categories": CATEGORIES, "count": len(CATEGORIES)}


@router.get("/classify/status")
async def classifier_status():
    try:
        from models.classifier import classifier
        return {"ready": classifier.is_ready(), "mode": classifier.get_mode()}
    except Exception as e:
        return {"ready": False, "error": str(e)}


@router.post("/classify/reload")
async def reload_classifier():
    try:
        from models.classifier import classifier
        success = classifier.reload()
        return {"reloaded": success, "mode": classifier.get_mode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))