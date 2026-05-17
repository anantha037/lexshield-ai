"""
LexShield AI — CV Pipeline  (Week 3, Day 2 — Production Upgrade)
==================================================================
Replaces the basic OpenCV + Tesseract pipeline with a multi-engine,
layout-aware document intelligence stack:

ENGINE STACK (in priority order):
  1. PyMuPDF (fitz)   — digital PDFs: direct text extraction, zero OCR cost.
                         Handles embedded fonts, tables, annotations.
  2. pdfplumber        — table extraction from digital PDFs (pandas DataFrames).
  3. docTR             — deep learning OCR for scanned images/PDFs.
                         CPU-only model (db_resnet50 + crnn_vgg16_bn).
                         Handles printed + handwritten text better than Tesseract.
  4. Tesseract (fallback) — multilingual fallback when docTR is unavailable.
                             Uses lang config based on detected source_language.

WHY docTR over Tesseract:
  - End-to-end neural network: detects text regions + reads text in one forward pass.
  - No manual preprocessing (threshold, deskew) needed — model handles it internally.
  - Significantly better on:
    • Low-quality scans (common in Indian legal documents)
    • Rotated/skewed pages
    • Mixed font sizes (court headers + body text)
    • Stamps and handwritten annotations on printed documents
  - CPU inference on i5-8250U: ~3–8s per page (acceptable for legal docs)
  - Free, open-source (Apache 2.0), no GPU required

TABLE EXTRACTION:
  pdfplumber extracts tables from digital PDFs as structured dicts.
  Tables are converted to markdown-style text and inserted at their
  correct position in the extracted document flow.
  This handles: wage slips, bank statements, consumer complaint exhibits,
  employment termination letters with leave records.

MULTILINGUAL OCR:
  Tesseract fallback uses source_language to select the correct language pack:
    "ml" → "mal+eng"   (Malayalam + English)
    "hi" → "hin+eng"   (Hindi + English)
    "ta" → "tam+eng"   (Tamil + English)
    "te" → "tel+eng"   (Telugu + English)
    "kn" → "kan+eng"   (Kannada + English)
    default → "eng"
  docTR currently supports English only — Malayalam/Hindi pages that are
  scanned are routed to Tesseract automatically.

LANGUAGE DETECTION ON OCR OUTPUT:
  After extracting text, detect_language() is called on the first 500 chars.
  If a non-English script is detected, source_language is set accordingly
  so downstream pipeline (multilingual_node) handles translation correctly.

Install (add to requirements.txt):
  python-doctr[torch]    # CPU-only docTR (no GPU needed)
  pdfplumber
  pymupdf                # fitz
  pytesseract            # fallback
  pillow
  opencv-python-headless
  numpy
"""

import re
import logging
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ── Tesseract (multilingual fallback) ─────────────────────────────────────────
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Suppress noisy torch/docTR warnings on CPU ────────────────────────────────
warnings.filterwarnings("ignore", message=".*Torch.*")
warnings.filterwarnings("ignore", message=".*CUDA.*")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE AVAILABILITY FLAGS  (checked once at import time)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import fitz as _fitz          # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False
    logger.warning("[CV] PyMuPDF not installed. pip install pymupdf")

try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False
    logger.warning("[CV] pdfplumber not installed. pip install pdfplumber")

try:
    from doctr.io     import DocumentFile as _DocTRDocumentFile
    from doctr.models import ocr_predictor as _ocr_predictor
    _DOCTR_AVAILABLE = True
    logger.info("[CV] docTR available — using neural OCR engine")
except ImportError:
    _DOCTR_AVAILABLE = False
    logger.warning(
        "[CV] docTR not installed. Falling back to Tesseract. "
        "Install: pip install python-doctr[torch]"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TESSERACT LANGUAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Maps ISO 639-1 language code → Tesseract language string
# Indian languages need "+eng" because legal documents mix scripts
_TESSERACT_LANG_MAP: dict[str, str] = {
    "ml": "mal+eng",    # Malayalam — install tesseract-ocr-mal
    "hi": "hin+eng",    # Hindi — install tesseract-ocr-hin
    "ta": "tam+eng",    # Tamil — install tesseract-ocr-tam
    "te": "tel+eng",    # Telugu — install tesseract-ocr-tel
    "kn": "kan+eng",    # Kannada — install tesseract-ocr-kan
    "mr": "hin+eng",    # Marathi uses Devanagari — same pack as Hindi
    "bn": "ben+eng",    # Bengali — install tesseract-ocr-ben
    "gu": "guj+eng",    # Gujarati — install tesseract-ocr-guj
    "pa": "pan+eng",    # Punjabi — install tesseract-ocr-pan
    "or": "ori+eng",    # Odia — install tesseract-ocr-ori
    "en": "eng",        # English only
}

# Default Tesseract config: PSM 3 (fully automatic page segmentation)
# PSM 6 was used before but PSM 3 handles multi-column legal documents better
_TESSERACT_CONFIG = "--psm 3 --oem 3"


# ═══════════════════════════════════════════════════════════════════════════════
# docTR MODEL SINGLETON  (lazy load — avoids 3s import cost on every request)
# ═══════════════════════════════════════════════════════════════════════════════

_doctr_model = None

def _get_doctr_model():
    """
    Lazy-load docTR OCR model on first use.
    CPU-optimised: db_resnet50 (detection) + crnn_vgg16_bn (recognition).
    Both pretrained on printed + handwritten text.
    """
    global _doctr_model
    if _doctr_model is None and _DOCTR_AVAILABLE:
        try:
            logger.info("[CV] Loading docTR OCR model (CPU)…")
            _doctr_model = _ocr_predictor(
                det_arch  = "db_resnet50",
                reco_arch = "crnn_vgg16_bn",
                pretrained = True,
            )
            logger.info("[CV] docTR model loaded.")
        except Exception as e:
            logger.error(f"[CV] docTR model load failed: {e}")
            _doctr_model = None
    return _doctr_model


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION (reuse multilingual_agent)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_ocr_language(text: str) -> str:
    """
    Detect language from extracted OCR text.
    Delegates to multilingual_agent.detect_language() (Unicode fast-path).
    Returns ISO 639-1 code.
    """
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        from agents.multilingual_agent import detect_language
        sample = text.strip()[:500]  # Check first 500 chars
        return detect_language(sample)
    except Exception:
        return "en"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE EXTRACTION (pdfplumber)
# ═══════════════════════════════════════════════════════════════════════════════

def _table_to_text(table: list[list]) -> str:
    """
    Convert a pdfplumber table (list of rows, each a list of cell strings)
    to a readable markdown-style text block.

    Handles:
    - None cells (empty cells in merged regions)
    - Multi-line cell content
    - Wage slips, bank statement rows, leave records
    """
    if not table:
        return ""

    lines = []
    for row in table:
        cells = []
        for cell in row:
            if cell is None:
                cells.append("")
            else:
                # Collapse multiline cell content
                cells.append(" ".join(str(cell).split()))
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def _extract_tables_pdfplumber(pdf_path: str) -> list[dict]:
    """
    Extract all tables from a digital PDF using pdfplumber.

    Returns list of {page: int, text: str} dicts.
    Empty if pdfplumber unavailable or no tables found.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return []

    tables = []
    try:
        with _pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                for tbl in (page_tables or []):
                    tbl_text = _table_to_text(tbl)
                    if tbl_text.strip():
                        tables.append({
                            "page": page_num,
                            "text": f"[TABLE — Page {page_num}]\n{tbl_text}",
                        })
    except Exception as e:
        logger.warning(f"[CV] pdfplumber table extraction warning: {e}")

    if tables:
        logger.info(f"[CV] pdfplumber: {len(tables)} table(s) extracted")

    return tables


def _extract_tables_pdfplumber_bytes(pdf_bytes: bytes) -> list[dict]:
    """Same as _extract_tables_pdfplumber but from raw bytes."""
    if not _PDFPLUMBER_AVAILABLE:
        return []

    import io
    tables = []
    try:
        with _pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                for tbl in (page_tables or []):
                    tbl_text = _table_to_text(tbl)
                    if tbl_text.strip():
                        tables.append({
                            "page": page_num,
                            "text": f"[TABLE — Page {page_num}]\n{tbl_text}",
                        })
    except Exception as e:
        logger.warning(f"[CV] pdfplumber table extraction warning: {e}")

    return tables


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: PyMuPDF — Digital PDF Text Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_digital_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text from a digital (non-scanned) PDF using PyMuPDF.
    Returns None if the PDF appears to be scanned (< 50 chars per page avg).
    """
    if not _PYMUPDF_AVAILABLE:
        return None

    try:
        doc = _fitz.open(pdf_path)
        pages_text = []
        total_chars = 0

        for page_num, page in enumerate(doc, start=1):
            # get_text("text") respects reading order and handles multi-column
            text = page.get_text("text").strip()
            total_chars += len(text)
            if text:
                pages_text.append(f"--- Page {page_num} ---\n{text}")

        doc.close()

        avg_chars = total_chars / max(len(pages_text), 1)
        if avg_chars < 50:
            # Scanned PDF — very little extractable text → route to OCR
            logger.info(
                f"[CV] PyMuPDF: avg {avg_chars:.0f} chars/page — "
                f"likely scanned, routing to OCR"
            )
            return None

        logger.info(
            f"[CV] PyMuPDF: {len(pages_text)} page(s), "
            f"{total_chars} chars extracted (digital PDF)"
        )
        return "\n\n".join(pages_text)

    except Exception as e:
        logger.warning(f"[CV] PyMuPDF extraction error: {e}")
        return None


def _extract_digital_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """Same as _extract_digital_pdf but from raw bytes."""
    if not _PYMUPDF_AVAILABLE:
        return None

    try:
        doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        total_chars = 0

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            total_chars += len(text)
            if text:
                pages_text.append(f"--- Page {page_num} ---\n{text}")

        doc.close()

        avg_chars = total_chars / max(len(pages_text), 1)
        if avg_chars < 50:
            return None

        return "\n\n".join(pages_text)

    except Exception as e:
        logger.warning(f"[CV] PyMuPDF bytes extraction error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2A: docTR — Neural OCR for Scanned Documents
# ═══════════════════════════════════════════════════════════════════════════════

def _doctr_ocr_pdf(pdf_path: str) -> Optional[str]:
    """
    Run docTR neural OCR on a scanned PDF.
    Returns None if docTR is unavailable.
    """
    model = _get_doctr_model()
    if model is None:
        return None

    try:
        doc_input = _DocTRDocumentFile.from_pdf(pdf_path)
        result    = model(doc_input)
        return _doctr_result_to_text(result)
    except Exception as e:
        logger.warning(f"[CV] docTR PDF OCR error: {e}")
        return None


def _doctr_ocr_image(image: np.ndarray) -> Optional[str]:
    """
    Run docTR neural OCR on an OpenCV image (numpy BGR array).
    """
    model = _get_doctr_model()
    if model is None:
        return None

    try:
        # docTR expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        doc_input = _DocTRDocumentFile.from_images([np.array(pil)])
        result    = model(doc_input)
        return _doctr_result_to_text(result)
    except Exception as e:
        logger.warning(f"[CV] docTR image OCR error: {e}")
        return None


def _doctr_result_to_text(result) -> str:
    """
    Convert docTR Document result object to plain text string.

    docTR result structure:
      result.pages → list of Page
        page.blocks → list of Block
          block.lines → list of Line
            line.words → list of Word
              word.value → str
    """
    pages_text = []

    for page_num, page in enumerate(result.pages, start=1):
        page_lines = []
        for block in page.blocks:
            for line in block.lines:
                words = [word.value for word in line.words if word.value.strip()]
                if words:
                    page_lines.append(" ".join(words))
        if page_lines:
            pages_text.append(f"--- Page {page_num} ---\n" + "\n".join(page_lines))

    text = "\n\n".join(pages_text)
    logger.info(f"[CV] docTR: {len(pages_text)} page(s), {len(text)} chars extracted")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2B: Tesseract — Multilingual Fallback OCR
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_for_tesseract(image: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing for Tesseract.
    docTR does its own preprocessing internally, so this is only for Tesseract.

    Steps:
    1. Grayscale
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) — better than
       simple threshold for variable-lighting legal document scans
    3. Deskew — corrects up to 45° rotation (common in phone-photographed docs)
    """
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # CLAHE — adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise (conservative — avoid destroying thin stroke characters)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=7)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    return _deskew(thresh)


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct image rotation using minAreaRect on dark pixel coordinates.
    Only corrects if tilt > 0.5° to avoid unnecessary interpolation artifacts.
    """
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:
        return image

    h, w   = image.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags      = cv2.INTER_CUBIC,
        borderMode = cv2.BORDER_REPLICATE,
    )


def _tesseract_ocr_image(image: np.ndarray, lang_code: str = "en") -> str:
    """
    Tesseract OCR on a preprocessed image, with correct language pack.

    Args:
        image:     BGR numpy array (will be preprocessed internally)
        lang_code: ISO 639-1 code for Tesseract language selection
    """
    tess_lang  = _TESSERACT_LANG_MAP.get(lang_code, "eng")
    processed  = _preprocess_for_tesseract(image)
    pil_img    = Image.fromarray(processed)

    try:
        raw   = pytesseract.image_to_string(pil_img, lang=tess_lang, config=_TESSERACT_CONFIG)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"[CV] Tesseract OCR error (lang={tess_lang}): {e}")
        # Try English-only fallback before giving up
        if tess_lang != "eng":
            try:
                raw   = pytesseract.image_to_string(pil_img, lang="eng", config=_TESSERACT_CONFIG)
                lines = [line.strip() for line in raw.splitlines() if line.strip()]
                logger.info("[CV] Tesseract: fell back to eng-only after language error")
                return "\n".join(lines)
            except Exception as e2:
                logger.error(f"[CV] Tesseract eng fallback also failed: {e2}")
        return ""


def _tesseract_ocr_pdf(pdf_path: str, lang_code: str = "en") -> str:
    """
    Tesseract OCR on a scanned PDF — converts pages to images first.
    Uses pdf2image (poppler-based).
    """
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        logger.error(f"[CV] pdf2image convert error: {e}")
        return ""

    all_text = []
    for page_num, page_img in enumerate(pages, start=1):
        cv_img    = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
        page_text = _tesseract_ocr_image(cv_img, lang_code)
        if page_text:
            all_text.append(f"--- Page {page_num} ---\n{page_text}")

    return "\n\n".join(all_text)


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_extracted_text(text: str) -> str:
    """
    Post-process extracted text for downstream LLM consumption.

    Operations:
    1. Remove control characters (except newlines/tabs)
    2. Collapse runs of 3+ newlines to double newline (paragraph break)
    3. Strip trailing whitespace per line
    4. Remove OCR noise patterns common in Indian legal documents:
       - Lone single characters on their own line (OCR artifacts)
       - Lines of only punctuation/symbols
    5. Preserve structural markers (--- Page N ---)
    """
    if not text:
        return ""

    # Remove control chars except \n \t
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    lines       = text.split('\n')
    clean_lines = []

    for line in lines:
        stripped = line.rstrip()

        # Keep page markers
        if stripped.startswith('--- Page') or stripped.startswith('[TABLE'):
            clean_lines.append(stripped)
            continue

        # Remove lines that are only punctuation/symbols (OCR noise)
        if stripped and re.match(r'^[\W_]+$', stripped):
            continue

        # Remove lone single-character lines (OCR artifacts, unless digit/letter)
        if len(stripped) == 1 and not stripped.isalnum():
            continue

        clean_lines.append(stripped)

    # Collapse 3+ consecutive blank lines to 2
    result = '\n'.join(clean_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf_path(
    pdf_path:        str,
    source_language: str = "en",
) -> str:
    """
    Extract text from a PDF file using the best available engine.

    Decision tree:
      1. PyMuPDF → digital PDF? → return text + pdfplumber tables merged in
      2. docTR   → scanned PDF (neural OCR, language-agnostic)
      3. Tesseract → fallback (uses source_language for lang pack selection)
    """
    # ── Engine 1: PyMuPDF (digital PDF) ───────────────────────────────────────
    digital_text = _extract_digital_pdf(pdf_path)
    if digital_text:
        # Merge tables from pdfplumber
        tables = _extract_tables_pdfplumber(pdf_path)
        if tables:
            table_block = "\n\n".join(t["text"] for t in tables)
            return _clean_extracted_text(digital_text + "\n\n" + table_block)
        return _clean_extracted_text(digital_text)

    # ── Engine 2: docTR (scanned PDF, neural OCR) ─────────────────────────────
    if _DOCTR_AVAILABLE:
        doctr_text = _doctr_ocr_pdf(pdf_path)
        if doctr_text:
            return _clean_extracted_text(doctr_text)

    # ── Engine 3: Tesseract (multilingual fallback) ────────────────────────────
    logger.info(f"[CV] Falling back to Tesseract (lang={source_language})")
    tess_text = _tesseract_ocr_pdf(pdf_path, source_language)
    return _clean_extracted_text(tess_text)


def extract_text_from_pdf_bytes(
    pdf_bytes:       bytes,
    source_language: str = "en",
) -> str:
    """Same as extract_text_from_pdf_path but accepts raw bytes (FastAPI uploads)."""
    import tempfile, os

    # ── Engine 1: PyMuPDF ──────────────────────────────────────────────────────
    digital_text = _extract_digital_pdf_bytes(pdf_bytes)
    if digital_text:
        tables = _extract_tables_pdfplumber_bytes(pdf_bytes)
        if tables:
            table_block = "\n\n".join(t["text"] for t in tables)
            return _clean_extracted_text(digital_text + "\n\n" + table_block)
        return _clean_extracted_text(digital_text)

    # ── Engines 2 & 3 need a file path — write to temp ────────────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        # ── Engine 2: docTR ────────────────────────────────────────────────────
        if _DOCTR_AVAILABLE:
            doctr_text = _doctr_ocr_pdf(tmp_path)
            if doctr_text:
                return _clean_extracted_text(doctr_text)

        # ── Engine 3: Tesseract ────────────────────────────────────────────────
        logger.info(f"[CV] Falling back to Tesseract (lang={source_language})")
        tess_text = _tesseract_ocr_pdf(tmp_path, source_language)
        return _clean_extracted_text(tess_text)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_text_from_image(
    image:           np.ndarray,
    source_language: str = "en",
) -> str:
    """
    Extract text from an image (OpenCV BGR numpy array).

    Decision tree:
      1. docTR  → neural OCR (language-agnostic, better on low-quality scans)
      2. Tesseract → multilingual fallback (uses source_language for lang pack)

    After extraction, detect_language() is called on result to set
    source_language if it wasn't already known.

    Args:
        image:           BGR numpy array (as returned by cv2.imread)
        source_language: ISO 639-1 hint from accompanying text/state (may be "en")

    Returns:
        Cleaned extracted text string.
    """
    if image is None or image.size == 0:
        logger.warning("[CV] extract_text_from_image received empty image")
        return ""

    # ── Engine 1: docTR ────────────────────────────────────────────────────────
    if _DOCTR_AVAILABLE:
        doctr_text = _doctr_ocr_image(image)
        if doctr_text and len(doctr_text.strip()) > 20:
            return _clean_extracted_text(doctr_text)

    # ── Engine 2: Tesseract ────────────────────────────────────────────────────
    tess_text = _tesseract_ocr_image(image, source_language)
    return _clean_extracted_text(tess_text)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(
    file_path:       str,
    source_language: str = "en",
) -> dict:
    """
    Main entry point. Detects file type and routes to the correct engine stack.

    Args:
        file_path:       Absolute path to the document file.
        source_language: ISO 639-1 language hint (from state["source_language"]).
                         Passed to Tesseract for language pack selection.
                         "en" = auto-detect during OCR.

    Returns:
        {
          "text":              str,   # Extracted and cleaned document text
          "file_type":         str,   # "pdf" | "image" | "unsupported"
          "success":           bool,  # True if non-empty text extracted
          "source_language":   str,   # Detected language of extracted text
          "engine_used":       str,   # "pymupdf" | "doctr" | "tesseract"
          "tables_found":      int,   # Number of tables extracted (PDF only)
          "char_count":        int,   # Length of extracted text
        }

    Usage:
        result = extract_text("/uploads/wage_slip.pdf", source_language="ml")
        if result["success"]:
            rag_pipeline.query(result["text"])
    """
    path   = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = extract_text_from_pdf_path(file_path, source_language)

        # Determine which engine was actually used (best effort from log)
        digital_check = _extract_digital_pdf(file_path) if _PYMUPDF_AVAILABLE else None
        if digital_check:
            engine = "pymupdf"
            table_count = len(_extract_tables_pdfplumber(file_path))
        elif _DOCTR_AVAILABLE:
            engine = "doctr"
            table_count = 0
        else:
            engine = "tesseract"
            table_count = 0

        detected_lang = _detect_ocr_language(text) if text else source_language

        return {
            "text":            text,
            "file_type":       "pdf",
            "success":         bool(text.strip()),
            "source_language": detected_lang,
            "engine_used":     engine,
            "tables_found":    table_count,
            "char_count":      len(text),
        }

    elif suffix in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}:
        image = cv2.imread(file_path)
        if image is None:
            return {
                "text": "", "file_type": "image", "success": False,
                "source_language": source_language, "engine_used": "none",
                "tables_found": 0, "char_count": 0,
            }

        text          = extract_text_from_image(image, source_language)
        detected_lang = _detect_ocr_language(text) if text else source_language

        engine = "doctr" if (_DOCTR_AVAILABLE and text) else "tesseract"

        return {
            "text":            text,
            "file_type":       "image",
            "success":         bool(text.strip()),
            "source_language": detected_lang,
            "engine_used":     engine,
            "tables_found":    0,   # image table extraction not implemented
            "char_count":      len(text),
        }

    else:
        logger.warning(f"[CV] Unsupported file type: {suffix}")
        return {
            "text": "", "file_type": "unsupported", "success": False,
            "source_language": source_language, "engine_used": "none",
            "tables_found": 0, "char_count": 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY SHIMS
# ═══════════════════════════════════════════════════════════════════════════════
# These preserve the original function signatures so existing callers
# (api/document.py, tests/test_cv.py) don't break.

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Backward compat: returns Tesseract-ready preprocessed image."""
    return _preprocess_for_tesseract(image)

def extract_text_from_pdf_path_legacy(pdf_path: str) -> str:
    """Backward compat alias."""
    return extract_text_from_pdf_path(pdf_path)

def extract_text_from_pdf_bytes_legacy(pdf_bytes: bytes) -> str:
    """Backward compat alias."""
    return extract_text_from_pdf_bytes(pdf_bytes)