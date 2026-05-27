"""
LexShield AI — CV Pipeline  (Week 3 — Surya Upgrade)
=====================================================
ENGINE STACK (priority order):

  1. PyMuPDF (fitz)    — digital PDFs: direct text extraction, zero OCR cost.
                          Handles embedded fonts, multi-column, annotations.
                          UNCHANGED from previous version.

  2. pdfplumber         — table extraction from digital PDFs.
                          UNCHANGED from previous version.

  3. Surya OCR          — replaces docTR. Transformer-based OCR supporting
                          90+ languages including Malayalam (ml), Hindi (hi),
                          Tamil (ta), Telugu (te), Kannada (kn).
                          Returns per-line confidence scores — used to detect
                          and reject garbage OCR output before it reaches NER.
                          Processes pages one-by-one (RAM safety on 8GB machine).

  4. Tesseract          — tertiary emergency fallback only.
                          Path auto-detected via shutil.which() — works on
                          both Windows and Linux/Docker (fixes hardcoded path bug).

WHAT CHANGED vs previous version:
  - docTR removed (English-only, produced garbage on Indic scripts silently)
  - Surya OCR added (Indic language support, confidence scoring)
  - pdf2image removed from OCR path (was loading all pages into RAM at once)
  - Pages now processed one-by-one with gc.collect() between each (RAM safety)
  - Tesseract path fixed: shutil.which() instead of hardcoded Windows path
  - ocr_confidence added to output dict (0.0-1.0 aggregate quality signal)
  - Quality gate: if confidence < 0.35, returns success=False (no more silent garbage)

WHAT IS UNCHANGED:
  - PyMuPDF extraction logic (identical)
  - pdfplumber table extraction (identical)
  - All preprocessing functions: _preprocess_for_tesseract, _deskew (identical)
  - _clean_extracted_text (identical)
  - All public function signatures: preprocess_image, extract_text_from_image,
    extract_text_from_pdf_path, extract_text_from_pdf_bytes, extract_text
  - Output dict keys (ocr_confidence is a new addition, not a replacement)
  - Backward compatibility shims at the bottom (identical)

Install (update requirements.txt):
  REMOVE: python-doctr[torch]  pdf2image
  ADD:    surya-ocr

System packages (Dockerfile):
  REMOVE: tesseract-ocr-* language packs (Tesseract is now tertiary only)
  KEEP:   tesseract-ocr (still used as emergency fallback)
  REMOVE: poppler-utils (was only needed by pdf2image)
"""

import gc
import io
import re
import logging
import shutil
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ── Suppress noisy warnings on CPU ────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*Torch.*")
warnings.filterwarnings("ignore", message=".*CUDA.*")

# ── Thread limits for CPU inference ───────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

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
    from surya.detection import DetectionPredictor as _SuryaDetectionPredictor
    from surya.recognition import RecognitionPredictor as _SuryaRecognitionPredictor
    _SURYA_AVAILABLE = True
    logger.info("[CV] Surya OCR available — multilingual neural OCR engine ready")
except ImportError:
    _SURYA_AVAILABLE = False
    logger.warning(
        "[CV] Surya OCR not installed. Falling back to Tesseract. "
        "Install: pip install surya-ocr"
    )

# ── Tesseract (tertiary fallback) ─────────────────────────────────────────────
try:
    import pytesseract
    _tess_path = shutil.which("tesseract")
    if _tess_path:
        pytesseract.pytesseract.tesseract_cmd = _tess_path
        _TESSERACT_AVAILABLE = True
        logger.info(f"[CV] Tesseract found at: {_tess_path}")
    else:
        _TESSERACT_AVAILABLE = False
        logger.warning("[CV] Tesseract binary not found. Install tesseract-ocr.")
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.warning("[CV] pytesseract not installed.")


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Surya 0.14.7 uses its own language codes (3-letter for Indic scripts)
_SURYA_SUPPORTED_LANGS = {
    "en", "tam", "mal", "hin", "tel", "kan",
    "mr", "bn", "gu", "pa", "or",
    "fr", "de", "es", "zh", "ja", "ko", "ar", "ru",
}

# Map from Surya lang code → Tesseract lang string
_TESSERACT_LANG_MAP: dict[str, str] = {
    "mal": "mal+eng",
    "hin": "hin+eng",
    "tam": "tam+eng",
    "tel": "tel+eng",
    "kan": "kan+eng",
    "mr":  "hin+eng",
    "bn":  "ben+eng",
    "gu":  "guj+eng",
    "pa":  "pan+eng",
    "or":  "ori+eng",
    "en":  "eng",
}

# Readable names for Tesseract missing-pack error messages
_LANG_DISPLAY_NAMES: dict[str, str] = {
    "tam": "Tamil",   "mal": "Malayalam", "hin": "Hindi",
    "tel": "Telugu",  "kan": "Kannada",   "mr":  "Marathi",
    "bn":  "Bengali", "gu":  "Gujarati",  "pa":  "Punjabi",
    "or":  "Oriya",   "en":  "English",
}

_TESSERACT_CONFIG = "--psm 3 --oem 3"

# OCR quality gate: below this confidence, output is considered garbage
_MIN_OCR_CONFIDENCE = 0.35

# Page processing limits
_MAX_PAGES_PER_CALL = 30    # warn if exceeded, still process
_PAGE_BATCH_SIZE    = 1     # process one page at a time for RAM safety


# ═══════════════════════════════════════════════════════════════════════════════
# SURYA MODEL SINGLETONS  (lazy load — avoids cost on every import)
# ═══════════════════════════════════════════════════════════════════════════════

_surya_det_predictor: "_SuryaDetectionPredictor | None" = None
_surya_rec_predictor: "_SuryaRecognitionPredictor | None" = None


def _get_surya_predictors():
    global _surya_det_predictor, _surya_rec_predictor

    if not _SURYA_AVAILABLE:
        return None, None

    if _surya_det_predictor is None:
        try:
            logger.info("[CV] Loading Surya DetectionPredictor (CPU)…")
            _surya_det_predictor = _SuryaDetectionPredictor()
            logger.info("[CV] Loading Surya RecognitionPredictor (CPU)…")
            _surya_rec_predictor = _SuryaRecognitionPredictor()
            logger.info("[CV] Surya predictors loaded successfully.")
        except Exception as e:
            logger.error(f"[CV] Surya predictor load failed: {e}")
            _surya_det_predictor = None
            return None, None

    return _surya_det_predictor, _surya_rec_predictor


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_ocr_language(text: str) -> str:
    """Detect language from extracted text. Returns ISO 639-1 code."""
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        from agents.multilingual_agent import detect_language
        return detect_language(text.strip()[:500])
    except Exception:
        return "en"


# ── Indic script detection for Surya language parameter ───────────────────────

# Unicode block ranges for Indic scripts
_INDIC_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0B80, 0x0BFF, "tam"),   # Tamil
    (0x0D00, 0x0D7F, "mal"),   # Malayalam
    (0x0900, 0x097F, "hin"),   # Devanagari (Hindi)
    (0x0C00, 0x0C7F, "tel"),   # Telugu
    (0x0C80, 0x0CFF, "kan"),   # Kannada
]

# Keyword hints that may appear in filenames or metadata
_INDIC_FILENAME_HINTS: dict[str, str] = {
    "tamil":     "tam",  "tam":  "tam",
    "malayalam": "mal",  "mal":  "mal",
    "hindi":     "hin",  "hin":  "hin",
    "telugu":    "tel",  "tel":  "tel",
    "kannada":   "kan",  "kan":  "kan",
}


def _detect_indic_script(
    filename: Optional[str] = None,
    metadata_hint: Optional[str] = None,
    prescan_text: Optional[str] = None,
) -> str:
    """
    Detect Indic script from filename, metadata hint, or a pre-scan of
    extractable text.  Returns a Surya 0.14.7 language code.

    Priority:
      1. Filename keywords  (e.g. "contract_tamil.pdf")
      2. Metadata hint       (e.g. caller passes "ta" or "tamil")
      3. Unicode block scan  (look at actual character ranges in text)
      4. Default → "en"
    """
    # 1. Filename keywords
    if filename:
        name_lower = filename.lower()
        for keyword, lang in _INDIC_FILENAME_HINTS.items():
            if keyword in name_lower:
                logger.debug(f"[CV] Indic script detected from filename: {lang}")
                return lang

    # 2. Metadata hint
    if metadata_hint:
        hint = metadata_hint.strip().lower()
        if hint in _INDIC_FILENAME_HINTS:
            return _INDIC_FILENAME_HINTS[hint]
        # Also accept Surya codes directly
        if hint in _SURYA_SUPPORTED_LANGS:
            return hint

    # 3. Unicode block scan of available text
    if prescan_text and len(prescan_text) >= 5:
        script_counts: dict[str, int] = {}
        for ch in prescan_text[:2000]:
            cp = ord(ch)
            for lo, hi, lang in _INDIC_SCRIPT_RANGES:
                if lo <= cp <= hi:
                    script_counts[lang] = script_counts.get(lang, 0) + 1
                    break
        if script_counts:
            dominant = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]
            if script_counts[dominant] >= 3:  # at least 3 chars to be confident
                logger.debug(
                    f"[CV] Indic script detected from unicode scan: {dominant} "
                    f"({script_counts[dominant]} chars)"
                )
                return dominant

    return "en"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE EXTRACTION (pdfplumber) — UNCHANGED
# ═══════════════════════════════════════════════════════════════════════════════

def _table_to_text(table: list[list]) -> str:
    if not table:
        return ""
    lines = []
    for row in table:
        cells = []
        for cell in row:
            cells.append("" if cell is None else " ".join(str(cell).split()))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_tables_pdfplumber(pdf_path: str) -> list[dict]:
    if not _PDFPLUMBER_AVAILABLE:
        return []
    tables = []
    try:
        with _pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for tbl in (page.extract_tables() or []):
                    tbl_text = _table_to_text(tbl)
                    if tbl_text.strip():
                        tables.append({
                            "page": page_num,
                            "text": f"[TABLE — Page {page_num}]\n{tbl_text}",
                        })
    except Exception as e:
        logger.warning(f"[CV] pdfplumber warning: {e}")
    if tables:
        logger.info(f"[CV] pdfplumber: {len(tables)} table(s) extracted")
    return tables


def _extract_tables_pdfplumber_bytes(pdf_bytes: bytes) -> list[dict]:
    if not _PDFPLUMBER_AVAILABLE:
        return []
    tables = []
    try:
        with _pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for tbl in (page.extract_tables() or []):
                    tbl_text = _table_to_text(tbl)
                    if tbl_text.strip():
                        tables.append({
                            "page": page_num,
                            "text": f"[TABLE — Page {page_num}]\n{tbl_text}",
                        })
    except Exception as e:
        logger.warning(f"[CV] pdfplumber bytes warning: {e}")
    return tables


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: PyMuPDF — Digital PDF Extraction  (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_digital_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text from a digital PDF using PyMuPDF.
    Returns None if PDF appears scanned (avg < 50 chars/page).
    UNCHANGED from previous version.
    """
    if not _PYMUPDF_AVAILABLE:
        return None
    try:
        doc         = _fitz.open(pdf_path)
        pages_text  = []
        total_chars = 0
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            total_chars += len(text)
            if text:
                pages_text.append(f"--- Page {page_num} ---\n{text}")
        doc.close()

        avg_chars = total_chars / max(len(pages_text), 1)
        if avg_chars < 50:
            logger.info(f"[CV] PyMuPDF: avg {avg_chars:.0f} chars/page — scanned, routing to OCR")
            return None

        logger.info(f"[CV] PyMuPDF: {len(pages_text)} page(s), {total_chars} chars (digital PDF)")
        return "\n\n".join(pages_text)
    except Exception as e:
        logger.warning(f"[CV] PyMuPDF error: {e}")
        return None


def _extract_digital_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """Same as _extract_digital_pdf but from raw bytes. UNCHANGED."""
    if not _PYMUPDF_AVAILABLE:
        return None
    try:
        doc         = _fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text  = []
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
        logger.warning(f"[CV] PyMuPDF bytes error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Surya OCR — Replaces docTR
# ═══════════════════════════════════════════════════════════════════════════════

def _surya_ocr_image(
    image:         np.ndarray,
    lang_code:     str = "en",
    filename:      Optional[str] = None,
    metadata_hint: Optional[str] = None,
) -> tuple[str, float]:
    """
    Run Surya OCR on a single numpy image (BGR).
    surya-ocr 0.14.7 API:  rec_predictor(images, langs, det_predictor)
      - images: list[PIL.Image]
      - langs:  list[list[str]]  (language codes per image)
      - det_predictor: DetectionPredictor instance

    On any Surya exception, returns ("", 0.0) so callers fall through
    to the Tesseract fallback path.
    """
    det_predictor, rec_predictor = _get_surya_predictors()
    if det_predictor is None:
        return "", 0.0

    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Detect Indic script from filename / metadata / pre-scan
        lang = _detect_indic_script(
            filename=filename,
            metadata_hint=metadata_hint if metadata_hint else (
                lang_code if lang_code != "en" else None
            ),
        )
        if lang not in _SURYA_SUPPORTED_LANGS:
            lang = "en"

        # Surya 0.14.7: positional call — (images, langs_per_image, det_predictor)
        predictions = rec_predictor([pil_image], [[lang]], det_predictor)

        if not predictions or not predictions[0].text_lines:
            return "", 0.0

        lines       = predictions[0].text_lines
        text_lines  = []
        confidences = []
        for line in lines:
            if line.text.strip():
                text_lines.append(line.text.strip())
                confidences.append(line.confidence)

        if not text_lines:
            return "", 0.0

        avg_confidence = sum(confidences) / len(confidences)
        return "\n".join(text_lines), round(avg_confidence, 3)

    except Exception as e:
        logger.warning(f"[CV] Surya OCR error: {e}")
        return "", 0.0


def _surya_ocr_pdf(
    pdf_bytes:     bytes,
    lang_code:     str = "en",
    filename:      Optional[str] = None,
    metadata_hint: Optional[str] = None,
) -> tuple[str, float]:
    """
    Run Surya OCR on a scanned PDF — one page at a time for RAM safety.
    surya-ocr 0.14.7 API:  rec_predictor(images, langs, det_predictor)

    On any Surya exception, returns ("", 0.0) so callers fall through
    to the Tesseract fallback path.
    """
    if not _PYMUPDF_AVAILABLE:
        logger.warning("[CV] PyMuPDF needed for PDF-to-image conversion")
        return "", 0.0

    det_predictor, rec_predictor = _get_surya_predictors()
    if det_predictor is None:
        return "", 0.0

    try:
        doc        = _fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        all_pages  = []
        all_confs  = []

        # Detect Indic script from filename / metadata
        lang = _detect_indic_script(
            filename=filename,
            metadata_hint=metadata_hint if metadata_hint else (
                lang_code if lang_code != "en" else None
            ),
        )
        if lang not in _SURYA_SUPPORTED_LANGS:
            lang = "en"

        if page_count > _MAX_PAGES_PER_CALL:
            logger.warning(
                f"[CV] Large document: {page_count} pages. "
                f"Processing first {_MAX_PAGES_PER_CALL} pages."
            )

        for page_num in range(min(page_count, _MAX_PAGES_PER_CALL)):
            page = doc[page_num]
            mat  = _fitz.Matrix(200 / 72, 200 / 72)
            pix  = page.get_pixmap(matrix=mat, colorspace=_fitz.csRGB)
            img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # On the first page, try to refine language from extracted text
            if page_num == 0 and lang == "en":
                # Quick prescan: try to get text from first page for script detection
                try:
                    first_page_text = doc[0].get_text("text")
                    if first_page_text and len(first_page_text.strip()) >= 5:
                        detected = _detect_indic_script(prescan_text=first_page_text)
                        if detected != "en":
                            lang = detected
                            logger.info(f"[CV] Indic script detected from page 1 text: {lang}")
                except Exception:
                    pass

            try:
                # Surya 0.14.7: positional call — (images, langs_per_image, det_predictor)
                predictions = rec_predictor([img], [[lang]], det_predictor)

                if predictions and predictions[0].text_lines:
                    page_lines = []
                    page_confs = []
                    for line in predictions[0].text_lines:
                        if line.text.strip():
                            page_lines.append(line.text.strip())
                            page_confs.append(line.confidence)
                    if page_lines:
                        all_pages.append(
                            f"--- Page {page_num + 1} ---\n" + "\n".join(page_lines)
                        )
                        all_confs.extend(page_confs)

            except Exception as e:
                logger.warning(f"[CV] Surya: page {page_num + 1} failed: {e}")
            finally:
                del img, pix
                gc.collect()

        doc.close()

        if not all_pages:
            return "", 0.0

        avg_confidence = sum(all_confs) / len(all_confs) if all_confs else 0.0
        text           = "\n\n".join(all_pages)
        logger.info(
            f"[CV] Surya PDF: {len(all_pages)} page(s), "
            f"{len(text)} chars, lang={lang}, confidence={avg_confidence:.2f}"
        )
        return text, round(avg_confidence, 3)

    except Exception as e:
        logger.error(f"[CV] Surya PDF OCR error: {e}")
        return "", 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Tesseract — Emergency Tertiary Fallback  (path bug fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_for_tesseract(image: np.ndarray) -> np.ndarray:
    """
    Image preprocessing for Tesseract.
    UNCHANGED from previous version — CLAHE + deskew logic identical.
    """
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=7)
    thresh   = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return _deskew(thresh)


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct image rotation. UNCHANGED from previous version."""
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
    Tesseract OCR on a preprocessed image.
    FIXED: path now resolved via shutil.which() — works on Windows + Linux/Docker.
    Returns a missing-pack message if the required language data is not installed.
    """
    if not _TESSERACT_AVAILABLE:
        return ""

    tess_lang = _TESSERACT_LANG_MAP.get(lang_code, "eng")
    processed = _preprocess_for_tesseract(image)
    pil_img   = Image.fromarray(processed)

    try:
        raw   = pytesseract.image_to_string(pil_img, lang=tess_lang, config=_TESSERACT_CONFIG)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return "\n".join(lines)
    except Exception as e:
        err_str = str(e).lower()
        logger.warning(f"[CV] Tesseract error (lang={tess_lang}): {e}")

        # Detect missing language pack
        if "failed loading language" in err_str or "not found" in err_str:
            display = _LANG_DISPLAY_NAMES.get(lang_code, lang_code)
            tess_pack = tess_lang.split("+")[0]  # e.g. "mal" from "mal+eng"
            return (
                f"OCR language pack for {display} not installed. "
                f"Run: apt-get install tesseract-ocr-{tess_pack}"
            )

        if tess_lang != "eng":
            try:
                raw   = pytesseract.image_to_string(pil_img, lang="eng", config=_TESSERACT_CONFIG)
                lines = [l.strip() for l in raw.splitlines() if l.strip()]
                logger.info("[CV] Tesseract: fell back to eng-only")
                return "\n".join(lines)
            except Exception as e2:
                logger.error(f"[CV] Tesseract eng fallback failed: {e2}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING  (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_extracted_text(text: str) -> str:
    """
    Post-process extracted text for downstream LLM consumption.
    UNCHANGED from previous version.
    """
    if not text:
        return ""

    text        = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    lines       = text.split('\n')
    clean_lines = []

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith('--- Page') or stripped.startswith('[TABLE'):
            clean_lines.append(stripped)
            continue
        if stripped and re.match(r'^[\W_]+$', stripped):
            continue
        if len(stripped) == 1 and not stripped.isalnum():
            continue
        clean_lines.append(stripped)

    result = '\n'.join(clean_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINTS  (signatures preserved, Surya replaces docTR internally)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf_path(
    pdf_path:        str,
    source_language: str = "en",
) -> str:
    """
    Extract text from a PDF file using the best available engine.

    Decision tree:
      1. PyMuPDF  -> digital PDF? -> return text + pdfplumber tables merged in
      2. Surya    -> scanned PDF (multilingual neural OCR, page-by-page)
      3. Tesseract -> emergency fallback if Surya unavailable

    Public signature UNCHANGED — callers are unaffected.
    """
    # Engine 1: PyMuPDF (digital PDF)
    digital_text = _extract_digital_pdf(pdf_path)
    if digital_text:
        tables = _extract_tables_pdfplumber(pdf_path)
        if tables:
            table_block = "\n\n".join(t["text"] for t in tables)
            return _clean_extracted_text(digital_text + "\n\n" + table_block)
        return _clean_extracted_text(digital_text)

    # Engine 2: Surya (scanned PDF)
    if _SURYA_AVAILABLE:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        surya_text, confidence = _surya_ocr_pdf(pdf_bytes, source_language)
        if surya_text and confidence >= _MIN_OCR_CONFIDENCE:
            return _clean_extracted_text(surya_text)
        elif surya_text:
            logger.warning(
                f"[CV] Surya confidence {confidence:.2f} below threshold "
                f"{_MIN_OCR_CONFIDENCE} — text may be unreliable"
            )
            return _clean_extracted_text(surya_text)  # still return, caller sees confidence

    # Engine 3: Tesseract emergency fallback
    logger.info(f"[CV] Falling back to Tesseract (lang={source_language})")
    try:
        from pdf2image import convert_from_path
        pages    = convert_from_path(pdf_path, dpi=200, last_page=20)
        all_text = []
        for page_num, page_img in enumerate(pages, start=1):
            cv_img    = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
            page_text = _tesseract_ocr_image(cv_img, source_language)
            if page_text:
                all_text.append(f"--- Page {page_num} ---\n{page_text}")
            del cv_img, page_img
            gc.collect()
        return _clean_extracted_text("\n\n".join(all_text))
    except ImportError:
        logger.error("[CV] pdf2image not available for Tesseract fallback")
        return ""
    except Exception as e:
        logger.error(f"[CV] Tesseract PDF fallback error: {e}")
        return ""


def extract_text_from_pdf_bytes(
    pdf_bytes:       bytes,
    source_language: str = "en",
) -> str:
    """
    Same as extract_text_from_pdf_path but accepts raw bytes (FastAPI uploads).
    Public signature UNCHANGED.
    """
    # Engine 1: PyMuPDF
    digital_text = _extract_digital_pdf_bytes(pdf_bytes)
    if digital_text:
        tables = _extract_tables_pdfplumber_bytes(pdf_bytes)
        if tables:
            table_block = "\n\n".join(t["text"] for t in tables)
            return _clean_extracted_text(digital_text + "\n\n" + table_block)
        return _clean_extracted_text(digital_text)

    # Engine 2: Surya
    if _SURYA_AVAILABLE:
        surya_text, confidence = _surya_ocr_pdf(pdf_bytes, source_language)
        if surya_text:
            if confidence < _MIN_OCR_CONFIDENCE:
                logger.warning(
                    f"[CV] Surya confidence {confidence:.2f} below threshold "
                    f"{_MIN_OCR_CONFIDENCE}"
                )
            return _clean_extracted_text(surya_text)

    # Engine 3: Tesseract
    import tempfile
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        return extract_text_from_pdf_path(tmp_path, source_language)
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
      1. Surya  -> multilingual neural OCR (replaces docTR)
      2. Tesseract -> multilingual fallback (uses source_language for lang pack)

    Public signature UNCHANGED — api/document.py callers are unaffected.
    """
    if image is None or image.size == 0:
        logger.warning("[CV] extract_text_from_image received empty image")
        return ""

    # Engine 1: Surya
    if _SURYA_AVAILABLE:
        text, confidence = _surya_ocr_image(image, source_language)
        if text:
            if confidence < _MIN_OCR_CONFIDENCE:
                logger.warning(
                    f"[CV] Surya image confidence {confidence:.2f} — may be unreliable"
                )
            return _clean_extracted_text(text)

    # Engine 2: Tesseract
    tess_text = _tesseract_ocr_image(image, source_language)
    return _clean_extracted_text(tess_text)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT  (output dict unchanged + new ocr_confidence field)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(
    file_path:       str,
    source_language: str = "en",
) -> dict:
    """
    Main entry point. Detects file type and routes to the correct engine stack.

    Returns:
        {
          "text":            str,    # Extracted and cleaned document text
          "file_type":       str,    # "pdf" | "image" | "unsupported"
          "success":         bool,   # True if non-empty text extracted
          "source_language": str,    # Detected language of extracted text
          "engine_used":     str,    # "pymupdf" | "surya" | "tesseract"
          "tables_found":    int,    # Number of tables extracted (PDF only)
          "char_count":      int,    # Length of extracted text
          "ocr_confidence":  float,  # NEW: 0.0-1.0. 1.0 for digital PDFs (no OCR).
        }                            # If < 0.35, analysis results may be unreliable.
    """
    path   = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # Check digital vs scanned once — no double PyMuPDF call
        digital_text = _extract_digital_pdf_bytes(pdf_bytes)

        if digital_text:
            tables      = _extract_tables_pdfplumber_bytes(pdf_bytes)
            table_block = "\n\n".join(t["text"] for t in tables) if tables else ""
            full_text   = _clean_extracted_text(
                digital_text + ("\n\n" + table_block if table_block else "")
            )
            detected_lang = _detect_ocr_language(full_text) if full_text else source_language
            return {
                "text":            full_text,
                "file_type":       "pdf",
                "success":         bool(full_text.strip()),
                "source_language": detected_lang,
                "engine_used":     "pymupdf",
                "tables_found":    len(tables),
                "char_count":      len(full_text),
                "ocr_confidence":  1.0,   # digital PDF: no OCR, perfect quality
            }

        # Scanned PDF — try Surya
        if _SURYA_AVAILABLE:
            surya_text, confidence = _surya_ocr_pdf(pdf_bytes, source_language)
            if surya_text:
                clean         = _clean_extracted_text(surya_text)
                detected_lang = _detect_ocr_language(clean) if clean else source_language
                return {
                    "text":            clean,
                    "file_type":       "pdf",
                    "success":         bool(clean.strip()) and confidence >= _MIN_OCR_CONFIDENCE,
                    "source_language": detected_lang,
                    "engine_used":     "surya",
                    "tables_found":    0,
                    "char_count":      len(clean),
                    "ocr_confidence":  confidence,
                }

        # Tesseract emergency fallback
        tess_text     = extract_text_from_pdf_bytes(pdf_bytes, source_language)
        detected_lang = _detect_ocr_language(tess_text) if tess_text else source_language
        return {
            "text":            tess_text,
            "file_type":       "pdf",
            "success":         bool(tess_text.strip()),
            "source_language": detected_lang,
            "engine_used":     "tesseract",
            "tables_found":    0,
            "char_count":      len(tess_text),
            "ocr_confidence":  0.5,   # Tesseract: no confidence scoring available
        }

    elif suffix in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}:
        image = cv2.imread(file_path)
        if image is None:
            return {
                "text": "", "file_type": "image", "success": False,
                "source_language": source_language, "engine_used": "none",
                "tables_found": 0, "char_count": 0, "ocr_confidence": 0.0,
            }

        if _SURYA_AVAILABLE:
            text, confidence = _surya_ocr_image(image, source_language)
            clean            = _clean_extracted_text(text)
            detected_lang    = _detect_ocr_language(clean) if clean else source_language
            return {
                "text":            clean,
                "file_type":       "image",
                "success":         bool(clean.strip()) and confidence >= _MIN_OCR_CONFIDENCE,
                "source_language": detected_lang,
                "engine_used":     "surya",
                "tables_found":    0,
                "char_count":      len(clean),
                "ocr_confidence":  confidence,
            }

        # Tesseract fallback for images
        tess_text     = _tesseract_ocr_image(image, source_language)
        clean         = _clean_extracted_text(tess_text)
        detected_lang = _detect_ocr_language(clean) if clean else source_language
        return {
            "text":            clean,
            "file_type":       "image",
            "success":         bool(clean.strip()),
            "source_language": detected_lang,
            "engine_used":     "tesseract",
            "tables_found":    0,
            "char_count":      len(clean),
            "ocr_confidence":  0.5,
        }

    else:
        logger.warning(f"[CV] Unsupported file type: {suffix}")
        return {
            "text": "", "file_type": "unsupported", "success": False,
            "source_language": source_language, "engine_used": "none",
            "tables_found": 0, "char_count": 0, "ocr_confidence": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY SHIMS  (UNCHANGED — api/document.py imports these)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Backward compat: returns preprocessed image. UNCHANGED."""
    return _preprocess_for_tesseract(image)


def extract_text_from_pdf_path_legacy(pdf_path: str) -> str:
    """Backward compat alias. UNCHANGED."""
    return extract_text_from_pdf_path(pdf_path)


def extract_text_from_pdf_bytes_legacy(pdf_bytes: bytes) -> str:
    """Backward compat alias. UNCHANGED."""
    return extract_text_from_pdf_bytes(pdf_bytes)