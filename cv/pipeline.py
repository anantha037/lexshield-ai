"""
LexShield CV Pipeline
======================
Handles image preprocessing, OCR, and PDF text extraction.

Three stages:
  1. preprocess_image  — grayscale, denoise, threshold, deskew
  2. extract_text_from_image — Tesseract OCR
  3. PDF handler       — convert pages to images, then run stages 1+2

Tesseract supports: English (eng)
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path

# ── Tesseract binary path (Windows) ──────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Language config ───────────────────────────────────────────────────────────
TESSERACT_LANG = "eng"


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — IMAGE PREPROCESSOR
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Prepares a raw image for OCR.
    Steps: grayscale → denoise → adaptive threshold → deskew

    Args:
        image: numpy array (BGR, as loaded by OpenCV)
    Returns:
        cleaned: 2D numpy array ready for Tesseract
    """
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise (removes small noise artifacts common in scanned legal docs)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold — better than global for uneven lighting
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    # Deskew — correct rotation
    return _deskew(thresh)


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Detects and corrects image rotation using image moments.
    Only corrects if tilt > 0.5° (avoids unnecessary transforms).
    """
    coords = np.column_stack(np.where(image < 128))
    if len(coords) == 0:
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


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — OCR EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_from_image(image: np.ndarray) -> str:
    """
    Runs Tesseract OCR on a preprocessed image.

    Args:
        image: preprocessed 2D numpy array
    Returns:
        Cleaned extracted text string
    """
    config  = "--psm 6 --oem 3"   # PSM 6 = single block of text
    pil_img = Image.fromarray(image)
    raw     = pytesseract.image_to_string(pil_img, lang=TESSERACT_LANG, config=config)

    lines   = [line.strip() for line in raw.splitlines() if line.strip()]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — PDF HANDLER
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf_path(pdf_path: str) -> str:
    """
    Converts each PDF page to an image, preprocesses, and runs OCR.
    Use for scanned PDFs. For digital PDFs use PyMuPDF (preprocessor.py).
    """
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=300)
    return _process_pdf_pages(pages)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Same as above but accepts raw bytes (for FastAPI uploads)."""
    from pdf2image import convert_from_bytes
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    return _process_pdf_pages(pages)


def _process_pdf_pages(pages: list) -> str:
    """Preprocess + OCR each page, return combined text."""
    all_text = []
    for page_num, page_image in enumerate(pages, start=1):
        cv_image     = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        preprocessed = preprocess_image(cv_image)
        page_text    = extract_text_from_image(preprocessed)
        if page_text:
            all_text.append(f"--- Page {page_num} ---\n{page_text}")
    return "\n\n".join(all_text)


# ═════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def extract_text(file_path: str) -> dict:
    """
    Main entry point. Detects file type and routes accordingly.

    Returns:
        dict with keys: text, file_type, success
    """
    path   = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = extract_text_from_pdf_path(file_path)
        return {"text": text, "file_type": "pdf", "success": bool(text.strip())}

    elif suffix in {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}:
        image = cv2.imread(file_path)
        if image is None:
            return {"text": "", "file_type": "image", "success": False}
        preprocessed = preprocess_image(image)
        text         = extract_text_from_image(preprocessed)
        return {"text": text, "file_type": "image", "success": bool(text.strip())}

    else:
        return {"text": "", "file_type": "unsupported", "success": False}