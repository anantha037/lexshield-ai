"""
LexShield Legal Text Preprocessor
==================================
Pipeline: raw PDF / judgment JSON → clean text → chunks → chunks.json

Sources handled:
  1. Statute PDFs     (data/raw/statutes/)
  2. IL-TUR judgments (data/raw/judgments/iltur_judgments.json)
  3. Pre-chunked SC   (data/raw/judgments/sc_prechunked.json)

Output:
  data/processed/chunks.json  — ~4,500–5,000 quality chunks

Run:
  python data/preprocessor.py
"""

import re
import json
import random
from pathlib import Path
from typing import Optional
import hashlib

# ── Optional PyMuPDF import ───────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("  PyMuPDF not installed. Run: pip install pymupdf")
    print("    PDF processing will be skipped.\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Chunking parameters ───────────────────────────────────────────────────────
# 1 token ≈ 0.75 words for English legal text
# Target: 500 tokens  → ~375 words
# Overlap: 50 tokens  → ~38 words
CHUNK_SIZE_WORDS   = 375
CHUNK_OVERLAP_WORDS = 38

# ── Sampling caps (keeps total under 5,000 for your hardware) ─────────────────
MAX_ILTUR_CHUNKS      = 1000
MAX_PRECHUNKED_CHUNKS = 2000

# ── Statute files to process ──────────────────────────────────────────────────
STATUTE_FILES = {
    "IPC_1860.pdf":                     ("IPC 1860",                     "statute"),
    "BNS_2023.pdf":                     ("BNS 2023",                     "statute"),
    "CrPC_1973.pdf":                    ("CrPC 1973",                    "statute"),
    "Consumer_Protection_Act_2019.pdf": ("Consumer Protection Act 2019", "statute"),
    "Payment_of_Wages_1936.pdf":        ("Payment of Wages Act 1936",    "statute"),
    "BNS_Handbook_2024.pdf":            ("BNS Handbook 2024",            "statute"),
    "Kerala_Rent_Control_Act_1965.pdf": ("Kerala Rent Control Act 1965", "statute"),
}


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — PDF TEXT EXTRACTION
# Uses PyMuPDF to read the text layer directly (much faster + cleaner than OCR)
# OCR pipeline (cv/pipeline.py) is reserved for scanned images/court uploads.
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extracts text page-by-page from a PDF using PyMuPDF.
    Returns list of {page_num, text} dicts.
    """
    if not PYMUPDF_AVAILABLE:
        print(f"  [SKIP] PyMuPDF unavailable — cannot read {pdf_path.name}")
        return []

    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append({"page_num": page_num + 1, "text": text})
        doc.close()
    except Exception as e:
        print(f"  [ERROR] Could not read {pdf_path.name}: {e}")
    return pages


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — TEXT CLEANING
# ═════════════════════════════════════════════════════════════════════════════

def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted text from legal PDFs and judgment files.

    Removes:
      - Standalone page numbers
      - Common Indian legal PDF headers/footers
      - Section separator lines (---, ___, ...)
      - Hyphenated line breaks  (word-\\nword → wordword)
      - Excess whitespace
      - Lines containing only special characters (table artifacts)
      - Runs of 3+ blank lines → collapsed to 1 blank line
    """
    text = raw_text

    # Standalone page numbers (a line that is just digits, optionally spaced)
    text = re.sub(r'(?m)^\s*\d{1,4}\s*$', '', text)

    # Common headers/footers in Indian legislative PDFs
    text = re.sub(
        r'(THE\s+GAZETTE\s+OF\s+INDIA'
        r'|MINISTRY\s+OF\s+LAW'
        r'|GOVERNMENT\s+OF\s+INDIA'
        r'|www\.legislative\.gov\.in'
        r'|legislative\.gov\.in'
        r'|www\.indiacode\.nic\.in'
        r'|indiacode\.nic\.in)',
        '', text, flags=re.IGNORECASE
    )

    # Section separator lines (---, ___, ===, ...)
    text = re.sub(r'(?m)^[\-_\.=]{3,}\s*$', '', text)

    # Fix hyphenated line breaks: "imprison-\nment" → "imprisonment"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Collapse multiple spaces/tabs → single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove lines with only special chars (table borders, junk)
    text = re.sub(
        r'(?m)^\s*[^a-zA-Z0-9\u0D00-\u0D7F\u0900-\u097F\n]{3,}\s*$',
        '', text
    )

    # Collapse 3+ consecutive blank lines → 1 blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SECTION-AWARE CHUNKING
# ═════════════════════════════════════════════════════════════════════════════

def detect_section_header(text: str) -> Optional[str]:
    """
    Tries to extract a section label from the start of a paragraph.

    Recognises patterns like:
      "Section 420."  /  "420. Cheating"  /  "CHAPTER IV"  /  "PART III"
    Returns the label string (≤80 chars) or None.
    """
    patterns = [
        r'^(Section\s+\d+[A-Z]?[\.\-]?\s*[A-Z][^\.]*)',  # Section 420. Cheating...
        r'^(\d+[A-Z]?[\.\)]\s+[A-Z][^\.]{5,60})',         # 420. Cheating and...
        r'^(CHAPTER\s+[IVXLC]+[\.\s][^\n]*)',              # CHAPTER IV OFFENCES
        r'^(PART\s+[IVXLC]+[\.\s][^\n]*)',                 # PART III
    ]
    for pattern in patterns:
        m = re.match(pattern, text.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip()[:80]
    return None


def _save_chunk(
    chunks: list,
    words: list[str],
    source: str,
    doc_type: str,
    section: Optional[str],
) -> None:
    text = " ".join(words).strip()
    if len(text) < 50:
        return

    source_slug = re.sub(r'[^a-z0-9]', '_', source.lower())[:25]
    # Add a short hash of the text to guarantee uniqueness
    text_hash   = hashlib.md5(text.encode()).hexdigest()[:6]
    chunk_id    = f"{source_slug}_{len(chunks) + 1:05d}_{text_hash}"

    chunks.append({
        "chunk_id":   chunk_id,
        "text":       text,
        "source":     source,
        "doc_type":   doc_type,
        "section":    section or "",
        "word_count": len(words),
    })


def chunk_text(
    text: str,
    source: str,
    doc_type: str,
    chunk_size_words: int  = CHUNK_SIZE_WORDS,
    overlap_words: int     = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """
    Splits text into overlapping chunks while respecting paragraph boundaries.

    Strategy:
      1. Split on paragraph boundaries (\\n\\n) to respect natural structure.
      2. If a single paragraph exceeds the chunk size, split by sentences.
      3. Accumulate paragraphs into a chunk until size limit is reached.
      4. Apply overlap: carry last `overlap_words` words into the next chunk.

    Each chunk dict has: chunk_id, text, source, doc_type, section, word_count
    """
    chunks: list[dict] = []
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    current_words:  list[str]    = []
    current_section: Optional[str] = None

    for para in paragraphs:

        # Update running section label if this paragraph is a header
        detected = detect_section_header(para)
        if detected:
            current_section = detected

        para_words = para.split()

        # If the paragraph alone exceeds chunk size → split by sentences first
        if len(para_words) > chunk_size_words:
            sentences = re.split(r'(?<=[.?!])\s+', para)
            for sentence in sentences:
                sent_words = sentence.split()
                if len(current_words) + len(sent_words) > chunk_size_words:
                    if current_words:
                        _save_chunk(chunks, current_words, source, doc_type, current_section)
                        current_words = current_words[-overlap_words:]
                current_words.extend(sent_words)

        else:
            # Normal paragraph: accumulate until size limit
            if len(current_words) + len(para_words) > chunk_size_words:
                if current_words:
                    _save_chunk(chunks, current_words, source, doc_type, current_section)
                    current_words = current_words[-overlap_words:]
            current_words.extend(para_words)

    # Flush any remaining words
    if current_words:
        _save_chunk(chunks, current_words, source, doc_type, current_section)

    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SOURCE PROCESSORS
# ═════════════════════════════════════════════════════════════════════════════

def process_statutes() -> list[dict]:
    """
    Reads every statute PDF in data/raw/statutes/, extracts + cleans text,
    chunks it and returns all chunks.
    All statute chunks are always kept (no sampling) — they are the legal core.
    """
    all_chunks: list[dict] = []
    statutes_dir = RAW_DIR / "statutes"

    for filename, (source_name, doc_type) in STATUTE_FILES.items():
        pdf_path = statutes_dir / filename

        if not pdf_path.exists():
            print(f"   Not found, skipping: {filename}")
            continue

        print(f"\nProcessing: {filename}")
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            print(f"   No text extracted — file may be image-only (needs OCR).")
            continue

        print(f"   Pages extracted : {len(pages)}")

        full_text = "\n\n".join(p["text"] for p in pages)
        cleaned   = clean_text(full_text)
        chunks    = chunk_text(cleaned, source=source_name, doc_type=doc_type)

        all_chunks.extend(chunks)
        print(f"   Chunks created  : {len(chunks)}")

    return all_chunks


def process_iltur_judgments() -> list[dict]:
    """
    Reads iltur_judgments.json, cleans and chunks each judgment text.
    Sampled down to MAX_ILTUR_CHUNKS for performance on 8GB RAM.
    """
    all_chunks: list[dict] = []
    filepath = RAW_DIR / "judgments" / "iltur_judgments.json"

    if not filepath.exists():
        print("   iltur_judgments.json not found — skipping.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        judgments = json.load(f)

    # Random sample so we don't spend hours embedding 130k docs
    if len(judgments) > MAX_ILTUR_CHUNKS * 3:
        # Sample 3x so that after chunking we land near the cap
        judgments = random.sample(judgments, MAX_ILTUR_CHUNKS * 3)

    print(f"\n  Processing IL-TUR judgments ({len(judgments)} docs sampled)...")

    for judgment in judgments:
        raw_text = judgment.get("text", "")
        if not raw_text.strip():
            continue

        source  = f"SC Judgment — {judgment.get('source_config', 'iltur')}"
        cleaned = clean_text(raw_text)
        chunks  = chunk_text(cleaned, source=source, doc_type="judgment")

        for chunk in chunks:
            chunk["court"]         = judgment.get("court", "Supreme Court of India")
            chunk["source_config"] = judgment.get("source_config", "")

        all_chunks.extend(chunks)

        # Stop once we hit the cap
        if len(all_chunks) >= MAX_ILTUR_CHUNKS:
            all_chunks = all_chunks[:MAX_ILTUR_CHUNKS]
            break

    print(f"   Final IL-TUR chunks : {len(all_chunks)}")
    return all_chunks


def process_prechunked_judgments() -> list[dict]:
    """
    Reads sc_prechunked.json — data already chunked by HuggingFace dataset.
    Just cleans and normalises metadata; no re-chunking needed.
    Sampled down to MAX_PRECHUNKED_CHUNKS for performance.
    """
    all_chunks: list[dict] = []
    filepath = RAW_DIR / "judgments" / "sc_prechunked.json"

    if not filepath.exists():
        print("   sc_prechunked.json not found — skipping.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    print(f"\n Processing pre-chunked SC judgments ({len(raw_chunks)} entries)...")

    # Shuffle then iterate so the sample is random, not just the first N
    random.shuffle(raw_chunks)

    for item in raw_chunks:
        text = item.get("text", "") or item.get("chunk", "")
        if not text or len(text.strip()) < 80:
            continue

        cleaned = clean_text(text)
        if len(cleaned) < 80:
            continue

        all_chunks.append({
            "chunk_id":   item.get("chunk_id", f"hf_sc_{len(all_chunks):05d}"),
            "text":       cleaned,
            "source":     item.get("source", "Supreme Court of India"),
            "doc_type":   "judgment",
            "section":    "",
            "word_count": len(cleaned.split()),
        })

        if len(all_chunks) >= MAX_PRECHUNKED_CHUNKS:
            break

    print(f"   Final pre-chunked SC chunks : {len(all_chunks)}")
    return all_chunks


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_preprocessing() -> list[dict]:
    """
    Runs the full pipeline and writes data/processed/chunks.json.
    Returns the final list of chunks.
    """
    random.seed(42)  # Reproducible sampling across runs

    print("=" * 60)
    print("  LexShield Legal Preprocessing Pipeline")
    print("=" * 60)

    all_chunks: list[dict] = []

    # ── 1. Statutes (always include all) ─────────────────────────────────────
    statute_chunks = process_statutes()
    all_chunks.extend(statute_chunks)

    # ── 2. IL-TUR judgments (capped at MAX_ILTUR_CHUNKS) ─────────────────────
    iltur_chunks = process_iltur_judgments()
    all_chunks.extend(iltur_chunks)

    # ── 3. Pre-chunked SC judgments (capped at MAX_PRECHUNKED_CHUNKS) ─────────
    prechunked_chunks = process_prechunked_judgments()
    all_chunks.extend(prechunked_chunks)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = PROCESSED_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print(f"  Statute chunks       : {len(statute_chunks)}")
    print(f"  IL-TUR chunks        : {len(iltur_chunks)}")
    print(f"  Pre-chunked SC chunks: {len(prechunked_chunks)}")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL CHUNKS         : {len(all_chunks)}")
    print(f"  Output               : {output_path}")
    print("=" * 60)

    # ── Quick quality spot-check ──────────────────────────────────────────────
    print("\n Random sample of 3 chunks for quality check:\n")
    for c in random.sample(all_chunks, min(3, len(all_chunks))):
        print(f"  [{c['doc_type'].upper()}] {c['source'][:55]}")
        print(f"  Section : {c['section'][:60] or '—'}")
        print(f"  Words   : {c['word_count']}")
        print(f"  Preview : {c['text'][:180]}...")
        print()

    return all_chunks


if __name__ == "__main__":
    run_preprocessing()