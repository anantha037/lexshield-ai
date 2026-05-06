"""
LexShield AI — Week 2 Contextual Chunking Preprocessor
=======================================================
Replaces Week 1 basic token-splitter with:
  1. Section-boundary detection — splits at legal section headers first
  2. Context injection — every chunk carries its section header as a prefix
  3. Hierarchical metadata — section_number, section_title, chapter, chunk_type
  4. Token-based splitting ONLY as fallback when a section > MAX_SECTION_WORDS

New chunk schema (superset of Week 1):
  chunk_id, text, context_text, source, doc_type,
  section, section_title, chapter, chunk_type, word_count

  context_text = "[source | chapter | section header]\\n{raw text}"
  → embedded for richer semantic signal; text = raw section (shown in prompts)
"""

import re
import json
import hashlib
import gc
import os
from pathlib import Path

# CPU safety — set before any numpy/torch import (even transitive)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Run: pip install PyMuPDF")

# ── Chunking constants ────────────────────────────────────────────────────────
MAX_SECTION_WORDS = 450   # sections larger than this get token-split
OVERLAP_WORDS     = 38    # ~50 token overlap when splitting
MIN_CHUNK_WORDS   = 15    # discard chunks shorter than this

# ── Regex patterns ────────────────────────────────────────────────────────────
CHAPTER_RE = re.compile(
    r'^(CHAPTER\s+(?:[IVXLCDM]+|\d+)[^\n]{0,80})',
    re.MULTILINE | re.IGNORECASE,
)

# Ordered most-specific → least-specific; first match wins for any position
SECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'^(Section\s+\d+[A-Za-z]?\.)',           re.MULTILINE),
    re.compile(r'^(\d{1,4}[A-Z]?\.\s+[A-Z][a-z])',      re.MULTILINE),
    re.compile(r'^(Article\s+\d+[A-Za-z]?\.?)',          re.MULTILINE),
    re.compile(r'^(Rule\s+\d+[A-Za-z]?\.)',              re.MULTILINE | re.IGNORECASE),
    re.compile(r'^([A-Z][A-Z\s\-]{8,60})$',             re.MULTILINE),
]

# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    # ── Existing cleans (Week 1) ──────────────────────────────────────────────
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    # Remove gazette/ministry headers
    text = re.sub(r'THE GAZETTE OF INDIA[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'MINISTRY OF[^\n]*',           '', text, flags=re.IGNORECASE)
    text = re.sub(r'GOVERNMENT OF[^\n]*',         '', text, flags=re.IGNORECASE)
    text = re.sub(r'EXTRAORDINARY\s+PART\s+II[^\n]*', '', text, flags=re.IGNORECASE)
    # Re-join hyphenated line breaks
    text = re.sub(r'-\n(\w)', r'\1', text)

    # ── FIX 1: strip amendment PREFIX from section headers ────────────────────
    # '1[108A. Abetment in India' → '108A. Abetment in India'
    # '8[121A. Conspiracy...'     → '121A. Conspiracy...'
    # Rule: only strip when digit[ is immediately followed by another digit
    # (section numbers) — never strips '1[imprisonment' (letter follows)
    text = re.sub(r'^\d{1,3}\[(\d)', r'\1', text, flags=re.MULTILINE)

    # ── FIX 2: strip footnote annotation lines ────────────────────────────────
    # These are page-footer notes in printed Indian statutes — NOT legal content.
    # Examples:
    #   '1. Subs. by Act 35 of 1969, s. 2, for section 153A.'
    #   '2. Ins. by Act 13 of 2013, s. 20 (w.e.f. 3-2-2013).'
    #   '11. Subs. by the A. O. 1950, for "Queen".'
    #   '6. Now see the Navy Act, 1957 (62 of 1957).'
    #   '5. The words "or that Act" omitted by the A. O. 1950.'
    # These lines start with a small number + period + known amendment keyword.
    # Zero false positives confirmed: no real IPC section starts with these words.
    text = re.sub(
        r'^\d{1,3}\.\s+(?:Subs\.|Ins\.|Rep\.|Added|Proviso|Omitted|Now\s+see|The\s+word)[^\n]*',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # ── FIX 3: unwrap inline amendment markers (cosmetic) ────────────────────
    # '1[imprisonment for life]' → 'imprisonment for life'
    # '2[India]'                 → 'India'
    # Only unwraps short non-nested markers so structural brackets are safe.
    text = re.sub(r'\d{1,3}\[([A-Za-z][^\[\]\n]{1,80})\]', r'\1', text)

    # ── Final normalisation ───────────────────────────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def make_chunk_id(slug: str, index: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
    return f"{slug}_{index:05d}_{h}"


def _is_toc_chunk(text: str) -> bool:
    """True if chunk looks like a Table of Contents (short lines + many dots/numbers)."""
    lines = text.strip().splitlines()
    if not lines:
        return False
    toc = sum(
        1 for l in lines
        if len(l.strip()) < 80 and (
            l.count('.') / max(len(l.strip()), 1) > 0.3
            or re.match(r'^\d[\d\s\.]+$', l.strip())
        )
    )
    return toc / max(len(lines), 1) > 0.65


# ── Section-boundary detection ────────────────────────────────────────────────

def find_chapters(text: str) -> list[tuple[int, str]]:
    return sorted(
        [(m.start(), m.group(1).strip()) for m in CHAPTER_RE.finditer(text)],
        key=lambda x: x[0],
    )


def find_section_boundaries(text: str) -> list[tuple[int, str]]:
    """
    Returns sorted [(char_pos, header_text)] for every section start.
    De-duplicates positions within 5 characters to avoid double-matches.
    """
    seen:  set[int]         = set()
    found: list[tuple[int, str]] = []

    for pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            if any(abs(pos - s) < 5 for s in seen):
                continue
            seen.add(pos)
            found.append((pos, m.group(1).strip()))

    found.sort(key=lambda x: x[0])
    return found


def chapter_at(position: int, chapters: list[tuple[int, str]]) -> str:
    name = "General Provisions"
    for pos, header in chapters:
        if pos <= position:
            name = header
        else:
            break
    return name


def parse_section_header(header: str) -> tuple[str, str]:
    """Returns (section_number, section_title)."""
    for pat, grp_num, grp_title in [
        (r'[Ss]ection\s+(\d+[A-Z]?)\.?\s*(.*)',       1, 2),
        (r'(\d+[A-Z]?)\.\s*(.*)',                       1, 2),
        (r'Article\s+(\d+[A-Z]?)\.?\s*(.*)',           1, 2),
        (r'Rule\s+(\d+[A-Z]?)\.?\s*(.*)',              1, 2),
    ]:
        m = re.match(pat, header, re.IGNORECASE)
        if m:
            num   = m.group(grp_num)
            title = m.group(grp_title).rstrip('.—').strip()
            return num, title
    return "", header


# ── Token-based split for over-long sections ──────────────────────────────────

def split_large_section(
    text: str,
    max_words: int = MAX_SECTION_WORDS,
    overlap:   int = OVERLAP_WORDS,
) -> list[str]:
    words  = text.split()
    parts: list[str] = []
    start  = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return parts


# ── Core contextual chunker ───────────────────────────────────────────────────

def contextual_chunk_document(
    text:         str,
    source:       str,
    doc_type:     str,
    source_slug:  str,
    start_index:  int = 0,
) -> list[dict]:
    """
    Converts a full document string into contextual chunks.

    For each detected section:
      • context_text = "[source | chapter | section header]\\n{section text}"
      • If section fits ≤ MAX_SECTION_WORDS → single chunk (chunk_type='section')
      • If section too long   → overlapping split  (chunk_type='split')
      • No sections detected  → fallback token-split (chunk_type='fallback_split')
    """
    text       = clean_text(text)
    chapters   = find_chapters(text)
    boundaries = find_section_boundaries(text)
    chunks:    list[dict] = []
    idx        = start_index

    # ── No section markers: pure token-split fallback ────────────────────────
    if not boundaries:
        words = text.split()
        i = 0
        while i < len(words):
            part = " ".join(words[i : i + MAX_SECTION_WORDS])
            if len(part.split()) >= MIN_CHUNK_WORDS and not _is_toc_chunk(part):
                cid = make_chunk_id(source_slug, idx, part)
                chunks.append({
                    "chunk_id":      cid,
                    "text":          part,
                    "context_text":  f"[{source}]\n{part}",
                    "source":        source,
                    "doc_type":      doc_type,
                    "section":       "",
                    "section_title": "",
                    "chapter":       "",
                    "chunk_type":    "fallback_split",
                    "word_count":    len(part.split()),
                })
                idx += 1
            i += MAX_SECTION_WORDS - OVERLAP_WORDS
        return chunks

    # ── Build (start, end, header) spans ─────────────────────────────────────
    spans: list[tuple[int, int, str]] = []

    if boundaries[0][0] > 150:
        preamble = text[: boundaries[0][0]].strip()
        if len(preamble.split()) >= MIN_CHUNK_WORDS:
            spans.append((0, boundaries[0][0], "Preamble"))

    for i, (pos, header) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        spans.append((pos, end, header))

    # ── Process each span ────────────────────────────────────────────────────
    for sec_start, sec_end, header in spans:
        raw = text[sec_start:sec_end].strip()
        if len(raw.split()) < MIN_CHUNK_WORDS or _is_toc_chunk(raw):
            continue

        sec_num, sec_title  = parse_section_header(header)
        chap_name           = chapter_at(sec_start, chapters)
        ctx_prefix          = f"[{source} | {chap_name} | {header}]\n"

        if len(raw.split()) <= MAX_SECTION_WORDS:
            # ── Single-chunk section ─────────────────────────────────────────
            cid = make_chunk_id(source_slug, idx, raw)
            chunks.append({
                "chunk_id":      cid,
                "text":          raw,
                "context_text":  ctx_prefix + raw,
                "source":        source,
                "doc_type":      doc_type,
                "section":       sec_num,
                "section_title": sec_title,
                "chapter":       chap_name,
                "chunk_type":    "section",
                "word_count":    len(raw.split()),
            })
            idx += 1
        else:
            # ── Overlapping sub-chunks ───────────────────────────────────────
            for part in split_large_section(raw):
                if len(part.split()) < MIN_CHUNK_WORDS:
                    continue
                cid = make_chunk_id(source_slug, idx, part)
                chunks.append({
                    "chunk_id":      cid,
                    "text":          part,
                    "context_text":  ctx_prefix + part,
                    "source":        source,
                    "doc_type":      doc_type,
                    "section":       sec_num,
                    "section_title": sec_title,
                    "chapter":       chap_name,
                    "chunk_type":    "split",
                    "word_count":    len(part.split()),
                })
                idx += 1

    return chunks


# ── PyMuPDF extraction ────────────────────────────────────────────────────────

def extract_text_pymupdf(pdf_path: str) -> str:
    doc   = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


# ── Statute configs ───────────────────────────────────────────────────────────

STATUTE_CONFIGS: list[dict] = [
    {"path": "data/raw/statutes/ipc.pdf",              "source": "Indian Penal Code (IPC)",                        "doc_type": "statute",  "slug": "ipc"},
    {"path": "data/raw/statutes/bns.pdf",              "source": "Bharatiya Nyaya Sanhita (BNS) 2023",             "doc_type": "statute",  "slug": "bns"},
    {"path": "data/raw/statutes/crpc.pdf",             "source": "Code of Criminal Procedure (CrPC)",              "doc_type": "statute",  "slug": "crpc"},
    {"path": "data/raw/statutes/consumer_protection.pdf","source":"Consumer Protection Act 2019",                  "doc_type": "statute",  "slug": "consumer"},
    {"path": "data/raw/statutes/wages.pdf",            "source": "Code on Wages 2019",                             "doc_type": "statute",  "slug": "wages"},
    {"path": "data/raw/statutes/kerala_rent.pdf",      "source": "Kerala Buildings (Lease and Rent Control) Act",  "doc_type": "statute",  "slug": "kerala_rent"},
    {"path": "data/raw/statutes/bns_handbook.pdf",     "source": "BNS Handbook",                                   "doc_type": "handbook", "slug": "bns_handbook"},
]


def process_all_statutes(start_index: int = 0) -> list[dict]:
    all_chunks: list[dict] = []
    idx = start_index
    for cfg in STATUTE_CONFIGS:
        p = Path(cfg["path"])
        if not p.exists():
            print(f"  [SKIP] {p}")
            continue
        print(f"  {cfg['source']}", end=" ... ", flush=True)
        chunks = contextual_chunk_document(
            text=extract_text_pymupdf(str(p)),
            source=cfg["source"], doc_type=cfg["doc_type"],
            source_slug=cfg["slug"], start_index=idx,
        )
        print(f"{len(chunks)} chunks")
        all_chunks.extend(chunks)
        idx += len(chunks)
        gc.collect()
    return all_chunks


# ── Judgment dataset wrappers ─────────────────────────────────────────────────

def chunk_judgment_records(
    records:      list[dict],
    source_field: str,
    text_field:   str,
    doc_type:     str,
    slug_prefix:  str,
    max_records:  int = 1000,
    start_index:  int = 0,
) -> list[dict]:
    all_chunks: list[dict] = []
    idx = start_index
    for i, rec in enumerate(records[:max_records]):
        text   = rec.get(text_field, "")
        source = str(rec.get(source_field, f"{doc_type}_{i}"))[:200]
        if not text or len(text.strip()) < 100:
            continue
        sub = contextual_chunk_document(
            text=text, source=source, doc_type=doc_type,
            source_slug=f"{slug_prefix}_{i:04d}", start_index=idx,
        )
        all_chunks.extend(sub)
        idx += len(sub)
        if i > 0 and i % 100 == 0:
            print(f"    {i} records → {len(all_chunks)} chunks")
            gc.collect()
    return all_chunks


def wrap_prechunked_records(
    records:     list[dict],
    slug_prefix: str,
    doc_type:    str = "judgment",
    max_records: int = 2000,
    start_index: int = 0,
) -> list[dict]:
    """Wraps already-chunked SC records in the new Week 2 schema."""
    chunks: list[dict] = []
    idx = start_index
    for i, rec in enumerate(records[:max_records]):
        text   = rec.get("text", rec.get("chunk", ""))
        source = str(rec.get("source", rec.get("case", f"SC_{i}")))[:200]
        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            continue
        cid = make_chunk_id(f"{slug_prefix}_{i:04d}", 0, text)
        chunks.append({
            "chunk_id":      cid,
            "text":          text,
            "context_text":  f"[Supreme Court Judgment | {source}]\n{text}",
            "source":        source,
            "doc_type":      doc_type,
            "section":       str(rec.get("section", "")),
            "section_title": str(rec.get("section_title", "")),
            "chapter":       "",
            "chunk_type":    "judgment_segment",
            "word_count":    len(text.split()),
        })
        idx += 1
    return chunks


# ── Full pipeline entry point ─────────────────────────────────────────────────

def run_full_pipeline(
    iltur_path:  str = "data/raw/judgments/iltur_judgments.json",
    sc_path:     str = "data/raw/judgments/sc_prechunked.json",
    output_path: str = "data/processed/chunks.json",
    max_iltur:   int = 1000,
    max_sc:      int = 2000,
) -> list[dict]:
    print("=" * 64)
    print("LexShield AI — Week 2 Contextual Chunking Pipeline")
    print("=" * 64)

    all_chunks: list[dict] = []

    print("\n[1/3] Statute PDFs ...")
    statute_chunks = process_all_statutes(start_index=0)
    all_chunks.extend(statute_chunks)
    print(f"  Statute total : {len(statute_chunks)}")
    gc.collect()

    print(f"\n[2/3] IL-TUR judgments (max {max_iltur}) ...")
    if Path(iltur_path).exists():
        with open(iltur_path, "r", encoding="utf-8") as f:
            iltur_recs = json.load(f)
        iltur_chunks = chunk_judgment_records(
            records=iltur_recs, source_field="case_name", text_field="text",
            doc_type="judgment", slug_prefix="iltur",
            max_records=max_iltur, start_index=len(all_chunks),
        )
        all_chunks.extend(iltur_chunks)
        print(f"  IL-TUR total  : {len(iltur_chunks)}")
    else:
        print(f"  [SKIP] {iltur_path}")
    gc.collect()

    print(f"\n[3/3] SC judgments (max {max_sc}) ...")
    if Path(sc_path).exists():
        with open(sc_path, "r", encoding="utf-8") as f:
            sc_recs = json.load(f)
        sc_chunks = wrap_prechunked_records(
            records=sc_recs, slug_prefix="sc",
            max_records=max_sc, start_index=len(all_chunks),
        )
        all_chunks.extend(sc_chunks)
        print(f"  SC total      : {len(sc_chunks)}")
    else:
        print(f"  [SKIP] {sc_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*64}")
    print(f"DONE  —  Total chunks : {len(all_chunks)}")
    print(f"Saved → {output_path}")

    type_counts: dict[str, int] = {}
    for c in all_chunks:
        k = c.get("chunk_type", "unknown")
        type_counts[k] = type_counts.get(k, 0) + 1
    print("\nchunk_type breakdown:")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:28s}  {n}")

    return all_chunks


if __name__ == "__main__":
    run_full_pipeline()