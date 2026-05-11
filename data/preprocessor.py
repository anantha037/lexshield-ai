"""
LexShield AI — Contextual Chunking Preprocessor
=======================================================
Changes in this version:

  FIX-1  category not threaded into chunks
         contextual_chunk_document() now accepts category + era params.
         process_all_statutes() passes cfg["category"] and cfg["era"].
         chunk_judgment_records() and wrap_prechunked_records() pass
         category="judgment", era="".

  FIX-2  section_title truncated to 2 chars ("Di" instead of full title)
         SECTION_PATTERNS[1] was r'^(\d{1,4}[A-Z]?\.\s+[A-Z][a-z])'
         which matched only 2 chars after the section number.
         Fixed to r'^(\d{1,4}[A-Z]?\.\s+[A-Z][^\n]{2,60})' which
         captures the full section title line (up to 60 chars).

  NEW    era field added to chunks and ChromaDB metadata
         "legacy"  → acts replaced by new codes (IPC, CrPC, Evidence Act)
         "current" → replacement acts (BNS, BNSS, BSA)
         ""        → all other acts (not part of a replacement pair)
         Used by pipeline to serve paired old+new answers with temporal
         guidance (pre / post July 1 2024).

Statute corpus: 10 categories | 50+ Indian Acts
File convention: data/raw/statutes/{slug}.pdf
"""

import re
import json
import hashlib
import gc
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Run: pip install PyMuPDF")

# ── Chunking constants ────────────────────────────────────────────────────────
MAX_SECTION_WORDS = 450
OVERLAP_WORDS     = 38
MIN_CHUNK_WORDS   = 15

# ── Regex patterns ────────────────────────────────────────────────────────────
CHAPTER_RE = re.compile(
    r'^(CHAPTER\s+(?:[IVXLCDM]+|\d+)[^\n]{0,80})',
    re.MULTILINE | re.IGNORECASE,
)

# FIX-2: pattern index 1 changed — captures full section title, not just 2 chars
SECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'^(Section\s+\d+[A-Za-z]?\.)',              re.MULTILINE),
    re.compile(r'^(\d{1,4}[A-Z]?\.\s+[A-Z][^\n]{2,60})',   re.MULTILINE),  # FIX-2
    re.compile(r'^(Article\s+\d+[A-Za-z]?\.?)',             re.MULTILINE),
    re.compile(r'^(Rule\s+\d+[A-Za-z]?\.)',                 re.MULTILINE | re.IGNORECASE),
    re.compile(r'^([A-Z][A-Z\s\-]{8,60})$',                re.MULTILINE),
]

# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'THE GAZETTE OF INDIA[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'MINISTRY OF[^\n]*',           '', text, flags=re.IGNORECASE)
    text = re.sub(r'GOVERNMENT OF[^\n]*',         '', text, flags=re.IGNORECASE)
    text = re.sub(r'EXTRAORDINARY\s+PART\s+II[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'-\n(\w)', r'\1', text)
    text = re.sub(r'^\d{1,3}\[(\d)', r'\1', text, flags=re.MULTILINE)
    text = re.sub(
        r'^\d{1,3}\.\s+(?:Subs\.|Ins\.|Rep\.|Added|Proviso|Omitted|Now\s+see|The\s+word)[^\n]*',
        '', text, flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(r'\d{1,3}\[([A-Za-z][^\[\]\n]{1,80})\]', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def make_chunk_id(slug: str, index: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
    return f"{slug}_{index:05d}_{h}"


def _is_toc_chunk(text: str) -> bool:
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
    seen:  set[int]              = set()
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
        (r'[Ss]ection\s+(\d+[A-Z]?)\.?\s*(.*)',   1, 2),
        (r'(\d+[A-Z]?)\.\s*(.*)',                   1, 2),
        (r'Article\s+(\d+[A-Z]?)\.?\s*(.*)',       1, 2),
        (r'Rule\s+(\d+[A-Z]?)\.?\s*(.*)',          1, 2),
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
    category:     str = "",   # FIX-1: was missing — now passed from STATUTE_CONFIGS
    era:          str = "",   # NEW: "legacy" | "current" | ""
) -> list[dict]:
    """
    Converts a full document string into contextual chunks.

    category: legal domain e.g. "criminal", "family", "corporate" ...
    era:      "legacy"  = replaced act (IPC, CrPC, Evidence Act)
              "current" = replacement act (BNS, BNSS, BSA)
              ""        = all other acts
    """
    text       = clean_text(text)
    chapters   = find_chapters(text)
    boundaries = find_section_boundaries(text)
    chunks:    list[dict] = []
    idx        = start_index

    # ── No section markers: fallback token-split ──────────────────────────────
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
                    "category":      category,   # FIX-1
                    "era":           era,         # NEW
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

        sec_num, sec_title = parse_section_header(header)
        chap_name          = chapter_at(sec_start, chapters)
        ctx_prefix         = f"[{source} | {chap_name} | {header}]\n"

        if len(raw.split()) <= MAX_SECTION_WORDS:
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
                "category":      category,   # FIX-1
                "era":           era,         # NEW
            })
            idx += 1
        else:
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
                    "category":      category,   # FIX-1
                    "era":           era,         # NEW
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
# era field values:
#   "legacy"  = act replaced by a new code on July 1 2024 (IPC, CrPC, Evidence Act)
#   "current" = the replacement act (BNS, BNSS, BSA)
#   ""        = not part of a replacement pair

STATUTE_CONFIGS: list[dict] = [

    # ── 1. Criminal & Justice ────────────────────────────────────────────────
    {
        "path": "data/raw/statutes/ipc.pdf",
        "source": "Indian Penal Code (IPC) 1860",
        "doc_type": "statute", "slug": "ipc",
        "category": "criminal", "era": "legacy",
    },
    {
        "path": "data/raw/statutes/crpc.pdf",
        "source": "Code of Criminal Procedure (CrPC) 1973",
        "doc_type": "statute", "slug": "crpc",
        "category": "criminal", "era": "legacy",
    },
    {
        "path": "data/raw/statutes/evidence_act.pdf",
        "source": "Indian Evidence Act 1872",
        "doc_type": "statute", "slug": "evidence_act",
        "category": "criminal", "era": "legacy",
    },
    {
        "path": "data/raw/statutes/bns.pdf",
        "source": "Bharatiya Nyaya Sanhita (BNS) 2023",
        "doc_type": "statute", "slug": "bns",
        "category": "criminal", "era": "current",
    },
    {
        "path": "data/raw/statutes/bnss.pdf",
        "source": "Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023",
        "doc_type": "statute", "slug": "bnss",
        "category": "criminal", "era": "current",
    },
    {
        "path": "data/raw/statutes/bsa.pdf",
        "source": "Bharatiya Sakshya Adhiniyam (BSA) 2023",
        "doc_type": "statute", "slug": "bsa",
        "category": "criminal", "era": "current",
    },
    {
        "path": "data/raw/statutes/pocso.pdf",
        "source": "Protection of Children from Sexual Offences (POCSO) Act 2012",
        "doc_type": "statute", "slug": "pocso",
        "category": "criminal", "era": "",
    },
    {
        "path": "data/raw/statutes/pmla.pdf",
        "source": "Prevention of Money Laundering Act (PMLA) 2002",
        "doc_type": "statute", "slug": "pmla",
        "category": "criminal", "era": "",
    },
    {
        "path": "data/raw/statutes/ndps.pdf",
        "source": "Narcotic Drugs and Psychotropic Substances (NDPS) Act 1985",
        "doc_type": "statute", "slug": "ndps",
        "category": "criminal", "era": "",
    },
    {
        "path": "data/raw/statutes/uapa.pdf",
        "source": "Unlawful Activities (Prevention) Act (UAPA) 1967",
        "doc_type": "statute", "slug": "uapa",
        "category": "criminal", "era": "",
    },
    {
        "path": "data/raw/statutes/juvenile_justice.pdf",
        "source": "Juvenile Justice (Care and Protection of Children) Act 2015",
        "doc_type": "statute", "slug": "juvenile_justice",
        "category": "criminal", "era": "",
    },
    {
        "path": "data/raw/statutes/prevention_corruption.pdf",
        "source": "Prevention of Corruption Act 1988",
        "doc_type": "statute", "slug": "prevention_corruption",
        "category": "criminal", "era": "",
    },

    # ── 2. Family & Personal Laws ────────────────────────────────────────────
    {
        "path": "data/raw/statutes/hindu_marriage.pdf",
        "source": "Hindu Marriage Act 1955",
        "doc_type": "statute", "slug": "hindu_marriage",
        "category": "family", "era": "",
    },
    {
        "path": "data/raw/statutes/special_marriage.pdf",
        "source": "Special Marriage Act 1954",
        "doc_type": "statute", "slug": "special_marriage",
        "category": "family", "era": "",
    },
    {
        "path": "data/raw/statutes/muslim_personal_law.pdf",
        "source": "Muslim Personal Law (Shariat) Application Act 1937",
        "doc_type": "statute", "slug": "muslim_personal_law",
        "category": "family", "era": "",
    },
    {
        "path": "data/raw/statutes/indian_succession.pdf",
        "source": "Indian Succession Act 1925",
        "doc_type": "statute", "slug": "indian_succession",
        "category": "family", "era": "",
    },
    {
        "path": "data/raw/statutes/hindu_succession.pdf",
        "source": "Hindu Succession Act 1956",
        "doc_type": "statute", "slug": "hindu_succession",
        "category": "family", "era": "",
    },
    {
        "path": "data/raw/statutes/domestic_violence.pdf",
        "source": "Protection of Women from Domestic Violence Act 2005",
        "doc_type": "statute", "slug": "domestic_violence",
        "category": "family", "era": "",
    },
    {
        "path": "data/raw/statutes/family_courts.pdf",
        "source": "Family Courts Act 1984",
        "doc_type": "statute", "slug": "family_courts",
        "category": "family", "era": "",
    },
    {
        "path": "data/raw/statutes/senior_citizens.pdf",
        "source": "Maintenance and Welfare of Parents and Senior Citizens Act 2007",
        "doc_type": "statute", "slug": "senior_citizens",
        "category": "family", "era": "",
    },

    # ── 3. Corporate, Business & Contract ────────────────────────────────────
    {
        "path": "data/raw/statutes/companies_act.pdf",
        "source": "Companies Act 2013",
        "doc_type": "statute", "slug": "companies_act",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/indian_partnership.pdf",
        "source": "Indian Partnership Act 1932",
        "doc_type": "statute", "slug": "indian_partnership",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/llp_act.pdf",
        "source": "Limited Liability Partnership (LLP) Act 2008",
        "doc_type": "statute", "slug": "llp_act",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/msmed.pdf",
        "source": "Micro, Small and Medium Enterprises Development (MSMED) Act 2006",
        "doc_type": "statute", "slug": "msmed",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/ibc.pdf",
        "source": "Insolvency and Bankruptcy Code (IBC) 2016",
        "doc_type": "statute", "slug": "ibc",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/competition_act.pdf",
        "source": "Competition Act 2002",
        "doc_type": "statute", "slug": "competition_act",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/negotiable_instruments.pdf",
        "source": "Negotiable Instruments Act 1881",
        "doc_type": "statute", "slug": "negotiable_instruments",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/contract_act.pdf",
        "source": "Indian Contract Act 1872",
        "doc_type": "statute", "slug": "contract_act",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/arbitration.pdf",
        "source": "Arbitration and Conciliation Act 1996",
        "doc_type": "statute", "slug": "arbitration",
        "category": "corporate", "era": "",
    },
    {
        "path": "data/raw/statutes/specific_relief.pdf",
        "source": "Specific Relief Act 1963",
        "doc_type": "statute", "slug": "specific_relief",
        "category": "corporate", "era": "",
    },

    # ── 4. Taxation & Finance ─────────────────────────────────────────────────
    {
        "path": "data/raw/statutes/income_tax.pdf",
        "source": "Income Tax Act 1961",
        "doc_type": "statute", "slug": "income_tax",
        "category": "taxation", "era": "",
    },
    {
        "path": "data/raw/statutes/cgst.pdf",
        "source": "Central Goods and Services Tax (CGST) Act 2017",
        "doc_type": "statute", "slug": "cgst",
        "category": "taxation", "era": "",
    },
    {
        "path": "data/raw/statutes/igst.pdf",
        "source": "Integrated Goods and Services Tax (IGST) Act 2017",
        "doc_type": "statute", "slug": "igst",
        "category": "taxation", "era": "",
    },
    {
        "path": "data/raw/statutes/customs_act.pdf",
        "source": "Customs Act 1962",
        "doc_type": "statute", "slug": "customs_act",
        "category": "taxation", "era": "",
    },
    {
        "path": "data/raw/statutes/fema.pdf",
        "source": "Foreign Exchange Management Act (FEMA) 1999",
        "doc_type": "statute", "slug": "fema",
        "category": "taxation", "era": "",
    },
    {
        "path": "data/raw/statutes/sebi_act.pdf",
        "source": "Securities and Exchange Board of India (SEBI) Act 1992",
        "doc_type": "statute", "slug": "sebi_act",
        "category": "taxation", "era": "",
    },
    {
        "path": "data/raw/statutes/banking_regulation.pdf",
        "source": "Banking Regulation Act 1949",
        "doc_type": "statute", "slug": "banking_regulation",
        "category": "taxation", "era": "",
    },

    # ── 5. Property, Real Estate & Tenancy ───────────────────────────────────
    {
        "path": "data/raw/statutes/transfer_of_property.pdf",
        "source": "Transfer of Property Act 1882",
        "doc_type": "statute", "slug": "transfer_of_property",
        "category": "property", "era": "",
    },
    {
        "path": "data/raw/statutes/registration_act.pdf",
        "source": "Registration Act 1908",
        "doc_type": "statute", "slug": "registration_act",
        "category": "property", "era": "",
    },
    {
        "path": "data/raw/statutes/rera.pdf",
        "source": "Real Estate (Regulation and Development) Act (RERA) 2016",
        "doc_type": "statute", "slug": "rera",
        "category": "property", "era": "",
    },
    {
        "path": "data/raw/statutes/kerala_rent.pdf",
        "source": "Kerala Buildings (Lease and Rent Control) Act 1965",
        "doc_type": "statute", "slug": "kerala_rent",
        "category": "property", "era": "",
    },

    # ── 6. Labour & Employment ────────────────────────────────────────────────
    {
        "path": "data/raw/statutes/wages.pdf",
        "source": "Code on Wages 2019",
        "doc_type": "statute", "slug": "wages",
        "category": "labour", "era": "",
    },
    {
        "path": "data/raw/statutes/industrial_relations.pdf",
        "source": "Industrial Relations Code 2020",
        "doc_type": "statute", "slug": "industrial_relations",
        "category": "labour", "era": "",
    },
    {
        "path": "data/raw/statutes/social_security_code.pdf",
        "source": "Code on Social Security 2020",
        "doc_type": "statute", "slug": "social_security_code",
        "category": "labour", "era": "",
    },
    {
        "path": "data/raw/statutes/osh_code.pdf",
        "source": "Occupational Safety, Health and Working Conditions Code 2020",
        "doc_type": "statute", "slug": "osh_code",
        "category": "labour", "era": "",
    },
    {
        "path": "data/raw/statutes/posh_act.pdf",
        "source": "Sexual Harassment of Women at Workplace (POSH) Act 2013",
        "doc_type": "statute", "slug": "posh_act",
        "category": "labour", "era": "",
    },

    # ── 7. Food, Health & Education ──────────────────────────────────────────
    {
        "path": "data/raw/statutes/fssai.pdf",
        "source": "Food Safety and Standards Act (FSSAI) 2006",
        "doc_type": "statute", "slug": "fssai",
        "category": "health", "era": "",
    },
    {
        "path": "data/raw/statutes/rte_act.pdf",
        "source": "Right of Children to Free and Compulsory Education (RTE) Act 2009",
        "doc_type": "statute", "slug": "rte_act",
        "category": "health", "era": "",
    },
    {
        "path": "data/raw/statutes/drugs_cosmetics.pdf",
        "source": "Drugs and Cosmetics Act 1940",
        "doc_type": "statute", "slug": "drugs_cosmetics",
        "category": "health", "era": "",
    },
    {
        "path": "data/raw/statutes/clinical_establishments.pdf",
        "source": "Clinical Establishments (Registration and Regulation) Act 2010",
        "doc_type": "statute", "slug": "clinical_establishments",
        "category": "health", "era": "",
    },

    # ── 8. Environment, Forest & Agriculture ─────────────────────────────────
    {
        "path": "data/raw/statutes/environment_protection.pdf",
        "source": "Environment (Protection) Act 1986",
        "doc_type": "statute", "slug": "environment_protection",
        "category": "environment", "era": "",
    },
    {
        "path": "data/raw/statutes/forest_conservation.pdf",
        "source": "Forest (Conservation) Act 1980",
        "doc_type": "statute", "slug": "forest_conservation",
        "category": "environment", "era": "",
    },
    {
        "path": "data/raw/statutes/indian_forest.pdf",
        "source": "Indian Forest Act 1927",
        "doc_type": "statute", "slug": "indian_forest",
        "category": "environment", "era": "",
    },
    {
        "path": "data/raw/statutes/wildlife_protection.pdf",
        "source": "Wildlife (Protection) Act 1972",
        "doc_type": "statute", "slug": "wildlife_protection",
        "category": "environment", "era": "",
    },
    {
        "path": "data/raw/statutes/water_pollution.pdf",
        "source": "Water (Prevention and Control of Pollution) Act 1974",
        "doc_type": "statute", "slug": "water_pollution",
        "category": "environment", "era": "",
    },
    {
        "path": "data/raw/statutes/air_pollution.pdf",
        "source": "Air (Prevention and Control of Pollution) Act 1981",
        "doc_type": "statute", "slug": "air_pollution",
        "category": "environment", "era": "",
    },

    # ── 9. Technology, Media & IP ─────────────────────────────────────────────
    {
        "path": "data/raw/statutes/it_act.pdf",
        "source": "Information Technology (IT) Act 2000",
        "doc_type": "statute", "slug": "it_act",
        "category": "technology", "era": "",
    },
    {
        "path": "data/raw/statutes/dpdp_act.pdf",
        "source": "Digital Personal Data Protection (DPDP) Act 2023",
        "doc_type": "statute", "slug": "dpdp_act",
        "category": "technology", "era": "",
    },
    {
        "path": "data/raw/statutes/copyright_act.pdf",
        "source": "Copyright Act 1957",
        "doc_type": "statute", "slug": "copyright_act",
        "category": "technology", "era": "",
    },
    {
        "path": "data/raw/statutes/patents_act.pdf",
        "source": "Patents Act 1970",
        "doc_type": "statute", "slug": "patents_act",
        "category": "technology", "era": "",
    },
    {
        "path": "data/raw/statutes/trademarks_act.pdf",
        "source": "Trade Marks Act 1999",
        "doc_type": "statute", "slug": "trademarks_act",
        "category": "technology", "era": "",
    },

    # ── 10. Citizen Rights & Daily Administration ─────────────────────────────
    {
        "path": "data/raw/statutes/constitution.pdf",
        "source": "Constitution of India 1950",
        "doc_type": "statute", "slug": "constitution",
        "category": "civil", "era": "",
    },
    {
        "path": "data/raw/statutes/motor_vehicles.pdf",
        "source": "Motor Vehicles Act 1988",
        "doc_type": "statute", "slug": "motor_vehicles",
        "category": "civil", "era": "",
    },
    {
        "path": "data/raw/statutes/rti_act.pdf",
        "source": "Right to Information (RTI) Act 2005",
        "doc_type": "statute", "slug": "rti_act",
        "category": "civil", "era": "",
    },
    {
        "path": "data/raw/statutes/consumer_protection.pdf",
        "source": "Consumer Protection Act 2019",
        "doc_type": "statute", "slug": "consumer",
        "category": "civil", "era": "",
    },
    {
        "path": "data/raw/statutes/aadhaar_act.pdf",
        "source": "Aadhaar (Targeted Delivery of Financial and Other Subsidies) Act 2016",
        "doc_type": "statute", "slug": "aadhaar_act",
        "category": "civil", "era": "",
    },
    {
        "path": "data/raw/statutes/cpc.pdf",
        "source": "Code of Civil Procedure (CPC) 1908",
        "doc_type": "statute", "slug": "cpc",
        "category": "civil", "era": "",
    },

    # ── Handbooks / Reference ─────────────────────────────────────────────────
    {
        "path": "data/raw/statutes/bns_handbook.pdf",
        "source": "BNS Handbook",
        "doc_type": "handbook", "slug": "bns_handbook",
        "category": "criminal", "era": "current",
    },
]

# ── Convenience lookups ───────────────────────────────────────────────────────

def get_configs_by_category(category: str) -> list[dict]:
    return [c for c in STATUTE_CONFIGS if c.get("category") == category]


def get_config_by_slug(slug: str) -> dict | None:
    for c in STATUTE_CONFIGS:
        if c["slug"] == slug:
            return c
    return None


def process_all_statutes(
    start_index: int = 0,
    category:    str | None = None,
    slugs:       list[str] | None = None,
) -> list[dict]:
    configs = STATUTE_CONFIGS
    if category:
        configs = [c for c in configs if c.get("category") == category]
    if slugs:
        configs = [c for c in configs if c["slug"] in slugs]

    all_chunks: list[dict] = []
    idx = start_index
    for cfg in configs:
        p = Path(cfg["path"])
        if not p.exists():
            print(f"  [SKIP] {p}")
            continue
        print(f"  {cfg['source']}", end=" ... ", flush=True)
        chunks = contextual_chunk_document(
            text         = extract_text_pymupdf(str(p)),
            source       = cfg["source"],
            doc_type     = cfg["doc_type"],
            source_slug  = cfg["slug"],
            start_index  = idx,
            category     = cfg.get("category", ""),   # FIX-1
            era          = cfg.get("era", ""),          # NEW
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
            category="judgment", era="",   # FIX-1
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
            "category":      "judgment",   # FIX-1
            "era":           "",            # NEW
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
    category:    str | None = None,
    slugs:       list[str] | None = None,
) -> list[dict]:
    print("=" * 64)
    print("LexShield AI — Contextual Chunking Pipeline")
    print(f"Statute configs loaded: {len(STATUTE_CONFIGS)} acts across 10 categories")
    print("=" * 64)

    all_chunks: list[dict] = []

    print("\n[1/3] Statute PDFs ...")
    statute_chunks = process_all_statutes(start_index=0, category=category, slugs=slugs)
    all_chunks.extend(statute_chunks)
    print(f"  Statute total : {len(statute_chunks)}")
    gc.collect()

    if category or slugs:
        print("\n  [Selective run — skipping judgment datasets]")
    else:
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