"""
LexShield AI — NER Pipeline  (Week 2, Day 4)
=============================================
Extracts structured legal entities from Indian legal documents.

Entity types returned:
  persons        — names of individuals (title-cased from ALL-CAPS)
  organizations  — companies, courts, government bodies
  dates          — all date formats in Indian legal text
  locations      — cities, states, districts
  monetary       — ₹50,000 / Rs. 50,000 / rupees amounts
  ipc_sections   — Section 420 IPC / u/s 498A / S. 302
  case_numbers   — W.P.(C) No. 1234/2023 / Crl.A. 456/2022
  acts           — Indian Penal Code, Consumer Protection Act etc.

Pipeline order:
  1. Preprocess — title-case ALL-CAPS words so spaCy NER fires on names
  2. spaCy en_core_web_sm — PERSON, ORG, GPE, DATE base entities
  3. OpenNyAI InLegalNER — Indian legal-specific entities (if available)
  4. Custom regex — IPC sections, case numbers, monetary, acts
  5. Merge + deduplicate all entity lists
  6. Return EntityResult dataclass

OpenNyAI is optional — pipeline degrades gracefully if not installed.
spaCy en_core_web_sm is required.
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ── spaCy load ────────────────────────────────────────────────────────────────
try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
        _SPACY_READY = True
    except OSError:
        _nlp = None
        _SPACY_READY = False
        print("[NER] en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
except ImportError:
    _nlp = None
    _SPACY_READY = False
    print("[NER] spaCy not installed. Run: pip install spacy")

# ── OpenNyAI load (optional) ──────────────────────────────────────────────────
_OPENNYAI_READY = False
_legal_nlp      = None

try:
    from opennyai import Pipeline as OpenNyAIPipeline
    _legal_nlp      = OpenNyAIPipeline(["InLegalNER"], use_gpu=False)
    _OPENNYAI_READY = True
    print("[NER] OpenNyAI InLegalNER loaded.")
except ImportError:
    print("[NER] OpenNyAI not installed — using spaCy + regex only.")
    print("      Optional install: pip install opennyai")
except Exception as e:
    print(f"[NER] OpenNyAI load failed ({e}) — using spaCy + regex only.")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EntityResult:
    """
    Structured output of NER extraction.
    All lists are deduplicated and sorted.
    """
    persons:       list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    dates:         list[str] = field(default_factory=list)
    locations:     list[str] = field(default_factory=list)
    monetary:      list[str] = field(default_factory=list)
    ipc_sections:  list[str] = field(default_factory=list)
    case_numbers:  list[str] = field(default_factory=list)
    acts:          list[str] = field(default_factory=list)
    raw_text_used: str       = ""   # the preprocessed text actually passed to NER

    def to_dict(self) -> dict:
        return {
            "persons":       self.persons,
            "organizations": self.organizations,
            "dates":         self.dates,
            "locations":     self.locations,
            "monetary":      self.monetary,
            "ipc_sections":  self.ipc_sections,
            "case_numbers":  self.case_numbers,
            "acts":          self.acts,
            "entity_counts": {
                "persons":       len(self.persons),
                "organizations": len(self.organizations),
                "dates":         len(self.dates),
                "locations":     len(self.locations),
                "monetary":      len(self.monetary),
                "ipc_sections":  len(self.ipc_sections),
                "case_numbers":  len(self.case_numbers),
                "acts":          len(self.acts),
            },
        }


# ── Step 1: ALL-CAPS preprocessor ────────────────────────────────────────────
# Indian legal docs write names as "RAJESH KUMAR" or "M/S ACME ENTERPRISES"
# spaCy fails on ALL-CAPS names. Title-casing fixes this.
# Strategy: title-case only tokens that are ALL-CAPS and > 1 char,
#           skip short abbreviations (IPC, CrPC, NRI etc.) and numbers.

# Known legal abbreviations to preserve in uppercase
_PRESERVE_UPPER: frozenset[str] = frozenset({
    "IPC", "BNS", "CPC", "CRPC", "PIL", "FIR", "RTI", "GST", "PAN", "TAN",
    "AADHAAR", "NGO", "NRI", "OCI", "SC", "HC", "CBI", "ED", "IT", "GST",
    "NEFT", "RTGS", "UPI", "EMI", "NOC", "LOC", "MOU", "SLA", "LLC", "LLP",
    "PVT", "LTD", "CO", "VS", "AND", "OR", "THE", "OF", "IN", "TO", "BY",
    "FOR", "AT", "ON", "WITH", "FROM", "UNDER", "OVER", "BEFORE", "AFTER",
})


def preprocess_allcaps(text: str) -> str:
    """
    Title-cases ALL-CAPS word sequences that look like names, while
    preserving legal abbreviations and mixed-case text.

    "RAJESH KUMAR filed a petition" → "Rajesh Kumar filed a petition"
    "under Section 420 IPC"         → unchanged (IPC preserved)
    "M/S ACME ENTERPRISES PVT LTD"  → "M/S Acme Enterprises PVT LTD"
    """
    tokens = text.split()
    result = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Is this token ALL-CAPS and long enough to be a name word?
        is_allcaps = (
            tok.isupper()
            and len(tok) > 2
            and tok not in _PRESERVE_UPPER
            and not tok.isdigit()
            and not re.match(r'^\d', tok)   # starts with digit (e.g. "123A")
        )
        if is_allcaps:
            result.append(tok.title())
        else:
            result.append(tok)
        i += 1

    return " ".join(result)


# ── Step 2: spaCy entity extraction ──────────────────────────────────────────

# spaCy label → our entity bucket
_SPACY_LABEL_MAP: dict[str, str] = {
    "PERSON":   "persons",
    "ORG":      "organizations",
    "GPE":      "locations",      # geopolitical entity
    "LOC":      "locations",
    "DATE":     "dates",
    "MONEY":    "monetary",
    "FAC":      "organizations",  # facility → org
    "NORP":     "organizations",  # nationalities/groups → org
}

# Noise words to exclude from NER results (spaCy false positives on legal text)
_NOISE_WORDS: frozenset[str] = frozenset({
    "hereinafter", "wherein", "thereof", "thereto", "hereby", "whereas",
    "aforesaid", "abovementioned", "hereunder", "notwithstanding",
    "plaintiff", "defendant", "petitioner", "respondent", "appellant",
    "applicant", "complainant", "accused", "witness", "deponent",
    "court", "tribunal", "bench", "judge", "justice", "magistrate",
    "section", "clause", "article", "rule", "schedule", "annexure",
    "first", "second", "third", "fourth", "fifth",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
})


def _run_spacy(text: str) -> dict[str, list[str]]:
    if not _SPACY_READY or _nlp is None:
        return {k: [] for k in ["persons", "organizations", "dates", "locations", "monetary"]}

    # spaCy max length guard (8GB RAM safe)
    doc    = _nlp(text[:50_000])
    result: dict[str, list[str]] = {
        "persons": [], "organizations": [], "dates": [], "locations": [], "monetary": [],
    }

    for ent in doc.ents:
        bucket = _SPACY_LABEL_MAP.get(ent.label_)
        if not bucket:
            continue
        val = ent.text.strip()
        if (
            len(val) < 2
            or val.lower() in _NOISE_WORDS
            or val.isdigit()
            or re.match(r'^[\d\s\.\,\-]+$', val)
        ):
            continue
        result[bucket].append(val)

    return result


# ── Step 3: OpenNyAI extraction ───────────────────────────────────────────────

# OpenNyAI InLegalNER label → our bucket
_OPENNYAI_LABEL_MAP: dict[str, str] = {
    "PETITIONER":        "persons",
    "RESPONDENT":        "persons",
    "JUDGE":             "persons",
    "LAWYER":            "persons",
    "COURT":             "organizations",
    "GPE":               "locations",
    "ORG":               "organizations",
    "DATE":              "dates",
    "STATUTE":           "acts",
    "PROVISION":         "ipc_sections",
    "CASE_NUMBER":       "case_numbers",
    "PRECEDENT":         "case_numbers",
    "WITNESS":           "persons",
}


def _run_opennyai(text: str) -> dict[str, list[str]]:
    if not _OPENNYAI_READY or _legal_nlp is None:
        return {}
    try:
        result_text = text[:30_000]   # RAM guard
        output = _legal_nlp([result_text])
        entities: dict[str, list[str]] = {}

        for doc_output in output:
            for sent_output in doc_output:
                for ent in sent_output.get("entities", []):
                    label  = ent.get("label", "")
                    val    = ent.get("text", "").strip()
                    bucket = _OPENNYAI_LABEL_MAP.get(label)
                    if bucket and val and len(val) > 1:
                        entities.setdefault(bucket, []).append(val)

        return entities
    except Exception as e:
        print(f"[NER] OpenNyAI extraction failed: {e}")
        return {}


# ── Step 4: Custom regex patterns ────────────────────────────────────────────

# IPC / BNS / CrPC section references
_SECTION_PATTERNS: list[re.Pattern] = [
    # "Section 420", "Section 420A", "Section 108A"
    re.compile(r'\bSection[s]?\s+(\d{1,4}[A-Za-z]{0,2})\b', re.IGNORECASE),
    # "u/s 420", "u/s 420A" (under section shorthand)
    re.compile(r'\bu[/\\]s\s+(\d{1,4}[A-Za-z]{0,2})\b', re.IGNORECASE),
    # "S. 420", "S.420", "Ss. 420, 421"
    re.compile(r'\bSs?\.\s*(\d{1,4}[A-Za-z]{0,2})\b', re.IGNORECASE),
    # "sec. 420", "sec 420"
    re.compile(r'\bsec\.?\s+(\d{1,4}[A-Za-z]{0,2})\b', re.IGNORECASE),
]

# Act names — matched as complete phrases
_ACT_PATTERNS: list[re.Pattern] = [
    re.compile(
        r'\b('
        r'Indian Penal Code(?:\s+\d{4})?'
        r'|Bharatiya Nyaya Sanhita(?:\s+\d{4})?'
        r'|Code of Criminal Procedure(?:\s+\d{4})?'
        r'|Bharatiya Nagarik Suraksha Sanhita(?:\s+\d{4})?'
        r'|Consumer Protection Act(?:\s+\d{4})?'
        r'|Code on Wages(?:\s+\d{4})?'
        r'|Indian Contract Act(?:\s+\d{4})?'
        r'|Transfer of Property Act(?:\s+\d{4})?'
        r'|Registration Act(?:\s+\d{4})?'
        r'|Specific Relief Act(?:\s+\d{4})?'
        r'|Limitation Act(?:\s+\d{4})?'
        r'|Arbitration and Conciliation Act(?:\s+\d{4})?'
        r'|Right to Information Act(?:\s+\d{4})?'
        r'|Motor Vehicles Act(?:\s+\d{4})?'
        r'|Income Tax Act(?:\s+\d{4})?'
        r'|Companies Act(?:\s+\d{4})?'
        r'|Negotiable Instruments Act(?:\s+\d{4})?'
        r'|Prevention of Corruption Act(?:\s+\d{4})?'
        r'|POCSO Act(?:\s+\d{4})?'
        r'|NDPS Act(?:\s+\d{4})?'
        r'|Domestic Violence Act(?:\s+\d{4})?'
        r'|Dowry Prohibition Act(?:\s+\d{4})?'
        r'|Kerala Buildings\s+\([^)]+\)\s+Act(?:\s+\d{4})?'
        r'|[\w\s]+ Act,?\s+\d{4}'   # generic fallback: "Any Named Act, 1980"
        r')',
        re.IGNORECASE,
    )
]

# Case number patterns in Indian courts
_CASE_NUMBER_PATTERNS: list[re.Pattern] = [
    # W.P.(C) No. 1234/2023 — Writ Petition Civil
    re.compile(
        r'\b(W\.?P\.?\s*(?:\([A-Z]+\))?\s*No\.?\s*\d+\s*/\s*\d{4})',
        re.IGNORECASE,
    ),
    # Crl.A. 456/2022, Crl.Rev. 789/2021
    re.compile(
        r'\b(Crl\.?\s*(?:A|Rev|P|M|Petn)\.?\s*(?:No\.?)?\s*\d+\s*/\s*\d{4})',
        re.IGNORECASE,
    ),
    # Civil Appeal / Criminal Appeal
    re.compile(
        r'\b((?:Civil|Criminal|Misc)\s+(?:Appeal|Revision|Petition|Application|Suit)'
        r'\s+(?:No\.?)?\s*\d+\s*/\s*\d{4})',
        re.IGNORECASE,
    ),
    # O.S. No. / C.S. No. (Original Suit)
    re.compile(
        r'\b([A-Z]\.?[A-Z]\.?\s+No\.?\s*\d+\s*/\s*\d{4})',
        re.IGNORECASE,
    ),
    # SLP (Crl) 1234/2023
    re.compile(
        r'\b(SLP\s*(?:\([A-Za-z]+\))?\s*(?:No\.?)?\s*\d+\s*/\s*\d{4})',
        re.IGNORECASE,
    ),
]

# Monetary amounts — Indian formats
_MONETARY_PATTERNS: list[re.Pattern] = [
    # ₹50,000  /  ₹ 50000  /  ₹50.5 lakhs
    re.compile(
        r'₹\s*[\d,]+(?:\.\d+)?\s*(?:lakhs?|crores?|thousands?)?',
        re.IGNORECASE,
    ),
    # Rs. 50,000  /  Rs 50000  /  INR 50,000
    re.compile(
        r'\b(?:Rs\.?|INR)\s*[\d,]+(?:\.\d+)?\s*(?:lakhs?|crores?|thousands?)?',
        re.IGNORECASE,
    ),
    # "50 lakhs"  /  "2 crores"  /  "5 crore rupees"
    re.compile(
        r'\b(\d+(?:\.\d+)?\s*(?:lakhs?|crores?)\s*(?:rupees?)?)\b',
        re.IGNORECASE,
    ),
]

_INDIAN_LOCATIONS: list[str] = [
    # Kerala cities and districts
    "Thiruvananthapuram", "Trivandrum", "Ernakulam", "Kochi", "Cochin",
    "Kozhikode", "Calicut", "Thrissur", "Trichur", "Kollam", "Quilon",
    "Alappuzha", "Alleppey", "Palakkad", "Palghat", "Malappuram",
    "Kannur", "Cannanore", "Kasaragod", "Wayanad", "Idukki", "Pathanamthitta",
    "Kottayam", "Pattom", "Kowdiar", "Kazhakuttam", "Vanchiyoor",
    "Infopark", "Technopark", "Kakkanad",
    # Indian metros and major cities
    "Mumbai", "Bombay", "Delhi", "New Delhi", "Bangalore", "Bengaluru",
    "Chennai", "Madras", "Hyderabad", "Kolkata", "Calcutta", "Pune",
    "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Bhopal", "Patna",
    "Bhubaneswar", "Guwahati", "Dehradun", "Shimla", "Panaji", "Itanagar",
    "Aizawl", "Imphal", "Shillong", "Kohima", "Gangtok", "Agartala",
    # Indian states and UTs
    "Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana",
    "Maharashtra", "Gujarat", "Rajasthan", "Uttar Pradesh", "Bihar",
    "West Bengal", "Odisha", "Madhya Pradesh", "Chhattisgarh", "Jharkhand",
    "Punjab", "Haryana", "Himachal Pradesh", "Uttarakhand", "Goa",
    "Assam", "Meghalaya", "Manipur", "Mizoram", "Nagaland", "Tripura",
    "Arunachal Pradesh", "Sikkim", "Jammu", "Kashmir", "Ladakh",
    "Delhi", "Puducherry", "Chandigarh", "Andaman", "Lakshadweep",
]

# Build single compiled regex for all locations (word boundary match)
_LOCATION_RE = re.compile(
    r'\b(' + '|'.join(re.escape(loc) for loc in _INDIAN_LOCATIONS) + r')\b',
    re.IGNORECASE,
)

# Indian person name extraction — uses legal document introduction patterns
_PERSON_PATTERNS: list[re.Pattern] = [
    # "Mr. Rajesh Kumar" / "Mrs. Priya Sharma" / "Dr. Arun Nair"
    re.compile(
        r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Shri\.?|Smt\.?|Adv\.?|Advocate)\s+'
        r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})',
    ),
    # "son of Gopinath Menon" / "daughter of Ramesh Sharma" — capture the parent
    # But more importantly capture the subject: person named before "son of"
    # Pattern: Capital Name(s), aged / son of / daughter of
    re.compile(
        r'([A-Z][A-Z\s]{3,40}),\s*(?:aged|son of|daughter of|wife of|husband of)',
        re.IGNORECASE,
    ),

    re.compile(
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*\n?\s*'
    r'\((?:HR\s+Director|Director|Manager|Employee|Employer|Landlord|Tenant|'
    r'Witness|Signatory|Authorized\s+Signatory|Partner|Proprietor|'
    r'Chairman|CEO|CFO|COO|CTO|President|Secretary|Trustee|Guardian)\)',
    re.IGNORECASE,
    ),
    
    # "petitioner MOHAMMED IBRAHIM" / "respondent STATE" — capture after role keyword
    re.compile(
        r'(?:Petitioner|Respondent|Appellant|Complainant|Accused|Plaintiff|Defendant)'
        r'\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})',
        re.IGNORECASE,
    ),
]



def _run_regex(text: str) -> dict[str, list[str]]:
    """Extract entities using custom regex patterns."""
    result: dict[str, list[str]] = {
        "ipc_sections": [],
        "case_numbers": [],
        "monetary":     [],
        "acts":         [],
    }

    # IPC/BNS section numbers
    for pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            sec = m.group(1).strip().upper()
            if len(sec) >= 1:
                result["ipc_sections"].append(f"Section {sec}")

    # Acts
    for pat in _ACT_PATTERNS:
        for m in pat.finditer(text):
            act = m.group(0).strip()
            # Clean trailing punctuation
            act = re.sub(r'[,\.\s]+$', '', act)
            if len(act) > 5:
                result["acts"].append(act)

    # Case numbers
    for pat in _CASE_NUMBER_PATTERNS:
        for m in pat.finditer(text):
            cn = m.group(0).strip()
            if len(cn) > 5:
                result["case_numbers"].append(cn)

    # Monetary amounts
    for pat in _MONETARY_PATTERNS:
        for m in pat.finditer(text):
            amount = m.group(0).strip()
            if len(amount) > 1:
                result["monetary"].append(amount)
    
    # Organizations — State/Government/Court patterns spaCy misses
    _ORG_PATTERNS = [
        re.compile(r'\b(State of [A-Z][a-zA-Z\s]+?)(?=\s+(?:represented|through|vs|and|,|\.))',
                re.IGNORECASE),
        re.compile(r'\b(Government of [A-Z][a-zA-Z\s]+?)(?=\s+(?:represented|through|,|\.))',
                re.IGNORECASE),
        re.compile(r'\b(High Court of [A-Z][a-zA-Z\s]+?)(?=\s)',
                re.IGNORECASE),
        re.compile(r'\b(Supreme Court of India)\b', re.IGNORECASE),
        re.compile(r'\b(District Court[,\s])', re.IGNORECASE),
        re.compile(r'\b(Bar Council of [A-Z][a-zA-Z\s]+?)(?=\s*[\.,])',
                re.IGNORECASE),
        re.compile(r'\bM[/\\]S\.?\s+([A-Z][A-Za-z\s]+?(?:Pvt\.?\s*Ltd\.?|Private\s+Limited|LLP|Limited))\b',
                re.IGNORECASE),
    ]

    # Organization regex extraction
    result.setdefault("organizations", [])
    for pat in _ORG_PATTERNS:
        for m in pat.finditer(text):
            org = m.group(0).strip().rstrip('.,;')
            if len(org) > 4:
                result["organizations"].append(org)

    return result

def _run_location_regex(text: str) -> dict[str, list[str]]:
    """Regex-based Indian location extraction as fallback for spaCy."""
    found = _LOCATION_RE.findall(text)
    # Title-case to normalize "KERALA" → "Kerala"
    return {"locations": [loc.title() for loc in found]}

def _run_person_regex(text: str) -> dict[str, list[str]]:
    """Regex-based person extraction using Indian legal document patterns."""
    persons = []
    preprocessed = preprocess_allcaps(text)  # title-case ALL-CAPS names first

    for pat in _PERSON_PATTERNS:
        for m in pat.finditer(preprocessed):
            name = m.group(1).strip().title()
            # Filter noise: must be 2+ words or a known-name length
            words = name.split()
            if len(words) >= 2 and all(len(w) >= 2 for w in words):
                persons.append(name)
            elif len(words) == 1 and len(name) >= 4:
                persons.append(name)

    return {"persons": persons}
# ── Step 5: Merge and deduplicate ─────────────────────────────────────────────

def _clean_val(val: str) -> str:
    """Strip trailing punctuation and normalize whitespace."""
    return re.sub(r'\s+', ' ', val).strip().strip('.,;:()[]')


def _deduplicate(items: list[str]) -> list[str]:
    """
    Deduplicate case-insensitively, keep longest version when overlap exists.
    Sort alphabetically for consistent output.
    """
    seen:   dict[str, str] = {}   # lowercase → canonical form
    for item in items:
        cleaned = _clean_val(item)
        if not cleaned or len(cleaned) < 2:
            continue
        key = cleaned.lower()
        # Keep longer version (e.g. "Indian Penal Code 1860" > "Indian Penal Code")
        if key not in seen or len(cleaned) > len(seen[key]):
            seen[key] = cleaned

    # Also deduplicate substrings: remove "Section 4" if "Section 420" exists
    result   = list(seen.values())
    filtered = []
    for item in result:
        # Keep item unless a longer item contains it as a substring (case-insensitive)
        dominated = any(
            item.lower() != other.lower() and item.lower() in other.lower()
            for other in result
        )
        if not dominated:
            filtered.append(item)

    return sorted(filtered)


def _merge_results(*dicts: dict[str, list[str]]) -> dict[str, list[str]]:
    """Merge multiple entity dicts, combining lists for each key."""
    merged: dict[str, list[str]] = {}
    for d in dicts:
        for key, vals in d.items():
            merged.setdefault(key, []).extend(vals)
    return merged


# ── Main NER function ─────────────────────────────────────────────────────────

def extract_entities(text: str) -> EntityResult:
    if not text or not text.strip():
        return EntityResult()

    text      = text[:50_000]
    processed = preprocess_allcaps(text)

    # Step 2: spaCy
    spacy_ents    = _run_spacy(processed)

    # Step 3: OpenNyAI (optional)
    opennyai_ents = _run_opennyai(text)

    # Step 4: Regex
    regex_ents    = _run_regex(text)

    # Step 4b: NEW — targeted fallback extractors
    location_ents = _run_location_regex(text)
    person_ents   = _run_person_regex(text)

    # Step 5: Merge ALL sources
    all_ents = _merge_results(
        spacy_ents,
        opennyai_ents,
        regex_ents,
        location_ents,   # ← new
        person_ents,     # ← new
    )

    return EntityResult(
        persons       = _deduplicate(all_ents.get("persons",       [])),
        organizations = _deduplicate(all_ents.get("organizations", [])),
        dates         = _deduplicate(all_ents.get("dates",         [])),
        locations     = _deduplicate(all_ents.get("locations",     [])),
        monetary      = _deduplicate(all_ents.get("monetary",      [])),
        ipc_sections  = _deduplicate(all_ents.get("ipc_sections",  [])),
        case_numbers  = _deduplicate(all_ents.get("case_numbers",  [])),
        acts          = _deduplicate(all_ents.get("acts",          [])),
        raw_text_used = processed[:500],
    )

# ── Singleton convenience function ───────────────────────────────────────────

def run_ner(text: str) -> dict:
    """
    Convenience wrapper — returns dict directly.
    Use this in API endpoints.
    """
    return extract_entities(text).to_dict()


print(f"[NER] Pipeline ready. "
      f"spaCy={'✓' if _SPACY_READY else '✗'}  "
      f"OpenNyAI={'✓' if _OPENNYAI_READY else '✗ (optional)'}")