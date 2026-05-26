"""
LexShield AI — NER Pipeline  (Bug 2 Fix)
=========================================
Root cause (found during analysis):
  The existing pipeline had the RIGHT architecture — 5-stage merge with
  OpenNyAI as optional layer — but the WRONG loader. It was calling:
      from opennyai import Pipeline as OpenNyAIPipeline
  which is the old opennyai SDK API. The HuggingFace model
  opennyaiorg/en_legal_ner_trf is a spaCy model and must be loaded via:
      spacy.load("en_legal_ner_trf")
  Because the import silently failed, the system was falling back to
  en_core_web_sm ONLY on every request — causing wrong entity labels,
  wrong risk scores, and zero BNS/BNSS/CrPC section support.

What this fix does (and does NOT do):
  ✓ Fixes the loader: en_legal_ner_trf loaded via spacy.load()
  ✓ Fixes _run_opennyai() to use the spaCy doc.ents API (not SDK dict API)
  ✓ Adds USE_LEGAL_NER env-flag: "false" = stub locally, "true" = real model
  ✓ Keeps the entire 5-stage pipeline, all regex patterns, EntityResult,
    to_dict() keys, run_ner() — NOTHING downstream changes
  ✗ Does NOT touch risk_scorer.py — it reads dict keys not spaCy labels,
    so it requires zero changes (bug report was wrong on this point)
  ✗ Does NOT touch api/document.py — same reason

Deployment:
  Local dev  → USE_LEGAL_NER=false  (stub, instant, no RAM cost)
  GCP / prod → USE_LEGAL_NER=true   (real model, ~1.5 GB RAM, correct output)

Install on the server (not locally):
  pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl

Entity types returned (unchanged):
  persons        — PETITIONER, RESPONDENT, JUDGE, LAWYER, WITNESS from legal NER
  organizations  — COURT, ORG
  dates          — DATE
  locations      — GPE
  monetary       — regex only (legal NER has no monetary label)
  ipc_sections   — PROVISION label (e.g. "Section 302 IPC") + regex
  case_numbers   — CASE_NUMBER, PRECEDENT + regex
  acts           — STATUTE label + regex
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ── Env flag: controls which NER backend is active ────────────────────────────
# Set USE_LEGAL_NER=true in your Cloud Run / Docker environment.
# Leave unset or false locally — the stub fires instantly with zero RAM cost.
_USE_LEGAL_NER: bool = os.getenv("USE_LEGAL_NER", "false").lower() == "true"

# ── spaCy base model load (en_core_web_sm) ────────────────────────────────────
# Always loaded when USE_LEGAL_NER=false.
# When USE_LEGAL_NER=true, this is the fallback if en_legal_ner_trf fails.
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    spacy = None
    print("[NER] spaCy not installed. Run: pip install spacy")

_nlp         = None
_SPACY_READY = False

if _SPACY_AVAILABLE and not _USE_LEGAL_NER:
    try:
        _nlp         = spacy.load("en_core_web_sm")
        _SPACY_READY = True
        print("[NER] en_core_web_sm loaded (dev mode).")
    except OSError:
        print("[NER] en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")

# ── Legal NER model load (en_legal_ner_trf) ───────────────────────────────────
# Only attempted when USE_LEGAL_NER=true.
# Falls back to en_core_web_sm if the model is not installed.
# This is a spaCy model (transformer-based), NOT the old opennyai SDK.
#
# Install:
#   pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl
#
# Why spacy.load() and not opennyai.Pipeline():
#   The HuggingFace model opennyaiorg/en_legal_ner_trf IS a spaCy model.
#   opennyai.Pipeline() is the old SDK that wraps a different runtime.
#   Using spacy.load() is correct, faster, and avoids the SDK dependency.

_legal_nlp      = None
_LEGAL_NER_READY = False

if _USE_LEGAL_NER and _SPACY_AVAILABLE:
    # Try to load the legal NER transformer model
    try:
        _legal_nlp       = spacy.load("en_legal_ner_trf")
        _LEGAL_NER_READY = True
        print("[NER] en_legal_ner_trf loaded (production mode). Legal entity extraction active.")
    except OSError:
        print("[NER] en_legal_ner_trf not found — falling back to en_core_web_sm.")
        print("      Install: pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf"
              "/resolve/main/en_legal_ner_trf-any-py3-none-any.whl")
        # Graceful fallback: load en_core_web_sm so the endpoint still works
        try:
            _nlp         = spacy.load("en_core_web_sm")
            _SPACY_READY = True
            print("[NER] Fallback: en_core_web_sm loaded.")
        except OSError:
            print("[NER] en_core_web_sm also not found. NER will use regex only.")
    except Exception as e:
        print(f"[NER] en_legal_ner_trf load failed ({e}) — falling back to en_core_web_sm.")
        try:
            _nlp         = spacy.load("en_core_web_sm")
            _SPACY_READY = True
        except OSError:
            pass
elif not _USE_LEGAL_NER and _SPACY_AVAILABLE and not _SPACY_READY:
    # USE_LEGAL_NER=false but en_core_web_sm wasn't loaded yet (shouldn't happen, safety net)
    try:
        _nlp         = spacy.load("en_core_web_sm")
        _SPACY_READY = True
    except OSError:
        pass


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EntityResult:
    """
    Structured output of NER extraction.
    All lists are deduplicated and sorted.
    Keys are STABLE — downstream code (risk_scorer, document.py) reads
    these dict keys directly and does NOT depend on spaCy label names.
    """
    persons:       list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    dates:         list[str] = field(default_factory=list)
    locations:     list[str] = field(default_factory=list)
    monetary:      list[str] = field(default_factory=list)
    ipc_sections:  list[str] = field(default_factory=list)
    case_numbers:  list[str] = field(default_factory=list)
    acts:          list[str] = field(default_factory=list)
    raw_text_used: str       = ""

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
_PRESERVE_UPPER: frozenset[str] = frozenset({
    "IPC", "BNS", "CPC", "CRPC", "PIL", "FIR", "RTI", "GST", "PAN", "TAN",
    "AADHAAR", "NGO", "NRI", "OCI", "SC", "HC", "CBI", "ED", "IT", "GST",
    "NEFT", "RTGS", "UPI", "EMI", "NOC", "LOC", "MOU", "SLA", "LLC", "LLP",
    "PVT", "LTD", "CO", "VS", "AND", "OR", "THE", "OF", "IN", "TO", "BY",
    "FOR", "AT", "ON", "WITH", "FROM", "UNDER", "OVER", "BEFORE", "AFTER",
})

def preprocess_allcaps(text: str) -> str:
    tokens = text.split()
    result = []
    for tok in tokens:
        is_allcaps = (
            tok.isupper()
            and len(tok) > 2
            and tok not in _PRESERVE_UPPER
            and not tok.isdigit()
            and not re.match(r'^\d', tok)
        )
        result.append(tok.title() if is_allcaps else tok)
    return " ".join(result)


# ── Step 2a: spaCy en_core_web_sm extraction (dev / fallback) ────────────────
_SPACY_LABEL_MAP: dict[str, str] = {
    "PERSON": "persons",
    "ORG":    "organizations",
    "GPE":    "locations",
    "LOC":    "locations",
    "DATE":   "dates",
    "MONEY":  "monetary",
    "FAC":    "organizations",
    "NORP":   "organizations",
}

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

_SPACY_VERB_NOISE = re.compile(
    r'\b(produced|filed|prays|dismissing|seeking|leasing|approved|'
    r'dated|represented|impose|has|vide|along|with)\b',
    re.IGNORECASE,
)
_SPACY_PLACE_NOISE = re.compile(
    r'\b(bhavan|station|panchayat|house|madam|pin|exhibit)\b',
    re.IGNORECASE,
)

def _run_spacy_base(text: str) -> dict[str, list[str]]:
    """
    Run en_core_web_sm for base entity extraction.
    Called when USE_LEGAL_NER=false or as fallback.
    """
    empty = {k: [] for k in ["persons", "organizations", "dates", "locations", "monetary"]}
    if not _SPACY_READY or _nlp is None:
        return empty

    doc    = _nlp(text[:50_000])
    result = {k: [] for k in empty}

    for ent in doc.ents:
        bucket = _SPACY_LABEL_MAP.get(ent.label_)
        if not bucket:
            continue
        val = ent.text.strip()
        if (len(val) < 2 or val.lower() in _NOISE_WORDS
                or val.isdigit() or re.match(r'^[\d\s\.\,\-]+$', val)):
            continue
        if bucket in ("persons", "organizations") and _SPACY_VERB_NOISE.search(val):
            continue
        if bucket == "persons" and _SPACY_PLACE_NOISE.search(val):
            continue
        word_count = len(val.split())
        if bucket == "persons"       and word_count > 5:  continue
        if bucket == "organizations" and word_count > 10: continue
        result[bucket].append(val)

    return result


# ── Step 2b: en_legal_ner_trf extraction (production) ────────────────────────
#
# en_legal_ner_trf is a spaCy transformer model trained on Indian court
# judgments. It returns doc.ents with these labels:
#   COURT, PETITIONER, RESPONDENT, JUDGE, LAWYER, PROVISION, STATUTE,
#   PRECEDENT, CASE_NUMBER, DATE, WITNESS, OTHER_PERSON, GPE, ORG
#
# The label→bucket mapping is already correct in the existing codebase.
# The only thing that was broken was the loader (opennyai.Pipeline vs spacy.load).

_LEGAL_NER_LABEL_MAP: dict[str, str] = {
    # People
    "PETITIONER":   "persons",
    "RESPONDENT":   "persons",
    "JUDGE":        "persons",
    "LAWYER":       "persons",
    "WITNESS":      "persons",
    "OTHER_PERSON": "persons",
    # Organizations
    "COURT":        "organizations",
    "ORG":          "organizations",
    # Locations
    "GPE":          "locations",
    # Dates
    "DATE":         "dates",
    # Legal provisions — map to ipc_sections (same key risk_scorer reads)
    "PROVISION":    "ipc_sections",
    # Statutes — map to acts (same key risk_scorer reads)
    "STATUTE":      "acts",
    # Case references
    "CASE_NUMBER":  "case_numbers",
    "PRECEDENT":    "case_numbers",
}

def _run_legal_ner(text: str) -> dict[str, list[str]]:
    """
    Run en_legal_ner_trf for Indian legal entity extraction.
    Called when USE_LEGAL_NER=true and model loaded successfully.

    This replaces _run_opennyai() from the old code.
    The old code called opennyai.Pipeline() which is the wrong API for
    this model. en_legal_ner_trf is a spaCy model: use doc.ents directly.
    """
    if not _LEGAL_NER_READY or _legal_nlp is None:
        return {}

    try:
        # Transformer models are slower — cap at 30k chars (same as old opennyai limit)
        doc    = _legal_nlp(text[:30_000])
        result: dict[str, list[str]] = {}

        for ent in doc.ents:
            bucket = _LEGAL_NER_LABEL_MAP.get(ent.label_)
            if not bucket:
                continue
            val = ent.text.strip()
            if len(val) < 2:
                continue
            result.setdefault(bucket, []).append(val)

        return result

    except Exception as e:
        print(f"[NER] en_legal_ner_trf extraction failed: {e}")
        return {}


# ── Step 2c: Stub (local dev when USE_LEGAL_NER=false and no spaCy) ──────────
def _run_stub(_text: str) -> dict[str, list[str]]:
    """
    Zero-cost stub used during local development.
    Returns empty buckets. Regex pipeline (Step 4) still fires and
    extracts ipc_sections, acts, case_numbers, monetary — which is
    enough for development and smoke-testing the risk scorer.
    """
    return {}


# ── Step 4: Custom regex patterns (unchanged from original) ──────────────────
_SECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'\bSection[s]?\s+(\d{1,4}[A-Za-z]{0,2})\b', re.IGNORECASE),
    re.compile(r'\bu[/\\]s\s+(\d{1,4}[A-Za-z]{0,2})\b',    re.IGNORECASE),
    re.compile(r'\bSs?\.\s*(\d{1,4}[A-Za-z]{0,2})\b',      re.IGNORECASE),
    re.compile(r'\bsec\.?\s+(\d{1,4}[A-Za-z]{0,2})\b',     re.IGNORECASE),
    re.compile(r'\bArticle\s+(\d+[A-Za-z]?(?:\([a-z]\))?)\b', re.IGNORECASE),
]

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
        r')',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b((?:Water|Air|Environment(?:al)?|Forest|Wildlife|Pollution)\s+(?:Protection\s+)?'
        r'(?:Act|Rules?|Regulations?),?\s*(?:\d{4})?)\b',
        re.IGNORECASE,
    ),
    # Generic fallback — capitalized words before Act/Rules/Code
    re.compile(r'\b(?:[A-Z][A-Za-z]*\s+){1,6}(?:Act|Rules?|Code|Regulations?)(?:\s*,?\s*\d{4})?'),
]

_CASE_NUMBER_PATTERNS: list[re.Pattern] = [
    re.compile(r'\b(W\.?P\.?\s*(?:\([A-Z]+\))?\s*No\.?\s*\d+\s*/\s*\d{4})',               re.IGNORECASE),
    re.compile(r'\b(W\.?P\.?\s*\([A-Z]\)\.?\s*(?:No\.?)?\s*\d+\s*(?:of|/)\s*\d{4})',     re.IGNORECASE),
    re.compile(r'\b(W\.?P\.?\s*\([A-Z]\)\s*\d+\s*(?:of|/)\s*\d{4})',                     re.IGNORECASE),
    re.compile(r'\b(Crl\.?\s*(?:A|Rev|P|M|Petn)\.?\s*(?:No\.?)?\s*\d+\s*/\s*\d{4})',     re.IGNORECASE),
    re.compile(r'\b((?:Civil|Criminal|Misc)\s+(?:Appeal|Revision|Petition|Application|Suit)\s+(?:No\.?)?\s*\d+\s*/\s*\d{4})', re.IGNORECASE),
    re.compile(r'\b([A-Z]\.?[A-Z]\.?\s+No\.?\s*\d+\s*/\s*\d{4})',                         re.IGNORECASE),
    re.compile(r'\b(SLP\s*(?:\([A-Za-z]+\))?\s*(?:No\.?)?\s*\d+\s*/\s*\d{4})',           re.IGNORECASE),
]

_MONETARY_PATTERNS: list[re.Pattern] = [
    re.compile(r'₹\s*[\d,]+(?:\.\d+)?\s*(?:lakhs?|crores?|thousands?)?',                 re.IGNORECASE),
    re.compile(r'\b(?:Rs\.?|INR)\s*[\d,]+(?:\.\d+)?\s*(?:lakhs?|crores?|thousands?)?',   re.IGNORECASE),
    re.compile(r'\b(\d+(?:\.\d+)?\s*(?:lakhs?|crores?)\s*(?:rupees?)?)\b',               re.IGNORECASE),
]

_INDIAN_LOCATIONS: list[str] = [
    "Thiruvananthapuram", "Trivandrum", "Ernakulam", "Kochi", "Cochin",
    "Kozhikode", "Calicut", "Thrissur", "Trichur", "Kollam", "Quilon",
    "Alappuzha", "Alleppey", "Palakkad", "Palghat", "Malappuram",
    "Kannur", "Cannanore", "Kasaragod", "Wayanad", "Idukki", "Pathanamthitta",
    "Kottayam", "Pattom", "Kowdiar", "Kazhakuttam", "Vanchiyoor",
    "Infopark", "Technopark", "Kakkanad",
    "Mumbai", "Bombay", "Delhi", "New Delhi", "Bangalore", "Bengaluru",
    "Chennai", "Madras", "Hyderabad", "Kolkata", "Calcutta", "Pune",
    "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Bhopal", "Patna",
    "Bhubaneswar", "Guwahati", "Dehradun", "Shimla", "Panaji", "Itanagar",
    "Aizawl", "Imphal", "Shillong", "Kohima", "Gangtok", "Agartala",
    "Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana",
    "Maharashtra", "Gujarat", "Rajasthan", "Uttar Pradesh", "Bihar",
    "West Bengal", "Odisha", "Madhya Pradesh", "Chhattisgarh", "Jharkhand",
    "Punjab", "Haryana", "Himachal Pradesh", "Uttarakhand", "Goa",
    "Assam", "Meghalaya", "Manipur", "Mizoram", "Nagaland", "Tripura",
    "Arunachal Pradesh", "Sikkim", "Jammu", "Kashmir", "Ladakh",
    "Delhi", "Puducherry", "Chandigarh", "Andaman", "Lakshadweep",
]

_LOCATION_RE = re.compile(
    r'\b(' + '|'.join(re.escape(loc) for loc in _INDIAN_LOCATIONS) + r')\b',
    re.IGNORECASE,
)

_PERSON_PATTERNS: list[re.Pattern] = [
    re.compile(r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Shri\.?|Smt\.?|Adv\.?|Advocate)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})'),
    re.compile(r'([A-Z][A-Z\s]{3,40}),\s*(?:aged|son of|daughter of|wife of|husband of)', re.IGNORECASE),
    re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*\n?\s*\((?:HR\s+Director|Director|Manager|Employee|Employer|Landlord|Tenant|Witness|Signatory|Authorized\s+Signatory|Partner|Proprietor|Chairman|CEO|CFO|COO|CTO|President|Secretary|Trustee|Guardian)\)', re.IGNORECASE),
    re.compile(r'(?:Petitioner|Respondent|Appellant|Complainant|Accused|Plaintiff|Defendant)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})', re.IGNORECASE),
]

_ACT_VERB_NOISE = re.compile(
    r'\b(stipulated|directed|ordered|filed|passed|amended|'
    r'issued|published|notified|held|stated|provided)\b',
    re.IGNORECASE,
)

_ORG_PATTERNS: list[re.Pattern] = [
    re.compile(r'\b(State of [A-Z][a-zA-Z\s]+?)(?=\s+(?:represented|through|vs|and|,|\.))',   re.IGNORECASE),
    re.compile(r'\b(Government of [A-Z][a-zA-Z\s]+?)(?=\s+(?:represented|through|,|\.))',     re.IGNORECASE),
    re.compile(r'\b(High Court of [A-Z][a-zA-Z\s]+?)(?=\s)',                                  re.IGNORECASE),
    re.compile(r'\b(Supreme Court of India)\b',                                                re.IGNORECASE),
    re.compile(r'\b(District Court[,\s])',                                                     re.IGNORECASE),
    re.compile(r'\b(Bar Council of [A-Z][a-zA-Z\s]+?)(?=\s*[\.,])',                           re.IGNORECASE),
    re.compile(r'\bM[/\\]S\.?\s+([A-Z][A-Za-z\s]+?(?:Pvt\.?\s*Ltd\.?|Private\s+Limited|LLP|Limited))\b', re.IGNORECASE),
]


def _run_regex(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        "ipc_sections": [], "case_numbers": [], "monetary": [], "acts": [], "organizations": [],
    }

    for pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            sec = m.group(1).strip().upper()
            if len(sec) >= 1:
                result["ipc_sections"].append(f"Section {sec}")

    for pat in _ACT_PATTERNS:
        for m in pat.finditer(text):
            act = re.sub(r'[,\.\s]+$', '', m.group(0).strip())
            if 5 < len(act) < 75 and not _ACT_VERB_NOISE.search(act):
                result["acts"].append(act)

    for pat in _CASE_NUMBER_PATTERNS:
        for m in pat.finditer(text):
            cn = m.group(0).strip()
            if len(cn) > 5:
                result["case_numbers"].append(cn)

    for pat in _MONETARY_PATTERNS:
        for m in pat.finditer(text):
            amt = m.group(0).strip()
            if len(amt) > 1:
                result["monetary"].append(amt)

    for pat in _ORG_PATTERNS:
        for m in pat.finditer(text):
            org = m.group(0).strip().rstrip('.,;')
            if len(org) > 4:
                result["organizations"].append(org)

    return result


def _run_location_regex(text: str) -> dict[str, list[str]]:
    found = _LOCATION_RE.findall(text)
    found = [f for f in found if 'indiankanoon' not in f.lower() and len(f) < 40]
    return {"locations": [loc.title() for loc in found]}


def _run_person_regex(text: str) -> dict[str, list[str]]:
    persons      = []
    preprocessed = preprocess_allcaps(text)

    preprocessed = re.sub(
        r'\b(?:R[\-\d\,\s&]+\s+)?BY\s+(?:ADV[S]?|ADVOCATE[S]?)\.?\s*'
        r'(?:SRI|SMT|DR|MR|MRS)?\.?\s*[^\n]+',
        '', preprocessed, flags=re.IGNORECASE,
    )
    preprocessed = re.sub(
        r'\b\w[\w\.\s]+v\.\s+(?:Union|State)\s+of\s+India[^\n]*',
        '', preprocessed, flags=re.IGNORECASE,
    )

    for pat in _PERSON_PATTERNS:
        for m in pat.finditer(preprocessed):
            name  = m.group(1).strip().title()
            words = name.split()
            if len(words) >= 2 and all(len(w) >= 2 for w in words):
                persons.append(name)
            elif len(words) == 1 and len(name) >= 4:
                persons.append(name)

    return {"persons": persons}


# ── Step 5: Merge and deduplicate ─────────────────────────────────────────────
_EDGE_NOISE_RE = re.compile(
    r'^(?:the|a|an|of|to|for|by|in|on|at|and|from)\s+'
    r'|\s+(?:the|a|an|of|to|for|by|in|on|at|and|from)$',
    re.IGNORECASE,
)

def _clean_val(val: str) -> str:
    cleaned = re.sub(r'\s+', ' ', val).strip().strip('.,;:()[]')
    prev    = ""
    while cleaned != prev:
        prev    = cleaned
        cleaned = _EDGE_NOISE_RE.sub('', cleaned).strip()
    return cleaned.strip('.,;:()[]')

def _deduplicate(items: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for item in items:
        cleaned = _clean_val(item)
        if not cleaned or len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key not in seen or len(cleaned) > len(seen[key]):
            seen[key] = cleaned

    result   = list(seen.values())
    filtered = []
    for item in result:
        dominated = any(
            item.lower() != other.lower() and item.lower() in other.lower()
            for other in result
        )
        if not dominated:
            filtered.append(item)
    return sorted(filtered)

def _merge_results(*dicts: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for d in dicts:
        for key, vals in d.items():
            merged.setdefault(key, []).extend(vals)
    return merged


# ── Global person scrubber ────────────────────────────────────────────────────
def _global_person_scrub(names: list[str]) -> list[str]:
    cleaned = []
    for name in names:
        if re.search(r'\b(?:vs\.?|v\.?|versus)\b', name, re.IGNORECASE):
            continue
        scrubbed = re.sub(
            r'\s+(?:R[\-\d\,a-z&]+|BY\s+ADV.*|Adv\..*|SC|SR\.|Senior Advocate|Represented\s+By).*$',
            '', name, flags=re.IGNORECASE,
        ).strip('.,;()[] ')
        if len(scrubbed) > 3:
            cleaned.append(scrubbed)
    return cleaned


# ── Main NER function ─────────────────────────────────────────────────────────

def extract_entities(text: str) -> EntityResult:
    """
    5-stage NER pipeline.

    Stage routing by USE_LEGAL_NER env flag:
      false → en_core_web_sm (Stage 2a) + regex stages
      true  → en_legal_ner_trf (Stage 2b, spaCy model) + regex stages
              fallback to en_core_web_sm if model not installed

    All dict keys in EntityResult.to_dict() are STABLE regardless of
    which NER backend is active. risk_scorer.py and document.py are
    unaffected by this change.
    """
    if not text or not text.strip():
        return EntityResult()

    text      = text[:50_000]
    processed = preprocess_allcaps(text)

    # Stage 2: NER model (base or legal, depending on env flag)
    if _USE_LEGAL_NER and _LEGAL_NER_READY:
        # Production: legal transformer NER — best accuracy for Indian legal text
        model_ents = _run_legal_ner(processed)
    elif _SPACY_READY:
        # Dev / fallback: general English NER
        model_ents = _run_spacy_base(processed)
    else:
        # No model loaded at all — regex pipeline still runs below
        model_ents = _run_stub(processed)

    # Stage 4: Regex (always runs — catches ipc_sections, acts, monetary,
    # case_numbers that NER models routinely miss)
    regex_ents    = _run_regex(text)
    location_ents = _run_location_regex(text)
    person_ents   = _run_person_regex(text)

    # Stage 5: Merge all sources
    all_ents = _merge_results(model_ents, regex_ents, location_ents, person_ents)

    return EntityResult(
        persons       = _deduplicate(_global_person_scrub(all_ents.get("persons",       []))),
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
    """Convenience wrapper — returns dict directly. Use this in API endpoints."""
    return extract_entities(text).to_dict()


# ── Startup status ────────────────────────────────────────────────────────────
_active_backend = (
    "en_legal_ner_trf (production)"  if _LEGAL_NER_READY else
    "en_core_web_sm (dev/fallback)"  if _SPACY_READY     else
    "regex-only (no spaCy model)"
)
print(
    f"[NER] Pipeline ready. "
    f"USE_LEGAL_NER={_USE_LEGAL_NER}  "
    f"backend={_active_backend}"
)