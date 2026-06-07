"""
LexShield AI — Risk Scorer v3
==============================
Key fix: Section risk now checks (section_number, act) pairs — not bare numbers.
Cross-references NER's ipc_sections + acts fields to avoid false positives
across 68 acts where same section numbers appear in multiple acts.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Risk level thresholds ─────────────────────────────────────────────────────
# Score: Low=0.0–0.35, Medium=0.36–0.65, High=0.66–0.85, Critical=0.86–1.0

@dataclass
class RiskResult:
    score:               float
    level:               str
    factors:             list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    doc_type_risk:       float     = 0.0
    entity_risk:         float     = 0.0
    llm_risk:            Optional[float] = None
    method:              str       = "rule_based"

    def to_dict(self) -> dict:
        return {
            "score":               round(self.score, 3),
            "level":               self.level,
            "factors":             self.factors,
            "recommended_actions": self.recommended_actions,
            "breakdown": {
                "doc_type_risk": round(self.doc_type_risk, 3),
                "entity_risk":   round(self.entity_risk,   3),
                "llm_risk":      round(self.llm_risk, 3) if self.llm_risk else None,
            },
            "method": self.method,
        }


# ── Base risk by document type ────────────────────────────────────────────────
_DOC_TYPE_BASE_RISK: dict[str, float] = {
    "fir":                   0.85,
    "bail_application":      0.80,
    "cheque_bounce_notice":  0.75,
    "court_notice_summons":  0.65,
    "sc_judgment":           0.60,
    "hc_judgment":           0.55,
    "legal_notice":          0.55,
    "police_complaint":      0.50,
    "loan_agreement":        0.45,
    "power_of_attorney":     0.40,
    "property_deed":         0.35,
    "consumer_complaint":    0.35,
    "employment_contract":   0.30,
    "rental_agreement":      0.25,
    "affidavit":             0.20,
    "unknown":               0.50,
    "uncertain":             0.50,
}


# ── Act name normalizer ───────────────────────────────────────────────────────
# Maps partial/variant act names extracted by NER to canonical keys.
# NER may return "Indian Penal Code 1860", "IPC", "the IPC" etc.
# All map to the same canonical key used in risk tables below.

_ACT_ALIASES: list[tuple[re.Pattern, str]] = [
    # Criminal codes
    (re.compile(r'\bIndian Penal Code\b|\bIPC\b', re.IGNORECASE),                   "IPC"),
    (re.compile(r'\bBharatiya Nyaya Sanhita\b|\bBNS\b', re.IGNORECASE),             "BNS"),
    (re.compile(r'\bCode of Criminal Procedure\b|\bCrPC\b|\bCr\.P\.C\b', re.IGNORECASE), "CrPC"),
    (re.compile(r'\bBharatiya Nagarik Suraksha Sanhita\b|\bBNSS\b', re.IGNORECASE), "BNSS"),

    # Special criminal acts
    (re.compile(r'\bNegotiable Instruments Act\b|\bNI Act\b', re.IGNORECASE),        "NI_ACT"),
    (re.compile(r'\bPOCSO\b|Protection of Children from Sexual Offences', re.IGNORECASE), "POCSO"),
    (re.compile(r'\bNDPS Act\b|Narcotic Drugs and Psychotropic Substances', re.IGNORECASE), "NDPS"),
    (re.compile(r'\bPrevention of Corruption Act\b|\bPC Act\b', re.IGNORECASE),      "PC_ACT"),
    (re.compile(r'\bDomestic Violence Act\b|Protection of Women from Domestic Violence', re.IGNORECASE), "DV_ACT"),
    (re.compile(r'\bDowry Prohibition Act\b', re.IGNORECASE),                        "DOWRY_ACT"),
    (re.compile(r'\bScheduled Caste.*Scheduled Tribe.*Prevention\b|\bSC.ST Act\b', re.IGNORECASE), "SC_ST_ACT"),
    (re.compile(r'\bUnlawful Activities.*Prevention\b|\bUAPA\b', re.IGNORECASE),     "UAPA"),
    (re.compile(r'\bArms Act\b', re.IGNORECASE),                                     "ARMS_ACT"),
    (re.compile(r'\bExplosives Act\b', re.IGNORECASE),                               "EXPLOSIVES_ACT"),
    (re.compile(r'\bInformation Technology Act\b|\bIT Act\b', re.IGNORECASE),        "IT_ACT"),

    # Civil / property
    (re.compile(r'\bTransfer of Property Act\b|\bTP Act\b', re.IGNORECASE),          "TP_ACT"),
    (re.compile(r'\bRegistration Act\b', re.IGNORECASE),                             "REGISTRATION_ACT"),
    (re.compile(r'\bSpecific Relief Act\b', re.IGNORECASE),                          "SPECIFIC_RELIEF_ACT"),
    (re.compile(r'\bLimitation Act\b', re.IGNORECASE),                               "LIMITATION_ACT"),
    (re.compile(r'\bCode of Civil Procedure\b|\bCPC\b|\bC\.P\.C\b', re.IGNORECASE), "CPC"),

    # Consumer / labour
    (re.compile(r'\bConsumer Protection Act\b', re.IGNORECASE),                      "CONSUMER_ACT"),
    (re.compile(r'\bCode on Wages\b|Minimum Wages Act\b', re.IGNORECASE),            "WAGES_ACT"),
    (re.compile(r'\bIndustrial Disputes Act\b', re.IGNORECASE),                      "ID_ACT"),

    # Contract / commercial
    (re.compile(r'\bIndian Contract Act\b', re.IGNORECASE),                          "CONTRACT_ACT"),
    (re.compile(r'\bCompanies Act\b', re.IGNORECASE),                                "COMPANIES_ACT"),
    (re.compile(r'\bInsolvency.*Bankruptcy Code\b|\bIBC\b', re.IGNORECASE),          "IBC"),
    (re.compile(r'\bArbitration.*Conciliation Act\b', re.IGNORECASE),                "ARBITRATION_ACT"),

    # Tax
    (re.compile(r'\bIncome Tax Act\b|\bIT Act\b.*1961', re.IGNORECASE),              "INCOME_TAX_ACT"),
    (re.compile(r'\bGST Act\b|Goods and Services Tax\b', re.IGNORECASE),             "GST_ACT"),

    # Property / rent
    (re.compile(r'\bKerala Buildings.*Act\b|Kerala Rent.*Act\b', re.IGNORECASE),     "KERALA_RENT_ACT"),

    # RTI
    (re.compile(r'\bRight to Information Act\b|\bRTI Act\b', re.IGNORECASE),        "RTI_ACT"),

    # Constitutional
    (re.compile(r'\bConstitution of India\b', re.IGNORECASE),                        "CONSTITUTION"),

    # Motor vehicles
    (re.compile(r'\bMotor Vehicles Act\b', re.IGNORECASE),                           "MV_ACT"),
]


def _normalize_act(act_text: str) -> str | None:
    """Return canonical act key for a raw act string from NER, or None if unrecognized."""
    for pattern, canonical in _ACT_ALIASES:
        if pattern.search(act_text):
            return canonical
    return None


# ── Section risk table: {(canonical_act, section_number): risk_level} ─────────
# risk_level: "HIGH" | "MEDIUM"
# Section numbers stored as strings, uppercase, e.g. "302", "138", "498A"
# Only sections that CANNOT be inferred from doc_type alone need entries here.

_SECTION_RISK: dict[tuple[str, str], str] = {

    # ── IPC ──────────────────────────────────────────────────────────────────
    ("IPC", "302"):  "HIGH",    # Murder
    ("IPC", "304"):  "HIGH",    # Culpable homicide
    ("IPC", "304B"): "HIGH",    # Dowry death
    ("IPC", "307"):  "HIGH",    # Attempt to murder
    ("IPC", "376"):  "HIGH",    # Rape
    ("IPC", "376A"): "HIGH",    # Rape (aggravated)
    ("IPC", "377"):  "HIGH",    # Unnatural offences
    ("IPC", "395"):  "HIGH",    # Dacoity
    ("IPC", "396"):  "HIGH",    # Dacoity with murder
    ("IPC", "364A"): "HIGH",    # Kidnapping for ransom
    ("IPC", "120B"): "HIGH",    # Criminal conspiracy
    ("IPC", "121"):  "HIGH",    # Waging war against state
    ("IPC", "124A"): "HIGH",    # Sedition
    ("IPC", "498A"): "HIGH",    # Cruelty by husband/relatives
    ("IPC", "406"):  "HIGH",    # Criminal breach of trust
    ("IPC", "409"):  "HIGH",    # CBT by public servant
    ("IPC", "420"):  "HIGH",    # Cheating
    ("IPC", "467"):  "HIGH",    # Forgery of valuable security
    ("IPC", "468"):  "HIGH",    # Forgery for cheating
    ("IPC", "471"):  "HIGH",    # Using forged document
    ("IPC", "326"):  "HIGH",    # Grievous hurt with dangerous weapon
    ("IPC", "326A"): "HIGH",    # Acid attack
    ("IPC", "354"):  "HIGH",    # Assault on woman with intent to outrage modesty
    ("IPC", "354A"): "HIGH",    # Sexual harassment
    ("IPC", "354D"): "HIGH",    # Stalking
    ("IPC", "323"):  "MEDIUM",  # Voluntarily causing hurt
    ("IPC", "324"):  "MEDIUM",  # Hurt with dangerous weapon
    ("IPC", "341"):  "MEDIUM",  # Wrongful restraint
    ("IPC", "342"):  "MEDIUM",  # Wrongful confinement
    ("IPC", "379"):  "MEDIUM",  # Theft
    ("IPC", "380"):  "MEDIUM",  # Theft in dwelling
    ("IPC", "392"):  "MEDIUM",  # Robbery
    ("IPC", "415"):  "MEDIUM",  # Cheating (general)
    ("IPC", "417"):  "MEDIUM",  # Punishment for cheating
    ("IPC", "499"):  "MEDIUM",  # Defamation
    ("IPC", "500"):  "MEDIUM",  # Punishment for defamation
    ("IPC", "107"):  "MEDIUM",  # Abetment
    ("IPC", "143"):  "MEDIUM",  # Unlawful assembly
    ("IPC", "147"):  "MEDIUM",  # Rioting
    ("IPC", "148"):  "MEDIUM",  # Rioting with deadly weapon
    ("IPC", "506"):  "MEDIUM",  # Criminal intimidation
    ("IPC", "509"):  "MEDIUM",  # Word/gesture to insult modesty of woman

    # ── BNS (Bharatiya Nyaya Sanhita 2023 — replaces IPC) ────────────────────
    ("BNS", "103"):  "HIGH",    # Murder (≈ IPC 302)
    ("BNS", "109"):  "HIGH",    # Attempt to murder (≈ IPC 307)
    ("BNS", "64"):   "HIGH",    # Rape (≈ IPC 376)
    ("BNS", "111"):  "HIGH",    # Organised crime
    ("BNS", "113"):  "HIGH",    # Terrorist act
    ("BNS", "316"):  "HIGH",    # CBT (≈ IPC 406/409)
    ("BNS", "318"):  "HIGH",    # Cheating (≈ IPC 420)
    ("BNS", "85"):   "HIGH",    # Cruelty by husband (≈ IPC 498A)
    ("BNS", "80"):   "MEDIUM",  # Hurt (≈ IPC 323)
    ("BNS", "115"):  "MEDIUM",  # Voluntarily causing hurt

    # ── NI Act ───────────────────────────────────────────────────────────────
    ("NI_ACT", "138"):  "HIGH",   # Dishonour of cheque — criminal liability
    ("NI_ACT", "141"):  "HIGH",   # Offences by companies
    ("NI_ACT", "142"):  "MEDIUM", # Cognizance of offences

    # ── POCSO ────────────────────────────────────────────────────────────────
    ("POCSO", "3"):   "HIGH",   # Penetrative sexual assault
    ("POCSO", "4"):   "HIGH",   # Punishment for penetrative sexual assault
    ("POCSO", "5"):   "HIGH",   # Aggravated penetrative sexual assault
    ("POCSO", "6"):   "HIGH",   # Punishment for aggravated
    ("POCSO", "7"):   "HIGH",   # Sexual assault
    ("POCSO", "8"):   "HIGH",   # Punishment for sexual assault
    ("POCSO", "11"):  "HIGH",   # Sexual harassment of child
    ("POCSO", "12"):  "HIGH",   # Punishment for sexual harassment

    # ── NDPS ─────────────────────────────────────────────────────────────────
    ("NDPS", "8"):    "HIGH",   # Prohibition of certain operations
    ("NDPS", "20"):   "HIGH",   # Punishment for cannabis
    ("NDPS", "21"):   "HIGH",   # Punishment for manufactured drugs
    ("NDPS", "22"):   "HIGH",   # Punishment for psychotropic substances
    ("NDPS", "25"):   "HIGH",   # Punishment for allowing premises to be used
    ("NDPS", "27"):   "MEDIUM", # Punishment for consumption
    ("NDPS", "37"):   "HIGH",   # Offences to be cognizable and non-bailable
    ("NDPS", "50"):   "MEDIUM", # Conditions under which search of persons shall be conducted

    # ── PC Act (Prevention of Corruption) ────────────────────────────────────
    ("PC_ACT", "7"):   "HIGH",  # Offence relating to public servant being bribed
    ("PC_ACT", "11"):  "HIGH",  # Public servant obtaining valuable thing without consideration
    ("PC_ACT", "13"):  "HIGH",  # Criminal misconduct by public servant

    # ── UAPA ─────────────────────────────────────────────────────────────────
    ("UAPA", "13"):  "HIGH",    # Punishment for unlawful activities
    ("UAPA", "16"):  "HIGH",    # Punishment for terrorist act
    ("UAPA", "17"):  "HIGH",    # Punishment for raising funds for terrorist act
    ("UAPA", "18"):  "HIGH",    # Punishment for conspiracy
    ("UAPA", "38"):  "HIGH",    # Offence relating to membership of terrorist organisation
    ("UAPA", "39"):  "HIGH",    # Offence relating to support for terrorist organisation

    # ── DV Act ───────────────────────────────────────────────────────────────
    ("DV_ACT", "18"): "HIGH",   # Protection orders
    ("DV_ACT", "19"): "HIGH",   # Residence orders
    ("DV_ACT", "31"): "HIGH",   # Penalty for breach of protection order

    # ── Dowry Prohibition Act ─────────────────────────────────────────────────
    ("DOWRY_ACT", "3"):  "HIGH",  # Penalty for giving or taking dowry
    ("DOWRY_ACT", "4"):  "HIGH",  # Penalty for demanding dowry

    # ── Arms Act ─────────────────────────────────────────────────────────────
    ("ARMS_ACT", "25"): "HIGH",   # Punishment for certain offences
    ("ARMS_ACT", "27"): "HIGH",   # Punishment for using arms

    # ── IT Act ───────────────────────────────────────────────────────────────
    ("IT_ACT", "66"):   "HIGH",   # Computer related offences
    ("IT_ACT", "66A"):  "HIGH",   # Punishment for sending offensive messages (note: struck down but still cited)
    ("IT_ACT", "66C"):  "HIGH",   # Identity theft
    ("IT_ACT", "66D"):  "HIGH",   # Cheating by personation
    ("IT_ACT", "66E"):  "HIGH",   # Violation of privacy
    ("IT_ACT", "67"):   "HIGH",   # Publishing obscene material
    ("IT_ACT", "67B"):  "HIGH",   # Publishing child sexual abuse material
    ("IT_ACT", "43"):   "MEDIUM", # Penalty for damage to computer system

    # ── SC/ST Act ─────────────────────────────────────────────────────────────
    ("SC_ST_ACT", "3"):   "HIGH",  # Atrocities
    ("SC_ST_ACT", "3A"):  "HIGH",  # Offences after 2015 amendment
    ("SC_ST_ACT", "14A"): "HIGH",  # Appeals

    # ── CPC ──────────────────────────────────────────────────────────────────
    ("CPC", "9"):    "MEDIUM",  # Courts to try all civil suits
    ("CPC", "80"):   "MEDIUM",  # Notice to government before suit
    ("CPC", "151"):  "MEDIUM",  # Inherent powers of court

    # ── TP Act ───────────────────────────────────────────────────────────────
    ("TP_ACT", "53A"): "MEDIUM", # Part performance
    ("TP_ACT", "54"):  "MEDIUM", # Sale defined
    ("TP_ACT", "58"):  "MEDIUM", # Mortgage defined

    # ── Consumer Protection Act 2019 ─────────────────────────────────────────
    ("CONSUMER_ACT", "88"): "MEDIUM", # Penalty for failure to comply with order
    ("CONSUMER_ACT", "89"): "HIGH",   # Penalty for false/misleading advertisement

    # ── Income Tax Act ───────────────────────────────────────────────────────
    ("INCOME_TAX_ACT", "271"):  "MEDIUM", # Penalty for concealment
    ("INCOME_TAX_ACT", "276C"): "HIGH",   # Prosecution for willful attempt to evade tax
    ("INCOME_TAX_ACT", "276B"): "HIGH",   # Failure to pay TDS

    # ── Motor Vehicles Act ───────────────────────────────────────────────────
    ("MV_ACT", "184"):  "HIGH",   # Driving dangerously
    ("MV_ACT", "185"):  "HIGH",   # Driving under influence of alcohol/drugs
    ("MV_ACT", "304A"): "HIGH",   # Causing death by negligence (IPC applied via MV cases)

    # ── Companies Act ────────────────────────────────────────────────────────
    ("COMPANIES_ACT", "447"): "HIGH",   # Punishment for fraud
    ("COMPANIES_ACT", "448"): "HIGH",   # Punishment for false statement
    ("COMPANIES_ACT", "449"): "HIGH",   # Punishment for false evidence

    # ── IBC ──────────────────────────────────────────────────────────────────
    ("IBC", "74"): "HIGH",    # Punishment for offences by insolvency professional
    ("IBC", "76"): "HIGH",    # Punishment for concealment of property

    # ── Constitution of India ─────────────────────────────────────────────────
    # Articles, not sections — but extracted under ipc_sections by NER
    # These themselves aren't risk signals; writ petitions invoking these
    # are handled by doc_type (hc_judgment, sc_judgment) base risk
}


# ── Keyword risk patterns ─────────────────────────────────────────────────────
_RISK_KEYWORDS_HIGH: list[re.Pattern] = [
    re.compile(r'\b(non[- ]bailable|life imprisonment|death penalty|capital punishment)\b', re.IGNORECASE),
    re.compile(r'\b(warrant of arrest|remand|judicial custody|arrested|taken into custody)\b', re.IGNORECASE),
    re.compile(r'\b(chargesheet|charge sheet|cognizance taken|committed to sessions)\b', re.IGNORECASE),
]

_RISK_KEYWORDS_MEDIUM: list[re.Pattern] = [
    re.compile(r'\b(bailable offence|show cause|contempt of court)\b', re.IGNORECASE),
    re.compile(r'\b(civil suit|recovery proceedings|attachment of property|injunction)\b', re.IGNORECASE),
    re.compile(r'\b(termination|eviction|dispossession|penalty|forfeiture)\b', re.IGNORECASE),
]

_RISK_AMOUNTS_HIGH   = 1_000_000   # > 10 lakh
_RISK_AMOUNTS_MEDIUM = 50_000     # > 50 thousand


# ── Recommended actions ───────────────────────────────────────────────────────
_ACTIONS: dict[str, dict[str, list[str]]] = {
    "fir": {
        "Critical": ["Engage a criminal lawyer immediately — do not speak to police without counsel",
                     "Apply for anticipatory bail before appearing before police",
                     "Do not surrender documents or make statements unilaterally"],
        "High":     ["Consult a criminal lawyer within 24 hours",
                     "Do not make any statements to police without lawyer present",
                     "Apply for anticipatory bail if arrest is likely"],
        "Medium":   ["Consult a criminal lawyer within 48 hours",
                     "Gather documentary evidence supporting your version"],
        "Low":      ["Keep a copy of the FIR", "Monitor case status at police station"],
    },
    "cheque_bounce_notice": {
        "Critical": ["Pay immediately or engage lawyer to negotiate settlement today",
                     "Criminal prosecution under Section 138 NI Act can begin after 15 days",
                     "If unable to pay, explore compounding of offence with payee"],
        "High":     ["Pay the cheque amount within 15 days to avoid criminal prosecution",
                     "Consult a lawyer about Section 138 NI Act liability immediately"],
        "Medium":   ["Respond to notice within 15 days", "Consult lawyer about notice validity"],
        "Low":      ["Acknowledge receipt and respond in writing"],
    },
    "court_notice_summons": {
        "High":     ["Engage a lawyer immediately and appear on the date mentioned",
                     "Non-appearance results in ex-parte proceedings and possible warrant"],
        "Medium":   ["Consult a lawyer to prepare your response",
                     "File counter-affidavit if required"],
        "Low":      ["Appear before court on the specified date with all relevant documents"],
    },
    "legal_notice": {
        "High":     ["Reply through your lawyer within the stipulated time — silence is harmful",
                     "Gather all evidence related to the dispute immediately"],
        "Medium":   ["Consult a lawyer to draft a reply", "Do not ignore the notice"],
        "Low":      ["Acknowledge receipt", "Consider amicable resolution"],
    },
    "bail_application": {
        "High":     ["Engage experienced criminal lawyer for bail hearing",
                     "Arrange sureties and surety documents in advance",
                     "Compile documents proving roots in community and no flight risk"],
        "Medium":   ["Consult a lawyer about bail conditions",
                     "Prepare documents for surety"],
        "Low":      ["Appear before court as directed"],
    },
    "default": {
        "Critical": ["Consult a qualified criminal lawyer immediately"],
        "High":     ["Consult a qualified advocate immediately",
                     "Do not sign or agree to anything without legal advice"],
        "Medium":   ["Consult a lawyer before taking any action", "Keep all documents safely"],
        "Low":      ["Read the document carefully",
                     "Consult a lawyer if any clause is unclear"],
    },
}


def _get_actions(doc_type: str, level: str) -> list[str]:
    actions = _ACTIONS.get(doc_type, _ACTIONS["default"])
    return actions.get(level, _ACTIONS["default"].get(level, []))


def _score_to_level(score: float) -> str:
    if score >= 0.86: return "Critical"
    if score >= 0.66: return "High"
    if score >= 0.36: return "Medium"
    return "Low"


def _to_0_100(raw: float) -> int:
    """
    Normalize any raw risk score to an integer in [0, 100].

    Range detection is automatic:
      - raw in [0.0, 1.0]   → scale ×100  (standard probability / fraction)
      - raw in (1, 100]     → treat as already percentage-scaled, use as-is
      - raw > 100           → normalize against a fixed ceiling of 10_000
                              (observed scores up to ~5500; ceiling gives headroom)
                              e.g. 5500 → 55, 3500 → 35, 10000 → 100

    The result is always clamped to [0, 100] and cast to int.
    """
    _MAX_RAW = 10_000.0
    if raw <= 0.0:
        return 0
    if raw <= 1.0:
        # [0, 1] float range — standard scorer output
        normalized = raw * 100.0
    elif raw <= 100.0:
        # (1, 100] — already percentage-scaled
        normalized = raw
    else:
        # > 100 — normalize proportionally against fixed ceiling
        normalized = (raw / _MAX_RAW) * 100.0
    return int(min(max(round(normalized), 0), 100))


def _extract_section_number(section_str: str) -> str:
    """Extract normalized section number from NER output like 'Section 302' -> '302'."""
    m = re.search(r'(\d+[A-Z]?)', section_str.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _normalize_monetary(amt_str: str) -> float:
    """Convert monetary string to float rupee amount."""
    try:
        amt_str = amt_str.replace(",", "")
        nums    = re.findall(r'[\d.]+', amt_str)
        if not nums:
            return 0.0
        amount = float(nums[0])
        if "crore" in amt_str.lower():
            amount *= 10_000_000
        elif "lakh" in amt_str.lower():
            amount *= 100_000
        return amount
    except Exception:
        return 0.0


# ── Main scorer ───────────────────────────────────────────────────────────────

class RiskScorer:

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def score(
        self,
        text:     str,
        doc_type: str,
        entities: Optional[dict] = None,
        use_llm:  Optional[bool] = None,
    ) -> RiskResult:

        # ── Non-legal early exit ──────────────────────────────────────────
        if doc_type == "non_legal":
            return RiskResult(
                score               = 0,
                level               = "Low",
                factors             = ["This does not appear to be a legal document"],
                recommended_actions = [],
                doc_type_risk       = 0.0,
                entity_risk         = 0.0,
                llm_risk            = None,
                method              = "rejected",
            )

        should_use_llm = use_llm if use_llm is not None else self.use_llm
        entities       = entities or {}
        factors        = []

        # ── Layer 1: Document type base risk ──────────────────────────────────
        doc_risk = _DOC_TYPE_BASE_RISK.get(doc_type, 0.50)
        if doc_risk >= 0.66:
            factors.append(f"Document type '{doc_type.replace('_', ' ')}' carries inherent high legal risk")
        elif doc_risk >= 0.40:
            factors.append(f"Document type '{doc_type.replace('_', ' ')}' carries moderate legal risk")

        # ── Layer 2: Section + Act cross-reference ────────────────────────────
        entity_risk    = 0.0
        entity_factors = []

        if doc_type == "consumer_complaint":
            entity_risk = max(entity_risk, 0.40)
            factors.append("Consumer dispute — Medium risk floor applied")

        ipc_sections = entities.get("ipc_sections", [])  # e.g. ["Section 302", "Section 376"]
        acts_raw     = entities.get("acts", [])           # e.g. ["Indian Penal Code", "NDPS Act"]

        # Normalize all acts found in document
        canonical_acts: list[str] = []
        for act_str in acts_raw:
            canonical = _normalize_act(act_str)
            if canonical:
                canonical_acts.append(canonical)

        # If no acts extracted from NER, infer from doc_type as fallback
        if not canonical_acts:
            canonical_acts = _infer_acts_from_doctype(doc_type)

        # Cross-reference each section against each act
        for section_str in ipc_sections:
            sec_num = _extract_section_number(section_str)
            if not sec_num:
                continue

            matched = False
            for act in canonical_acts:
                key       = (act, sec_num)
                risk_level = _SECTION_RISK.get(key)
                if risk_level == "HIGH":
                    entity_risk = max(entity_risk, 0.82)
                    entity_factors.append(
                        f"High-risk provision: Section {sec_num} of {act.replace('_', ' ')}"
                    )
                    matched = True
                    break
                elif risk_level == "MEDIUM":
                    entity_risk = max(entity_risk, 0.50)
                    entity_factors.append(
                        f"Medium-risk provision: Section {sec_num} of {act.replace('_', ' ')}"
                    )
                    matched = True

            # If section found but act not in canonical list, flag ambiguously
            if not matched and sec_num and canonical_acts:
                # Check if this section is HIGH risk in ANY act we know
                high_in_any = [
                    act for act in canonical_acts
                    if _SECTION_RISK.get((act, sec_num)) == "HIGH"
                ]
                if high_in_any:
                    entity_risk = max(entity_risk, 0.70)
                    entity_factors.append(
                        f"Potentially high-risk section {sec_num} "
                        f"(matched in {high_in_any[0].replace('_', ' ')})"
                    )

        # ── Layer 3: Monetary amounts ─────────────────────────────────────────
        for amt_str in entities.get("monetary", []):
            amount = _normalize_monetary(amt_str)
            if amount >= _RISK_AMOUNTS_HIGH:
                entity_risk = max(entity_risk, 0.60)
                entity_factors.append(f"High monetary amount involved: {amt_str}")
            elif amount >= _RISK_AMOUNTS_MEDIUM:
                entity_risk = max(entity_risk, 0.38)
                entity_factors.append(f"Significant monetary amount: {amt_str}")

        # ── Layer 4: Text keyword signals ─────────────────────────────────────
        keyword_boost = 0.0
        for pat in _RISK_KEYWORDS_HIGH:
            if pat.search(text[:5000]):
                keyword_boost = max(keyword_boost, 0.20)
                factors.append("High-risk legal term detected in document")
                break
        for pat in _RISK_KEYWORDS_MEDIUM:
            if pat.search(text[:5000]):
                keyword_boost = max(keyword_boost, 0.10)
                break

        factors.extend(entity_factors)
        entity_risk = min(entity_risk + keyword_boost, 1.0)

        # ── Layer 5: Combine ──────────────────────────────────────────────────
        if entity_factors:
            rule_score = 0.40 * doc_risk + 0.60 * entity_risk
        else:
            rule_score = 0.80 * doc_risk + 0.20 * entity_risk

        rule_score = min(rule_score, 1.0)
        method     = "rule_based"
        llm_risk   = None

        # ── Layer 6: LLM (high-stakes docs only) ─────────────────────────────
        run_llm = (
            should_use_llm
            and doc_type in {"fir", "cheque_bounce_notice", "bail_application",
                             "court_notice_summons", "sc_judgment", "hc_judgment"}
            and len(text.strip()) > 200
        )
        if run_llm:
            llm_risk = self._llm_score(text, doc_type)
            if llm_risk is not None:
                rule_score = 0.60 * rule_score + 0.40 * llm_risk
                method     = "rule_llm_hybrid"

        final_score = _to_0_100(min(rule_score, 1.0))
        level       = _score_to_level(final_score / 100.0)

        if not factors:
            factors.append("No specific high-risk signals detected in document")

        return RiskResult(
            score               = final_score,
            level               = level,
            factors             = factors,
            recommended_actions = _get_actions(doc_type, level),
            doc_type_risk       = round(doc_risk, 3),
            entity_risk         = round(entity_risk, 3),
            llm_risk            = round(llm_risk, 3) if llm_risk else None,
            method              = method,
        )

    def _llm_score(self, text: str, doc_type: str) -> Optional[float]:
        try:
            from rag.llm import llm
            import json

            prompt = (
                f"You are a legal risk assessment AI for Indian law.\n\n"
                f"Analyze this {doc_type.replace('_', ' ')} document excerpt and assess "
                f"the legal risk to the person named or receiving it.\n\n"
                f"Document:\n{text[:1500]}\n\n"
                f"Respond ONLY with this JSON (no markdown, no extra text):\n"
                f'{{ "risk_score": 0.75, "key_risk": "one sentence reason" }}\n\n'
                f"risk_score: float 0.0 (no risk) to 1.0 (critical risk)."
            )
            response = llm.generate(prompt, max_tokens=80, temperature=0.1)
            m        = re.search(r'\{[^}]+\}', response)
            if m:
                data  = json.loads(m.group(0))
                score = float(data.get("risk_score", 0.5))
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.exception("LLM risk scoring failed")
        return None


def _infer_acts_from_doctype(doc_type: str) -> list[str]:
    """
    Fallback: if NER finds no acts, infer likely acts from document type.
    This prevents false negatives when NER misses the act name.
    """
    _DOCTYPE_ACT_MAP: dict[str, list[str]] = {
        "fir":                  ["IPC", "BNS", "CrPC", "BNSS"],
        "bail_application":     ["IPC", "BNS", "CrPC", "BNSS", "NDPS", "POCSO"],
        "cheque_bounce_notice": ["NI_ACT"],
        "court_notice_summons": ["CrPC", "CPC", "BNSS"],
        "legal_notice":         ["CONTRACT_ACT", "IPC", "CPC"],
        "consumer_complaint":   ["CONSUMER_ACT"],
        "employment_contract":  ["ID_ACT", "WAGES_ACT"],
        "rental_agreement":     ["KERALA_RENT_ACT", "CONTRACT_ACT", "TP_ACT"],
        "property_deed":        ["TP_ACT", "REGISTRATION_ACT"],
        "loan_agreement":       ["CONTRACT_ACT", "NI_ACT"],
        "sc_judgment":          ["IPC", "BNS", "CPC", "CONSTITUTION"],
        "hc_judgment":          ["IPC", "BNS", "CPC", "CONSTITUTION"],
    }
    return _DOCTYPE_ACT_MAP.get(doc_type, ["IPC", "CrPC"])


# ── DocumentRisk result (for api/document.py analyze endpoint) ────────────────

@dataclass
class DocumentClauseRisk:
    """Per-clause risk result returned by score_document()."""
    clause_number: int
    clause_text:   str
    score:         int          # 0-100
    risk_level:    str
    flags:         list[str] = field(default_factory=list)
    legal_refs:    list[str] = field(default_factory=list)
    explanation:   str        = ""

    def to_dict(self) -> dict:
        return {
            "clause_number": self.clause_number,
            "clause_text":   self.clause_text,
            "score":         self.score,
            "risk_level":    self.risk_level,
            "flags":         self.flags,
            "legal_refs":    self.legal_refs,
            "explanation":   self.explanation,
        }


@dataclass
class DocumentRisk:
    """Structured risk result for document analysis endpoint."""
    overall_score:   int              # 0-100
    risk_level:      str
    high_risk_count: int
    summary:         str
    clause_risks:    list[DocumentClauseRisk] = field(default_factory=list)


# ── Clause-level risk patterns for document analysis ─────────────────────────
_CLAUSE_RISK_PATTERNS: list[tuple[re.Pattern, str, str, list[str], list[str]]] = [
    # (pattern, flag, explanation, legal_refs, risk_keyword for level)
    (re.compile(r'\bnon[- ]refundable\b', re.IGNORECASE),
     "NON_REFUNDABLE_DEPOSIT",
     "Non-refundable deposit/payment clauses may be void under Indian law.",
     ["Section 74 Indian Contract Act", "State Rent Control Acts"],
     "high"),
    (re.compile(r'\bunilateral\b.*\btermination\b|\bwithout\s+notice\b', re.IGNORECASE),
     "UNILATERAL_TERMINATION",
     "Unilateral or no-notice termination clauses may be challenged under Indian contract law.",
     ["Section 73-74 Indian Contract Act"],
     "high"),
    (re.compile(r'\bindemnify\b.*\ball.*\blosses\b|\bindemnify.*harmless\b', re.IGNORECASE),
     "BROAD_INDEMNITY",
     "Broad indemnity clauses expose a party to unlimited liability.",
     ["Section 124-125 Indian Contract Act"],
     "high"),
    (re.compile(r'\bnon[- ]compete\b|\bnot\s+to\s+compete\b', re.IGNORECASE),
     "NON_COMPETE",
     "Non-compete clauses are generally unenforceable under Section 27 of the Indian Contract Act.",
     ["Section 27 Indian Contract Act"],
     "medium"),
    (re.compile(r'\barbitration\b.*\bDelhi\b|\barbitration\s+clause\b', re.IGNORECASE),
     "ARBITRATION_CLAUSE",
     "Arbitration clause detected. Ensure venue/jurisdiction is acceptable.",
     ["Arbitration and Conciliation Act 1996"],
     "medium"),
    (re.compile(r'\bpenalty\b.*\b(\d+)\s*(?:times?|x)\b|\bliquidated\s+damages\b', re.IGNORECASE),
     "PENALTY_CLAUSE",
     "Penalty/liquidated damages clause detected. May be subject to court reduction.",
     ["Section 74 Indian Contract Act"],
     "medium"),
    (re.compile(r'\bwithout\s+prejudice\b', re.IGNORECASE),
     "WITHOUT_PREJUDICE",
     "Without-prejudice clause restricts admissibility of communications.",
     ["Indian Evidence Act"],
     "low"),
    (re.compile(r'\bforce\s+majeure\b', re.IGNORECASE),
     "FORCE_MAJEURE",
     "Force majeure clause detected. Check scope and notice requirements.",
     ["Section 56 Indian Contract Act"],
     "low"),
]


def _score_clause(text: str) -> tuple[int, str, list[str], list[str], str]:
    """Score a clause 0-100, returning (score, level, flags, legal_refs, explanation)."""
    score    = 0
    flags    = []
    refs     = []
    explain  = ""
    highest  = "low"

    for pattern, flag, exp, legal_refs, level in _CLAUSE_RISK_PATTERNS:
        if pattern.search(text):
            flags.append(flag)
            refs.extend(legal_refs)
            explain = exp
            if level == "high":
                score = max(score, 80)
                highest = "high"
            elif level == "medium":
                score = max(score, 50)
                if highest not in ("high",):
                    highest = "medium"
            elif level == "low":
                score = max(score, 25)

    if score == 0:
        return 0, "low", [], [], ""
    return score, highest, flags, list(dict.fromkeys(refs)), explain


class RiskScorer:

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def score(
        self,
        text:     str,
        doc_type: str,
        entities: Optional[dict] = None,
        use_llm:  Optional[bool] = None,
    ) -> RiskResult:

        # ── Non-legal early exit ──────────────────────────────────────────
        if doc_type == "non_legal":
            return RiskResult(
                score               = 0,
                level               = "Low",
                factors             = ["This does not appear to be a legal document"],
                recommended_actions = [],
                doc_type_risk       = 0.0,
                entity_risk         = 0.0,
                llm_risk            = None,
                method              = "rejected",
            )

        should_use_llm = use_llm if use_llm is not None else self.use_llm
        entities       = entities or {}
        factors        = []

        # ── Layer 1: Document type base risk ──────────────────────────────────
        doc_risk = _DOC_TYPE_BASE_RISK.get(doc_type, 0.50)
        if doc_risk >= 0.66:
            factors.append(f"Document type '{doc_type.replace('_', ' ')}' carries inherent high legal risk")
        elif doc_risk >= 0.40:
            factors.append(f"Document type '{doc_type.replace('_', ' ')}' carries moderate legal risk")

        # ── Layer 2: Section + Act cross-reference ────────────────────────────
        entity_risk    = 0.0
        entity_factors = []

        if doc_type == "consumer_complaint":
            entity_risk = max(entity_risk, 0.40)
            factors.append("Consumer dispute — Medium risk floor applied")

        ipc_sections = entities.get("ipc_sections", [])
        acts_raw     = entities.get("acts", [])

        canonical_acts: list[str] = []
        for act_str in acts_raw:
            canonical = _normalize_act(act_str)
            if canonical:
                canonical_acts.append(canonical)

        if not canonical_acts:
            canonical_acts = _infer_acts_from_doctype(doc_type)

        for section_str in ipc_sections:
            sec_num = _extract_section_number(section_str)
            if not sec_num:
                continue

            matched = False
            for act in canonical_acts:
                key        = (act, sec_num)
                risk_level = _SECTION_RISK.get(key)
                if risk_level == "HIGH":
                    entity_risk = max(entity_risk, 0.82)
                    entity_factors.append(
                        f"High-risk provision: Section {sec_num} of {act.replace('_', ' ')}"
                    )
                    matched = True
                    break
                elif risk_level == "MEDIUM":
                    entity_risk = max(entity_risk, 0.50)
                    entity_factors.append(
                        f"Medium-risk provision: Section {sec_num} of {act.replace('_', ' ')}"
                    )
                    matched = True

            if not matched and sec_num and canonical_acts:
                high_in_any = [
                    act for act in canonical_acts
                    if _SECTION_RISK.get((act, sec_num)) == "HIGH"
                ]
                if high_in_any:
                    entity_risk = max(entity_risk, 0.70)
                    entity_factors.append(
                        f"Potentially high-risk section {sec_num} "
                        f"(matched in {high_in_any[0].replace('_', ' ')})"
                    )

        # ── Layer 3: Monetary amounts ─────────────────────────────────────────
        for amt_str in entities.get("monetary", []):
            amount = _normalize_monetary(amt_str)
            if amount >= _RISK_AMOUNTS_HIGH:
                entity_risk = max(entity_risk, 0.60)
                entity_factors.append(f"High monetary amount involved: {amt_str}")
            elif amount >= _RISK_AMOUNTS_MEDIUM:
                entity_risk = max(entity_risk, 0.38)
                entity_factors.append(f"Significant monetary amount: {amt_str}")

        # ── Layer 4: Text keyword signals ─────────────────────────────────────
        keyword_boost = 0.0
        for pat in _RISK_KEYWORDS_HIGH:
            if pat.search(text[:5000]):
                keyword_boost = max(keyword_boost, 0.20)
                factors.append("High-risk legal term detected in document")
                break
        for pat in _RISK_KEYWORDS_MEDIUM:
            if pat.search(text[:5000]):
                keyword_boost = max(keyword_boost, 0.10)
                break

        factors.extend(entity_factors)
        entity_risk = min(entity_risk + keyword_boost, 1.0)

        # ── Layer 5: Combine ──────────────────────────────────────────────────
        if entity_factors:
            rule_score = 0.40 * doc_risk + 0.60 * entity_risk
        else:
            rule_score = 0.80 * doc_risk + 0.20 * entity_risk

        rule_score = min(rule_score, 1.0)
        method     = "rule_based"
        llm_risk   = None

        # ── Layer 6: LLM (high-stakes docs only) ─────────────────────────────
        run_llm = (
            should_use_llm
            and doc_type in {"fir", "cheque_bounce_notice", "bail_application",
                             "court_notice_summons", "sc_judgment", "hc_judgment"}
            and len(text.strip()) > 200
        )
        if run_llm:
            llm_risk = self._llm_score(text, doc_type)
            if llm_risk is not None:
                rule_score = 0.60 * rule_score + 0.40 * llm_risk
                method     = "rule_llm_hybrid"

        final_score = _to_0_100(min(rule_score, 1.0))
        level       = _score_to_level(final_score / 100.0)

        if not factors:
            factors.append("No specific high-risk signals detected in document")

        return RiskResult(
            score               = final_score,
            level               = level,
            factors             = factors,
            recommended_actions = _get_actions(doc_type, level),
            doc_type_risk       = round(doc_risk, 3),
            entity_risk         = round(entity_risk, 3),
            llm_risk            = round(llm_risk, 3) if llm_risk else None,
            method              = method,
        )

    def score_document(
        self,
        text:     str,
        doc_type: str,
    ) -> DocumentRisk:
        """
        Clause-level risk scoring for the document analysis endpoint.
        Splits text into clauses, scores each with rule-based patterns,
        returns a DocumentRisk with overall_score, risk_level, clause_risks.
        """
        # ── Non-legal early exit ──────────────────────────────────────────
        if doc_type == "non_legal":
            return DocumentRisk(
                overall_score   = 0,
                risk_level      = "Low",
                high_risk_count = 0,
                summary         = "This does not appear to be a legal document",
                clause_risks    = [],
            )

        # Split into clauses (numbered clauses or double-newline paragraphs)
        clause_split = re.split(
            r'\n{2,}|\d+\.\s+(?=[A-Z])|(?:CLAUSE|ARTICLE|SECTION)\s+\d+\.',
            text[:8000],
        )
        clauses = [c.strip() for c in clause_split if len(c.strip()) > 30][:20]

        clause_risks = []
        for i, clause in enumerate(clauses, start=1):
            score, level, flags, refs, explain = _score_clause(clause)
            if score > 0:
                clause_risks.append(DocumentClauseRisk(
                    clause_number = i,
                    clause_text   = clause[:200],
                    score         = score,
                    risk_level    = level,
                    flags         = flags,
                    legal_refs    = refs,
                    explanation   = explain,
                ))

        high_risk_count = sum(1 for c in clause_risks if c.risk_level in ("high", "critical"))

        # Overall score: max of clause scores or base doc risk
        base = _DOC_TYPE_BASE_RISK.get(doc_type, 0.5)
        max_clause_score = max((c.score for c in clause_risks), default=0) / 100.0
        overall_float    = max(base, max_clause_score)
        overall_score    = int(round(overall_float * 100))
        level            = _score_to_level(overall_float)

        if high_risk_count > 0:
            summary = f"{level.upper()} RISK: {high_risk_count} clause(s) contain high-risk terms. Review before signing."
        elif clause_risks:
            summary = f"{level.upper()} RISK: {len(clause_risks)} clause(s) flagged for review."
        else:
            summary = f"{level.upper()} RISK: No specific clause-level risk flags detected."

        return DocumentRisk(
            overall_score   = overall_score,
            risk_level      = level,
            high_risk_count = high_risk_count,
            summary         = summary,
            clause_risks    = clause_risks,
        )

    def _llm_score(self, text: str, doc_type: str) -> Optional[float]:
        try:
            from rag.llm import llm
            import json

            prompt = (
                f"You are a legal risk assessment AI for Indian law.\n\n"
                f"Analyze this {doc_type.replace('_', ' ')} document excerpt and assess "
                f"the legal risk to the person named or receiving it.\n\n"
                f"Document:\n{text[:1500]}\n\n"
                f"Respond ONLY with this JSON (no markdown, no extra text):\n"
                f'{{ "risk_score": 0.75, "key_risk": "one sentence reason" }}\n\n'
                f"risk_score: float 0.0 (no risk) to 1.0 (critical risk)."
            )
            response = llm.generate(prompt, max_tokens=80, temperature=0.1)
            m        = re.search(r'\{[^}]+\}', response)
            if m:
                data  = json.loads(m.group(0))
                score = float(data.get("risk_score", 0.5))
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.exception("LLM risk scoring failed")
        return None


def _infer_acts_from_doctype(doc_type: str) -> list[str]:
    """
    Fallback: if NER finds no acts, infer likely acts from document type.
    This prevents false negatives when NER misses the act name.
    """
    _DOCTYPE_ACT_MAP: dict[str, list[str]] = {
        "fir":                  ["IPC", "BNS", "CrPC", "BNSS"],
        "bail_application":     ["IPC", "BNS", "CrPC", "BNSS", "NDPS", "POCSO"],
        "cheque_bounce_notice": ["NI_ACT"],
        "court_notice_summons": ["CrPC", "CPC", "BNSS"],
        "legal_notice":         ["CONTRACT_ACT", "IPC", "CPC"],
        "consumer_complaint":   ["CONSUMER_ACT"],
        "employment_contract":  ["ID_ACT", "WAGES_ACT"],
        "rental_agreement":     ["KERALA_RENT_ACT", "CONTRACT_ACT", "TP_ACT"],
        "property_deed":        ["TP_ACT", "REGISTRATION_ACT"],
        "loan_agreement":       ["CONTRACT_ACT", "NI_ACT"],
        "sc_judgment":          ["IPC", "BNS", "CPC", "CONSTITUTION"],
        "hc_judgment":          ["IPC", "BNS", "CPC", "CONSTITUTION"],
    }
    return _DOCTYPE_ACT_MAP.get(doc_type, ["IPC", "CrPC"])


# ═══════════════════════════════════════════════════════════════════════════════
# PROACTIVE RIGHTS VIOLATION DETECTOR  (Task 3)
# ═══════════════════════════════════════════════════════════════════════════════
# Rule-based — no LLM, instant, zero extra API cost.
# Called by api/document.py after every /analyze request.

def detect_rights_violations(
    doc_type:   str,
    ner_result: dict,
    doc_text:   str,
) -> list[dict]:
    """
    Detect likely rights violations in a document based on its type and content.

    Returns a list of alert dicts:
      {
        "right":     str,   # name of the right
        "violation": str,   # description of the potential violation
        "section":   str,   # applicable legal provision
        "severity":  str    # "info" | "low" | "medium" | "high" | "critical"
      }

    All checks are rule-based (regex + keyword), with zero LLM cost.
    """
    alerts: list[dict] = []
    text_lower = doc_text.lower()

    # ── Rental agreement checks ───────────────────────────────────────────────
    if doc_type == "rental_agreement":

        # Check: excess security deposit (> 2 months rent)
        # Heuristic: if "security deposit" appears and a high amount is present
        # Monetary NER extracts amounts — compare deposit vs monthly rent mentions
        monetary = ner_result.get("monetary", [])
        if "security deposit" in text_lower and len(monetary) >= 2:
            # Extract all numbers from monetary strings
            amounts = []
            for m in monetary:
                nums = re.findall(r'[\d,]+', m.replace(",", ""))
                for n in nums:
                    try:
                        amounts.append(int(n))
                    except ValueError:
                        pass
            if amounts:
                max_amt = max(amounts)
                min_amt = min(amounts)
                # If max is more than 2× min, likely exceeds 2 months
                if min_amt > 0 and max_amt / min_amt > 2.5:
                    alerts.append({
                        "right":     "Excess security deposit",
                        "violation": "Deposit may exceed the legal limit of 2 months rent in most states.",
                        "section":   "State Rent Control Acts",
                        "severity":  "medium",
                    })

        # Check: no notice period for termination
        has_notice = any(kw in text_lower for kw in [
            "notice period", "termination notice", "days notice", "days' notice",
            "months notice", "months' notice", "written notice",
        ])
        if not has_notice:
            alerts.append({
                "right":     "Right to notice before eviction",
                "violation": "Agreement may not specify the legally required notice period before termination.",
                "section":   "Transfer of Property Act S.106",
                "severity":  "high",
            })

    # ── Employment contract checks ─────────────────────────────────────────────
    elif doc_type == "employment_contract":

        # Check: PF/EPF not mentioned
        has_pf = any(kw in text_lower for kw in [
            "provident fund", "epf", "pf contribution", "employees' provident",
            "employee provident", "pf act",
        ])
        if not has_pf:
            alerts.append({
                "right":     "Right to Provident Fund",
                "violation": "Contract does not mention EPF contribution as required by law.",
                "section":   "Employees' Provident Funds and Miscellaneous Provisions Act 1952",
                "severity":  "high",
            })

        # Check: non-compete clause
        has_non_compete = any(kw in text_lower for kw in [
            "non-compete", "non compete", "not compete", "not to compete",
            "restrictive covenant", "restraint of trade",
        ])
        if has_non_compete:
            alerts.append({
                "right":     "Right to employment after resignation",
                "violation": "Non-compete clause detected — such clauses are generally unenforceable in India.",
                "section":   "Indian Contract Act S.27",
                "severity":  "medium",
            })

        # Check: no gratuity mention for long-term contracts
        has_gratuity = any(kw in text_lower for kw in [
            "gratuity", "payment of gratuity",
        ])
        if not has_gratuity:
            alerts.append({
                "right":     "Right to Gratuity",
                "violation": "Contract does not mention gratuity entitlement (payable after 5 years of service).",
                "section":   "Payment of Gratuity Act 1972",
                "severity":  "low",
            })

    # ── Loan agreement checks ─────────────────────────────────────────────────
    elif doc_type == "loan_agreement":

        # Check: no foreclosure/prepayment clause
        has_prepay = any(kw in text_lower for kw in [
            "foreclosure", "prepayment", "pre-payment", "pre-closure",
            "part payment", "early repayment",
        ])
        if not has_prepay:
            alerts.append({
                "right":     "Right to prepay loan",
                "violation": "Agreement may not specify prepayment rights — you may be charged penalties.",
                "section":   "RBI Guidelines on Prepayment of Loans",
                "severity":  "medium",
            })

        # Check: no cooling-off period
        has_cooling = any(kw in text_lower for kw in [
            "cooling off", "cooling-off", "cancellation period", "right to cancel",
        ])
        if not has_cooling:
            alerts.append({
                "right":     "Right to cooling-off period",
                "violation": "Loan agreement may not provide a cooling-off / cancellation window.",
                "section":   "RBI Fair Practices Code for Lenders",
                "severity":  "low",
            })

    # ── Criminal / court documents ────────────────────────────────────────────
    if doc_type in ("fir", "court_notice_summons", "bail_application"):
        alerts.append({
            "right":     "Right to free legal aid",
            "violation": (
                "Criminal matter detected — if you cannot afford a lawyer, "
                "you are entitled to free legal aid under Article 39A."
            ),
            "section":   "Legal Services Authorities Act 1987 / Article 39A Constitution",
            "severity":  "info",
        })

    # ── Cheque bounce notice ──────────────────────────────────────────────────
    if doc_type == "cheque_bounce_notice":
        alerts.append({
            "right":     "Right to reply within 15 days",
            "violation": (
                "Cheque dishonour notice: you have 15 days from receipt to pay "
                "the amount — failing this triggers criminal prosecution under S.138."
            ),
            "section":   "Negotiable Instruments Act S.138",
            "severity":  "high",
        })

    return alerts


# ── Singleton ─────────────────────────────────────────────────────────────────
risk_scorer = RiskScorer(use_llm=True)