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


def _extract_section_number(section_str: str) -> str:
    """Extract normalized section number from NER output like 'Section 302' → '302'."""
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

        final_score = round(min(rule_score, 1.0), 3)
        level       = _score_to_level(final_score)

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
            logger.warning("LLM risk scoring failed: %s", e)
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


# ── Singleton ─────────────────────────────────────────────────────────────────
risk_scorer = RiskScorer(use_llm=True)