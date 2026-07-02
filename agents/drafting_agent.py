"""
LexShield AI — Drafting Agent (Session 3 — Full Implementation)
================================================================
Multi-turn, PostgreSQL-persisted, 6-stage workflow for generating
professional Indian legal complaint drafts.

Stage Flow:
  INIT -> CLARIFY -> RETRIEVE_SECTIONS -> IDENTIFY_AUTHORITY -> CONFIRM -> GENERATE -> DONE

8 Complaint Categories:
  wage_theft | illegal_eviction | cheque_bounce | consumer_complaint |
  fir_complaint | domestic_violence | employment_termination | loan_default

Architecture:
  - All draft state is stored in PostgreSQL (drafts table).
  - In-memory dict is NOT used — every read/write goes to PostgreSQL.
  - This survives server restarts, unlike the previous in-memory approach.
  - graph.py calls has_active_draft() before intent routing.
  - draft_node calls handle() which dispatches by stage.

LLM:
  - Gemini 2.0 Flash (primary) — 1M tokens/day free.
  - Groq LLaMA 3.3 70B (fallback) — preserves quota.
  - Prompts are designed to produce real-world quality Indian legal documents.
"""

import json
import logging

logger = logging.getLogger(__name__)
import os
import re
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class FactExtractionResult(BaseModel):
    """Result of a single Groq JSON-mode fact-extraction call."""
    extracted_facts: dict[str, str]  # field_name -> extracted value
    missing_fields:  list[str]       # fields that could not be extracted
    confidence:      float           # 0.0 to 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class DraftStage(str, Enum):
    INIT               = "INIT"
    # ── New explicit stages (graph-aware) ─────────────────────────────────────
    MENU_SHOWN         = "MENU_SHOWN"          # menu presented, waiting for selection
    COLLECTING_FIELDS  = "COLLECTING_FIELDS"   # complaint type set, collecting fields
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"  # all fields done, confirm/cancel
    # ── Legacy internal stages (kept for backward-compat) ─────────────────────
    CLARIFY            = "CLARIFY"
    RETRIEVE_SECTIONS  = "RETRIEVE_SECTIONS"
    IDENTIFY_AUTHORITY = "IDENTIFY_AUTHORITY"
    CONFIRM            = "CONFIRM"
    GENERATE           = "GENERATE"
    DONE               = "DONE"


# ═══════════════════════════════════════════════════════════════════════════════
# DB SETUP
# ═══════════════════════════════════════════════════════════════════════════════

from agents.pg_sessions import get_conn


def _init_drafts_table() -> None:
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drafts (
                session_id  TEXT PRIMARY KEY,
                stage       TEXT NOT NULL,
                category    TEXT NOT NULL,
                draft_data  TEXT NOT NULL,
                created_at  DOUBLE PRECISION NOT NULL,
                updated_at  DOUBLE PRECISION NOT NULL
            )
        """)


_init_drafts_table()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLAINT CATEGORIES — KEYWORD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_CATEGORY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r'\b(wage\s+theft|salary\s+(not\s+paid|withheld|pending|dues)|unpaid\s+(wages|salary)|'
        r'payment\s+of\s+wages|labour\s+dues|pf\s+not\s+deposited)\b', re.IGNORECASE),
        "wage_theft"),

    (re.compile(
        r'\b(illegal\s+eviction|forcible\s+(eviction|dispossession)|unlawful\s+eviction|'
        r'evict(ed|ion)|thrown\s+out\s+of|locked\s+out\s+of|dispossessed)\b', re.IGNORECASE),
        "illegal_eviction"),

    (re.compile(
        r'\b(cheque\s+bounce|cheque\s+dishonour(ed)?|section\s+138|ni\s+act|'
        r'negotiable\s+instruments?|bounced?\s+cheque|dishonoured?\s+cheque)\b', re.IGNORECASE),
        "cheque_bounce"),

    (re.compile(
        r'\b(consumer\s+complaint|defective\s+(product|goods)|deficiency\s+in\s+service|'
        r'consumer\s+forum|consumer\s+court|consumer\s+protection|product\s+defect|'
        r'service\s+deficiency|unfair\s+trade\s+practice)\b', re.IGNORECASE),
        "consumer_complaint"),

    (re.compile(
        r'\b(fir\s+complaint|first\s+information\s+report|police\s+complaint|'
        r'theft\b|robbery|assault|fraud\s+complaint|criminal\s+complaint|'
        r'file\s+a?\s*fir|register\s+fir|file\s+complaint\s+(with\s+)?police)\b',
        re.IGNORECASE),
        "fir_complaint"),

    (re.compile(
        r'\b(domestic\s+violence|cruelty\s+by\s+husband|498[Aa]|dv\s+act|'
        r'domestic\s+abuse|matrimonial\s+cruelty|protection\s+order|'
        r'dowry\s+(harassment|demand|cruelty)|wife\s+beating|marital\s+violence)\b',
        re.IGNORECASE),
        "domestic_violence"),

    (re.compile(
        r'\b(wrongful\s+termination|illegal\s+termination|unfair\s+dismissal|'
        r'employment\s+termination|job\s+(terminated|dismissed|fired|sacked)|'
        r'terminated\s+from\s+(job|service|employment)|retrenchment|industrial\s+dispute)\b',
        re.IGNORECASE),
        "employment_termination"),

    (re.compile(
        r'\b(loan\s+default|emi\s+(default|not\s+paid|missed)|debt\s+recovery|'
        r'drt\b|bank\s+(loan|debt)|nbfc\s+complaint|recovery\s+agent|'
        r'loan\s+harassment|rbi\s+ombudsman)\b', re.IGNORECASE),
        "loan_default"),
]


def _detect_category(description: str) -> Optional[str]:
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(description):
            return category
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED FIELDS PER DOC TYPE  (used by _extract_facts for LLM extraction)
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS: dict[str, list[str]] = {
    "wage_theft": [
        "complainant_name", "employer_name", "employer_address",
        "unpaid_amount", "unpaid_period", "jurisdiction",
        "documentary_proof", "complainant_address", "complainant_contact", "designation", "date_of_joining",
    ],
    "illegal_eviction": [
        "complainant_name", "landlord_name", "property_address",
        "monthly_rent", "jurisdiction", "eviction_date",
        "complainant_address", "complainant_contact", "tenancy_duration",
    ],
    "cheque_bounce": [
        "complainant_name", "accused_name", "cheque_number",
        "cheque_amount", "dishonour_date", "dishonour_reason",
        "complainant_address", "complainant_contact", "accused_address", "complainant_bank",
    ],
    "consumer_complaint": [
        "complainant_name", "respondent_name", "product_service",
        "defect_description", "relief_sought", "jurisdiction",
        "complainant_address", "complainant_contact", "respondent_address",
    ],
    "fir_complaint": [
        "complainant_name", "accused_name_or_description",
        "incident_description", "incident_date",
        "incident_location", "sections_applicable",
        "complainant_address", "complainant_contact",
    ],
    "domestic_violence": [
        "complainant_name", "respondent_name", "relationship",
        "violence_type", "jurisdiction",
        "supporting_evidence", "complainant_address", "complainant_contact", "complainant_father_or_husband_name", "respondent_address", "relationship_start_year",
    ],
    "employment_termination": [
        "complainant_name", "employer_name", "jurisdiction",
        "termination_reason", "service_period", "last_salary",
        "complainant_address", "complainant_contact", "designation", "date_of_joining", "date_of_termination",
    ],
    "loan_default": [
        "complainant_name", "lender_name", "loan_amount",
        "loan_type", "defaulted_emis", "outstanding_amount",
        "complainant_address", "complainant_contact", "lender_address", "loan_date",
    ],
}

# Fields that are ALWAYS added to missing_fields regardless of what the LLM
# extracts.  These map to documentary-evidence questions in _CLARIFYING_QUESTIONS
# that must always be asked explicitly — the LLM cannot reliably extract
# "do you have documents?" intent from the user's initial query.
ALWAYS_ASK_FIELDS: dict[str, list[str]] = {
    "wage_theft":        ["documentary_proof"],
    "domestic_violence": ["supporting_evidence"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLARIFYING QUESTIONS  (asked one at a time, stored as list)
# ═══════════════════════════════════════════════════════════════════════════════

_CLARIFYING_QUESTIONS: dict[str, list[str]] = {
    "wage_theft": [
        "In which State and District is your employer located? (This determines the correct Labour Commissioner's jurisdiction.)",
        "Please provide the full name and complete address of your employer / company.",
        "What is the total amount of unpaid wages/salary, and for which period? (e.g., ₹45,000 for April–June 2024)",
        "Do you have any documentary proof? (e.g., appointment letter, salary slips, bank statement showing last salary credited, Form 16, or any written communication from employer)",
    ],
    "illegal_eviction": [
        "In which State and District is the property located?",
        "Is the property residential or commercial? Please provide its full address.",
        "What is the monthly rent amount and how was it paid? (cash / bank transfer / cheque — and do you have receipts?)",
        "Did the landlord give any written notice before eviction? If yes, what reason was stated and on what date?",
    ],
    "cheque_bounce": [
        "Please provide the cheque details: cheque number, amount (₹), date on cheque, name of issuing bank and branch.",
        "On what date was the cheque dishonoured, and what reason did the bank give? (e.g., 'funds insufficient', 'account closed', 'signature mismatch')",
        "What was the cheque issued for? (e.g., loan repayment, goods supplied, services rendered, security deposit)",
        "Have you already sent a demand notice to the drawer within 30 days of dishonour? If yes, on what date, and do you have the postal/courier receipt?",
    ],
    "consumer_complaint": [
        "What is the name of the product or service and the name of the company / seller / service provider?",
        "When was the purchase made, and what was the amount paid? (Please provide invoice/receipt number if available.)",
        "Describe the specific defect in the product or deficiency in the service clearly.",
        "Have you already lodged a complaint with the company? If yes, on what date and what was their response (or did they fail to respond)?",
    ],
    "fir_complaint": [
        "In which State and Police Station jurisdiction did the incident occur? (Provide district and nearest police station name.)",
        "What is the nature of the offence? (e.g., theft, house breaking, cheating/fraud, assault, extortion, criminal intimidation)",
        "What was the date, time, and exact location of the incident?",
        "Please provide the names and addresses of the accused, if known. (Write 'Not identified' if unknown.)",
    ],
    "domestic_violence": [
        "In which State and District do you currently reside?",
        "What is your relationship with the respondent (perpetrator)? (e.g., husband, father-in-law, mother-in-law, brother-in-law)",
        "What type of violence have you experienced? (physical / emotional / verbal / economic / sexual — you may specify more than one)",
        "Do you have any supporting evidence? (e.g., medical reports, photographs of injuries, witness names, police diary entries, written communications)",
    ],
    "employment_termination": [
        "In which State and District is your employer located?",
        "What reason, if any, was given for your termination? Was it communicated orally or in writing?",
        "What was your total period of service, designation, and last drawn monthly salary?",
        "Did you receive a termination letter, charge-sheet, or show-cause notice? Were proper retrenchment benefits or notice pay paid?",
    ],
    "loan_default": [
        "What is the full name of the lender? (Bank / NBFC / private individual — include branch name if applicable.)",
        "What was the total loan amount, type of loan (home/personal/business/vehicle), and date of sanction?",
        "How many EMIs have been defaulted, and what is the total outstanding amount as of today?",
        "Was any collateral, security, or guarantor provided? Have you received any recovery notice, SARFAESI notice, or legal notice from the lender?",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIELD -> QUESTION MAPPING  (direct per-category lookup — Bug 1 fix)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The old keyword-scan approach caused multiple fields (complainant_name,
# accused_name, cheque_number, cheque_amount) to all resolve to the same
# "cheque details" question — so Q1 was repeated 4 times in a row.
# This direct map gives every field its own unique question string.

_FIELD_QUESTION_MAP: dict[str, dict[str, str]] = {
    "wage_theft": {
        "jurisdiction":       "In which State and District is your employer located? (This determines the correct Labour Commissioner's jurisdiction.)",
        "complainant_name":   "What is your full name? (as it should appear on the complaint)",
        "employer_name":      "Please provide the full name of your employer / company.",
        "employer_address":   "Please provide the complete address of your employer / company.",
        "unpaid_amount":      "What is the total amount of unpaid wages/salary? (e.g., ₹45,000)",
        "unpaid_period":      "For which period are the wages unpaid? (e.g., April–June 2024)",
        "documentary_proof":  "Do you have any documentary proof? (e.g., appointment letter, salary slips, bank statement showing last salary credited, Form 16, or any written communication from employer)",
        "complainant_address": "What is your current residential address?",
        "complainant_contact": "What is your contact phone number or email?",
        "designation":         "What was your job title / designation?",
        "date_of_joining":     "On what date did you join this employer? (DD.MM.YYYY, or approximate if unsure)",
    },
    "illegal_eviction": {
        "jurisdiction":       "In which State and District is the property located?",
        "complainant_name":   "What is your full name? (as it should appear on the complaint)",
        "landlord_name":      "What is the full name of your landlord / property owner?",
        "property_address":   "Is the property residential or commercial? Please provide its full address.",
        "monthly_rent":       "What is the monthly rent amount and how was it paid? (cash / bank transfer / cheque — and do you have receipts?)",
        "eviction_date":      "Did the landlord give any written notice before eviction? If yes, what reason was stated and on what date?",
        "complainant_address": "What is your current residential address?",
        "complainant_contact": "What is your contact phone number or email?",
        "tenancy_duration":    "How long have you been a tenant at this property? (in years/months)",
    },
    "cheque_bounce": {
        "complainant_name":   "What is your full name? (the payee — person/company to whom the cheque was issued)",
        "accused_name":       "What is the full name of the cheque issuer (drawer/accused)?",
        "cheque_number":      "Please provide the cheque number and the name of the bank and branch it was drawn on.",
        "cheque_amount":      "What is the cheque amount (₹) and the date written on the cheque?",
        "dishonour_date":     "On what date was the cheque dishonoured (returned by the bank)?",
        "dishonour_reason":   "What reason did the bank give for dishonour? (e.g., 'funds insufficient', 'account closed', 'signature mismatch')",
        "complainant_address": "What is your current residential address?",
        "complainant_contact": "What is your contact phone number or email?",
        "accused_address":     "What is the full address of the cheque issuer (accused)?",
        "complainant_bank":    "What is the name of your bank where the cheque was deposited?",
    },
    "consumer_complaint": {
        "complainant_name":   "What is your full name? (as it should appear on the complaint)",
        "respondent_name":    "What is the name of the company / seller / service provider you are complaining against?",
        "product_service":    "What is the name of the product or service involved?",
        "defect_description": "Describe the specific defect in the product or deficiency in the service clearly.",
        "relief_sought":      "Have you already lodged a complaint with the company? If yes, on what date and what was their response?",
        "jurisdiction":       "In which State and District did you purchase the product / receive the service?",
        "complainant_address": "What is your current residential address?",
        "complainant_contact": "What is your contact phone number or email?",
        "respondent_address":  "What is the full address of the company / seller / service provider?",
    },
    "fir_complaint": {
        "complainant_name":          "What is your full name and contact address?",
        "incident_location":         "In which State and Police Station jurisdiction did the incident occur? (Provide district and nearest police station name.)",
        "incident_description":      "What is the nature of the offence? (e.g., theft, assault, cheating/fraud, extortion, criminal intimidation)",
        "incident_date":             "What was the date, time, and exact location of the incident?",
        "accused_name_or_description": "Please provide the names and addresses of the accused, if known. (Write 'Not identified' if unknown.)",
        "sections_applicable":       "Do you know which IPC/BNS sections apply? (Leave blank if unsure — we will identify them for you.)",
        "complainant_address":       "What is your current residential address?",
        "complainant_contact":       "What is your contact phone number or email?",
    },
    "domestic_violence": {
        "complainant_name":  "What is your full name? (as it should appear on the application)",
        "jurisdiction":      "In which State and District do you currently reside?",
        "respondent_name":   "What is the full name of the respondent (perpetrator)?",
        "relationship":      "What is your relationship with the respondent? (e.g., husband, father-in-law, mother-in-law, brother-in-law)",
        "violence_type":     "What type of violence have you experienced? (physical / emotional / verbal / economic / sexual — you may specify more than one)",
        "supporting_evidence": "Do you have any supporting evidence? (e.g., medical reports, photographs of injuries, witness names, police diary entries, written communications)",
        "complainant_address": "What is your current residential address?",
        "complainant_contact": "What is your contact phone number or email?",
        "complainant_father_or_husband_name": "What is the name of your father or husband? (Please specify which)",
        "respondent_address":  "What is the full address of the respondent?",
        "relationship_start_year": "In which year did your marriage or domestic relationship begin?",
    },
    "employment_termination": {
        "complainant_name":    "What is your full name? (as it should appear on the complaint)",
        "employer_name":       "What is the full name of your employer / company?",
        "jurisdiction":        "In which State and District is your employer located?",
        "termination_reason":  "What reason, if any, was given for your termination? Was it communicated orally or in writing?",
        "service_period":      "What was your total period of service and your designation?",
        "last_salary":         "What was your last drawn monthly salary (₹)?",
        "complainant_address": "What is your current residential address?",
        "complainant_contact": "What is your contact phone number or email?",
        "designation":         "What was your job title / designation?",
        "date_of_joining":     "On what date did you join this employer? (DD.MM.YYYY, or approximate if unsure)",
        "date_of_termination": "On what date were you terminated? (DD.MM.YYYY, or approximate if unsure)",
    },
    "loan_default": {
        "complainant_name":    "What is your full name? (as it should appear on the complaint)",
        "lender_name":         "What is the full name of the lender? (Bank / NBFC / private individual — include branch name if applicable.)",
        "loan_amount":         "What was the total loan amount (₹) sanctioned?",
        "loan_type":           "What type of loan is it? (home / personal / business / vehicle / gold / other)",
        "defaulted_emis":      "How many EMIs have been defaulted, and from which month?",
        "outstanding_amount":  "What is the total outstanding amount as of today (₹)?",
        "complainant_address": "What is your current residential address?",
        "complainant_contact": "What is your contact phone number or email?",
        "lender_address":      "What is the full address of the lender / branch?",
        "loan_date":           "On what date was the loan sanctioned or disbursed?",
    },
}

# Explicit overrides for always-ask fields (shared across categories)
_FIELD_QUESTION_OVERRIDES: dict[str, str] = {
    "documentary_proof": (
        "Do you have any documentary proof? "
        "(e.g., appointment letter, salary slips, bank statement showing last salary credited, "
        "Form 16, or any written communication from employer)"
    ),
    "supporting_evidence": (
        "Do you have any supporting evidence? "
        "(e.g., medical reports, photographs of injuries, witness names, "
        "police diary entries, written communications)"
    ),
}


def _field_question(field: str, category: str) -> str:
    """
    Return a human-readable question for a given REQUIRED_FIELDS field name.

    Resolution order:
      1. Per-category direct lookup in _FIELD_QUESTION_MAP — unique question per field.
      2. _FIELD_QUESTION_OVERRIDES — shared overrides for always-ask fields.
      3. Generic prettified fallback.
    """
    # 1. Direct per-category lookup (primary path — prevents question repetition)
    cat_map = _FIELD_QUESTION_MAP.get(category, {})
    if field in cat_map:
        return cat_map[field]

    # 2. Shared always-ask overrides
    override = _FIELD_QUESTION_OVERRIDES.get(field)
    if override:
        return override

    # 3. Generic fallback
    pretty = field.replace("_", " ").title()
    return f"Please provide the {pretty} for your {category.replace('_', ' ')} document."


# ═══════════════════════════════════════════════════════════════════════════════
# FILING AUTHORITY  (rule-based lookup — no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

_FILING_AUTHORITY: dict[str, str] = {
    "wage_theft": (
        "Labour Commissioner / Labour Court under the Payment of Wages Act, 1936 "
        "and the Industrial Disputes Act, 1947.\n"
        "— File Form L-1 / Application under Section 15, Payment of Wages Act "
        "at the Office of the District Labour Commissioner.\n"
        "— For amounts above ₹24,000/month: file under Section 33C, "
        "Industrial Disputes Act, 1947 before the Labour Court.\n"
        "— Additional remedy: File complaint with the Regional Provident Fund "
        "Commissioner if PF/ESI was not deposited."
    ),
    "illegal_eviction": (
        "Rent Controller / Civil Court under the Rent Control Act of the "
        "respective State and the Transfer of Property Act, 1882.\n"
        "— File an Objection Petition / Application for Restoration of Possession "
        "before the Rent Controller of the concerned district.\n"
        "— For interim relief (stay on eviction): File an application for "
        "Temporary Injunction under Order 39 Rules 1 & 2, CPC before the "
        "Civil Judge (Junior Division) of the district.\n"
        "— Police complaint under Section 441/442 BNS (trespass) may also be filed."
    ),
    "cheque_bounce": (
        "Judicial Magistrate First Class (JMFC) under Section 138 read with "
        "Section 142 of the Negotiable Instruments Act, 1881.\n"
        "— Jurisdiction: Court where the cheque was presented for payment, "
        "or where the payee resides/carries on business.\n"
        "— Mandatory pre-condition: Demand notice must be sent within 30 days "
        "of dishonour. Complaint must be filed within 30 days of expiry of "
        "15-day notice period.\n"
        "— Also file a complaint with the banker under RBI Cheque Bounce guidelines."
    ),
    "consumer_complaint": (
        "District Consumer Disputes Redressal Commission (DCDRC) under "
        "Section 35 of the Consumer Protection Act, 2019.\n"
        "— Pecuniary jurisdiction: Claims up to ₹50 lakh -> DCDRC; "
        "₹50 lakh to ₹2 crore -> State Consumer Disputes Redressal Commission (SCDRC); "
        "above ₹2 crore -> National Consumer Disputes Redressal Commission (NCDRC).\n"
        "— File at the DCDRC of the district where the opposite party resides, "
        "carries on business, or where the cause of action arose.\n"
        "— Online filing also available at: consumerhelpline.gov.in"
    ),
    "fir_complaint": (
        "Police Station (Station House Officer) for cognizable offences "
        "under Section 173 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS).\n"
        "— For cognizable offences (theft, assault, cheating, etc.): "
        "Police are bound to register FIR; refusal is grounds for complaint "
        "to SP/DIG or Magistrate under Section 175 BNSS.\n"
        "— For non-cognizable offences: File private complaint before the "
        "Judicial Magistrate First Class (JMFC) under Section 223 BNSS.\n"
        "— If police refuse to register FIR: File complaint before the "
        "Superintendent of Police (SP) or directly before the Chief Judicial "
        "Magistrate under Section 175(3) BNSS."
    ),
    "domestic_violence": (
        "Protection Officer / Judicial Magistrate First Class (JMFC) under "
        "the Protection of Women from Domestic Violence Act, 2005.\n"
        "— File Domestic Incident Report (DIR) with the Protection Officer "
        "of the concerned district.\n"
        "— File application for Protection Order / Residence Order / "
        "Monetary Relief before the JMFC under Section 12, DV Act.\n"
        "— Also file FIR under Section 498A IPC / Section 85 BNS "
        "(cruelty by husband/relatives) at the local police station.\n"
        "— For immediate safety: Contact One Stop Centre (OSC) — "
        "Toll Free: 181 (Women Helpline)."
    ),
    "employment_termination": (
        "Labour Court / Industrial Tribunal under the Industrial Disputes Act, 1947.\n"
        "— File Statement of Claim / Reference Application at the "
        "Office of the Labour Commissioner for conciliation under Section 12, IDA.\n"
        "— If conciliation fails: Matter is referred to Labour Court / "
        "Industrial Tribunal under Section 10, IDA.\n"
        "— Limitation: Must file within 3 years of termination.\n"
        "— For government employees: File before the Central / State "
        "Administrative Tribunal (CAT/SAT) as applicable.\n"
        "— Additional remedy under Section 25F IDA: Claim for retrenchment "
        "compensation if service exceeded 1 year."
    ),
    "loan_default": (
        "Debt Recovery Tribunal (DRT) for secured/unsecured loans above ₹20 lakh "
        "under the Recovery of Debts and Bankruptcy Act, 1993.\n"
        "— For loans below ₹20 lakh: File summary suit before Civil Court "
        "under Order 37, Code of Civil Procedure, 1908.\n"
        "— Against bank/NBFC harassment: File complaint with the "
        "RBI Integrated Ombudsman Scheme at cms.rbi.org.in (toll-free: 14448).\n"
        "— Against SARFAESI action: File application under Section 17, "
        "SARFAESI Act before DRT within 45 days of notice.\n"
        "— Insolvency option: Approach NCLT under Insolvency and Bankruptcy Code, 2016."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# RAG QUERIES PER CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════

_RAG_QUERIES: dict[str, str] = {
    "wage_theft":             "Payment of Wages Act 1936 Section 15 recovery of unpaid wages Industrial Disputes Act Section 33C",
    "illegal_eviction":       "illegal eviction Transfer of Property Act Rent Control Act tenant rights restoration of possession",
    "cheque_bounce":          "Section 138 Negotiable Instruments Act cheque dishonour demand notice criminal complaint procedure",
    "consumer_complaint":     "Consumer Protection Act 2019 Section 35 District Consumer Commission defect deficiency complaint procedure",
    "fir_complaint":          "Section 173 BNSS FIR registration cognizable offence police complaint procedure",
    "domestic_violence":      "Protection of Women from Domestic Violence Act 2005 Section 12 protection order Section 498A IPC",
    "employment_termination": "Industrial Disputes Act 1947 Section 25F retrenchment wrongful termination Labour Court claim",
    "loan_default":           "Debt Recovery Tribunal Recovery of Debts Bankruptcy Act SARFAESI Act Section 17 RBI Ombudsman",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORTING DOCUMENTS PER CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════

_SUPPORTING_DOCUMENTS: dict[str, list[str]] = {
    "wage_theft": [
        "Appointment letter / offer letter",
        "Salary slips for last 3–6 months (if available)",
        "Bank statements showing salary credits",
        "Form 16 / Income Tax returns (if applicable)",
        "Any written communication from employer regarding salary",
        "PF passbook / UAN details",
        "Identity proof (Aadhaar / PAN)",
        "Address proof",
    ],
    "illegal_eviction": [
        "Rent agreement / lease deed",
        "Rent receipts (last 6 months)",
        "Identity proof (Aadhaar / PAN)",
        "Photographs of the property and any damage",
        "Photographs/video of illegal eviction (if available)",
        "Written communications from landlord (WhatsApp messages / emails / letters)",
        "Proof of address at the rented premises",
        "Witness affidavits (if available)",
    ],
    "cheque_bounce": [
        "Original dishonoured cheque (or certified copy from bank)",
        "Bank return memo / cheque return slip with reason for dishonour",
        "Copy of demand notice sent to drawer",
        "Postal receipt / speed post acknowledgement / courier tracking proof",
        "Envelope of returned legal notice (if undelivered)",
        "Underlying agreement / invoice / receipt evidencing transaction",
        "Bank statement showing deposit of cheque",
        "Identity proof (Aadhaar / PAN)",
    ],
    "consumer_complaint": [
        "Purchase invoice / bill / receipt",
        "Warranty card / service agreement",
        "Photographs / video evidence of defective product / poor service",
        "Complaint made to company (email / letter / complaint number)",
        "Company's response (or proof of no response after 30 days)",
        "Expert opinion / inspection report (if applicable)",
        "Medical records (if injury caused by defective product)",
        "Identity proof (Aadhaar / PAN)",
    ],
    "fir_complaint": [
        "Detailed written complaint statement (this document)",
        "Photographs / video evidence of the incident (if available)",
        "Witness names, addresses, and contact details",
        "Medical report / injury certificate (if assault)",
        "Receipts / inventory of stolen property (if theft)",
        "Bank statements / transaction records (if fraud/cheating)",
        "Electronic evidence (emails, WhatsApp chats, call recordings — if applicable)",
        "Identity proof (Aadhaar / PAN)",
    ],
    "domestic_violence": [
        "Medical reports / injury certificates from government hospital",
        "Photographs of injuries",
        "Domestic Incident Report (DIR) filed with Protection Officer",
        "Witness statements / affidavits",
        "Any written or recorded threats / abusive communications",
        "Documents showing economic violence (bank statements, property documents)",
        "Marriage certificate",
        "Identity proof (Aadhaar / PAN)",
        "Address proof",
    ],
    "employment_termination": [
        "Appointment / offer letter",
        "Termination letter / charge-sheet / show-cause notice (if given)",
        "Last drawn salary slips (3–6 months)",
        "Resignation letter / acceptance (if voluntary)",
        "Service record / experience certificate",
        "Bank statements showing salary credits",
        "PF / gratuity payment proof or absence thereof",
        "Any written communications with employer regarding termination",
        "Identity proof (Aadhaar / PAN)",
    ],
    "loan_default": [
        "Loan sanction letter / loan agreement",
        "Repayment schedule / amortisation table",
        "Bank statements showing EMI payments and defaults",
        "SARFAESI notice / recovery notice received (if any)",
        "All correspondence with bank / NBFC",
        "Security / collateral documents (if applicable)",
        "Insurance policy on loan (if applicable)",
        "Identity proof (Aadhaar / PAN)",
        "Address proof",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD RELIEF SOUGHT PER CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════

_STANDARD_RELIEF: dict[str, str] = {
    "wage_theft": (
        "Recovery of unpaid wages/salary with interest at 6% p.a.; "
        "penalty under Section 20, Payment of Wages Act; "
        "compensation for wrongful withholding; direction to deposit PF/ESI."
    ),
    "illegal_eviction": (
        "Immediate restoration of possession; "
        "permanent injunction restraining illegal eviction; "
        "damages for loss caused by illegal dispossession; "
        "costs of litigation."
    ),
    "cheque_bounce": (
        "Conviction and imprisonment up to 2 years and/or fine up to twice the "
        "cheque amount under Section 138, Negotiable Instruments Act; "
        "recovery of the full cheque amount with interest and costs."
    ),
    "consumer_complaint": (
        "Replacement / refund of defective product or completion of deficient service; "
        "compensation for physical, mental, and financial suffering; "
        "cost of litigation; punitive damages if unfair trade practice established."
    ),
    "fir_complaint": (
        "Registration of FIR and investigation under appropriate sections; "
        "arrest of accused as warranted; recovery of stolen property; "
        "prosecution of accused before competent court."
    ),
    "domestic_violence": (
        "Protection Order under Section 18, DV Act; "
        "Residence Order under Section 19, DV Act; "
        "Monetary Relief under Section 20, DV Act; "
        "Custody Order under Section 21, DV Act (if applicable); "
        "Compensation Order under Section 22, DV Act; "
        "FIR under Section 498A IPC / Section 85 BNS."
    ),
    "employment_termination": (
        "Reinstatement with full back wages; "
        "alternatively, payment of retrenchment compensation under Section 25F, IDA; "
        "payment of notice pay, earned leave, gratuity; "
        "costs of proceedings."
    ),
    "loan_default": (
        "Stay on coercive recovery action; "
        "restructuring / rescheduling of loan repayment; "
        "damages for harassment by recovery agents; "
        "compliance with RBI Fair Practices Code and Fair Debt Collection guidelines."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# NEXT STEPS PER CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════

_NEXT_STEPS: dict[str, str] = {
    "wage_theft": (
        "Step 1: Print this complaint on plain paper. Attach copies of all documentary evidence listed above.\n"
        "Step 2: Submit two copies to the Office of the District Labour Commissioner. Obtain acknowledgement on your copy.\n"
        "Step 3: If no action in 30 days, escalate to the Labour Court or High Court under Article 226 of the Constitution."
    ),
    "illegal_eviction": (
        "Step 1: Print this complaint and get it notarised. Attach all supporting documents.\n"
        "Step 2: File before the Rent Controller / Civil Court. Simultaneously file an urgent application for interim injunction (stay on eviction).\n"
        "Step 3: File a police complaint at the local police station for criminal trespass under Section 441/442 BNS."
    ),
    "cheque_bounce": (
        "Step 1: Send the demand notice (separate document) by registered post AND speed post to the drawer immediately.\n"
        "Step 2: Preserve the postal receipt and tracking proof. If no payment in 15 days, file this complaint before the JMFC within 30 days.\n"
        "Step 3: File complaint at the JMFC court with all documents. Pay court fee. The case will be registered as CC (Criminal Complaint)."
    ),
    "consumer_complaint": (
        "Step 1: Send a legal notice to the company giving 15–30 days to resolve the issue. Keep postal proof.\n"
        "Step 2: If unresolved, file this complaint at the DCDRC along with a filing fee (nominal, based on claim amount) and all evidence.\n"
        "Step 3: Alternatively, file online at consumerhelpline.gov.in (National Consumer Helpline: 1800-11-4000)."
    ),
    "fir_complaint": (
        "Step 1: Present this written complaint to the SHO of the concerned police station. The SHO is legally bound to register FIR.\n"
        "Step 2: If the SHO refuses, file a written complaint to the Superintendent of Police (SP) or approach the JMFC directly.\n"
        "Step 3: Retain a signed, stamped copy of the FIR for your records. Follow up on investigation progress."
    ),
    "domestic_violence": (
        "Step 1: Contact the Protection Officer of your district immediately. File the Domestic Incident Report (DIR).\n"
        "Step 2: File this complaint and application before the JMFC under Section 12, DV Act, for urgent protection order.\n"
        "Step 3: Call Women Helpline 181 or One Stop Centre (OSC) for immediate assistance and shelter if needed."
    ),
    "employment_termination": (
        "Step 1: Send a legal notice to employer demanding reinstatement / compensation within 15 days.\n"
        "Step 2: File conciliation application with the District Labour Commissioner under Section 12, Industrial Disputes Act.\n"
        "Step 3: If conciliation fails, file reference petition before the Labour Court / Industrial Tribunal."
    ),
    "loan_default": (
        "Step 1: Send a written complaint to the bank/NBFC's grievance redressal officer. Keep postal proof.\n"
        "Step 2: If unresolved in 30 days, file complaint with RBI Integrated Ombudsman at cms.rbi.org.in or call 14448.\n"
        "Step 3: If SARFAESI notice received, file application under Section 17, SARFAESI Act before DRT within 45 days."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY DISPLAY NAMES
# ═══════════════════════════════════════════════════════════════════════════════

_CATEGORY_LABELS: dict[str, str] = {
    "wage_theft":             "Non-Payment / Withholding of Wages Complaint",
    "illegal_eviction":       "Illegal Eviction / Unlawful Dispossession Complaint",
    "cheque_bounce":          "Complaint under Section 138, Negotiable Instruments Act (Cheque Bounce)",
    "consumer_complaint":     "Consumer Complaint under Consumer Protection Act, 2019",
    "fir_complaint":          "Written Complaint to Police for FIR Registration (BNSS Section 173)",
    "domestic_violence":      "Application under Protection of Women from Domestic Violence Act, 2005",
    "employment_termination": "Complaint for Wrongful / Illegal Employment Termination",
    "loan_default":           "Complaint against Lender / Bank (Loan Harassment / Unfair Recovery)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are LexShield AI, a senior Indian legal document drafter with 20 years of experience.
You draft precise, professional Indian legal documents that are:
- Formatted exactly as accepted by Indian courts, tribunals, and authorities
- Written in formal Indian legal English with correct legal terminology
- Structured with proper numbered paragraphs, headings, and signature blocks
- Based strictly on facts provided — never hallucinate facts not given
- Complete and ready for submission after legal review

Always use:
- "Complainant" for the person filing the complaint
- "Respondent / Opposite Party / Accused" as appropriate to the forum
- Correct Indian legal date format: DD.MM.YYYY or DD/MM/YYYY
- "Respectfully Sheweth" or "Most Respectfully Sheweth" for complaints to courts
- "Submitted for your kind perusal and necessary action" for complaints to authorities
- Proper verification clause: "VERIFICATION: I, [name], the above-named Complainant, do hereby verify that the contents of paragraphs __ to __ above are true and correct to the best of my knowledge and belief, and that nothing material has been concealed therefrom. Verified at [place] on this __ day of __, 20__."
- When filling the VERIFICATION clause, always use the exact Current Date value provided in the facts — never invent or guess a date.

Do not add disclaimers inside the document body."""


# ═══════════════════════════════════════════════════════════════════════════════
# DRAFT GENERATION PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_generation_prompt(
    category:  str,
    draft_data: dict,
) -> str:
    """Build a highly specific, professional prompt for each complaint type."""

    from datetime import datetime
    current_date_str = datetime.now().strftime("%d.%m.%Y")

    answers      = draft_data.get("answers", {})
    authority    = draft_data.get("authority", _FILING_AUTHORITY.get(category, "Competent Authority"))
    sections_txt = draft_data.get("applicable_sections_text", "")

    # Format answers into a readable block
    if draft_data.get("use_dynamic"):
        # Dynamic path — labeled facts (LLM-extracted + user-answered)
        facts_block = "\n".join(
            f"{k.replace('_', ' ').title()}: {v}"
            for k, v in answers.items()
            if k != "correction"
        )
    else:
        # Fallback path — original Q/A format
        questions = _CLARIFYING_QUESTIONS.get(category, [])
        facts_block = "\n".join(
            f"Q{i+1}: {q}\nA{i+1}: {answers.get(str(i), 'Not provided')}"
            for i, q in enumerate(questions)
        )

    facts_block += f"\n\nCurrent Date (use this for verification clause): {current_date_str}"

    missing = draft_data.get("missing_elements_to_inject", [])
    if missing:
        facts_block += "\n\nCRITICAL FEEDBACK ON PREVIOUS DRAFT:\nYour previous draft was rejected for missing the following required elements:\n- " + "\n- ".join(missing) + "\nYou MUST include them in this new draft."

    relief = _STANDARD_RELIEF.get(category, "Such other relief as the authority deems fit.")

    prompts: dict[str, str] = {

        "wage_theft": f"""Draft a formal complaint for recovery of unpaid wages under the Payment of Wages Act, 1936 and the Industrial Disputes Act, 1947.

FACTS PROVIDED BY COMPLAINANT:
{facts_block}

APPLICABLE LEGAL SECTIONS (from RAG retrieval):
{sections_txt}

FILING AUTHORITY:
{authority}

Generate a complete, court-ready complaint with the following structure:

TO
THE DISTRICT LABOUR COMMISSIONER
[District and State from facts]

COMPLAINT UNDER SECTION 15, PAYMENT OF WAGES ACT, 1936 / SECTION 33C, INDUSTRIAL DISPUTES ACT, 1947 FOR RECOVERY OF UNPAID WAGES

COMPLAINANT: [Full name, address, contact from facts]
RESPONDENT / EMPLOYER: [Employer name, address from facts]

MOST RESPECTFULLY SHEWETH:

1. That the Complainant was employed by the Respondent as [designation inferred from context] since [period inferred].

2. That the Complainant's wages of ₹[amount] per month were fixed as per the appointment letter / mutual agreement.

3. [Facts of non-payment, period, and amounts in numbered paragraphs]

4. That the Complainant made repeated oral/written requests to the Respondent for payment of wages, which were ignored.

5. That the non-payment of wages is in direct violation of Section 4 and Section 5 of the Payment of Wages Act, 1936 / Section 33C of the Industrial Disputes Act, 1947.

APPLICABLE LEGAL PROVISIONS:
[List all applicable sections]

RELIEF SOUGHT:
{relief}

DECLARATION: I, the Complainant, declare that the facts stated herein are true and correct.

VERIFICATION CLAUSE

PLACE: __________ DATE: __________
SIGNATURE OF COMPLAINANT
[Name]
[Address]
[Contact]

List of documents enclosed: [numbered list of enclosures]""",


        "cheque_bounce": f"""Draft a formal criminal complaint under Section 138 read with Section 142 of the Negotiable Instruments Act, 1881 for cheque dishonour.

FACTS PROVIDED BY COMPLAINANT:
{facts_block}

APPLICABLE LEGAL SECTIONS (from RAG retrieval):
{sections_txt}

FILING AUTHORITY:
{authority}

Generate a complete complaint in the format prescribed for filing before the Judicial Magistrate First Class (JMFC):

IN THE COURT OF THE JUDICIAL MAGISTRATE FIRST CLASS
[District] — [State]

COMPLAINT NO. _________ OF 20__

IN THE MATTER OF:

[Complainant name, s/o or d/o, address, contact]          ... COMPLAINANT

VERSUS

[Drawer/Accused name, s/o or d/o, address]                 ... ACCUSED

COMPLAINT UNDER SECTION 138 OF THE NEGOTIABLE INSTRUMENTS ACT, 1881
(Triable as Summons Case)

MOST RESPECTFULLY SHEWETH:

1. That the Complainant is a law-abiding citizen carrying on [business / employment] at [address].

2. That the Accused, in discharge of a legally enforceable liability / debt, drew and issued a cheque bearing No. _______, dated ________, for a sum of ₹________/- (Rupees _________ only), drawn on [Bank name], [Branch], in favour of the Complainant. (The said cheque is marked as EXHIBIT 'A'.)

3. That the Complainant presented the aforesaid cheque for encashment through his/her bank, [Complainant's bank], on __________. The cheque was returned dishonoured on __________ with the bank's memo stating the reason as "___________". (The bank return memo is marked as EXHIBIT 'B'.)

4. That within 30 days of receipt of information of dishonour, the Complainant caused a legal demand notice dated __________ to be sent to the Accused at the address mentioned above by registered post / speed post. The said notice was duly served / attempted to be served on __________. (The notice copy and postal receipt are marked as EXHIBITS 'C' and 'D'.)

5. That despite receipt of the said demand notice, the Accused has failed and neglected to make payment of the said amount within 15 days of receipt of the notice, thus committing an offence punishable under Section 138 of the Negotiable Instruments Act, 1881.

6. That the Complainant is filing this complaint within 30 days of the expiry of the 15-day notice period as mandated by Section 142 of the N.I. Act.

APPLICABLE LEGAL PROVISIONS:
{sections_txt if sections_txt else "Section 138, Section 139, Section 141, Section 142, Negotiable Instruments Act, 1881."}

PRAYER:
It is, therefore, most respectfully prayed that this Hon'ble Court may be pleased to:
(a) Take cognizance of the offence under Section 138 of the Negotiable Instruments Act, 1881;
(b) Issue summons to the Accused;
(c) Convict the Accused and sentence him/her to imprisonment up to 2 years and/or fine up to twice the cheque amount (i.e., ₹_________/-)
(d) Grant such other and further relief as this Hon'ble Court may deem fit and proper.

VERIFICATION:
I, __________, the above-named Complainant, do hereby verify that the contents of paragraphs 1 to 6 above are true and correct to the best of my knowledge and belief, and that nothing material has been concealed therefrom. Verified at __________ on this ______ day of __________, 20__.

COMPLAINANT
[Name]
[Address]
[Contact]

LIST OF ENCLOSURES:
1. Original/copy of dishonoured cheque (Exhibit A)
2. Bank return memo (Exhibit B)
3. Copy of demand notice (Exhibit C)
4. Postal/speed post receipt and tracking (Exhibit D)
5. Proof of identity""",


        "consumer_complaint": f"""Draft a formal consumer complaint under Section 35 of the Consumer Protection Act, 2019 before the District Consumer Disputes Redressal Commission (DCDRC).

FACTS PROVIDED BY COMPLAINANT:
{facts_block}

APPLICABLE LEGAL SECTIONS (from RAG retrieval):
{sections_txt}

FILING AUTHORITY:
{authority}

Generate a complete complaint:

IN THE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION
[District] — [State]

CONSUMER COMPLAINT NO. _________ OF 20__

IN THE MATTER OF:

[Complainant name, s/o or d/o, address, contact]          ... COMPLAINANT

VERSUS

[Company/Seller name, address, through its Proprietor / Managing Director / Authorized Representative]   ... OPPOSITE PARTY

COMPLAINT UNDER SECTIONS 35, 2(1)(e), AND 2(47) OF THE CONSUMER PROTECTION ACT, 2019

MOST RESPECTFULLY SHEWETH:

1. That the Complainant is a consumer within the meaning of Section 2(7) of the Consumer Protection Act, 2019.

2. That the Opposite Party is engaged in the business of [selling goods/providing services] and is an establishment within the jurisdiction of this Commission.

3. [Facts of purchase, date, amount, invoice details in numbered paragraphs]

4. [Facts of defect in goods / deficiency in service — specific and detailed]

5. That the Complainant lodged a complaint with the Opposite Party on __________ which was not resolved / was ignored.

6. That the above constitutes "deficiency in service" / "defect in goods" / "unfair trade practice" within the meaning of Sections 2(11), 2(10), and 2(47) respectively of the Consumer Protection Act, 2019.

7. That the cause of action arose within the jurisdiction of this Commission and the claim falls within the pecuniary limit of this Commission.

APPLICABLE LEGAL PROVISIONS:
{sections_txt if sections_txt else "Section 2(7), 2(10), 2(11), 2(47), Section 35, Section 39, Consumer Protection Act, 2019."}

RELIEF SOUGHT:
{relief}

PRAYER:
The Complainant, therefore, most humbly prays that this Hon'ble Commission may be pleased to pass an order directing the Opposite Party to:
(a) Replace the defective product / complete the deficient service / refund ₹________/-;
(b) Pay compensation of ₹________/- for mental agony, harassment, and loss caused;
(c) Pay costs of this litigation;
(d) Grant such other relief as deemed fit.

VERIFICATION:
I, __________, the above-named Complainant, do hereby verify that the contents of paragraphs 1 to 7 are true and correct to the best of my knowledge and belief. Verified at __________ on this ______ day of __________, 20__.

COMPLAINANT
[Name] [Address] [Contact]

LIST OF ENCLOSURES: [numbered list]""",


        "fir_complaint": f"""Draft a complete written complaint to police for FIR registration under Section 173 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS).

FACTS PROVIDED BY COMPLAINANT:
{facts_block}

APPLICABLE LEGAL SECTIONS (from RAG retrieval):
{sections_txt}

Generate a complete, professionally worded written complaint:

TO
THE STATION HOUSE OFFICER (SHO)
[Police Station Name], [District], [State]

SUBJECT: Written Complaint for Registration of First Information Report (FIR) under Section 173, BNSS, 2023

RESPECTFULLY SUBMITTED:

1. COMPLAINANT DETAILS:
   Name: [From facts]
   S/o or D/o: ____________
   Address: [From facts]
   Contact: [From facts]

2. ACCUSED / SUSPECT DETAILS:
   Name(s): [From facts, or "Not identified / Unknown"]
   Address(es): [From facts]

3. NATURE OF OFFENCE:
   [Classify the offence from facts — be specific: theft, cheating, assault, house-breaking, etc.]
   Applicable Sections: {sections_txt if sections_txt else "[infer from facts: BNS/IPC sections]"}

4. FACTS OF THE COMPLAINT:
   (i) That the Complainant states as under:
   [Detailed chronological narration of facts from Q&A — use sub-paragraphs]
   (ii) Date, time, and place of incident: [From facts]
   (iii) Mode of commission of offence: [From facts]
   (iv) Witnesses (if any): [Names and addresses, or "Will be produced if required"]

5. LOSS/DAMAGE CAUSED:
   [Specify property lost, injury sustained, financial loss, etc.]

6. PREVIOUS COMPLAINT:
   That the Complainant [has / has not] previously reported this matter to any police station.

7. RELIEF REQUESTED:
   {relief}

PRAYER:
I, the Complainant, respectfully request your goodself to:
(a) Register the above complaint as FIR under appropriate sections of BNS/BNSS;
(b) Investigate the matter and arrest the accused;
(c) Recover the stolen property / take other appropriate action.

I state that the contents of this complaint are true and correct to the best of my knowledge and belief.

Yours respectfully,

Date: __________
Place: __________
Signature: __________
Name: __________
Address: __________

ACKNOWLEDGEMENT (to be given by receiving officer):
FIR No.: ___________ Date: ___________ Time: ___________
Received by: ___________ Designation: ___________""",


        "domestic_violence": f"""Draft a formal application under Section 12 of the Protection of Women from Domestic Violence Act, 2005 for reliefs including Protection Order, Residence Order, and Monetary Relief.

FACTS PROVIDED BY COMPLAINANT:
{facts_block}

APPLICABLE LEGAL SECTIONS (from RAG retrieval):
{sections_txt}

FILING AUTHORITY:
{authority}

Generate a complete application:

IN THE COURT OF THE JUDICIAL MAGISTRATE FIRST CLASS / METROPOLITAN MAGISTRATE
[District] — [State]

APPLICATION NO. _________ OF 20__
(Under Section 12 of the Protection of Women from Domestic Violence Act, 2005)

IN THE MATTER OF:

Smt. [Name], W/o [Husband's name], D/o [Father's name],
Aged ____ years, R/o ____________________________          ... AGGRIEVED PERSON / APPLICANT

VERSUS

[Respondent name, S/o, Address]                            ... RESPONDENT

APPLICATION FOR PROTECTION ORDER, RESIDENCE ORDER, MONETARY RELIEF, AND COMPENSATION

MOST RESPECTFULLY SHEWETH:

1. That the Applicant is the lawfully wedded wife / [relationship] of the Respondent. They were married / have been in a domestic relationship since [year].

2. That the parties shared a shared household at [address].

3. That the Respondent has subjected the Applicant to domestic violence as defined under Section 3 of the DV Act, 2005, in the following manner:

   (a) PHYSICAL VIOLENCE: [Details from facts — dates, nature of injury, medical treatment]
   
   (b) EMOTIONAL AND VERBAL ABUSE: [Details from facts]
   
   (c) ECONOMIC ABUSE: [Details — denial of money, dispossession of property]
   
   (d) SEXUAL VIOLENCE: [If applicable — details]

4. That the Applicant has [medical reports / photographs / witnesses] evidencing the domestic violence suffered.

5. That the Applicant apprehends further acts of domestic violence and is in immediate need of protection.

6. That the Applicant [has / has not] filed a Domestic Incident Report with the Protection Officer.

APPLICABLE LEGAL PROVISIONS:
{sections_txt if sections_txt else "Sections 3, 12, 17, 18, 19, 20, 21, 22, 23 of the Protection of Women from Domestic Violence Act, 2005; Section 498A IPC / Section 85 BNS."}

RELIEFS PRAYED FOR:
{relief}

PRAYER:
It is, therefore, most respectfully prayed that this Hon'ble Court may be pleased to:
(a) Pass an ex parte interim Protection Order under Section 18, DV Act, forthwith;
(b) Pass a Residence Order under Section 19, DV Act, directing that the Applicant not be dispossessed from the shared household;
(c) Direct payment of Monetary Relief under Section 20, DV Act, for medical expenses, loss of earnings, and maintenance;
(d) Award Compensation under Section 22, DV Act, for physical and mental injuries;
(e) Pass any other order as this Court may deem fit in the interest of justice.

VERIFICATION:
I, __________, the above-named Applicant, do hereby verify that the contents of paragraphs 1 to 6 above are true and correct to the best of my knowledge and belief. Verified at __________ on this ______ day of __________, 20__.

APPLICANT
[Name] [Address] [Contact]

Counsel for Applicant: ____________ (if applicable)

LIST OF ENCLOSURES: [numbered]""",


        "illegal_eviction": f"""Draft a formal complaint against illegal eviction before the Rent Controller / Civil Court under the applicable State Rent Control Act and Transfer of Property Act, 1882.

FACTS PROVIDED BY COMPLAINANT:
{facts_block}

APPLICABLE LEGAL SECTIONS (from RAG retrieval):
{sections_txt}

FILING AUTHORITY:
{authority}

Generate a complete complaint / petition:

IN THE COURT OF THE RENT CONTROLLER / CIVIL JUDGE (JUNIOR DIVISION)
[District] — [State]

PETITION / COMPLAINT NO. _________ OF 20__

IN THE MATTER OF:

[Complainant / Tenant name, address, contact]              ... PETITIONER / COMPLAINANT

VERSUS

[Landlord / Owner name, address]                           ... RESPONDENT

PETITION FOR RESTORATION OF POSSESSION AND INJUNCTION AGAINST ILLEGAL EVICTION

MOST RESPECTFULLY SHEWETH:

1. That the Petitioner has been a tenant of the premises described hereunder, belonging to the Respondent, [for the past __ years/months]:
   Property: [Full address from facts]
   Type: Residential / Commercial
   Monthly Rent: ₹[amount] per month

2. That the Petitioner has been regularly paying rent to the Respondent [by cash / bank transfer / cheque] and possesses receipts therefor.

3. [Detailed facts of illegal eviction — date, manner, whether force was used, whether notice was given]

4. That the Respondent illegally and unlawfully evicted / attempted to evict the Petitioner without following due process of law.

5. That no valid notice of termination of tenancy has been served as required under Section 106 of the Transfer of Property Act, 1882 and the applicable State Rent Control Act.

6. That the said illegal eviction / threatened eviction is contrary to law and entitles the Petitioner to seek restoration of possession and injunction.

APPLICABLE LEGAL PROVISIONS:
{sections_txt if sections_txt else "Section 106, Transfer of Property Act, 1882; Relevant sections of State Rent Control Act; Section 441/442, BNS (criminal trespass — for police complaint)."}

RELIEF SOUGHT:
{relief}

PRAYER:
The Petitioner, therefore, most respectfully prays that this Hon'ble Court / Authority may be pleased to:
(a) Restore possession of the said premises to the Petitioner immediately;
(b) Pass a permanent injunction restraining the Respondent from interfering with the Petitioner's peaceful possession;
(c) Award damages of ₹_____/- for loss and inconvenience caused;
(d) Grant costs of proceedings;
(e) Pass such other order as deemed fit.

VERIFICATION: [Standard]

PETITIONER
[Name] [Address] [Contact]

LIST OF ENCLOSURES: [numbered]""",


        "employment_termination": f"""Draft a formal complaint against wrongful / illegal termination of employment under the Industrial Disputes Act, 1947 before the Labour Commissioner.

FACTS PROVIDED BY COMPLAINANT:
{facts_block}

APPLICABLE LEGAL SECTIONS (from RAG retrieval):
{sections_txt}

FILING AUTHORITY:
{authority}

Generate a complete complaint:

TO
THE DISTRICT LABOUR COMMISSIONER
[District] — [State]

COMPLAINT / STATEMENT OF CLAIM UNDER SECTION 2A / SECTION 10, INDUSTRIAL DISPUTES ACT, 1947

COMPLAINANT / WORKMAN:
[Name, address, contact, designation from facts]

RESPONDENT / EMPLOYER:
[Company name, address from facts]

SUBJECT: Complaint against Wrongful / Illegal Termination of Service

MOST RESPECTFULLY SHEWETH:

1. That the Complainant was employed as [designation] with the Respondent since [date]. The nature of employment was [permanent / contractual / casual].

2. That the Complainant's last drawn monthly wages were ₹[amount], which included basic salary, HRA, and other allowances.

3. That the Complainant was terminated from service on [date]. [The termination was oral / communicated by letter dated ____ for reasons stated as ____.]

4. That the said termination is illegal and in violation of the following:
   (a) No charge-sheet or show-cause notice was issued to the Complainant prior to termination.
   (b) No domestic enquiry was conducted as required under the principles of natural justice.
   (c) No retrenchment compensation under Section 25F of the Industrial Disputes Act, 1947 was paid.
   (d) No notice / pay in lieu of notice was given as required under the terms of employment.

5. That the Complainant has been continuously employed with the Respondent for [period], completing more than 1 year of service, and is therefore entitled to retrenchment protection under Section 25F, IDA.

6. That the Complainant is an "industrial workman" within the meaning of Section 2(s) of the Industrial Disputes Act, 1947.

APPLICABLE LEGAL PROVISIONS:
{sections_txt if sections_txt else "Sections 2(s), 2A, 10, 25F, 25G, 25H, Industrial Disputes Act, 1947."}

RELIEF SOUGHT:
{relief}

PRAYER:
It is, therefore, most respectfully prayed that this Hon'ble Authority may be pleased to:
(a) Declare the termination of the Complainant as illegal, unjustified, and void;
(b) Direct the Respondent to reinstate the Complainant with full back wages and continuity of service;
(c) Alternatively, direct payment of retrenchment compensation as per Section 25F, IDA;
(d) Direct payment of notice pay, earned leave dues, gratuity, and PF contributions;
(e) Grant costs of proceedings.

VERIFICATION: [Standard]

COMPLAINANT
[Name] [Address] [Contact]

LIST OF ENCLOSURES: [numbered]""",


        "loan_default": f"""Draft a formal complaint against a bank / NBFC for harassment, unfair recovery practices, and violation of RBI guidelines under the Recovery of Debts and Bankruptcy Act, 1993.

FACTS PROVIDED BY COMPLAINANT:
{facts_block}

APPLICABLE LEGAL SECTIONS (from RAG retrieval):
{sections_txt}

FILING AUTHORITY:
{authority}

Generate a complete complaint:

TO
THE BANKING OMBUDSMAN / NODAL OFFICER
[RBI / Bank Name / DRT, as applicable]
[City and State]

COMPLAINT UNDER THE RBI INTEGRATED OMBUDSMAN SCHEME, 2021 / RECOVERY OF DEBTS AND BANKRUPTCY ACT, 1993

COMPLAINANT:
[Name, address, contact from facts]

OPPOSITE PARTY / RESPONDENT:
[Bank / NBFC name, branch, and address from facts]

LOAN DETAILS:
Loan Type: [From facts]
Loan Account No.: ________________
Loan Amount: ₹[Amount]
Lender: [Name and branch]

SUBJECT: Complaint against Unfair Loan Recovery Practices / Harassment / Violation of RBI Guidelines

RESPECTFULLY SUBMITTED:

1. That the Complainant availed of a [type] loan of ₹______/- from the Respondent on [date], repayable in _____ monthly instalments of ₹______/- each.

2. That due to [genuine financial hardship — specify reason], the Complainant was unable to pay [number] EMIs from [period], aggregating to ₹______/-.

3. That the Respondent / its recovery agents have engaged in the following unfair / unlawful recovery practices in violation of the RBI Fair Practices Code:
   [List specific acts: phone harassment, threats, visiting home/workplace, public humiliation, use of force, calling at odd hours, contacting family members, etc. — from facts]

4. That the Complainant made a written representation to the Respondent's Grievance Redressal Officer on [date], which was [not responded to / inadequately addressed].

5. That the conduct of the Respondent violates:
   (a) RBI Circular on Fair Practices Code for Lenders;
   (b) RBI Guidelines on Recovery Agents;
   (c) Reserve Bank — Integrated Ombudsman Scheme, 2021.

APPLICABLE LEGAL PROVISIONS:
{sections_txt if sections_txt else "Recovery of Debts and Bankruptcy Act, 1993; SARFAESI Act, 2002; RBI Fair Practices Code; RBI Integrated Ombudsman Scheme, 2021."}

RELIEF SOUGHT:
{relief}

PRAYER:
The Complainant most respectfully prays:
(a) To issue directions to the Respondent to immediately stop harassing recovery practices;
(b) To direct restructuring / rescheduling of the loan on humanitarian grounds;
(c) To award compensation of ₹_____/- for mental agony and harassment;
(d) To take penal action against the Respondent for violation of RBI guidelines;
(e) To pass such other order as deemed just and proper.

DECLARATION:
I hereby declare that the information given above is true and correct to the best of my knowledge and belief.

Yours faithfully,
Date: __________
Place: __________
[Name]
[Address]
[Contact / Email]

LIST OF ENCLOSURES: [numbered]""",
    }

    return prompts.get(category, prompts.get("fir_complaint", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: DETECT CORRECTION CONTENT IN A "CONFIRM" MESSAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _looks_like_correction(query: str) -> bool:
    """True if the query contains correction content beyond just 'confirm'."""
    # Remove the confirm word and check if meaningful content remains
    stripped = re.sub(
        r'\b(confirm|yes|proceed|ok|okay|go ahead)\b', '', query,
        flags=re.IGNORECASE
    ).strip()
    # If more than 5 words remain after removing confirm, it's likely a correction
    return len(stripped.split()) >= 5


# ═══════════════════════════════════════════════════════════════════════════════
# DRAFTING AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DraftingAgent:

    def __init__(self):
        pass

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _load(self, session_id: str) -> Optional[dict]:
        """Load draft row from DB. Returns None if not found."""
        with get_conn() as conn:
            row = conn.execute(
                "SELECT stage, category, draft_data FROM drafts WHERE session_id = %s",
                (session_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "stage":      row["stage"],
            "category":   row["category"],
            "draft_data": json.loads(row["draft_data"]),
        }

    def _save(self, session_id: str, stage: str, category: str, draft_data: dict) -> None:
        now = time.time()
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO drafts (session_id, stage, category, draft_data, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(session_id) DO UPDATE SET
                    stage      = excluded.stage,
                    category   = excluded.category,
                    draft_data = excluded.draft_data,
                    updated_at = excluded.updated_at
            """, (session_id, stage, category, json.dumps(draft_data), now, now))

    def _delete(self, session_id: str) -> None:
        with get_conn() as conn:
            conn.execute("DELETE FROM drafts WHERE session_id = %s", (session_id,))

    # ── Public API ─────────────────────────────────────────────────────────────

    def has_active_draft(self, session_id: str, query: str = "") -> bool:
        """True if this session has a draft. For DONE stage, use heuristic to distinguish correction vs new query."""
        row = self._load(session_id)
        if row is None:
            return False
            
        if row["stage"] != DraftStage.DONE:
            return True
            
        if not query:
            return False
            
        # Heuristic for DONE stage: differentiate correction vs new question
        q_lower = query.lower().strip()
        is_question = bool(re.match(r'^(what|how|why|who|when|can)\b', q_lower) or '?' in q_lower)
        correction_keywords = ["change", "update", "fix", "instead", "add", "remove", "typo", "mistake", "wrong", "draft", "document", "address", "name", "rent", "salary"]
        has_keyword = any(kw in q_lower for kw in correction_keywords)
        
        # Very short phrases (e.g. "Mumbai", "Rent was 5000") are usually corrections
        is_short = len(query.split()) <= 6
        
        if (has_keyword or is_short) and not is_question:
            return True
            
        return False

    def handle(self, query: str, session_id: str) -> dict:
        """
        Main dispatch entry point called by draft_node in graph.py.
        Routes to the appropriate stage handler based on DB state.

        Stage dispatch order:
          None              -> _start_draft  (detects category or shows menu)
          MENU_SHOWN        -> _handle_menu_shown  (parse selection, advance to COLLECTING_FIELDS)
          COLLECTING_FIELDS -> _process_answer     (collect field values, advance to AWAITING_CONFIRMATION)
          AWAITING_CONFIRMATION -> _generate_draft (only reached on confirmed intent from graph.py)
          CLARIFY           -> _process_answer     (legacy fallback path)
          RETRIEVE_SECTIONS -> _retrieve_sections
          IDENTIFY_AUTHORITY-> _identify_authority
          CONFIRM           -> _handle_confirm      (legacy — kept for back-compat)
          GENERATE          -> _generate_draft
        """
        row = self._load(session_id)

        if row is None:
            return self._start_draft(session_id, query)

        stage = row["stage"]

        # ── New explicit stages ────────────────────────────────────────────────
        if stage == DraftStage.MENU_SHOWN:
            return self._handle_menu_shown(session_id, query, row)

        if stage == DraftStage.COLLECTING_FIELDS:
            return self._process_answer(session_id, query, row)

        if stage == DraftStage.AWAITING_CONFIRMATION:
            # graph.py only routes here on confirmed intent — generate immediately
            return self._generate_draft(session_id, row)

        # ── Legacy internal stages (backward-compat) ───────────────────────────
        if stage == DraftStage.CLARIFY:
            return self._process_answer(session_id, query, row)

        if stage == DraftStage.RETRIEVE_SECTIONS:
            return self._retrieve_sections(session_id, row)

        if stage == DraftStage.IDENTIFY_AUTHORITY:
            return self._identify_authority(session_id, row)

        if stage == DraftStage.CONFIRM:
            return self._handle_confirm(session_id, query, row)

        if stage == DraftStage.GENERATE:
            return self._generate_draft(session_id, row)

        if stage == DraftStage.DONE:
            category = row["category"]
            draft_data = row["draft_data"]
            
            # Apply correction
            draft_data = self._apply_correction(draft_data, query, category)
            row["draft_data"] = draft_data
            
            # Regenerate draft
            return self._generate_draft(session_id, row)

        # Unexpected state
        self._delete(session_id)
        return {
            "answer":   "Your draft session has expired or reached an unexpected state. Please start again.",
            "stage":    0,
            "doc_type": "",
            "complete": False,
            "draft":    "",
        }

    # ── FACT EXTRACTION (Groq JSON-mode) ───────────────────────────────────────

    def _extract_facts(self, query: str, doc_type: str) -> FactExtractionResult:
        """
        Extract facts from the user's initial query via a single Groq JSON-mode call.

        Returns FactExtractionResult with extracted_facts, missing_fields, confidence.
        On any failure -> returns empty extraction (all fields missing, confidence=0.0)
        so the caller silently falls back to the full sequential question flow.
        """
        required = REQUIRED_FIELDS.get(doc_type)
        if not required:
            return FactExtractionResult(
                extracted_facts={}, missing_fields=[], confidence=0.0
            )

        always_ask = set(ALWAYS_ASK_FIELDS.get(doc_type, []))
        # Fields the LLM should attempt to extract (exclude always-ask)
        extractable = [f for f in required if f not in always_ask]

        field_list = ", ".join(extractable)
        system_prompt = (
            f"You are a legal fact extractor for LexShield AI.\n"
            f"Extract facts for a {doc_type.replace('_', ' ')} legal document "
            f"from the user's query.\n\n"
            f"Required fields: {field_list}\n\n"
            f"For each field, extract the value if clearly present in the query.\n"
            f"If a field is NOT present or cannot be inferred with confidence, "
            f"include it in missing_fields.\n\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{\n'
            f'  "extracted_facts": {{"field_name": "value", ...}},\n'
            f'  "missing_fields": ["field_name", ...],\n'
            f'  "confidence": 0.0\n'
            f'}}\n\n'
            f"Rules:\n"
            f"- Only include a field in extracted_facts if the value is "
            f"explicitly present in the query.\n"
            f"- Every required field must appear in exactly one of: "
            f"extracted_facts or missing_fields.\n"
            f"- confidence: 0.0–1.0, reflecting overall extraction quality.\n"
            f"- Do NOT invent or hallucinate values."
        )

        try:
            import os
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set")

            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model           = "llama-3.3-70b-versatile",
                messages        = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": query},
                ],
                temperature     = 0,
                max_tokens      = 512,
                response_format = {"type": "json_object"},
                timeout         = 5,
            )
            raw = resp.choices[0].message.content.strip()

            # Parse and validate
            parsed = json.loads(raw)
            extracted = parsed.get("extracted_facts", {})
            missing   = parsed.get("missing_fields", [])
            conf      = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))

            # Ensure only valid fields are in extracted_facts
            valid_extracted = {
                k: v for k, v in extracted.items()
                if k in extractable and isinstance(v, str) and v.strip()
            }
            # Rebuild missing: anything not successfully extracted
            valid_missing = [
                f for f in extractable if f not in valid_extracted
            ]
            # Always-ask fields go into missing_fields unconditionally
            for field in always_ask:
                if field not in valid_missing:
                    valid_missing.append(field)

            return FactExtractionResult(
                extracted_facts=valid_extracted,
                missing_fields=valid_missing,
                confidence=conf,
            )

        except Exception as e:
            logger.exception("[DraftingAgent] Fact extraction failed (non-fatal)")
            # Fallback: all fields missing
            return FactExtractionResult(
                extracted_facts={},
                missing_fields=list(required),
                confidence=0.0,
            )

    # ── STAGE: START (INIT -> CLARIFY) ──────────────────────────────────────────


    def _field_question(self, field: str, category: str) -> str:
        """Delegates to the module-level _field_question() which uses
        _FIELD_QUESTION_MAP for a unique per-field question (Bug 1 fix)."""
        return _field_question(field, category)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE: MENU_SHOWN handler
    # ─────────────────────────────────────────────────────────────────────────

    # Map of digit / word / partial name -> category key
    _MENU_MAP: dict = {
        "1": "wage_theft",          "one": "wage_theft",
        "2": "illegal_eviction",     "two": "illegal_eviction",
        "3": "cheque_bounce",        "three": "cheque_bounce",
        "4": "consumer_complaint",   "four": "consumer_complaint",
        "5": "fir_complaint",        "five": "fir_complaint",
        "6": "domestic_violence",    "six": "domestic_violence",
        "7": "employment_termination", "seven": "employment_termination",
        "8": "loan_default",         "eight": "loan_default",
        # partial name aliases
        "wage": "wage_theft", "salary": "wage_theft", "unpaid": "wage_theft",
        "eviction": "illegal_eviction", "evict": "illegal_eviction",
        "cheque": "cheque_bounce", "bounce": "cheque_bounce", "ni act": "cheque_bounce",
        "consumer": "consumer_complaint",
        "fir": "fir_complaint", "police": "fir_complaint",
        "domestic": "domestic_violence", "violence": "domestic_violence", "dv": "domestic_violence",
        "termination": "employment_termination", "wrongful": "employment_termination", "employment": "employment_termination",
        "loan": "loan_default", "bank": "loan_default", "emi": "loan_default",
    }

    def _handle_menu_shown(self, session_id: str, query: str, row: dict) -> dict:
        """
        MENU_SHOWN stage: parse user's complaint-type selection.
        Accepts: digit ("8"), number word ("eight"), or partial name ("loan", "cheque bounce").
        On match  -> save COLLECTING_FIELDS, start field collection.
        On no match -> re-display menu.
        """
        q = query.strip().lower()

        # Try exact digit / word first
        category = self._MENU_MAP.get(q)

        # Try partial match anywhere in the query
        if category is None:
            for key, cat in self._MENU_MAP.items():
                if key in q:
                    category = cat
                    break

        if category is None:
            # Could not parse — check attempts and re-prompt or exit
            draft_data = row.get("draft_data", {})
            attempts = draft_data.get("menu_attempts", 0) + 1
            if attempts >= 2:
                self._delete(session_id)
                return {
                    "answer": (
                        "I'm only able to help draft the 8 specific complaint types listed above "
                        "— I can't draft general documents like emails or letters outside these "
                        "categories. If your situation doesn't match any of them, let me know more "
                        "details and I'll tell you if there's a related legal remedy."
                    ),
                    "stage": 0,
                    "doc_type": "",
                    "complete": False,
                    "draft": "",
                }
            
            draft_data["menu_attempts"] = attempts
            self._save(session_id, DraftStage.MENU_SHOWN, "__menu__", draft_data)

            return {
                "answer": (
                    "Sorry, I didn't catch that. Please reply with the number (1–8) "
                    "or name of the complaint type you need:\n\n"
                    "1. **Unpaid Wages / Salary**\n"
                    "2. **Illegal Eviction**\n"
                    "3. **Cheque Bounce** (Section 138 NI Act)\n"
                    "4. **Consumer Complaint**\n"
                    "5. **FIR / Police Complaint**\n"
                    "6. **Domestic Violence**\n"
                    "7. **Wrongful Termination**\n"
                    "8. **Loan / Bank Harassment**"
                ),
                "stage":    DraftStage.MENU_SHOWN,
                "doc_type": "",
                "complete": False,
                "draft":    "",
            }

        logger.info(f"[DraftingAgent] Menu selection -> category={category!r}")

        # Build initial draft_data and immediately start field collection
        # Treat the selection query as an empty description (no facts to extract)
        label     = _CATEGORY_LABELS[category]
        questions = _CLARIFYING_QUESTIONS[category]
        required  = REQUIRED_FIELDS.get(category, [])

        draft_data = {
            "answers":          {},
            "current_q_index":  0,
            "missing_fields":   list(required),
            "use_dynamic":      True,
            "applicable_sections_text": "",
            "authority":        "",
        }
        self._save(session_id, DraftStage.COLLECTING_FIELDS, category, draft_data)

        first_field = required[0] if required else None
        first_q = _field_question(first_field, category) if first_field else questions[0]
        return {
            "answer": (
                f"I will help you draft a **{label}**.\n\n"
                f"I need a few details. Please answer the following questions one at a time.\n\n"
                f"**Question 1 of {len(required)}:**\n{first_q}"
            ),
            "stage":    DraftStage.COLLECTING_FIELDS,
            "doc_type": category,
            "complete": False,
            "draft":    "",
        }

    def _start_draft(self, session_id: str, description: str) -> dict:
        category = _detect_category(description)

        if category is None:
            # ── Persist MENU_SHOWN so the next reply is routed to draft_node ──
            self._save(session_id, DraftStage.MENU_SHOWN, "__menu__", {})
            return {
                "answer": (
                    "I can help you draft complaints and legal documents for the following situations:\n\n"
                    "1. **Unpaid Wages / Salary** — complaint to Labour Commissioner\n"
                    "2. **Illegal Eviction** — complaint to Rent Controller / Civil Court\n"
                    "3. **Cheque Bounce** — complaint under Section 138, NI Act to JMFC\n"
                    "4. **Consumer Complaint** — complaint to District Consumer Commission\n"
                    "5. **FIR / Police Complaint** — written complaint to Police Station\n"
                    "6. **Domestic Violence** — application under DV Act to Magistrate\n"
                    "7. **Wrongful Termination** — complaint to Labour Court\n"
                    "8. **Loan / Bank Harassment** — complaint to RBI Ombudsman / DRT\n\n"
                    "Please reply with the number (1–8) or describe your situation and I will "
                    "identify the right complaint type."
                ),
                "stage":    DraftStage.MENU_SHOWN,
                "doc_type": "",
                "complete": False,
                "draft":    "",
            }

        label      = _CATEGORY_LABELS[category]
        questions  = _CLARIFYING_QUESTIONS[category]
        required   = REQUIRED_FIELDS.get(category, [])

        # ── Attempt dynamic fact extraction ────────────────────────────────────
        res = self._extract_facts(description, category)

        if res.confidence >= 0.5 and res.extracted_facts:
            logger.debug(
                f"[DraftingAgent] session={session_id[:8]} "
                f"-> extracted {len(res.extracted_facts)} facts. "
                f"missing={len(res.missing_fields)} conf={res.confidence:.2f}"
            )

            draft_data = {
                "answers":                  dict(res.extracted_facts),
                "current_q_index":          0,
                "missing_fields":           list(res.missing_fields),
                "use_dynamic":              True,
                "applicable_sections_text": "",
                "authority":                "",
            }

            if not res.missing_fields:
                # All facts extracted — advance straight to AWAITING_CONFIRMATION
                self._save(session_id, DraftStage.RETRIEVE_SECTIONS, category, draft_data)
                return self._retrieve_sections(
                    session_id, {"category": category, "draft_data": draft_data}
                )

            # Some fields missing — persist COLLECTING_FIELDS, ask the first one
            first_field = res.missing_fields[0]
            first_q     = _field_question(first_field, category)
            self._save(session_id, DraftStage.COLLECTING_FIELDS, category, draft_data)

            return {
                "answer": (
                    f"I will help you draft a **{label}**.\n\n"
                    f"I've noted the details you provided. I just need a few more:\n\n"
                    f"**Question 1 of {len(res.missing_fields)}:**\n{first_q}"
                ),
                "stage":    DraftStage.COLLECTING_FIELDS,
                "doc_type": category,
                "complete": False,
                "draft":    "",
            }

        # ── Fallback: no extraction — sequential questioning ───────────────────
        logger.debug(
            f"[DraftingAgent] fact extraction fall-through: "
            f"extracted={len(res.extracted_facts)} missing={len(res.missing_fields)}"
        )

        # BUG1 fix: use _field_question so the shown question matches required[0],
        # not just the first clarifying question (which may be for a different field).
        first_q    = _field_question(required[0], category) if required else questions[0]
        draft_data = {
            "answers":          {},
            "current_q_index":  0,
            "missing_fields":   list(required),
            "use_dynamic":      True,   # use field-based loop for consistency
            "applicable_sections_text": "",
            "authority":        "",
        }

        self._save(session_id, DraftStage.COLLECTING_FIELDS, category, draft_data)

        return {
            "answer": (
                f"I will help you draft a **{label}**.\n\n"
                f"I need to gather some details first. Please answer the following questions "
                f"one at a time.\n\n"
                f"**Question 1 of {len(required)}:**\n{first_q}"
            ),
            "stage":    DraftStage.COLLECTING_FIELDS,
            "doc_type": category,
            "complete": False,
            "draft":    "",
        }

    # ── STAGE: COLLECTING_FIELDS / CLARIFY (collect answers one turn at a time) ─

    def _process_answer(self, session_id: str, answer: str, row: dict) -> dict:
        """
        Handles both COLLECTING_FIELDS (new) and CLARIFY (legacy) stages.
        Always uses the dynamic missing_fields loop when use_dynamic=True.
        Advances to RETRIEVE_SECTIONS (-> IDENTIFY_AUTHORITY -> AWAITING_CONFIRMATION)
        when all fields are collected.
        """
        category    = row["category"]
        draft_data  = row["draft_data"]
        use_dynamic = draft_data.get("use_dynamic", False)
        cur_stage   = row["stage"]  # COLLECTING_FIELDS or CLARIFY

        if use_dynamic:
            missing = draft_data.get("missing_fields", [])
            idx     = draft_data["current_q_index"]

            if idx < len(missing):
                field = missing[idx]
                draft_data["answers"][field] = answer.strip()

            next_idx = idx + 1
            if next_idx < len(missing):
                draft_data["current_q_index"] = next_idx
                next_field = missing[next_idx]
                next_q     = _field_question(next_field, category)
                # Stay in same stage (COLLECTING_FIELDS or CLARIFY)
                self._save(session_id, cur_stage, category, draft_data)
                return {
                    "answer": (
                        f"**Question {next_idx + 1} of {len(missing)}:**\n"
                        f"{next_q}"
                    ),
                    "stage":    cur_stage,
                    "doc_type": category,
                    "complete": False,
                    "draft":    "",
                }

            # All fields collected -> pipeline advances to AWAITING_CONFIRMATION
            self._save(session_id, DraftStage.RETRIEVE_SECTIONS, category, draft_data)
            return self._retrieve_sections(session_id, {"category": category, "draft_data": draft_data})

        # ── Legacy fallback: sequential index-based loop ───────────────────────
        questions = _CLARIFYING_QUESTIONS[category]
        idx       = draft_data["current_q_index"]
        draft_data["answers"][str(idx)] = answer.strip()
        next_idx  = idx + 1

        if next_idx < len(questions):
            draft_data["current_q_index"] = next_idx
            self._save(session_id, cur_stage, category, draft_data)
            return {
                "answer": (
                    f"**Question {next_idx + 1} of {len(questions)}:**\n"
                    f"{questions[next_idx]}"
                ),
                "stage":    cur_stage,
                "doc_type": category,
                "complete": False,
                "draft":    "",
            }

        self._save(session_id, DraftStage.RETRIEVE_SECTIONS, category, draft_data)
        return self._retrieve_sections(session_id, {"category": category, "draft_data": draft_data})

    # ── STAGE: RETRIEVE_SECTIONS ───────────────────────────────────────────────

    def _retrieve_sections(self, session_id: str, row: dict) -> dict:
        category   = row["category"]
        draft_data = row["draft_data"]

        rag_query = _RAG_QUERIES.get(category, f"Indian law complaint {category.replace('_', ' ')}")

        try:
            from rag.pipeline import rag_pipeline
            answer = rag_pipeline.query(rag_query)

            # Extract section references from the answer
            section_text = answer.answer_text[:1500] if answer.answer_text else ""
            draft_data["applicable_sections_text"] = section_text
        except Exception as e:
            logger.exception("[DraftingAgent] RAG retrieval failed (non-fatal)")
            draft_data["applicable_sections_text"] = ""

        self._save(session_id, DraftStage.IDENTIFY_AUTHORITY, category, draft_data)
        return self._identify_authority(session_id, {"category": category, "draft_data": draft_data})

    # ── STAGE: IDENTIFY_AUTHORITY ──────────────────────────────────────────────

    def _identify_authority(self, session_id: str, row: dict) -> dict:
        category   = row["category"]
        draft_data = row["draft_data"]

        authority = _FILING_AUTHORITY.get(category, "Competent Authority as per applicable law")
        draft_data["authority"] = authority

        self._save(session_id, DraftStage.CONFIRM, category, draft_data)
        return self._confirm_draft(session_id, {"category": category, "draft_data": draft_data})

    # ── STAGE: CONFIRM ─────────────────────────────────────────────────────────

    def _confirm_draft(self, session_id: str, row: dict) -> dict:
        category   = row["category"]
        draft_data = row["draft_data"]
        answers    = draft_data.get("answers", {})

        outline_lines = []
        if draft_data.get("use_dynamic"):
            # Dynamic path — labeled field->value pairs
            for field, value in answers.items():
                if field == "correction":
                    continue
                pretty = field.replace("_", " ").title()
                val_display = value[:100] + ("..." if len(value) > 100 else "")
                outline_lines.append(f"  • {pretty}: {val_display}")
        else:
            # Fallback path — original Q/A index-based outline
            questions = _CLARIFYING_QUESTIONS.get(category, [])
            for i, q in enumerate(questions):
                a = answers.get(str(i), "Not provided")
                outline_lines.append(f"  • {q.split('?')[0]}: {a[:100]}{'...' if len(a) > 100 else ''}")
        outline_str = "\n".join(outline_lines)

        authority_short = draft_data.get("authority", "").split("\n")[0]
        label           = _CATEGORY_LABELS.get(category, category)
        relief          = _STANDARD_RELIEF.get(category, "Standard relief")

        return {
            "answer": (
                f"✅ **I have all the information I need.** Here is a summary of what your draft will contain:\n\n"
                f"**Document:** {label}\n\n"
                f"**Details collected:**\n{outline_str}\n\n"
                f"**Filing Authority:** {authority_short}\n\n"
                f"**Relief Sought:** {relief[:200]}{'...' if len(relief) > 200 else ''}\n\n"
                f"---\n"
                f"Reply **'confirm'** to generate your complete, professionally formatted legal draft.\n"
                f"Or tell me if any detail needs to be corrected before drafting."
            ),
            "stage":    DraftStage.CONFIRM,
            "doc_type": category,
            "complete": False,
            "draft":    "",
        }

    def _apply_correction(self, draft_data: dict, correction_text: str, category: str) -> dict:
        """Helper to apply a correction to a specific field via LLM."""
        required = REQUIRED_FIELDS.get(category, [])
        current_answers = draft_data.get("answers", {})
        
        # Build prompt with current fields and values
        fields_context = "\n".join([f"- {f}: {current_answers.get(f, 'Not provided')}" for f in required])
        
        system_prompt = (
            f"You are a legal data correction assistant.\n"
            f"The user is correcting one or more fields in their complaint draft.\n\n"
            f"Current field values:\n{fields_context}\n\n"
            f"User correction: \"{correction_text}\"\n\n"
            f"Identify which field the user wants to update, and what the new value should be.\n"
            f"Respond ONLY with valid JSON:\n"
            f"{{\"field_to_update\": \"field_name\", \"new_value\": \"value\", \"found\": true}}\n"
            f"If you cannot confidently determine which field to update, return {{\"found\": false}}.\n"
            f"The 'field_to_update' MUST be exactly one of the fields listed above."
        )
        
        try:
            import os
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set")

            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model           = "llama-3.3-70b-versatile",
                messages        = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": correction_text},
                ],
                temperature     = 0,
                max_tokens      = 128,
                response_format = {"type": "json_object"},
                timeout         = 5,
            )
            raw = resp.choices[0].message.content.strip()
            
            # Parse JSON
            parsed = json.loads(raw)
            if parsed.get("found") is True:
                field = parsed.get("field_to_update")
                val = parsed.get("new_value")
                if field in required and val:
                    # Successfully mapped to a specific field
                    draft_data["answers"][field] = val
                    return draft_data

        except Exception as e:
            logger.exception("[DraftingAgent] Correction application failed, falling back")
            
        # Fallback: just store under "correction"
        draft_data["answers"]["correction"] = correction_text
        return draft_data

    def _handle_confirm(self, session_id: str, query: str, row: dict) -> dict:
        """Check if user confirmed, cancelled, or wants to make changes.

        Any message that is NOT a clear confirmation or cancellation is
        treated as a potential correction and routed to _apply_correction()
        so the LLM can map freeform text (e.g. "the property is in Delhi")
        to the correct field without requiring trigger words.
        """
        category = row["category"]
        draft_data = row["draft_data"]

        is_confirm = re.search(
            r'\b(confirm|yes|proceed|generate|draft it|go ahead|ok|okay|yes please)\b',
            query, re.IGNORECASE
        )
        is_cancel = re.search(
            r'\b(cancel|stop|abort|quit|start over|restart)\b',
            query, re.IGNORECASE
        )

        if is_cancel:
            self._delete(session_id)
            return {
                "answer": "Your draft session has been cancelled. You can start a new draft anytime.",
                "stage": 0, "doc_type": "", "complete": False, "draft": "",
            }

        if is_confirm and not _looks_like_correction(query):
            # Pure confirmation with no correction content — generate immediately
            self._save(session_id, DraftStage.GENERATE, category, draft_data)
            return self._generate_draft(session_id, row)

        # Everything else: treat as potential correction
        # (includes "the property is in delhi", "change salary to 60000",
        #  "her name is Priya", or any freeform correction)
        draft_data = self._apply_correction(draft_data, query, category)

        # After correction, check if they also said confirm in the same message
        if is_confirm:
            self._save(session_id, DraftStage.GENERATE, category, draft_data)
            return self._generate_draft(session_id, row)

        # Correction applied, re-show updated summary for user to verify
        self._save(session_id, DraftStage.CONFIRM, category, draft_data)
        return self._confirm_draft(session_id, {"category": category, "draft_data": draft_data})

    # ── STAGE: GENERATE ────────────────────────────────────────────────────────

    def _generate_draft(self, session_id: str, row: dict) -> dict:
        category   = row["category"]
        draft_data = row["draft_data"]

        logger.info(f"[DraftingAgent] Generating {category} draft for session {session_id[:8]}…")

        validation_status = "passed"
        try:
            draft_text = self._call_llm(category, draft_data)
            
            # Validation Step
            val_result = self._validate_draft(draft_text, category)
            if not val_result.get("passed", True):
                missing_items = val_result.get("missing", [])
                logger.warning(f"[DraftingAgent] Validation failed. Missing: {missing_items}. Regenerating...")
                
                draft_data["missing_elements_to_inject"] = missing_items
                draft_text_2 = self._call_llm(category, draft_data)
                
                # Validate second attempt
                val_result_2 = self._validate_draft(draft_text_2, category)
                if not val_result_2.get("passed", True):
                    logger.warning("[DraftingAgent] Second attempt failed validation. Returning as-is.")
                    validation_status = "failed_returned"
                else:
                    logger.info("[DraftingAgent] Second attempt passed validation.")
                    validation_status = "failed_regenerated"
                
                draft_text = draft_text_2
                draft_data["missing_elements"] = missing_items

        except Exception as e:
            logger.exception("[DraftingAgent] LLM generation failed")
            draft_text = f"[Draft generation failed: {e}. Please try again.]"
            validation_status = "failed_returned"

        # Mark DONE
        self._save(session_id, DraftStage.DONE, category, draft_data)

        docs  = _SUPPORTING_DOCUMENTS.get(category, [])
        steps = _NEXT_STEPS.get(category, "Consult a lawyer to file this document.")
        label = _CATEGORY_LABELS.get(category, category)

        answer = (
            f"✅ **Your {label} is ready.**\n\n"
            f"---\n\n"
            f"{draft_text}\n\n"
            f"---\n\n"
            f"**📎 Supporting Documents to Attach:**\n"
            + "\n".join(f"{i+1}. {d}" for i, d in enumerate(docs))
            + f"\n\n**📌 Filing Authority:**\n{draft_data.get('authority', '').split(chr(10))[0]}\n\n"
            f"**🚀 Next Steps:**\n{steps}\n\n"
            f"⚠️ *This is an AI-generated draft for reference. Please review with a qualified advocate "
            f"before filing. The advocate may need to customise specific details for your jurisdiction.*"
        )

        return {
            "answer":    answer,
            "stage":     DraftStage.DONE,
            "doc_type":  category,
            "complete":  True,
            "draft":     draft_text,
            "draft_data": draft_data,
            "validation_status": validation_status,
        }

    # ── Validation ─────────────────────────────────────────────────────────────
    
    def _validate_draft(self, draft_text: str, doc_type: str) -> dict:
        prompt = f"""You are a legal document validator. Check if this {doc_type} draft contains ALL of:
1. Party names (complainant and respondent)
2. Specific legal basis or act cited
3. Clear statement of facts
4. Specific relief sought
5. Verification/declaration clause
Respond ONLY as JSON: {{"passed": true}} or {{"passed": false, "missing": ["item1", "item2"]}}

Draft to validate:
{draft_text}"""
        try:
            from rag.llm import llm
            response = llm.generate(
                prompt=prompt, 
                system_prompt="You are a strict legal validator outputting ONLY JSON.", 
                temperature=0.0, 
                max_tokens=200
            )
            
            clean_json = response.strip().strip('`').replace('json', '').strip()
            return json.loads(clean_json)
        except Exception as e:
            logger.exception("[DraftingAgent] Validation parse error")
            return {"passed": True}

    # ── LLM call ───────────────────────────────────────────────────────────────

    def _call_llm(self, category: str, draft_data: dict) -> str:
        prompt = _build_generation_prompt(category, draft_data)

        # Try Gemini 2.0 Flash first
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                return self._via_gemini(prompt, gemini_key)
            except Exception as e:
                logger.warning(f"[DraftingAgent] Gemini failed ({e}), trying Groq")

        # Fallback: Groq LLaMA 3.3 70B
        from rag.llm import llm
        return llm.generate(
            prompt        = prompt,
            system_prompt = _SYSTEM_PROMPT,
            temperature   = 0.15,
            max_tokens    = 2000,
        )

    def _via_gemini(self, prompt: str, api_key: str) -> str:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model    = "gemini-2.0-flash",
            contents = prompt,
            config   = types.GenerateContentConfig(
                system_instruction = _SYSTEM_PROMPT,
                temperature        = 0.15,
                max_output_tokens  = 2000,
            ),
        )
        return response.text

    def cancel_draft(self, session_id: str) -> bool:
        """Cancel and delete an in-progress draft."""
        row = self._load(session_id)
        if row is not None:
            self._delete(session_id)
            return True
        return False


# ── Singleton ──────────────────────────────────────────────────────────────────
drafting_agent = DraftingAgent()