"""
LexShield AI — Drafting Agent
================================
Multi-turn workflow for generating Indian legal documents.

Supported document types:
  fir                   → FIR under Section 173 BNSS / 154 CrPC
  legal_notice_ni       → Legal Notice under Section 138 NI Act (cheque bounce)
  legal_notice_contract → Legal Notice under Indian Contract Act 1872
  rental_agreement      → Rental Agreement under Transfer of Property Act
  legal_notice_generic  → General legal notice

Workflow (3 turns):
  Turn 1 (Stage 0→1): Detect doc_type from query → ask incident details
  Turn 2 (Stage 1→2): Store incident details     → ask party details
  Turn 3 (Stage 2→3): Store party details        → generate draft via LLM

LLM used:
  Gemini 2.0 Flash (google-genai SDK) — 1M tokens/day free tier.
  Saves Groq quota for RAG queries.
  Falls back to Groq if Gemini fails.

State:
  _draft_sessions dict keyed by session_id.
  Persists across LangGraph invocations within same server process.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# DRAFT STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DraftState:
    doc_type:  str
    stage:     int  = 1        # 1=collecting incident, 2=collecting parties, 3=done
    collected: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_DOC_PATTERNS = [
    (re.compile(r'\bfir\b|first\s+information\s+report|police\s+complaint', re.IGNORECASE),           "fir"),
    (re.compile(r'\bcheque\s+bounce\b|section\s+138\b|ni\s+act|negotiable\s+instrument', re.IGNORECASE), "legal_notice_ni"),
    (re.compile(r'\bbreach\s+of\s+contract\b|contract\s+(act|dispute|violation|breach)', re.IGNORECASE), "legal_notice_contract"),
    (re.compile(r'\brental\s+agreement\b|lease\s+agreement\b|landlord|tenant', re.IGNORECASE),         "rental_agreement"),
    (re.compile(r'\blegal\s+notice\b', re.IGNORECASE),                                                 "legal_notice_generic"),
]

_DOC_LABELS = {
    "fir":                    "FIR (First Information Report) under Section 173 BNSS",
    "legal_notice_ni":        "Legal Notice under Section 138 NI Act (Cheque Bounce)",
    "legal_notice_contract":  "Legal Notice for Breach of Contract",
    "rental_agreement":       "Rental Agreement",
    "legal_notice_generic":   "Legal Notice",
}


def _detect_doc_type(query: str) -> Optional[str]:
    for pattern, doc_type in _DOC_PATTERNS:
        if pattern.search(query):
            return doc_type
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_STAGE1_QUESTIONS: dict[str, str] = {
    "fir": (
        "To draft your FIR, please provide the following incident details:\n\n"
        "1. What happened? (describe the offence clearly)\n"
        "2. When did it happen? (date and approximate time)\n"
        "3. Where did it happen? (full address or location)\n"
        "4. How did it happen? (any additional context or method)"
    ),
    "legal_notice_ni": (
        "To draft your cheque bounce legal notice, please provide:\n\n"
        "1. Cheque details (cheque number, amount in Rs., date on cheque, bank name and branch)\n"
        "2. When was it dishonoured? (date of return and reason given by bank)\n"
        "3. What was the cheque issued for? (loan repayment, goods, services etc.)\n"
        "4. Have you previously communicated with the drawer about this?"
    ),
    "legal_notice_contract": (
        "To draft your legal notice for breach of contract, please provide:\n\n"
        "1. What was the contract about? (nature and purpose)\n"
        "2. When was the contract executed? (date)\n"
        "3. What specific obligation was breached? (describe clearly)\n"
        "4. What loss or damage has been caused? (in Rs. if applicable)"
    ),
    "rental_agreement": (
        "To draft your rental agreement, please provide the property details:\n\n"
        "1. Full address of the property being rented\n"
        "2. Monthly rent amount (in Rs.)\n"
        "3. Lease duration (e.g. 11 months, 1 year)\n"
        "4. Security deposit amount (in Rs.)"
    ),
    "legal_notice_generic": (
        "To draft your legal notice, please provide the dispute details:\n\n"
        "1. What is the dispute about? (describe clearly)\n"
        "2. When did this issue arise? (date)\n"
        "3. What relief or action are you seeking?\n"
        "4. What deadline do you want to give? (e.g. 15 days, 30 days from receipt)"
    ),
}

_STAGE2_QUESTIONS: dict[str, str] = {
    "fir": (
        "Thank you. Now please provide the party and relief details:\n\n"
        "1. Your full name (complainant)\n"
        "2. Your full address\n"
        "3. Your contact number\n"
        "4. Accused person's name and address (write 'unknown' if not identified)\n"
        "5. Relief sought (e.g. arrest of accused, recovery of property, registration of case)"
    ),
    "legal_notice_ni": (
        "Thank you. Now please provide the party details:\n\n"
        "1. Your full name (payee / person who received the cheque)\n"
        "2. Your full address\n"
        "3. Drawer's full name (person who issued the cheque)\n"
        "4. Drawer's full address\n"
        "5. Total amount demanded (principal + interest if any)\n"
        "6. Time limit you want to give for payment (typically 15 days from receipt)"
    ),
    "legal_notice_contract": (
        "Thank you. Now please provide the party details:\n\n"
        "1. Your full name (aggrieved party)\n"
        "2. Your full address\n"
        "3. Opposite party's full name\n"
        "4. Opposite party's full address\n"
        "5. Relief or remedy sought (compensation amount, specific performance, etc.)\n"
        "6. Time limit to comply (e.g. 15 days, 30 days from receipt of notice)"
    ),
    "rental_agreement": (
        "Thank you. Now please provide the party details:\n\n"
        "1. Landlord's full name and complete address\n"
        "2. Tenant's full name and complete address\n"
        "3. Agreement start date\n"
        "4. Any special conditions? (maintenance responsibility, pet policy, parking, etc. — write 'none' if not applicable)"
    ),
    "legal_notice_generic": (
        "Thank you. Now please provide the party details:\n\n"
        "1. Your full name (sender of notice)\n"
        "2. Your full address\n"
        "3. Recipient's full name\n"
        "4. Recipient's full address\n"
        "5. Consequences if demand is not met (legal suit, criminal complaint, etc.)"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are LexShield AI, an expert Indian legal document drafter. "
    "Draft precise, professional Indian legal documents using correct legal terminology. "
    "Follow the exact format and sections requested. Use Indian legal conventions. "
    "Always include proper headings, date placeholders, and signature blocks. "
    "Do not add disclaimers inside the document itself."
)


def _build_generation_prompt(doc_type: str, collected: dict) -> str:
    s1 = collected.get("stage1", "")
    s2 = collected.get("stage2", "")

    templates = {

        "fir": f"""Draft a complete, professional FIR (First Information Report) under Section 173 of the Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS) — formerly Section 154 CrPC — using the details below.

INCIDENT DETAILS PROVIDED BY COMPLAINANT:
{s1}

PARTY AND RELIEF DETAILS:
{s2}

Structure the FIR with exactly these sections:
1. HEADER — "FIRST INFORMATION REPORT" centered, bold
   - To: The Station House Officer, [Police Station Name], [District]
   - FIR No.: _______ / [Year]
   - Date: _____________ Time: _____________
2. SECTIONS OF LAW — infer appropriate IPC/BNS sections from the incident
3. TYPE OF INFORMATION — First/Written/Oral (use Written)
4. COMPLAINANT DETAILS — name, address, contact from party details
5. ACCUSED DETAILS — as provided
6. INCIDENT DESCRIPTION — detailed factual narrative using incident details
7. WITNESSES — as provided or "To be ascertained"
8. PROPERTY INVOLVED — list if mentioned, else "Not Applicable"
9. RELIEF REQUESTED — as provided
10. DECLARATION — standard declaration by complainant that information is true
11. COMPLAINANT SIGNATURE BLOCK — name, date, place
12. OFFICER RECEIVING FIR — signature block

Be complete and use formal legal language throughout.""",

        "legal_notice_ni": f"""Draft a formal Legal Notice under Section 138 of the Negotiable Instruments Act, 1881 using the details below.

CHEQUE AND DISHONOUR DETAILS:
{s1}

PARTY DETAILS:
{s2}

Structure the Legal Notice with:
1. HEADER — "LEGAL NOTICE" centered
2. DATE — [Date of Notice]
3. TO — recipient's name and full address (from party details)
4. SUBJECT — "Legal Notice under Section 138 of the Negotiable Instruments Act, 1881"
5. SALUTATION — Sir/Madam
6. PARA 1 — Introduction of sender and relationship/transaction
7. PARA 2 — Cheque details: number, amount, date, bank, issued for what purpose
8. PARA 3 — Dishonour details: date of return, reason given by bank
9. PARA 4 — Legal position: Section 138 NI Act, criminal liability
10. PARA 5 — Demand: payment of full amount within [time limit] of receipt of this notice
11. PARA 6 — Consequence: criminal complaint under Section 138 NI Act without further notice
12. CLOSING — Yours faithfully
13. SENDER SIGNATURE BLOCK — name, address, date

Use precise legal language. Mention that demand is being made within 30 days of dishonour as required by law.""",

        "legal_notice_contract": f"""Draft a formal Legal Notice for Breach of Contract under the Indian Contract Act, 1872 using the details below.

CONTRACT AND BREACH DETAILS:
{s1}

PARTY DETAILS:
{s2}

Structure with:
1. HEADER — "LEGAL NOTICE"
2. DATE — [Date]
3. TO — opposite party name and address
4. SUBJECT — "Legal Notice for Breach of Contract under the Indian Contract Act, 1872"
5. SALUTATION
6. PARA 1 — Identification of sender, contract details, date of execution
7. PARA 2 — Nature of contractual obligation that was breached
8. PARA 3 — Specific breach and how it occurred
9. PARA 4 — Loss and damage caused (mention amount if provided)
10. PARA 5 — Legal basis: Sections 73 and 74 of the Indian Contract Act 1872
11. PARA 6 — Demand: specific relief within [time limit] of receipt
12. PARA 7 — Consequence: civil suit for damages and specific performance without further notice
13. CLOSING and SIGNATURE BLOCK""",

        "rental_agreement": f"""Draft a complete Rental Agreement (Lease Deed) under the Transfer of Property Act, 1882 using the details below.

PROPERTY AND RENT DETAILS:
{s1}

PARTY DETAILS:
{s2}

Structure the agreement with:
1. TITLE — "RENTAL AGREEMENT / LEASE DEED"
2. DATE AND PLACE of execution
3. PARTIES — "This Agreement is made between:" — Landlord and Tenant with full details
4. RECITALS — property description and purpose of lease
5. CLAUSES:
   Clause 1: Lease Period — start date, end date, renewal terms
   Clause 2: Monthly Rent — amount, due date, mode of payment
   Clause 3: Security Deposit — amount, conditions for refund
   Clause 4: Tenant Obligations — maintenance, no subletting, no structural changes, timely rent
   Clause 5: Landlord Obligations — peaceful possession, essential services
   Clause 6: Use of Premises — residential/commercial purpose only
   Clause 7: Termination — notice period (minimum 1 month)
   Clause 8: Dispute Resolution — jurisdiction of courts at [City]
   Clause 9: Governing Law — laws of India
6. SIGNATURE BLOCKS — Landlord, Tenant, Witness 1, Witness 2 with date and place""",

        "legal_notice_generic": f"""Draft a formal Legal Notice using the details below.

DISPUTE DETAILS:
{s1}

PARTY DETAILS:
{s2}

Structure with:
1. HEADER — "LEGAL NOTICE"
2. DATE — [Date]
3. TO — recipient name and address
4. SUBJECT — brief subject line
5. SALUTATION
6. PARA 1 — Introduction of sender
7. PARA 2 — Facts of the dispute in chronological order
8. PARA 3 — Legal basis for the claim
9. PARA 4 — Specific demand with time limit
10. PARA 5 — Consequences of non-compliance
11. CLOSING — Yours faithfully
12. SENDER SIGNATURE BLOCK — name, address, date""",
    }

    return templates.get(doc_type, templates["legal_notice_generic"])


# ═══════════════════════════════════════════════════════════════════════════════
# DRAFTING AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class DraftingAgent:

    def __init__(self):
        # { session_id: DraftState }
        self._draft_sessions: dict[str, DraftState] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def has_active_draft(self, session_id: str) -> bool:
        """True if this session has an in-progress draft (stages 1 or 2)."""
        state = self._draft_sessions.get(session_id)
        return state is not None and state.stage < 3

    def handle(self, query: str, session_id: str) -> dict:
        """
        Main entry point for drafting workflow.

        Returns:
          answer:   str   — question to ask user, or completed draft
          stage:    int   — current stage after this call
          doc_type: str
          complete: bool  — True when draft is fully generated
          draft:    str   — populated only when complete=True
        """
        state = self._draft_sessions.get(session_id)

        # ── No active draft — start new one ────────────────────────────────────
        if state is None:
            doc_type = _detect_doc_type(query)

            if doc_type is None:
                return self._unknown_doc_type_response()

            label = _DOC_LABELS.get(doc_type, doc_type)
            self._draft_sessions[session_id] = DraftState(doc_type=doc_type, stage=1)

            return {
                "answer":   f"I'll help you draft a **{label}**.\n\n{_STAGE1_QUESTIONS[doc_type]}",
                "stage":    1,
                "doc_type": doc_type,
                "complete": False,
                "draft":    "",
            }

        # ── Stage 1 complete — store incident details, ask party details ────────
        if state.stage == 1:
            state.collected["stage1"] = query
            state.stage = 2

            return {
                "answer":   _STAGE2_QUESTIONS[state.doc_type],
                "stage":    2,
                "doc_type": state.doc_type,
                "complete": False,
                "draft":    "",
            }

        # ── Stage 2 complete — store party details, generate draft ──────────────
        if state.stage == 2:
            state.collected["stage2"] = query
            state.stage = 3

            print(f"[DraftingAgent] Generating {state.doc_type} draft for session {session_id[:8]}")
            draft_text = self._generate_draft(state.doc_type, state.collected)

            # Clear session
            del self._draft_sessions[session_id]

            return {
                "answer": (
                    f"Here is your drafted document:\n\n"
                    f"---\n\n{draft_text}\n\n---\n\n"
                    "⚠️ This is an AI-generated draft. Please review with a qualified lawyer before use."
                ),
                "stage":    3,
                "doc_type": state.doc_type,
                "complete": True,
                "draft":    draft_text,
            }

        # ── Unexpected state — reset ────────────────────────────────────────────
        self._draft_sessions.pop(session_id, None)
        return {
            "answer":   "Something went wrong with the draft session. Please start again by describing the document you need.",
            "stage":    0,
            "doc_type": "",
            "complete": False,
            "draft":    "",
        }

    def cancel_draft(self, session_id: str) -> bool:
        """Cancel an in-progress draft and clear session state."""
        if session_id in self._draft_sessions:
            del self._draft_sessions[session_id]
            return True
        return False

    # ── Document generation ────────────────────────────────────────────────────

    def _generate_draft(self, doc_type: str, collected: dict) -> str:
        """
        Generate the legal document via LLM.
        Uses Gemini 2.0 Flash (1M tokens/day) to preserve Groq quota.
        Falls back to Groq if Gemini is unavailable.
        """
        prompt = _build_generation_prompt(doc_type, collected)

        # Try Gemini first
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                return self._generate_via_gemini(prompt, gemini_key)
            except Exception as e:
                print(f"[DraftingAgent] Gemini failed ({e}), falling back to Groq")

        # Fallback to Groq
        from rag.llm import llm
        return llm.generate(
            prompt        = prompt,
            system_prompt = _SYSTEM_PROMPT,
            temperature   = 0.2,
            max_tokens    = 1500,
        )

    def _generate_via_gemini(self, prompt: str, api_key: str) -> str:
        """Call Gemini 2.0 Flash via google-genai SDK."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model    = "gemini-2.0-flash",
            contents = prompt,
            config   = types.GenerateContentConfig(
                system_instruction = _SYSTEM_PROMPT,
                temperature        = 0.2,
                max_output_tokens  = 1500,
            ),
        )
        return response.text

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _unknown_doc_type_response(self) -> dict:
        return {
            "answer": (
                "I can help you draft the following Indian legal documents:\n\n"
                "• FIR (First Information Report)\n"
                "• Legal Notice for Cheque Bounce (Section 138 NI Act)\n"
                "• Legal Notice for Breach of Contract\n"
                "• Rental Agreement\n"
                "• General Legal Notice\n\n"
                "Please specify which document you need. For example:\n"
                "\"Draft an FIR for theft\" or \"Draft a cheque bounce legal notice\""
            ),
            "stage":    0,
            "doc_type": "",
            "complete": False,
            "draft":    "",
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
drafting_agent = DraftingAgent()