"""
LexShield AI — Risk Scoring Engine  (Week 2, Day 5)
====================================================
Splits a document into clauses and scores each clause for legal risk.

Risk rules are derived from:
  - Indian Contract Act 1872 (illegal/void clauses)
  - Specific Relief Act 1963 (enforceability)
  - Consumer Protection Act 2019 (unfair contract terms)
  - Kerala Buildings (Lease and Rent Control) Act (rent caps)
  - Code on Wages 2019 (wage deduction limits)
  - Transfer of Property Act 1882 (property rights)
  - Supreme Court precedents on non-compete, arbitration clauses

Each clause gets:
  score        — 0 (no risk) to 100 (very high risk)
  risk_level   — "low" / "medium" / "high" / "critical"
  flags        — list of triggered risk rules
  legal_refs   — relevant legal provisions
  explanation  — plain-English explanation

Overall document risk:
  overall_score  — weighted average of clause scores
  high_risk_count— number of clauses scoring ≥ 70
  summary        — human-readable risk summary
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Risk result dataclasses ───────────────────────────────────────────────────

@dataclass
class ClauseRisk:
    clause_number:  int
    clause_text:    str
    score:          int           # 0-100
    risk_level:     str           # low/medium/high/critical
    flags:          list[str]     = field(default_factory=list)
    legal_refs:     list[str]     = field(default_factory=list)
    explanation:    str           = ""

    def to_dict(self) -> dict:
        return {
            "clause_number": self.clause_number,
            "clause_text":   self.clause_text[:300],   # truncate for API
            "score":         self.score,
            "risk_level":    self.risk_level,
            "flags":         self.flags,
            "legal_refs":    self.legal_refs,
            "explanation":   self.explanation,
        }


@dataclass
class DocumentRisk:
    overall_score:    int
    risk_level:       str
    high_risk_count:  int
    clause_risks:     list[ClauseRisk]  = field(default_factory=list)
    summary:          str               = ""
    doc_type_context: str               = ""

    def to_dict(self) -> dict:
        return {
            "overall_score":   self.overall_score,
            "risk_level":      self.risk_level,
            "high_risk_count": self.high_risk_count,
            "summary":         self.summary,
            "clause_risks":    [c.to_dict() for c in self.clause_risks],
        }


# ── Risk rules ────────────────────────────────────────────────────────────────
# Each rule is a dict:
#   pattern      — regex to match in clause text
#   score        — base risk score (0-100)
#   flag         — short flag name
#   legal_refs   — relevant legal provisions
#   explanation  — plain-English explanation
#   doc_types    — list of doc types this rule applies to (empty = all types)

RISK_RULES: list[dict] = [

    # ── RENTAL AGREEMENT RULES ────────────────────────────────────────────────
    {
        "pattern":     re.compile(
            r'\b(non[- ]?refundable|not\s+refundable|forfeited?\s+deposit|'
            r'deposit.*?shall\s+not\s+be\s+returned)\b',
            re.IGNORECASE
        ),
        "score":       80,
        "flag":        "NON_REFUNDABLE_DEPOSIT",
        "legal_refs":  [
            "Section 10 of the Kerala Buildings (Lease and Rent Control) Act",
            "Section 73 of the Indian Contract Act, 1872",
        ],
        "explanation": (
            "Non-refundable deposit clauses are void under Kerala Rent Control law. "
            "Security deposits must be returned at the end of tenancy. "
            "Such clauses are unenforceable and landlords may face legal action."
        ),
        "doc_types":   ["rental_agreement"],
    },
    {
        "pattern":     re.compile(
            r'\b(landlord\s+(?:may|can|shall|will)\s+(?:enter|inspect|access)'
            r'|right\s+to\s+enter\s+without\s+notice'
            r'|entry\s+without\s+(?:prior\s+)?notice)\b',
            re.IGNORECASE
        ),
        "score":       75,
        "flag":        "ENTRY_WITHOUT_NOTICE",
        "legal_refs":  [
            "Section 108(m) of the Transfer of Property Act, 1882",
            "Article 21 of the Constitution of India (right to privacy)",
        ],
        "explanation": (
            "Landlords cannot enter rented premises without prior notice. "
            "A minimum notice period (typically 24 hours) is required by law. "
            "Unconsented entry may amount to trespass under IPC."
        ),
        "doc_types":   ["rental_agreement"],
    },
    {
        # FIX: removed the trailing \b after % — % is not a word character so
        # \b after it never matches, causing the rule to silently never fire.
        "pattern":     re.compile(
            r'\b(rent\s+(?:shall\s+)?(?:increase|escalate)\s+(?:by\s+)?'
            r'(\d+)\s*%'
            r'|annual\s+increment\s+of\s+(\d+)\s*%)',
            re.IGNORECASE
        ),
        "score":       50,   # computed dynamically based on % value
        "flag":        "EXCESSIVE_RENT_INCREASE",
        "legal_refs":  [
            "Section 5 of the Kerala Buildings (Lease and Rent Control) Act",
            "Section 23 of the Indian Contract Act, 1872",
        ],
        "explanation": (
            "Rent increases exceeding 10% per year may be challengeable under "
            "rent control legislation. Excessive increase clauses may be void."
        ),
        "doc_types":   ["rental_agreement"],
        "dynamic":     True,   # score computed in _compute_dynamic_score()
    },
    {
        "pattern":     re.compile(
            r'\b(tenant\s+shall\s+(?:not|never)\s+(?:sublet|sublease|assign)'
            r'|no\s+subletting\s+(?:shall\s+be\s+)?(?:permitted|allowed))\b',
            re.IGNORECASE
        ),
        "score":       30,
        "flag":        "SUBLETTING_RESTRICTION",
        "legal_refs":  [
            "Section 108(j) of the Transfer of Property Act, 1882",
        ],
        "explanation": (
            "Subletting restrictions are generally permissible but should be "
            "reasonable. Complete blanket restrictions may be challenged."
        ),
        "doc_types":   ["rental_agreement"],
    },

    # ── EMPLOYMENT CONTRACT RULES ─────────────────────────────────────────────
    {
        "pattern":     re.compile(
            r'\b(non[- ]?compete|shall\s+not\s+(?:work|join|engage)\s+with'
            r'|prohibited\s+from\s+(?:joining|working)\s+(?:any\s+)?competitor)\b',
            re.IGNORECASE
        ),
        "score":       65,
        "flag":        "NON_COMPETE_CLAUSE",
        "legal_refs":  [
            "Section 27 of the Indian Contract Act, 1872",
            "Niranjan Shankar Golikari v. Century Spinning Co. (1967) AIR SC 1098",
        ],
        "explanation": (
            "Post-employment non-compete clauses are void under Section 27 of the "
            "Indian Contract Act as restraint of trade. Only in-employment "
            "restrictions are valid. This clause is likely unenforceable."
        ),
        "doc_types":   ["employment_contract"],
    },
    {
        # FIX: broadened pattern to also catch "waives any right to approach courts"
        # and "employee waives right to approach" phrasings from the test document.
        "pattern": re.compile(
            r'('
            r'waiv(?:e|es|ing|ed)\s+(?:any\s+)?(?:right\s+to\s+)?(?:court|legal|judicial)'
            r'|shall\s+not\s+(?:approach|file|initiate)\s+(?:any\s+)?(?:court|legal\s+proceedings)'
            r'|bar(?:red|ring)\s+(?:from\s+)?(?:court|legal\s+proceedings)'
            r'|waiv(?:e|es|ing|ed)\s+(?:any\s+)?(?:right\s+to\s+)?approach\s+courts?'
            r'|right\s+to\s+approach\s+courts?\s+(?:is\s+)?(?:hereby\s+)?waived'
            r'|waives?\s+(?:any\s+)?right\s+to\s+(?:approach|file|seek)'
            r')',
            re.IGNORECASE
        ),
        "score":       95,
        "flag":        "WAIVER_OF_COURT_ACCESS",
        "legal_refs":  [
            "Article 21 of the Constitution of India",
            "Article 32 of the Constitution of India",
            "Section 28 of the Indian Contract Act, 1872",
        ],
        "explanation": (
            "Any clause waiving the right to approach courts is void and "
            "unconstitutional. Article 21 guarantees access to justice and "
            "Section 28 of the Contract Act voids agreements in restraint of "
            "legal proceedings. This clause has no legal effect."
        ),
        "doc_types":   [],   # applies to ALL document types
    },
    {
        "pattern": re.compile(
            r'('
            r'salary\s+(?:may\s+be\s+)?deduct(?:ed)?\s+without\s+notice'
            r'|wages?\s+(?:may\s+be\s+)?withheld?\s+(?:at\s+)?(?:employer\'?s?\s+)?discretion'
            r'|deduction\s+(?:shall\s+be\s+)?made\s+without\s+(?:prior\s+)?notice'
            r'|deduct\s+salary\s+without\s+notice'
            r'|deduct\s+salary\s+without\s+notice\s+at'
            r'|may\s+deduct\s+salary\s+without\s+notice'
            r')',
            re.IGNORECASE
        ),
        "score":       85,
        "flag":        "UNLAWFUL_WAGE_DEDUCTION",
        "legal_refs":  [
            "Section 18 of the Code on Wages, 2019",
            "Section 7 of the Payment of Wages Act, 1936",
        ],
        "explanation": (
            "Wage deductions without written notice and proper authorization are "
            "prohibited under the Code on Wages 2019. Permissible deductions are "
            "exhaustively listed in Section 18 and cannot be expanded by contract."
        ),
        "doc_types":   ["employment_contract"],
    },
    {
        "pattern":     re.compile(
            r'\b(employer\s+(?:may|can|shall)\s+terminate\s+(?:without|with\s+no)\s+'
            r'(?:notice|cause|reason)'
            r'|termination\s+at\s+(?:will|pleasure|discretion)\s+of\s+(?:the\s+)?employer'
            r'|employment\s+(?:may\s+be\s+)?terminated\s+(?:summarily|immediately)\s+'
            r'without\s+(?:cause|notice|reason))\b',
            re.IGNORECASE
        ),
        "score":       70,
        "flag":        "ARBITRARY_TERMINATION",
        "legal_refs":  [
            "Section 25F of the Industrial Disputes Act, 1947",
            "Section 11 of the Code on Wages, 2019",
        ],
        "explanation": (
            "Clauses allowing termination without notice or cause may violate "
            "the Industrial Disputes Act and natural justice principles. "
            "Minimum notice periods are mandated by law."
        ),
        "doc_types":   ["employment_contract"],
    },

    # ── PROPERTY DEED RULES ───────────────────────────────────────────────────
    {
        "pattern":     re.compile(
            r'\b(sold\s+on\s+(?:\")?as\s*is(?:\s*where\s*is)?\s*(?:basis|condition)?'
            r'|seller\s+(?:gives?\s+)?no\s+(?:warranty|guarantee|representation)'
            r'|no\s+(?:warranty|guarantee)\s+(?:is\s+)?(?:given|provided)\s+(?:by\s+)?(?:the\s+)?seller)\b',
            re.IGNORECASE
        ),
        "score":       60,
        "flag":        "NO_TITLE_WARRANTY",
        "legal_refs":  [
            "Section 55(1)(a) of the Transfer of Property Act, 1882",
        ],
        "explanation": (
            "Under Section 55 of the Transfer of Property Act, a seller has an "
            "implied duty to disclose material defects. 'As is' clauses do not "
            "completely negate this liability for hidden defects known to the seller."
        ),
        "doc_types":   ["property_deed"],
    },

    # ── GENERAL / CROSS-DOCUMENT RULES ────────────────────────────────────────
    {
        "pattern":     re.compile(
            r'\b(arbitration\s+(?:shall\s+be\s+)?(?:the\s+)?(?:sole|only|exclusive)\s+remedy'
            r'|disputes?\s+(?:shall\s+be\s+)?(?:exclusively\s+)?(?:referred\s+to|resolved\s+by)\s+arbitration'
            r'|(?:binding\s+)?arbitration\s+clause(?:\s+applies)?)\b',
            re.IGNORECASE
        ),
        "score":       40,
        "flag":        "MANDATORY_ARBITRATION",
        "legal_refs":  [
            "Section 8 of the Arbitration and Conciliation Act, 1996",
            "Vidya Drolia v. Durga Trading Corporation (2021) SC",
        ],
        "explanation": (
            "Mandatory arbitration clauses are generally enforceable in India under "
            "the Arbitration Act 1996, but courts have held that certain "
            "consumer disputes and employment matters cannot be compulsorily "
            "arbitrated. Review if this clause applies appropriately."
        ),
        "doc_types":   [],
    },
    {
        "pattern":     re.compile(
            r'\b(liquidated\s+damages\s+(?:of\s+)?(\d+)\s*%'
            r'|penalty\s+(?:of\s+)?(\d+)\s*%\s+(?:of\s+)?(?:the\s+)?(?:total\s+)?(?:contract\s+)?(?:value|amount)'
            r'|forfeit(?:ure)?\s+of\s+(?:entire|full|all)\s+(?:amount|payment|deposit))\b',
            re.IGNORECASE
        ),
        "score":       55,
        "flag":        "EXCESSIVE_PENALTY",
        "legal_refs":  [
            "Section 74 of the Indian Contract Act, 1872",
            "Fateh Chand v. Balkishan Das (1964) AIR SC 1405",
        ],
        "explanation": (
            "Under Section 74 of the Contract Act, courts can reduce penalty "
            "clauses to reasonable compensation. Forfeiture of entire amounts or "
            "very high penalty percentages may be reduced by courts."
        ),
        "doc_types":   [],
    },
    {
        "pattern":     re.compile(
            r'\b((?:sole\s+)?jurisdiction\s+(?:of\s+)?(?:courts?\s+(?:in|at|of))\s+[A-Z][a-z]+'
            r'|(?:exclusive\s+)?jurisdiction.*?(?:courts?\s+(?:in|at))\s+[A-Z][a-z]+'
            r'|disputes?\s+subject\s+to\s+(?:exclusive\s+)?jurisdiction\s+of)\b',
            re.IGNORECASE
        ),
        "score":       25,
        "flag":        "EXCLUSIVE_JURISDICTION",
        "legal_refs":  [
            "Section 20 of the Code of Civil Procedure, 1908",
        ],
        "explanation": (
            "Exclusive jurisdiction clauses are generally valid in commercial "
            "contracts but can be challenged in consumer matters. Verify that "
            "the chosen court actually has territorial jurisdiction."
        ),
        "doc_types":   [],
    },
    {
        "pattern":     re.compile(
            r'\b(this\s+agreement\s+(?:shall\s+be\s+)?(?:automatically\s+)?renewed'
            r'|auto[- ]?renew(?:al|s|ed)?'
            r'|deemed\s+(?:to\s+be\s+)?renewed\s+automatically)\b',
            re.IGNORECASE
        ),
        "score":       35,
        "flag":        "AUTO_RENEWAL",
        "legal_refs":  [
            "Section 2(b) of the Indian Contract Act, 1872 (acceptance definition)",
        ],
        "explanation": (
            "Auto-renewal clauses bind parties without affirmative consent. "
            "In consumer contracts, these may be challenged. Ensure there is "
            "an explicit opt-out mechanism and advance notice requirement."
        ),
        "doc_types":   [],
    },
    {
        "pattern":     re.compile(
            r'\b((?:completely|fully|absolutely|entirely)\s+indemnif(?:y|ies|ied)'
            r'|indemnif(?:y|ies|ied)\s+(?:and\s+hold\s+harmless\s+)?from\s+(?:any\s+and\s+)?all'
            r'|blanket\s+indemnity)\b',
            re.IGNORECASE
        ),
        "score":       50,
        "flag":        "BROAD_INDEMNITY",
        "legal_refs":  [
            "Section 124 of the Indian Contract Act, 1872",
            "Section 125 of the Indian Contract Act, 1872",
        ],
        "explanation": (
            "Overly broad indemnity clauses covering 'any and all' losses may be "
            "disproportionate. Indian courts scrutinize indemnity clauses and may "
            "limit their scope to foreseeable and reasonable losses."
        ),
        "doc_types":   [],
    },
]


# ── Score thresholds ──────────────────────────────────────────────────────────
def _score_to_level(score: int) -> str:
    if score >= 80: return "critical"
    if score >= 60: return "high"
    if score >= 35: return "medium"
    return "low"


def _compute_dynamic_score(rule: dict, clause: str) -> int:
    """Compute score for dynamic rules (e.g. percentage-based rent increase)."""
    if rule["flag"] == "EXCESSIVE_RENT_INCREASE":
        # Extract percentage
        pct_match = re.search(r'(\d+)\s*%', clause, re.IGNORECASE)
        if pct_match:
            pct = int(pct_match.group(1))
            if pct <= 5:   return 10
            if pct <= 10:  return 20
            if pct <= 15:  return 50
            if pct <= 25:  return 70
            return 85   # > 25% is very high risk
    return rule["score"]


# ── Clause splitter ───────────────────────────────────────────────────────────

def split_into_clauses(text: str) -> list[str]:
    """
    Split document text into individual clauses for per-clause scoring.

    Strategy:
      1. Split on numbered clauses: "1.", "2.", "1)", "a)", "CLAUSE 1"
      2. Split on paragraph breaks if no numbered structure
      3. Cap clause length at 500 words, minimum 10 words
    """
    # Try numbered clause splitting first
    numbered_re = re.compile(
        r'(?:^|\n)\s*(?:\d{1,3}\.|\d{1,3}\)|[a-z]\.|[A-Z]\.|\([a-z]\)|\([A-Z]\)|'
        r'CLAUSE\s+\d+|Article\s+\d+|Paragraph\s+\d+)\s+',
        re.MULTILINE,
    )
    splits = numbered_re.split(text)

    # If we got meaningful splits, use them
    if len(splits) > 3:
        clauses = [s.strip() for s in splits if len(s.strip().split()) >= 10]
        if clauses:
            return clauses[:50]   # max 50 clauses per document

    # Fallback: paragraph splitting
    paragraphs = re.split(r'\n\s*\n', text)
    clauses    = []
    for para in paragraphs:
        para = para.strip()
        words = para.split()
        if len(words) < 10:
            continue
        if len(words) > 500:
            # Split long paragraphs into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            chunk, chunk_words = [], 0
            for sent in sentences:
                sent_words = len(sent.split())
                if chunk_words + sent_words > 150:
                    if chunk:
                        clauses.append(" ".join(chunk))
                    chunk, chunk_words = [sent], sent_words
                else:
                    chunk.append(sent)
                    chunk_words += sent_words
            if chunk:
                clauses.append(" ".join(chunk))
        else:
            clauses.append(para)

    return clauses[:50]


# ── Main risk scorer ──────────────────────────────────────────────────────────

class RiskScorer:
    """
    Scores a document for legal risk at the clause level.

    Usage:
        from models.risk_scorer import risk_scorer
        result = risk_scorer.score(text, doc_type="rental_agreement")
        print(result.overall_score, result.summary)
    """

    def score(
        self,
        text:     str,
        doc_type: str = "",
    ) -> DocumentRisk:
        """
        Score all clauses in a document.
        doc_type: optional label_name from classifier (filters rules by doc type)
        """
        if not text or len(text.strip()) < 50:
            return DocumentRisk(
                overall_score   = 0,
                risk_level      = "low",
                high_risk_count = 0,
                summary         = "Document too short to score.",
            )

        clauses      = split_into_clauses(text)
        clause_risks = []

        for i, clause in enumerate(clauses, start=1):
            cr = self._score_clause(i, clause, doc_type)
            clause_risks.append(cr)

        # Compute overall score
        if not clause_risks:
            overall = 0
        else:
            scores  = [cr.score for cr in clause_risks]
            # Weight: average of top 3 highest-risk clauses + mean of rest
            scores_sorted = sorted(scores, reverse=True)
            top3          = scores_sorted[:3]
            rest          = scores_sorted[3:] or [0]
            overall = int(
                0.6 * (sum(top3) / len(top3)) +
                0.4 * (sum(rest) / len(rest))
            )

        high_risk_count = sum(1 for cr in clause_risks if cr.score >= 70)
        critical_count  = sum(1 for cr in clause_risks if cr.score >= 80)
        overall_level   = _score_to_level(overall)

        # Build summary
        if critical_count > 0:
            summary = (
                f"CRITICAL: {critical_count} clause(s) contain critically risky terms "
                f"(score ≥ 80). Immediate legal review recommended before signing."
            )
        elif high_risk_count > 0:
            summary = (
                f"HIGH RISK: {high_risk_count} clause(s) contain high-risk terms. "
                f"Seek legal advice before signing."
            )
        elif overall >= 35:
            summary = (
                f"MEDIUM RISK: Some clauses require attention. "
                f"Review flagged clauses carefully."
            )
        else:
            summary = (
                f"LOW RISK: No major legal risk clauses detected. "
                f"Standard document review still recommended."
            )

        return DocumentRisk(
            overall_score    = overall,
            risk_level       = overall_level,
            high_risk_count  = high_risk_count,
            clause_risks     = clause_risks,
            summary          = summary,
            doc_type_context = doc_type,
        )

    def _score_clause(
        self,
        clause_num: int,
        clause:     str,
        doc_type:   str,
    ) -> ClauseRisk:
        """Score a single clause against all applicable risk rules."""
        max_score  = 0
        all_flags: list[str]      = []
        all_refs:  list[str]      = []
        all_expls: list[str]      = []

        for rule in RISK_RULES:
            # Check if rule applies to this document type
            applicable_types = rule.get("doc_types", [])
            if applicable_types and doc_type and doc_type not in applicable_types:
                continue

            if not rule["pattern"].search(clause):
                continue

            # Compute score
            if rule.get("dynamic"):
                score = _compute_dynamic_score(rule, clause)
            else:
                score = rule["score"]

            if score > max_score:
                max_score = score

            all_flags.append(rule["flag"])
            all_refs.extend(rule.get("legal_refs",  []))
            all_expls.append(rule.get("explanation", ""))

        # Deduplicate refs
        seen_refs: set[str] = set()
        deduped_refs = []
        for ref in all_refs:
            if ref not in seen_refs:
                seen_refs.add(ref)
                deduped_refs.append(ref)

        explanation = " | ".join(all_expls) if all_expls else ""

        return ClauseRisk(
            clause_number = clause_num,
            clause_text   = clause[:300],
            score         = max_score,
            risk_level    = _score_to_level(max_score),
            flags         = all_flags,
            legal_refs    = deduped_refs,
            explanation   = explanation,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
risk_scorer = RiskScorer()