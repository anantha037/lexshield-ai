"""
LexShield AI — Structured Output
==================================
LexShieldResponse wraps every agent response with standardized fields.

Fields:
  answer_text   — full LLM answer (existing)
  summary       — 2-3 sentence plain-English summary (rule-based extraction)
  key_clauses   — list of legal provisions/sections mentioned
  risk          — RiskResult from risk_scorer (score, level, factors, actions)
  suggestions   — plain-English action list from risk_scorer
  citations     — list of Citation objects (existing from LegalAnswer)
  draft         — completed draft text (DraftingAgent only, else empty)
  intent        — classified intent
  session_id    — session identifier
  confidence    — intent confidence
  mode          — which agent/node handled this
  sources_consulted — number of chunks used
  synthesis_note    — retrieval note
  grounding_warning — hallucination warning if any
  rewritten_queries — query expansion list
  reranker_used     — whether NVIDIA NIM reranker was used

Builder:
  build_structured_response() — takes raw agent output, runs risk scorer,
  extracts clauses, builds summary. No extra LLM calls — rule-based only.
  Fast, free, works offline.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from rag.synthesizer import Citation, LegalAnswer
from models.risk_scorer import RiskResult, risk_scorer


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURED RESPONSE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LexShieldResponse:
    # Core answer
    answer_text:       str
    summary:           str
    key_clauses:       list[str]
    suggestions:       list[str]

    # Risk
    risk_score:        float
    risk_level:        str
    risk_factors:      list[str]

    # Citations and draft
    citations:         list[Citation]
    draft:             str

    # Routing metadata
    intent:            str
    session_id:        str
    confidence:        float
    mode:              str
    citation_status:   str

    # RAG metadata
    sources_consulted: int
    synthesis_note:    str
    grounding_warning: str
    rewritten_queries: list[str]
    reranker_used:     bool

    # Case law (populated only when intent == case_law_search)
    case_law_results:  list[dict] = field(default_factory=list)
    
    validation_status: str = "not_applicable"
    scope_status:      str = "in_scope"
    scope_message:     Optional[str] = None

    # Debug scratchpad (populated from final graph state when ?debug=true)
    debug_scratchpad:  Optional[dict] = None

    @staticmethod
    def _safe_str(value) -> str:
        """Ensure value is a clean UTF-8 string — handles bytes from ChromaDB chunks."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if value is None:
            return ""
        try:
            return str(value).encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            return ""

    def to_dict(self) -> dict:
        s = self._safe_str
        return {
            "answer_text":       s(self.answer_text),
            "summary":           s(self.summary),
            "key_clauses":       [s(c) for c in self.key_clauses],
            "suggestions":       [s(c) for c in self.suggestions],
            "risk": {
                "score":   round(self.risk_score, 3),
                "level":   s(self.risk_level),
                "factors": [s(f) for f in self.risk_factors],
            },
            "citations": [
                {
                    "source_number":    c.source_number,
                    "source":           s(c.source),
                    "section":          s(c.section),
                    "section_title":    s(c.section_title),
                    "preview":          s(c.preview),
                    "relevance_score":  c.relevance_score,
                    "era":              s(c.era),
                }
                for c in self.citations
            ],
            "draft":             s(self.draft),
            "intent":            s(self.intent),
            "session_id":        s(self.session_id),
            "confidence":        self.confidence,
            "mode":              s(self.mode),
            "citation_status":   s(self.citation_status),
            "validation_status": s(self.validation_status),
            "scope_status":      s(self.scope_status),
            "scope_message":     s(self.scope_message),
            "kg_sections_used":  [s(k) for k in self.kg_sections_used],
            "sources_consulted": self.sources_consulted,
            "synthesis_note":    s(self.synthesis_note),
            "grounding_warning": s(self.grounding_warning),
            "rewritten_queries": [s(q) for q in self.rewritten_queries],
            "reranker_used":     self.reranker_used,
            "case_law_results":  self.case_law_results,
            "debug_scratchpad":  self.debug_scratchpad,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

_SECTION_CLAUSE_RE = re.compile(
    r'Section\s+\d+[A-Z]?\s+(?:of\s+)?(?:the\s+)?'
    r'(?:Indian Penal Code|Bharatiya Nyaya Sanhita|Code of Criminal Procedure'
    r'|Bharatiya Nagarik Suraksha Sanhita|Indian Evidence Act'
    r'|Bharatiya Sakshya Adhiniyam|Negotiable Instruments Act'
    r'|Protection of Children|POCSO|Consumer Protection Act'
    r'|Information Technology Act|Motor Vehicles Act'
    r'|Transfer of Property Act|Indian Contract Act'
    r'|Prevention of Corruption|NDPS|UAPA|IPC|BNS|CrPC|BNSS|NI Act|BSA)?',
    re.IGNORECASE,
)

_ACT_ONLY_RE = re.compile(
    r'\b(?:Indian Penal Code|Bharatiya Nyaya Sanhita|Code of Criminal Procedure'
    r'|Bharatiya Nagarik Suraksha Sanhita|Indian Evidence Act'
    r'|Bharatiya Sakshya Adhiniyam|Negotiable Instruments Act'
    r'|Protection of Children from Sexual Offences Act'
    r'|Consumer Protection Act|Information Technology Act'
    r'|Motor Vehicles Act|Transfer of Property Act'
    r'|Indian Contract Act|Prevention of Corruption Act'
    r'|Narcotic Drugs.*?Act|Unlawful Activities.*?Act'
    r'|IPC|BNS|CrPC|BNSS|NI Act|BSA|POCSO|NDPS|UAPA)\b',
    re.IGNORECASE,
)

_INLINE_CITE_RE = re.compile(r'\[\d+\]')


def _extract_summary(answer_text: str, max_sentences: int = 3) -> str:
    """
    Extract first 2-3 sentences of answer as summary.
    Strips inline citations like [1], [2] for cleaner reading.
    """
    clean  = _INLINE_CITE_RE.sub("", answer_text).strip()
    sents  = _SENTENCE_SPLIT.split(clean)
    chosen = [s.strip() for s in sents[:max_sentences] if s.strip()]
    return " ".join(chosen)


def _extract_key_clauses(answer_text: str, citations: list[Citation]) -> list[str]:
    """
    Extract unique legal clause references from answer text + citations.
    Returns deduplicated list of clause strings.
    """
    clauses: list[str] = []
    seen:    set[str]  = set()

    # From citations
    for c in citations:
        if c.source == "Knowledge Graph" or c.source == "System":
            continue
        if c.section:
            label = f"Section {c.section}"
            if c.section_title:
                label += f" — {c.section_title}"
            label += f" ({c.source})"
            if label not in seen:
                seen.add(label)
                clauses.append(label)

    # From answer text — regex
    for m in _SECTION_CLAUSE_RE.finditer(answer_text):
        clause = m.group(0).strip().rstrip(".,;")
        if clause not in seen and len(clause) > 8:
            seen.add(clause)
            clauses.append(clause)

    # Acts mentioned without section
    for m in _ACT_ONLY_RE.finditer(answer_text):
        act = m.group(0).strip()
        if act not in seen:
            seen.add(act)
            # Only add if no section-level entry already covers this act
            act_already_covered = any(act.lower() in c.lower() for c in clauses)
            if not act_already_covered:
                clauses.append(act)

    return clauses[:10]  # Cap at 10


def _detect_doc_type_from_intent(intent: str, answer_text: str) -> str:
    """
    Infer doc_type for risk scorer from intent + answer content.
    Risk scorer needs doc_type for base risk calculation.
    """
    if intent == "risk_check":
        text_lower = answer_text.lower()
        if "fir" in text_lower or "criminal" in text_lower:
            return "fir"
        if "cheque" in text_lower or "section 138" in text_lower:
            return "cheque_bounce_notice"
        if "bail" in text_lower:
            return "bail_application"
        if "notice" in text_lower:
            return "legal_notice"
        return "legal_notice"

    intent_doctype_map = {
        "legal_query":        "unknown",
        "document_analysis":  "unknown",
        "draft_request":      "legal_notice",
        "translation_request":"unknown",
        "general":            "unknown",
    }
    return intent_doctype_map.get(intent, "unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_structured_response(
    answer_text:       str,
    intent:            str,
    session_id:        str,
    confidence:        float,
    mode:              str,
    citations:         list[Citation]      = None,
    draft:             str                 = "",
    sources_consulted: int                 = 0,
    synthesis_note:    str                 = "",
    grounding_warning: str                 = "",
    rewritten_queries: Optional[list[str]] = None,
    reranker_used:     bool                = False,
    citation_status:   str                 = "unverified",
    validation_status: str                 = "not_applicable",
    scope_status:      str                 = "in_scope",
    scope_message:     Optional[str]       = None,
    kg_sections_used:  Optional[list[str]] = None,
    doc_type:          str                 = "",
    entities:          dict                = None,
    case_law_results:  list[dict]          = None,
    debug_scratchpad:  Optional[dict]      = None,
) -> LexShieldResponse:
    """
    Build a fully structured LexShieldResponse from raw agent output.
    No extra LLM calls — all extraction is rule-based.

    Args:
        answer_text:       main LLM answer string
        intent:            classified intent
        session_id:        session identifier
        confidence:        intent confidence
        mode:              agent node that handled request
        citation_status:   'cited', 'partial', or 'unverified'
        validation_status: 'passed', 'failed_regenerated', 'failed_returned', 'not_applicable'
        scope_status:      'in_scope', 'out_of_scope', etc.
        scope_message:     reason if out of scope
        kg_sections_used:  list of KG sections referenced
        citations:         Citation list from LegalAnswer
        draft:             completed draft (DraftingAgent only)
        sources_consulted: chunk count used
        synthesis_note:    note from synthesizer
        grounding_warning: hallucination warning
        rewritten_queries: expanded queries
        reranker_used:     whether reranker was used
        doc_type:          document type (for risk scorer)
        entities:          NER entities dict (for risk scorer)

    Returns:
        LexShieldResponse with all fields populated
    """
    citations         = citations         or []
    rewritten_queries = rewritten_queries or []
    entities          = entities          or {}
    case_law_results  = case_law_results  or []
    kg_sections_used  = kg_sections_used  or []

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = _extract_summary(answer_text)

    # ── Key clauses ────────────────────────────────────────────────────────────
    key_clauses = _extract_key_clauses(answer_text, citations)

    # ── Risk scoring ───────────────────────────────────────────────────────────
    effective_doc_type = doc_type or _detect_doc_type_from_intent(intent, answer_text)

    # Skip expensive LLM risk for non-risk intents to save Groq quota
    use_llm_risk = (intent == "risk_check")

    try:
        risk_result: RiskResult = risk_scorer.score(
            text     = answer_text,
            doc_type = effective_doc_type,
            entities = entities,
            use_llm  = use_llm_risk,
        )
    except Exception as e:
        print(f"[StructuredOutput] Risk scorer error: {e}")
        risk_result = RiskResult(
            score               = 0.0,
            level               = "Low",
            factors             = ["Risk scoring unavailable"],
            recommended_actions = [],
        )

    # ── Suggestions ────────────────────────────────────────────────────────────
    suggestions = risk_result.recommended_actions or [
        "Review the legal provisions mentioned above carefully.",
        "Consult a qualified Indian advocate for advice specific to your situation.",
    ]

    # Honor the citation_status derived by the orchestrator.
    # Fall back to citation-list inference only when the caller
    # did not supply an explicit status.
    if citation_status == "unverified":          # default / not overridden by caller
        if citations and not grounding_warning:
            citation_status = "cited"
        elif citations and grounding_warning:
            citation_status = "partial"
        # else stays "unverified"

    return LexShieldResponse(
        answer_text       = answer_text,
        summary           = summary,
        key_clauses       = key_clauses,
        suggestions       = suggestions,
        risk_score        = risk_result.score,
        risk_level        = risk_result.level,
        risk_factors      = risk_result.factors,
        citations         = citations,
        draft             = draft,
        intent            = intent,
        session_id        = session_id,
        confidence        = confidence,
        mode              = mode,
        citation_status   = citation_status,
        validation_status = validation_status,
        sources_consulted = sources_consulted,
        synthesis_note    = synthesis_note,
        grounding_warning = grounding_warning or "",
        rewritten_queries = rewritten_queries,
        reranker_used     = reranker_used,
        case_law_results  = case_law_results,
        debug_scratchpad  = debug_scratchpad,
    )