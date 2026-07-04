"""
LexShield AI — LangGraph Multi-Agent Graph  (Week 3, Day 2 Session 2 — Updated)
==================================================================================
Changes from previous session:

1. rights_node added as 8th node.
   Routes on intent == "rights_check".
   Calls RightsAgent.get_rights_with_rag_enrichment() -> structured rights guide
   from data/rights_guide.json merged with live RAG context.

2. AgentState gains "case_law_result" field (dict) for case_law_node output.

3. route_by_intent Priority 2 updated:
   "rights_check" is NOT redirected to multilingual_node even for non-English
   queries — the rights guide content is English, and translation is handled
   by the response layer. Future: translate formatted rights response.

4. All nodes registered and wired to END.

Graph topology (8 terminal nodes):
---------------------------------------------------------------------------
[START]
   │
   ▼
classify_intent_node  <- detect_language() + intent_classifier.classify()
   │
   ▼  route_by_intent() conditional edge
┌─────────────────────────────────────────────────────────────────────────┐
│ legal_rag_node       │ document_analysis_node │ case_law_node           │
│ risk_check_node      │ draft_node             │ multilingual_node       │
│ rights_node          │ general_node           │                         │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ▼
[END]
---------------------------------------------------------------------------

Priority order in route_by_intent:
  1. Active draft session in SQLite        -> draft_node
  2. Non-English source_language + legal/risk/general intent -> multilingual_node
  3. Intent map                            -> correct agent node
"""

import os
import logging

logger = logging.getLogger(__name__)
import re
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver


# ═══════════════════════════════════════════════════════════════════════════════
# SCRATCHPAD KEY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SCRATCH_DETECTED_SECTIONS = "detected_sections"   # list[str]
SCRATCH_DETECTED_ACTS     = "detected_acts"        # list[str]
SCRATCH_JURISDICTION      = "jurisdiction"          # str
SCRATCH_DOC_TYPE          = "doc_type"              # str
SCRATCH_QUERY_COMPLEXITY  = "query_complexity"      # "simple" | "complex"


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL REGEX CONSTANTS  (compiled once at import, not per-query)
# ═══════════════════════════════════════════════════════════════════════════════

_SEC_RE = re.compile(
    r'\b[Ss]ections?\s*\.?\s*(\d{1,4}[A-Za-z]?)\b',
    re.IGNORECASE,
)

_ACT_RE = re.compile(
    r'\b(?:Indian Penal Code|Bharatiya Nyaya Sanhita'
    r'|Code of Criminal Procedure|Bharatiya Nagarik Suraksha Sanhita'
    r'|Indian Evidence Act|Bharatiya Sakshya Adhiniyam'
    r'|Negotiable Instruments Act|Protection of Children from Sexual Offences Act'
    r'|Consumer Protection Act|Information Technology Act'
    r'|Motor Vehicles Act|Transfer of Property Act'
    r'|Indian Contract Act|Prevention of Corruption Act'
    r'|Narcotic Drugs and Psychotropic Substances Act'
    r'|Unlawful Activities \(Prevention\) Act'
    r'|IPC|BNS|CrPC|BNSS|NI Act|BSA|POCSO|NDPS|UAPA)\b',
    re.IGNORECASE,
)

_JURISDICTION_RE = re.compile(
    r'\b(?:'
    # 28 States
    r'Andhra Pradesh|Arunachal Pradesh|Assam|Bihar|Chhattisgarh'
    r'|Goa|Gujarat|Haryana|Himachal Pradesh|Jharkhand'
    r'|Karnataka|Kerala|Madhya Pradesh|Maharashtra|Manipur'
    r'|Meghalaya|Mizoram|Nagaland|Odisha|Punjab'
    r'|Rajasthan|Sikkim|Tamil Nadu|Telangana|Tripura'
    r'|Uttar Pradesh|Uttarakhand|West Bengal'
    # 8 Union Territories
    r'|Andaman and Nicobar Islands|Chandigarh|Dadra and Nagar Haveli and Daman and Diu'
    r'|Delhi|Jammu and Kashmir|Ladakh|Lakshadweep|Puducherry'
    r')\b',
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# POSTGRESQL CHECKPOINTER
# ═══════════════════════════════════════════════════════════════════════════════

from agents.pg_sessions import pool as _pg_pool

checkpointer = PostgresSaver(_pg_pool)
checkpointer.setup()  # Idempotent: creates checkpoint tables if missing


# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    query:            str
    session_id:       str

    # ── Classification ─────────────────────────────────────────────────────────
    intent:           str
    confidence:       float

    # ── RAG / NER / Risk outputs ───────────────────────────────────────────────
    rag_result:       dict
    ner_result:       dict
    risk_result:      dict

    # ── Drafting ───────────────────────────────────────────────────────────────
    draft_stage:      str
    draft_data:       dict

    # ── Case law ───────────────────────────────────────────────────────────────
    case_law_result:  dict    # {query, results, total_found} from search_and_summarize

    # ── Rights guide ───────────────────────────────────────────────────────────
    rights_category:  str     # detected category: tenant/employee/consumer/women/bail
    rights_result:    dict    # enriched rights dict from RightsAgent

    # ── Diagnostics ────────────────────────────────────────────────────────────
    pipeline_depth:   int
    rag_grade:        str     # "good" | "poor" | ""

    # ── Multilingual ───────────────────────────────────────────────────────────
    source_language:  str     # ISO 639-1: "en", "ml", "hi", "ta", "te", ...

    # ── Output ─────────────────────────────────────────────────────────────────
    scope_status:     str     # "in_scope" | "out_of_scope"
    scope_message:    str
    response:         str
    error:            str
    validation_status: str    # "passed" | "failed_regenerated" | "failed_returned" | "not_applicable"

    # ── Shared scratchpad ──────────────────────────────────────────────────────
    scratchpad:       dict    # cross-agent shared memory, fresh per invoke


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: INTENT CLASSIFIER  (+ language detection)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_intent_node(state: AgentState) -> dict:
    """
    Entry node. Always runs first.

    1. detect_language(query) — Unicode fast-path + langdetect.
       Stores ISO 639-1 code in state["source_language"].
    2. intent_classifier.classify_with_llm(query, groq_client) — LLM-first
       classification with regex override pre-filter and fallback chain.
       Returns LLMIntentResult (primary) or IntentResult (fallback).
    3. Entity fields (sections, acts, jurisdiction, complexity) are read from
       the result directly.  If they are empty (override fast-path or fallback),
       the module-level regex constants run as a backup extraction pass.
    4. intent_reasoning is written to scratchpad for JSONL logging.
    """
    from agents.intent_classifier  import intent_classifier
    from agents.multilingual_agent import detect_language
    from rag.llm                   import llm as groq_client

    query = state.get("query", "").strip()
    if not query:
        return {
            "intent":          "general",
            "confidence":      0.0,
            "source_language": "en",
            "pipeline_depth":  1,
        }

    # ── Priority 0: active draft session — short-circuit to draft_node ─────────
    _session_id_early = state.get("session_id", "")
    if _session_id_early:
        try:
            from agents.drafting_agent import drafting_agent as _da
            if _da.has_active_draft(_session_id_early, query):
                _row = _da._load(_session_id_early)
                _draft_stage = _row["stage"] if _row else ""

                # NOTE: confirm / cancel / correction handling at the CONFIRM
                # stage lives in DraftingAgent._handle_confirm — the single
                # source of truth.  A previous graph-level branch here keyed
                # on an "AWAITING_CONFIRMATION" stage that DraftingAgent never
                # persists; it was dead code and has been removed.
                logger.debug(
                    f"[Graph] classify_intent_node -> draft stage={_draft_stage} "
                    f"session={_session_id_early[:8]}… -> short-circuit to draft_request"
                )
                return {
                    "intent":          "draft_request",
                    "confidence":      1.0,
                    "source_language": "en",
                    "pipeline_depth":  state.get("pipeline_depth", 0) + 1,
                    "scratchpad":      dict(state.get("scratchpad", {})),
                }
        except Exception as _e:
            logger.exception(f"[Graph] classify_intent_node active-draft check failed (non-fatal)")
    # ── End short-circuit ──────────────────────────────────────────────────────
    
    ui_language = state.get("source_language", "en")
    detected_language = detect_language(query)

    final_language = ui_language if ui_language != "en" else detected_language

    if final_language != "en":
        logger.debug(f"[Graph] classify_intent_node -> final language: {final_language!r}")

    # Primary: Tool-calling classification (new) -> LLM classification (existing fallback)
    # Step 1: Try tool-calling via bound_llm (returns None on failure/no tool_calls)
    result = intent_classifier.classify_with_tool_calls(query, state.get("session_id", ""))

    # Step 2: If tool-calling didn't produce a result, fall back to existing LLM classifier
    if result is None:
        try:
            result = intent_classifier.classify_with_llm(query, groq_client)
        except Exception as e:
            logger.exception("[Graph] classify_intent_node LLM call failed, falling back to general")
            return {
                "intent": "general",
                "confidence": 0.0,
                "detected_sections": [],
                "detected_acts": [],
                "source_language": final_language,
            }
    logger.debug(
        f"[Graph] classify_intent_node -> intent={result.intent!r} "
        f"conf={result.confidence:.2f} lang={final_language!r}"
    )

    # ── Read entity fields (same attribute names on LLMIntentResult & IntentResult)
    detected_sections = list(result.detected_sections)
    detected_acts     = list(result.detected_acts)
    jurisdiction      = result.jurisdiction
    complexity        = result.query_complexity
    reasoning         = getattr(result, "reasoning", "")

    # ── Regex fallback entity extraction ──────────────────────────────────────
    # Runs when entity fields are empty: override fast-path returned no entities,
    # or the LLM call fell back to classify() which leaves fields at defaults.
    if not detected_sections and not detected_acts and not jurisdiction:
        detected_sections = list(set(_SEC_RE.findall(query)))
        detected_acts     = list(set(m.group(0) for m in _ACT_RE.finditer(query)))
        juri_match        = _JURISDICTION_RE.search(query)
        jurisdiction      = juri_match.group(0) if juri_match else ""
        complexity        = "complex" if (len(detected_sections) > 1 or len(detected_acts) > 1) else "simple"

    # ── Scratchpad population ──────────────────────────────────────────────────
    # IMPORTANT: always start with a FRESH scratchpad.  Do NOT read from
    # state.get("scratchpad", {}) — the PostgresSaver checkpointer persists
    # AgentState across invoke() calls on the same thread_id, so stale
    # detected_sections / detected_acts from a previous turn would bleed
    # into the current turn's retrieval and cause cross-query contamination.
    scratchpad = {}

    scratchpad[SCRATCH_DETECTED_SECTIONS] = detected_sections
    scratchpad[SCRATCH_DETECTED_ACTS]     = detected_acts
    scratchpad[SCRATCH_JURISDICTION]      = jurisdiction
    scratchpad[SCRATCH_QUERY_COMPLEXITY]  = complexity
    scratchpad["intent_reasoning"]        = reasoning   # for JSONL logger

    if detected_sections or detected_acts or jurisdiction:
        logger.debug(
            f"[Graph] scratchpad -> sections={detected_sections} "
            f"acts={detected_acts} jurisdiction={jurisdiction!r} "
            f"complexity={complexity!r}"
        )

    return {
        "intent":          result.intent,
        "confidence":      result.confidence,
        "source_language": final_language,
        "pipeline_depth":  state.get("pipeline_depth", 0) + 1,
        "scratchpad":      scratchpad,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge after classify_intent_node.

    Priority:
      1. Active SQLite draft -> draft_node
      2. Non-English + legal/risk/general intent -> multilingual_node
      3. Intent map -> correct node
    """
    from agents.drafting_agent import drafting_agent

    session_id      = state.get("session_id", "")
    source_language = state.get("source_language", "en")
    intent          = state.get("intent", "general")

    # Priority 0: response already handled (cancellation / re-prompt)
    if intent == "_draft_handled":
        logger.debug("[Graph] route -> _draft_handled -> general_node (response pre-set)")
        return "general_node"

    # Priority 1: active draft
    if session_id and drafting_agent.has_active_draft(session_id, state.get("query", "")):
        logger.debug(f"[Graph] route -> active draft -> draft_node")
        return "draft_node"

    # Priority 2: non-English auto-detection or EXPLICIT UI selection
    # We check if source_language is NOT 'en' (which will catch 'ml', 'hi', etc. passed from the UI)
    _multilingual_eligible = {"legal_query", "risk_check", "general"}
    if source_language != "en" and intent in _multilingual_eligible:
        logger.debug(
            f"[Graph] route -> non-English {source_language!r} "
            f"intent={intent!r} -> multilingual_node"
        )
        return "multilingual_node"

    # Priority 3: intent map
    _map: dict[str, str] = {
        "legal_query":          "legal_rag_node",
        "document_analysis":    "document_analysis_node",
        "draft_request":        "draft_node",
        "risk_check":           "risk_check_node",
        "translation_request":  "multilingual_node",
        "case_law_search":      "case_law_node",
        "rights_check":         "rights_node",
        "general":              "general_node",
    }
    node = _map.get(intent, "general_node")
    logger.debug(f"[Graph] route -> intent={intent!r} -> {node}")
    return node


def route_after_rights(state: AgentState) -> str:
    """
    Conditional edge after rights_node.

    If rights_node found no matching category (scope_status == "out_of_scope"
    and rights_category is empty), reroute to legal_rag_node so the query
    still gets a real answer instead of a dead-end menu message.

    Loop guard: rights_node sets scratchpad["rights_fallback_used"] = True
    on this exact path, so this function never routes the same query through
    rights_node -> legal_rag_node -> rights_node more than once.
    """
    scratchpad = state.get("scratchpad", {})
    no_category_matched = (
        state.get("scope_status") == "out_of_scope"
        and not state.get("rights_category")
    )
    already_fell_back = scratchpad.get("rights_fallback_used", False)

    if no_category_matched and not already_fell_back:
        logger.debug("[Graph] route_after_rights -> no category matched -> legal_rag_node")
        return "legal_rag_node"

    return "end"


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: LEGAL RAG  (+ optional case law enrichment)
# ═══════════════════════════════════════════════════════════════════════════════

def legal_rag_node(state: AgentState) -> dict:
    """
    Intent: legal_query (English queries — non-English -> multilingual_node)

    Flow: NER -> KG enrich -> RAG pipeline -> optional case law enrichment
    """
    from rag.pipeline            import rag_pipeline
    from nlp.ner_pipeline        import run_ner
    from rag.knowledge_graph     import enrich_retrieval
    from agents.case_law_agent   import (
        enrich_rag_response_with_case_law,
        ENABLE_CASE_LAW_ENRICHMENT,
    )
    from agents.memory import session_memory

    query         = state.get("query", "")
    session_id    = state.get("session_id", "")
    context_block = state.get("rag_result", {}).get("context_block", "")

    # ── Bind session_id into ContextVar so query_rewriter can read it ─────────
    # Must run before any RAG/rewriter call.  Safe when session_id is empty.
    if session_id:
        session_memory.set_session_context(session_id)

    # ── Scratchpad reader: use current turn's scratchpad (set by classify_intent_node)
    # Copy it so mutations in this node don't affect the upstream state dict.
    scratchpad = dict(state.get("scratchpad", {}))
    jurisdiction = scratchpad.get(SCRATCH_JURISDICTION, "")
    if jurisdiction and context_block:
        context_block = f"[JURISDICTION CONTEXT: {jurisdiction}]\n{context_block}"
    elif jurisdiction:
        context_block = f"[JURISDICTION CONTEXT: {jurisdiction}]"

    logger.debug("[Graph] legal_rag_node -> NER + KG + RAG")


    # NER
    ner_out: dict        = {"entities": []}
    ner_sections: list[str] = []
    try:
        ner_out = run_ner(query[:2000])
        for ent in ner_out.get("entities", []):
            if ent.get("label") in ("SECTION", "LEGAL_SECTION"):
                src = ent.get("source", "")
                txt = ent.get("text", "")
                if src and txt:
                    ner_sections.append(f"{src}_{txt}")
                elif txt:
                    ner_sections.append(txt)
    except Exception as e:
        logger.exception(f"[Graph] legal_rag_node NER warning")

    # ── Scratchpad writer: overwrite with current-turn NER sections ────────
    # IMPORTANT: do NOT merge with existing_sections from the scratchpad.
    # classify_intent_node already wrote its regex/LLM detections into the
    # scratchpad; those are from the CURRENT query and are fine.  NER sections
    # are additive to those.  We combine them but never carry over from a
    # previous turn — classify_intent_node now starts with a fresh scratchpad.
    intent_sections = scratchpad.get(SCRATCH_DETECTED_SECTIONS, [])
    merged = list(set(intent_sections + ner_sections))
    scratchpad[SCRATCH_DETECTED_SECTIONS] = merged

    # KG enrichment
    kg_chunks: list[dict] = []
    if ner_sections:
        try:
            from rag.hybrid_search import extract_act_hint
            act_hint = extract_act_hint(query)
            kg_chunks = enrich_retrieval(
                ner_sections        = ner_sections,
                chunk_pool          = [],
                bypass_score_filter = True,
                act_hint            = act_hint,
            )
            if kg_chunks:
                logger.debug(f"[Graph] KG: {len(kg_chunks)} extra chunk(s)")
        except Exception as e:
            logger.exception(f"[Graph] legal_rag_node KG warning")

    # RAG pipeline
    try:
        answer          = rag_pipeline.query(user_query=query, context_block=context_block)
        kg_count        = len(kg_chunks)
        rag_answer_text = answer.answer_text

        # Optional case law enrichment
        if ENABLE_CASE_LAW_ENRICHMENT and ner_sections and answer.sources_consulted > 0:
            from rag.llm import llm as _groq
            try:
                rag_answer_text = enrich_rag_response_with_case_law(
                    rag_answer_text = rag_answer_text,
                    ner_sections    = ner_sections,
                    groq_client     = _groq,
                )
            except Exception as e:
                logger.exception(f"[Graph] Case law enrichment warning")

        scope_status  = "in_scope"
        scope_message = ""
        if answer.sources_consulted == 0:
            scope_status  = "out_of_scope"
            scope_message = "No relevant Indian legal provisions could be found for this query."
            if not getattr(answer, "fallback", False):
                rag_answer_text = ""
            ner_sections = []
            # Clear stale act context — no valid retrieval means no act to persist.
            if session_id:
                session_memory.clear_last_act(session_id)
        else:
            # ── Persist last_act + last_section for follow-up queries ────────
            # Prefer the scratchpad act (LLM/regex extracted from the current
            # query) over the NER act.  Use first section from scratchpad too.
            detected_acts     = scratchpad.get(SCRATCH_DETECTED_ACTS, [])
            detected_sections = scratchpad.get(SCRATCH_DETECTED_SECTIONS, [])
            resolved_act = detected_acts[0] if detected_acts else ""

            # Fallback: pull act name from NER entity labels if scratchpad empty
            if not resolved_act:
                for ent in ner_out.get("entities", []):
                    if ent.get("label") in ("ACT", "LEGISLATION"):
                        resolved_act = ent.get("text", "").strip()
                        if resolved_act:
                            break

            if resolved_act and session_id:
                resolved_section = detected_sections[0] if detected_sections else ""
                session_memory.set_last_act(session_id, resolved_act, resolved_section)

        return {
            "rag_result": {
                "answer":            rag_answer_text,
                "sources_consulted": answer.sources_consulted + kg_count if answer.sources_consulted > 0 else 0,
                "synthesis_note":    (answer.synthesis_note or "")
                                     + (f" [KG+{kg_count}]" if kg_count and answer.sources_consulted > 0 else ""),
                "grounding_warning": answer.grounding_warning or "",
                "rewritten_queries": answer.rewritten_queries or [],
                "reranker_used":     answer.reranker_used,
                "fallback":          getattr(answer, "fallback", False),
                "mode":              "legal_rag_node",
                "kg_sections_used":  ner_sections,
            },
            "ner_result":     ner_out if answer.sources_consulted > 0 else {"entities": []},
            "response":       rag_answer_text,
            "rag_grade":      "good" if answer.sources_consulted > 0 else "poor",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "scope_status":   scope_status,
            "scope_message":  scope_message,
            "scratchpad":     scratchpad,
            "error":          "",
        }
    except Exception as exc:
        logger.exception(f"[Graph] legal_rag_node ERROR")
        return {
            "rag_result": {},
            "ner_result": {"entities": []},
            "response":   "I encountered an error processing your legal query. Please try again.",
            "rag_grade":  "poor",
            "scope_status": "in_scope",
            "scope_message": "",
            "scratchpad":   scratchpad,
            "error":      str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: DOCUMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def document_analysis_node(state: AgentState) -> dict:
    from rag.pipeline     import rag_pipeline
    from nlp.ner_pipeline import run_ner

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")
    enriched      = f"{context_block}\n\n{query}" if context_block else query

    logger.debug("[Graph] document_analysis_node -> RAG + NER")

    try:
        answer  = rag_pipeline.query(enriched)
        rag_out = {
            "answer":            answer.answer_text,
            "sources_consulted": answer.sources_consulted,
            "synthesis_note":    answer.synthesis_note    or "",
            "grounding_warning": answer.grounding_warning or "",
            "rewritten_queries": answer.rewritten_queries or [],
            "reranker_used":     answer.reranker_used,
            "fallback":          getattr(answer, "fallback", False),
            "mode":              "document_analysis_node",
        }
        doc_body = query.split("\n\n", 1)[-1][:3000]
        try:
            ner_out = run_ner(doc_body)
        except Exception as e:
            ner_out = {"entities": [], "error": str(e)}

        return {
            "rag_result":     rag_out,
            "ner_result":     ner_out,
            "response":       answer.answer_text,
            "rag_grade":      "good",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }
    except Exception as exc:
        logger.exception(f"[Graph] document_analysis_node ERROR")
        return {
            "rag_result": {},
            "ner_result": {},
            "response":   "I encountered an error analysing the document. Please try again.",
            "rag_grade":  "poor",
            "error":      str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: RISK CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def risk_check_node(state: AgentState) -> dict:
    from rag.pipeline       import rag_pipeline
    from models.risk_scorer import risk_scorer

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")
    risk_prefix   = (
        "Provide a detailed legal risk assessment for the following situation. "
        "Identify applicable Indian laws, potential penalties, liabilities, "
        "and legal consequences:\n\n"
    )
    enriched = (
        f"{context_block}\n\n{risk_prefix}{query}"
        if context_block else f"{risk_prefix}{query}"
    )
    logger.debug("[Graph] risk_check_node -> RAG + scorer")

    try:
        answer  = rag_pipeline.query(enriched)
        rag_out = {
            "answer":            answer.answer_text,
            "sources_consulted": answer.sources_consulted,
            "synthesis_note":    answer.synthesis_note    or "",
            "grounding_warning": answer.grounding_warning or "",
            "rewritten_queries": answer.rewritten_queries or [],
            "reranker_used":     answer.reranker_used,
            "fallback":          getattr(answer, "fallback", False),
            "mode":              "risk_check_node",
        }
        try:
            risk_out  = risk_scorer.score(text=query, doc_type="unknown")
            risk_dict = {"score": risk_out.score, "level": risk_out.level, "factors": risk_out.factors}
        except Exception as e:
            logger.exception(f"[Graph] risk_scorer warning")
            risk_dict = {"score": 0.0, "level": "unknown", "factors": []}

        return {
            "rag_result":     rag_out,
            "risk_result":    risk_dict,
            "response":       answer.answer_text,
            "rag_grade":      "good",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }
    except Exception as exc:
        logger.exception(f"[Graph] risk_check_node ERROR")
        return {
            "rag_result":  {},
            "risk_result": {"score": 0.0, "level": "unknown", "factors": []},
            "response":    "I encountered an error during risk assessment. Please try again.",
            "rag_grade":   "poor",
            "error":       str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: DRAFT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def draft_node(state: AgentState) -> dict:
    from agents.drafting_agent import drafting_agent

    query      = state.get("query", "")
    session_id = state.get("session_id", "")
    logger.debug(f"[Graph] draft_node -> session={session_id[:8] if session_id else '?'}…")

    try:
        result    = drafting_agent.handle(query=query, session_id=session_id)
        stage_str = result.get("stage", "INIT")
        if hasattr(stage_str, "value"):
            stage_str = stage_str.value

        complete  = result.get("complete", False)
        doc_type  = result.get("doc_type", "")
        draft_txt = result.get("draft", "")

        # ── Persist completed draft text into draft_data for PDF export ────────
        # drafting_agent._save() already wrote draft_data at DONE stage, but the
        # final draft_text is only returned transiently.  Re-save with the text
        # so the /export-pdf endpoint can retrieve it by session_id.
        if complete and draft_txt:
            _persisted = dict(result.get("draft_data", {}))
            _persisted["completed_draft"] = draft_txt
            drafting_agent._save(session_id, "DONE", doc_type, _persisted)

        scope_status = "in_scope"
        scope_message = ""
        if stage_str == "0" or (stage_str == "INIT" and not doc_type) or (stage_str == 0):
            scope_status = "out_of_scope"
            scope_message = "The requested document type is not in our supported template list."

        # ── Scratchpad writer: doc_type ──────────────────────────────────────
        scratchpad = dict(state.get("scratchpad", {}))
        if doc_type:
            scratchpad[SCRATCH_DOC_TYPE] = doc_type

        draft_data = result.get("draft_data", {})
        if "missing_elements" in draft_data:
            scratchpad["missing_elements"] = draft_data["missing_elements"]

        return {
            "draft_stage": stage_str,
            "draft_data":  draft_data,
            "response":    result.get("answer", ""),
            "rag_result": {
                "answer":            result.get("answer", ""),
                "sources_consulted": 0,
                "synthesis_note":    f"DraftingAgent stage={stage_str} category={doc_type} complete={complete}",
                "grounding_warning": "",
                "rewritten_queries": [],
                "reranker_used":     False,
                "mode":              f"draft_node_{stage_str}",
                "complete":          complete,
                "draft":             draft_txt,
                "doc_type":          doc_type,
            },
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "scope_status":   scope_status,
            "scope_message":  scope_message,
            "scratchpad":     scratchpad,
            "error":          "",
            "validation_status": result.get("validation_status", "not_applicable"),
        }
    except Exception as exc:
        logger.exception(f"[Graph] draft_node ERROR")
        return {
            "draft_stage": "ERROR",
            "draft_data":  {},
            "rag_result":  {},
            "response":    "I encountered an error with the drafting workflow. Please try again.",
            "scope_status": "in_scope",
            "scope_message": "",
            "error":       str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: MULTILINGUAL  (two sub-flows)
# ═══════════════════════════════════════════════════════════════════════════════

def multilingual_node(state: AgentState) -> dict:
    """
    Sub-flow A: intent == translation_request -> translation_agent.handle()
    Sub-flow B: source_language != "en" (auto-detected) -> process_multilingual_query()
    """
    from agents.multilingual_agent import process_multilingual_query
    from agents.translation_agent  import translation_agent
    from rag.pipeline              import rag_pipeline
    from rag.llm                   import llm as groq_client

    query           = state.get("query", "")
    session_id      = state.get("session_id", "")
    intent          = state.get("intent", "")
    source_language = state.get("source_language", "en")

    if intent == "translation_request":
        logger.debug("[Graph] multilingual_node -> sub-flow B: explicit translation")
        try:
            result = translation_agent.handle(query=query, session_id=session_id)
            return {
                "source_language": result.get("target_language", source_language),
                "rag_result": {
                    "answer":            result.get("answer", ""),
                    "sources_consulted": result.get("sources_consulted", 0),
                    "synthesis_note":    result.get("synthesis_note", ""),
                    "grounding_warning": result.get("grounding_warning", ""),
                    "rewritten_queries": result.get("rewritten_queries", []),
                    "reranker_used":     result.get("reranker_used", False),
                    "mode":              result.get("mode", "translation_request"),
                },
                "response":       result.get("answer", ""),
                "pipeline_depth": state.get("pipeline_depth", 1) + 1,
                "error":          "",
            }
        except Exception as exc:
            logger.exception(f"[Graph] multilingual_node B ERROR")
            return {"source_language": source_language, "rag_result": {},
                    "response": "Translation error. Please try again.", "error": str(exc)}

    logger.debug(f"[Graph] multilingual_node -> sub-flow A: auto {source_language!r}")
    try:
        result = process_multilingual_query(
            query        = query,
            session_id   = session_id,
            rag_pipeline = rag_pipeline,
            groq_client  = groq_client,
        )
        return {
            "source_language": result.get("detected_language", source_language),
            "rag_result": {
                "answer":            result.get("english_response", ""),
                "sources_consulted": result.get("sources_consulted", 0),
                "synthesis_note":    (result.get("synthesis_note", "") +
                                      f" [translated from {result.get('original_language', '?')}]"),
                "grounding_warning": result.get("grounding_warning", ""),
                "rewritten_queries": result.get("rewritten_queries", []),
                "reranker_used":     result.get("reranker_used", False),
                "mode":              result.get("mode", f"multilingual_{source_language}"),
                "translated_query":  result.get("translated_query", ""),
                "original_language": result.get("original_language", ""),
            },
            "response":       result.get("response", ""),
            "rag_grade":      "good",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }
    except Exception as exc:
        logger.exception(f"[Graph] multilingual_node A ERROR")
        return {"source_language": source_language, "rag_result": {},
                "response": "I encountered an error. Please try again.", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: CASE LAW SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def case_law_node(state: AgentState) -> dict:
    """Intent: case_law_search -> Indian Kanoon live judgment search."""
    from agents.case_law_agent import search_and_summarize, format_case_law_response
    from rag.llm               import llm as groq_client

    query = state.get("query", "")
    logger.debug(f"[Graph] case_law_node -> Indian Kanoon: {query[:60]!r}")

    # ── Scratchpad reader: enrich search query ───────────────────────────
    scratchpad = state.get("scratchpad", {})
    extra_terms = []
    for sec in scratchpad.get(SCRATCH_DETECTED_SECTIONS, []):
        extra_terms.append(f"Section {sec}")
    for act in scratchpad.get(SCRATCH_DETECTED_ACTS, []):
        extra_terms.append(act)
    enriched_query = query
    if extra_terms:
        enriched_query = f"{query} {' '.join(extra_terms)}"
        logger.debug(f"[Graph] case_law_node -> enriched query: {enriched_query[:80]!r}")

    try:
        import asyncio
        search_result = asyncio.run(search_and_summarize(query=enriched_query, groq_client=groq_client, max_results=3))
        
        scope_status = "in_scope"
        scope_message = ""
        if search_result["total_found"] == 0:
            scope_status = "out_of_scope"
            scope_message = "No matching judgments found on Indian Kanoon for this query."
            formatted_response = ""
        else:
            formatted_response = format_case_law_response(search_result)

        return {
            "case_law_result": search_result,
            "rag_result": {
                "answer":            formatted_response,
                "sources_consulted": search_result["total_found"],
                "synthesis_note":    f"Indian Kanoon: {search_result['total_found']} judgment(s)",
                "grounding_warning": "" if search_result["total_found"] > 0
                                     else "No judgments found — refine query.",
                "rewritten_queries": [],
                "reranker_used":     False,
                "mode":              "case_law_node",
            },
            "response":       formatted_response,
            "rag_grade":      "good" if search_result["total_found"] > 0 else "poor",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "scope_status":   scope_status,
            "scope_message":  scope_message,
            "error":          "",
        }
    except Exception as exc:
        logger.exception(f"[Graph] case_law_node ERROR")
        return {
            "case_law_result": {"query": query, "results": [], "total_found": 0},
            "rag_result":      {},
            "response":        "Error searching case law. Check INDIANKANOON_API_KEY in .env.",
            "rag_grade":       "poor",
            "scope_status":    "in_scope",
            "scope_message":   "",
            "error":           str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: RIGHTS AGENT  (new — Week 3 Day 2 Session 2)
# ═══════════════════════════════════════════════════════════════════════════════

def rights_node(state: AgentState) -> dict:
    """
    Intent: rights_check

    Detects the rights category from the query, loads the structured guide
    from data/rights_guide.json, and enriches with live RAG context.

    Category detection strategy:
      1. Keyword scan of query for: tenant, employee/worker/labour,
         consumer, women/woman/wife/domestic, bail/arrest/accused/detained
      2. Falls back to presenting all categories if no match found

    Output: formatted markdown rights guide with legal sections + what-to-do.
    """
    from agents.rights_agent import (
        get_rights_with_rag_enrichment,
        format_rights_response,
        get_all_categories,
    )
    from rag.pipeline import rag_pipeline

    query = state.get("query", "").lower()
    logger.debug(f"[Graph] rights_node -> detecting category from: {query[:60]!r}")

    # ── Category detection ─────────────────────────────────────────────────────
    category = _detect_rights_category(query)

    if category:
        logger.debug(f"[Graph] rights_node -> category={category!r}")
        try:
            rights_dict = get_rights_with_rag_enrichment(
                category     = category,
                rag_pipeline = rag_pipeline,
            )
            formatted   = format_rights_response(rights_dict)

            return {
                "rights_category": category,
                "rights_result":   rights_dict,
                "rag_result": {
                    "answer":            formatted,
                    "sources_consulted": rights_dict.get("rag_enrichment", {}).get("sources_consulted", 0),
                    "synthesis_note":    f"RightsAgent: category={category}, RAG enriched",
                    "grounding_warning": "",
                    "rewritten_queries": [],
                    "reranker_used":     False,
                    "mode":              f"rights_node_{category}",
                },
                "response":       formatted,
                "rag_grade":      "good",
                "pipeline_depth": state.get("pipeline_depth", 1) + 1,
                "scope_status":   "in_scope",
                "scope_message":  "",
                "error":          "",
            }
        except Exception as exc:
            logger.exception(f"[Graph] rights_node ERROR for category={category!r}")
            return {
                "rights_category": category,
                "rights_result":   {},
                "rag_result":      {},
                "response": (
                    f"I encountered an error loading rights for '{category}'. "
                    "Please try again or ask a specific legal question."
                ),
                "rag_grade": "poor",
                "scope_status": "in_scope",
                "scope_message": "",
                "error":     str(exc),
            }

    # ── No category detected — out of scope ──────────────────────────────────
    logger.debug("[Graph] rights_node -> no category detected, out of scope")
    try:
        categories    = get_all_categories()
        menu_lines    = [
            "We currently support legal rights guides for the following categories:"
        ]
        for cat in categories:
            menu_lines.append(f"  • {cat['display']}")
        
        scope_msg = "Requested rights category is not supported. " + " ".join(menu_lines)
        
        scratchpad = dict(state.get("scratchpad", {}))
        scratchpad["rights_fallback_used"] = True

        return {
            "rights_category": "",
            "rights_result":   {},
            "rag_result":      {},
            "response":        "",
            "rag_grade":       "poor",
            "pipeline_depth":  state.get("pipeline_depth", 1) + 1,
            "scope_status":    "out_of_scope",
            "scope_message":   scope_msg,
            "scratchpad":      scratchpad,
            "error":           "",
        }
    except Exception as exc:
        return {
            "rights_category": "",
            "rights_result":   {},
            "response":        "Error generating rights guide.",
            "rag_grade":       "poor",
            "scope_status":    "in_scope",
            "scope_message":   "",
            "error":           str(exc),
        }


def _detect_rights_category(query_lower: str) -> str:
    """
    Detect rights category from query text.

    Returns category key or empty string if ambiguous.

    Strategy:
      1. Fast regex keyword scan (zero-cost, handles explicit keywords).
      2. If regex returns empty, fallback to a cheap LLM classification call
         that maps natural-language queries to one of the 5 valid categories
         or "none".  Follows the same Groq JSON-mode pattern used by
         IntentClassifier._call_groq_json().
    """
    import re

    _CATEGORY_PATTERNS: list[tuple[re.Pattern, str]] = [
        # women — check before bail (domestic violence / protection order overlap)
        (re.compile(
            r'\b(women?|wife|wife\'s|domestic\s+violence|dowry|498[aA]|'
            r'maternity|dv\s+act|pwdva|sexual\s+harassment|posh)\b',
            re.IGNORECASE,
        ), "women"),
        # bail/arrested person — check before general legal
        (re.compile(
            r'\b(bail|arrested|arrest|detained|detention|accused|custody|'
            r'lockup|police\s+station|remand|chargesheet|default\s+bail)\b',
            re.IGNORECASE,
        ), "bail"),
        # tenant
        (re.compile(
            r'\b(tenant|tenants?|renter|landlord|rent|evict|eviction|'
            r'security\s+deposit|lease|rental)\b',
            re.IGNORECASE,
        ), "tenant"),
        # employee/worker
        (re.compile(
            r'\b(employee|employees?|worker|workers?|labour|employer|'
            r'salary|wages?|gratuity|pf|provident\s+fund|epf|'
            r'termination|fired|dismissed|workplace)\b',
            re.IGNORECASE,
        ), "employee"),
        # consumer
        (re.compile(
            r'\b(consumer|consumers?|buyer|customer|product|refund|'
            r'defective|e.commerce|amazon|flipkart|complaint\s+forum|'
            r'consumer\s+commission|ncdrc|scdrc|dcdrc)\b',
            re.IGNORECASE,
        ), "consumer"),
    ]

    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(query_lower):
            return category

    # ── LLM fallback: classify paraphrased queries into a rights category ─────
    return _llm_classify_rights_category(query_lower)


def _llm_classify_rights_category(query: str) -> str:
    """
    LLM-based fallback for rights category detection.

    Uses a single cheap Groq JSON-mode call (same pattern as
    IntentClassifier._call_groq_json) to classify the query into one of
    the 5 valid categories or "none".

    Returns category key ("tenant"/"employee"/"consumer"/"women"/"bail")
    or empty string if the query doesn't fit any category.
    """
    import os
    import json

    _VALID = {"tenant", "employee", "consumer", "women", "bail"}

    _SYSTEM = (
        "You are a legal rights category classifier for an Indian legal AI platform.\n\n"
        "Given a user query, classify it into exactly ONE of these 5 rights categories:\n"
        "- tenant: Issues about rent, eviction, landlord disputes, security deposits, lease agreements\n"
        "- employee: Issues about salary, wages, termination, workplace rights, provident fund, gratuity\n"
        "- consumer: Issues about defective products, poor services, refunds, subscriptions, "
        "e-commerce disputes, unfair trade practices\n"
        "- women: Issues about domestic violence, dowry, sexual harassment, maternity rights, "
        "protection orders\n"
        "- bail: Issues about arrest, detention, bail rights, custody, police station rights, "
        "rights of accused\n\n"
        "If the query does NOT fit any of these 5 categories, return \"none\".\n\n"
        "Respond ONLY with this JSON (no markdown, no extra text):\n"
        "{\"category\": \"<one of: tenant, employee, consumer, women, bail, none>\", "
        "\"reasoning\": \"one sentence explanation\"}"
    )

    try:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("[Graph] _llm_classify_rights_category: GROQ_API_KEY not set")
            return ""

        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model           = "llama-3.3-70b-versatile",
            messages        = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": query},
            ],
            temperature     = 0,
            max_tokens      = 100,
            response_format = {"type": "json_object"},
            timeout         = 6,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(raw)
        category = data.get("category", "none").strip().lower()
        reasoning = data.get("reasoning", "")

        if category in _VALID:
            logger.info(
                f"[Graph] LLM rights category fallback -> {category!r} "
                f"reasoning={reasoning!r}"
            )
            return category

        logger.debug(
            f"[Graph] LLM rights category fallback -> none "
            f"(raw={category!r}, reasoning={reasoning!r})"
        )
        return ""

    except Exception as exc:
        logger.exception("[Graph] _llm_classify_rights_category failed (non-fatal)")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: GENERAL
# ═══════════════════════════════════════════════════════════════════════════════

def general_node(state: AgentState) -> dict:
    # ── Early return if response was pre-set (draft cancel / re-prompt) ───────
    existing_response = state.get("response", "")
    if existing_response:
        logger.debug("[Graph] general_node -> returning pre-set response")
        return {
            "response": existing_response,
            "rag_result": {
                "answer": existing_response, "sources_consulted": 0,
                "synthesis_note": "pre-set response", "grounding_warning": "",
                "rewritten_queries": [], "reranker_used": False, "mode": "general_node",
            },
            "source": "llm_only",
            "sourced": False,
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error": "",
        }

    from rag.llm import llm

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")

    system_prompt = (
        "You are LexShield AI, an Indian legal intelligence assistant. "
        "Help users understand Indian law, their legal rights, and legal documents. "
        "Be concise, friendly, and direct. "
        "If asked a specific legal question, encourage the user to ask it directly. "
        "You support queries in Malayalam, Hindi, Tamil, Telugu, and other Indian languages."
    )
    prompt = f"{context_block}\n\nUser: {query}" if context_block else f"User: {query}"
    logger.debug("[Graph] general_node -> direct LLM")

    try:
        answer = llm.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=512)
        return {
            "response": answer,
            "rag_result": {
                "answer": answer, "sources_consulted": 0,
                "synthesis_note": "", "grounding_warning": "",
                "rewritten_queries": [], "reranker_used": False, "mode": "general_node",
            },
            "source": "llm_only",
            "sourced": False,
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error": "",
        }
    except Exception as exc:
        logger.exception(f"[Graph] general_node ERROR")
        return {
            "response":   "Hello! I'm LexShield AI, your Indian legal assistant. How can I help you today?",
            "rag_result": {},
            "source":     "llm_only",
            "sourced":    False,
            "error":      str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Assemble and compile the 8-node LangGraph StateGraph.

    Nodes:
      classify_intent_node   — entry, always runs
      legal_rag_node         — English legal Q&A
      document_analysis_node — uploaded document analysis
      draft_node             — 8-category complaint drafting
      risk_check_node        — legal risk assessment
      multilingual_node      — Malayalam/Hindi auto + explicit translation
      case_law_node          — Indian Kanoon live judgment search
      rights_node            — Know Your Rights structured guide
      general_node           — greetings and capability questions
    """
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent_node",   classify_intent_node)
    builder.add_node("legal_rag_node",         legal_rag_node)
    builder.add_node("document_analysis_node", document_analysis_node)
    builder.add_node("draft_node",             draft_node)
    builder.add_node("risk_check_node",        risk_check_node)
    builder.add_node("multilingual_node",      multilingual_node)
    builder.add_node("case_law_node",          case_law_node)
    builder.add_node("rights_node",            rights_node)
    builder.add_node("general_node",           general_node)

    builder.set_entry_point("classify_intent_node")

    builder.add_conditional_edges(
        "classify_intent_node",
        route_by_intent,
        {
            "legal_rag_node":         "legal_rag_node",
            "document_analysis_node": "document_analysis_node",
            "draft_node":             "draft_node",
            "risk_check_node":        "risk_check_node",
            "multilingual_node":      "multilingual_node",
            "case_law_node":          "case_law_node",
            "rights_node":            "rights_node",
            "general_node":           "general_node",
        },
    )

    for node_name in [
        "legal_rag_node", "document_analysis_node", "draft_node",
        "risk_check_node", "multilingual_node", "case_law_node",
        "general_node",
    ]:
        builder.add_edge(node_name, END)

    builder.add_conditional_edges(
        "rights_node",
        route_after_rights,
        {
            "legal_rag_node": "legal_rag_node",
            "end":             END,
        },
    )

    return builder.compile(checkpointer=checkpointer)


agent_graph = build_graph()