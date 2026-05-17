"""
LexShield AI — LangGraph Multi-Agent Graph  (Week 3, Day 2 — Updated)
=======================================================================
Changes from Session 3:

1. classify_intent_node — now calls detect_language() first.
   Stores ISO 639-1 code in state["source_language"].
   Non-English queries (ml/hi/ta/te/...) detected here, before routing.

2. route_by_intent — Priority 3 added: if source_language != "en"
   AND intent is legal_query, risk_check, or general → multilingual_node.
   This ensures a Malayalam query asking about Section 302 is handled by
   the multilingual pipeline, not raw RAG (which would get a Malayalam
   query it can't process meaningfully).

3. multilingual_node — split into two sub-flows:
   - intent == "translation_request" → translation_agent.handle()
     (explicit: "explain in Malayalam", "translate to Hindi")
   - source_language != "en" (auto-detected) → multilingual_agent.process_multilingual_query()
     (auto: Malayalam/Hindi query detected by script recognition)

4. case_law_node — new node (7th).
   Calls case_law_agent.search_and_summarize() → Indian Kanoon live judgments.
   Routes on intent == "case_law_search".

5. legal_rag_node — optional case law enrichment.
   When ENABLE_CASE_LAW_ENRICHMENT=true and NER finds section numbers,
   appends top 2 real judgments from Indian Kanoon after RAG answer.

6. AgentState — source_language field already present from Session 3.
   No schema change needed.

Graph topology:
---------------------------------------------------------------------------
[START]
   │
   ▼
classify_intent_node  ← now also calls detect_language()
   │
   ▼  route_by_intent() — conditional edge
┌─────────────────────────────────────────────────────────────────────┐
│ legal_rag_node       │ document_analysis_node │ case_law_node       │
│ risk_check_node      │ draft_node             │ multilingual_node   │
│ general_node         │                        │                     │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼
[END]
---------------------------------------------------------------------------

Priority order in route_by_intent:
  1. Active draft session in SQLite     → draft_node
  2. Non-English source_language + legal/risk/general intent → multilingual_node
  3. Intent map                         → correct agent node
"""

import os
import sqlite3
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ═══════════════════════════════════════════════════════════════════════════════
# SQLITE CHECKPOINTER  (module-level singleton)
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH         = os.path.join(_PROJECT_ROOT, "data", "sessions.db")

_checkpoint_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
checkpointer     = SqliteSaver(_checkpoint_conn)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query:           str
    session_id:      str

    # ── Classification ────────────────────────────────────────────────────────
    intent:          str
    confidence:      float

    # ── RAG / NER / Risk outputs ──────────────────────────────────────────────
    rag_result:      dict
    ner_result:      dict
    risk_result:     dict

    # ── Drafting (multi-turn, stage managed by DraftingAgent in SQLite) ───────
    draft_stage:     str    # DraftStage enum value as string
    draft_data:      dict

    # ── Case law ──────────────────────────────────────────────────────────────
    case_law_result: dict   # {query, results, total_found} from search_and_summarize

    # ── Diagnostics ───────────────────────────────────────────────────────────
    pipeline_depth:  int
    rag_grade:       str    # "good" | "poor" | ""

    # ── Multilingual ──────────────────────────────────────────────────────────
    source_language: str    # ISO 639-1: "en", "ml", "hi", "ta", "te", ...

    # ── Output ────────────────────────────────────────────────────────────────
    response:        str
    error:           str


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: INTENT CLASSIFIER  (+ language detection)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_intent_node(state: AgentState) -> dict:
    """
    Entry node. Runs two classifiers in sequence:

    1. detect_language(query) — Unicode fast-path + langdetect fallback.
       Sets state["source_language"] to ISO 639-1 code.
       This happens BEFORE intent classification so route_by_intent
       can redirect non-English queries to multilingual_node.

    2. intent_classifier.classify(query) — keyword + regex scorer.
       Sets state["intent"] and state["confidence"].

    Non-English queries are classified by intent normally so we can
    still distinguish "Malayalam legal query" from "Malayalam draft request".
    The multilingual_node only handles legal_query / risk_check / general —
    draft_request goes to draft_node even for non-English input (DraftingAgent
    generates English drafts which are then translated in a future iteration).
    """
    from agents.intent_classifier  import intent_classifier
    from agents.multilingual_agent import detect_language

    query = state.get("query", "").strip()
    if not query:
        return {
            "intent":          "general",
            "confidence":      0.0,
            "source_language": "en",
            "pipeline_depth":  1,
        }

    # ── Step 1: Language detection ─────────────────────────────────────────────
    source_language = detect_language(query)
    if source_language != "en":
        print(f"[Graph] classify_intent_node → detected language: {source_language!r}")

    # ── Step 2: Intent classification ─────────────────────────────────────────
    result = intent_classifier.classify(query)
    print(
        f"[Graph] classify_intent_node → intent={result.intent!r} "
        f"conf={result.confidence:.2f} lang={source_language!r}"
    )

    return {
        "intent":          result.intent,
        "confidence":      result.confidence,
        "source_language": source_language,
        "pipeline_depth":  state.get("pipeline_depth", 0) + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION  (conditional edge)
# ═══════════════════════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge after classify_intent_node.

    Priority order:
      1. Active SQLite draft session → draft_node (always, regardless of language)
      2. Non-English source_language AND intent is legal/risk/general
         → multilingual_node (wraps translate→RAG→translate flow)
      3. Intent map → correct agent node

    Notes on Priority 2:
    - draft_request is NOT redirected to multilingual_node — the DraftingAgent
      generates structured English drafts regardless of query language.
      Future enhancement: translate the final draft to user's language.
    - document_analysis is NOT redirected — OCR already handles multilingual
      via source_language-aware Tesseract config in cv/pipeline.py.
    - case_law_search is NOT redirected — Indian Kanoon is English-only.
    """
    from agents.drafting_agent import drafting_agent

    session_id      = state.get("session_id", "")
    source_language = state.get("source_language", "en")
    intent          = state.get("intent", "general")

    # ── Priority 1: Active multi-turn draft ────────────────────────────────────
    if session_id and drafting_agent.has_active_draft(session_id):
        print(
            f"[Graph] route_by_intent → active draft "
            f"(session={session_id[:8]}…) → draft_node"
        )
        return "draft_node"

    # ── Priority 2: Non-English auto-detection ─────────────────────────────────
    _multilingual_eligible_intents = {"legal_query", "risk_check", "general"}
    if source_language != "en" and intent in _multilingual_eligible_intents:
        print(
            f"[Graph] route_by_intent → non-English source_language={source_language!r} "
            f"intent={intent!r} → multilingual_node"
        )
        return "multilingual_node"

    # ── Priority 3: Intent map ─────────────────────────────────────────────────
    _intent_to_node: dict[str, str] = {
        "legal_query":          "legal_rag_node",
        "document_analysis":    "document_analysis_node",
        "draft_request":        "draft_node",
        "risk_check":           "risk_check_node",
        "translation_request":  "multilingual_node",
        "case_law_search":      "case_law_node",
        "general":              "general_node",
    }
    node = _intent_to_node.get(intent, "general_node")
    print(f"[Graph] route_by_intent → intent={intent!r} → {node}")
    return node


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: LEGAL RAG AGENT  (+ optional case law enrichment)
# ═══════════════════════════════════════════════════════════════════════════════

def legal_rag_node(state: AgentState) -> dict:
    """
    Intent: legal_query (English queries only — non-English routes to multilingual_node)

    Flow:
      1. NER → extract section IDs from query
      2. KG enrich → add graph-connected chunks (IPC↔BNS equivalents)
      3. RAG pipeline → hybrid search + CRAG + rerank + synthesise
      4. [Optional] Case law enrichment → append real judgments from Indian Kanoon
      5. Write rag_result, ner_result, response to state
    """
    from rag.pipeline            import rag_pipeline
    from nlp.ner_pipeline        import run_ner
    from rag.knowledge_graph     import enrich_retrieval
    from agents.case_law_agent   import (
        enrich_rag_response_with_case_law,
        ENABLE_CASE_LAW_ENRICHMENT,
    )

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")
    enriched      = f"{context_block}\n\n{query}" if context_block else query

    print("[Graph] legal_rag_node → NER + KG enrich + RAG")

    # ── Step 1: NER ────────────────────────────────────────────────────────────
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
    except Exception as ner_exc:
        print(f"[Graph] legal_rag_node NER warning (non-fatal): {ner_exc}")

    # ── Step 2: Knowledge Graph enrichment ────────────────────────────────────
    kg_chunks: list[dict] = []
    if ner_sections:
        try:
            kg_chunks = enrich_retrieval(
                ner_sections        = ner_sections,
                chunk_pool          = [],
                bypass_score_filter = True,
            )
            if kg_chunks:
                print(
                    f"[Graph] legal_rag_node KG: {len(kg_chunks)} chunk(s) "
                    f"for {ner_sections}"
                )
        except Exception as kg_exc:
            print(f"[Graph] legal_rag_node KG warning (non-fatal): {kg_exc}")

    # ── Step 3: RAG pipeline ───────────────────────────────────────────────────
    try:
        answer   = rag_pipeline.query(enriched)
        kg_count = len(kg_chunks)

        rag_answer_text = answer.answer_text

        # ── Step 4: Case law enrichment (optional) ─────────────────────────────
        if ENABLE_CASE_LAW_ENRICHMENT and ner_sections:
            from rag.llm import llm as _groq
            try:
                rag_answer_text = enrich_rag_response_with_case_law(
                    rag_answer_text = rag_answer_text,
                    ner_sections    = ner_sections,
                    groq_client     = _groq,
                )
            except Exception as cl_exc:
                # Non-fatal: RAG answer is still valid without case law
                print(f"[Graph] legal_rag_node case law enrichment warning: {cl_exc}")

        return {
            "rag_result": {
                "answer":            rag_answer_text,
                "sources_consulted": answer.sources_consulted + kg_count,
                "synthesis_note":    (answer.synthesis_note or "")
                                     + (f" [KG+{kg_count}]" if kg_count else ""),
                "grounding_warning": answer.grounding_warning or "",
                "rewritten_queries": answer.rewritten_queries or [],
                "reranker_used":     answer.reranker_used,
                "mode":              "legal_rag_node",
                "kg_sections_used":  ner_sections,
            },
            "ner_result":     ner_out,
            "response":       rag_answer_text,
            "rag_grade":      "good",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }

    except Exception as exc:
        print(f"[Graph] legal_rag_node ERROR: {exc}")
        return {
            "rag_result": {},
            "ner_result": ner_out,
            "response":   "I encountered an error processing your legal query. Please try again.",
            "rag_grade":  "poor",
            "error":      str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: DOCUMENT ANALYSIS AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def document_analysis_node(state: AgentState) -> dict:
    """
    Intent: document_analysis
    RAG pipeline for document text + NER for entity extraction.
    """
    from rag.pipeline     import rag_pipeline
    from nlp.ner_pipeline import run_ner

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")
    enriched      = f"{context_block}\n\n{query}" if context_block else query

    print("[Graph] document_analysis_node → RAG + NER")

    try:
        answer  = rag_pipeline.query(enriched)
        rag_out = {
            "answer":            answer.answer_text,
            "sources_consulted": answer.sources_consulted,
            "synthesis_note":    answer.synthesis_note    or "",
            "grounding_warning": answer.grounding_warning or "",
            "rewritten_queries": answer.rewritten_queries or [],
            "reranker_used":     answer.reranker_used,
            "mode":              "document_analysis_node",
        }

        doc_body = query.split("\n\n", 1)[-1][:3000]
        try:
            ner_out = run_ner(doc_body)
        except Exception as ner_exc:
            print(f"[Graph] NER warning (non-fatal): {ner_exc}")
            ner_out = {"entities": [], "error": str(ner_exc)}

        return {
            "rag_result":     rag_out,
            "ner_result":     ner_out,
            "response":       answer.answer_text,
            "rag_grade":      "good",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }

    except Exception as exc:
        print(f"[Graph] document_analysis_node ERROR: {exc}")
        return {
            "rag_result": {},
            "ner_result": {},
            "response":   "I encountered an error analysing the document. Please try again.",
            "rag_grade":  "poor",
            "error":      str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: RISK CHECK AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def risk_check_node(state: AgentState) -> dict:
    """
    Intent: risk_check
    Risk-focused RAG + risk_scorer post-processing.
    """
    from rag.pipeline       import rag_pipeline
    from models.risk_scorer import risk_scorer

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")

    risk_prefix = (
        "Provide a detailed legal risk assessment for the following situation. "
        "Identify applicable Indian laws, potential penalties, liabilities, "
        "and legal consequences:\n\n"
    )
    enriched = (
        f"{context_block}\n\n{risk_prefix}{query}"
        if context_block else f"{risk_prefix}{query}"
    )
    print("[Graph] risk_check_node → risk assessment via RAG + scorer")

    try:
        answer  = rag_pipeline.query(enriched)
        rag_out = {
            "answer":            answer.answer_text,
            "sources_consulted": answer.sources_consulted,
            "synthesis_note":    answer.synthesis_note    or "",
            "grounding_warning": answer.grounding_warning or "",
            "rewritten_queries": answer.rewritten_queries or [],
            "reranker_used":     answer.reranker_used,
            "mode":              "risk_check_node",
        }

        try:
            risk_out  = risk_scorer.score(text=query, doc_type="unknown")
            risk_dict = {
                "score":   risk_out.score,
                "level":   risk_out.level,
                "factors": risk_out.factors,
            }
        except Exception as risk_exc:
            print(f"[Graph] risk_scorer warning (non-fatal): {risk_exc}")
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
        print(f"[Graph] risk_check_node ERROR: {exc}")
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
    """
    Intent: draft_request  OR  active multi-turn draft (priority override).
    Dispatches to DraftingAgent.handle() — stateless from LangGraph's perspective.
    """
    from agents.drafting_agent import drafting_agent

    query      = state.get("query", "")
    session_id = state.get("session_id", "")

    print(
        f"[Graph] draft_node → session={session_id[:8] if session_id else '?'}… "
        f"stage={state.get('draft_stage', 'INIT')!r} query={query[:50]!r}"
    )

    try:
        result    = drafting_agent.handle(query=query, session_id=session_id)
        stage_str = result.get("stage", "INIT")
        if hasattr(stage_str, "value"):
            stage_str = stage_str.value

        complete  = result.get("complete", False)
        doc_type  = result.get("doc_type", "")
        draft_txt = result.get("draft", "")

        return {
            "draft_stage": stage_str,
            "draft_data":  result.get("draft_data", {}),
            "response":    result.get("answer", ""),
            "rag_result": {
                "answer":            result.get("answer", ""),
                "sources_consulted": 0,
                "synthesis_note": (
                    f"DraftingAgent stage={stage_str} "
                    f"category={doc_type} complete={complete}"
                ),
                "grounding_warning": "",
                "rewritten_queries": [],
                "reranker_used":     False,
                "mode":              f"draft_node_{stage_str}",
                "complete":          complete,
                "draft":             draft_txt,
                "doc_type":          doc_type,
            },
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[Graph] draft_node ERROR: {exc}")
        return {
            "draft_stage": "ERROR",
            "draft_data":  {},
            "rag_result":  {},
            "response": (
                "I encountered an error with the drafting workflow. "
                "Please try again by describing your legal situation."
            ),
            "error": str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: MULTILINGUAL AGENT  (updated — two sub-flows)
# ═══════════════════════════════════════════════════════════════════════════════

def multilingual_node(state: AgentState) -> dict:
    """
    Handles two distinct multilingual scenarios:

    Sub-flow A — Auto-detected non-English query (e.g. Malayalam, Hindi):
      source_language != "en" in state.
      Uses multilingual_agent.process_multilingual_query():
        translate query → English → RAG → translate answer → source language.
      Example: "Section 302 IPC ൽ ശിക്ഷ എന്ത്?" → Malayalam answer

    Sub-flow B — Explicit translation request (English query):
      intent == "translation_request", source_language == "en".
      Uses translation_agent.handle():
        strip translation instruction → RAG → translate answer to target language.
      Example: "Explain Section 138 NI Act in Hindi" → bilingual answer

    The sub-flow is selected by checking state["intent"]:
      "translation_request" → sub-flow B
      anything else         → sub-flow A (auto-detected non-English)
    """
    from agents.multilingual_agent import process_multilingual_query
    from agents.translation_agent  import translation_agent
    from rag.pipeline              import rag_pipeline
    from rag.llm                   import llm as groq_client

    query           = state.get("query", "")
    session_id      = state.get("session_id", "")
    intent          = state.get("intent", "")
    source_language = state.get("source_language", "en")

    # ── Sub-flow B: Explicit translation request (English query) ───────────────
    if intent == "translation_request":
        print("[Graph] multilingual_node → sub-flow B: explicit translation request")
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
            print(f"[Graph] multilingual_node (sub-flow B) ERROR: {exc}")
            return {
                "source_language": source_language,
                "rag_result":      {},
                "response": "I encountered an error processing the translation request. Please try again.",
                "error":    str(exc),
            }

    # ── Sub-flow A: Auto-detected non-English query ────────────────────────────
    print(
        f"[Graph] multilingual_node → sub-flow A: "
        f"auto-detected {source_language!r} query"
    )
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
                "synthesis_note": (
                    result.get("synthesis_note", "") +
                    f" [translated from {result.get('original_language', 'unknown')}]"
                ),
                "grounding_warning": result.get("grounding_warning", ""),
                "rewritten_queries": result.get("rewritten_queries", []),
                "reranker_used":     result.get("reranker_used", False),
                "mode":              result.get("mode", f"multilingual_{source_language}"),
                "translated_query":  result.get("translated_query", ""),
                "original_language": result.get("original_language", ""),
            },
            "response":       result.get("response", ""),  # in user's language
            "rag_grade":      "good",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }
    except Exception as exc:
        print(f"[Graph] multilingual_node (sub-flow A) ERROR: {exc}")
        return {
            "source_language": source_language,
            "rag_result":      {},
            "response": "I encountered an error processing your query. Please try again.",
            "error":    str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: CASE LAW SEARCH AGENT  (new — Week 3 Day 2)
# ═══════════════════════════════════════════════════════════════════════════════

def case_law_node(state: AgentState) -> dict:
    """
    Intent: case_law_search

    Queries Indian Kanoon for live Supreme Court and High Court judgments.
    Returns structured case law results with:
      - Case title, court, date, citation
      - AI-generated 2-sentence precedent summary (Groq)
      - Clickable URL to full judgment on indiankanoon.org

    This node is what differentiates LexShield from static legal chatbots —
    it retrieves REAL, CURRENT judgments from India's primary case law database.

    Flow:
      1. search_cases(query)          → Indian Kanoon API → case list
      2. summarize_case(case, llm)    → Groq 2-sentence summary per case
      3. format_case_law_response()   → markdown formatted output
      4. Write to state["response"] + state["case_law_result"]
    """
    from agents.case_law_agent import search_and_summarize, format_case_law_response
    from rag.llm               import llm as groq_client

    query = state.get("query", "")
    print(f"[Graph] case_law_node → Indian Kanoon search for: {query[:60]!r}")

    try:
        search_result = search_and_summarize(
            query       = query,
            groq_client = groq_client,
            max_results = 3,
        )
        formatted_response = format_case_law_response(search_result)

        return {
            "case_law_result": search_result,
            "rag_result": {
                "answer":            formatted_response,
                "sources_consulted": search_result["total_found"],
                "synthesis_note": (
                    f"Indian Kanoon: {search_result['total_found']} "
                    f"judgment(s) found for '{query[:40]}'"
                ),
                "grounding_warning": (
                    ""
                    if search_result["total_found"] > 0
                    else "No judgments found — query may need refinement."
                ),
                "rewritten_queries": [],
                "reranker_used":     False,
                "mode":              "case_law_node",
            },
            "response":       formatted_response,
            "rag_grade":      "good" if search_result["total_found"] > 0 else "poor",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[Graph] case_law_node ERROR: {exc}")
        return {
            "case_law_result": {"query": query, "results": [], "total_found": 0},
            "rag_result":      {},
            "response": (
                "I encountered an error searching for case law. "
                "Please check that INDIANKANOON_API_KEY is set in .env and try again."
            ),
            "rag_grade": "poor",
            "error":     str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: GENERAL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def general_node(state: AgentState) -> dict:
    """
    Intent: general  (greetings, capability questions, off-topic)
    Direct Groq call — no RAG, no retrieval.
    """
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
    prompt = (
        f"{context_block}\n\nUser: {query}"
        if context_block else f"User: {query}"
    )
    print("[Graph] general_node → direct LLM (no RAG)")

    try:
        answer = llm.generate(
            prompt        = prompt,
            system_prompt = system_prompt,
            max_tokens    = 512,
        )
        return {
            "response": answer,
            "rag_result": {
                "answer":            answer,
                "sources_consulted": 0,
                "synthesis_note":    "",
                "grounding_warning": "",
                "rewritten_queries": [],
                "reranker_used":     False,
                "mode":              "general_node",
            },
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }
    except Exception as exc:
        print(f"[Graph] general_node ERROR: {exc}")
        return {
            "response": (
                "Hello! I'm LexShield AI, your Indian legal assistant. "
                "How can I help you today?"
            ),
            "rag_result": {},
            "error":      str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Assembles and compiles the 7-node LangGraph StateGraph.

    Node registry:
      classify_intent_node   — entry point, always runs
      legal_rag_node         — English legal queries
      document_analysis_node — uploaded document analysis
      draft_node             — 8-category complaint drafting
      risk_check_node        — legal risk assessment
      multilingual_node      — Malayalam/Hindi/Tamil/Telugu auto + explicit
      case_law_node          — Indian Kanoon live case law search
      general_node           — greetings and capability questions

    SqliteSaver checkpointer: writes full AgentState to data/sessions.db
    after every node. LangGraph reloads previous checkpoint on same thread_id.
    """
    builder = StateGraph(AgentState)

    # ── Register all nodes ─────────────────────────────────────────────────────
    builder.add_node("classify_intent_node",   classify_intent_node)
    builder.add_node("legal_rag_node",         legal_rag_node)
    builder.add_node("document_analysis_node", document_analysis_node)
    builder.add_node("draft_node",             draft_node)
    builder.add_node("risk_check_node",        risk_check_node)
    builder.add_node("multilingual_node",      multilingual_node)
    builder.add_node("case_law_node",          case_law_node)
    builder.add_node("general_node",           general_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    builder.set_entry_point("classify_intent_node")

    # ── Conditional routing from classifier ────────────────────────────────────
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
            "general_node":           "general_node",
        },
    )

    # ── All terminal nodes → END ───────────────────────────────────────────────
    for node_name in [
        "legal_rag_node",
        "document_analysis_node",
        "draft_node",
        "risk_check_node",
        "multilingual_node",
        "case_law_node",
        "general_node",
    ]:
        builder.add_edge(node_name, END)

    return builder.compile(checkpointer=checkpointer)


# ── Singleton compiled graph ───────────────────────────────────────────────────
agent_graph = build_graph()