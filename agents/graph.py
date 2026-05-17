"""
LexShield AI — LangGraph Multi-Agent Graph  (Week 3, Day 2 Session 2 — Updated)
==================================================================================
Changes from previous session:

1. rights_node added as 8th node.
   Routes on intent == "rights_check".
   Calls RightsAgent.get_rights_with_rag_enrichment() → structured rights guide
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
classify_intent_node  ← detect_language() + intent_classifier.classify()
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
  1. Active draft session in SQLite        → draft_node
  2. Non-English source_language + legal/risk/general intent → multilingual_node
  3. Intent map                            → correct agent node
"""

import os
import sqlite3
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ═══════════════════════════════════════════════════════════════════════════════
# SQLITE CHECKPOINTER
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH         = os.path.join(_PROJECT_ROOT, "data", "sessions.db")

_checkpoint_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
checkpointer     = SqliteSaver(_checkpoint_conn)


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
    response:         str
    error:            str


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: INTENT CLASSIFIER  (+ language detection)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_intent_node(state: AgentState) -> dict:
    """
    Entry node. Always runs first.

    1. detect_language(query) — Unicode fast-path + langdetect.
       Stores ISO 639-1 code in state["source_language"].
    2. intent_classifier.classify(query) — 8-intent keyword+regex scorer.
       Stores intent and confidence in state.
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

    source_language = detect_language(query)
    if source_language != "en":
        print(f"[Graph] classify_intent_node → detected language: {source_language!r}")

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
# ROUTING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge after classify_intent_node.

    Priority:
      1. Active SQLite draft → draft_node
      2. Non-English + legal/risk/general intent → multilingual_node
      3. Intent map → correct node
    """
    from agents.drafting_agent import drafting_agent

    session_id      = state.get("session_id", "")
    source_language = state.get("source_language", "en")
    intent          = state.get("intent", "general")

    # Priority 1: active draft
    if session_id and drafting_agent.has_active_draft(session_id):
        print(f"[Graph] route → active draft → draft_node")
        return "draft_node"

    # Priority 2: non-English auto-detection
    # rights_check NOT redirected — English guide content, translated at response layer
    _multilingual_eligible = {"legal_query", "risk_check", "general"}
    if source_language != "en" and intent in _multilingual_eligible:
        print(
            f"[Graph] route → non-English {source_language!r} "
            f"intent={intent!r} → multilingual_node"
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
    print(f"[Graph] route → intent={intent!r} → {node}")
    return node


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: LEGAL RAG  (+ optional case law enrichment)
# ═══════════════════════════════════════════════════════════════════════════════

def legal_rag_node(state: AgentState) -> dict:
    """
    Intent: legal_query (English queries — non-English → multilingual_node)

    Flow: NER → KG enrich → RAG pipeline → optional case law enrichment
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

    print("[Graph] legal_rag_node → NER + KG + RAG")

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
        print(f"[Graph] legal_rag_node NER warning: {e}")

    # KG enrichment
    kg_chunks: list[dict] = []
    if ner_sections:
        try:
            kg_chunks = enrich_retrieval(
                ner_sections        = ner_sections,
                chunk_pool          = [],
                bypass_score_filter = True,
            )
            if kg_chunks:
                print(f"[Graph] KG: {len(kg_chunks)} extra chunk(s)")
        except Exception as e:
            print(f"[Graph] legal_rag_node KG warning: {e}")

    # RAG pipeline
    try:
        answer          = rag_pipeline.query(enriched)
        kg_count        = len(kg_chunks)
        rag_answer_text = answer.answer_text

        # Optional case law enrichment
        if ENABLE_CASE_LAW_ENRICHMENT and ner_sections:
            from rag.llm import llm as _groq
            try:
                rag_answer_text = enrich_rag_response_with_case_law(
                    rag_answer_text = rag_answer_text,
                    ner_sections    = ner_sections,
                    groq_client     = _groq,
                )
            except Exception as e:
                print(f"[Graph] Case law enrichment warning: {e}")

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
# NODE: DOCUMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def document_analysis_node(state: AgentState) -> dict:
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
        print(f"[Graph] document_analysis_node ERROR: {exc}")
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
    print("[Graph] risk_check_node → RAG + scorer")

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
            risk_dict = {"score": risk_out.score, "level": risk_out.level, "factors": risk_out.factors}
        except Exception as e:
            print(f"[Graph] risk_scorer warning: {e}")
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
    from agents.drafting_agent import drafting_agent

    query      = state.get("query", "")
    session_id = state.get("session_id", "")
    print(f"[Graph] draft_node → session={session_id[:8] if session_id else '?'}…")

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
            "error":          "",
        }
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[Graph] draft_node ERROR: {exc}")
        return {
            "draft_stage": "ERROR",
            "draft_data":  {},
            "rag_result":  {},
            "response":    "I encountered an error with the drafting workflow. Please try again.",
            "error":       str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: MULTILINGUAL  (two sub-flows)
# ═══════════════════════════════════════════════════════════════════════════════

def multilingual_node(state: AgentState) -> dict:
    """
    Sub-flow A: intent == translation_request → translation_agent.handle()
    Sub-flow B: source_language != "en" (auto-detected) → process_multilingual_query()
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
        print("[Graph] multilingual_node → sub-flow B: explicit translation")
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
            print(f"[Graph] multilingual_node B ERROR: {exc}")
            return {"source_language": source_language, "rag_result": {},
                    "response": "Translation error. Please try again.", "error": str(exc)}

    print(f"[Graph] multilingual_node → sub-flow A: auto {source_language!r}")
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
        print(f"[Graph] multilingual_node A ERROR: {exc}")
        return {"source_language": source_language, "rag_result": {},
                "response": "I encountered an error. Please try again.", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: CASE LAW SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def case_law_node(state: AgentState) -> dict:
    """Intent: case_law_search → Indian Kanoon live judgment search."""
    from agents.case_law_agent import search_and_summarize, format_case_law_response
    from rag.llm               import llm as groq_client

    query = state.get("query", "")
    print(f"[Graph] case_law_node → Indian Kanoon: {query[:60]!r}")

    try:
        search_result      = search_and_summarize(query=query, groq_client=groq_client, max_results=3)
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
            "error":          "",
        }
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[Graph] case_law_node ERROR: {exc}")
        return {
            "case_law_result": {"query": query, "results": [], "total_found": 0},
            "rag_result":      {},
            "response":        "Error searching case law. Check INDIANKANOON_API_KEY in .env.",
            "rag_grade":       "poor",
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
    print(f"[Graph] rights_node → detecting category from: {query[:60]!r}")

    # ── Category detection ─────────────────────────────────────────────────────
    category = _detect_rights_category(query)

    if category:
        print(f"[Graph] rights_node → category={category!r}")
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
                "error":          "",
            }
        except Exception as exc:
            print(f"[Graph] rights_node ERROR for category={category!r}: {exc}")
            return {
                "rights_category": category,
                "rights_result":   {},
                "rag_result":      {},
                "response": (
                    f"I encountered an error loading rights for '{category}'. "
                    "Please try again or ask a specific legal question."
                ),
                "rag_grade": "poor",
                "error":     str(exc),
            }

    # ── No category detected — show menu ──────────────────────────────────────
    print("[Graph] rights_node → no category detected, showing menu")
    try:
        categories    = get_all_categories()
        menu_lines    = [
            "⚖️ **Know Your Rights — LexShield AI**",
            "",
            "Please specify which rights you'd like to explore:",
            "",
        ]
        for cat in categories:
            menu_lines.append(
                f"{cat['icon']} **{cat['display']}** — "
                f"{cat['num_rights']} rights covered"
            )
        menu_lines.extend([
            "",
            "**Examples:**",
            '  • "What are my rights as a tenant?"',
            '  • "Know my rights as an employee"',
            '  • "Bail rights of arrested person"',
            '  • "Consumer rights India"',
            '  • "Women\'s rights domestic violence"',
        ])
        menu_response = "\n".join(menu_lines)
    except Exception:
        menu_response = (
            "Please specify which rights you need: tenant, employee, consumer, women, or bail."
        )

    return {
        "rights_category": "",
        "rights_result":   {},
        "rag_result":      {"answer": menu_response, "sources_consulted": 0,
                            "mode": "rights_node_menu"},
        "response":       menu_response,
        "rag_grade":      "good",
        "pipeline_depth": state.get("pipeline_depth", 1) + 1,
        "error":          "",
    }


def _detect_rights_category(query_lower: str) -> str:
    """
    Detect rights category from query text.

    Returns category key or empty string if ambiguous.
    Detection order matters — "domestic violence" should hit "women",
    not "bail" even though arrest is mentioned in the same query.
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

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: GENERAL
# ═══════════════════════════════════════════════════════════════════════════════

def general_node(state: AgentState) -> dict:
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
    print("[Graph] general_node → direct LLM")

    try:
        answer = llm.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=512)
        return {
            "response": answer,
            "rag_result": {
                "answer": answer, "sources_consulted": 0,
                "synthesis_note": "", "grounding_warning": "",
                "rewritten_queries": [], "reranker_used": False, "mode": "general_node",
            },
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error": "",
        }
    except Exception as exc:
        print(f"[Graph] general_node ERROR: {exc}")
        return {
            "response":   "Hello! I'm LexShield AI, your Indian legal assistant. How can I help you today?",
            "rag_result": {},
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
        "rights_node", "general_node",
    ]:
        builder.add_edge(node_name, END)

    return builder.compile(checkpointer=checkpointer)


agent_graph = build_graph()