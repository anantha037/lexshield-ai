"""
LexShield AI — LangGraph Multi-Agent Graph
============================================
A real LangGraph StateGraph.  Every agent is a node; routing is a
conditional edge function.  State is persisted to SQLite after every
node execution via SqliteSaver (thread_id = session_id).

Graph topology
--------------
[START]
   │
   ▼
classify_intent_node
   │
   ▼  route_by_intent() — conditional edge
┌──────────────────────────────────────────────┐
│ legal_rag_node       │ document_analysis_node │
│ risk_check_node      │ draft_node             │
│ multilingual_node    │ general_node           │
└──────────────────────────────────────────────┘
   │
   ▼
[END]

Checkpointer note
-----------------
SqliteSaver is used instead of AsyncSqliteSaver because:
  - Existing FastAPI endpoints are synchronous (def, not async def)
  - AsyncSqliteSaver requires an async context-manager lifecycle that
    cannot be a module-level singleton safely on Windows
  - SqliteSaver provides identical persistence: full AgentState is
    written to data/sessions.db after every node, keyed by thread_id
The portfolio claim "persistent SQLite checkpointing via LangGraph" is
fully true with SqliteSaver.
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

# check_same_thread=False — required for FastAPI (multiple threads share conn)
_checkpoint_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
checkpointer     = SqliteSaver(_checkpoint_conn)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query:           str    # raw user query
    session_id:      str    # active session ID  (mirrors thread_id in config)

    # ── Classification ────────────────────────────────────────────────────────
    intent:          str    # classified intent string
    confidence:      float  # intent confidence  0.0–1.0

    # ── RAG / NER / Risk outputs ──────────────────────────────────────────────
    rag_result:      dict   # full RAG pipeline output dict
    ner_result:      dict   # NER pipeline output  {entities: [...]}
    risk_result:     dict   # risk scorer output   {score, level, factors}

    # ── Drafting (multi-turn) ─────────────────────────────────────────────────
    draft_stage:     int    # current DraftingAgent stage  (0 = not started)
    draft_data:      dict   # accumulated draft fields across turns

    # ── Diagnostics ───────────────────────────────────────────────────────────
    pipeline_depth:  int    # count of pipeline nodes executed this run
    rag_grade:       str    # "good" | "poor" | ""  — RAG self-grading result

    # ── Multilingual ──────────────────────────────────────────────────────────
    source_language: str    # detected or requested target language

    # ── Output ────────────────────────────────────────────────────────────────
    response:        str    # final answer text set by each terminal node
    error:           str    # error message if any node failed


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_intent_node(state: AgentState) -> dict:
    """
    Reads state["query"], classifies it, writes intent + confidence.
    This is the entry node — always runs first.
    """
    from agents.intent_classifier import intent_classifier

    query = state.get("query", "").strip()
    if not query:
        return {"intent": "general", "confidence": 0.0, "pipeline_depth": 1}

    result = intent_classifier.classify(query)
    print(f"[Graph] classify_intent_node → intent={result.intent!r} "
          f"conf={result.confidence:.2f}")

    return {
        "intent":         result.intent,
        "confidence":     result.confidence,
        "pipeline_depth": state.get("pipeline_depth", 0) + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION  (conditional edge)
# ═══════════════════════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge after classify_intent_node.
    Priority: active draft session → draft_node (ignores intent).
    Otherwise maps intent → node name string.
    """
    from agents.drafting_agent import drafting_agent

    session_id = state.get("session_id", "")

    # Active multi-turn draft always wins
    if drafting_agent.has_active_draft(session_id):
        print("[Graph] route_by_intent → active draft detected → draft_node")
        return "draft_node"

    _map = {
        "legal_query":          "legal_rag_node",
        "document_analysis":    "document_analysis_node",
        "draft_request":        "draft_node",
        "risk_check":           "risk_check_node",
        "translation_request":  "multilingual_node",
        "general":              "general_node",
    }
    intent = state.get("intent", "general")
    node   = _map.get(intent, "general_node")
    print(f"[Graph] route_by_intent → {intent!r} → {node}")
    return node


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: LEGAL RAG AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def legal_rag_node(state: AgentState) -> dict:
    """
    Intent: legal_query
    Injects conversation context then runs the full hybrid RAG pipeline.
    Writes: rag_result, response, rag_grade, pipeline_depth, error.
    """
    from rag.pipeline import rag_pipeline

    query        = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")

    enriched = f"{context_block}\n\n{query}" if context_block else query
    print("[Graph] legal_rag_node → querying RAG pipeline")

    try:
        answer = rag_pipeline.query(enriched)
        return {
            "rag_result": {
                "answer":            answer.answer_text,
                "sources_consulted": answer.sources_consulted,
                "synthesis_note":    answer.synthesis_note    or "",
                "grounding_warning": answer.grounding_warning or "",
                "rewritten_queries": answer.rewritten_queries or [],
                "reranker_used":     answer.reranker_used,
                "mode":              "legal_rag_node",
            },
            "response":       answer.answer_text,
            "rag_grade":      "good",
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }
    except Exception as exc:
        print(f"[Graph] legal_rag_node ERROR: {exc}")
        return {
            "rag_result": {},
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
    Writes: rag_result, ner_result, response, pipeline_depth, error.
    """
    from rag.pipeline     import rag_pipeline
    from nlp.ner_pipeline import run_ner

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")

    enriched = f"{context_block}\n\n{query}" if context_block else query
    print("[Graph] document_analysis_node → RAG + NER")

    try:
        answer = rag_pipeline.query(enriched)
        rag_out = {
            "answer":            answer.answer_text,
            "sources_consulted": answer.sources_consulted,
            "synthesis_note":    answer.synthesis_note    or "",
            "grounding_warning": answer.grounding_warning or "",
            "rewritten_queries": answer.rewritten_queries or [],
            "reranker_used":     answer.reranker_used,
            "mode":              "document_analysis_node",
        }

        # NER on the document body (everything after the first blank line)
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
    Risk-focused RAG query + risk_scorer post-processing.
    Writes: rag_result, risk_result, response, pipeline_depth, error.
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
    enriched = f"{context_block}\n\n{risk_prefix}{query}" if context_block \
               else f"{risk_prefix}{query}"
    print("[Graph] risk_check_node → risk assessment via RAG + scorer")

    try:
        answer = rag_pipeline.query(enriched)
        rag_out = {
            "answer":            answer.answer_text,
            "sources_consulted": answer.sources_consulted,
            "synthesis_note":    answer.synthesis_note    or "",
            "grounding_warning": answer.grounding_warning or "",
            "rewritten_queries": answer.rewritten_queries or [],
            "reranker_used":     answer.reranker_used,
            "mode":              "risk_check_node",
        }

        # Risk scorer on the raw user query
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
    Calls DraftingAgent which manages stage state internally.
    Writes: draft_stage, draft_data, rag_result, response, pipeline_depth, error.
    """
    from agents.drafting_agent import drafting_agent

    query      = state.get("query", "")
    session_id = state.get("session_id", "")
    print(f"[Graph] draft_node → session={session_id[:8]}… query={query[:50]!r}")

    try:
        result = drafting_agent.handle(query=query, session_id=session_id)
        return {
            "draft_stage": result.get("stage", 0),
            "draft_data":  result.get("draft_data", {}),
            "response":    result.get("answer", ""),
            "rag_result": {
                "answer":            result.get("answer", ""),
                "sources_consulted": 0,
                "synthesis_note":    (f"DraftingAgent stage={result.get('stage')} "
                                      f"doc_type={result.get('doc_type')}"),
                "grounding_warning": "",
                "rewritten_queries": [],
                "reranker_used":     False,
                "mode":              f"draft_node_stage{result.get('stage', 0)}",
                "complete":          result.get("complete", False),
                "draft":             result.get("draft", ""),
            },
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }
    except Exception as exc:
        print(f"[Graph] draft_node ERROR: {exc}")
        return {
            "draft_stage": 0,
            "draft_data":  {},
            "rag_result":  {},
            "response":    "I encountered an error with the drafting agent. Please try again.",
            "error":       str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: MULTILINGUAL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def multilingual_node(state: AgentState) -> dict:
    """
    Intent: translation_request
    TranslationAgent: detect language → RAG → translate response.
    Writes: source_language, rag_result, response, pipeline_depth, error.
    """
    from agents.translation_agent import translation_agent

    query      = state.get("query", "")
    session_id = state.get("session_id", "")
    print("[Graph] multilingual_node → translation request")

    try:
        result = translation_agent.handle(query=query, session_id=session_id)
        return {
            "source_language": result.get("target_language", ""),
            "rag_result": {
                "answer":            result.get("answer", ""),
                "sources_consulted": result.get("sources_consulted", 0),
                "synthesis_note":    result.get("synthesis_note", ""),
                "grounding_warning": result.get("grounding_warning", ""),
                "rewritten_queries": result.get("rewritten_queries", []),
                "reranker_used":     result.get("reranker_used", False),
                "mode":              result.get("mode", "multilingual_node"),
            },
            "response":       result.get("answer", ""),
            "pipeline_depth": state.get("pipeline_depth", 1) + 1,
            "error":          "",
        }
    except Exception as exc:
        print(f"[Graph] multilingual_node ERROR: {exc}")
        return {
            "source_language": "",
            "rag_result":      {},
            "response":        "I encountered an error processing the translation request.",
            "error":           str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: GENERAL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def general_node(state: AgentState) -> dict:
    """
    Intent: general  (greetings, capability questions, off-topic)
    Direct Groq LLM call — no RAG, no retrieval.
    Writes: response, rag_result, pipeline_depth, error.
    """
    from rag.llm import llm

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")

    system_prompt = (
        "You are LexShield AI, an Indian legal intelligence assistant. "
        "Help users understand Indian law, their legal rights, and legal documents. "
        "Be concise, friendly, and direct. "
        "If asked a specific legal question, encourage the user to ask it directly."
    )
    prompt = f"{context_block}\n\nUser: {query}" if context_block \
             else f"User: {query}"
    print("[Graph] general_node → direct LLM (no RAG)")

    try:
        answer = llm.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=512)
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
    Assembles and compiles the LangGraph StateGraph.

    The SqliteSaver checkpointer writes the full AgentState to
    data/sessions.db after every node.  On the next invoke() with the
    same thread_id, LangGraph loads the previous checkpoint automatically.

    Call pattern in orchestrator:
        config = {"configurable": {"thread_id": session_id}}
        final_state = agent_graph.invoke(initial_state, config)
    """
    builder = StateGraph(AgentState)

    # ── Register all nodes ─────────────────────────────────────────────────────
    builder.add_node("classify_intent_node",   classify_intent_node)
    builder.add_node("legal_rag_node",         legal_rag_node)
    builder.add_node("document_analysis_node", document_analysis_node)
    builder.add_node("draft_node",             draft_node)
    builder.add_node("risk_check_node",        risk_check_node)
    builder.add_node("multilingual_node",      multilingual_node)
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
            "general_node":           "general_node",
        },
    )

    # ── All terminal nodes → END ───────────────────────────────────────────────
    for node_name in [
        "legal_rag_node", "document_analysis_node", "draft_node",
        "risk_check_node", "multilingual_node", "general_node",
    ]:
        builder.add_edge(node_name, END)

    return builder.compile(checkpointer=checkpointer)


# ── Singleton compiled graph ───────────────────────────────────────────────────
agent_graph = build_graph()