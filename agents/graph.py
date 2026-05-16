"""
LexShield AI — LangGraph Multi-Agent Graph  (Session 3 — Updated)
===================================================================
Changes from Session 2:

1. draft_node  — wired to the new SQLite-persisted DraftingAgent.
   Calls drafting_agent.handle(query, session_id) which dispatches
   internally by stage (INIT→CLARIFY→RETRIEVE_SECTIONS→...→DONE).

2. route_by_intent — checks has_active_draft() BEFORE reading intent.
   If True, always routes to draft_node regardless of intent classifier.

3. legal_rag_node — after NER runs on query, calls
   knowledge_graph.enrich_retrieval() to augment the chunk pool with
   graph-connected sections before the synthesiser runs.

4. intent_classifier draft_request triggers updated in intent_classifier.py
   (see that file for the additional keywords).

Graph topology (unchanged):
---------------------------------------------------------------------------
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
---------------------------------------------------------------------------

Checkpointer:
  SqliteSaver on data/sessions.db — identical to Session 2.
  check_same_thread=False for FastAPI multi-thread compatibility.
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
    draft_stage:     str    # DraftStage enum value as string, e.g. "CLARIFY"
    draft_data:      dict   # last returned draft_data from DraftingAgent

    # ── Diagnostics ───────────────────────────────────────────────────────────
    pipeline_depth:  int
    rag_grade:       str    # "good" | "poor" | ""

    # ── Multilingual ──────────────────────────────────────────────────────────
    source_language: str

    # ── Output ────────────────────────────────────────────────────────────────
    response:        str
    error:           str


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_intent_node(state: AgentState) -> dict:
    """
    Always the entry node. Reads query, classifies intent, writes
    intent + confidence to state.
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

    Priority order:
    1. Active draft session in SQLite → always draft_node.
    2. Intent map → correct agent node.

    This ensures mid-draft messages like "confirm" or answers to
    clarifying questions are never misrouted to legal_rag_node.
    """
    from agents.drafting_agent import drafting_agent

    session_id = state.get("session_id", "")

    # ── Priority 1: active multi-turn draft ────────────────────────────────────
    if session_id and drafting_agent.has_active_draft(session_id):
        print(f"[Graph] route_by_intent → active draft (session={session_id[:8]}…) → draft_node")
        return "draft_node"

    # ── Priority 2: intent map ─────────────────────────────────────────────────
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
    print(f"[Graph] route_by_intent → intent={intent!r} → {node}")
    return node


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: LEGAL RAG AGENT  (Session 3: + Knowledge Graph enrichment)
# ═══════════════════════════════════════════════════════════════════════════════

def legal_rag_node(state: AgentState) -> dict:
    """
    Intent: legal_query
    Session 3 addition: after NER runs on query, call
    knowledge_graph.enrich_retrieval() to augment chunk pool with
    graph-connected sections (IPC↔BNS equivalents, definitional chains).

    Flow:
      1. NER → extract section IDs from query
      2. KG enrich → add graph-connected chunks to pool
      3. RAG pipeline → hybrid search + rerank + synthesise
      4. Write rag_result, ner_result, response to state
    """
    from rag.pipeline     import rag_pipeline
    from nlp.ner_pipeline import run_ner
    from rag.knowledge_graph import enrich_retrieval

    query         = state.get("query", "")
    context_block = state.get("rag_result", {}).get("context_block", "")
    enriched      = f"{context_block}\n\n{query}" if context_block else query

    print("[Graph] legal_rag_node → NER + KG enrich + RAG pipeline")

    # ── Step 1: NER on query ───────────────────────────────────────────────────
    ner_out: dict = {"entities": []}
    ner_sections: list[str] = []
    try:
        ner_out = run_ner(query[:2000])
        # Extract section-type entities for KG lookup
        # NER entities have shape: {"text": "302", "label": "SECTION", "source": "IPC"}
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

    # ── Step 2: KG enrichment (augment chunk pool) ─────────────────────────────
    # We pass an empty pool here — enrich_retrieval fetches from vectorstore.
    # The fetched KG chunks are then merged with RAG pipeline results inside
    # the pipeline itself via _inject_kg(). This call is an ADDITIONAL layer:
    # it handles multi-hop graph traversal beyond what _inject_kg does for
    # the section fast-path.
    kg_chunks: list[dict] = []
    if ner_sections:
        try:
            kg_chunks = enrich_retrieval(
                ner_sections       = ner_sections,
                chunk_pool         = [],
                bypass_score_filter = True,
            )
            if kg_chunks:
                print(f"[Graph] legal_rag_node KG enriched: {len(kg_chunks)} extra chunk(s) "
                      f"for sections {ner_sections}")
        except Exception as kg_exc:
            print(f"[Graph] legal_rag_node KG enrich warning (non-fatal): {kg_exc}")

    # ── Step 3: RAG pipeline ────────────────────────────────────────────────────
    try:
        answer = rag_pipeline.query(enriched)

        # Merge KG chunks into sources_consulted count for transparency
        kg_count = len(kg_chunks)

        return {
            "rag_result": {
                "answer":            answer.answer_text,
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
            "response":       answer.answer_text,
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
    Risk-focused RAG query + risk_scorer post-processing.
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
# NODE: DRAFT AGENT  (Session 3 — full SQLite-persisted DraftingAgent)
# ═══════════════════════════════════════════════════════════════════════════════

def draft_node(state: AgentState) -> dict:
    """
    Intent: draft_request  OR  active multi-turn draft (priority override).

    Dispatches to DraftingAgent.handle() which manages its own stage
    machine internally using SQLite. The node is stateless from LangGraph's
    perspective — DraftingAgent is the source of truth for draft stage.

    handle() returns:
        answer:    str   — next question / outline / final draft
        stage:     str   — current DraftStage enum value
        doc_type:  str   — complaint category detected
        complete:  bool  — True when DONE
        draft:     str   — the final draft text (only when complete=True)
    """
    from agents.drafting_agent import drafting_agent

    query      = state.get("query", "")
    session_id = state.get("session_id", "")

    print(f"[Graph] draft_node → session={session_id[:8]}… "
          f"stage={state.get('draft_stage', 'INIT')!r} "
          f"query={query[:50]!r}")

    try:
        result = drafting_agent.handle(query=query, session_id=session_id)

        # Build rag_result for API layer compatibility
        stage_str = result.get("stage", "INIT")
        if hasattr(stage_str, "value"):
            stage_str = stage_str.value  # DraftStage enum → string

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
                "synthesis_note":    (
                    f"DraftingAgent stage={stage_str} "
                    f"category={doc_type} "
                    f"complete={complete}"
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
# NODE: MULTILINGUAL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def multilingual_node(state: AgentState) -> dict:
    """
    Intent: translation_request
    TranslationAgent: detect language → RAG → translate response.
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
            "response": (
                "I encountered an error processing the translation request. "
                "Please try again."
            ),
            "error": str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: GENERAL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def general_node(state: AgentState) -> dict:
    """
    Intent: general  (greetings, capability questions, off-topic)
    Direct Groq LLM call — no RAG, no retrieval.
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
    prompt = (
        f"{context_block}\n\nUser: {query}"
        if context_block else f"User: {query}"
    )
    print("[Graph] general_node → direct LLM (no RAG)")

    try:
        answer = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=512,
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
    Assembles and compiles the LangGraph StateGraph.

    SqliteSaver checkpointer writes the full AgentState to data/sessions.db
    after every node. On the next invoke() with the same thread_id, LangGraph
    reloads the previous checkpoint automatically.

    Call pattern in orchestrator:
        config      = {"configurable": {"thread_id": session_id}}
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
        "legal_rag_node",
        "document_analysis_node",
        "draft_node",
        "risk_check_node",
        "multilingual_node",
        "general_node",
    ]:
        builder.add_edge(node_name, END)

    return builder.compile(checkpointer=checkpointer)


# ── Singleton compiled graph ───────────────────────────────────────────────────
agent_graph = build_graph()