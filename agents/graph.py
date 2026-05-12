"""
LexShield AI — LangGraph Multi-Agent Graph
============================================
AgentState flows through nodes based on classified intent.

Graph topology:
  [START]
     │
     ▼
  classify_intent_node
     │
     ▼ (conditional on state["intent"])
  ┌──────────────────────────────────────────┐
  │ legal_rag_node  │ document_node          │
  │ risk_node       │ draft_node             │
  │ translation_node│ general_node           │
  └──────────────────────────────────────────┘
     │
     ▼
  [END]

Each node receives full AgentState and returns a partial
state dict — LangGraph merges it automatically.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END


# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    query:      str               # raw user query
    intent:     str               # classified intent
    confidence: float             # intent confidence score
    context:    str               # conversation history block from SessionMemory
    session_id: str               # active session ID
    result:     dict              # final answer payload
    draft:      str               # populated by DraftingAgent (Day 3)
    language:   str               # target language for translation (Day 4-5)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_intent_node(state: AgentState) -> dict:
    """
    Classifies intent from state["query"].
    Writes: intent, confidence
    """
    from agents.intent_classifier import intent_classifier

    query = state.get("query", "").strip()
    if not query:
        return {"intent": "general", "confidence": 0.0}

    result = intent_classifier.classify(query)
    print(f"[Graph] classify_intent_node → intent={result.intent!r} conf={result.confidence:.2f}")

    return {
        "intent":     result.intent,
        "confidence": result.confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge function.
    Maps intent → next node name.
    Active draft sessions always route to draft_node regardless of intent.
    """
    from agents.drafting_agent import drafting_agent

    session_id = state.get("session_id", "")

    # Active multi-turn draft takes priority over intent classification
    if drafting_agent.has_active_draft(session_id):
        print(f"[Graph] route_by_intent → active draft detected → draft_node")
        return "draft_node"

    intent_to_node = {
        "legal_query":          "legal_rag_node",
        "document_analysis":    "document_node",
        "draft_request":        "draft_node",
        "risk_check":           "risk_node",
        "translation_request":  "translation_node",
        "general":              "general_node",
    }
    intent = state.get("intent", "general")
    node   = intent_to_node.get(intent, "general_node")
    print(f"[Graph] route_by_intent → {intent!r} → {node}")
    return node


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: LEGAL RAG AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def legal_rag_node(state: AgentState) -> dict:
    """
    Handles legal_query intent.
    Injects conversation context into query before RAG call.
    Writes: result
    """
    from rag.pipeline import rag_pipeline

    query   = state.get("query", "")
    context = state.get("context", "")

    enriched = f"{context}\n\n{query}" if context else query
    print(f"[Graph] legal_rag_node → querying RAG")

    legal_answer = rag_pipeline.query(enriched)

    return {
        "result": {
            "answer":            legal_answer.answer_text,
            "sources_consulted": legal_answer.sources_consulted,
            "synthesis_note":    legal_answer.synthesis_note or "",
            "grounding_warning": legal_answer.grounding_warning or "",
            "rewritten_queries": legal_answer.rewritten_queries or [],
            "reranker_used":     legal_answer.reranker_used,
            "mode":              "legal_rag_node",
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: DOCUMENT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def document_node(state: AgentState) -> dict:
    """
    Handles document_analysis intent.
    Runs document text through RAG for analysis and summary.
    Writes: result
    """
    from rag.pipeline import rag_pipeline

    query   = state.get("query", "")
    context = state.get("context", "")

    enriched = f"{context}\n\n{query}" if context else query
    print(f"[Graph] document_node → analysing document via RAG")

    legal_answer = rag_pipeline.query(enriched)

    return {
        "result": {
            "answer":            legal_answer.answer_text,
            "sources_consulted": legal_answer.sources_consulted,
            "synthesis_note":    legal_answer.synthesis_note or "",
            "grounding_warning": legal_answer.grounding_warning or "",
            "rewritten_queries": legal_answer.rewritten_queries or [],
            "reranker_used":     legal_answer.reranker_used,
            "mode":              "document_node",
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: RISK NODE
# ═══════════════════════════════════════════════════════════════════════════════

def risk_node(state: AgentState) -> dict:
    """
    Handles risk_check intent.
    Prepends a risk assessment instruction before calling RAG.
    Writes: result
    """
    from rag.pipeline import rag_pipeline

    query   = state.get("query", "")
    context = state.get("context", "")

    risk_query = (
        "Provide a detailed legal risk assessment for the following. "
        "Identify applicable Indian laws, potential penalties, liabilities, "
        f"and legal consequences:\n\n{query}"
    )
    enriched = f"{context}\n\n{risk_query}" if context else risk_query
    print(f"[Graph] risk_node → risk assessment via RAG")

    legal_answer = rag_pipeline.query(enriched)

    return {
        "result": {
            "answer":            legal_answer.answer_text,
            "sources_consulted": legal_answer.sources_consulted,
            "synthesis_note":    legal_answer.synthesis_note or "",
            "grounding_warning": legal_answer.grounding_warning or "",
            "rewritten_queries": legal_answer.rewritten_queries or [],
            "reranker_used":     legal_answer.reranker_used,
            "mode":              "risk_node",
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: DRAFT NODE (Stub — Day 3)
# ═══════════════════════════════════════════════════════════════════════════════

def draft_node(state: AgentState) -> dict:
    """
    Handles draft_request intent and active multi-turn draft sessions.
    Calls DraftingAgent which manages stage state internally.
    Writes: result, draft
    """
    from agents.drafting_agent import drafting_agent

    query      = state.get("query", "")
    session_id = state.get("session_id", "")

    print(f"[Graph] draft_node → session={session_id[:8]} query={query[:50]!r}")

    draft_result = drafting_agent.handle(query=query, session_id=session_id)

    return {
        "draft": draft_result.get("draft", ""),
        "result": {
            "answer":            draft_result["answer"],
            "sources_consulted": 0,
            "synthesis_note":    f"DraftingAgent stage={draft_result['stage']} doc_type={draft_result['doc_type']}",
            "grounding_warning": "",
            "rewritten_queries": [],
            "reranker_used":     False,
            "mode":              f"draft_node_stage{draft_result['stage']}",
            "complete":          draft_result.get("complete", False),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: TRANSLATION NODE (Stub — Day 4-5)
# ═══════════════════════════════════════════════════════════════════════════════

def translation_node(state: AgentState) -> dict:
    """
    Handles translation_request intent.
    Stub — MultilingualAgent implemented in Day 4-5.
    Writes: result, language
    """
    print(f"[Graph] translation_node → stub")
    answer = (
        "The multilingual translation agent is being built and will be "
        "available shortly. It will support Malayalam, Hindi, Tamil, "
        "Telugu, Kannada and other Indian languages."
    )
    return {
        "language": "",
        "result":   {
            "answer":            answer,
            "sources_consulted": 0,
            "synthesis_note":    "MultilingualAgent stub — Day 4-5",
            "grounding_warning": "",
            "rewritten_queries": [],
            "reranker_used":     False,
            "mode":              "translation_node_stub",
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE: GENERAL NODE
# ═══════════════════════════════════════════════════════════════════════════════

def general_node(state: AgentState) -> dict:
    """
    Handles general intent.
    Direct LLM call — no RAG, no retrieval.
    Writes: result
    """
    from rag.llm import llm

    query   = state.get("query", "")
    context = state.get("context", "")

    system_prompt = (
        "You are LexShield AI, an Indian legal intelligence assistant. "
        "Help users understand Indian law, their legal rights, and legal documents. "
        "Be concise, friendly, and direct. "
        "If asked about specific legal questions, encourage the user to ask them directly."
    )
    prompt = f"{context}\n\nUser: {query}" if context else f"User: {query}"

    print(f"[Graph] general_node → direct LLM")
    answer = llm.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=512)

    return {
        "result": {
            "answer":            answer,
            "sources_consulted": 0,
            "synthesis_note":    "",
            "grounding_warning": "",
            "rewritten_queries": [],
            "reranker_used":     False,
            "mode":              "general_node",
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Assembles and compiles the LangGraph StateGraph.
    Returns a compiled graph ready for .invoke().
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    builder.add_node("classify_intent_node", classify_intent_node)
    builder.add_node("legal_rag_node",       legal_rag_node)
    builder.add_node("document_node",        document_node)
    builder.add_node("draft_node",           draft_node)
    builder.add_node("risk_node",            risk_node)
    builder.add_node("translation_node",     translation_node)
    builder.add_node("general_node",         general_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    builder.set_entry_point("classify_intent_node")

    # ── Conditional edges from classifier ──────────────────────────────────────
    builder.add_conditional_edges(
        "classify_intent_node",
        route_by_intent,
        {
            "legal_rag_node":    "legal_rag_node",
            "document_node":     "document_node",
            "draft_node":        "draft_node",
            "risk_node":         "risk_node",
            "translation_node":  "translation_node",
            "general_node":      "general_node",
        },
    )

    # ── All agent nodes terminate at END ───────────────────────────────────────
    for node in ["legal_rag_node", "document_node", "draft_node",
                 "risk_node", "translation_node", "general_node"]:
        builder.add_edge(node, END)

    return builder.compile()


# ── Singleton compiled graph ───────────────────────────────────────────────────
agent_graph = build_graph()