"""
Day 2 Checkpoint — LangGraph Multi-Agent Graph Tests
=====================================================
3 queries hitting 3 different nodes.
Does NOT call real RAG/LLM — patches them to keep tests fast.

Run: pytest tests/test_graph.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.graph import agent_graph, AgentState


# ── Shared mock LegalAnswer ────────────────────────────────────────────────────

def _mock_legal_answer(text="Mock RAG answer"):
    ans = MagicMock()
    ans.answer_text       = text
    ans.sources_consulted = 3
    ans.synthesis_note    = "mocked"
    ans.grounding_warning = ""
    ans.rewritten_queries = []
    ans.reranker_used     = False
    return ans


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — legal_query → legal_rag_node
# ═══════════════════════════════════════════════════════════════════════════════

def test_graph_legal_query():
    """
    Query about IPC section must route through legal_rag_node.
    Verifies: intent=legal_query, mode=legal_rag_node, answer present.
    """
    with patch("rag.pipeline.rag_pipeline") as mock_rag:
        mock_rag.query.return_value = _mock_legal_answer("Section 379 IPC covers theft.")

        initial: AgentState = {
            "query":      "What is the punishment for theft under Section 379 IPC?",
            "intent":     "",
            "confidence": 0.0,
            "context":    "",
            "session_id": "test-session-1",
            "result":     {},
            "draft":      "",
            "language":   "",
        }

        final = agent_graph.invoke(initial)

        assert final["intent"]          == "legal_query",    f"Expected legal_query, got {final['intent']}"
        assert final["result"]["mode"]  == "legal_rag_node", f"Expected legal_rag_node, got {final['result']['mode']}"
        assert final["result"]["answer"] != ""
        print(f"\n✓ legal_query → legal_rag_node | answer: {final['result']['answer'][:60]}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — draft_request → draft_node
# ═══════════════════════════════════════════════════════════════════════════════

def test_graph_draft_request():
    """
    Drafting request must route to draft_node stub.
    Verifies: intent=draft_request, mode=draft_node_stub, answer present.
    No mocking needed — draft_node is a pure stub.
    """
    initial: AgentState = {
        "query":      "Help me draft a legal notice for cheque bounce under Section 138 NI Act",
        "intent":     "",
        "confidence": 0.0,
        "context":    "",
        "session_id": "test-session-2",
        "result":     {},
        "draft":      "",
        "language":   "",
    }

    final = agent_graph.invoke(initial)

    assert final["intent"]         == "draft_request",    f"Expected draft_request, got {final['intent']}"
    assert final["result"]["mode"] == "draft_node_stub",  f"Expected draft_node_stub, got {final['result']['mode']}"
    assert "drafting agent" in final["result"]["answer"].lower()
    print(f"\n✓ draft_request → draft_node | answer: {final['result']['answer'][:60]}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — general → general_node
# ═══════════════════════════════════════════════════════════════════════════════

def test_graph_general():
    """
    Greeting must route to general_node with direct LLM call.
    Verifies: intent=general, mode=general_node, answer present.
    """
    with patch("rag.llm.llm") as mock_llm:
        mock_llm.generate.return_value = "Hello! I am LexShield AI. How can I help you today?"

        initial: AgentState = {
            "query":      "Hello, what can you do?",
            "intent":     "",
            "confidence": 0.0,
            "context":    "",
            "session_id": "test-session-3",
            "result":     {},
            "draft":      "",
            "language":   "",
        }

        final = agent_graph.invoke(initial)

        assert final["intent"]         == "general",      f"Expected general, got {final['intent']}"
        assert final["result"]["mode"] == "general_node", f"Expected general_node, got {final['result']['mode']}"
        assert final["result"]["answer"] != ""
        print(f"\n✓ general → general_node | answer: {final['result']['answer'][:60]}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 — State keys always present
# ═══════════════════════════════════════════════════════════════════════════════

def test_graph_state_keys_always_present():
    """
    Final state must always contain all AgentState keys regardless of node taken.
    """
    initial: AgentState = {
        "query":      "Translate this into Malayalam",
        "intent":     "",
        "confidence": 0.0,
        "context":    "",
        "session_id": "test-session-4",
        "result":     {},
        "draft":      "",
        "language":   "",
    }

    final = agent_graph.invoke(initial)

    for key in ["query", "intent", "confidence", "context",
                "session_id", "result", "draft", "language"]:
        assert key in final, f"Missing key in final state: {key!r}"

    print(f"\n✓ All AgentState keys present | intent={final['intent']}")