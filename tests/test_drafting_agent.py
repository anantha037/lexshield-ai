"""
Day 3 Checkpoint — Drafting Agent Multi-Turn Tests
====================================================
Simulates complete 3-turn FIR and legal notice drafting workflows.
LLM calls are mocked to keep tests fast and free.

Run: pytest tests/test_drafting_agent.py -v
"""

import pytest
from unittest.mock import patch
from agents.drafting_agent import drafting_agent, DraftingAgent


# ── Fresh agent per test to avoid session bleed ───────────────────────────────

@pytest.fixture
def agent():
    """Return a fresh DraftingAgent instance for each test."""
    return DraftingAgent()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Unknown doc type shows menu
# ═══════════════════════════════════════════════════════════════════════════════

def test_unknown_doc_type_returns_menu(agent):
    result = agent.handle("I need some legal help", session_id="s1")
    assert result["stage"]    == 0
    assert result["complete"] == False
    assert "FIR" in result["answer"]
    assert "Legal Notice" in result["answer"]
    print(f"\nOK Unknown doc type -> menu shown")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Full 3-turn FIR workflow
# ═══════════════════════════════════════════════════════════════════════════════

def test_fir_three_turn_workflow(agent):
    """
    Turn 1: Trigger FIR draft
    Turn 2: Provide incident details
    Turn 3: Provide party details -> get draft
    """
    sid = "fir-session-001"

    # Turn 1 — trigger
    r1 = agent.handle("Help me draft an FIR for theft", session_id=sid)
    assert r1["stage"]    == 1,    f"Expected stage 1, got {r1['stage']}"
    assert r1["doc_type"] == "fir"
    assert r1["complete"] == False
    assert "incident" in r1["answer"].lower() or "happened" in r1["answer"].lower()
    assert agent.has_active_draft(sid)
    print(f"\nOK FIR Turn 1 — stage=1, asking for incident details")

    # Turn 2 — incident details
    r2 = agent.handle(
        "My laptop worth Rs 55000 was stolen from my house on 10th May 2025 at around 2pm. "
        "The thief broke the window lock and entered the house while I was at work.",
        session_id=sid,
    )
    assert r2["stage"]    == 2,    f"Expected stage 2, got {r2['stage']}"
    assert r2["complete"] == False
    assert "party" in r2["answer"].lower() or "name" in r2["answer"].lower()
    assert agent.has_active_draft(sid)
    print(f"\nOK FIR Turn 2 — stage=2, asking for party details")

    # Turn 3 — party details + draft generation (mock LLM)
    mock_draft = (
        "FIRST INFORMATION REPORT\n"
        "FIR No.: ___/2025\n"
        "Complainant: Anantha Krishnan K\n"
        "Offence: Theft under Section 303 BNS\n"
        "...[complete FIR draft]..."
    )

    with patch.object(agent, "_generate_draft", return_value=mock_draft):
        r3 = agent.handle(
            "My name is Anantha Krishnan K, residing at MG Road Ernakulam Kerala. "
            "Contact: 9876543210. Accused is unknown. "
            "I want the FIR registered and stolen laptop recovered.",
            session_id=sid,
        )

    assert r3["stage"]    == 3,        f"Expected stage 3, got {r3['stage']}"
    assert r3["complete"] == True,     "Expected complete=True at stage 3"
    assert r3["draft"]    == mock_draft
    assert mock_draft in r3["answer"]
    assert not agent.has_active_draft(sid)  # session cleaned up
    print(f"\nOK FIR Turn 3 — draft generated, session cleared")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Full 3-turn Legal Notice (NI Act) workflow
# ═══════════════════════════════════════════════════════════════════════════════

def test_legal_notice_ni_three_turn_workflow(agent):
    sid = "ni-session-001"

    # Turn 1
    r1 = agent.handle("Draft a cheque bounce legal notice under Section 138 NI Act", session_id=sid)
    assert r1["stage"]    == 1
    assert r1["doc_type"] == "legal_notice_ni"
    assert "cheque" in r1["answer"].lower()
    print(f"\nOK NI Notice Turn 1 — stage=1, asking for cheque details")

    # Turn 2
    r2 = agent.handle(
        "Cheque No. 004521, Rs. 1,50,000, dated 1st April 2025, State Bank of India Ernakulam. "
        "Dishonoured on 5th April 2025 with reason 'Insufficient Funds'. "
        "Cheque was for repayment of personal loan.",
        session_id=sid,
    )
    assert r2["stage"]    == 2
    assert r2["complete"] == False
    print(f"\nOK NI Notice Turn 2 — stage=2, asking for party details")

    # Turn 3 with mock
    mock_draft = "LEGAL NOTICE\nUnder Section 138 NI Act...[complete notice]..."
    with patch.object(agent, "_generate_draft", return_value=mock_draft):
        r3 = agent.handle(
            "My name is Ravi Kumar, 45 Gandhi Nagar, Kochi. "
            "Drawer: Suresh Menon, 12 Park Street, Kochi. "
            "Demanding Rs 1,50,000 within 15 days.",
            session_id=sid,
        )

    assert r3["complete"] == True
    assert r3["draft"]    == mock_draft
    assert not agent.has_active_draft(sid)
    print(f"\nOK NI Notice Turn 3 — draft generated, session cleared")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 — has_active_draft lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

def test_has_active_draft_lifecycle(agent):
    sid = "lifecycle-session"

    assert not agent.has_active_draft(sid)                          # before start

    agent.handle("Draft an FIR", session_id=sid)
    assert agent.has_active_draft(sid)                              # after stage 1

    agent.handle("Theft at my home yesterday", session_id=sid)
    assert agent.has_active_draft(sid)                              # after stage 2

    with patch.object(agent, "_generate_draft", return_value="draft"):
        agent.handle("My name is Test User", session_id=sid)
    assert not agent.has_active_draft(sid)                          # after stage 3

    print(f"\nOK has_active_draft lifecycle correct across all 3 stages")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Cancel draft mid-session
# ═══════════════════════════════════════════════════════════════════════════════

def test_cancel_draft(agent):
    sid = "cancel-session"

    agent.handle("Draft a rental agreement", session_id=sid)
    assert agent.has_active_draft(sid)

    cancelled = agent.cancel_draft(sid)
    assert cancelled == True
    assert not agent.has_active_draft(sid)

    # Cancel non-existent returns False
    assert agent.cancel_draft("no-such-session") == False
    print(f"\nOK cancel_draft clears session correctly")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Graph routes follow-up turns to draft_node via active draft check
# ═══════════════════════════════════════════════════════════════════════════════

def test_graph_routes_active_draft_followup():
    """
    After stage 1, a follow-up message like 'The theft happened yesterday'
    must route to draft_node (not legal_query) because has_active_draft=True.
    """
    from agents.graph import route_by_intent, AgentState

    sid = "graph-draft-route-test"

    # Start a draft session on the singleton drafting_agent
    drafting_agent.handle("Draft an FIR for assault", session_id=sid)
    assert drafting_agent.has_active_draft(sid)

    # Now simulate a follow-up that looks like legal_query by intent
    state: AgentState = {
        "query":      "Section 351 BNS — the assault happened yesterday at 6pm",
        "intent":     "legal_query",   # classifier would say this
        "confidence": 0.8,
        "context":    "",
        "session_id": sid,
        "result":     {},
        "draft":      "",
        "language":   "",
    }

    node = route_by_intent(state)
    assert node == "draft_node", f"Expected draft_node for active draft session, got {node}"

    # Cleanup
    drafting_agent.cancel_draft(sid)
    print(f"\nOK Active draft session overrides intent routing -> draft_node")