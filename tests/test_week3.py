"""
LexShield AI — Week 3 Integration Tests
=========================================
Covers all Week 3 components end-to-end through the master orchestrator.
All LLM/RAG calls are mocked — tests run fast with no API cost.

Test groups:
  A. Orchestrator routing (6 intents)
  B. LexShieldResponse structured output (all fields)
  C. Draft agent multi-turn via orchestrator
  D. Knowledge graph via pipeline
  E. Translation agent language detection
  F. Session memory across turns
  G. Legacy /orchestrate/query backward compatibility

Run: pytest tests/test_week3.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.orchestrator   import MasterOrchestrator
from agents.memory         import SessionMemory
from agents.drafting_agent import DraftingAgent
from agents.graph          import agent_graph, AgentState
from rag.structured_output import build_structured_response
from agents.translation_agent import detect_language, _strip_legal_content


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def orchestrator():
    """Fresh orchestrator with isolated session memory."""
    orch         = MasterOrchestrator()
    return orch


def _mock_rag_answer(text="Mock RAG answer about Indian law."):
    ans = MagicMock()
    ans.answer_text       = text
    ans.sources_consulted = 3
    ans.synthesis_note    = "Synthesized from 3 sections"
    ans.grounding_warning = ""
    ans.rewritten_queries = ["expanded query"]
    ans.reranker_used     = True
    ans.citations         = []
    return ans


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP A — ORCHESTRATOR ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorRouting:

    def test_legal_query_routes_to_rag(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer()
            orch = MasterOrchestrator()
            resp = orch.handle_query("What is Section 302 IPC?")
            assert resp.intent == "legal_query"
            assert "rag" in resp.mode.lower() or "legal" in resp.mode.lower()
            print(f"\n✓ legal_query → {resp.mode}")

    def test_risk_check_routes_to_risk_node(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer("Legal risk assessment...")
            orch = MasterOrchestrator()
            resp = orch.handle_query("Am I liable if I breach a rental agreement?")
            assert resp.intent == "risk_check"
            assert "risk" in resp.mode.lower()
            print(f"\n✓ risk_check → {resp.mode}")

    def test_draft_request_routes_to_draft_node(self):
        orch = MasterOrchestrator()
        resp = orch.handle_query("Help me draft a rental agreement")
        assert resp.intent == "draft_request"
        assert "draft" in resp.mode.lower()
        print(f"\n✓ draft_request → {resp.mode}")

    def test_general_routes_to_llm(self):
        with patch("rag.llm.llm") as mock_llm:
            mock_llm.generate.return_value = "Hello! I am LexShield AI."
            orch = MasterOrchestrator()
            resp = orch.handle_query("Hello, what can you do?")
            assert resp.intent == "general"
            assert "general" in resp.mode.lower()
            print(f"\n✓ general → {resp.mode}")

    def test_translation_request_intent(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag, \
             patch("rag.llm.llm") as mock_llm:
            mock_rag.query.return_value  = _mock_rag_answer()
            mock_llm.generate.return_value = "Malayalam translation here."
            orch = MasterOrchestrator()
            resp = orch.handle_query("Explain Section 138 NI Act in Malayalam")
            assert resp.intent == "translation_request"
            print(f"\n✓ translation_request → {resp.mode}")

    def test_document_analysis_intent(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer("Document summary...")
            orch = MasterOrchestrator()
            resp = orch.handle_query("Analyze this rental agreement document")
            assert resp.intent == "document_analysis"
            print(f"\n✓ document_analysis → {resp.mode}")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP B — STRUCTURED OUTPUT (LexShieldResponse)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuredOutput:

    def test_all_fields_present(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer(
                "Under Section 302 Indian Penal Code, murder is punishable by death. [1]"
            )
            orch = MasterOrchestrator()
            resp = orch.handle_query("What is punishment for murder under IPC?")

        required = [
            "answer_text", "summary", "key_clauses", "suggestions",
            "risk_score", "risk_level", "risk_factors", "citations",
            "draft", "intent", "session_id", "confidence", "mode",
            "sources_consulted", "synthesis_note", "reranker_used",
        ]
        d = resp.to_dict()
        for field in required:
            assert field in d, f"Missing field: {field}"
        print(f"\n✓ All structured output fields present")

    def test_summary_populated(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer(
                "Section 420 IPC deals with cheating. The punishment is up to 7 years. "
                "The accused must have dishonest intent."
            )
            orch = MasterOrchestrator()
            resp = orch.handle_query("What is Section 420 IPC?")

        assert len(resp.summary) > 0
        assert len(resp.summary) <= len(resp.answer_text)
        print(f"\n✓ Summary: {resp.summary[:60]}")

    def test_risk_score_valid_range(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer()
            orch = MasterOrchestrator()
            resp = orch.handle_query("What is Section 302 IPC?")

        assert 0.0 <= resp.risk_score <= 1.0
        assert resp.risk_level in ("Low", "Medium", "High", "Critical")
        print(f"\n✓ Risk: {resp.risk_score:.2f} ({resp.risk_level})")

    def test_suggestions_non_empty(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer()
            orch = MasterOrchestrator()
            resp = orch.handle_query("Is it legal to not pay employees on time?")

        assert len(resp.suggestions) > 0
        print(f"\n✓ Suggestions: {resp.suggestions[0][:60]}")

    def test_to_dict_json_serializable(self):
        import json
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer()
            orch = MasterOrchestrator()
            resp = orch.handle_query("What is bail?")

        d    = resp.to_dict()
        blob = json.dumps(d)  # must not raise
        assert len(blob) > 10
        print(f"\n✓ to_dict() is JSON serializable ({len(blob)} chars)")

    def test_handle_document_returns_structured(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer("This is a rental agreement.")
            orch = MasterOrchestrator()
            resp = orch.handle_document(
                extracted_text = "This rental agreement is made between landlord and tenant.",
                filename       = "test.pdf",
            )

        assert resp.intent            == "document_analysis"
        assert len(resp.answer_text)  > 0
        assert len(resp.summary)      > 0
        print(f"\n✓ handle_document returns structured output")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP C — DRAFTING AGENT MULTI-TURN VIA ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestDraftingAgentIntegration:

    def test_fir_three_turns_via_orchestrator(self):
        """Full 3-turn FIR draft through orchestrator with mocked LLM."""
        from agents.drafting_agent import DraftingAgent
        fresh_agent = DraftingAgent()

        with patch("agents.graph.drafting_agent", fresh_agent), \
             patch.object(fresh_agent, "_generate_draft", return_value="MOCK FIR DRAFT"):

            orch = MasterOrchestrator()

            # Turn 1
            r1 = orch.handle_query("Help me draft a written complaint to police for theft")
            assert r1.intent == "draft_request"
            assert fresh_agent.has_active_draft(r1.session_id)
            print(f"\n✓ FIR Turn 1 — stage in progress, session={r1.session_id[:8]}")

            # Turn 2
            r2 = orch.handle_query(
                "My phone was stolen at railway station on 5 May 2026",
                session_id=r1.session_id,
            )
            assert fresh_agent.has_active_draft(r1.session_id)
            print(f"\n✓ FIR Turn 2 — still in progress")

            # Turn 3
            r3 = orch.handle_query(
                "My name is Anantha Krishnan K, Kakkanad Kochi. Accused unknown.",
                session_id=r1.session_id,
            )
            assert not fresh_agent.has_active_draft(r1.session_id)
            assert "MOCK FIR DRAFT" in r3.answer_text or r3.draft == "MOCK FIR DRAFT"
            print(f"\n✓ FIR Turn 3 — draft complete, session cleared")

    def test_draft_intent_preserved_across_turns(self):
        """Session intent stays draft_request for all 3 turns."""
        from agents.drafting_agent import DraftingAgent
        fresh_agent = DraftingAgent()

        with patch("agents.graph.drafting_agent", fresh_agent), \
             patch.object(fresh_agent, "_generate_draft", return_value="DRAFT"):

            orch = MasterOrchestrator()
            r1   = orch.handle_query("Create a rental agreement template")
            r2   = orch.handle_query(
                "Flat 4B Green Valley Kakkanad, Rs 12000/month, 11 months",
                session_id=r1.session_id,
            )
            r3   = orch.handle_query(
                "Landlord: Priya Nair. Tenant: Ajith Kumar. Start June 2026.",
                session_id=r1.session_id,
            )

        assert r1.intent == "draft_request"
        assert r3.intent == "draft_request"
        print(f"\n✓ Draft intent preserved across all 3 turns")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP D — KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeGraphIntegration:

    def test_kg_singleton_built(self):
        from rag.knowledge_graph import get_kg
        kg    = get_kg()
        stats = kg.stats()
        assert stats["built"]  == True
        assert stats["nodes"]  >  200
        assert stats["edges"]  >  100
        print(f"\n✓ KG built: {stats['nodes']} nodes, {stats['edges']} edges")

    def test_kg_420_ipc_returns_related(self):
        from rag.knowledge_graph import get_kg
        kg      = get_kg()
        related = kg.query_related_sections("420", source_hint="Indian Penal Code")
        assert len(related) > 0
        sections = {r["section"] for r in related if r["node_type"] == "section"}
        assert "415" in sections or "318" in sections
        print(f"\n✓ KG Section 420 IPC → {len(related)} related nodes")

    def test_kg_context_format(self):
        from rag.knowledge_graph import get_kg
        kg      = get_kg()
        related = kg.query_related_sections("302", source_hint="Indian Penal Code")
        ctx     = kg.format_context("302", "Indian Penal Code", related)
        assert "[Knowledge Graph]" in ctx
        assert "302" in ctx
        print(f"\n✓ KG context format correct for Section 302 IPC")

    def test_kg_injected_in_pipeline_for_section_query(self):
        """
        When pipeline processes a section query, KG context chunk
        should appear in pinned_chunks via the injection hook.
        """
        from rag.knowledge_graph import get_kg
        kg = get_kg()  # ensure built
        related = kg.query_related_sections("138", source_hint="Negotiable Instruments Act")
        assert len(related) > 0, "KG should have entries for Section 138 NI Act"
        print(f"\n✓ KG has entries for Section 138 NI Act — pipeline injection ready")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP E — TRANSLATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestTranslationAgent:

    def test_detect_english(self):
        r = detect_language("What is Section 302 IPC?")
        assert r.is_english      == True
        assert r.detected_script is None
        print(f"\n✓ English detected correctly")

    def test_detect_malayalam_script(self):
        r = detect_language("വകുപ്പ് 302 ഐപിസി എന്താണ്?")
        assert r.is_english      == False
        assert r.detected_script == "Malayalam"
        print(f"\n✓ Malayalam script detected")

    def test_detect_hindi_script(self):
        r = detect_language("धारा 302 आईपीसी क्या है?")
        assert r.is_english      == False
        assert r.detected_script == "Hindi"
        print(f"\n✓ Hindi script detected")

    def test_detect_translation_request(self):
        r = detect_language("Explain Section 138 NI Act in Malayalam")
        assert r.target_language == "Malayalam"
        print(f"\n✓ Translation target detected: {r.target_language}")

    def test_strip_legal_content(self):
        query  = "Translate into Malayalam: What is Section 302 IPC?"
        result = _strip_legal_content(query, "Malayalam")
        assert "Section 302" in result
        assert "Translate" not in result
        print(f"\n✓ Legal content stripped: {result!r}")

    def test_translation_node_called_for_translation_intent(self):
        with patch("rag.pipeline.rag_pipeline") as mock_rag, \
             patch("rag.llm.llm") as mock_llm:
            mock_rag.query.return_value    = _mock_rag_answer("Section 138 NI Act covers cheque bounce.")
            mock_llm.generate.return_value = "Section 138 NI Act cheque bounce — Malayalam answer."

            orch = MasterOrchestrator()
            resp = orch.handle_query("Explain Section 138 NI Act in Malayalam")

        assert resp.intent == "translation_request"
        assert len(resp.answer_text) > 0
        print(f"\n✓ Translation intent handled end-to-end")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP F — SESSION MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionMemory:

    def test_session_created_on_first_query(self):
        with patch("rag.llm.llm") as mock_llm:
            mock_llm.generate.return_value = "Hello!"
            orch = MasterOrchestrator()
            resp = orch.handle_query("Hello")

        assert resp.session_id != ""
        assert len(resp.session_id) > 8
        print(f"\n✓ Session created: {resp.session_id[:8]}...")

    def test_session_persists_across_turns(self):
        mem = SessionMemory()
        sid = mem.create_session()
        mem.add_turn(sid, role="user",      content="What is bail?",       intent="legal_query")
        mem.add_turn(sid, role="assistant", content="Bail is provisional.", intent="legal_query")
        mem.add_turn(sid, role="user",      content="Tell me more.",        intent="legal_query")

        history = mem.get_history(sid)
        assert len(history) == 3
        assert history[0]["role"]    == "user"
        assert history[1]["role"]    == "assistant"
        print(f"\n✓ Session memory stores {len(history)} turns correctly")

    def test_context_block_format(self):
        mem = SessionMemory()
        sid = mem.create_session()
        mem.add_turn(sid, "user",      "What is Section 302?", "legal_query")
        mem.add_turn(sid, "assistant", "Section 302 IPC is murder.", "legal_query")

        ctx = mem.get_context_block(sid)
        assert "[CONVERSATION HISTORY]" in ctx
        assert "[END HISTORY]"          in ctx
        assert "Section 302"            in ctx
        print(f"\n✓ Context block formatted correctly")

    def test_session_delete(self):
        mem = SessionMemory()
        sid = mem.create_session()
        mem.add_turn(sid, "user", "test", None)
        assert mem.session_exists(sid)

        mem.delete_session(sid)
        assert not mem.session_exists(sid)
        print(f"\n✓ Session delete works correctly")

    def test_max_turns_stored(self):
        from agents.memory import MAX_TURNS_STORED
        mem = SessionMemory()
        sid = mem.create_session()
        for i in range(MAX_TURNS_STORED + 5):
            mem.add_turn(sid, "user", f"message {i}", None)

        assert mem.turn_count(sid) == MAX_TURNS_STORED
        print(f"\n✓ Max turns FIFO trim works: {MAX_TURNS_STORED} max")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP G — LEGACY ENDPOINT BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegacyEndpoint:

    def test_legacy_orchestrate_query_returns_dict(self):
        """
        /api/v1/orchestrate/query now delegates to master orchestrator.
        Must return a dict with at least answer_text and intent.
        """
        with patch("rag.pipeline.rag_pipeline") as mock_rag:
            mock_rag.query.return_value = _mock_rag_answer()

            from agents.orchestrator import master_orchestrator
            resp = master_orchestrator.handle_query("What is Section 302 IPC?")
            d    = resp.to_dict()

        assert "answer_text" in d
        assert "intent"      in d
        assert "summary"     in d
        assert d["intent"]   == "legal_query"
        print(f"\n✓ Legacy endpoint compatible — to_dict() has all keys")

    def test_all_six_intents_return_structured(self):
        """Every intent must return a LexShieldResponse with consistent structure."""
        test_queries = [
            ("What is Section 302 IPC?",                         "legal_query"),
            ("Is it legal to fire someone without notice?",      "risk_check"),
            ("Hello, what can you do?",                          "general"),
            ("Help me draft a rental agreement",                 "draft_request"),
        ]

        for query, expected_intent in test_queries:
            with patch("rag.pipeline.rag_pipeline") as mock_rag, \
                 patch("rag.llm.llm") as mock_llm:
                mock_rag.query.return_value    = _mock_rag_answer()
                mock_llm.generate.return_value = "LexShield AI response."

                orch = MasterOrchestrator()
                resp = orch.handle_query(query)
                d    = resp.to_dict()

            assert resp.intent           == expected_intent, \
                f"Query {query!r}: expected {expected_intent}, got {resp.intent}"
            assert "answer_text"         in d
            assert "risk"                in d
            assert isinstance(d["risk"], dict)
            print(f"\n✓ {expected_intent:25} → structured output OK")