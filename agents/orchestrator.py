"""
LexShield AI — Master Orchestrator
=====================================
Routes user input through the LangGraph multi-agent graph.

Graph handles:
  legal_query         → LegalRAGNode
  document_analysis   → DocumentNode
  draft_request       → DraftNode (Day 3)
  risk_check          → RiskNode
  translation_request → TranslationNode (Day 4-5)
  general             → GeneralNode (direct LLM)

Memory:
  Every user+assistant turn stored in SessionMemory.
  Last 5 turns injected as context block into AgentState.
"""

from dataclasses import dataclass
from typing import Optional

from agents.memory import session_memory
from agents.graph  import agent_graph, AgentState


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorResult:
    session_id:        str
    intent:            str
    confidence:        float
    answer:            str
    sources_consulted: int   = 0
    synthesis_note:    str   = ""
    grounding_warning: str   = ""
    rewritten_queries: list  = None
    reranker_used:     bool  = False
    mode:              str   = ""

    def __post_init__(self):
        if self.rewritten_queries is None:
            self.rewritten_queries = []


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MasterOrchestrator:

    def handle_query(
        self,
        query:      str,
        session_id: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        Main entry point for text queries.
        Builds AgentState, runs the LangGraph graph, returns result.

        Args:
            query:      raw user text
            session_id: existing session ID (creates new if None/invalid)

        Returns:
            OrchestratorResult
        """
        session_id    = session_memory.ensure_session(session_id)
        context_block = session_memory.get_context_block(session_id)

        # Store user turn before graph (intent filled in after)
        session_memory.add_turn(session_id, role="user", content=query, intent=None)

        # Build initial state
        initial_state: AgentState = {
            "query":      query,
            "intent":     "",
            "confidence": 0.0,
            "context":    context_block,
            "session_id": session_id,
            "result":     {},
            "draft":      "",
            "language":   "",
        }

        # Run graph
        print(f"[Orchestrator] Invoking graph for query: {query[:60]!r}")
        final_state = agent_graph.invoke(initial_state)

        # Extract results
        intent     = final_state.get("intent", "general")
        confidence = final_state.get("confidence", 0.0)
        result     = final_state.get("result", {})

        answer = result.get("answer", "I was unable to process your request. Please try again.")

        # Store assistant turn
        session_memory.add_turn(session_id, role="assistant", content=answer, intent=intent)

        return OrchestratorResult(
            session_id        = session_id,
            intent            = intent,
            confidence        = confidence,
            answer            = answer,
            sources_consulted = result.get("sources_consulted", 0),
            synthesis_note    = result.get("synthesis_note", ""),
            grounding_warning = result.get("grounding_warning", ""),
            rewritten_queries = result.get("rewritten_queries", []),
            reranker_used     = result.get("reranker_used", False),
            mode              = result.get("mode", ""),
        )

    def handle_document(
        self,
        extracted_text: str,
        session_id:     Optional[str] = None,
        filename:       Optional[str] = None,
    ) -> OrchestratorResult:
        """
        Entry point for pre-extracted document text.
        Forces document_analysis intent directly into AgentState.

        Args:
            extracted_text: OCR/PDF extracted text
            session_id:     existing session ID
            filename:       original filename for context (optional)
        """
        session_id    = session_memory.ensure_session(session_id)
        context_block = session_memory.get_context_block(session_id)

        label = f" (file: {filename})" if filename else ""
        query = f"Analyze and summarize this legal document{label}:\n\n{extracted_text[:3000]}"

        session_memory.add_turn(
            session_id, role="user",
            content=f"[Document analysis{label}]",
            intent="document_analysis",
        )

        # Force intent — skip classifier node for documents
        initial_state: AgentState = {
            "query":      query,
            "intent":     "document_analysis",
            "confidence": 1.0,
            "context":    context_block,
            "session_id": session_id,
            "result":     {},
            "draft":      "",
            "language":   "",
        }

        print(f"[Orchestrator] Document analysis via graph{label}")
        final_state = agent_graph.invoke(initial_state)

        result = final_state.get("result", {})
        answer = result.get("answer", "I was unable to process the document. Please try again.")

        session_memory.add_turn(session_id, role="assistant", content=answer, intent="document_analysis")

        return OrchestratorResult(
            session_id        = session_id,
            intent            = "document_analysis",
            confidence        = 1.0,
            answer            = answer,
            sources_consulted = result.get("sources_consulted", 0),
            synthesis_note    = result.get("synthesis_note", ""),
            grounding_warning = result.get("grounding_warning", ""),
            rewritten_queries = result.get("rewritten_queries", []),
            reranker_used     = result.get("reranker_used", False),
            mode              = result.get("mode", ""),
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
master_orchestrator = MasterOrchestrator()