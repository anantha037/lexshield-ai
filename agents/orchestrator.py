"""
LexShield AI — Master Orchestrator
=====================================
Routes user input through the LangGraph multi-agent graph.
Wraps every response in LexShieldResponse (structured output).
"""

from typing import Optional

from agents.memory         import session_memory
from agents.graph          import agent_graph, AgentState
from rag.structured_output import build_structured_response, LexShieldResponse


class MasterOrchestrator:

    def handle_query(
        self,
        query:      str,
        session_id: Optional[str] = None,
    ) -> LexShieldResponse:
        session_id    = session_memory.ensure_session(session_id)
        context_block = session_memory.get_context_block(session_id)

        session_memory.add_turn(session_id, role="user", content=query, intent=None)

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

        print(f"[Orchestrator] Invoking graph: {query[:60]!r}")
        final_state = agent_graph.invoke(initial_state)

        intent     = final_state.get("intent", "general")
        confidence = final_state.get("confidence", 0.0)
        result     = final_state.get("result", {})
        draft      = final_state.get("draft", "")

        answer = result.get("answer", "I was unable to process your request. Please try again.")

        session_memory.add_turn(session_id, role="assistant", content=answer, intent=intent)

        return build_structured_response(
            answer_text       = answer,
            intent            = intent,
            session_id        = session_id,
            confidence        = confidence,
            mode              = result.get("mode", ""),
            citations         = [],
            draft             = draft,
            sources_consulted = result.get("sources_consulted", 0),
            synthesis_note    = result.get("synthesis_note", ""),
            grounding_warning = result.get("grounding_warning", ""),
            rewritten_queries = result.get("rewritten_queries", []),
            reranker_used     = result.get("reranker_used", False),
        )

    def handle_document(
        self,
        extracted_text: str,
        session_id:     Optional[str] = None,
        filename:       Optional[str] = None,
    ) -> LexShieldResponse:
        session_id    = session_memory.ensure_session(session_id)
        context_block = session_memory.get_context_block(session_id)

        label = f" (file: {filename})" if filename else ""
        query = f"Analyze and summarize this legal document{label}:\n\n{extracted_text[:3000]}"

        session_memory.add_turn(
            session_id, role="user",
            content=f"[Document analysis{label}]",
            intent="document_analysis",
        )

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
        answer = result.get("answer", "I was unable to process the document.")

        session_memory.add_turn(
            session_id, role="assistant", content=answer, intent="document_analysis"
        )

        return build_structured_response(
            answer_text       = answer,
            intent            = "document_analysis",
            session_id        = session_id,
            confidence        = 1.0,
            mode              = result.get("mode", ""),
            citations         = [],
            draft             = "",
            sources_consulted = result.get("sources_consulted", 0),
            synthesis_note    = result.get("synthesis_note", ""),
            grounding_warning = result.get("grounding_warning", ""),
            rewritten_queries = result.get("rewritten_queries", []),
            reranker_used     = result.get("reranker_used", False),
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
master_orchestrator = MasterOrchestrator()