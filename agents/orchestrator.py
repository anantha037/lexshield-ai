"""
LexShield AI — Master Orchestrator
=====================================
Thin wrapper around the LangGraph agent_graph.
Passes thread_id=session_id in config so SqliteSaver checkpointer
stores and restores state across server restarts.

State field mapping (agents/graph.py AgentState → here)
--------------------------------------------------------
state["response"]    — final answer text  (was state["result"]["answer"])
state["rag_result"]  — RAG pipeline output dict  (was state["result"])
state["risk_result"] — risk scorer output  {score, level, factors}
state["ner_result"]  — NER output  {entities: [...]}
state["draft_stage"] — current DraftingAgent stage
state["draft_data"]  — accumulated draft fields
"""

from typing import Optional

from agents.memory         import session_memory
from agents.graph          import agent_graph, AgentState
from rag.structured_output import build_structured_response, LexShieldResponse


class MasterOrchestrator:

    # ── Query flow ─────────────────────────────────────────────────────────────

    def handle_query(
        self,
        query:      str,
        session_id: Optional[str] = None,
    ) -> LexShieldResponse:
        session_id = session_memory.ensure_session(session_id)

        # Inject conversation history into initial state via rag_result
        context_block = session_memory.get_context_block(session_id)

        # Record user turn BEFORE invoking (so context includes prior turns)
        session_memory.add_turn(session_id, role="user", content=query, intent=None)

        # Build initial state — nodes update their own fields
        initial_state: AgentState = {
            "query":           query,
            "session_id":      session_id,
            "intent":          "",
            "confidence":      0.0,
            "rag_result":      {"context_block": context_block},
            "ner_result":      {},
            "risk_result":     {},
            "draft_stage":     0,
            "draft_data":      {},
            "pipeline_depth":  0,
            "rag_grade":       "",
            "source_language": "",
            "response":        "",
            "error":           "",
        }

        # LangGraph config — thread_id links this invocation to its checkpoint
        config = {"configurable": {"thread_id": session_id}}

        print(f"[Orchestrator] graph.invoke — query={query[:60]!r} "
              f"session={session_id[:8]}…")
        final_state = agent_graph.invoke(initial_state, config)

        # Extract outputs from named state fields
        intent     = final_state.get("intent",     "general")
        confidence = final_state.get("confidence", 0.0)
        response   = final_state.get("response",   "")
        rag_result = final_state.get("rag_result", {})
        risk_result = final_state.get("risk_result", {})

        answer = response or rag_result.get("answer", "") or \
                 "I was unable to process your request. Please try again."

        # Pull draft text out of rag_result (set by draft_node)
        draft = rag_result.get("draft", "")

        # Pull structured case law results (set by case_law_node)
        case_law_raw = final_state.get("case_law_result", {})
        case_law_results = []
        if case_law_raw and case_law_raw.get("results"):
            for item in case_law_raw["results"]:
                c = item.get("case", {})
                case_law_results.append({
                    "title":    c.get("title", ""),
                    "court":    c.get("court", ""),
                    "date":     c.get("date", ""),
                    "citation": c.get("citation", ""),
                    "headline": c.get("headline", ""),
                    "url":      c.get("url", ""),
                    "summary":  item.get("summary", ""),
                })

        # Record assistant turn
        session_memory.add_turn(
            session_id, role="assistant", content=answer, intent=intent
        )

        return build_structured_response(
            answer_text       = answer,
            intent            = intent,
            session_id        = session_id,
            confidence        = confidence,
            mode              = rag_result.get("mode", ""),
            citations         = [],
            draft             = draft,
            sources_consulted = rag_result.get("sources_consulted", 0),
            synthesis_note    = rag_result.get("synthesis_note",    ""),
            grounding_warning = rag_result.get("grounding_warning", ""),
            rewritten_queries = rag_result.get("rewritten_queries", []),
            reranker_used     = rag_result.get("reranker_used",     False),
            case_law_results  = case_law_results,
        )

    # ── Document flow ──────────────────────────────────────────────────────────

    def handle_document(
        self,
        extracted_text: str,
        session_id:     Optional[str] = None,
        filename:       Optional[str] = None,
    ) -> LexShieldResponse:
        session_id = session_memory.ensure_session(session_id)

        context_block = session_memory.get_context_block(session_id)
        label = f" (file: {filename})" if filename else ""
        query = f"Analyze and summarize this legal document{label}:\n\n{extracted_text[:3000]}"

        session_memory.add_turn(
            session_id, role="user",
            content=f"[Document analysis{label}]",
            intent="document_analysis",
        )

        initial_state: AgentState = {
            "query":           query,
            "session_id":      session_id,
            "intent":          "document_analysis",   # pre-set — skip classifier
            "confidence":      1.0,
            "rag_result":      {"context_block": context_block},
            "ner_result":      {},
            "risk_result":     {},
            "draft_stage":     0,
            "draft_data":      {},
            "pipeline_depth":  0,
            "rag_grade":       "",
            "source_language": "",
            "response":        "",
            "error":           "",
        }

        config = {"configurable": {"thread_id": session_id}}

        print(f"[Orchestrator] Document analysis via graph{label}")
        final_state = agent_graph.invoke(initial_state, config)

        response   = final_state.get("response",   "")
        rag_result = final_state.get("rag_result", {})

        answer = response or rag_result.get("answer", "") or \
                 "I was unable to process the document."

        session_memory.add_turn(
            session_id, role="assistant", content=answer, intent="document_analysis"
        )

        return build_structured_response(
            answer_text       = answer,
            intent            = "document_analysis",
            session_id        = session_id,
            confidence        = 1.0,
            mode              = rag_result.get("mode", ""),
            citations         = [],
            draft             = "",
            sources_consulted = rag_result.get("sources_consulted", 0),
            synthesis_note    = rag_result.get("synthesis_note",    ""),
            grounding_warning = rag_result.get("grounding_warning", ""),
            rewritten_queries = rag_result.get("rewritten_queries", []),
            reranker_used     = rag_result.get("reranker_used",     False),
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
master_orchestrator = MasterOrchestrator()