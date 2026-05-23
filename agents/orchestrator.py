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
import json
import time
from datetime import datetime
import os
import logging
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from agents.memory         import session_memory, profile_memory
from agents.graph          import agent_graph, AgentState
from rag.structured_output import build_structured_response, LexShieldResponse

logger = logging.getLogger(__name__)


class MasterOrchestrator:

    # ── Query flow ─────────────────────────────────────────────────────────────

    @traceable(name="master_orchestrator.handle_query", run_type="chain")
    def handle_query(
        self,
        query:      str,
        session_id: Optional[str] = None,
    ) -> LexShieldResponse:
        start_time = time.time()
        session_id = session_memory.ensure_session(session_id)

        # Inject conversation history into initial state via rag_result
        context_block = session_memory.get_relevant_context(session_id, query)
        summary = session_memory.get_summary(session_id)
        profile_block = profile_memory.get_profile_block(session_id)
        
        if summary:
            if context_block:
                context_block = f"[CONVERSATION SUMMARY]\n{summary}\n\n{context_block}"
            else:
                context_block = f"[CONVERSATION SUMMARY]\n{summary}\n"
                
        if profile_block:
            if context_block:
                context_block = f"{profile_block}\n\n{context_block}"
            else:
                context_block = f"{profile_block}\n"

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
            "scratchpad":      {},
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

        # Derive citation_status
        citations = rag_result.get("citations", [])
        grounding_warning = rag_result.get("grounding_warning", "")
        
        if citations or case_law_results:
            if grounding_warning:
                citation_status = "partial"
            else:
                citation_status = "cited"
        else:
            citation_status = "unverified"

        # Extract scope and validation status
        scope_status  = final_state.get("scope_status", "in_scope")
        scope_message = final_state.get("scope_message", None)
        validation_status = final_state.get("validation_status", "not_applicable")

        # Extract entities for profile update
        doc_types = []
        domains = []
        query_lower = query.lower()
        if intent in ["draft_request", "document_analysis"]:
            for dt in ["FIR", "Legal Notice", "Rental Agreement", "Contract", "Bail Application", "Petition", "Affidavit", "Complaint"]:
                if dt.lower() in query_lower:
                    doc_types.append(dt)
                    
        if any(w in query_lower for w in ["divorce", "maintenance", "family", "dowry", "domestic violence"]):
            domains.append("Family Law")
        elif any(w in query_lower for w in ["property", "tenant", "eviction", "landlord", "rent"]):
            domains.append("Civil/Property Law")
        elif any(w in query_lower for w in ["bail", "fir", "arrest", "police", "crime", "ipc", "criminal", "murder", "theft"]):
            domains.append("Criminal Law")
        elif any(w in query_lower for w in ["consumer", "product", "refund", "defective"]):
            domains.append("Consumer Law")
            
        entities_for_profile = dict(final_state.get("ner_result", {}))
        entities_for_profile["doc_types"] = doc_types
        entities_for_profile["domains"] = domains
        
        profile_memory.update_profile(session_id, intent, entities_for_profile)

        # Record assistant turn
        session_memory.add_turn(
            session_id, role="assistant", content=answer, intent=intent
        )

        # Attach metadata to LangSmith trace
        rt = get_current_run_tree()
        if rt:
            rt.add_metadata({
                "session_id": session_id,
                "intent": intent,
                "citation_status": citation_status,
                "scope_status": scope_status
            })

        latency_ms = int((time.time() - start_time) * 1000)
        
        # Local metrics logging fallback
        try:
            os.makedirs("logs", exist_ok=True)
            
            # Extract crag_score directly from rag_result
            crag_score = rag_result.get("crag_score", 0)
            if not crag_score:
                note = rag_result.get("synthesis_note", "")
                if "score=" in note:
                    import re
                    m = re.search(r"score=(\d+)", note)
                    if m:
                        crag_score = int(m.group(1))
                elif "complexity=simple" in note:
                    # Simple complexity queries bypass CRAG scoring and proceed directly
                    # to synthesis. We default to 4 (good retrieval) as simple queries
                    # use the section fast-path which has high precision by design.
                    crag_score = 4

            metric = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "session_id": session_id,
                "intent": intent,
                "latency_ms": latency_ms,
                "citation_status": citation_status,
                "scope_status": scope_status,
                "validation_status": validation_status,
                "crag_score": crag_score,
                "chunks_retrieved": rag_result.get("sources_consulted", 0),
                "model_used": "groq-default"
            }
            with open(os.path.join("logs", "query_metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(metric) + "\n")
        except Exception as e:
            logger.warning(f"[Orchestrator] Failed to write local metric log: {e}")

        return build_structured_response(
            answer_text       = answer,
            intent            = intent,
            session_id        = session_id,
            confidence        = confidence,
            mode              = rag_result.get("mode", ""),
            citation_status   = citation_status,
            validation_status = validation_status,
            scope_status      = scope_status,
            scope_message     = scope_message,
            citations         = citations,
            draft             = draft,
            sources_consulted = rag_result.get("sources_consulted", 0),
            synthesis_note    = rag_result.get("synthesis_note",    ""),
            grounding_warning = grounding_warning,
            rewritten_queries = rag_result.get("rewritten_queries", []),
            reranker_used     = rag_result.get("reranker_used",     False),
            case_law_results  = case_law_results,
            debug_scratchpad  = final_state.get("scratchpad", {}),
        )

    # ── Document flow ──────────────────────────────────────────────────────────

    @traceable(name="master_orchestrator.handle_document", run_type="chain")
    def handle_document(
        self,
        extracted_text: str,
        session_id:     Optional[str] = None,
        filename:       Optional[str] = None,
    ) -> LexShieldResponse:
        session_id = session_memory.ensure_session(session_id)

        # Use BM25-scored context if extracted text is meaningful (≥20 chars)
        doc_snippet = extracted_text[:500].strip()
        if len(doc_snippet) >= 20:
            context_block = session_memory.get_relevant_context(session_id, doc_snippet)
        else:
            context_block = session_memory.get_context_block(session_id)
        summary = session_memory.get_summary(session_id)
        profile_block = profile_memory.get_profile_block(session_id)
        
        if summary:
            if context_block:
                context_block = f"[CONVERSATION SUMMARY]\n{summary}\n\n{context_block}"
            else:
                context_block = f"[CONVERSATION SUMMARY]\n{summary}\n"
                
        if profile_block:
            if context_block:
                context_block = f"{profile_block}\n\n{context_block}"
            else:
                context_block = f"{profile_block}\n"

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
            "scratchpad":      {},
        }

        config = {"configurable": {"thread_id": session_id}}

        print(f"[Orchestrator] Document analysis via graph{label}")
        final_state = agent_graph.invoke(initial_state, config)

        response   = final_state.get("response",   "")
        rag_result = final_state.get("rag_result", {})
        scope_status  = final_state.get("scope_status", "in_scope")
        scope_message = final_state.get("scope_message", None)

        answer = response or rag_result.get("answer", "") or \
                 "I was unable to process the document."

        # Extract entities for profile update
        doc_types = []
        domains = []
        query_lower = query.lower()
        if "document_analysis" in ["draft_request", "document_analysis"]:
            for dt in ["FIR", "Legal Notice", "Rental Agreement", "Contract", "Bail Application", "Petition", "Affidavit", "Complaint"]:
                if dt.lower() in query_lower:
                    doc_types.append(dt)
                    
        if any(w in query_lower for w in ["divorce", "maintenance", "family", "dowry", "domestic violence"]):
            domains.append("Family Law")
        elif any(w in query_lower for w in ["property", "tenant", "eviction", "landlord", "rent"]):
            domains.append("Civil/Property Law")
        elif any(w in query_lower for w in ["bail", "fir", "arrest", "police", "crime", "ipc", "criminal", "murder", "theft"]):
            domains.append("Criminal Law")
        elif any(w in query_lower for w in ["consumer", "product", "refund", "defective"]):
            domains.append("Consumer Law")
            
        entities_for_profile = dict(final_state.get("ner_result", {}))
        entities_for_profile["doc_types"] = doc_types
        entities_for_profile["domains"] = domains
        
        profile_memory.update_profile(session_id, "document_analysis", entities_for_profile)

        session_memory.add_turn(
            session_id, role="assistant", content=answer, intent="document_analysis"
        )

        return build_structured_response(
            answer_text       = answer,
            intent            = "document_analysis",
            session_id        = session_id,
            confidence        = 1.0,
            mode              = rag_result.get("mode", ""),
            citation_status   = "unverified",
            validation_status = "not_applicable",
            scope_status      = scope_status,
            scope_message     = scope_message,
            citations         = [],
            draft             = "",
            sources_consulted = rag_result.get("sources_consulted", 0),
            synthesis_note    = rag_result.get("synthesis_note",    ""),
            grounding_warning = rag_result.get("grounding_warning", ""),
            rewritten_queries = rag_result.get("rewritten_queries", []),
            reranker_used     = rag_result.get("reranker_used",     False),
            debug_scratchpad  = final_state.get("scratchpad", {}),
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
master_orchestrator = MasterOrchestrator()