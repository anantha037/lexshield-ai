"""
LexShield AI — Master Orchestrator
=====================================
Routes user input by intent to the correct agent/pipeline.

Routing table:
  legal_query         → RAGPipeline.query()
  document_analysis   → RAGPipeline.query() on extracted text
  draft_request       → DraftingAgent stub (Day 2-3)
  risk_check          → RAGPipeline.query() with risk prompt prefix
  translation_request → MultilingualAgent stub (Day 4-5)
  general             → Direct LLM call (no RAG)

Memory:
  Every user+assistant turn is stored in SessionMemory.
  Last 5 turns injected as context block into every prompt.
"""

import os
from dataclasses import dataclass
from typing import Optional

from agents.intent_classifier import intent_classifier, IntentResult
from agents.memory import session_memory


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorResult:
    session_id: str
    intent: str
    confidence: float
    answer: str
    sources_consulted: int = 0
    synthesis_note: str = ""
    grounding_warning: str = ""
    rewritten_queries: list = None
    reranker_used: bool = False
    mode: str = ""                  # which agent/path handled this

    def __post_init__(self):
        if self.rewritten_queries is None:
            self.rewritten_queries = []


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MasterOrchestrator:

    # ── RAG lazy import (avoids heavy startup cost when not used) ──────────────
    @staticmethod
    def _get_rag():
        from rag.pipeline import rag_pipeline
        return rag_pipeline

    @staticmethod
    def _get_llm():
        from rag.llm import llm
        return llm

    # ── Main entry ─────────────────────────────────────────────────────────────

    def handle_query(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        Main entry point for text queries.

        Args:
            query:      raw user text
            session_id: existing session ID (creates new if None/invalid)

        Returns:
            OrchestratorResult
        """
        session_id = session_memory.ensure_session(session_id)

        # Classify intent
        intent_result: IntentResult = intent_classifier.classify(query)
        intent = intent_result.intent

        # Fetch conversation context
        context_block = session_memory.get_context_block(session_id)

        # Store user turn
        session_memory.add_turn(session_id, role="user", content=query, intent=intent)

        # Route
        result = self._route(query, intent, context_block, session_id)

        # Store assistant turn
        session_memory.add_turn(session_id, role="assistant", content=result.answer, intent=intent)

        # Attach session/intent metadata
        result.session_id  = session_id
        result.intent      = intent
        result.confidence  = intent_result.confidence

        return result

    def handle_document(
        self,
        extracted_text: str,
        session_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        Entry point for pre-extracted document text.
        Routes as document_analysis intent.

        Args:
            extracted_text: OCR/PDF extracted text
            session_id:     existing session ID
            filename:       original filename for context (optional)
        """
        session_id = session_memory.ensure_session(session_id)

        label = f" (file: {filename})" if filename else ""
        query = f"Analyze and summarize this legal document{label}:\n\n{extracted_text[:3000]}"

        context_block = session_memory.get_context_block(session_id)
        session_memory.add_turn(session_id, role="user", content=f"[Document analysis{label}]", intent="document_analysis")

        result = self._handle_document_analysis(query, context_block)

        session_memory.add_turn(session_id, role="assistant", content=result.answer, intent="document_analysis")

        result.session_id = session_id
        result.intent     = "document_analysis"
        result.confidence = 1.0

        return result

    # ── Router ─────────────────────────────────────────────────────────────────

    def _route(
        self,
        query: str,
        intent: str,
        context_block: str,
        session_id: str,
    ) -> OrchestratorResult:

        if intent == "legal_query":
            return self._handle_legal_query(query, context_block)

        elif intent == "document_analysis":
            return self._handle_document_analysis(query, context_block)

        elif intent == "draft_request":
            return self._handle_draft_request(query, context_block)

        elif intent == "risk_check":
            return self._handle_risk_check(query, context_block)

        elif intent == "translation_request":
            return self._handle_translation_request(query, context_block)

        else:  # general
            return self._handle_general(query, context_block)

    # ── Handlers ───────────────────────────────────────────────────────────────

    def _handle_legal_query(self, query: str, context_block: str) -> OrchestratorResult:
        """Route to RAGPipeline. Inject conversation context into query."""
        rag = self._get_rag()
        enriched_query = f"{context_block}\n\n{query}" if context_block else query
        legal_answer = rag.query(enriched_query)

        return OrchestratorResult(
            session_id="",
            intent="legal_query",
            confidence=0.0,
            answer=legal_answer.answer_text,
            sources_consulted=legal_answer.sources_consulted,
            synthesis_note=legal_answer.synthesis_note or "",
            grounding_warning=legal_answer.grounding_warning or "",
            rewritten_queries=legal_answer.rewritten_queries or [],
            reranker_used=legal_answer.reranker_used,
            mode="rag_pipeline",
        )

    def _handle_document_analysis(self, query: str, context_block: str) -> OrchestratorResult:
        """Run document text through RAG pipeline for analysis."""
        rag = self._get_rag()
        enriched_query = f"{context_block}\n\n{query}" if context_block else query
        legal_answer = rag.query(enriched_query)

        return OrchestratorResult(
            session_id="",
            intent="document_analysis",
            confidence=0.0,
            answer=legal_answer.answer_text,
            sources_consulted=legal_answer.sources_consulted,
            synthesis_note=legal_answer.synthesis_note or "",
            grounding_warning=legal_answer.grounding_warning or "",
            rewritten_queries=legal_answer.rewritten_queries or [],
            reranker_used=legal_answer.reranker_used,
            mode="rag_pipeline_document",
        )

    def _handle_risk_check(self, query: str, context_block: str) -> OrchestratorResult:
        """Add risk assessment prefix and route through RAG."""
        rag = self._get_rag()
        risk_query = (
            f"Provide a detailed legal risk assessment for the following. "
            f"Identify applicable Indian laws, potential penalties, liabilities, "
            f"and legal consequences:\n\n{query}"
        )
        enriched_query = f"{context_block}\n\n{risk_query}" if context_block else risk_query
        legal_answer = rag.query(enriched_query)

        return OrchestratorResult(
            session_id="",
            intent="risk_check",
            confidence=0.0,
            answer=legal_answer.answer_text,
            sources_consulted=legal_answer.sources_consulted,
            synthesis_note=legal_answer.synthesis_note or "",
            grounding_warning=legal_answer.grounding_warning or "",
            rewritten_queries=legal_answer.rewritten_queries or [],
            reranker_used=legal_answer.reranker_used,
            mode="rag_pipeline_risk",
        )

    def _handle_draft_request(self, query: str, context_block: str) -> OrchestratorResult:
        """DraftingAgent stub — implemented Week 3 Day 2-3."""
        return OrchestratorResult(
            session_id="",
            intent="draft_request",
            confidence=0.0,
            answer=(
                "The drafting agent is being built and will be available shortly. "
                "Please describe what document you need (e.g. FIR, legal notice, "
                "rental agreement) and I will draft it for you once ready."
            ),
            mode="draft_stub",
        )

    def _handle_translation_request(self, query: str, context_block: str) -> OrchestratorResult:
        """MultilingualAgent stub — implemented Week 3 Day 4-5."""
        return OrchestratorResult(
            session_id="",
            intent="translation_request",
            confidence=0.0,
            answer=(
                "The multilingual translation agent is being built and will be "
                "available shortly. It will support Malayalam, Hindi, Tamil, "
                "Telugu, Kannada and other Indian languages."
            ),
            mode="translation_stub",
        )

    def _handle_general(self, query: str, context_block: str) -> OrchestratorResult:
        """Direct LLM call — no RAG, no retrieval."""
        llm = self._get_llm()

        system_prompt = (
            "You are LexShield AI, an Indian legal intelligence assistant. "
            "Help users understand Indian law, their legal rights, and legal documents. "
            "Be concise, friendly, and direct. "
            "If asked about specific legal questions, encourage the user to ask them directly."
        )

        prompt = f"{context_block}\n\nUser: {query}" if context_block else f"User: {query}"
        answer = llm.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=512)

        return OrchestratorResult(
            session_id="",
            intent="general",
            confidence=0.0,
            answer=answer,
            mode="direct_llm",
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
master_orchestrator = MasterOrchestrator()