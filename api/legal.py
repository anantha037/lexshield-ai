"""
LexShield AI — Legal Query API  (Week 2, Day 3 update)
=======================================================
POST /api/v1/legal/query

Request:
  { "query": "What is punishment for cheating under IPC?" }

Response (structured JSON):
  {
    "query": "...",
    "answer": "According to [1] Section 420 of the Indian Penal Code...",
    "citations": [
      {
        "source_number": 1,
        "source": "Indian Penal Code (IPC)",
        "section": "420",
        "section_title": "Cheating",
        "chapter": "CHAPTER XVII",
        "preview": "420. Cheating.—Whoever cheats...",
        "relevance_score": 0.9123,
        "retrieval_source": "both",
        "doc_type": "statute"
      }
    ],
    "sources_consulted": 3,
    "synthesis_note": "Synthesized from 3 sections across 2 sources (Statute)",
    "grounding_warning": null,
    "rewritten_queries": ["...", "...", "..."],
    "reranker_used": true
  }
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter(prefix="/api/v1/legal", tags=["legal"])


# ── Request model ─────────────────────────────────────────────────────────────

class LegalQueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=1000,
        description="The legal question to answer",
    )
    n_results: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="Number of sources to include in context (default: 5)",
    )
    enable_rewriting: Optional[bool] = Field(
        default=True,
        description="Set false to skip LLM query rewriting (faster)",
    )
    enable_reranking: Optional[bool] = Field(
        default=True,
        description="Set false to skip NVIDIA reranking (faster fallback)",
    )


# ── Response models ───────────────────────────────────────────────────────────

class CitationResponse(BaseModel):
    source_number:    int
    source:           str
    section:          str
    section_title:    str
    chapter:          str
    preview:          str
    relevance_score:  Optional[float]
    retrieval_source: str
    doc_type:         str


class LegalQueryResponse(BaseModel):
    query:             str
    answer:            str
    citations:         list[CitationResponse]
    sources_consulted: int
    synthesis_note:    str
    grounding_warning: Optional[str]
    rewritten_queries: list[str]
    reranker_used:     bool


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=LegalQueryResponse)
async def legal_query(request: LegalQueryRequest):
    """
    Answer a legal question using the full advanced RAG pipeline.

    Full pipeline on every request:
      1. Abbreviation expansion (IPC → Indian Penal Code)
      2. LLM query rewriting → 3 angle-diverse search queries
      3. Hybrid vector+BM25 search on all queries
      4. Deduplicate and merge results
      5. NVIDIA NIM reranking (top 10 → top 5)
      6. Multi-document synthesis with numbered [SOURCE N] citations
      7. Grounding / hallucination check
      8. Return structured JSON with full citation metadata
    """
    from rag.pipeline import rag_pipeline

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Apply per-request overrides
    rag_pipeline.enable_rewriting = (
        request.enable_rewriting if request.enable_rewriting is not None else True
    )
    rag_pipeline.enable_reranking = (
        request.enable_reranking if request.enable_reranking is not None else True
    )

    result = rag_pipeline.query(user_query=query, n_results=request.n_results)

    # Convert Citation dataclasses → CitationResponse pydantic models
    citation_responses = [
        CitationResponse(
            source_number    = c.source_number,
            source           = c.source,
            section          = c.section,
            section_title    = c.section_title,
            chapter          = c.chapter,
            preview          = c.preview,
            relevance_score  = c.relevance_score,
            retrieval_source = c.retrieval_source,
            doc_type         = c.doc_type,
        )
        for c in result.citations
    ]

    return LegalQueryResponse(
        query              = query,
        answer             = result.answer_text,
        citations          = citation_responses,
        sources_consulted  = result.sources_consulted,
        synthesis_note     = result.synthesis_note,
        grounding_warning  = result.grounding_warning,
        rewritten_queries  = result.rewritten_queries,
        reranker_used      = result.reranker_used,
    )