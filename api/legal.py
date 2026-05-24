"""
LexShield AI — Legal Query API
=======================================================
Changes in this version:
  - category field validator added: converts "" -> None before Pydantic
    Literal validation runs. Fixes 422 error when client sends category="".
  - era field added to CitationResponse and LegalQueryResponse.
  - Everything else unchanged.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

router = APIRouter(prefix="/api/v1/legal", tags=["legal"])

LegalCategory = Literal[
    "criminal", "family", "corporate", "taxation", "property",
    "labour", "health", "environment", "technology", "civil",
]

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
    category: Optional[LegalCategory] = Field(
        default=None,
        description=(
            "Restrict retrieval to a legal category. Omit (or pass null) to "
            "search the entire corpus. Sending an empty string is treated as null. "
            "Valid values: criminal | family | corporate | taxation | property | "
            "labour | health | environment | technology | civil"
        ),
    )

    # ── FIX: convert empty string -> None before Literal validation ────────────
    # When a client sends { "category": "" } Pydantic's Literal check raises a
    # 422 because "" is not one of the allowed values. This validator runs first
    # (mode="before") and normalises "" to None so the field is treated as
    # "no filter" rather than an invalid value.
    @field_validator("category", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: object) -> object:
        if v == "" or v is None:
            return None
        return v


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
    category:         str
    era:              str = ""    # "legacy" | "current" | ""


class LegalQueryResponse(BaseModel):
    query:             str
    answer:            str
    citations:         list[CitationResponse]
    sources_consulted: int
    synthesis_note:    str
    grounding_warning: Optional[str]
    rewritten_queries: list[str]
    reranker_used:     bool
    category_filter:   Optional[str]


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=LegalQueryResponse)
async def legal_query(request: LegalQueryRequest):
    """
    Answer a legal question using the full advanced RAG pipeline.

    Pipeline steps:
      1. Abbreviation expansion (40+ mappings)
      2. LLM query rewriting -> angle-diverse search queries
      3. Hybrid vector+BM25 search (with optional category filter)
      4. Section metadata fast-path — pins exact section chunks to top
      5. Paired act retrieval (IPC↔BNS, CrPC↔BNSS, Evidence↔BSA)
      6. NVIDIA NIM reranking
      7. Multi-document synthesis with [SOURCE N] citations
      8. Grounding / hallucination check

    Category filter (optional):
      Restrict to a legal domain to reduce cross-category noise.
      Pass null or omit entirely to search the full corpus.
      Example: { "query": "cheque bounce", "category": "corporate" }

    Note: Do NOT pass category="criminal" for IPC/BNS queries unless you
    want to restrict to criminal acts only. Omit category for paired act
    (old law + new law) responses to work correctly.
    """
    from rag.pipeline import rag_pipeline

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag_pipeline.enable_rewriting = (
        request.enable_rewriting if request.enable_rewriting is not None else True
    )
    rag_pipeline.enable_reranking = (
        request.enable_reranking if request.enable_reranking is not None else True
    )

    result = rag_pipeline.query(
        user_query      = query,
        n_results       = request.n_results,
        category_filter = request.category,   # None = full corpus
    )

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
            category         = getattr(c, "category", ""),
            era              = getattr(c, "era",      ""),
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
        category_filter    = request.category,
    )