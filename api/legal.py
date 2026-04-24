"""
LexShield Legal Q&A Endpoint
Accepts a query, returns a RAG-grounded answer with citations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from rag.pipeline import rag_pipeline

router = APIRouter(prefix="/api/v1/legal", tags=["Legal Q&A"])


class QueryRequest(BaseModel):
    query:           str
    doc_type_filter: Optional[str] = None   # 'statute' | 'judgment' | None


class CitationResponse(BaseModel):
    source:   str
    section:  str
    doc_type: str
    score:    float


class QueryResponse(BaseModel):
    query:        str
    answer:       str
    citations:    list[CitationResponse]
    context_used: bool
    warning:      Optional[str] = None


@router.post("/query", response_model=QueryResponse)
async def legal_query(request: QueryRequest):
    """
    Legal Q&A endpoint.
    Accepts a natural language legal question.
    Returns a grounded answer with citations from the knowledge base.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if len(request.query) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Query too long. Keep it under 1000 characters."
        )

    result = rag_pipeline.answer(
        query=request.query,
        doc_type_filter=request.doc_type_filter,
    )

    return QueryResponse(
        query        = result.query,
        answer       = result.answer,
        citations    = [CitationResponse(**c) for c in result.citations],
        context_used = result.context_used,
        warning      = result.warning,
    )