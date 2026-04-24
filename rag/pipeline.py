"""
LexShield RAG Pipeline
=======================
End-to-end: legal question → retrieved context → cited answer.

Pipeline steps:
  1. Receive user query
  2. Embed query → vector
  3. Retrieve top-K chunks from ChromaDB
  4. Build grounded prompt with retrieved context
  5. LLM generates answer
  6. Format response with citations

Usage:
  from rag.pipeline import rag_pipeline
  result = rag_pipeline.answer("What are my rights as a tenant?")
"""

from dataclasses import dataclass
from typing import Optional

from rag.embedder    import embedder
from rag.vectorstore import vectorstore
from rag.llm         import llm


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Structured response returned by the RAG pipeline."""
    query:          str
    answer:         str
    citations:      list[dict]          # list of {source, section, doc_type, score}
    retrieved_chunks: list[dict]        # raw retrieved chunks (for debugging)
    context_used:   bool                # False if no relevant context found
    warning:        Optional[str] = None


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are LexShield, an AI-powered Indian legal assistant.

Your role is to help Indian citizens understand their legal rights and obligations.

Rules you MUST follow:
1. Answer ONLY based on the retrieved legal sections provided. Do not use outside knowledge.
2. Always cite which law or judgment supports each claim (e.g., "Under Section 420 of IPC..." or "As per the Kerala Rent Control Act...").
3. If the retrieved context does not contain enough information to answer the question, say explicitly: "The retrieved legal sections do not contain sufficient information to answer this question. Please consult a qualified lawyer."
4. Use plain, simple language. Avoid legal jargon where possible. Explain technical terms when you must use them.
5. Never give a definitive legal opinion — you provide information, not legal advice. End responses with: "Note: This is legal information, not legal advice. For your specific situation, consult a qualified lawyer."
6. If the question involves a criminal matter or urgent situation, mention that the person should contact the police or a lawyer immediately."""


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Constructs the full RAG prompt by injecting retrieved legal context.

    Format:
      [Retrieved Legal Sections]
      --- Section 1 ---
      Source: IPC 1860 | Section: 420 | Type: statute
      <text>
      ...
      [User Question]
      <query>
      [Instructions]
      Answer based only on the above sections. Cite sources.
    """
    if not retrieved_chunks:
        return (
            f"The user asked: {query}\n\n"
            "No relevant legal sections were found in the knowledge base. "
            "Tell the user you cannot find relevant legal information and recommend they consult a lawyer."
        )

    # Build context block
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source   = chunk.get("source",   "Unknown Source")
        section  = chunk.get("section",  "")
        doc_type = chunk.get("doc_type", "")
        text     = chunk.get("text",     "")

        header = f"--- Retrieved Section {i} ---"
        meta   = f"Source: {source} | Type: {doc_type}"
        if section:
            meta += f" | Section: {section}"

        context_parts.append(f"{header}\n{meta}\n{text}")

    context_block = "\n\n".join(context_parts)

    prompt = f"""
[RETRIEVED LEGAL SECTIONS]
{context_block}

[USER QUESTION]
{query}

[YOUR TASK]
Using ONLY the retrieved legal sections above:
1. Answer the user's question clearly and in plain language.
2. For every legal claim you make, cite the source (e.g., "Under Section X of [Act Name]...").
3. Structure your answer with clear paragraphs.
4. If the sections above do not contain enough information, say so honestly.
5. End with the standard disclaimer about legal advice.

Answer:""".strip()

    return prompt


# ── Main pipeline class ───────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full retrieve-then-generate pipeline.
    """

    def __init__(
        self,
        n_results:           int   = 5,
        min_relevance_score: float = 0.25,
    ):
        """
        Args:
            n_results           : how many chunks to retrieve per query
            min_relevance_score : chunks below this similarity are discarded
                                  (0.30 is permissive — catches edge cases)
        """
        self.n_results           = n_results
        self.min_relevance_score = min_relevance_score

    def answer(
        self,
        query:           str,
        doc_type_filter: Optional[str] = None,
        verbose:         bool          = False,
    ) -> RAGResponse:
        """
        Full RAG pipeline for a single legal query.

        Args:
            query          : user's natural language legal question
            doc_type_filter: optionally restrict retrieval to 'statute' or 'judgment'
            verbose        : if True, prints retrieved chunks before generating

        Returns:
            RAGResponse with answer, citations, and retrieved chunks.
        """

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        raw_results = vectorstore.search(
            query,
            n_results=self.n_results,
            doc_type_filter=doc_type_filter,
        )

        # Filter out low-relevance chunks
        relevant = [r for r in raw_results if r["score"] >= self.min_relevance_score]

        if verbose:
            print(f"\n[RAG] Query: {query}")
            print(f"[RAG] Retrieved {len(raw_results)} chunks, {len(relevant)} above threshold")
            for i, r in enumerate(relevant, 1):
                print(f"  {i}. [{r['score']:.3f}] {r['source'][:50]} — {r['section'][:40]}")

        # ── Step 2: Build prompt ──────────────────────────────────────────────
        prompt = build_rag_prompt(query, relevant)

        # ── Step 3: Generate ──────────────────────────────────────────────────
        answer_text = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # ── Step 4: Build citations ───────────────────────────────────────────
        citations = []
        seen_sources = set()
        for chunk in relevant:
            key = f"{chunk['source']}|{chunk['section']}"
            if key not in seen_sources:
                seen_sources.add(key)
                citations.append({
                    "source":   chunk["source"],
                    "section":  chunk["section"],
                    "doc_type": chunk["doc_type"],
                    "score":    chunk["score"],
                })

        context_used = len(relevant) > 0
        warning      = None
        if not context_used:
            warning = "No relevant legal sections found. Answer may be incomplete."

        return RAGResponse(
            query           = query,
            answer          = answer_text,
            citations       = citations,
            retrieved_chunks = relevant,
            context_used    = context_used,
            warning         = warning,
        )

    def pretty_print(self, response: RAGResponse) -> None:
        """Prints a formatted RAGResponse to terminal for manual testing."""
        width = 65
        print("\n" + "=" * width)
        print("  LEXSHIELD LEGAL Q&A")
        print("=" * width)
        print(f"  Query: {response.query}")
        print("-" * width)
        print("\nANSWER:\n")
        print(response.answer)

        if response.citations:
            print("\n" + "-" * width)
            print("SOURCES USED:")
            for i, c in enumerate(response.citations, 1):
                section_str = f" › {c['section'][:55]}" if c["section"] else ""
                print(f"  {i}. [{c['doc_type'].upper()}] {c['source']}{section_str}")
                print(f"      Relevance score: {c['score']:.3f}")

        if response.warning:
            print(f"\n  Warning: {response.warning}")

        print("=" * width + "\n")


# ── Module-level singleton ────────────────────────────────────────────────────
rag_pipeline = RAGPipeline(n_results=5, min_relevance_score=0.25)