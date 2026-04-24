"""
LexShield Vector Store
======================
Wraps ChromaDB with ingest and search operations.

Storage: data/chroma_db/ (persistent on disk — survives restarts)

Collections:
  legal_documents — all statute + judgment chunks with metadata

Usage:
  from rag.vectorstore import vectorstore
  vectorstore.ingest_chunks(chunks)
  results = vectorstore.search("landlord not returning deposit", n_results=5)
"""

import chromadb
from pathlib import Path
from typing import Optional

from rag.embedder import embedder

# ── Storage path ──────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = str(Path("data/chroma_db").resolve())
COLLECTION_NAME = "legal_documents"


class LegalVectorStore:
    """
    Persistent ChromaDB vector store for LexShield legal chunks.
    """

    def __init__(self, db_path: str = CHROMA_DB_PATH):
        print(f"Connecting to ChromaDB at: {db_path}")
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        print(f"Collection '{COLLECTION_NAME}' ready.")
        print(f"Current document count: {self.collection.count()}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_chunks(
        self,
        chunks: list[dict],
        batch_size: int = 20,
        skip_existing: bool = True,
    ) -> int:
        """
        Embeds and stores chunks into ChromaDB.

        Args:
            chunks       : list of chunk dicts from chunks.json
            batch_size   : how many chunks to ingest per ChromaDB call.
                           50 is safe for 8GB RAM.
            skip_existing: if True, chunks whose chunk_id already exists
                           in ChromaDB are skipped (safe to re-run).

        Returns:
            Number of new chunks added.
        """
        if skip_existing:
            existing_ids = set(self.collection.get(include=[])["ids"])
            chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
            if not chunks:
                print("All chunks already ingested. Nothing to add.")
                return 0
            print(f"Skipping already-ingested chunks. Remaining: {len(chunks)}")

        total   = len(chunks)
        added   = 0
        n_batch = (total + batch_size - 1) // batch_size  # ceiling division

        print(f"\nIngesting {total} chunks in {n_batch} batches of {batch_size}...")

        for batch_idx in range(n_batch):
            start = batch_idx * batch_size
            end   = min(start + batch_size, total)
            batch = chunks[start:end]

            texts     = [c["text"]     for c in batch]
            ids       = [c["chunk_id"] for c in batch]
            metadatas = [
                {
                    "source":   c.get("source",   ""),
                    "doc_type": c.get("doc_type",  ""),
                    "section":  c.get("section",   ""),
                    "court":    c.get("court",     ""),
                    "word_count": str(c.get("word_count", 0)),
                }
                for c in batch
            ]

            # Embed the batch
            vectors = embedder.embed(texts, batch_size=32, show_progress=False)

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )

            added += len(batch)
            pct    = (added / total) * 100
            print(f"Batch {batch_idx + 1}/{n_batch} | {added}/{total} ({pct:.1f}%)")

            # Slow down slightly
            import time
            time.sleep(1.5)

        print(f"\nIngestion complete. Total in collection: {self.collection.count()}")
        return added

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 10,
        doc_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Embeds the query and retrieves the most semantically similar chunks.

        Args:
            query          : natural language legal question
            n_results      : how many chunks to return (default 10)
            doc_type_filter: optionally restrict to 'statute' or 'judgment'

        Returns:
            List of result dicts, each containing:
              text, source, doc_type, section, court, score
        """
        query_vector = embedder.embed_single(query)

        where_filter = {"doc_type": doc_type_filter} if doc_type_filter else None

        raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score 0–1: higher is better
            score = round(1 - (dist / 2), 4)

            results.append({
                "text":     doc,
                "source":   meta.get("source",   ""),
                "doc_type": meta.get("doc_type",  ""),
                "section":  meta.get("section",   ""),
                "court":    meta.get("court",     ""),
                "score":    score,
            })

        return results

    def count(self) -> int:
        """Returns total number of chunks stored."""
        return self.collection.count()

    def reset(self) -> None:
        """
        Deletes and recreates the collection.
        Use only if you want to re-ingest from scratch.
        """
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Collection reset. Count: {self.collection.count()}")


# ── Module-level singleton ────────────────────────────────────────────────────
vectorstore = LegalVectorStore()