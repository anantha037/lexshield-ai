"""
LexShield AI — Vector Store  (Week 2 update)
============================================
Changes from Week 1:
  • ingest_chunks() now uses chunk["context_text"] as the document to embed
    (richer signal: source + chapter + section header + text)
  • Metadata now includes: section_title, chapter, chunk_type
  • search() returns chunk_id, context_text, section_title, chapter, chunk_type
  • New reset_collection() — wipes collection before re-ingestion
  • Backward-compatible: old chunks without context_text fall back to text
"""

import os
import time
import gc
import json
from typing import Optional

# CPU safety
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import chromadb
from chromadb.config import Settings

from rag.embedder import embedder   # LegalEmbedder singleton (Week 1, unchanged)

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "legal_documents"
INGEST_BATCH     = 16    # ChromaDB batch size (RAM safe on 8 GB)
BATCH_SLEEP      = 1.5   # seconds between batches
GC_EVERY_N       = 5     # run gc.collect() every N batches


class LegalVectorStore:
    """
    ChromaDB-backed vector store for LexShield legal corpus.

    Stores:
      document  = context_text  (section-header-prefixed text, used for embedding)
      metadata  = source, doc_type, section, section_title, chapter, chunk_type
      id        = chunk_id
    """

    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorStore] Collection '{COLLECTION_NAME}' — {self.count()} docs")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_chunks(self, chunks: list[dict], skip_existing: bool = True) -> int:
        """
        Batch-ingest chunk dicts into ChromaDB.

        Uses context_text for embedding (falls back to text if absent).
        Returns number of newly added chunks.
        """
        if not chunks:
            return 0

        # Deduplicate against existing ids if requested
        if skip_existing:
            existing_ids: set[str] = set()
            try:
                all_ids = self.collection.get(include=[])["ids"]
                existing_ids = set(all_ids)
            except Exception:
                pass
            new_chunks = [c for c in chunks if c.get("chunk_id", "") not in existing_ids]
        else:
            new_chunks = chunks

        if not new_chunks:
            print("[VectorStore] All chunks already present, skipping.")
            return 0

        total   = len(new_chunks)
        added   = 0
        batches = [new_chunks[i : i + INGEST_BATCH] for i in range(0, total, INGEST_BATCH)]

        print(f"[VectorStore] Ingesting {total} chunks in {len(batches)} batches ...")

        for batch_idx, batch in enumerate(batches):
            # Prepare document text for embedding
            docs = [
                c.get("context_text") or c.get("text", "")
                for c in batch
            ]
            ids = [c["chunk_id"] for c in batch]

            # Build metadata dicts — ChromaDB requires str/int/float/bool values
            metadatas = [
                {
                    "source":        str(c.get("source",        "")),
                    "doc_type":      str(c.get("doc_type",      "")),
                    "section":       str(c.get("section",       "")),
                    "section_title": str(c.get("section_title", "")),
                    "chapter":       str(c.get("chapter",       "")),
                    "chunk_type":    str(c.get("chunk_type",    "")),
                    "word_count":    int(c.get("word_count",    0)),
                }
                for c in batch
            ]

            # Embed using the Week 1 LegalEmbedder (batch_size=8 safe)
            embeddings = embedder.embed(docs)

            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            added += len(batch)

            # Memory management
            if (batch_idx + 1) % GC_EVERY_N == 0:
                gc.collect()

            time.sleep(BATCH_SLEEP)

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                print(f"  batch {batch_idx + 1}/{len(batches)}  ({added}/{total})")

        gc.collect()
        print(f"[VectorStore] Done. Added {added} chunks. Total: {self.count()}")
        return added

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_collection(self):
        """
        Delete and recreate the ChromaDB collection.
        Call before re-ingestion to start fresh.
        """
        print(f"[VectorStore] Deleting collection '{COLLECTION_NAME}' ...")
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection reset. Empty.")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 8) -> list[dict]:
        """
        Embed query → cosine similarity search → return results.

        Each result dict:
          chunk_id, text (=context_text stored in ChromaDB),
          source, doc_type, section, section_title, chapter, chunk_type,
          score  (float 0–1, higher = more similar)
        """
        if self.count() == 0:
            return []

        query_embedding = embedder.embed([query])[0]

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.count()),
            include=["documents", "distances", "metadatas"],
        )

        results: list[dict] = []
        for cid, doc, dist, meta in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["distances"][0],
            raw["metadatas"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity [0, 1]
            score = max(0.0, 1.0 - dist / 2.0)
            results.append({
                "chunk_id":      cid,
                "text":          doc,            # context_text stored as document
                "source":        meta.get("source",        ""),
                "doc_type":      meta.get("doc_type",      ""),
                "section":       meta.get("section",       ""),
                "section_title": meta.get("section_title", ""),
                "chapter":       meta.get("chapter",       ""),
                "chunk_type":    meta.get("chunk_type",    ""),
                "score":         round(score, 4),
            })

        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        try:
            r = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if not r["ids"]:
                return None
            meta = r["metadatas"][0]
            return {
                "chunk_id": chunk_id,
                "text":     r["documents"][0],
                **meta,
            }
        except Exception:
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────
vectorstore = LegalVectorStore()