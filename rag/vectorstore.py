"""
LexShield AI — Vector Store  (section lookup patch)
====================================================
Key addition: get_by_section() — direct ChromaDB metadata query by section number.
Bypasses BM25 and vector text-similarity when an exact section is requested.
Everything else unchanged from Day 1.
"""

import os
import time
import gc
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import chromadb
from chromadb.config import Settings
from rag.embedder import embedder

COLLECTION_NAME = "legal_documents"
INGEST_BATCH    = 16
BATCH_SLEEP     = 1.5
GC_EVERY_N      = 5

# Maps lowercase query keywords → partial source strings for optional filtering
SOURCE_KEYWORDS: dict[str, str] = {
    "ipc":                      "Indian Penal Code",
    "indian penal code":        "Indian Penal Code",
    "bns":                      "Bharatiya Nyaya Sanhita",
    "bharatiya nyaya sanhita":  "Bharatiya Nyaya Sanhita",
    "crpc":                     "Code of Criminal Procedure",
    "code of criminal procedure": "Code of Criminal Procedure",
    "consumer":                 "Consumer Protection",
    "wages":                    "Code on Wages",
    "kerala":                   "Kerala Buildings",
}


class LegalVectorStore:

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

    # ── Direct metadata lookup — the section fast path ────────────────────────

    def get_by_section(
        self,
        section_number: str,
        source_hint:    Optional[str] = None,
    ) -> list[dict]:
        """
        Query ChromaDB metadata directly by section number.
        Returns all chunks where metadata.section == section_number.
        Optionally filters to chunks whose source contains source_hint.

        Returned chunks get hybrid_score=1.0 so they pin to the top of results.
        retrieval_source is tagged "metadata" to distinguish from BM25/vector.

        This is intentionally called BEFORE hybrid search when an explicit
        section number is detected in the query.
        """
        section_number = section_number.strip().upper()
        
        try:
            raw = self.collection.get(
                where={"section": {"$eq": section_number}},
                include=["documents", "metadatas",],
                limit=20,
            )

            if not raw["ids"]:
                return []

            results: list[dict] = []
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
                source = meta.get("source", "")

                # Optional source filter — "Indian Penal Code" filters out CrPC s.108 etc.
                if source_hint and source_hint.lower() not in source.lower():
                    continue

                results.append({
                    "chunk_id":        cid,
                    "text":            doc,
                    "source":          source,
                    "doc_type":        meta.get("doc_type",      ""),
                    "section":         meta.get("section",       ""),
                    "section_title":   meta.get("section_title", ""),
                    "chapter":         meta.get("chapter",       ""),
                    "chunk_type":      meta.get("chunk_type",    ""),
                    # Scores — pinned high so these always appear at the top
                    "score":           1.0,
                    "vector_score":    1.0,
                    "bm25_score":      1.0,
                    "bm25_score_norm": 1.0,
                    "hybrid_score":    1.0,
                    "retrieval_source": "metadata",
                    "rerank_score":    None,
                })

            return results

        except Exception as e:
            print(f"[VectorStore] get_by_section({section_number!r}) error: {e}")
            return []

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest_chunks(self, chunks: list[dict], skip_existing: bool = True) -> int:
        if not chunks:
            return 0
        if skip_existing:
            try:
                existing_ids = set(self.collection.get(include=[])["ids"])
            except Exception:
                existing_ids = set()
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
            docs      = [c.get("context_text") or c.get("text", "") for c in batch]
            ids       = [c["chunk_id"] for c in batch]
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
            embeddings = embedder.embed(docs)
            self.collection.add(
                ids=ids, documents=docs,
                embeddings=embeddings, metadatas=metadatas,
            )
            added += len(batch)
            if (batch_idx + 1) % GC_EVERY_N == 0:
                gc.collect()
            time.sleep(BATCH_SLEEP)
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                print(f"  batch {batch_idx + 1}/{len(batches)}  ({added}/{total})")

        gc.collect()
        print(f"[VectorStore] Done. Added {added}. Total: {self.count()}")
        return added

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_collection(self):
        print(f"[VectorStore] Deleting '{COLLECTION_NAME}' ...")
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection reset.")

    # ── Vector search ─────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 8) -> list[dict]:
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
            raw["ids"][0], raw["documents"][0],
            raw["distances"][0], raw["metadatas"][0],
        ):
            score = max(0.0, 1.0 - dist / 2.0)
            results.append({
                "chunk_id":      cid,
                "text":          doc,
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
            r = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if not r["ids"]:
                return None
            return {"chunk_id": chunk_id, "text": r["documents"][0], **r["metadatas"][0]}
        except Exception:
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────
vectorstore = LegalVectorStore()