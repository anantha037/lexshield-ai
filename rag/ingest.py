"""
LexShield Ingestion Script — Memory-Safe Version
=================================================
Optimised for 8GB RAM, no GPU.
Uses small batches + garbage collection to prevent OOM freeze.

Run:
  python rag/ingest.py
"""

import json
import time
import gc
from pathlib import Path

CHUNKS_FILE = Path("data/processed/chunks.json")

# ── Tuned for 8GB RAM ─────────────────────────────────────────────────────────
EMBED_BATCH_SIZE  = 8    # texts embedded at once (was 32 — too high)
INGEST_BATCH_SIZE = 16   # chunks sent to ChromaDB at once (was 50 — too high)
GC_EVERY_N_BATCHES = 5   # force garbage collection every N batches


def main():
    if not CHUNKS_FILE.exists():
        print(f"{CHUNKS_FILE} not found. Run data/preprocessor.py first.")
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    by_type: dict[str, int] = {}
    for c in chunks:
        t = c.get("doc_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    print("Breakdown by type:")
    for doc_type, count in by_type.items():
        print(f"  {doc_type:12s}: {count}")
    print()

    # Import after printing so model-load message appears cleanly
    from rag.embedder import embedder
    from rag.vectorstore import vectorstore

    # Check what's already in ChromaDB (safe re-run)
    existing_ids = set(vectorstore.collection.get(include=[])["ids"])
    chunks_to_add = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not chunks_to_add:
        print("All chunks already ingested.")
        print(f"   Total in ChromaDB: {vectorstore.count()}")
        return

    print(f"Already ingested : {len(existing_ids)}")
    print(f"Remaining        : {len(chunks_to_add)}")
    print()

    total    = len(chunks_to_add)
    n_batches = (total + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE
    added    = 0
    start    = time.time()

    print(f"Starting ingestion: {n_batches} batches of {INGEST_BATCH_SIZE}")
    print(f"Embed batch size  : {EMBED_BATCH_SIZE} (memory-safe for 8GB)\n")

    for batch_idx in range(n_batches):
        s = batch_idx * INGEST_BATCH_SIZE
        e = min(s + INGEST_BATCH_SIZE, total)
        batch = chunks_to_add[s:e]

        texts     = [c["text"]     for c in batch]
        ids       = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "source":     c.get("source",   ""),
                "doc_type":   c.get("doc_type",  ""),
                "section":    c.get("section",   ""),
                "court":      c.get("court",     ""),
                "word_count": str(c.get("word_count", 0)),
            }
            for c in batch
        ]

        # Embed with small batch size to control RAM spike
        vectors = embedder.embed(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress=False,
        )

        # Deduplicate within batch
        seen = set()
        deduped = [
            (i, v, t, m)
            for i, v, t, m in zip(ids, vectors, texts, metadatas)
            if i not in seen and not seen.add(i)
        ]

        if deduped:
            d_ids, d_vecs, d_texts, d_metas = zip(*deduped)
            vectorstore.collection.add(
                ids=list(d_ids),
                embeddings=list(d_vecs),
                documents=list(d_texts),
                metadatas=list(d_metas),
            )
            added += len(d_ids)

        # Progress
        pct     = (added / total) * 100
        elapsed = time.time() - start
        eta_sec = (elapsed / max(added, 1)) * (total - added)
        eta_min = eta_sec / 60
        print(
            f"  Batch {batch_idx + 1:3d}/{n_batches} | "
            f"{added:4d}/{total} ({pct:5.1f}%) | "
            f"ETA: {eta_min:.0f} min"
        )

        # Force garbage collection every N batches to free RAM
        if (batch_idx + 1) % GC_EVERY_N_BATCHES == 0:
            gc.collect()

    elapsed_total = (time.time() - start) / 60
    print(f"\nIngestion complete!")
    print(f"   Chunks added     : {added}")
    print(f"   Total in ChromaDB: {vectorstore.count()}")
    print(f"   Time taken       : {elapsed_total:.1f} minutes")


if __name__ == "__main__":
    main()