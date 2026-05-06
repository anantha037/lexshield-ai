"""
LexShield AI — Week 2 Re-ingestion Script
==========================================
Run this ONCE to migrate from Week 1 to Week 2:
  1. Re-chunk entire corpus with contextual_chunk_document()
  2. Reset ChromaDB collection (delete old, create fresh)
  3. Ingest all new chunks (context_text as document)
  4. Rebuild BM25 index

Usage (from project root, venv active):
    python -m rag.ingest

Expected runtime on i5-8250U:
  - Chunking:    2–4 minutes (PyMuPDF + regex, CPU only)
  - Ingestion:   15–25 minutes (embedding 4000+ chunks, batch_size=8)
  - BM25 build:  < 30 seconds

Flags:
  --skip-chunk   Skip re-chunking if chunks.json already updated
  --skip-reset   Skip ChromaDB reset (add-only, deduplication handles repeats)
  --dry-run      Run chunking only, no ChromaDB/BM25 changes
"""

import sys
import os
import gc
import time
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ── Parse flags ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LexShield Week 2 re-ingestion")
parser.add_argument("--skip-chunk",  action="store_true", help="Skip chunking step")
parser.add_argument("--skip-reset",  action="store_true", help="Skip ChromaDB reset")
parser.add_argument("--dry-run",     action="store_true", help="Chunk only, no DB changes")
parser.add_argument("--max-iltur",   type=int, default=1000)
parser.add_argument("--max-sc",      type=int, default=2000)
args = parser.parse_args()


def separator(title: str = "") -> None:
    line = "=" * 64
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ── Step 1: Contextual chunking ───────────────────────────────────────────────
chunks = []

if args.skip_chunk:
    separator("Step 1: SKIPPED (--skip-chunk)")
    import json
    from pathlib import Path
    path = Path("data/processed/chunks.json")
    if not path.exists():
        print("ERROR: chunks.json not found. Remove --skip-chunk.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} existing chunks from {path}")
else:
    separator("Step 1: Contextual chunking")
    from data.preprocessor import run_full_pipeline
    chunks = run_full_pipeline(
        max_iltur=args.max_iltur,
        max_sc=args.max_sc,
    )

if not chunks:
    print("ERROR: No chunks produced. Aborting.")
    sys.exit(1)

print(f"\nChunks ready: {len(chunks)}")
gc.collect()

if args.dry_run:
    print("\n[DRY RUN] Chunking complete. Skipping DB/BM25 changes.")
    sys.exit(0)

# ── Step 2: ChromaDB reset + ingestion ───────────────────────────────────────
separator("Step 2: ChromaDB re-ingestion")

from rag.vectorstore import vectorstore

if not args.skip_reset:
    print("Resetting ChromaDB collection ...")
    vectorstore.reset_collection()
    print("Collection cleared.\n")
else:
    print(f"(--skip-reset: existing {vectorstore.count()} docs kept)\n")

t0 = time.time()
added = vectorstore.ingest_chunks(chunks, skip_existing=args.skip_reset)
elapsed = time.time() - t0
print(f"\nIngestion done in {elapsed/60:.1f} min.")
print(f"ChromaDB total docs: {vectorstore.count()}")
gc.collect()

# ── Step 3: BM25 rebuild ──────────────────────────────────────────────────────
separator("Step 3: BM25 index rebuild")
from rag.bm25_retriever import bm25_retriever
bm25_retriever.rebuild()
print(f"BM25 index: {bm25_retriever.count()} docs indexed.")

# ── Step 4: Quick smoke-test ──────────────────────────────────────────────────
separator("Step 4: Smoke tests")

test_queries = [
    "Section 420 cheating dishonestly",       # exact keyword
    "punishment for murder under IPC",         # semantic
    "tenant eviction notice period Kerala",    # mixed
]

from rag.hybrid_search import hybrid_searcher

for q in test_queries:
    results = hybrid_searcher.search_explain(q, n_results=3)
    print(f"\nQuery: '{q}'")
    for r in results:
        src   = r.get("source",         "?")[:45]
        sec   = r.get("section",        "")
        breakdown = r.get("score_breakdown", "")
        print(f"  {breakdown}  |  {src}  sec={sec}")

separator("DONE")
print("Week 2 ingestion complete.\n"
      "Run: uvicorn api.main:app --reload")