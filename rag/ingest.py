"""
LexShield AI — Re-ingestion Script
====================================
Run this ONCE for a full corpus reset, or with --category/--slugs
for selective addition of new acts without touching existing embeddings.

Usage (from project root, venv active):

  Full re-ingest (all 50+ acts + judgments):
    python -m rag.ingest

  Add only new acts in a category (no ChromaDB reset, dedup handles repeats):
    python -m rag.ingest --category criminal --skip-reset

  Add a single act by slug:
    python -m rag.ingest --slugs bnss bsa --skip-reset

  Chunk only, no DB writes:
    python -m rag.ingest --dry-run

  Skip re-chunking if chunks.json is already current:
    python -m rag.ingest --skip-chunk

Available categories:
  criminal | family | corporate | taxation | property
  labour   | health | environment | technology | civil

Expected runtime on i5-8250U (full run):
  Chunking:    8–15 min  (50+ PDFs via PyMuPDF, CPU only)
  Ingestion:   40–70 min (embedding ~25 000 chunks, batch_size=8)
  BM25 build:  < 60 sec
"""

import sys
import os
import gc
import time
import argparse

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ── Parse flags ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LexShield re-ingestion")
parser.add_argument("--skip-chunk",  action="store_true",
                    help="Skip chunking step (use existing chunks.json)")
parser.add_argument("--skip-reset",  action="store_true",
                    help="Skip ChromaDB reset — add-only, dedup handles repeats")
parser.add_argument("--dry-run",     action="store_true",
                    help="Run chunking only, no ChromaDB/BM25 changes")
parser.add_argument("--max-iltur",   type=int, default=1000,
                    help="Max IL-TUR judgment records (default: 1000)")
parser.add_argument("--max-sc",      type=int, default=2000,
                    help="Max SC judgment records (default: 2000)")

# ── Selective ingestion ───────────────────────────────────────────────────────
# These two flags let you add new acts without a full re-ingest.
# When either is set, judgment datasets are skipped automatically.
parser.add_argument("--category",    type=str, default=None,
                    help="Only process statutes in this category "
                         "(criminal|family|corporate|taxation|property|"
                         "labour|health|environment|technology|civil)")
parser.add_argument("--slugs",       type=str, nargs="+", default=None,
                    help="Only process specific statute slugs "
                         "(e.g. --slugs bnss bsa pocso)")

args = parser.parse_args()

# Guard: selective flags should always pair with --skip-reset
# (you don't want to wipe existing embeddings when adding 1 new act)
if (args.category or args.slugs) and not args.skip_reset and not args.dry_run:
    logger.info("⚠️  WARNING: --category/--slugs without --skip-reset will wipe "
          "existing ChromaDB data.")
    ans = input("Continue and reset? [y/N] ").strip().lower()
    if ans != "y":
        logger.info("Aborted. Re-run with --skip-reset to append instead.")
        sys.exit(0)


def separator(title: str = "") -> None:
    line = "=" * 64
    if title:
        logger.info(f"\n{line}\n  {title}\n{line}")
    else:
        logger.info(line)


# ── Step 1: Contextual chunking ───────────────────────────────────────────────
chunks = []

if args.skip_chunk:
    separator("Step 1: SKIPPED (--skip-chunk)")
    import json
    from pathlib import Path
    path = Path("data/processed/chunks.json")
    if not path.exists():
        logger.info("ERROR: chunks.json not found. Remove --skip-chunk.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} existing chunks from {path}")
else:
    separator("Step 1: Contextual chunking")
    from data.preprocessor import run_full_pipeline, STATUTE_CONFIGS

    # Show what will be processed
    if args.category:
        relevant = [c for c in STATUTE_CONFIGS if c.get("category") == args.category]
        logger.info(f"Category filter: '{args.category}' -> {len(relevant)} statute(s)")
    elif args.slugs:
        relevant = [c for c in STATUTE_CONFIGS if c["slug"] in args.slugs]
        logger.info(f"Slug filter: {args.slugs} -> {len(relevant)} statute(s)")
        missing = set(args.slugs) - {c["slug"] for c in relevant}
        if missing:
            logger.info(f"  ⚠️  Unknown slugs (check STATUTE_CONFIGS): {missing}")
    else:
        logger.info(f"Full run: {len(STATUTE_CONFIGS)} statutes configured")

    chunks = run_full_pipeline(
        max_iltur=args.max_iltur,
        max_sc=args.max_sc,
        category=args.category,
        slugs=args.slugs,
    )

if not chunks:
    logger.info("ERROR: No chunks produced. Aborting.")
    sys.exit(1)

logger.info(f"\nChunks ready: {len(chunks)}")
gc.collect()

if args.dry_run:
    logger.info("\n[DRY RUN] Chunking complete. Skipping DB/BM25 changes.")
    sys.exit(0)

# ── Step 2: ChromaDB reset + ingestion ───────────────────────────────────────
separator("Step 2: ChromaDB re-ingestion")

from rag.vectorstore import vectorstore

if not args.skip_reset:
    logger.info("Resetting ChromaDB collection ...")
    vectorstore.reset_collection()
    logger.info("Collection cleared.\n")
else:
    existing = vectorstore.count()
    logger.info(f"(--skip-reset: keeping {existing} existing docs, appending {len(chunks)} new)\n")

t0 = time.time()
added = vectorstore.ingest_chunks(chunks, skip_existing=args.skip_reset)
elapsed = time.time() - t0
logger.info(f"\nIngestion done in {elapsed/60:.1f} min.")
logger.info(f"ChromaDB total docs: {vectorstore.count()}")
gc.collect()

# ── Step 3: BM25 rebuild ──────────────────────────────────────────────────────
separator("Step 3: BM25 index rebuild")
from rag.bm25_retriever import bm25_retriever
bm25_retriever.rebuild()
logger.info(f"BM25 index: {bm25_retriever.count()} docs indexed.")

# ── Step 4: Quick smoke-test ──────────────────────────────────────────────────
separator("Step 4: Smoke tests")

test_queries = [
    "Section 420 cheating dishonestly",           # IPC/BNS exact keyword
    "punishment for murder under IPC",             # semantic criminal
    "tenant eviction notice period Kerala",        # property — existing
    "cheque bounce dishonour section 138",         # Negotiable Instruments — new
    "GST input tax credit eligibility",            # CGST — new
    "data breach penalty DPDP Act",                # technology — new
    "motor vehicle accident compensation",         # civil — new
    "domestic violence protection order",          # family — new
]

from rag.hybrid_search import hybrid_searcher

for q in test_queries:
    results = hybrid_searcher.search_explain(q, n_results=3)
    logger.info(f"\nQuery: '{q}'")
    for r in results:
        src       = r.get("source",         "?")[:50]
        sec       = r.get("section",        "")
        breakdown = r.get("score_breakdown", "")
        logger.info(f"  {breakdown}  |  {src}  sec={sec}")

separator("DONE")

if args.category or args.slugs:
    logger.info("Selective ingestion complete.")
    logger.info("Tip: run with --skip-chunk --skip-reset if you need to add more acts.")
else:
    logger.info("Full corpus ingestion complete.")
logger.info("Run: uvicorn api.main:app --reload")