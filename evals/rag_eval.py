"""
evals/rag_eval.py
=================
Step 1 — RAG Pipeline Evaluation with RAGAS
============================================

Evaluates LexShield's RAG pipeline across 4 metrics:
  - context_precision:  Are retrieved chunks relevant to the question?
  - context_recall:     Do retrieved chunks cover the ground truth answer?
  - answer_faithfulness: Does the generated answer stay within retrieved chunks?
  - answer_correctness: How close is the generated answer to ground truth?

KEY DESIGN — Phase separation (solves Groq rate limit):
  Phase 1: Retrieval only — runs hybrid_searcher.search() for all 25 questions.
           Zero LLM calls. Saves contexts to JSON. Can run any time.
  Phase 2: Generation — 5 batches of 5 questions with 25s sleep between batches.
           ~5 Groq calls per batch (plus query rewriting). Saves answers to JSON.
  Phase 3: RAGAS scoring — loads saved data, runs RAGAS evaluator.
           RAGAS LLM judge uses a separate lighter Groq model (lower rate limit cost).

You can run each phase independently. If Phase 2 fails mid-way, it resumes
from the last saved answer (skips already-generated questions).

Usage:
    # Full run (all 3 phases)
    python -m evals.rag_eval

    # Individual phases
    python -m evals.rag_eval --phase 1    # retrieval only (fast, no rate limits)
    python -m evals.rag_eval --phase 2    # generation only (needs Phase 1 data)
    python -m evals.rag_eval --phase 3    # RAGAS scoring (needs Phase 1+2 data)

    # Phase 1 + retrieval metrics only (no Groq calls at all)
    python -m evals.rag_eval --phase 1 --retrieval-only

Outputs:
    evals/results/rag_phase1_contexts.json    — retrieved contexts per question
    evals/results/rag_phase2_answers.json     — generated answers per question
    evals/results/rag_ragas_results.json      — RAGAS per-question scores
    evals/results/rag_summary_latest.json     — summary metrics
"""

import os
import sys
import json
import time
import logging
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_DATA_PATH = Path(__file__).parent / "test_data" / "rag_questions.json"
RESULTS_DIR    = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHASE1_PATH = RESULTS_DIR / "rag_phase1_contexts.json"
PHASE2_PATH = RESULTS_DIR / "rag_phase2_answers.json"

# Batching config — tuned for Groq free tier
GEN_BATCH_SIZE  = 5    # Questions per generation batch
GEN_BATCH_SLEEP = 25   # Seconds between generation batches

# Number of chunks to retrieve per question for eval
N_RETRIEVAL = 8


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — RETRIEVAL (Zero LLM calls)
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1_retrieval() -> list[dict]:
    """
    Run hybrid search for all 25 questions and save {question, contexts, ground_truth}.
    No LLM calls — uses ChromaDB + BM25 directly.
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Retrieval (zero LLM calls)")
    logger.info("=" * 60)

    from rag.hybrid_search import hybrid_searcher
    from rag.pipeline import preprocess_query

    with open(TEST_DATA_PATH) as f:
        questions = json.load(f)

    phase1_data = []

    for i, q in enumerate(questions):
        logger.info(f"[{i+1:02d}/{len(questions)}] {q['id']}: {q['question'][:70]}")
        try:
            expanded = preprocess_query(q["question"])
            chunks   = hybrid_searcher.search(
                expanded,
                n_results=N_RETRIEVAL,
                category_filter=None,
                act_hint=None,
            )
            # Extract text from chunks
            contexts = [c.get("text", "").strip() for c in chunks if c.get("text")]

            entry = {
                "id":           q["id"],
                "question":     q["question"],
                "ground_truth": q["ground_truth"],
                "contexts":     contexts,
                "n_chunks":     len(chunks),
                "complexity":   q.get("complexity", ""),
                "category":     q.get("category", ""),
                "expanded_q":   expanded,
                "chunk_meta":   [
                    {
                        "source":       c.get("source", ""),
                        "section":      c.get("section", ""),
                        "hybrid_score": round(float(c.get("hybrid_score", 0)), 4),
                    }
                    for c in chunks
                ],
            }
            phase1_data.append(entry)
            logger.info(f"         Retrieved {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"         Error: {e}")
            phase1_data.append({
                "id":           q["id"],
                "question":     q["question"],
                "ground_truth": q["ground_truth"],
                "contexts":     [],
                "n_chunks":     0,
                "error":        str(e),
            })

        # Light GC every 5 questions — helps on 8GB RAM
        if (i + 1) % 5 == 0:
            gc.collect()

    with open(PHASE1_PATH, "w") as f:
        json.dump(phase1_data, f, indent=2)
    logger.info(f"\nPhase 1 complete. Saved to {PHASE1_PATH}")

    # Print retrieval stats without LLM
    _print_retrieval_stats(phase1_data)
    return phase1_data


def _print_retrieval_stats(phase1_data: list[dict]) -> None:
    """Compute and print retrieval metrics that need no LLM."""
    total           = len(phase1_data)
    zero_chunks     = sum(1 for d in phase1_data if d.get("n_chunks", 0) == 0)
    avg_chunks      = sum(d.get("n_chunks", 0) for d in phase1_data) / total if total else 0
    empty_ctx_rate  = zero_chunks / total if total else 0

    # Coverage proxy: does any retrieved chunk mention an expected act?
    coverage_hits = 0
    for d in phase1_data:
        if not d.get("contexts"):
            continue
        all_text = " ".join(d["contexts"]).lower()
        # Crude check — if ground truth keywords appear in retrieved text
        gt_words = set(d.get("ground_truth", "").lower().split())
        ctx_words = set(all_text.split())
        overlap   = len(gt_words & ctx_words) / len(gt_words) if gt_words else 0
        if overlap > 0.3:   # 30% keyword overlap proxy for coverage
            coverage_hits += 1
    coverage_proxy = coverage_hits / total if total else 0

    print("\n── Phase 1 Retrieval Stats (no LLM) ──────────────────────────")
    print(f"  Questions evaluated:   {total}")
    print(f"  Zero chunks retrieved: {zero_chunks} ({empty_ctx_rate*100:.1f}%)")
    print(f"  Avg chunks/question:   {avg_chunks:.1f}")
    print(f"  Coverage proxy (>30% keyword overlap): {coverage_proxy*100:.1f}%")
    print("  (Full context_recall requires RAGAS in Phase 3)")
    print("─" * 62)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — GENERATION (Batched, rate-limit safe)
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2_generation() -> list[dict]:
    """
    Generate answers using the full RAG pipeline for all 25 questions.
    Batched with sleep between batches. Resumes from existing answers.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Generation (batched, rate-limit safe)")
    logger.info(f"         Batch size: {GEN_BATCH_SIZE} | Sleep: {GEN_BATCH_SLEEP}s")
    logger.info("=" * 60)

    if not PHASE1_PATH.exists():
        raise FileNotFoundError(
            f"Phase 1 data not found at {PHASE1_PATH}. Run --phase 1 first."
        )

    from rag.pipeline import rag_pipeline

    with open(PHASE1_PATH) as f:
        phase1_data = json.load(f)

    # Load existing answers to enable resume
    existing_answers: dict[str, dict] = {}
    if PHASE2_PATH.exists():
        with open(PHASE2_PATH) as f:
            saved = json.load(f)
        existing_answers = {d["id"]: d for d in saved}
        logger.info(f"Resuming: {len(existing_answers)} answers already saved")

    phase2_data = list(existing_answers.values())
    pending     = [d for d in phase1_data if d["id"] not in existing_answers]
    logger.info(f"Pending: {len(pending)} questions")

    for batch_start in range(0, len(pending), GEN_BATCH_SIZE):
        batch = pending[batch_start: batch_start + GEN_BATCH_SIZE]
        batch_num = batch_start // GEN_BATCH_SIZE + 1
        total_batches = (len(pending) + GEN_BATCH_SIZE - 1) // GEN_BATCH_SIZE

        logger.info(f"\nBatch {batch_num}/{total_batches} "
                    f"({len(batch)} questions)")

        for item in batch:
            logger.info(f"  Generating: {item['id']} — {item['question'][:60]}")
            t0 = time.perf_counter()
            try:
                answer_obj = rag_pipeline.query(
                    user_query=item["question"],
                    n_results=N_RETRIEVAL,
                )
                latency = (time.perf_counter() - t0) * 1000
                phase2_data.append({
                    "id":              item["id"],
                    "question":        item["question"],
                    "ground_truth":    item["ground_truth"],
                    "answer":          answer_obj.answer_text,
                    "sources":         answer_obj.sources_consulted,
                    "synthesis_note":  answer_obj.synthesis_note,
                    "grounding_warn":  answer_obj.grounding_warning,
                    "latency_ms":      round(latency, 1),
                    "error":           None,
                })
                logger.info(f"    ✓ {latency:.0f}ms — {answer_obj.answer_text[:60]}...")
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                logger.error(f"    ✗ Error: {e}")
                phase2_data.append({
                    "id":           item["id"],
                    "question":     item["question"],
                    "ground_truth": item["ground_truth"],
                    "answer":       "",
                    "latency_ms":   round(latency, 1),
                    "error":        str(e),
                })

            # Save after every question — no data loss on crash
            with open(PHASE2_PATH, "w") as f:
                json.dump(phase2_data, f, indent=2)

        gc.collect()

        if batch_start + GEN_BATCH_SIZE < len(pending):
            logger.info(f"  Sleeping {GEN_BATCH_SLEEP}s before next batch...")
            time.sleep(GEN_BATCH_SLEEP)

    logger.info(f"\nPhase 2 complete. Saved to {PHASE2_PATH}")
    return phase2_data


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — RAGAS SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase3_ragas(retrieval_only: bool = False) -> dict:
    """
    Load Phase 1 + Phase 2 data and compute RAGAS metrics.

    retrieval_only=True: Only compute context_precision and context_recall
    (using RAGAS's context metrics without the generation metrics).
    This requires only the Phase 1 data and uses a lighter judge model.
    """
    logger.info("=" * 60)
    logger.info("Phase 3: RAGAS Scoring")
    logger.info(f"         Mode: {'retrieval only' if retrieval_only else 'full'}")
    logger.info("=" * 60)

    # ── Check RAGAS is installed ───────────────────────────────────────────────
    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
        )
        if not retrieval_only:
            from ragas.metrics import faithfulness, answer_correctness
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"RAGAS not installed: {e}")
        logger.error("Install with: pip install ragas datasets --break-system-packages")
        logger.error("For CPU-only: pip install ragas datasets sentence-transformers --break-system-packages")
        raise

    # ── Load data ──────────────────────────────────────────────────────────────
    if not PHASE1_PATH.exists():
        raise FileNotFoundError(f"Phase 1 data missing: {PHASE1_PATH}")

    with open(PHASE1_PATH) as f:
        phase1_data = json.load(f)

    phase2_lookup: dict[str, str] = {}
    if not retrieval_only:
        if not PHASE2_PATH.exists():
            raise FileNotFoundError(f"Phase 2 data missing: {PHASE2_PATH}")
        with open(PHASE2_PATH) as f:
            phase2_data = json.load(f)
        phase2_lookup = {d["id"]: d.get("answer", "") for d in phase2_data}

    # ── Build RAGAS Dataset ────────────────────────────────────────────────────
    rows = []
    for item in phase1_data:
        if not item.get("contexts"):
            logger.warning(f"  Skipping {item['id']}: no contexts retrieved")
            continue

        row = {
            "question":   item["question"],
            "contexts":   item["contexts"],
            "ground_truth": item["ground_truth"],
        }
        if not retrieval_only:
            answer = phase2_lookup.get(item["id"], "")
            if not answer:
                logger.warning(f"  Skipping {item['id']}: no answer generated")
                continue
            row["answer"] = answer

        rows.append(row)

    if not rows:
        raise ValueError("No valid rows for RAGAS evaluation. Check Phase 1 and 2 data.")

    logger.info(f"Evaluating {len(rows)} questions with RAGAS...")
    dataset = Dataset.from_list(rows)

    # ── Configure RAGAS LLM ────────────────────────────────────────────────────
    # Use Groq's faster 8B model as the RAGAS judge to save rate limit quota.
    # RAGAS supports any LangChain-compatible LLM via langchain_groq.
    ragas_llm, ragas_embeddings = _get_ragas_llm()

    metrics = [context_precision, context_recall]
    if not retrieval_only:
        metrics += [faithfulness, answer_correctness]

    # ── Run RAGAS ─────────────────────────────────────────────────────────────
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
        )
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        logger.error("If rate limit hit, try --phase 3 again in a minute.")
        raise

    # ── Parse results ──────────────────────────────────────────────────────────
    result_df = result.to_pandas()

    scores = {
        "context_precision": round(float(result_df["context_precision"].mean()), 4),
        "context_recall":    round(float(result_df["context_recall"].mean()), 4),
    }
    if not retrieval_only:
        scores["faithfulness"]       = round(float(result_df["faithfulness"].mean()), 4)
        scores["answer_correctness"] = round(float(result_df["answer_correctness"].mean()), 4)

    # Per-question scores
    per_question = result_df.to_dict(orient="records")

    # ── Identify worst-performing questions ────────────────────────────────────
    q_col = "user_input" if "user_input" in result_df.columns else "question"

    recall_cols = [c for c in [q_col, "context_recall"] if c in result_df.columns]
    worst_recall = result_df.nsmallest(3, "context_recall")[recall_cols].to_dict(orient="records") if "context_recall" in result_df.columns else []

    prec_cols = [c for c in [q_col, "context_precision"] if c in result_df.columns]
    worst_precision = result_df.nsmallest(3, "context_precision")[prec_cols].to_dict(orient="records") if "context_precision" in result_df.columns else []

    summary = {
        "timestamp":       datetime.now().isoformat(),
        "n_evaluated":     len(rows),
        "mode":            "retrieval_only" if retrieval_only else "full",
        "scores":          scores,
        "worst_recall":    worst_recall,
        "worst_precision": worst_precision,
        "per_question":    per_question,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ragas_path  = RESULTS_DIR / f"rag_ragas_results_{ts}.json"
    latest_path = RESULTS_DIR / "rag_summary_latest.json"

    with open(ragas_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    for metric, score in scores.items():
        bar   = "█" * int(score * 20)
        grade = "GOOD" if score >= 0.7 else ("OK" if score >= 0.5 else "NEEDS WORK")
        print(f"  {metric:<26} {score:.4f}  {bar}  [{grade}]")
    print()
    print("Worst context_recall questions:")
    for w in worst_recall:
        q_text = w.get("user_input", w.get("question", ""))
        print(f"  {w.get('context_recall', 0):.3f} — {str(q_text)[:70]}")
    print()
    print(f"Results saved to: {ragas_path}")
    print("=" * 60)

    return summary


def _get_ragas_llm():
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return llm, embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS — Complexity Router Evaluation (no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════════

def run_complexity_eval() -> dict:
    """
    Evaluate classify_query_complexity() on 15 test cases.
    Zero Groq calls — purely local inference.
    """
    logger.info("=" * 60)
    logger.info("Bonus: Complexity Router Evaluation (zero LLM calls)")
    logger.info("=" * 60)

    from rag.adaptive_router import classify_query_complexity

    complexity_path = Path(__file__).parent / "test_data" / "complexity_test_set.json"
    with open(complexity_path) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        t0         = time.perf_counter()
        predicted  = classify_query_complexity(case["query"])
        latency_ms = (time.perf_counter() - t0) * 1000
        correct    = predicted == case["expected_complexity"]
        results.append({
            "id":                  case["id"],
            "query":               case["query"][:80],
            "expected_complexity": case["expected_complexity"],
            "predicted":           predicted,
            "correct":             correct,
            "latency_ms":          round(latency_ms, 1),
            "note":                case.get("note", ""),
        })
        status = "✓" if correct else "✗"
        logger.info(f"  {status} {case['id']}: expected={case['expected_complexity']} "
                    f"predicted={predicted}")

    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc     = correct / total if total else 0

    wrong = [r for r in results if not r["correct"]]

    summary = {
        "timestamp":         datetime.now().isoformat(),
        "total":             total,
        "correct":           correct,
        "accuracy":          round(acc, 4),
        "accuracy_pct":      f"{acc*100:.1f}%",
        "wrong_predictions": wrong,
        "per_question":      results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"complexity_eval_{ts}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComplexity Router Accuracy: {acc*100:.1f}% ({correct}/{total})")
    if wrong:
        print("Wrong:")
        for w in wrong:
            print(f"  {w['id']}: expected={w['expected_complexity']} got={w['predicted']}")
            print(f"       {w['query']}")
    print(f"Saved to: {path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LexShield RAG evaluation")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="Run a specific phase only. Omit to run all phases."
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Phase 3: score only context_precision and context_recall (no generation needed)."
    )
    parser.add_argument(
        "--complexity", action="store_true",
        help="Also run complexity router evaluation (zero LLM calls)."
    )
    args = parser.parse_args()

    if args.phase == 1 or args.phase is None:
        run_phase1_retrieval()
    if args.phase == 2 or args.phase is None:
        run_phase2_generation()
    if args.phase == 3 or args.phase is None:
        run_phase3_ragas(retrieval_only=args.retrieval_only)
    if args.complexity:
        run_complexity_eval()