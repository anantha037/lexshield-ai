"""
evals/run_all.py
================
Unified Eval Runner
===================
Runs all LexShield evaluations and produces a single summary report.

Run order (each can be skipped with flags):
  1. routing_eval     — intent classifier accuracy (Step 2)
  2. rag_eval Phase 1 — retrieval only, zero LLM calls
  3. pipeline_health  — hallucination rate + latency profiling
  4. rag_eval Phase 2 — generation (batched, rate-limit safe)
  5. rag_eval Phase 3 — RAGAS scoring
  6. langsmith_eval   — agent tracing + node-level feedback (Step 3)
  7. complexity_eval  — adaptive router accuracy (bonus, zero LLM)

The order is intentional:
  - Fast/zero-cost evals run first so you get signal immediately.
  - Heavy LLM evals run later with sleep buffers between them.
  - If any eval crashes, others still run and the summary reflects what succeeded.

Usage:
    # Full run
    cd C:\\Projects\\LexShield-AI
    python -m evals.run_all

    # Quick run (no Groq calls at all — routing regex-only + retrieval + latency)
    python -m evals.run_all --quick

    # Skip specific steps
    python -m evals.run_all --skip-ragas --skip-langsmith

    # Only routing eval
    python -m evals.run_all --only routing
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            Path(__file__).parent / "results" / "run_all.log",
            encoding = "utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INTER_EVAL_SLEEP = 30   # seconds between major evals (Groq cooldown)


def _run_step(name: str, fn, results: dict, *args, **kwargs) -> bool:
    """Run a single eval step, catch all errors, record outcome."""
    logger.info("\n" + "█" * 60)
    logger.info(f"  STEP: {name}")
    logger.info("█" * 60)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - t0)
        results[name] = {
            "status":     "ok",
            "elapsed_s":  round(elapsed, 1),
            "summary":    _extract_key_metric(name, result),
        }
        logger.info(f"  ✓ {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = (time.perf_counter() - t0)
        results[name] = {
            "status":    "error",
            "elapsed_s": round(elapsed, 1),
            "error":     str(e),
            "traceback": traceback.format_exc()[-800:],
        }
        logger.error(f"  ✗ {name} failed: {e}")
        return False


def _extract_key_metric(name: str, result) -> str:
    """Pull the single most important number from each eval result."""
    if not isinstance(result, dict):
        return str(result)[:80]
    try:
        if "accuracy_pct" in result:
            return f"accuracy={result['accuracy_pct']}"
        if "scores" in result:
            s = result["scores"]
            parts = [f"{k}={v:.3f}" for k, v in s.items()]
            return "  ".join(parts)
        if "clean_pct" in result:
            return f"clean_rate={result['clean_pct']}"
        if "overall" in result and "p50_ms" in result.get("overall", {}):
            o = result["overall"]
            return f"p50={o['p50_ms']}ms  p95={o['p95_ms']}ms"
    except Exception:
        pass
    return str(result)[:80]


def run_all(
    quick:           bool = False,
    skip_ragas:      bool = False,
    skip_langsmith:  bool = False,
    skip_health:     bool = False,
    only:            Optional[str] = None,
    regex_only:      bool = False,
) -> dict:
    """
    Run all evals. Returns a summary dict of all results.

    Args:
        quick:          Zero Groq calls. Routing regex-only + Phase1 + latency only.
        skip_ragas:     Skip RAG Phase 2+3 (skip Groq generation + RAGAS scoring).
        skip_langsmith: Skip LangSmith agent tracing eval.
        skip_health:    Skip hallucination + latency profiling.
        only:           Run only this eval name ('routing', 'rag', 'health', 'langsmith').
        regex_only:     For routing eval: use keyword/regex only (no Groq calls).
    """
    logger.info("=" * 60)
    logger.info("LexShield Full Evaluation Suite")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Mode: {'quick' if quick else 'full'}")
    logger.info("=" * 60)

    results: dict = {}
    start_time = time.perf_counter()

    from evals.routing_eval          import run_routing_eval
    from evals.rag_eval              import (
        run_phase1_retrieval,
        run_phase2_generation,
        run_phase3_ragas,
        run_complexity_eval,
    )
    from evals.pipeline_health_eval  import run_hallucination_eval, run_latency_profiling
    from evals.langsmith_eval        import run_agent_trace_eval

    # ── Step 1: Routing eval ───────────────────────────────────────────────────
    if only is None or only == "routing":
        use_llm = not (quick or regex_only)
        _run_step("routing_eval", run_routing_eval, results, use_llm=use_llm)
        if not quick:
            time.sleep(10)

    # ── Step 2: Complexity routing eval (zero LLM) ─────────────────────────────
    if only is None or only == "complexity":
        _run_step("complexity_eval", run_complexity_eval, results)

    # ── Step 3: RAG Phase 1 (retrieval only, zero LLM) ────────────────────────
    if only is None or only == "rag":
        _run_step("rag_phase1_retrieval", run_phase1_retrieval, results)

    # ── Step 4: Hallucination + latency (uses pipeline, batched) ───────────────
    if not skip_health and (only is None or only == "health"):
        if not quick:
            logger.info(f"Sleeping {INTER_EVAL_SLEEP}s before health eval...")
            time.sleep(INTER_EVAL_SLEEP)
        _run_step("hallucination_eval", run_hallucination_eval, results)
        _run_step("latency_profiling",  run_latency_profiling,  results)

    # ── Step 5: RAG Phase 2 (generation, batched) ─────────────────────────────
    if not skip_ragas and not quick and (only is None or only == "rag"):
        logger.info(f"Sleeping {INTER_EVAL_SLEEP}s before generation phase...")
        time.sleep(INTER_EVAL_SLEEP)
        _run_step("rag_phase2_generation", run_phase2_generation, results)

    # ── Step 6: RAGAS scoring ─────────────────────────────────────────────────
    if not skip_ragas and not quick and (only is None or only == "rag"):
        logger.info(f"Sleeping {INTER_EVAL_SLEEP}s before RAGAS scoring...")
        time.sleep(INTER_EVAL_SLEEP)
        _run_step("rag_phase3_ragas", run_phase3_ragas, results,
                  retrieval_only=False)

    # ── Step 7: LangSmith agent tracing ───────────────────────────────────────
    if not skip_langsmith and not quick and (only is None or only == "langsmith"):
        logger.info(f"Sleeping {INTER_EVAL_SLEEP}s before LangSmith eval...")
        time.sleep(INTER_EVAL_SLEEP)
        _run_step("langsmith_eval", run_agent_trace_eval, results)

    # ── Final summary report ──────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - start_time
    n_ok    = sum(1 for v in results.values() if v["status"] == "ok")
    n_err   = sum(1 for v in results.values() if v["status"] == "error")

    final_summary = {
        "timestamp":     datetime.now().isoformat(),
        "total_elapsed": round(total_elapsed, 1),
        "steps_ok":      n_ok,
        "steps_failed":  n_err,
        "results":       results,
    }

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"eval_suite_{ts}.json"
    with open(path, "w") as f:
        json.dump(final_summary, f, indent=2)

    # ── Print master summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEXSHIELD EVAL SUITE — FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total time:    {total_elapsed:.0f}s  "
          f"({n_ok} ok, {n_err} failed)")
    print()
    for step, data in results.items():
        icon = "✓" if data["status"] == "ok" else "✗"
        metric = data.get("summary", data.get("error", "")[:50])
        print(f"  {icon} {step:<30} {metric}")
    print()
    print(f"  Full report: {path}")
    print("=" * 60)

    return final_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LexShield full eval suite")
    parser.add_argument(
        "--quick", action="store_true",
        help="Zero Groq calls: routing regex-only + Phase 1 retrieval + latency only"
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAG generation (Phase 2) and RAGAS scoring (Phase 3)"
    )
    parser.add_argument(
        "--skip-langsmith", action="store_true",
        help="Skip LangSmith agent tracing eval"
    )
    parser.add_argument(
        "--skip-health", action="store_true",
        help="Skip hallucination + latency profiling"
    )
    parser.add_argument(
        "--only", choices=["routing", "rag", "health", "langsmith", "complexity"],
        help="Run only one eval module"
    )
    parser.add_argument(
        "--regex-only", action="store_true",
        help="Routing eval: use keyword/regex only (no Groq LLM calls)"
    )
    args = parser.parse_args()

    run_all(
        quick          = args.quick,
        skip_ragas     = args.skip_ragas,
        skip_langsmith = args.skip_langsmith,
        skip_health    = args.skip_health,
        only           = args.only,
        regex_only     = args.regex_only,
    )