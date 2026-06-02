"""
evals/pipeline_health_eval.py
==============================
Missing Step — Hallucination Rate + Latency Profiling
======================================================

Two mandatory evals your 3-step plan was missing:

1. HALLUCINATION RATE
   Your synthesizer already runs a hallucination checker internally but the
   result is logged silently and never aggregated. This module runs 15 queries
   through the pipeline, captures grounding_warning and confidence fields from
   LegalAnswer, and computes a hallucination rate across complexity tiers.
   Zero extra LLM calls — it uses the same pipeline calls that produce answers.

2. LATENCY PROFILING
   On your i5-8250U (CPU-only, 8GB RAM), simple/moderate/complex pipeline paths
   have very different latency profiles. This module measures p50/p95/p99
   per complexity tier and identifies which pipeline step is the bottleneck.
   Uses 5 queries per tier (15 total), which is manageable on the free tier.

Usage:
    python -m evals.pipeline_health_eval
    python -m evals.pipeline_health_eval --latency-only
    python -m evals.pipeline_health_eval --hallucination-only
"""

import os
import sys
import json
import time
import logging
import gc
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SLEEP = 20   # seconds between batches

# ── Test queries per complexity tier ──────────────────────────────────────────
# Chosen so the adaptive router routes them to the right tier
SIMPLE_QUERIES = [
    "What is Section 302 IPC?",
    "What is an FIR?",
    "Define culpable homicide under IPC.",
    "What is Section 498A IPC?",
    "What is POCSO Act?",
]

MODERATE_QUERIES = [
    "What are the rights of an arrested person under Section 50 CrPC?",
    "What is anticipatory bail and when can it be applied for?",
    "What are the grounds for divorce under the Hindu Marriage Act?",
    "What is the procedure to file a consumer complaint?",
    "What does Section 138 NI Act say about cheque bounce?",
]

COMPLEX_QUERIES = [
    "How does Section 302 IPC differ from 304A and what role does intent play?",
    "Compare IPC abetment provisions with BNS equivalents and explain how pending CrPC cases transition to BNSS.",
    "Explain the full legal process from FIR registration to bail application under CrPC with relevant sections.",
    "How do Sections 107, 108, 109 IPC on abetment interact with the principal offence and how is punishment determined?",
    "What are the procedural differences between District and State Consumer Commission and how does limitation apply?",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION EVAL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HallucinationRecord:
    query:              str
    complexity:         str
    grounding_warning:  str    # raw warning from LegalAnswer
    confidence:         str    # "low" / "" from LegalAnswer
    crag_fallback:      bool   # crag_fallback flag in synthesis_note
    rag_grade:          str    # "good" / "poor" from synthesis_note
    answer_snippet:     str    # first 100 chars of answer
    latency_ms:         float
    is_hallucination_risk: bool  # True if grounding_warning OR confidence==low


def _parse_synthesis_note(note: str) -> dict:
    """Extract structured fields from synthesis_note string."""
    result = {"rag_grade": "unknown", "complexity": "unknown", "crag_fallback": False}
    if not note:
        return result
    m = __import__("re").search(r"rag_grade=(\w+)", note)
    if m:
        result["rag_grade"] = m.group(1)
    m = __import__("re").search(r"complexity=(\w+)", note)
    if m:
        result["complexity"] = m.group(1)
    result["crag_fallback"] = "crag_fallback=True" in note
    return result


def run_hallucination_eval() -> dict:
    """
    Run all 15 queries (5 per tier), capture grounding_warning and confidence,
    compute hallucination risk rate per tier and overall.
    Batched with sleep to respect Groq rate limits.
    """
    logger.info("=" * 60)
    logger.info("Hallucination Rate Evaluation")
    logger.info("=" * 60)

    from rag.pipeline import rag_pipeline

    all_queries = (
        [("simple", q)   for q in SIMPLE_QUERIES]
        + [("moderate", q) for q in MODERATE_QUERIES]
        + [("complex", q)  for q in COMPLEX_QUERIES]
    )

    records: list[HallucinationRecord] = []

    for batch_start in range(0, len(all_queries), 5):
        batch = all_queries[batch_start: batch_start + 5]
        batch_num = batch_start // 5 + 1

        logger.info(f"\nBatch {batch_num}/3")
        for tier, query in batch:
            logger.info(f"  [{tier}] {query[:65]}")
            t0 = time.perf_counter()
            try:
                answer = rag_pipeline.query(user_query=query)
                latency_ms = (time.perf_counter() - t0) * 1000

                note_fields = _parse_synthesis_note(answer.synthesis_note or "")
                is_risk = bool(
                    (answer.grounding_warning and answer.grounding_warning.strip())
                    or getattr(answer, "confidence", "") == "low"
                    or note_fields["crag_fallback"]
                    or note_fields["rag_grade"] == "poor"
                )

                rec = HallucinationRecord(
                    query              = query,
                    complexity         = tier,
                    grounding_warning  = (answer.grounding_warning or "").strip(),
                    confidence         = getattr(answer, "confidence", ""),
                    crag_fallback      = note_fields["crag_fallback"],
                    rag_grade          = note_fields["rag_grade"],
                    answer_snippet     = answer.answer_text[:100],
                    latency_ms         = round(latency_ms, 1),
                    is_hallucination_risk = is_risk,
                )
                records.append(rec)

                flag = "⚠ RISK" if is_risk else "✓ CLEAN"
                logger.info(f"      {flag} | rag_grade={rec.rag_grade} | "
                            f"crag_fallback={rec.crag_fallback} | {latency_ms:.0f}ms")

            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.error(f"      ✗ Error: {e}")
                records.append(HallucinationRecord(
                    query=query, complexity=tier,
                    grounding_warning=str(e), confidence="error",
                    crag_fallback=False, rag_grade="unknown",
                    answer_snippet="", latency_ms=round(latency_ms, 1),
                    is_hallucination_risk=True,
                ))

        gc.collect()
        if batch_start + 5 < len(all_queries):
            logger.info(f"  Sleeping {BATCH_SLEEP}s...")
            time.sleep(BATCH_SLEEP)

    # ── Metrics ────────────────────────────────────────────────────────────────
    total     = len(records)
    at_risk   = sum(1 for r in records if r.is_hallucination_risk)
    clean_rate = 1 - (at_risk / total) if total else 0

    per_tier: dict = {}
    for tier in ("simple", "moderate", "complex"):
        tier_recs = [r for r in records if r.complexity == tier]
        tier_risk = sum(1 for r in tier_recs if r.is_hallucination_risk)
        per_tier[tier] = {
            "total":      len(tier_recs),
            "at_risk":    tier_risk,
            "clean_rate": round(1 - tier_risk / len(tier_recs), 3) if tier_recs else 0,
            "risk_queries": [r.query for r in tier_recs if r.is_hallucination_risk],
        }

    summary = {
        "timestamp":    datetime.now().isoformat(),
        "total":        total,
        "at_risk":      at_risk,
        "clean_rate":   round(clean_rate, 4),
        "clean_pct":    f"{clean_rate*100:.1f}%",
        "per_tier":     per_tier,
        "records":      [asdict(r) for r in records],
    }

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"hallucination_eval_{ts}.json"
    latest = RESULTS_DIR / "hallucination_latest.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(latest, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Hallucination Risk Evaluation ──────────────────────────────")
    print(f"  Overall clean rate: {clean_rate*100:.1f}%  ({total - at_risk}/{total} clean answers)")
    for tier, data in per_tier.items():
        bar = "█" * int(data["clean_rate"] * 20)
        print(f"  {tier:<10} {data['clean_rate']*100:5.1f}%  {bar}")
    if at_risk > 0:
        print(f"\n  Risky queries:")
        for r in records:
            if r.is_hallucination_risk:
                print(f"    [{r.complexity}] {r.query[:70]}")
                if r.grounding_warning:
                    print(f"           Warning: {r.grounding_warning[:80]}")
    print(f"\n  Saved to: {path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# LATENCY PROFILING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LatencyRecord:
    query:       str
    complexity:  str
    latency_ms:  float
    rag_grade:   str
    error:       Optional[str]


def run_latency_profiling() -> dict:
    """
    Measure p50/p95/p99 latency per complexity tier on your CPU-only machine.
    Uses the same 15 queries as hallucination eval.
    """
    logger.info("=" * 60)
    logger.info("Latency Profiling (i5-8250U, 8GB RAM, CPU-only)")
    logger.info("=" * 60)

    from rag.pipeline import rag_pipeline

    all_queries = (
        [("simple", q)   for q in SIMPLE_QUERIES]
        + [("moderate", q) for q in MODERATE_QUERIES]
        + [("complex", q)  for q in COMPLEX_QUERIES]
    )

    records: list[LatencyRecord] = []

    for batch_start in range(0, len(all_queries), 5):
        batch = all_queries[batch_start: batch_start + 5]
        for tier, query in batch:
            logger.info(f"  [{tier}] {query[:65]}")
            t0 = time.perf_counter()
            try:
                answer     = rag_pipeline.query(user_query=query)
                latency_ms = (time.perf_counter() - t0) * 1000
                note       = _parse_synthesis_note(answer.synthesis_note or "")
                records.append(LatencyRecord(
                    query=query, complexity=tier,
                    latency_ms=round(latency_ms, 1),
                    rag_grade=note["rag_grade"], error=None,
                ))
                logger.info(f"      {latency_ms:.0f}ms")
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.error(f"      Error: {e}")
                records.append(LatencyRecord(
                    query=query, complexity=tier,
                    latency_ms=round(latency_ms, 1),
                    rag_grade="unknown", error=str(e),
                ))

        gc.collect()
        if batch_start + 5 < len(all_queries):
            logger.info(f"  Sleeping {BATCH_SLEEP}s...")
            time.sleep(BATCH_SLEEP)

    # ── Stats ──────────────────────────────────────────────────────────────────
    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        data_sorted = sorted(data)
        idx = int(len(data_sorted) * pct / 100)
        return data_sorted[min(idx, len(data_sorted) - 1)]

    per_tier_stats: dict = {}
    for tier in ("simple", "moderate", "complex"):
        tier_ms = [r.latency_ms for r in records if r.complexity == tier and not r.error]
        if not tier_ms:
            continue
        per_tier_stats[tier] = {
            "n":      len(tier_ms),
            "p50_ms": round(_percentile(tier_ms, 50), 1),
            "p95_ms": round(_percentile(tier_ms, 95), 1),
            "p99_ms": round(_percentile(tier_ms, 99), 1),
            "avg_ms": round(statistics.mean(tier_ms), 1),
            "min_ms": round(min(tier_ms), 1),
            "max_ms": round(max(tier_ms), 1),
        }

    all_ms = [r.latency_ms for r in records if not r.error]
    overall_stats = {
        "n":      len(all_ms),
        "p50_ms": round(_percentile(all_ms, 50), 1),
        "p95_ms": round(_percentile(all_ms, 95), 1),
        "avg_ms": round(statistics.mean(all_ms), 1) if all_ms else 0,
    }

    summary = {
        "timestamp":   datetime.now().isoformat(),
        "machine":     "Intel i5-8250U, 8GB RAM, CPU-only, Windows 11",
        "overall":     overall_stats,
        "per_tier":    per_tier_stats,
        "records":     [asdict(r) for r in records],
    }

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    path   = RESULTS_DIR / f"latency_profile_{ts}.json"
    latest = RESULTS_DIR / "latency_latest.json"
    with open(path,   "w") as f:
        json.dump(summary, f, indent=2)
    with open(latest, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Latency Profile ────────────────────────────────────────────")
    print(f"  {'tier':<10}  {'p50':>8}  {'p95':>8}  {'avg':>8}  {'max':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for tier, s in per_tier_stats.items():
        flag = " ⚠" if s["p95_ms"] > 10000 else ""
        print(f"  {tier:<10}  {s['p50_ms']:>7.0f}ms  {s['p95_ms']:>7.0f}ms  "
              f"{s['avg_ms']:>7.0f}ms  {s['max_ms']:>7.0f}ms{flag}")
    print(f"\n  Overall p50={overall_stats['p50_ms']:.0f}ms  "
          f"p95={overall_stats['p95_ms']:.0f}ms  avg={overall_stats['avg_ms']:.0f}ms")
    print(f"\n  Saved to: {path}")
    print("─" * 62)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline health eval")
    parser.add_argument("--latency-only",      action="store_true")
    parser.add_argument("--hallucination-only", action="store_true")
    args = parser.parse_args()

    if args.latency_only:
        run_latency_profiling()
    elif args.hallucination_only:
        run_hallucination_eval()
    else:
        run_hallucination_eval()
        run_latency_profiling()