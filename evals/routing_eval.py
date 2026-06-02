"""
evals/routing_eval.py
=====================
Step 2 — Intent Classifier Routing Accuracy Evaluation
=======================================================

Evaluates LexShield's 8-intent classifier (agents/intent_classifier.py) on a
30-question test set with known correct intents.

Key design decisions from reading intent_classifier.py:
  - Hard regex overrides (RIGHTS_OVERRIDE, DRAFT_OVERRIDE, etc.) fire FIRST
    and skip the LLM call entirely — zero Groq cost for those cases.
  - LLM JSON-mode call (llama-3.3-70b) fires only when no override matches.
  - Keyword/pattern fallback fires if Groq call fails.

Rate limit strategy:
  - Override cases: 0 Groq calls (about 15/30 questions)
  - LLM cases: ~15 calls with BATCH_SLEEP seconds between batches
  - Total: ~15 Groq calls, comfortably within free tier RPM

Usage:
    cd C:\\Projects\\LexShield-AI
    python -m evals.routing_eval

Outputs:
    evals/results/routing_results.json   — per-question results
    evals/results/routing_summary.json  — accuracy metrics
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_DATA_PATH  = Path(__file__).parent / "test_data" / "intent_test_set.json"
RESULTS_DIR     = Path(__file__).parent / "results"
BATCH_SIZE      = 5     # Number of LLM queries per batch
BATCH_SLEEP     = 20    # Seconds between batches (Groq free tier: ~30 RPM)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RouteResult:
    id:                str
    query:             str
    expected_intent:   str
    predicted_intent:  str
    correct:           bool
    confidence:        float
    expected_override: Optional[str]
    actual_override:   Optional[str]   # which override fired (if any)
    used_llm:          bool            # True if Groq was called
    latency_ms:        float
    error:             Optional[str]
    detected_sections: list
    detected_acts:     list
    reasoning:         str


def _detect_which_override(classifier, query: str) -> Optional[str]:
    """Detect which hard override would fire, in priority order."""
    if classifier._RIGHTS_OVERRIDE.search(query):
        return "RIGHTS_OVERRIDE"
    if classifier._DRAFT_OVERRIDE.search(query):
        return "DRAFT_OVERRIDE"
    if classifier._TRANSLATION_OVERRIDE.search(query):
        return "TRANSLATION_OVERRIDE"
    if classifier._CASE_LAW_OVERRIDE.search(query):
        return "CASE_LAW_OVERRIDE"
    return None


def _result_from_intent(result, query: str, expected: dict,
                        actual_override: Optional[str], latency_ms: float,
                        used_llm: bool, error: Optional[str] = None) -> RouteResult:
    """Normalise both IntentResult and LLMIntentResult into RouteResult."""
    return RouteResult(
        id                = expected["id"],
        query             = query,
        expected_intent   = expected["expected_intent"],
        predicted_intent  = result.intent,
        correct           = result.intent == expected["expected_intent"],
        confidence        = round(float(result.confidence), 3),
        expected_override = expected.get("expected_override"),
        actual_override   = actual_override,
        used_llm          = used_llm,
        latency_ms        = round(latency_ms, 1),
        error             = error,
        detected_sections = getattr(result, "detected_sections", []),
        detected_acts     = getattr(result, "detected_acts", []),
        reasoning         = getattr(result, "reasoning", ""),
    )


def run_routing_eval(use_llm: bool = True) -> dict:
    """
    Run routing evaluation on all 30 test cases.

    Args:
        use_llm: If True, use classify_with_llm() (Groq calls for non-override
                 cases). If False, use the keyword/regex classify() only —
                 zero Groq calls, faster, useful for quick sanity checks.
    """
    from agents.intent_classifier import intent_classifier
    from rag.llm import llm  # MultiLLMRouter — used to get groq_client reference

    logger.info("=" * 60)
    logger.info("LexShield Routing Evaluation")
    logger.info(f"Mode: {'LLM + regex' if use_llm else 'Regex only'}")
    logger.info("=" * 60)

    # Load test set
    with open(TEST_DATA_PATH) as f:
        test_cases = json.load(f)
    logger.info(f"Loaded {len(test_cases)} test cases")

    # Get Groq client — only needed for LLM mode
    groq_client = None
    if use_llm:
        try:
            import groq as _groq
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                logger.warning("GROQ_API_KEY not set — falling back to regex-only mode")
                use_llm = False
            else:
                groq_client = _groq.Groq(api_key=api_key)
                logger.info("Groq client initialised")
        except ImportError:
            logger.warning("groq package not installed — falling back to regex-only mode")
            use_llm = False

    results: list[RouteResult] = []
    llm_batch_count = 0

    for i, case in enumerate(test_cases):
        query  = case["query"]
        actual_override = _detect_which_override(intent_classifier, query)
        will_use_llm    = use_llm and (actual_override is None) and (groq_client is not None)

        # Rate limit: sleep between LLM batches
        if will_use_llm and llm_batch_count > 0 and llm_batch_count % BATCH_SIZE == 0:
            logger.info(f"  [Rate limit] Sleeping {BATCH_SLEEP}s before next batch...")
            time.sleep(BATCH_SLEEP)

        logger.info(f"[{i+1:02d}/{len(test_cases)}] {case['id']} — {query[:70]}")
        logger.info(f"         Expected: {case['expected_intent']} | "
                    f"Override: {actual_override or 'none'} | LLM: {will_use_llm}")

        t0 = time.perf_counter()
        error = None
        try:
            if use_llm and groq_client is not None:
                result = intent_classifier.classify_with_llm(query, groq_client)
                if will_use_llm:  # actual LLM was called (no override)
                    llm_batch_count += 1
            else:
                result = intent_classifier.classify(query)
        except Exception as exc:
            # Fallback: keyword-only classify
            logger.warning(f"  Classification error: {exc} — using keyword fallback")
            error  = str(exc)
            result = intent_classifier.classify(query)
        latency_ms = (time.perf_counter() - t0) * 1000

        route_result = _result_from_intent(
            result, query, case, actual_override, latency_ms,
            used_llm=will_use_llm, error=error,
        )
        results.append(route_result)

        status = "✓" if route_result.correct else "✗"
        logger.info(f"         {status} Predicted: {route_result.predicted_intent} "
                    f"(conf={route_result.confidence:.2f}, {latency_ms:.0f}ms)")

    # ── Compute metrics ────────────────────────────────────────────────────────
    total        = len(results)
    correct      = sum(1 for r in results if r.correct)
    accuracy     = correct / total if total else 0.0

    override_results = [r for r in results if r.actual_override]
    llm_results      = [r for r in results if r.used_llm]
    keyword_results  = [r for r in results if not r.used_llm and not r.actual_override]

    override_acc = (
        sum(1 for r in override_results if r.correct) / len(override_results)
        if override_results else 0.0
    )
    llm_acc = (
        sum(1 for r in llm_results if r.correct) / len(llm_results)
        if llm_results else 0.0
    )
    keyword_acc = (
        sum(1 for r in keyword_results if r.correct) / len(keyword_results)
        if keyword_results else 0.0
    )

    # Per-intent breakdown
    intent_breakdown: dict[str, dict] = {}
    all_intents = set(r.expected_intent for r in results)
    for intent in sorted(all_intents):
        intent_cases = [r for r in results if r.expected_intent == intent]
        n_correct = sum(1 for r in intent_cases if r.correct)
        intent_breakdown[intent] = {
            "total":    len(intent_cases),
            "correct":  n_correct,
            "accuracy": round(n_correct / len(intent_cases), 3) if intent_cases else 0.0,
            "errors":   [r.query[:60] for r in intent_cases if not r.correct],
        }

    # Confusion — wrong predictions
    wrong = [r for r in results if not r.correct]
    confusion = [
        {
            "id":        r.id,
            "query":     r.query[:80],
            "expected":  r.expected_intent,
            "predicted": r.predicted_intent,
            "override":  r.actual_override,
            "reasoning": r.reasoning,
        }
        for r in wrong
    ]

    avg_latency = sum(r.latency_ms for r in results) / total if total else 0.0

    summary = {
        "timestamp":          datetime.now().isoformat(),
        "mode":               "llm+regex" if use_llm else "regex_only",
        "total_questions":    total,
        "correct":            correct,
        "accuracy":           round(accuracy, 4),
        "accuracy_pct":       f"{accuracy * 100:.1f}%",
        "by_tier": {
            "override_cases":   {"n": len(override_results), "accuracy": round(override_acc, 3)},
            "llm_cases":        {"n": len(llm_results),      "accuracy": round(llm_acc, 3)},
            "keyword_cases":    {"n": len(keyword_results),  "accuracy": round(keyword_acc, 3)},
        },
        "per_intent":         intent_breakdown,
        "avg_latency_ms":     round(avg_latency, 1),
        "total_llm_calls":    llm_batch_count,
        "wrong_predictions":  confusion,
    }

    # ── Save results ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path  = RESULTS_DIR / f"routing_results_{ts}.json"
    summary_path  = RESULTS_DIR / f"routing_summary_{ts}.json"
    latest_path   = RESULTS_DIR / "routing_summary_latest.json"

    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROUTING EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall accuracy:  {accuracy * 100:.1f}%  ({correct}/{total})")
    print(f"Override accuracy: {override_acc * 100:.1f}%  ({len(override_results)} cases)")
    print(f"LLM accuracy:      {llm_acc * 100:.1f}%  ({len(llm_results)} cases)")
    print(f"Keyword accuracy:  {keyword_acc * 100:.1f}%  ({len(keyword_results)} cases)")
    print(f"Avg latency:       {avg_latency:.0f}ms")
    print(f"Total LLM calls:   {llm_batch_count}")
    print()
    print("Per-intent accuracy:")
    for intent, data in intent_breakdown.items():
        bar = "█" * int(data["accuracy"] * 20)
        print(f"  {intent:<22} {data['accuracy']*100:5.1f}%  {bar}")
    if wrong:
        print(f"\nWrong predictions ({len(wrong)}):")
        for w in confusion:
            print(f"  [{w['id']}] Expected={w['expected']} → Got={w['predicted']}")
            print(f"        Query: {w['query']}")
            if w["reasoning"]:
                print(f"        Reason: {w['reasoning']}")
    print()
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LexShield routing evaluation")
    parser.add_argument(
        "--regex-only", action="store_true",
        help="Use keyword/regex classifier only (no Groq calls). Fast sanity check."
    )
    args = parser.parse_args()
    run_routing_eval(use_llm=not args.regex_only)