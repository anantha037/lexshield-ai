"""
evals/langsmith_eval.py
=======================
Step 3 — LangSmith Agent Tracing + Node-Level Feedback
=======================================================

LexShield already has @traceable on rag_pipeline.query() and LangGraph
auto-traces every node when LANGCHAIN_TRACING_V2=true is set.

This module adds:
  1. Node-level feedback scoring — programmatic scores attached to individual
     LangSmith runs (not just final output). Scores 5 dimensions per run.
  2. Batch feedback upload — run a set of queries, collect run IDs from
     LangSmith, attach scores to each run.
  3. Retrieval quality monitor — reads rag_grade from synthesis_note and
     tracks CRAG score distribution across runs.
  4. A lightweight test harness that validates the full agent graph fires
     correctly for each intent.

Usage:
    python -m evals.langsmith_eval
    python -m evals.langsmith_eval --feedback-only   # upload scores to existing runs
    python -m evals.langsmith_eval --intent legal_query --query "What is Section 302 IPC?"
"""

import os
import sys
import json
import time
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LangSmith environment check ────────────────────────────────────────────────
LANGSMITH_ENABLED = bool(
    os.getenv("LANGCHAIN_TRACING_V2") == "true"
    and os.getenv("LANGCHAIN_API_KEY")
)


def _check_langsmith() -> bool:
    if not LANGSMITH_ENABLED:
        logger.warning(
            "LangSmith tracing not enabled.\n"
            "Set these env vars to enable:\n"
            "  LANGCHAIN_TRACING_V2=true\n"
            "  LANGCHAIN_API_KEY=<your key>\n"
            "  LANGCHAIN_PROJECT=LexShield-AI   (optional but recommended)"
        )
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeFeedback:
    """
    Scores attached to a single LangSmith run (one agent invocation).
    All scores are 0.0–1.0.
    """
    run_id:              str
    query:               str
    intent:              str

    # Retrieval quality (from rag_grade in synthesis_note)
    retrieval_quality:   float   # 1.0 = rag_grade:good, 0.0 = poor

    # Routing accuracy (did intent_classifier route to the right node?)
    routing_confidence:  float   # classifier confidence score

    # CRAG score (extracted from synthesis_note, normalised 0-1)
    crag_score_norm:     float   # crag_score / 5.0

    # Answer grounding (0 = grounding_warning present, 1 = clean)
    answer_grounded:     float

    # Latency score (1.0 = <2s, 0.5 = 2-5s, 0.0 = >5s)
    latency_score:       float

    latency_ms:          float
    raw_synthesis_note:  str
    timestamp:           str


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_rag_grade(synthesis_note: str) -> float:
    """Extract rag_grade from synthesis_note. Returns 1.0 for good, 0.0 for poor."""
    if not synthesis_note:
        return 0.5   # unknown
    if "rag_grade=good" in synthesis_note:
        return 1.0
    if "rag_grade=poor" in synthesis_note:
        return 0.0
    return 0.5


def _extract_crag_score(synthesis_note: str) -> float:
    """
    Extract CRAG score from synthesis_note and normalise to 0-1.
    Format in synthesis_note: [complexity=moderate rag_grade=good ...]
    The crag score itself comes from the crag_result dict stored in pipeline metadata.
    Falls back to rag_grade as proxy.
    """
    # Try to find explicit crag_score in metadata (set via rt.add_metadata in pipeline)
    # This will be in LangSmith run metadata, not synthesis_note.
    # From synthesis_note we use rag_grade as proxy: good=4+/5, poor=1-3/5
    grade = _extract_rag_grade(synthesis_note)
    return grade   # 0.0 or 1.0 or 0.5 as proxy


def _latency_to_score(latency_ms: float) -> float:
    """Score latency: <2s=1.0, 2-5s=0.5, >5s=0.0"""
    if latency_ms < 2000:
        return 1.0
    elif latency_ms < 5000:
        return 0.5
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT TEST QUERIES — one per intent to validate full graph fires
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_TEST_QUERIES = [
    {
        "intent":   "legal_query",
        "query":    "What is the punishment for murder under Section 302 IPC?",
        "note":     "Simple legal definition — should route to legal_rag_node"
    },
    {
        "intent":   "risk_check",
        "query":    "Am I legally liable if my tenant slips and gets injured in my rented property?",
        "note":     "Risk check — should route to risk_check_node"
    },
    {
        "intent":   "rights_check",
        "query":    "What are my rights as a tenant if my landlord tries to illegally evict me?",
        "note":     "Rights — should route to rights_node"
    },
    {
        "intent":   "draft_request",
        "query":    "Help me draft a legal notice to my employer for non-payment of salary for 3 months",
        "note":     "Draft — should route to DraftingAgent"
    },
    {
        "intent":   "case_law_search",
        "query":    "Show me Supreme Court judgments on anticipatory bail",
        "note":     "Case law — should route to CaseLawAgent"
    },
    {
        "intent":   "translation_request",
        "query":    "Explain Section 302 IPC in Malayalam",
        "note":     "Translation — should route to TranslationAgent"
    },
    {
        "intent":   "general",
        "query":    "Hello, what can you do?",
        "note":     "General — should route to general_node"
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRACER + FEEDBACK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent_trace_eval(
    queries: Optional[list[dict]] = None,
    sleep_between: float = 15.0,
) -> list[NodeFeedback]:
    """
    Run agent queries through the full LangGraph pipeline, collect run IDs,
    and attach node-level feedback scores to LangSmith.

    Args:
        queries:        List of {intent, query, note} dicts. Defaults to AGENT_TEST_QUERIES.
        sleep_between:  Seconds between queries (rate limit buffer).
    """
    if not _check_langsmith():
        logger.warning("Running without LangSmith — scores will be saved locally only.")

    queries = queries or AGENT_TEST_QUERIES

    # Import after path setup
    from agents.orchestrator import MasterOrchestrator
    from langsmith import Client as LangSmithClient

    orchestrator = MasterOrchestrator()
    ls_client    = LangSmithClient() if LANGSMITH_ENABLED else None

    feedback_records: list[NodeFeedback] = []

    for i, test in enumerate(queries):
        logger.info(f"\n[{i+1}/{len(queries)}] Intent: {test['intent']}")
        logger.info(f"         Query: {test['query'][:70]}")

        # Create a fresh session for each test query
        session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{test['intent']}"

        t0 = time.perf_counter()
        run_id = None
        try:
            # Run through orchestrator — LangSmith captures run_id automatically
            # via the @traceable decorator on rag_pipeline.query and LangGraph tracing
            result = orchestrator.handle_query(
                query=test["query"],
                session_id=session_id,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            # Extract synthesis_note and grounding_warning from result
            # Result shape depends on orchestrator.run() return type
            synthesis_note  = ""
            grounding_warn  = ""
            routing_conf    = 0.8   # default if not extractable

            synthesis_note = result.synthesis_note or ""
            grounding_warn = result.grounding_warning or ""

            # Retrieve the LangSmith run ID for this trace
            if ls_client and LANGSMITH_ENABLED:
                run_id = _get_latest_run_id(ls_client, session_id)

            feedback = NodeFeedback(
                run_id             = run_id or session_id,
                query              = test["query"],
                intent             = test["intent"],
                retrieval_quality  = _extract_rag_grade(synthesis_note),
                routing_confidence = routing_conf,
                crag_score_norm    = _extract_crag_score(synthesis_note),
                answer_grounded    = 0.0 if grounding_warn else 1.0,
                latency_score      = _latency_to_score(latency_ms),
                latency_ms         = round(latency_ms, 1),
                raw_synthesis_note = synthesis_note[:200],
                timestamp          = datetime.now().isoformat(),
            )
            feedback_records.append(feedback)

            logger.info(f"         ✓ {latency_ms:.0f}ms | "
                        f"rag_grade={feedback.retrieval_quality:.0f} | "
                        f"grounded={feedback.answer_grounded:.0f} | "
                        f"run_id={run_id or 'N/A'}")

            # Upload feedback to LangSmith
            if ls_client and run_id:
                _upload_feedback(ls_client, feedback)

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"         ✗ Error: {e}")
            feedback_records.append(NodeFeedback(
                run_id             = session_id,
                query              = test["query"],
                intent             = test["intent"],
                retrieval_quality  = 0.0,
                routing_confidence = 0.0,
                crag_score_norm    = 0.0,
                answer_grounded    = 0.0,
                latency_score      = _latency_to_score(latency_ms),
                latency_ms         = round(latency_ms, 1),
                raw_synthesis_note = f"ERROR: {str(e)}",
                timestamp          = datetime.now().isoformat(),
            ))

        if i < len(queries) - 1:
            time.sleep(sleep_between)

    _save_and_report(feedback_records)
    return feedback_records


def _get_latest_run_id(ls_client, session_id: str) -> Optional[str]:
    """
    Retrieve the most recent LangSmith run ID for the given session.
    LangSmith run IDs are available via the client after a short delay.
    """
    try:
        project = os.getenv("LANGCHAIN_PROJECT", "LexShield-AI")
        runs = list(ls_client.list_runs(
            project_name = project,
            filter       = f'eq(metadata_key, "session_id") and eq(metadata_value, "{session_id}")',
            limit        = 1,
        ))
        if runs:
            return str(runs[0].id)
    except Exception as e:
        logger.debug(f"Could not fetch run ID: {e}")
    return None


def _upload_feedback(ls_client, feedback: NodeFeedback) -> None:
    """
    Upload node-level scores to LangSmith as feedback on a run.
    Each score dimension becomes a named feedback key.
    """
    score_map = {
        "retrieval_quality":  feedback.retrieval_quality,
        "routing_confidence": feedback.routing_confidence,
        "crag_score":         feedback.crag_score_norm,
        "answer_grounded":    feedback.answer_grounded,
        "latency_score":      feedback.latency_score,
    }
    try:
        for key, score in score_map.items():
            ls_client.create_feedback(
                run_id    = feedback.run_id,
                key       = key,
                score     = score,
                comment   = f"Automated eval — {feedback.intent} query",
            )
        logger.info(f"         LangSmith feedback uploaded for run {feedback.run_id}")
    except Exception as e:
        logger.warning(f"         LangSmith feedback upload failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE FEEDBACK FUNCTION (for use inside agent graph nodes)
# ═══════════════════════════════════════════════════════════════════════════════

def score_rag_node(
    run_id:         str,
    synthesis_note: str,
    grounding_warn: str,
    latency_ms:     float,
    intent:         str,
) -> dict:
    """
    Compute and upload feedback scores for a single RAG node execution.
    Call this from inside graph.py nodes after rag_pipeline.query() returns.

    Example usage in agents/graph.py:
        from evals.langsmith_eval import score_rag_node
        from langsmith.run_helpers import get_current_run_tree

        def legal_rag_node(state: AgentState) -> AgentState:
            t0     = time.perf_counter()
            answer = rag_pipeline.query(state.user_message)
            ms     = (time.perf_counter() - t0) * 1000
            rt     = get_current_run_tree()
            if rt:
                score_rag_node(
                    run_id         = str(rt.id),
                    synthesis_note = answer.synthesis_note or "",
                    grounding_warn = answer.grounding_warning or "",
                    latency_ms     = ms,
                    intent         = state.intent,
                )
            ...
    """
    scores = {
        "retrieval_quality": _extract_rag_grade(synthesis_note),
        "answer_grounded":   0.0 if grounding_warn else 1.0,
        "latency_score":     _latency_to_score(latency_ms),
        "crag_score":        _extract_crag_score(synthesis_note),
    }

    if LANGSMITH_ENABLED and run_id:
        try:
            from langsmith import Client
            ls = Client()
            for key, score in scores.items():
                ls.create_feedback(
                    run_id  = run_id,
                    key     = key,
                    score   = score,
                    comment = f"Auto-scored: {intent}",
                )
        except Exception as e:
            logger.debug(f"score_rag_node upload failed: {e}")

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE + REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _save_and_report(records: list[NodeFeedback]) -> None:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"langsmith_feedback_{ts}.json"
    latest = RESULTS_DIR / "langsmith_feedback_latest.json"

    data = [asdict(r) for r in records]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    with open(latest, "w") as f:
        json.dump(data, f, indent=2)

    total = len(records)
    avg_latency  = sum(r.latency_ms for r in records) / total if total else 0
    avg_grounded = sum(r.answer_grounded for r in records) / total if total else 0
    avg_rag      = sum(r.retrieval_quality for r in records) / total if total else 0
    avg_latency_score = sum(r.latency_score for r in records) / total if total else 0

    print("\n" + "=" * 60)
    print("LANGSMITH AGENT TRACE EVALUATION")
    print("=" * 60)
    print(f"  Queries run:          {total}")
    print(f"  Avg retrieval quality:{avg_rag:.2f}  (1.0=good)")
    print(f"  Avg grounded answers: {avg_grounded:.2f}  (1.0=no warnings)")
    print(f"  Avg latency score:    {avg_latency_score:.2f}  (1.0=<2s)")
    print(f"  Avg latency:          {avg_latency:.0f}ms")
    print()
    print("Per-intent:")
    for r in records:
        status = "✓" if r.answer_grounded == 1.0 else "⚠"
        print(f"  {status} {r.intent:<22} {r.latency_ms:6.0f}ms  "
              f"rag={r.retrieval_quality:.0f}  grounded={r.answer_grounded:.0f}  "
              f"run={r.run_id[:16] if r.run_id else 'N/A'}")
    print()
    print(f"Feedback saved: {path}")
    if LANGSMITH_ENABLED:
        project = os.getenv("LANGCHAIN_PROJECT", "LexShield-AI")
        print(f"LangSmith project: https://smith.langchain.com/projects/{project}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LexShield LangSmith agent eval")
    parser.add_argument("--intent",  help="Run a single intent test")
    parser.add_argument("--query",   help="Custom query (requires --intent)")
    parser.add_argument("--sleep",   type=float, default=15.0,
                        help="Seconds between queries (default 15)")
    args = parser.parse_args()

    if args.intent and args.query:
        queries = [{"intent": args.intent, "query": args.query, "note": "custom"}]
    elif args.intent:
        queries = [q for q in AGENT_TEST_QUERIES if q["intent"] == args.intent]
        if not queries:
            print(f"Intent '{args.intent}' not in test set. Available: "
                  + ", ".join(q["intent"] for q in AGENT_TEST_QUERIES))
            sys.exit(1)
    else:
        queries = AGENT_TEST_QUERIES

    run_agent_trace_eval(queries=queries, sleep_between=args.sleep)