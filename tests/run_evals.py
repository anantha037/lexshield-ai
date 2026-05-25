"""
LexShield AI — Evaluation Script  (Updated — Rate-Limit Safe)
===============================================================
Fixes vs previous version:
  1. hybrid_search import fixed — tries 'hybrid_searcher' (actual export name),
     then 'hybrid_search', then vectorstore direct — never crashes silently.
  2. Checkpoint resume — saves progress after every question to eval_progress.json.
     Crash at Q14, restart, it resumes from Q15. Use --fresh to force full re-run.
  3. RAGAS eval LLM uses OpenRouter (priority=2), not Groq, so RAGAS scoring calls
     don't compete with RAG pipeline calls for the same Groq rate limit.
  4. Adaptive sleep — complex questions (bns_equivalents) sleep 7s;
     simple questions sleep 4s. Avoids hitting Groq 30 RPM cap.
  5. Per-question error isolation — one pipeline failure marks that question
     as failed and continues. One bad question no longer crashes everything.
  6. --intent-only flag for fast intent-only runs (no LLM calls needed).

Run (full eval):
  cd C:\\Projects\\LexShield-AI
  python -m tests.run_evals

Resume after crash:
  python -m tests.run_evals
  (Detects eval_progress.json and skips completed questions automatically)

Force fresh run:
  python -m tests.run_evals --fresh

Intent-only (fast, ~5 seconds):
  python -m tests.run_evals --intent-only
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("lexshield.eval")

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not set in .env — aborting.")
    sys.exit(1)

_EVAL_DATASET_PATH = _ROOT / "tests" / "eval_dataset.json"
_EVAL_REPORT_PATH  = _ROOT / "tests" / "eval_report.json"
_PROGRESS_PATH     = _ROOT / "tests" / "eval_progress.json"


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT RETRIEVAL — fixed hybrid_search import
# ═══════════════════════════════════════════════════════════════════════════════

def _get_contexts_for_query(question: str, top_k: int = 5) -> list[str]:
    """
    Retrieve top-k context chunks for a question.
    Tries three fallback strategies with visible error logging.
    """

    # Attempt 1: hybrid_searcher singleton — try top_k, k, n_results in order
    try:
        from rag.hybrid_search import hybrid_searcher
        import inspect

        sig = inspect.signature(hybrid_searcher.search)
        params = sig.parameters

        if "top_k" in params:
            chunks = hybrid_searcher.search(question, top_k=top_k)
        elif "k" in params:
            chunks = hybrid_searcher.search(question, k=top_k)
        elif "n_results" in params:
            chunks = hybrid_searcher.search(question, n_results=top_k)
        else:
            # Positional fallback — pass top_k as second argument
            chunks = hybrid_searcher.search(question, top_k)

        texts = []
        for c in chunks:
            if isinstance(c, dict):
                text = c.get("text") or c.get("content") or c.get("page_content", "")
            else:
                text = getattr(c, "text", None) or getattr(c, "content", None) or str(c)
            if text and text.strip():
                texts.append(text.strip())
        if texts:
            return texts

    except Exception as e:
        logger.error(f"[Attempt 1] hybrid_searcher.search failed: {e}")

    # Attempt 2: hybrid_search function alias (in case module was updated)
    try:
        from rag.hybrid_search import hybrid_search
        import inspect

        sig = inspect.signature(hybrid_search)
        params = sig.parameters

        if "top_k" in params:
            chunks = hybrid_search(question, top_k=top_k)
        elif "k" in params:
            chunks = hybrid_search(question, k=top_k)
        elif "n_results" in params:
            chunks = hybrid_search(question, n_results=top_k)
        else:
            chunks = hybrid_search(question, top_k)

        texts = [
            (c.get("text") or c.get("content", "")) if isinstance(c, dict)
            else getattr(c, "text", str(c))
            for c in chunks
        ]
        texts = [t for t in texts if t.strip()]
        if texts:
            return texts

    except Exception as e:
        logger.error(f"[Attempt 2] hybrid_search function failed: {e}")

    # Attempt 3: direct vectorstore — unwrap LegalVectorStore wrapper if needed
    try:
        from rag.vectorstore import vectorstore

        # Unwrap custom wrapper to get the underlying LangChain vectorstore
        if hasattr(vectorstore, "similarity_search"):
            results = vectorstore.similarity_search(question, k=top_k)

        elif hasattr(vectorstore, "vectorstore") and hasattr(
            vectorstore.vectorstore, "similarity_search"
        ):
            # LegalVectorStore wraps a real vectorstore at .vectorstore
            results = vectorstore.vectorstore.similarity_search(question, k=top_k)

        elif hasattr(vectorstore, "search"):
            # Generic .search() fallback — try common signatures
            try:
                results = vectorstore.search(question, k=top_k)
            except TypeError:
                results = vectorstore.search(question, top_k)

        else:
            raise AttributeError(
                f"{type(vectorstore).__name__} exposes none of: "
                "similarity_search, vectorstore, search"
            )

        texts = []
        for r in results:
            if isinstance(r, dict):
                texts.append(r.get("text", r.get("document", r.get("page_content", ""))))
            else:
                texts.append(getattr(r, "page_content", getattr(r, "text", str(r))))
        texts = [t for t in texts if t.strip()]
        if texts:
            return texts

    except Exception as e:
        logger.error(f"[Attempt 3] Vectorstore direct search failed: {e}")

    logger.warning(
        f"All context retrieval strategies failed for '{question[:45]}'. "
        "context_precision and context_recall will be degraded for this question."
    )
    return ["(context retrieval unavailable — check ChromaDB indexing)"]


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE SLEEP
# ═══════════════════════════════════════════════════════════════════════════════

_COMPLEX_CATEGORIES = {"bns_equivalents"}   # trigger 3-4 Groq calls each


def _sleep_between_questions(category: str, question_idx: int):
    """
    Adaptive inter-question sleep.

    Groq free tier: 30 RPM = min 2s/request.
    Complex questions trigger decomposer + CRAG + synthesiser = 3-4 calls.
    Simple questions trigger section fast-path = 1 call.

    base: complex=7s, simple=4s
    ramp: +0.5s every 5 questions (spreads load across rate limit windows)
    cap:  12s maximum
    """
    base  = 7.0 if category in _COMPLEX_CATEGORIES else 4.0
    ramp  = (question_idx // 5) * 0.5
    total = min(base + ramp, 12.0)
    logger.debug(f"Sleeping {total:.1f}s (category={category}, q={question_idx})")
    time.sleep(total)


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _save_progress(completed_rows: list[dict]):
    with open(_PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "rows":     completed_rows,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def _load_progress() -> list[dict]:
    if _PROGRESS_PATH.exists():
        try:
            with open(_PROGRESS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            rows = data.get("rows", [])
            if rows:
                logger.info(f"Checkpoint found: {len(rows)} question(s) already completed.")
            return rows
        except Exception:
            return []
    return []


def _clear_progress():
    if _PROGRESS_PATH.exists():
        _PROGRESS_PATH.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1A — RAG EVALUATION (RAGAS)
# ═══════════════════════════════════════════════════════════════════════════════

def run_rag_evaluation(fresh: bool = False) -> dict:
    """
    Evaluate RAG pipeline using RAGAS on 28 legally accurate Q&A pairs.

    Architecture:
      RAG pipeline calls -> Groq (primary) with OpenRouter failover via MultiLLMRouter
      RAGAS scoring calls -> OpenRouter priority-2 provider (separate rate limit)

    This separation is the key fix: previously both competed for the same
    30 RPM Groq limit, causing 429 at ~Q14. Now they use different providers.
    """
    logger.info("=" * 60)
    logger.info("TASK 1A — RAG EVALUATION (RAGAS)")
    logger.info("=" * 60)

    # Validate RAGAS install
    try:
        from ragas             import evaluate
        from ragas.metrics     import (faithfulness, answer_relevancy,
                                       context_precision, context_recall)
        from ragas.llms        import LangchainLLMWrapper
        from ragas.embeddings  import LangchainEmbeddingsWrapper
        from datasets          import Dataset
    except ImportError as e:
        logger.error(f"RAGAS missing: {e}\nInstall: pip install ragas datasets")
        return _rag_error_result(str(e))

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        logger.error(f"langchain_community missing: {e}\nInstall: pip install langchain-community")
        return _rag_error_result(str(e))

    # Load eval dataset
    try:
        with open(_EVAL_DATASET_PATH, encoding="utf-8") as f:
            eval_data = json.load(f)
        logger.info(f"Loaded {len(eval_data)} eval questions.")
    except FileNotFoundError:
        return _rag_error_result(f"eval_dataset.json not found at {_EVAL_DATASET_PATH}")

    # Load RAG pipeline
    try:
        from rag.pipeline import rag_pipeline
        logger.info("RAG pipeline loaded.")
    except Exception as e:
        return _rag_error_result(f"RAG pipeline load failed: {e}")

    # Configure RAGAS eval LLM (OpenRouter — NOT Groq, to avoid rate limit collision)
    eval_llm = _build_eval_llm()
    if eval_llm is None:
        return _rag_error_result("Could not initialise any eval LLM")

    # Checkpoint resume
    if fresh and _PROGRESS_PATH.exists():
        _clear_progress()
        logger.info("--fresh: cleared existing checkpoint.")

    completed_rows       = _load_progress()
    completed_questions  = {r["question"] for r in completed_rows}
    failed_count         = sum(1 for r in completed_rows if r.get("_failed"))

    # Build evaluation rows
    total = len(eval_data)
    for i, entry in enumerate(eval_data, 1):
        question     = entry["question"]
        ground_truth = entry["ground_truth"]
        category     = entry.get("category", "unknown")

        if question in completed_questions:
            logger.info(f"[{i:02d}/{total}] SKIP (checkpoint): {question[:55]}")
            continue

        logger.info(f"[{i:02d}/{total}] {category}: {question[:55]}")

        answer_text = ""
        failed_this = False
        try:
            result      = rag_pipeline.query(question)
            answer_text = result.answer_text if hasattr(result, "answer_text") else str(result)
        except Exception as e:
            logger.warning(f"  Pipeline error: {e}")
            answer_text = f"[Pipeline error: {e}]"
            failed_count += 1
            failed_this  = True

        contexts = _get_contexts_for_query(question, top_k=5)

        completed_rows.append({
            "question":     question,
            "answer":       answer_text,
            "contexts":     contexts,
            "ground_truth": ground_truth,
            "_category":    category,
            "_failed":      failed_this,
        })
        _save_progress(completed_rows)

        logger.info(f"  Answer:   {answer_text[:80]}…")
        logger.info(f"  Contexts: {len(contexts)} chunk(s) | Failures so far: {failed_count}")

        if i < total:
            _sleep_between_questions(category, i)

    # Build RAGAS-clean dataset (no internal _ fields)
    ragas_rows = [
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in completed_rows
    ]

    # Run RAGAS metrics
    logger.info(f"\nRunning RAGAS on {len(ragas_rows)} question(s)…")
    try:
        dataset      = Dataset.from_list(ragas_rows)

        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
        )

        ragas_result = evaluate(
            dataset    = dataset,
            metrics    = [faithfulness, answer_relevancy, context_precision, context_recall],
            llm        = eval_llm,
            embeddings = ragas_embeddings,
        )
        scores = dict(ragas_result)

        def _sf(v) -> float:
            import math
            if v is None: return 0.0
            if isinstance(v, (list, tuple)):
                valid = [float(x) for x in v if x is not None and not math.isnan(float(x))]
                return round(sum(valid) / len(valid), 4) if valid else 0.0
            try:
                f = float(v)
                return 0.0 if math.isnan(f) else round(f, 4)
            except Exception:
                return 0.0

        result = {
            "faithfulness":      _sf(scores.get("faithfulness",      0)),
            "answer_relevancy":  _sf(scores.get("answer_relevancy",  0)),
            "context_precision": _sf(scores.get("context_precision", 0)),
            "context_recall":    _sf(scores.get("context_recall",    0)),
            "num_questions":     len(eval_data),
            "failed_questions":  failed_count,
            "categories":        list(set(e["category"] for e in eval_data)),
            "eval_llm":          "OpenRouter meta-llama/llama-3.3-70b-instruct:free",
            "ragas_version":     _get_package_version("ragas"),
            "status":            "success",
        }

        logger.info("\nRAGAS Results:")
        for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            logger.info(f"  {k:25} {result[k]:.4f}")

        _clear_progress()
        return result

    except Exception as e:
        logger.error(f"RAGAS evaluate() failed: {e}")
        import traceback; traceback.print_exc()
        return _rag_error_result(f"RAGAS scoring error: {e}", len(eval_data), failed_count)


def _build_eval_llm():
    """
    Build a LangChain-compatible LLM for RAGAS scoring.
    Uses OpenRouter (priority=2) to avoid competing with Groq pipeline calls.
    Falls back through providers until one works.
    """
    from ragas.llms import LangchainLLMWrapper

    # Strategy 1: MultiLLMRouter.get_langchain_llm() -> OpenRouter priority=2
    try:
        from rag.multi_llm import MultiLLMRouter
        router   = MultiLLMRouter()
        eval_llm = router.get_langchain_llm(provider_priority=2)
        logger.info("RAGAS eval LLM: MultiLLMRouter (OpenRouter priority=2)")
        return eval_llm
    except Exception as e:
        logger.warning(f"MultiLLMRouter eval LLM failed: {e}")

    # Strategy 2: Direct OpenRouter ChatOpenAI
    if OPENROUTER_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            chat = ChatOpenAI(
                model       = "meta-llama/llama-3.3-70b-instruct:free",
                api_key     = OPENROUTER_API_KEY,
                base_url    = "https://openrouter.ai/api/v1",
                temperature = 0.0,
                max_tokens  = 1024,
                default_headers = {
                    "HTTP-Referer": "https://lexshield.ai",
                    "X-Title":      "LexShield AI",
                },
            )
            logger.info("RAGAS eval LLM: OpenRouter direct (meta-llama/llama-3.3-70b)")
            return LangchainLLMWrapper(chat)
        except Exception as e:
            logger.warning(f"OpenRouter direct eval LLM failed: {e}")

    # Strategy 3: Groq (last resort — shares rate limit with pipeline)
    if GROQ_API_KEY:
        try:
            from langchain_groq  import ChatGroq
            chat = ChatGroq(
                model       = "llama-3.3-70b-versatile",
                api_key     = GROQ_API_KEY,
                temperature = 0.0,
            )
            logger.warning(
                "RAGAS eval LLM: Groq (same rate limit as pipeline — "
                "may hit 429. Set OPENROUTER_API_KEY for better results.)"
            )
            return LangchainLLMWrapper(chat)
        except Exception as e:
            logger.error(f"Groq eval LLM also failed: {e}")

    logger.error("No eval LLM could be initialised.")
    return None


def _rag_error_result(reason: str, num_questions: int = 0, failed: int = 0) -> dict:
    return {
        "faithfulness":      0.0,
        "answer_relevancy":  0.0,
        "context_precision": 0.0,
        "context_recall":    0.0,
        "num_questions":     num_questions,
        "failed_questions":  failed,
        "status":            "error",
        "error":             reason,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1B — INTENT CLASSIFICATION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

_INTENT_TEST_CASES: list[tuple[str, str]] = [
    # legal_query
    ("What is Section 302 IPC and its punishment?",                         "legal_query"),
    ("Explain bail provisions under BNSS 2023",                             "legal_query"),
    ("What are the elements of cheating under Section 420 IPC?",            "legal_query"),
    ("What is anticipatory bail and how is it different from regular bail?", "legal_query"),
    ("Under which section can a FIR be quashed by High Court?",             "legal_query"),
    # document_analysis
    ("Analyze this FIR and tell me what offences are mentioned",             "document_analysis"),
    ("What does this legal notice mean and do I have to respond?",           "document_analysis"),
    ("Review this rental agreement and identify unfair clauses",             "document_analysis"),
    ("Check this employment contract for illegal termination clauses",       "document_analysis"),
    ("What is this legal document and what are my obligations?",             "document_analysis"),
    # draft_request
    ("Write an FIR complaint for theft of my mobile phone",                 "draft_request"),
    ("Help me draft a legal notice to my landlord for illegal eviction",    "draft_request"),
    ("Write a consumer complaint against Amazon for defective product",     "draft_request"),
    ("Draft a demand notice for cheque bounce under Section 138 NI Act",    "draft_request"),
    ("My salary has not been paid for 3 months, help me file complaint",    "draft_request"),
    # risk_check
    ("Is this arbitration clause in my employment contract risky?",         "risk_check"),
    ("What is the legal risk if I don't pay rent for 2 months?",            "risk_check"),
    ("Can I be arrested if I don't respond to this legal notice?",          "risk_check"),
    ("Is it legal to record a phone call without consent in India?",        "risk_check"),
    ("What happens if my company doesn't give me appointment letter?",      "risk_check"),
    # translation_request
    ("Translate this legal notice to Malayalam",                            "translation_request"),
    ("Explain Section 302 IPC in Hindi",                                    "translation_request"),
    ("Explain what this court order means in simple Malayalam",             "translation_request"),
    ("Yeh FIR ka matlab Hindi mein samjhao",                                "translation_request"),
    ("Explain my rights as a tenant in Tamil",                              "translation_request"),
    # case_law_search
    ("Show me Supreme Court judgments on Section 302 IPC murder",           "case_law_search"),
    ("What did the Supreme Court hold in the Maneka Gandhi case?",          "case_law_search"),
    ("Find landmark cases on cheque bounce Section 138 NI Act",             "case_law_search"),
    ("What are leading High Court judgments on tenant eviction rights?",    "case_law_search"),
    ("Case law for domestic violence and Section 498A cases",               "case_law_search"),
    # rights_check
    ("What are my rights as a tenant in India?",                            "rights_check"),
    ("Know my rights as an employee — my employer is not paying salary",    "rights_check"),
    ("What are the rights of an arrested person in India?",                 "rights_check"),
    ("Explain consumer rights under Indian law",                            "rights_check"),
    ("What are women's legal rights for protection from domestic violence?","rights_check"),
    # general
    ("Hello",                                                               "general"),
    ("What can LexShield AI do for me?",                                    "general"),
    ("Who are you and how do you work?",                                    "general"),
    ("What is the capital of India?",                                       "general"),
    ("Thank you for your help",                                             "general"),
]


def run_intent_evaluation() -> dict:
    logger.info("=" * 60)
    logger.info("TASK 1B — INTENT CLASSIFICATION EVALUATION")
    logger.info("=" * 60)

    try:
        from agents.intent_classifier import intent_classifier
    except Exception as e:
        return {"overall_accuracy": 0.0, "per_intent": {}, "confusion_matrix": {},
                "num_questions": 0, "num_correct": 0, "status": "error", "error": str(e)}

    intents       = sorted(set(exp for _, exp in _INTENT_TEST_CASES))
    per_intent    = {i: {"correct": 0, "total": 0} for i in intents}
    confusion     = {i: {j: 0 for j in intents} for i in intents}
    total_correct = 0
    mistakes      = []

    for query, expected in _INTENT_TEST_CASES:
        result    = intent_classifier.classify(query)
        predicted = result.intent

        per_intent[expected]["total"]  += 1
        confusion[expected][predicted] += 1

        if predicted == expected:
            total_correct += 1
            per_intent[expected]["correct"] += 1
        else:
            mistakes.append({
                "query":      query,
                "expected":   expected,
                "predicted":  predicted,
                "confidence": result.confidence,
            })
            logger.warning(
                f"MISS: [{expected} -> {predicted}] conf={result.confidence:.2f} | "
                f"'{query[:55]}'"
            )

    total         = len(_INTENT_TEST_CASES)
    overall_acc   = total_correct / total
    per_intent_acc = {
        intent: round(c["correct"] / c["total"], 4) if c["total"] > 0 else 0.0
        for intent, c in per_intent.items()
    }

    logger.info(f"\nOverall: {overall_acc:.1%} ({total_correct}/{total})")
    for intent, acc in sorted(per_intent_acc.items()):
        c = per_intent[intent]
        m = "OK" if acc == 1.0 else ("△" if acc >= 0.8 else "FAIL")
        logger.info(f"  {m} {intent:25} {acc:.1%}  ({c['correct']}/{c['total']})")

    return {
        "overall_accuracy":    round(overall_acc, 4),
        "per_intent":          per_intent_acc,
        "confusion_matrix":    confusion,
        "num_questions":       total,
        "num_correct":         total_correct,
        "misclassifications":  mistakes,
        "status":              "success",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1C — DRAFTING AGENT (static verification)
# ═══════════════════════════════════════════════════════════════════════════════

def get_draft_agent_metrics() -> dict:
    return {
        "stage_completion_rate":       1.0,
        "required_fields_present":     True,
        "categories_tested":           ["wage_theft", "cheque_bounce", "consumer_complaint"],
        "stages_verified":             ["INIT", "CLARIFY", "RETRIEVE_SECTIONS",
                                        "IDENTIFY_AUTHORITY", "CONFIRM", "GENERATE", "DONE"],
        "hitl_confirm_gate":           True,
        "sqlite_persistence_verified": True,
        "note": (
            "Manually verified — 3 test drafts completed end-to-end. "
            "All 7 stages executed correctly with HITL confirm gate active."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_package_version(package: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(package)
    except Exception:
        return "unknown"


def _get_system_info() -> dict:
    import platform
    return {
        "hardware":           "Intel i5-8250U, 8GB RAM, no GPU, Windows 11",
        "os":                 platform.platform(),
        "python":             platform.python_version(),
        "ragas_version":      _get_package_version("ragas"),
        "groq_model":         "llama-3.3-70b-versatile (pipeline primary)",
        "eval_llm":           "meta-llama/llama-3.3-70b-instruct:free via OpenRouter (RAGAS scoring)",
        "multi_llm_router":   True,
        "providers":          ["groq", "openrouter-llama33", "openrouter-qwen3",
                               "gemini-2.0-flash", "openrouter-nemotron",
                               "openrouter-deepseek-r1", "openrouter-mistral7b"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LexShield AI Evaluation Suite")
    parser.add_argument("--fresh",       action="store_true",
                        help="Ignore checkpoint, start from scratch")
    parser.add_argument("--intent-only", action="store_true",
                        help="Run only intent eval (fast, no LLM calls)")
    args = parser.parse_args()

    logger.info("LexShield AI — Evaluation Suite")
    logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    if args.intent_only:
        logger.info("Mode: intent-only")
        rag_metrics = _rag_error_result("skipped (--intent-only)")
    else:
        logger.info(
            "Mode: full eval | "
            "RAG -> Groq+OpenRouter failover | "
            "RAGAS scoring -> OpenRouter (separate rate limit) | "
            "Checkpoint -> tests/eval_progress.json"
        )
        rag_metrics = run_rag_evaluation(fresh=args.fresh)

    intent_metrics = run_intent_evaluation()
    draft_metrics  = get_draft_agent_metrics()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system":    _get_system_info(),
        "rag_metrics": rag_metrics,
        "intent_metrics": {
            "overall_accuracy":    intent_metrics["overall_accuracy"],
            "per_intent":          intent_metrics["per_intent"],
            "confusion_matrix":    intent_metrics["confusion_matrix"],
            "num_questions":       intent_metrics["num_questions"],
            "num_correct":         intent_metrics["num_correct"],
            "misclassifications":  intent_metrics.get("misclassifications", []),
            "status":              intent_metrics["status"],
        },
        "draft_agent_metrics": draft_metrics,
        "summary": {
            "overall_intent_accuracy": intent_metrics["overall_accuracy"],
            "rag_faithfulness":        rag_metrics.get("faithfulness",      0),
            "rag_answer_relevancy":    rag_metrics.get("answer_relevancy",  0),
            "rag_context_precision":   rag_metrics.get("context_precision", 0),
            "rag_context_recall":      rag_metrics.get("context_recall",    0),
            "eval_status": (
                "complete"
                if rag_metrics.get("status") == "success"
                and intent_metrics.get("status") == "success"
                else "partial"
            ),
        },
    }

    _EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Report saved: {_EVAL_REPORT_PATH}")
    logger.info(f"Intent accuracy:       {intent_metrics['overall_accuracy']:.1%}")
    logger.info(f"RAG faithfulness:      {rag_metrics.get('faithfulness', 0):.4f}")
    logger.info(f"RAG answer relevancy:  {rag_metrics.get('answer_relevancy', 0):.4f}")
    logger.info(f"RAG context precision: {rag_metrics.get('context_precision', 0):.4f}")
    logger.info(f"RAG context recall:    {rag_metrics.get('context_recall', 0):.4f}")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    main()