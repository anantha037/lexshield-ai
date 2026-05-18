"""
Offline RAGAS evaluation using saved eval_progress.json
Run: python -m tests.eval_from_progress
"""

import json
import os
import sys
import math
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lexshield.eval_offline")

_PROGRESS_PATH = _ROOT / "tests" / "eval_progress.json"
_REPORT_PATH   = _ROOT / "tests" / "eval_report_offline.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def build_eval_llm():
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    chat = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_retries=5,
        timeout=120,
        default_headers={
            "HTTP-Referer": "https://github.com/lexshield", # Required by OpenRouter
            "X-Title": "LexShield-Eval"                     # Required by OpenRouter
        }
    )

    logger.info("Eval LLM: OpenRouter Gemma 3 27B")

    return LangchainLLMWrapper(chat)


def build_eval_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        print("Run: pip install langchain-community sentence-transformers")
        sys.exit(1)

    logger.info("Loading local HuggingFace embeddings (all-MiniLM-L6-v2)...")
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name   = "all-MiniLM-L6-v2",
            model_kwargs = {"device": "cpu"},
        )
    )


def run():
    # ── Load progress ──────────────────────────────────────────────
    if not _PROGRESS_PATH.exists():
        logger.error(f"Progress file not found: {_PROGRESS_PATH}")
        sys.exit(1)

    with open(_PROGRESS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["rows"]
    logger.info(f"Loaded {len(rows)} completed questions from progress file.")

    # ── Validate rows ──────────────────────────────────────────────
    MAX_CONTEXT_CHARS = 600
    valid_rows = []
    for r in rows:
        if r.get("_failed"):
            logger.warning(f"Skipping failed row: {r['question'][:55]}")
            continue
        if not r.get("answer") or not r.get("contexts"):
            logger.warning(f"Skipping incomplete row: {r['question'][:55]}")
            continue
            
        trimmed_contexts = [
            c[:MAX_CONTEXT_CHARS] for c in r["contexts"][:1]
        ]
        trimmed_answer = r["answer"][:2000] if isinstance(r["answer"], str) else r["answer"]

        valid_rows.append({
            "question":     r["question"],
            "answer":       trimmed_answer,
            "contexts":     trimmed_contexts,
            "ground_truth": r["ground_truth"],
        })

    valid_rows = valid_rows[:1]
    logger.info(f"Valid rows for RAGAS: {len(valid_rows)}/{len(rows)}")

    # ── Build RAGAS components ─────────────────────────────────────
    try:
        from ragas           import evaluate
        from ragas.metrics import faithfulness
        from ragas.run_config import RunConfig
        from datasets        import Dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}\nRun: pip install ragas datasets")
        sys.exit(1)

    eval_llm        = build_eval_llm()
    eval_embeddings = build_eval_embeddings()

    # ── Run RAGAS Row-by-Row ───────────────────────────────────────
    logger.info("Running RAGAS scoring (row by row to strictly respect 30 RPM limit)...")

    metrics_list = [faithfulness]
    all_scores = {m.name: [] for m in metrics_list}
    
    # Configure RAGAS to fail fast instead of looping infinitely on 429s
    run_config = RunConfig(max_workers=1, max_retries=3, timeout=120)

    for row_idx, row in enumerate(valid_rows):
        logger.info(f"\nProcessing Question {row_idx + 1}/{len(valid_rows)}...")
        row_dataset = Dataset.from_list([row])
        
        for metric in metrics_list:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        evaluate,
                        dataset=row_dataset,
                        metrics=[metric],
                        llm=eval_llm,
                        embeddings=eval_embeddings,
                        run_config=run_config
                    )
                    r = future.result(timeout=180)
                
                scores_dict = dict(r)
                val = scores_dict.get(metric.name)
                
                if val is not None and not math.isnan(float(val)):
                    all_scores[metric.name].append(float(val))
                    logger.info(f"  {metric.name}: {val:.4f}")
                else:
                    all_scores[metric.name].append(0.0)
                    logger.info(f"  {metric.name}: 0.0000 (returned NaN)")

            except FutureTimeoutError:
                logger.error(f"  Metric {metric.name} timed out after 180s. Recording 0.0.")
                all_scores[metric.name].append(0.0)
            except Exception as e:
                logger.error(f"  Metric {metric.name} failed: {e}")
                all_scores[metric.name].append(0.0)

            # Hard sleep after EVERY metric call to guarantee we never exceed 30 RPM
            time.sleep(8.0)

    def sf(vals):
        if not vals: return 0.0
        valid = [v for v in vals if v is not None and not math.isnan(v)]
        if not valid: return 0.0
        return round(sum(valid) / len(valid), 4)

    final = {
        "faithfulness":  sf(all_scores.get("faithfulness", [])),
        "num_questions": len(valid_rows),
        "source":        "eval_progress.json (offline scoring)",
        "status":        "success",
    }

    # ── Print & save ───────────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("RAGAS RESULTS (offline from saved progress)")
    logger.info("="*55)
    for k in ["faithfulness"]:
        logger.info(f"  {k:25} {final[k]:.4f}")
    logger.info("="*55)

    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    logger.info(f"Report saved: {_REPORT_PATH}")


if __name__ == "__main__":
    run()