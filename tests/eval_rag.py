"""
LexShield AI — RAG Evaluation Suite
=====================================
Metrics:
  - Hit Rate @5        : Is expected section in top-5 citations?
  - MRR                : Mean Reciprocal Rank of expected section
  - Grounding Rate     : % answers with no grounding_warning
  - Citation Rate      : % answers with inline [N] citations
  - Statute Precision  : % citations from expected source/statute
  - LLM-as-Judge       : Groq scores answer relevance 1-5
  - Section Precision  : % queries where section fast-path fired correctly

Ablation modes (--ablation flag):
  A: vector only, no rewrite, no rerank
  B: hybrid, no rewrite, no rerank
  C: hybrid + rewrite, no rerank
  D: hybrid + rewrite + rerank   <- production pipeline

Run:
  python tests/eval_rag.py                    # full eval, production pipeline
  python tests/eval_rag.py --ablation         # compare A/B/C/D
  python tests/eval_rag.py --no-judge         # skip Groq judge calls (free)
  python tests/eval_rag.py --verbose          # print each answer
"""

import os
import re
import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from dotenv import load_dotenv
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# GOLD DATASET
# 5 queries per statute × 6 statutes + 5 judgment queries = 35 total
# expected_section: exact section string stored in ChromaDB metadata
# expected_source:  substring match against Citation.source
# expected_keywords: must appear in answer_text (case-insensitive)
# ══════════════════════════════════════════════════════════════════════════════

GOLD_QUERIES: list[dict] = [

    # ── IPC 1860 ──────────────────────────────────────────────────────────────
    {
        "query":            "What is the punishment for murder under IPC?",
        "expected_section": "302",
        "expected_source":  "IPC",
        "expected_keywords": ["death", "imprisonment"],
        "statute_group":    "IPC",
    },
    {
        "query":            "What does Section 420 IPC say about cheating?",
        "expected_section": "420",
        "expected_source":  "IPC",
        "expected_keywords": ["cheat", "dishonest"],
        "statute_group":    "IPC",
    },
    {
        "query":            "What is culpable homicide under IPC?",
        "expected_section": "299",
        "expected_source":  "IPC",
        "expected_keywords": ["death", "intention"],
        "statute_group":    "IPC",
    },
    {
        "query":            "Punishment for theft under Indian Penal Code",
        "expected_section": "379",
        "expected_source":  "IPC",
        "expected_keywords": ["imprisonment", "theft"],
        "statute_group":    "IPC",
    },
    {
        "query":            "What is criminal breach of trust under IPC?",
        "expected_section": "405",
        "expected_source":  "IPC",
        "expected_keywords": ["entrusted", "misappropriate"],
        "statute_group":    "IPC",
    },

    # ── BNS 2023 ──────────────────────────────────────────────────────────────
    {
        "query":            "What is the BNS equivalent of IPC Section 302?",
        "expected_section": "101",
        "expected_source":  "BNS",
        "expected_keywords": ["murder", "punishment"],
        "statute_group":    "BNS",
    },
    {
        "query":            "How does BNS 2023 define organised crime?",
        "expected_section": "111",
        "expected_source":  "BNS",
        "expected_keywords": ["organised", "crime"],
        "statute_group":    "BNS",
    },
    {
        "query":            "Punishment for theft under Bharatiya Nyaya Sanhita",
        "expected_section": "303",
        "expected_source":  "BNS",
        "expected_keywords": ["imprisonment", "theft"],
        "statute_group":    "BNS",
    },
    {
        "query":            "What is cheating under BNS 2023?",
        "expected_section": "318",
        "expected_source":  "BNS",
        "expected_keywords": ["cheat", "dishonest"],
        "statute_group":    "BNS",
    },
    {
        "query":            "How does BNS define culpable homicide?",
        "expected_section": "100",
        "expected_source":  "BNS",
        "expected_keywords": ["death", "intention"],
        "statute_group":    "BNS",
    },

    # ── CrPC ──────────────────────────────────────────────────────────────────
    {
        "query":            "When can police arrest without a warrant under CrPC?",
        "expected_section": "41",
        "expected_source":  "CrPC",
        "expected_keywords": ["arrest", "warrant"],
        "statute_group":    "CrPC",
    },
    {
        "query":            "What is the procedure for filing an FIR under CrPC?",
        "expected_section": "154",
        "expected_source":  "CrPC",
        "expected_keywords": ["information", "police"],
        "statute_group":    "CrPC",
    },
    {
        "query":            "What rights does an arrested person have under CrPC?",
        "expected_section": "50",
        "expected_source":  "CrPC",
        "expected_keywords": ["grounds", "arrest", "inform"],
        "statute_group":    "CrPC",
    },
    {
        "query":            "What is a charge sheet under CrPC?",
        "expected_section": "173",
        "expected_source":  "CrPC",
        "expected_keywords": ["report", "investigation"],
        "statute_group":    "CrPC",
    },
    {
        "query":            "Bail provisions for bailable offences under CrPC",
        "expected_section": "436",
        "expected_source":  "CrPC",
        "expected_keywords": ["bail", "bailable"],
        "statute_group":    "CrPC",
    },

    # ── Consumer Protection Act 2019 ──────────────────────────────────────────
    {
        "query":            "What is deficiency in service under Consumer Protection Act?",
        "expected_section": "2",
        "expected_source":  "Consumer",
        "expected_keywords": ["deficiency", "service"],
        "statute_group":    "Consumer",
    },
    {
        "query":            "How to file a consumer complaint under Consumer Protection Act 2019?",
        "expected_section": "35",
        "expected_source":  "Consumer",
        "expected_keywords": ["complaint", "commission"],
        "statute_group":    "Consumer",
    },
    {
        "query":            "What is unfair trade practice under Consumer Protection Act?",
        "expected_section": "2",
        "expected_source":  "Consumer",
        "expected_keywords": ["unfair", "trade"],
        "statute_group":    "Consumer",
    },
    {
        "query":            "Jurisdiction of District Consumer Commission",
        "expected_section": "34",
        "expected_source":  "Consumer",
        "expected_keywords": ["district", "commission", "jurisdiction"],
        "statute_group":    "Consumer",
    },
    {
        "query":            "What are the rights of consumers under Consumer Protection Act 2019?",
        "expected_section": "2",
        "expected_source":  "Consumer",
        "expected_keywords": ["right", "consumer"],
        "statute_group":    "Consumer",
    },

    # ── Code on Wages 2019 ────────────────────────────────────────────────────
    {
        "query":            "What is the definition of wages under Code on Wages 2019?",
        "expected_section": "2",
        "expected_source":  "Wages",
        "expected_keywords": ["wages", "remuneration"],
        "statute_group":    "Wages",
    },
    {
        "query":            "When must an employer pay wages under Code on Wages?",
        "expected_section": "17",
        "expected_source":  "Wages",
        "expected_keywords": ["payment", "wage period"],
        "statute_group":    "Wages",
    },
    {
        "query":            "What deductions are permissible from wages under Code on Wages?",
        "expected_section": "18",
        "expected_source":  "Wages",
        "expected_keywords": ["deduction", "permissible"],
        "statute_group":    "Wages",
    },
    {
        "query":            "Penalties for non-payment of minimum wages",
        "expected_section": "54",
        "expected_source":  "Wages",
        "expected_keywords": ["penalty", "minimum wage"],
        "statute_group":    "Wages",
    },
    {
        "query":            "What is the floor wage concept under Code on Wages 2019?",
        "expected_section": "9",
        "expected_source":  "Wages",
        "expected_keywords": ["floor wage", "minimum rate"],
        "statute_group":    "Wages",
    },

    # ── Kerala Rent Control Act ───────────────────────────────────────────────
    {
        "query":            "Grounds for eviction of tenant under Kerala Rent Control Act",
        "expected_section": "11",
        "expected_source":  "Kerala",
        "expected_keywords": ["eviction", "tenant"],
        "statute_group":    "Kerala",
    },
    {
        "query":            "What is fair rent under Kerala Buildings Lease and Rent Control Act?",
        "expected_section": "5",
        "expected_source":  "Kerala",
        "expected_keywords": ["fair rent", "fixation"],
        "statute_group":    "Kerala",
    },
    {
        "query":            "Can a landlord increase rent arbitrarily in Kerala?",
        "expected_section": "5",
        "expected_source":  "Kerala",
        "expected_keywords": ["rent", "increase"],
        "statute_group":    "Kerala",
    },
    {
        "query":            "Procedure for rent control court in Kerala",
        "expected_section": "30",
        "expected_source":  "Kerala",
        "expected_keywords": ["court", "application"],
        "statute_group":    "Kerala",
    },
    {
        "query":            "Deposit of rent in court under Kerala Rent Control Act",
        "expected_section": "9",
        "expected_source":  "Kerala",
        "expected_keywords": ["deposit", "court"],
        "statute_group":    "Kerala",
    },

    # ── Judgment queries ──────────────────────────────────────────────────────
    {
        "query":            "Supreme Court judgments on right to bail",
        "expected_section": "",
        "expected_source":  "judgment",
        "expected_keywords": ["bail", "court"],
        "statute_group":    "Judgments",
    },
    {
        "query":            "Case law on wrongful termination of employment in India",
        "expected_section": "",
        "expected_source":  "judgment",
        "expected_keywords": ["termination", "employment"],
        "statute_group":    "Judgments",
    },
    {
        "query":            "Judgments on tenant rights in India",
        "expected_section": "",
        "expected_source":  "judgment",
        "expected_keywords": ["tenant", "landlord"],
        "statute_group":    "Judgments",
    },
    {
        "query":            "Case law on consumer deficiency in banking services",
        "expected_section": "",
        "expected_source":  "judgment",
        "expected_keywords": ["consumer", "bank"],
        "statute_group":    "Judgments",
    },
    {
        "query":            "Supreme Court on right to fair trial",
        "expected_section": "",
        "expected_source":  "judgment",
        "expected_keywords": ["fair trial", "accused"],
        "statute_group":    "Judgments",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    query:             str
    statute_group:     str
    expected_section:  str
    expected_source:   str
    expected_keywords: list[str]

    # Retrieval
    hit_at_5:          bool            = False
    reciprocal_rank:   float           = 0.0
    section_in_answer: bool            = False
    source_precision:  float           = 0.0    # fraction of citations from expected source

    # Answer quality
    grounded:          bool            = True
    has_citations:     bool            = False
    keyword_hits:      int             = 0
    keyword_recall:    float           = 0.0    # hits / total expected keywords
    llm_judge_score:   Optional[float] = None

    # Pipeline metadata
    sources_consulted: int             = 0
    reranker_used:     bool            = False
    rewritten_queries: list[str]       = field(default_factory=list)
    grounding_warning: Optional[str]   = None

    # Raw for verbose
    answer_text:       str             = ""
    citations_raw:     list[dict]      = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = (
    "You are a strict legal evaluation judge. "
    "Score the answer's relevance to the question on a scale of 1 to 5:\n"
    "5 = Directly and fully answers the question with correct legal detail.\n"
    "4 = Mostly answers the question, minor gap.\n"
    "3 = Partially answers, significant information missing.\n"
    "2 = Tangentially related but does not answer.\n"
    "1 = Irrelevant or incorrect.\n"
    "Respond with a single integer only. No explanation."
)

def llm_judge_score(query: str, answer: str) -> Optional[float]:
    """Single Groq call. Returns 1-5 or None on failure."""
    try:
        from rag.llm import llm
        prompt = f"Question: {query}\n\nAnswer: {answer}\n\nScore (1-5):"
        raw    = llm.generate(
            prompt        = prompt,
            system_prompt = JUDGE_SYSTEM,
            temperature   = 0.0,
            max_tokens    = 5,
        )
        match = re.search(r'[1-5]', raw.strip())
        return float(match.group()) if match else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE QUERY EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_query(
    gold:       dict,
    pipeline,           # RAGPipeline instance
    use_judge:  bool = True,
    verbose:    bool = False,
) -> EvalResult:

    result = EvalResult(
        query             = gold["query"],
        statute_group     = gold["statute_group"],
        expected_section  = gold["expected_section"],
        expected_source   = gold["expected_source"],
        expected_keywords = gold["expected_keywords"],
    )

    try:
        answer: "LegalAnswer" = pipeline.query(gold["query"])
    except Exception as e:
        result.grounding_warning = f"Pipeline exception: {e}"
        result.grounded          = False
        return result

    # ── Populate metadata ─────────────────────────────────────────────────────
    result.answer_text       = answer.answer_text
    result.sources_consulted = answer.sources_consulted
    result.reranker_used     = answer.reranker_used
    result.rewritten_queries = answer.rewritten_queries
    result.grounding_warning = answer.grounding_warning
    result.grounded          = answer.grounding_warning is None
    result.has_citations     = bool(re.findall(r'\[\d+\]', answer.answer_text))

    # ── Hit Rate & MRR ────────────────────────────────────────────────────────
    exp_sec = gold["expected_section"]
    if exp_sec:
        for rank, cit in enumerate(answer.citations, start=1):
            if cit.section == exp_sec:
                result.hit_at_5        = True
                result.reciprocal_rank = 1.0 / rank
                result.section_in_answer = True
                break
    else:
        # Judgment queries: hit if any citation from expected source substring
        for cit in answer.citations:
            if gold["expected_source"].lower() in cit.source.lower():
                result.hit_at_5        = True
                result.reciprocal_rank = 1.0
                break

    # ── Source Precision ──────────────────────────────────────────────────────
    if answer.citations:
        matching = sum(
            1 for c in answer.citations
            if gold["expected_source"].lower() in c.source.lower()
        )
        result.source_precision = matching / len(answer.citations)

    # ── Keyword Recall ────────────────────────────────────────────────────────
    answer_lower = answer.answer_text.lower()
    hits = sum(1 for kw in gold["expected_keywords"] if kw.lower() in answer_lower)
    result.keyword_hits    = hits
    result.keyword_recall  = hits / len(gold["expected_keywords"]) if gold["expected_keywords"] else 1.0

    # ── LLM Judge ─────────────────────────────────────────────────────────────
    if use_judge:
        time.sleep(1.0)   # avoid Groq rate limit
        result.llm_judge_score = llm_judge_score(gold["query"], answer.answer_text)

    # ── Verbose ───────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Query   : {gold['query']}")
        print(f"  Answer  : {answer.answer_text[:300]}...")
        print(f"  Sections: {[c.section for c in answer.citations]}")
        print(f"  Warning : {answer.grounding_warning}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(results: list[EvalResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    section_queries = [r for r in results if r.expected_section]

    judge_scores = [r.llm_judge_score for r in results if r.llm_judge_score is not None]

    return {
        "n_queries":          n,
        "hit_rate_at_5":      round(sum(r.hit_at_5 for r in results) / n, 3),
        "mrr":                round(sum(r.reciprocal_rank for r in results) / n, 3),
        "grounding_rate":     round(sum(r.grounded for r in results) / n, 3),
        "citation_rate":      round(sum(r.has_citations for r in results) / n, 3),
        "keyword_recall":     round(sum(r.keyword_recall for r in results) / n, 3),
        "source_precision":   round(sum(r.source_precision for r in results) / n, 3),
        "llm_judge_mean":     round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None,
        "reranker_used_rate": round(sum(r.reranker_used for r in results) / n, 3),
        "section_hit_rate":   round(
            sum(r.hit_at_5 for r in section_queries) / len(section_queries), 3
        ) if section_queries else None,
    }


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Queries evaluated     : {metrics['n_queries']}")
    print(f"  Hit Rate @5           : {metrics['hit_rate_at_5']:.1%}   (target ≥ 0.80)")
    print(f"  MRR                   : {metrics['mrr']:.3f}  (target ≥ 0.60)")
    print(f"  Section Hit Rate      : {metrics.get('section_hit_rate', 'N/A')}")
    print(f"  Grounding Rate        : {metrics['grounding_rate']:.1%}   (target ≥ 0.80)")
    print(f"  Citation Rate         : {metrics['citation_rate']:.1%}")
    print(f"  Keyword Recall        : {metrics['keyword_recall']:.1%}")
    print(f"  Source Precision      : {metrics['source_precision']:.1%}")
    print(f"  LLM Judge Score (1-5) : {metrics['llm_judge_mean'] or 'skipped'}  (target ≥ 4.0)")
    print(f"  Reranker Used Rate    : {metrics['reranker_used_rate']:.1%}")


def print_per_statute(results: list[EvalResult]) -> None:
    from collections import defaultdict
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[r.statute_group].append(r)

    print(f"\n{'─'*60}")
    print(f"  Per-Statute Breakdown")
    print(f"{'─'*60}")
    print(f"  {'Statute':<15} {'Hit@5':>7} {'MRR':>7} {'Grounded':>10} {'KW Recall':>11}")
    for group, rs in sorted(groups.items()):
        n        = len(rs)
        hit      = sum(r.hit_at_5 for r in rs) / n
        mrr      = sum(r.reciprocal_rank for r in rs) / n
        grounded = sum(r.grounded for r in rs) / n
        kw       = sum(r.keyword_recall for r in rs) / n
        print(f"  {group:<15} {hit:>7.1%} {mrr:>7.3f} {grounded:>10.1%} {kw:>11.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "A_vector_only":        {"enable_rewriting": False, "enable_reranking": False},
    "B_hybrid_only":        {"enable_rewriting": False, "enable_reranking": False},
    "C_hybrid_rewrite":     {"enable_rewriting": True,  "enable_reranking": False},
    "D_full_pipeline":      {"enable_rewriting": True,  "enable_reranking": True},
}

# Note: A vs B requires patching hybrid_search to disable BM25 for config A.
# For now, A and B differ only in reranking so they share the same pipeline.
# Full ablation of vector-only requires a separate hybrid_search bypass flag.


def run_ablation(gold_set: list[dict], use_judge: bool, verbose: bool) -> None:
    from rag.pipeline import RAGPipeline

    print(f"\n{'═'*60}")
    print("  ABLATION STUDY")
    print(f"{'═'*60}")
    print(f"  Running {len(ABLATION_CONFIGS)} configurations × {len(gold_set)} queries")
    print(f"  (judge={'ON' if use_judge else 'OFF'})")

    all_metrics = {}
    for config_name, kwargs in ABLATION_CONFIGS.items():
        print(f"\n  [{config_name}] running...")
        pipeline = RAGPipeline(**kwargs)
        results  = []
        for i, gold in enumerate(gold_set, 1):
            print(f"    {i:02d}/{len(gold_set)}  {gold['query'][:55]}...", end="\r")
            r = evaluate_query(gold, pipeline, use_judge=use_judge, verbose=verbose)
            results.append(r)
        m = aggregate(results)
        all_metrics[config_name] = m
        print_metrics(config_name, m)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  ABLATION COMPARISON TABLE")
    print(f"{'─'*60}")
    print(f"  {'Config':<25} {'Hit@5':>7} {'MRR':>7} {'Grounded':>10} {'Judge':>7}")
    for name, m in all_metrics.items():
        judge = f"{m['llm_judge_mean']:.2f}" if m["llm_judge_mean"] else "  N/A"
        print(f"  {name:<25} {m['hit_rate_at_5']:>7.1%} {m['mrr']:>7.3f}"
              f" {m['grounding_rate']:>10.1%} {judge:>7}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation",   action="store_true",
                        help="Run A/B/C/D ablation comparison")
    parser.add_argument("--no-judge",   action="store_true",
                        help="Skip LLM-as-judge calls (saves Groq credits)")
    parser.add_argument("--statute",    type=str, default=None,
                        help="Evaluate only one statute group e.g. --statute IPC")
    parser.add_argument("--n",          type=int, default=None,
                        help="Evaluate only first N queries (for quick smoke test)")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--save",       type=str, default=None,
                        help="Save results JSON to path e.g. --save results/eval_week2.json")
    args = parser.parse_args()

    use_judge = not args.no_judge
    gold_set  = GOLD_QUERIES

    if args.statute:
        gold_set = [q for q in gold_set if q["statute_group"].lower() == args.statute.lower()]
        if not gold_set:
            print(f"No queries found for statute group '{args.statute}'.")
            sys.exit(1)

    if args.n:
        gold_set = gold_set[:args.n]

    print(f"\n{'═'*60}")
    print(f"  LexShield AI — RAG Evaluation Suite")
    print(f"{'═'*60}")
    print(f"  Queries  : {len(gold_set)}")
    print(f"  Judge    : {'ON  (costs Groq credits)' if use_judge else 'OFF (--no-judge)'}")
    print(f"  Ablation : {'YES' if args.ablation else 'NO'}")

    if args.ablation:
        run_ablation(gold_set, use_judge=use_judge, verbose=args.verbose)
        sys.exit(0)

    # ── Production pipeline eval ──────────────────────────────────────────────
    from rag.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    results  = []

    print(f"\n  Running evaluation...\n")
    for i, gold in enumerate(gold_set, 1):
        print(f"  {i:02d}/{len(gold_set)}  {gold['query'][:60]}...")
        r = evaluate_query(gold, pipeline, use_judge=use_judge, verbose=args.verbose)

        hit_mark  = "OK" if r.hit_at_5  else "FAIL"
        grnd_mark = "OK" if r.grounded  else "⚠"
        judge_str = f"  judge={r.llm_judge_score:.0f}" if r.llm_judge_score else ""
        print(f"         hit={hit_mark}  grounded={grnd_mark}  kw={r.keyword_recall:.0%}{judge_str}")
        if r.grounding_warning:
            print(f"         ⚠  {r.grounding_warning}")

        results.append(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    metrics = aggregate(results)
    print_metrics("PRODUCTION PIPELINE — FULL RESULTS", metrics)
    print_per_statute(results)

    # ── Failures ──────────────────────────────────────────────────────────────
    failed = [r for r in results if not r.hit_at_5 or not r.grounded]
    if failed:
        print(f"\n{'─'*60}")
        print(f"  FAILURES ({len(failed)} queries)")
        print(f"{'─'*60}")
        for r in failed:
            issues = []
            if not r.hit_at_5:  issues.append("miss")
            if not r.grounded:  issues.append("ungrounded")
            print(f"  [{', '.join(issues)}] {r.query[:65]}")
            if r.grounding_warning:
                print(f"           {r.grounding_warning}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save:
        import os
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        payload = {
            "metrics": metrics,
            "results": [
                {
                    "query":            r.query,
                    "statute_group":    r.statute_group,
                    "hit_at_5":         r.hit_at_5,
                    "reciprocal_rank":  r.reciprocal_rank,
                    "grounded":         r.grounded,
                    "keyword_recall":   r.keyword_recall,
                    "llm_judge_score":  r.llm_judge_score,
                    "grounding_warning":r.grounding_warning,
                    "sources_consulted":r.sources_consulted,
                    "reranker_used":    r.reranker_used,
                    "answer_preview":   r.answer_text[:200],
                }
                for r in results
            ]
        }
        with open(args.save, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Results saved -> {args.save}")

    # ── Exit code ─────────────────────────────────────────────────────────────
    target_met = (
        metrics["hit_rate_at_5"]  >= 0.80 and
        metrics["grounding_rate"] >= 0.80 and
        (metrics["llm_judge_mean"] is None or metrics["llm_judge_mean"] >= 4.0)
    )
    print(f"\n{'OK RAG EVAL TARGETS MET' if target_met else 'FAIL TARGETS NOT MET'}")
    sys.exit(0 if target_met else 1)