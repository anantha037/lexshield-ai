"""
LexShield AI — Day 3 Grounding Test Suite
==========================================
10 real legal queries covering different areas of Indian law.
Each query is checked against 4 criteria:

  Check 1 — Cites specific sources    : answer has [1], [2], etc. inline
  Check 2 — Grounded in retrieved text: grounding_warning is None
  Check 3 — Multi-source synthesis    : >1 source cited when relevant
  Check 4 — No phantom sections       : no section numbers absent from chunks

Prints a per-query pass/fail table.
Target: 8 / 10 queries pass all four checks.

Usage:
    python tests/test_day3.py
    python tests/test_day3.py --verbose   (show full answer for each query)
    python tests/test_day3.py --query 3   (run only query #3)
"""

import sys
import re
import argparse
import time
from dataclasses import dataclass

# ── Test queries ──────────────────────────────────────────────────────────────
# Chosen to cover: statutes, criminal, civil, consumer, labour, procedural,
# multi-source (IPC + BNS overlap), single-source (Kerala Rent), FIR procedure

TEST_QUERIES: list[dict] = [
    {
        "id":       1,
        "query":    "What is the punishment for murder under IPC?",
        "tags":     ["criminal", "single-act", "ipc"],
        "expect_sections": ["302"],
        "multi_source_expected": False,
    },
    {
        "id":       2,
        "query":    "What is cheating and what is the punishment under Section 420 IPC?",
        "tags":     ["criminal", "ipc", "exact-section"],
        "expect_sections": ["420"],
        "multi_source_expected": False,
    },
    {
        "id":       3,
        "query":    "How does the BNS 2023 treat cheating compared to IPC?",
        "tags":     ["criminal", "ipc-bns-compare", "multi-source"],
        "expect_sections": ["420", "318"],
        "multi_source_expected": True,
    },
    {
        "id":       4,
        "query":    "What are the rights of a tenant facing eviction in Kerala?",
        "tags":     ["civil", "kerala-rent", "state-law"],
        "expect_sections": [],
        "multi_source_expected": False,
    },
    {
        "id":       5,
        "query":    "What is the procedure to file an FIR under CrPC?",
        "tags":     ["procedure", "crpc", "fir"],
        "expect_sections": ["154"],
        "multi_source_expected": False,
    },
    {
        "id":       6,
        "query":    "What consumer rights does a buyer have against a defective product?",
        "tags":     ["consumer", "redressal", "multi-source"],
        "expect_sections": [],
        "multi_source_expected": True,
    },
    {
        "id":       7,
        "query":    "What is the punishment for theft under the Bharatiya Nyaya Sanhita?",
        "tags":     ["criminal", "bns", "theft"],
        "expect_sections": [],
        "multi_source_expected": False,
    },
    {
        "id":       8,
        "query":    "Can an employer deduct wages without notice under the Code on Wages?",
        "tags":     ["labour", "wages", "deduction"],
        "expect_sections": [],
        "multi_source_expected": False,
    },
    {
        "id":       9,
        "query":    "What are the conditions for granting bail in a non-bailable offence?",
        "tags":     ["procedure", "bail", "crpc"],
        "expect_sections": ["437"],
        "multi_source_expected": False,
    },
    {
        "id":       10,
        "query":    "What are the penalties for kidnapping under Indian law?",
        "tags":     ["criminal", "ipc-bns-compare", "multi-source"],
        "expect_sections": [],
        "multi_source_expected": True,
    },
]


# ── Check functions ───────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query_id:   int
    query:      str
    passed:     bool
    checks:     dict[str, bool]
    reasons:    dict[str, str]
    answer:     str
    citations:  list
    warning:    str | None
    elapsed:    float


def check_cites_sources(answer_text: str) -> tuple[bool, str]:
    """Check 1: answer must contain at least one [N] inline citation."""
    inline = re.findall(r'\[\d+\]', answer_text)
    if inline:
        return True, f"Found {len(inline)} inline citation(s): {inline[:3]}"
    return False, "No [N] inline citations found in answer."


def check_grounding(grounding_warning: str | None) -> tuple[bool, str]:
    """Check 2: grounding_warning must be None."""
    if grounding_warning is None:
        return True, "No grounding issues detected."
    return False, f"Warning: {grounding_warning}"


def check_multi_source(citations: list, answer_text: str, expected: bool) -> tuple[bool, str]:
    """
    Check 3: multi-source synthesis.
    If expected=True  → at least 2 different sources must be cited.
    If expected=False → single source is acceptable; just verify at least 1 cited.
    """
    distinct_sources = {c.source for c in citations}
    # Count how many distinct sources appear in the answer via [N] references
    cited_numbers = set(int(m) for m in re.findall(r'\[(\d+)\]', answer_text))
    distinct_cited = {citations[n - 1].source for n in cited_numbers if 1 <= n <= len(citations)}

    if expected:
        if len(distinct_cited) >= 2:
            return True, f"Synthesized {len(distinct_cited)} sources: {list(distinct_cited)[:2]}"
        return False, f"Expected multi-source synthesis but only found: {distinct_cited or 'none'}"
    else:
        if distinct_cited:
            return True, f"Source cited: {list(distinct_cited)[:1]}"
        return False, "No sources cited in answer."


def check_no_phantom_sections(answer_text: str, citations: list) -> tuple[bool, str]:
    """Check 4: no section numbers in answer that aren't in retrieved chunks."""
    cited_secs     = set(re.findall(r'\bSection\s+(\d+[A-Z]?)\b', answer_text, re.IGNORECASE))
    available_secs = {c.section for c in citations if c.section}
    phantom        = cited_secs - available_secs

    if not phantom:
        return True, f"All cited sections ({cited_secs or 'none'}) are in retrieved sources."
    return False, f"Phantom section(s) cited: {phantom} — not in retrieved chunks."


# ── Test runner ───────────────────────────────────────────────────────────────

def run_single_query(test: dict, verbose: bool = False) -> QueryResult:
    from rag.pipeline import rag_pipeline

    print(f"  Running query {test['id']}: {test['query'][:60]}...", end=" ", flush=True)
    t0     = time.time()
    result = rag_pipeline.query(test["query"])
    elapsed = time.time() - t0
    print(f"({elapsed:.1f}s)")

    answer    = result.answer_text
    citations = result.citations
    warning   = result.grounding_warning

    # Run all 4 checks
    c1_pass, c1_reason = check_cites_sources(answer)
    c2_pass, c2_reason = check_grounding(warning)
    c3_pass, c3_reason = check_multi_source(citations, answer, test["multi_source_expected"])
    c4_pass, c4_reason = check_no_phantom_sections(answer, citations)

    checks  = {"cites_sources": c1_pass, "grounding": c2_pass,
                "multi_source":  c3_pass, "no_phantom": c4_pass}
    reasons = {"cites_sources": c1_reason, "grounding": c2_reason,
                "multi_source":  c3_reason, "no_phantom": c4_reason}
    passed  = all(checks.values())

    if verbose:
        print(f"\n  ANSWER:\n{answer}\n")
        print(f"  CITATIONS ({len(citations)}):")
        for c in citations:
            print(f"    [{c.source_number}] {c.source} sec={c.section or '-'} score={c.relevance_score}")

    return QueryResult(
        query_id  = test["id"],
        query     = test["query"],
        passed    = passed,
        checks    = checks,
        reasons   = reasons,
        answer    = answer,
        citations = citations,
        warning   = warning,
        elapsed   = elapsed,
    )


def print_results(results: list[QueryResult]) -> None:
    CHECK_NAMES = ["cites_sources", "grounding", "multi_source", "no_phantom"]
    COL_W       = 14

    # Header
    print("\n" + "=" * 80)
    print(f"{'Q':>2}  {'QUERY':<40}  {'C1':^4} {'C2':^4} {'C3':^4} {'C4':^4}  {'PASS':^5}")
    print("=" * 80)

    passed_total = 0
    for r in results:
        marks = [("✓" if r.checks[cn] else "✗") for cn in CHECK_NAMES]
        status = "✓ YES" if r.passed else "✗ NO "
        query_short = r.query[:40]
        print(f"{r.query_id:>2}  {query_short:<40}  "
              f"{marks[0]:^4} {marks[1]:^4} {marks[2]:^4} {marks[3]:^4}  {status}")
        if r.passed:
            passed_total += 1

    print("=" * 80)
    print(f"\nC1=cites_sources  C2=grounding  C3=multi_source  C4=no_phantom_sections")
    print(f"\nRESULT: {passed_total} / {len(results)} queries passed all 4 checks.")

    target = 8
    if passed_total >= target:
        print(f"✓ TARGET MET ({passed_total} >= {target}). Day 3 checkpoint passed.")
    else:
        print(f"✗ TARGET NOT MET ({passed_total} < {target}). "
              f"Review failing checks below and tune the synthesis prompt.")

    # Failure details
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\nFAILING QUERIES — detail:")
        for r in failures:
            print(f"\n  Q{r.query_id}: {r.query}")
            for cn in CHECK_NAMES:
                if not r.checks[cn]:
                    print(f"    ✗ {cn}: {r.reasons[cn]}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Print full answer + citations for each query")
    parser.add_argument("--query",   type=int, default=None,
                        help="Run only this query number (1-10)")
    args = parser.parse_args()

    queries_to_run = TEST_QUERIES
    if args.query:
        queries_to_run = [q for q in TEST_QUERIES if q["id"] == args.query]
        if not queries_to_run:
            print(f"Query {args.query} not found. Valid: 1-{len(TEST_QUERIES)}")
            sys.exit(1)

    print(f"\nLexShield AI — Day 3 Grounding Test")
    print(f"Running {len(queries_to_run)} queries ...\n")

    results = [run_single_query(q, verbose=args.verbose) for q in queries_to_run]
    print_results(results)