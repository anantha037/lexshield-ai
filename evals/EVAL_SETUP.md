# evals/EVAL_SETUP.md
# LexShield Evaluation Suite — Setup & Run Guide

## Install eval dependencies

```bash
pip install ragas datasets langchain-groq langsmith --break-system-packages
```

If `langchain-groq` fails (Python version issues), use the OpenAI-compat fallback:
```bash
pip install ragas datasets langchain-openai langsmith --break-system-packages
```

---

## Required environment variables

Add to your `.env` file:

```
GROQ_API_KEY=your_key_here

# LangSmith (for Step 3)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=LexShield-AI
```

LangSmith free account: https://smith.langchain.com

---

## Run commands

### Quickest possible run (zero Groq calls, instant results):
```bash
cd C:\Projects\LexShield-AI
python -m evals.run_all --quick
```
This runs: routing (regex only) + Phase 1 retrieval + complexity router + latency profile.
No rate limits. Takes ~3-5 minutes.

### Full eval suite (all steps):
```bash
python -m evals.run_all
```
Takes ~25-35 minutes total due to Groq rate-limit sleeps. Leave it running.

### Individual evals:

```bash
# Step 2 — Routing accuracy (regex + LLM, ~15 Groq calls)
python -m evals.routing_eval

# Step 2 — Routing (regex only, zero Groq calls)
python -m evals.routing_eval --regex-only

# Step 1 — RAG Phase 1: retrieval only (zero LLM)
python -m evals.rag_eval --phase 1

# Step 1 — RAG Phase 2: generation (batched)
python -m evals.rag_eval --phase 2

# Step 1 — RAG Phase 3: RAGAS scoring
python -m evals.rag_eval --phase 3

# Step 1 — Retrieval metrics only (no generation needed)
python -m evals.rag_eval --phase 1 --retrieval-only
python -m evals.rag_eval --phase 3 --retrieval-only

# Missing Step — Hallucination rate
python -m evals.pipeline_health_eval --hallucination-only

# Missing Step — Latency profiling
python -m evals.pipeline_health_eval --latency-only

# Step 3 — LangSmith agent tracing
python -m evals.langsmith_eval

# Bonus — Complexity router accuracy (zero LLM)
python -m evals.rag_eval --phase 1 --complexity
```

---

## Recommended first run order (minimal rate limit risk)

1. `python -m evals.routing_eval --regex-only`   → instant, validates test set loads
2. `python -m evals.rag_eval --phase 1`           → retrieval quality, no LLM
3. `python -m evals.pipeline_health_eval --latency-only` → latency baseline
4. `python -m evals.routing_eval`                 → full routing with LLM (~15 calls)
5. `python -m evals.rag_eval --phase 2`           → generation (batched)
6. `python -m evals.rag_eval --phase 3`           → RAGAS scoring
7. `python -m evals.langsmith_eval`               → agent tracing

---

## What each eval measures

| Eval | Metric | LLM calls | What it tells you |
|------|--------|-----------|-------------------|
| routing_eval | Intent accuracy % | ~15 | Does your classifier route correctly? |
| rag_eval Phase 1 | Context coverage proxy | 0 | Are the right chunks retrieved? |
| rag_eval Phase 3 | context_precision, context_recall, faithfulness, answer_correctness | ~25 (RAGAS judge) | Full RAG quality score |
| pipeline_health | Hallucination risk rate | 0 | How often does grounding_warning fire? |
| pipeline_health | p50/p95 latency per tier | 0 | How fast is each complexity path? |
| langsmith_eval | Node-level scores in LangSmith | ~7 | Per-node quality visible in dashboard |
| complexity_eval | Complexity routing accuracy % | 0 | Does simple/moderate/complex route correctly? |

---

## Results location

All results save to `evals/results/`:
- `routing_summary_latest.json`
- `rag_phase1_contexts.json`
- `rag_phase2_answers.json`
- `rag_summary_latest.json`
- `hallucination_latest.json`
- `latency_latest.json`
- `langsmith_feedback_latest.json`
- `eval_suite_<timestamp>.json`  ← master summary from run_all.py

---

## Adding node-level scoring to graph.py

In `agents/graph.py`, after any `rag_pipeline.query()` call, add:

```python
from evals.langsmith_eval import score_rag_node
from langsmith.run_helpers import get_current_run_tree
import time

def legal_rag_node(state: AgentState) -> AgentState:
    t0     = time.perf_counter()
    answer = rag_pipeline.query(state.user_message)
    ms     = (time.perf_counter() - t0) * 1000

    rt = get_current_run_tree()
    if rt:
        score_rag_node(
            run_id         = str(rt.id),
            synthesis_note = answer.synthesis_note or "",
            grounding_warn = answer.grounding_warning or "",
            latency_ms     = ms,
            intent         = state.intent or "legal_query",
        )
    ...
```

This makes every RAG node execution visible as a scored run in LangSmith.