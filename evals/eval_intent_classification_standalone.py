"""
Standalone intent classification eval — avoids the full agent import chain
(which requires psycopg/PostgreSQL). Imports intent_classifier directly.
Run from repo root: python evals/eval_intent_classification_standalone.py
"""
import os, sys, json, time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Direct import — bypasses agents/__init__.py which pulls in memory.py -> psycopg
import importlib.util, pathlib

# Load intent_classifier.py directly — avoids agents/__init__.py which
# transitively imports psycopg via agents/memory.py.
_spec = importlib.util.spec_from_file_location(
    "agents.intent_classifier",
    pathlib.Path(__file__).parent.parent / "agents" / "intent_classifier.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
IntentClassifier = _mod.IntentClassifier
intent_classifier = _mod.intent_classifier

# Build a minimal groq client the same way the eval expects
from dotenv import load_dotenv
load_dotenv()
try:
    from groq import Groq
    _api_key = os.getenv("GROQ_API_KEY", "")
    groq_raw = Groq(api_key=_api_key) if _api_key else None
except Exception:
    groq_raw = None

TEST_QUERIES = [
    {"query": "What is Section 302 IPC?",                              "expected": "legal_query",          "type": "clear"},
    {"query": "Find landmark cases on cheque bounce Section 138",       "expected": "case_law_search",      "type": "clear"},
    {"query": "Draft a consumer complaint against Amazon",              "expected": "draft_request",        "type": "clear"},
    {"query": "What are my rights as a tenant?",                       "expected": "rights_check",         "type": "clear"},
    {"query": "Check the legal risk in this employment contract",       "expected": "risk_check",           "type": "clear"},
    {"query": "Translate this to Malayalam",                           "expected": "translation_request",  "type": "clear"},
    {"query": "Hello, what can you do?",                               "expected": "general",              "type": "clear"},
    {"query": "I got fired without any notice, what can I do?",        "expected": "rights_check",         "type": "nuanced"},
    {"query": "My landlord locked me out of my house",                 "expected": "rights_check",         "type": "nuanced"},
    {"query": "Can you help me write a legal notice to my employer?",  "expected": "draft_request",        "type": "clear"},
    {"query": "Kesavananda Bharati judgment significance",             "expected": "case_law_search",      "type": "clear"},
    {"query": "Is my non-compete clause enforceable in Kerala?",       "expected": "legal_query",          "type": "nuanced"},
    {"query": "My employer hasn't paid salary for 3 months",           "expected": "rights_check",         "type": "nuanced"},
    {"query": "What happened in the Vishaka case?",                    "expected": "case_law_search",      "type": "clear"},
    {"query": "Explain Article 21 of the Constitution",               "expected": "legal_query",          "type": "clear"},
]

def run_eval(label="BASELINE (pre-fix)"):
    print(f"\n{'='*125}")
    print(f"Intent Classification Eval — {label}")
    print(f"{'='*125}")
    print(f"{'Query':<55} | {'Expected':<20} | {'Predicted':<20} | {'OK':<5} | {'Conf':<5} | Path")
    print("-" * 125)

    correct = clear_c = nuanced_c = clear_t = nuanced_t = 0
    total_latency = total_conf = 0.0
    llm_p = fallback_p = 0
    rows = []

    for item in TEST_QUERIES:
        q, exp, qt = item["query"], item["expected"], item["type"]
        t0 = time.time()
        result = intent_classifier.classify_with_llm(q, groq_raw)
        lat = int((time.time() - t0) * 1000)
        pred = result.intent
        conf = getattr(result, "confidence", 0.0)
        ok = pred == exp
        path = "LLM" if type(result).__name__ == "LLMIntentResult" else "Fallback"
        rows.append(dict(query=q, expected=exp, predicted=pred, correct=ok,
                         confidence=conf, latency_ms=lat, path=path, type=qt,
                         reasoning=getattr(result, "reasoning", "")))
        if ok:
            correct += 1
            if qt == "clear": clear_c += 1
            else:             nuanced_c += 1
        if qt == "clear": clear_t += 1
        else:             nuanced_t += 1
        total_latency += lat
        total_conf    += conf
        if path == "LLM": llm_p += 1
        else:             fallback_p += 1
        print(f"{q[:53]:<55} | {exp:<20} | {pred:<20} | {str(ok):<5} | {conf:.2f} | {path}")

    n = len(TEST_QUERIES)
    oa = correct / n
    ca = clear_c / clear_t if clear_t else 0
    na = nuanced_c / nuanced_t if nuanced_t else 0
    print("-" * 125)
    print(f"Overall: {oa*100:.1f}%  |  Clear: {ca*100:.1f}%  |  Nuanced: {na*100:.1f}%  |  "
          f"Avg conf: {total_conf/n:.2f}  |  Avg latency: {total_latency/n:.0f}ms  |  "
          f"LLM path: {llm_p}/{n}  Fallback: {fallback_p}/{n}")
    return {"label": label, "overall": oa, "clear": ca, "nuanced": na,
            "avg_conf": total_conf/n, "avg_latency_ms": total_latency/n,
            "llm_path": llm_p, "rows": rows}

if __name__ == "__main__":
    baseline = run_eval("BASELINE (pre-fix)")
    out = os.path.join(os.path.dirname(__file__), "results", "intent_eval_baseline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nBaseline saved to {out}")

    print("\n" + "=" * 125)
    print("NOTE: The eval above tests classify_with_llm() (Groq JSON-mode path).")
    print("Bug 4 fixes classify_with_tool_calls() (LangChain bound_llm path),")
    print("which is called from graph.py classify_intent_node — not directly here.")
    print("The scores above are the BASELINE for classify_with_llm and should")
    print("be UNCHANGED after the Bug 4 fix (tool-calling is a separate code path).")
    print("=" * 125)

