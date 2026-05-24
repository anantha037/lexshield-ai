"""
LexShield AI — Week 3 Full Evaluation
=======================================
Evaluates all Week 3 components with real quality metrics.

Dimensions:
  1. RAG Retrieval Quality      — Precision@5, Recall@5, MRR
  2. RAG Answer Quality         — LLM-as-Judge via Gemini (4 dimensions, 1-5)
  3. RAG Grounding              — Section/act citation accuracy
  4. Intent Classifier          — Accuracy + per-class F1 (30 queries)
  5. Knowledge Graph            — Connectivity + lookup precision
  6. Drafting Agent             — Real draft structural completeness

Run:
  python -m tests.eval_week3

Output:
  Console report + tests/eval_results/week3_eval.txt

Cost:
  ~15 Gemini API calls (free tier: 1M tokens/day)
  ~5 Groq API calls for RAG (free tier: 100k tokens/day)
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAG TEST SET
# (query, expected_sections, expected_acts, answer_must_contain)
# ═══════════════════════════════════════════════════════════════════════════════

RAG_TEST_CASES = [
    {
        "query":              "What is the punishment for murder under IPC?",
        "expected_sections":  ["302", "300"],
        "expected_acts":      ["Indian Penal Code"],
        "answer_must_contain": ["imprisonment", "death", "murder"],
    },
    {
        "query":              "What is Section 420 IPC cheating?",
        "expected_sections":  ["420", "415"],
        "expected_acts":      ["Indian Penal Code"],
        "answer_must_contain": ["cheating", "dishonest", "imprisonment"],
    },
    {
        "query":              "Explain bail provisions for non-bailable offences under BNSS",
        "expected_sections":  ["480", "482"],
        "expected_acts":      ["Bharatiya Nagarik Suraksha Sanhita"],
        "answer_must_contain": ["bail", "non-bailable", "court"],
    },
    {
        "query":              "What is Section 138 NI Act cheque bounce punishment?",
        "expected_sections":  ["138"],
        "expected_acts":      ["Negotiable Instruments Act"],
        "answer_must_contain": ["cheque", "imprisonment", "fine"],
    },
    {
        "query":              "What are the rights of an arrested person under BNSS?",
        "expected_sections":  ["47", "48", "50"],
        "expected_acts":      ["Bharatiya Nagarik Suraksha Sanhita"],
        "answer_must_contain": ["arrested", "inform", "grounds"],
    },
    {
        "query":              "Define theft under BNS",
        "expected_sections":  ["302", "303"],
        "expected_acts":      ["Bharatiya Nyaya Sanhita"],
        "answer_must_contain": ["moveable property", "consent", "theft"],
    },
    {
        "query":              "What is Section 376 IPC rape punishment?",
        "expected_sections":  ["376"],
        "expected_acts":      ["Indian Penal Code"],
        "answer_must_contain": ["imprisonment", "rape", "years"],
    },
    {
        "query":              "Explain anticipatory bail under CrPC",
        "expected_sections":  ["438"],
        "expected_acts":      ["Code of Criminal Procedure"],
        "answer_must_contain": ["anticipatory", "bail", "arrest"],
    },
    {
        "query":              "What is criminal breach of trust under IPC Section 406?",
        "expected_sections":  ["406", "405"],
        "expected_acts":      ["Indian Penal Code"],
        "answer_must_contain": ["trust", "entrusted", "property"],
    },
    {
        "query":              "Explain Section 498A IPC cruelty by husband",
        "expected_sections":  ["498A", "498"],
        "expected_acts":      ["Indian Penal Code"],
        "answer_must_contain": ["cruelty", "husband", "harassment"],
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INTENT CLASSIFIER TEST SET (30 queries, 5 per intent)
# ═══════════════════════════════════════════════════════════════════════════════

INTENT_TEST_CASES = [
    # legal_query (5)
    ("What is Section 302 IPC punishment for murder?",                      "legal_query"),
    ("Explain the definition of theft under BNS Section 303",               "legal_query"),
    ("What are the bail conditions under BNSS Section 480?",                "legal_query"),
    ("What is Section 138 NI Act cheque bounce process?",                   "legal_query"),
    ("How does anticipatory bail under Section 438 CrPC work?",             "legal_query"),
    # document_analysis (5)
    ("Analyze this employment contract for key clauses",                    "document_analysis"),
    ("Review this property deed and extract important terms",               "document_analysis"),
    ("Scan this rental agreement and summarize it",                         "document_analysis"),
    ("Extract all clauses from this attached legal notice",                 "document_analysis"),
    ("What does this document say about termination conditions?",           "document_analysis"),
    # draft_request (5)
    ("Help me draft a written complaint to police for theft",               "draft_request"),
    ("Create a rental agreement template for 11 months",                    "draft_request"),
    ("Write a legal notice for cheque bounce under Section 138",            "draft_request"),
    ("Prepare a legal notice for breach of contract",                       "draft_request"),
    ("Help me write a consumer complaint against a builder",                "draft_request"),
    # risk_check (5)
    ("What are the legal consequences of not paying EMI on a loan?",        "risk_check"),
    ("Am I liable if my employee gets injured at my workplace?",            "risk_check"),
    ("Is it legal to record a phone call without consent in India?",        "risk_check"),
    ("What happens if I breach a rental agreement before the lease ends?",  "risk_check"),
    ("Can I be arrested for not repaying a personal loan?",                 "risk_check"),
    # translation_request (5)
    ("Translate this legal notice into Malayalam please",                   "translation_request"),
    ("Explain this FIR in Hindi",                                           "translation_request"),
    ("Convert this notice into Tamil",                                      "translation_request"),
    ("Say this in Kannada — what are my rights when arrested?",             "translation_request"),
    ("Explain the cheque bounce law in Malayalam",                          "translation_request"),
    # general (5)
    ("Hello, I need some help with a legal issue",                          "general"),
    ("What is LexShield AI and what can it do?",                            "general"),
    ("Thank you, that was very helpful",                                    "general"),
    ("Good morning! Can you help me understand Indian law?",                "general"),
    ("Who are you and what services do you provide?",                       "general"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KG LOOKUP TEST SET
# ═══════════════════════════════════════════════════════════════════════════════

KG_LOOKUP_CASES = [
    ("420", "Indian Penal Code",          ["415", "417", "318"],    ["cheating"]),
    ("302", "Indian Penal Code",          ["300", "101"],           ["murder"]),
    ("138", "Negotiable Instruments Act", ["139", "141"],           ["cheque_bounce"]),
    ("154", "Code of Criminal Procedure", ["173"],                  ["fir"]),
    ("379", "Indian Penal Code",          ["378", "303"],           ["theft"]),
    ("438", "Code of Criminal Procedure", ["482"],                  ["bail"]),
    ("376", "Indian Penal Code",          ["63"],                   ["rape"]),
    ("498A", "Indian Penal Code",         ["85"],                   ["cruelty"]),
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _gemini_judge(query: str, context: str, answer: str, api_key: str) -> dict:
    """
    Ask Gemini 2.0 Flash to score a RAG answer on 4 dimensions.
    Returns dict with scores 1-5 and reasoning.
    """
    from google import genai
    from google.genai import types

    prompt = f"""You are an Indian legal expert evaluating an AI legal assistant's answer.

QUERY: {query}

RETRIEVED CONTEXT (what the AI had access to):
{context[:1500]}

AI ANSWER:
{answer[:1000]}

Score this answer on each dimension from 1 to 5:
1 = very poor, 3 = acceptable, 5 = excellent

Respond ONLY with valid JSON, no explanation outside the JSON:
{{
  "faithfulness": <1-5>,
  "faithfulness_reason": "<one sentence>",
  "relevance": <1-5>,
  "relevance_reason": "<one sentence>",
  "completeness": <1-5>,
  "completeness_reason": "<one sentence>",
  "legal_accuracy": <1-5>,
  "legal_accuracy_reason": "<one sentence>",
  "overall": <1-5>
}}

Faithfulness: Does the answer stay within what the context says? No hallucination?
Relevance: Does it actually answer the specific query asked?
Completeness: Are key legal details (section numbers, penalties, conditions) covered?
Legal Accuracy: Are the cited laws, sections and legal principles correct for Indian law?"""

    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model    = "gemini-2.0-flash",
        contents = prompt,
        config   = types.GenerateContentConfig(
            temperature       = 0.1,
            max_output_tokens = 400,
        ),
    )
    raw = response.text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


def _section_ids_in_text(text: str) -> set[str]:
    """Extract section numbers mentioned in text."""
    return set(re.findall(r'\bSection\s+(\d+[A-Z]?)\b', text, re.IGNORECASE))


def _acts_in_text(text: str) -> set[str]:
    """Check which of a list of acts are mentioned in text."""
    found = set()
    act_keywords = {
        "Indian Penal Code":                   ["Indian Penal Code", "IPC"],
        "Bharatiya Nyaya Sanhita":             ["Bharatiya Nyaya Sanhita", "BNS"],
        "Code of Criminal Procedure":          ["Code of Criminal Procedure", "CrPC"],
        "Bharatiya Nagarik Suraksha Sanhita":  ["Bharatiya Nagarik Suraksha Sanhita", "BNSS"],
        "Negotiable Instruments Act":          ["Negotiable Instruments Act", "NI Act"],
        "Indian Evidence Act":                 ["Indian Evidence Act", "IEA"],
    }
    for act, keywords in act_keywords.items():
        if any(kw.lower() in text.lower() for kw in keywords):
            found.add(act)
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR 1 — RAG RETRIEVAL QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

def eval_rag_retrieval(all_results: list) -> dict:
    print("\n" + "═" * 65)
    print("DIMENSION 1 — RAG RETRIEVAL QUALITY  (Precision@5, Recall@5, MRR)")
    print("═" * 65)

    from rag.pipeline import RAGPipeline
    from rag.hybrid_search import hybrid_searcher

    pipeline = RAGPipeline(enable_rewriting=False, enable_reranking=False)

    precision_scores = []
    recall_scores    = []
    mrr_scores       = []
    K = 5

    for case in RAG_TEST_CASES:
        query    = case["query"]
        exp_secs = set(case["expected_sections"])

        try:
            chunks = hybrid_searcher.search(query, n_results=K)
            retrieved_sections = {c.get("section", "") for c in chunks}

            tp          = len(exp_secs & retrieved_sections)
            precision   = tp / K
            recall      = tp / len(exp_secs) if exp_secs else 0.0

            # MRR — rank of first relevant chunk
            rr = 0.0
            for i, chunk in enumerate(chunks, 1):
                if chunk.get("section", "") in exp_secs:
                    rr = 1.0 / i
                    break

            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(rr)

            mark = "OK" if precision > 0 else "FAIL"
            print(f"  {mark}  P@5={precision:.2f} R@5={recall:.2f} MRR={rr:.2f}  {query[:52]!r}")

        except Exception as e:
            print(f"  FAIL  ERROR: {e}  {query[:50]!r}")
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            mrr_scores.append(0.0)

    avg_p   = sum(precision_scores) / len(precision_scores) * 100
    avg_r   = sum(recall_scores)    / len(recall_scores)    * 100
    avg_mrr = sum(mrr_scores)       / len(mrr_scores)

    print(f"\n  Avg Precision@5 : {avg_p:.1f}%")
    print(f"  Avg Recall@5    : {avg_r:.1f}%")
    print(f"  Avg MRR         : {avg_mrr:.3f}")

    score = (avg_p + avg_r) / 2
    all_results.append(("RAG Retrieval (avg P@5+R@5)/2", round(score, 1)))
    return {"precision": avg_p, "recall": avg_r, "mrr": avg_mrr}


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR 2 — RAG ANSWER QUALITY (LLM-as-Judge)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_rag_answer_quality(all_results: list) -> dict:
    print("\n" + "═" * 65)
    print("DIMENSION 2 — RAG ANSWER QUALITY  (LLM-as-Judge via Gemini)")
    print("═" * 65)

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        print("  ⚠  GEMINI_API_KEY not set — skipping LLM-as-Judge evaluation")
        all_results.append(("RAG Answer Quality (LLM Judge)", "SKIPPED"))
        return {}

    from rag.pipeline import rag_pipeline

    dim_scores: dict[str, list] = {
        "faithfulness":   [],
        "relevance":      [],
        "completeness":   [],
        "legal_accuracy": [],
        "overall":        [],
    }

    # Evaluate first 5 cases only — saves Gemini quota
    for case in RAG_TEST_CASES[:5]:
        query = case["query"]
        try:
            print(f"  Evaluating: {query[:55]!r}")
            legal_answer = rag_pipeline.query(query)
            answer       = legal_answer.answer_text

            # Build context string from what was retrieved
            context_str = f"Sources consulted: {legal_answer.sources_consulted}. " \
                          f"Note: {legal_answer.synthesis_note}"

            time.sleep(4)  # Gemini 15 RPM rate limit
            scores = _gemini_judge(query, context_str, answer, gemini_key)

            for dim in dim_scores:
                dim_scores[dim].append(scores.get(dim, 3))

            print(
                f"    F={scores['faithfulness']} "
                f"R={scores['relevance']} "
                f"C={scores['completeness']} "
                f"LA={scores['legal_accuracy']} "
                f"Overall={scores['overall']}"
            )
            print(f"    Faithfulness: {scores['faithfulness_reason']}")

        except Exception as e:
            print(f"  FAIL  ERROR for query: {e}")
            for dim in dim_scores:
                dim_scores[dim].append(3)

    avgs = {dim: sum(v) / len(v) for dim, v in dim_scores.items() if v}

    print(f"\n  Avg Faithfulness   : {avgs.get('faithfulness',   0):.2f}/5")
    print(f"  Avg Relevance      : {avgs.get('relevance',       0):.2f}/5")
    print(f"  Avg Completeness   : {avgs.get('completeness',    0):.2f}/5")
    print(f"  Avg Legal Accuracy : {avgs.get('legal_accuracy',  0):.2f}/5")
    print(f"  Avg Overall        : {avgs.get('overall',         0):.2f}/5")

    overall_pct = avgs.get("overall", 0) / 5 * 100
    all_results.append(("RAG Answer Quality (LLM Judge /5)", round(avgs.get("overall", 0), 2)))
    return avgs


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR 3 — RAG GROUNDING
# ═══════════════════════════════════════════════════════════════════════════════

def eval_rag_grounding(all_results: list) -> dict:
    print("\n" + "═" * 65)
    print("DIMENSION 3 — RAG ANSWER GROUNDING  (Section + Act citation check)")
    print("═" * 65)

    from rag.pipeline import rag_pipeline

    grounded       = 0
    section_hits   = 0
    act_hits       = 0
    total          = len(RAG_TEST_CASES)

    for case in RAG_TEST_CASES:
        query    = case["query"]
        exp_secs = case["expected_sections"]
        exp_acts = case["expected_acts"]

        try:
            legal_answer  = rag_pipeline.query(query)
            answer        = legal_answer.answer_text

            found_secs    = _section_ids_in_text(answer)
            found_acts    = _acts_in_text(answer)

            sec_ok = any(s in found_secs for s in exp_secs)
            act_ok = any(a in found_acts for a in exp_acts)

            # Check answer_must_contain keywords
            keywords      = case.get("answer_must_contain", [])
            kw_ok         = sum(1 for kw in keywords if kw.lower() in answer.lower())
            kw_pct        = kw_ok / len(keywords) if keywords else 1.0

            case_grounded = sec_ok and act_ok and kw_pct >= 0.5

            section_hits += int(sec_ok)
            act_hits     += int(act_ok)
            grounded     += int(case_grounded)

            mark = "OK" if case_grounded else ("~" if (sec_ok or act_ok) else "FAIL")
            print(
                f"  {mark}  sec={'OK' if sec_ok else 'FAIL'} "
                f"act={'OK' if act_ok else 'FAIL'} "
                f"kw={kw_ok}/{len(keywords)}  "
                f"{query[:48]!r}"
            )

        except Exception as e:
            print(f"  FAIL  ERROR: {e}")

    sec_pct = section_hits / total * 100
    act_pct = act_hits     / total * 100
    grd_pct = grounded     / total * 100

    print(f"\n  Section citation rate : {sec_pct:.1f}%  ({section_hits}/{total})")
    print(f"  Act citation rate     : {act_pct:.1f}%  ({act_hits}/{total})")
    print(f"  Fully grounded        : {grd_pct:.1f}%  ({grounded}/{total})")

    all_results.append(("RAG Grounding (full)", round(grd_pct, 1)))
    return {"section_pct": sec_pct, "act_pct": act_pct, "grounded_pct": grd_pct}


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR 4 — INTENT CLASSIFIER F1
# ═══════════════════════════════════════════════════════════════════════════════

def eval_intent_classifier(all_results: list) -> dict:
    print("\n" + "═" * 65)
    print("DIMENSION 4 — INTENT CLASSIFIER  (Accuracy + Per-class F1, n=30)")
    print("═" * 65)

    from agents.intent_classifier import intent_classifier

    intents = ["legal_query", "document_analysis", "draft_request",
               "risk_check", "translation_request", "general"]

    tp_map: dict[str, int] = defaultdict(int)
    fp_map: dict[str, int] = defaultdict(int)
    fn_map: dict[str, int] = defaultdict(int)
    correct = 0

    for query, expected in INTENT_TEST_CASES:
        result = intent_classifier.classify(query)
        got    = result.intent

        if got == expected:
            correct         += 1
            tp_map[expected] += 1
        else:
            fp_map[got]      += 1
            fn_map[expected] += 1

    accuracy = correct / len(INTENT_TEST_CASES) * 100

    print(f"\n  {'Intent':<22} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print(f"  {'-'*22}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*6}")

    f1_scores = []
    for intent in intents:
        tp   = tp_map[intent]
        fp   = fp_map[intent]
        fn   = fn_map[intent]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        print(f"  {intent:<22} {tp:>4} {fp:>4} {fn:>4} {prec*100:>9.1f}% {rec*100:>7.1f}% {f1*100:>5.1f}%")

    macro_f1 = sum(f1_scores) / len(f1_scores) * 100
    print(f"\n  Accuracy   : {accuracy:.1f}%  ({correct}/{len(INTENT_TEST_CASES)})")
    print(f"  Macro F1   : {macro_f1:.1f}%")

    all_results.append(("Intent Classifier Accuracy", round(accuracy, 1)))
    all_results.append(("Intent Classifier Macro F1", round(macro_f1, 1)))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR 5 — KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def eval_knowledge_graph(all_results: list) -> dict:
    print("\n" + "═" * 65)
    print("DIMENSION 5 — KNOWLEDGE GRAPH  (Connectivity + Lookup Precision)")
    print("═" * 65)

    from rag.knowledge_graph import get_kg
    import networkx as nx

    kg    = get_kg()
    stats = kg.stats()
    G     = kg.graph

    # ── Connectivity metrics ───────────────────────────────────────────────────
    density     = nx.density(G)
    n_components = nx.number_connected_components(G)
    avg_degree  = sum(d for _, d in G.degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

    section_nodes = [n for n, a in G.nodes(data=True) if a.get("node_type") == "section"]
    concept_nodes = [n for n, a in G.nodes(data=True) if a.get("node_type") == "concept"]
    statute_nodes = [n for n, a in G.nodes(data=True) if a.get("node_type") == "statute"]

    print(f"\n  Graph Connectivity:")
    print(f"    Total nodes       : {stats['nodes']}")
    print(f"    Total edges       : {stats['edges']}")
    print(f"    Section nodes     : {len(section_nodes)}")
    print(f"    Concept nodes     : {len(concept_nodes)}")
    print(f"    Statute nodes     : {len(statute_nodes)}")
    print(f"    Graph density     : {density:.4f}")
    print(f"    Avg degree        : {avg_degree:.2f}")
    print(f"    Connected parts   : {n_components}")

    # ── Lookup precision ───────────────────────────────────────────────────────
    print(f"\n  Lookup Precision (expected sections + concepts found within 2 hops):")
    print(f"  {'Section':<10} {'Source':<30} {'Sec Hits':>8} {'Con Hits':>8}  Result")
    print(f"  {'-'*10}  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*6}")

    lookup_scores = []

    for sec, source, exp_secs, exp_cons in KG_LOOKUP_CASES:
        related  = kg.query_related_sections(sec, source_hint=source)
        got_secs = {r["section"] for r in related if r["node_type"] == "section"}
        got_cons = {r["concept"] for r in related if r["node_type"] == "concept"}

        sec_hit = sum(1 for s in exp_secs if s in got_secs)
        con_hit = sum(1 for c in exp_cons if c in got_cons)

        total_exp = len(exp_secs) + len(exp_cons)
        total_hit = sec_hit + con_hit
        prec      = total_hit / total_exp if total_exp > 0 else 0.0
        lookup_scores.append(prec)

        mark = "OK" if prec >= 0.5 else "FAIL"
        print(f"  {sec:<10}  {source[:30]:<30}  {sec_hit}/{len(exp_secs):>5}   {con_hit}/{len(exp_cons):>5}  {mark} {prec*100:.0f}%")

    avg_lookup = sum(lookup_scores) / len(lookup_scores) * 100

    # ── Coverage — what % of corpus sections have at least 1 KG edge ──────────
    total_section_nodes = len(section_nodes)
    has_edges           = sum(1 for n in section_nodes if G.degree(n) > 1)  # >1 means edge beyond belongs_to
    coverage            = has_edges / total_section_nodes * 100 if total_section_nodes > 0 else 0

    print(f"\n  Avg Lookup Precision : {avg_lookup:.1f}%")
    print(f"  Relationship Coverage: {coverage:.1f}%  ({has_edges}/{total_section_nodes} sections have cross-edges)")

    all_results.append(("KG Lookup Precision",    round(avg_lookup, 1)))
    all_results.append(("KG Relationship Coverage", round(coverage, 1)))
    return {"lookup_precision": avg_lookup, "density": density, "coverage": coverage}


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR 6 — DRAFTING AGENT (Real LLM output)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_drafting_agent_real(all_results: list) -> dict:
    print("\n" + "═" * 65)
    print("DIMENSION 6 — DRAFTING AGENT  (Real LLM structural completeness)")
    print("═" * 65)

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        print("  ⚠  GEMINI_API_KEY not set — skipping real LLM draft evaluation")
        all_results.append(("Drafting Agent Real LLM", "SKIPPED"))
        return {}

    from agents.drafting_agent import DraftingAgent

    REAL_DRAFT_CASES = [
        {
            "doc_type": "fir",
            "turns":    [
                "Help me draft a written complaint to police for mobile phone theft",
                "My Samsung Galaxy S23 worth Rs 80000 was stolen from my bag at Ernakulam railway station on 5 May 2026 at 3pm. I was waiting on platform 2 and noticed the phone missing after boarding the train.",
                "My name is Anantha Krishnan K, residing at Flat 4B Green Valley Apartments Kakkanad Kochi 682030. Contact 9876543210. I don't know the accused. I want the complaint registered and phone recovered.",
            ],
            "required_words": ["complaint", "police", "stolen", "Anantha", "Ernakulam", "Section"],
        },
        {
            "doc_type": "legal_notice_ni",
            "turns":    [
                "Draft a cheque bounce legal notice under Section 138 NI Act",
                "Cheque number 998877, Rs 2,00,000, dated 15 April 2026, HDFC Bank Kakkanad branch. Dishonoured on 20 April 2026, reason: Insufficient Funds. It was for repayment of a business loan.",
                "I am Ravi Kumar, 12 MG Road Kochi 682001. The drawer is Suresh Menon, 45 Park Avenue Thrissur 680001. I demand Rs 2,00,000 within 15 days of receipt of this notice.",
            ],
            "required_words": ["138", "Ravi Kumar", "Suresh Menon", "2,00,000", "15 days", "dishonoured"],
        },
    ]

    total_words  = 0
    found_words  = 0
    min_length   = 200  # chars
    length_ok    = 0

    for case in REAL_DRAFT_CASES:
        agent = DraftingAgent()
        sid   = f"real-{case['doc_type']}"

        print(f"\n  Generating real {case['doc_type']} draft via Gemini...")

        r1 = agent.handle(case["turns"][0], session_id=sid)
        print(f"    Turn 1 stage={r1['stage']} OK")

        r2 = agent.handle(case["turns"][1], session_id=sid)
        print(f"    Turn 2 stage={r2['stage']} OK")

        time.sleep(5)  # Rate limit
        r3 = agent.handle(case["turns"][2], session_id=sid)
        draft = r3.get("draft", "")

        print(f"    Turn 3 stage={r3['stage']} complete={r3['complete']} len={len(draft)} chars")

        req_words = case["required_words"]
        found     = [w for w in req_words if w.lower() in draft.lower()]
        missing   = [w for w in req_words if w.lower() not in draft.lower()]

        total_words += len(req_words)
        found_words += len(found)

        if len(draft) >= min_length:
            length_ok += 1

        mark = "OK" if len(missing) == 0 else "~"
        print(f"    {mark} Keywords: {len(found)}/{len(req_words)} found", end="")
        if missing:
            print(f"  [missing: {', '.join(missing)}]")
        else:
            print()
        print(f"    Draft length: {len(draft)} chars {'OK' if len(draft) >= min_length else 'FAIL (too short)'}")

        # Save draft to file for manual review
        os.makedirs("tests/eval_results", exist_ok=True)
        path = f"tests/eval_results/real_draft_{case['doc_type']}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(draft)
        print(f"    Draft saved -> {path}")

    kw_score     = found_words / total_words * 100 if total_words > 0 else 0
    length_score = length_ok   / len(REAL_DRAFT_CASES) * 100

    print(f"\n  Keyword Coverage  : {kw_score:.1f}%  ({found_words}/{total_words})")
    print(f"  Length Adequacy   : {length_score:.1f}%  ({length_ok}/{len(REAL_DRAFT_CASES)} drafts >{min_length} chars)")
    print(f"  ℹ  Review actual drafts in tests/eval_results/ for legal quality")

    overall = (kw_score + length_score) / 2
    all_results.append(("Drafting Agent Real LLM", round(overall, 1)))
    return {"keyword_coverage": kw_score, "length_score": length_score}


def eval_structured_output_and_translation(all_results: list) -> dict:
    print("\n" + "═" * 65)
    print("DIMENSION 7 — STRUCTURED OUTPUT + TRANSLATION  (automated)")
    print("═" * 65)

    from rag.structured_output import build_structured_response, _extract_summary, _extract_key_clauses
    from agents.translation_agent import detect_language, _strip_legal_content

    # ── Structured output field checks ────────────────────────────────────────
    ANSWER_SAMPLES = [
        ("Section 302 Indian Penal Code prescribes death or life imprisonment for murder. "
         "Section 101 BNS is the equivalent under the new law. The court considers rarest of rare cases. [1][2]",
         "legal_query", ["Section 302", "Section 101", "BNS", "Indian Penal Code"]),
        ("Section 138 NI Act makes cheque bounce a criminal offence punishable with "
         "imprisonment up to 2 years or fine or both. [1]",
         "risk_check", ["Section 138", "NI Act"]),
        ("Your rental agreement contains a clause on maintenance. "
         "Review Section 108 Transfer of Property Act for landlord obligations. [1]",
         "document_analysis", ["Section 108", "Transfer of Property Act"]),
        ("Hello! I am LexShield AI. I help with Indian legal questions.",
         "general", []),
        ("The drafting agent is being built.", "draft_request", []),
    ]

    field_scores  = []
    clause_scores = []

    print(f"\n  Structured Output — Field Population:")
    for answer, intent, exp_clauses in ANSWER_SAMPLES:
        resp = build_structured_response(
            answer_text = answer,
            intent      = intent,
            session_id  = "eval",
            confidence  = 0.9,
            mode        = "test",
        )
        d = resp.to_dict()

        required = ["answer_text", "summary", "key_clauses", "suggestions",
                    "risk", "citations", "intent", "session_id"]
        populated = sum(1 for f in required if d.get(f) is not None)
        field_scores.append(populated / len(required))

        summary_ok  = len(resp.summary) > 0
        risk_ok     = 0.0 <= resp.risk_score <= 1.0
        suggest_ok  = len(resp.suggestions) > 0

        mark = "OK" if (summary_ok and risk_ok and suggest_ok) else "~"
        print(f"  {mark}  [{intent:20}] fields={populated}/{len(required)} "
              f"summary={'OK' if summary_ok else 'FAIL'} "
              f"risk={'OK' if risk_ok else 'FAIL'} "
              f"suggest={'OK' if suggest_ok else 'FAIL'}")

        # Clause extraction check
        if exp_clauses:
            clauses = _extract_key_clauses(answer, citations=[])
            hits    = sum(1 for ec in exp_clauses
                         if any(ec.lower() in c.lower() for c in clauses))
            clause_scores.append(hits / len(exp_clauses))

    avg_fields  = sum(field_scores)  / len(field_scores)  * 100
    avg_clauses = sum(clause_scores) / len(clause_scores)  * 100 if clause_scores else 0.0

    print(f"\n  Avg field population   : {avg_fields:.1f}%")
    print(f"  Avg clause extraction  : {avg_clauses:.1f}%")

    # ── Translation language detection ────────────────────────────────────────
    LANG_CASES = [
        ("What is Section 302 IPC?",                      True,  None,         None),
        ("Explain Section 138 in Malayalam",              True,  None,         "Malayalam"),
        ("Translate this to Hindi: What is bail?",        True,  None,         "Hindi"),
        ("வகுப்பு 302 ஐபிசி என்ன?",                      False, "Tamil",      None),
        ("धारा 302 आईपीसी क्या है?",                       False, "Hindi",      None),
        ("വകുപ്പ് 138 NI ആക്ട് എന്താണ്?",                False, "Malayalam",  None),
        ("Explain bail in Kannada",                       True,  None,         "Kannada"),
        ("Section 302 IPC punishment explain in Telugu",  True,  None,         "Telugu"),
    ]

    print(f"\n  Translation Detection (n={len(LANG_CASES)}):")
    lang_hits = 0
    for query, exp_eng, exp_script, exp_target in LANG_CASES:
        r  = detect_language(query)
        ok = (r.is_english == exp_eng and
              r.detected_script == exp_script and
              r.target_language == exp_target)
        lang_hits += int(ok)
        mark = "OK" if ok else "FAIL"
        print(f"  {mark}  is_eng={r.is_english} script={r.detected_script} "
              f"target={r.target_language}  {query[:40]!r}")

    lang_score = lang_hits / len(LANG_CASES) * 100
    print(f"\n  Language Detection Accuracy : {lang_score:.1f}%  ({lang_hits}/{len(LANG_CASES)})")

    overall = (avg_fields + avg_clauses + lang_score) / 3
    all_results.append(("Structured Output Field Population", round(avg_fields, 1)))
    all_results.append(("Clause Extraction Accuracy",         round(avg_clauses, 1)))
    all_results.append(("Translation Detection Accuracy",     round(lang_score, 1)))
    return {"fields": avg_fields, "clauses": avg_clauses, "lang": lang_score}


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_final_report(all_results: list, elapsed: float):
    print("\n" + "═" * 65)
    print("WEEK 3 EVALUATION SUMMARY")
    print("═" * 65)

    numeric = [(n, v) for n, v in all_results if isinstance(v, (int, float))]
    skipped = [(n, v) for n, v in all_results if not isinstance(v, (int, float))]

    print(f"\n  {'Metric':<40} {'Score':>10}")
    print(f"  {'-'*40}  {'-'*10}")
    for name, score in numeric:
        unit = "/5" if "/5" in name else "%"
        print(f"  {name:<40} {score:>9.1f}{unit}")
    for name, val in skipped:
        print(f"  {name:<40} {val:>10}")

    if numeric:
        scores_pct = []
        for name, score in numeric:
            if "/5" in name:
                scores_pct.append(score / 5 * 100)
            else:
                scores_pct.append(score)
        overall = sum(scores_pct) / len(scores_pct)
        grade   = ("EXCELLENT" if overall >= 85 else
                   "GOOD"      if overall >= 70 else
                   "FAIR"      if overall >= 55 else "NEEDS WORK")
        print(f"\n  {'OVERALL AVERAGE':<40} {overall:>9.1f}%")
        print(f"  Grade: {grade}")

    print(f"\n  Elapsed   : {elapsed:.1f}s")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  ℹ  Drafts for manual review -> tests/eval_results/")

    # Save report
    os.makedirs("tests/eval_results", exist_ok=True)
    path = "tests/eval_results/week3_eval.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("LexShield AI — Week 3 Full Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")
        for name, score in all_results:
            unit = "/5" if "/5" in str(name) else "%"
            f.write(f"{name:<40} {score}\n")
        if numeric:
            f.write(f"\nOVERALL AVERAGE: {overall:.1f}%\n")
            f.write(f"Grade: {grade}\n")
        f.write(f"\nElapsed: {elapsed:.1f}s\n")
    print(f"  Report saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("LexShield AI — Week 3 Full Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This will make real API calls. Estimated time: 5-10 minutes.\n")

    all_results = []
    t0 = time.time()

    eval_intent_classifier(all_results)      # Fast, no API calls
    eval_knowledge_graph(all_results)        # Fast, no API calls
    eval_rag_retrieval(all_results)          # ChromaDB + BM25 only, no LLM
    eval_rag_grounding(all_results)          # Real Groq calls (~10 queries)
    eval_rag_answer_quality(all_results)     # Real Gemini calls (~5 queries)
    eval_drafting_agent_real(all_results)    # Real Gemini calls (~2 drafts)
    eval_structured_output_and_translation(all_results)

    write_final_report(all_results, time.time() - t0)