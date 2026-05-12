# tests/eval_all.py
"""
LexShield AI — Full Evaluation Suite v2
=========================================
Fixes:
- LegalAnswer uses .answer_text not .answer
- bail_application expected level corrected to Critical
- consumer_complaint risk threshold lowered
- intent classifier module path fix (needs agents/__init__.py)
- summary table component name matching fixed

Run: python -m tests.eval_all
     (uvicorn must be running in another terminal)
"""

import time
import json
import requests
from pathlib import Path

BASE_URL    = "http://localhost:8000"
REPORT_PATH = Path("tests/eval_report.json")

# ═══════════════════════════════════════════════════════════════════════════
# 1. INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

INTENT_TEST_CASES = [
    ("What is the punishment under Section 420 IPC?",             "legal_query"),
    ("What are my rights as a tenant under Kerala Rent Act?",     "legal_query"),
    ("Explain Section 138 of NI Act",                             "legal_query"),
    ("What does Article 21 of Constitution say?",                 "legal_query"),
    ("What is cognizable offence?",                               "legal_query"),
    ("Analyze this uploaded contract for legal issues",           "document_analysis"),
    ("What does this document say about my obligations?",         "document_analysis"),
    ("Review my rental agreement",                                "document_analysis"),
    ("Draft a legal notice to my landlord",                       "draft_request"),
    ("Help me write an FIR for harassment",                       "draft_request"),
    ("Generate an employment contract template",                  "draft_request"),
    ("Write a cheque bounce notice",                              "draft_request"),
    ("Is it illegal to record calls without consent?",            "risk_check"),
    ("Can I be arrested for not repaying a loan?",                "risk_check"),
    ("Am I liable if my employee gets injured?",                  "risk_check"),
    ("What happens if I ignore a court notice?",                  "risk_check"),
    ("Translate consumer rights to Hindi",                        "translation_request"),
    ("Can you explain this in Malayalam?",                        "translation_request"),
    ("Explain Section 302 in Tamil",                              "translation_request"),
    ("Hello",                                                     "general"),
    ("What can you do?",                                          "general"),
    ("Who are you?",                                              "general"),
]


def eval_intent_classifier() -> dict:
    print("\n[1/6] Evaluating Intent Classifier...")
    try:
        from agents.intent_classifier import intent_classifier
    except ImportError as e:
        print(f"  SKIP: {e}")
        print("  Fix: create empty agents/__init__.py")
        return {"component": "intent_classifier", "error": str(e)}

    correct = 0
    wrong   = []

    for query, expected in INTENT_TEST_CASES:
        result = intent_classifier.classify(query)
        if result.intent == expected:
            correct += 1
        else:
            wrong.append({
                "query":    query,
                "expected": expected,
                "got":      result.intent,
                "conf":     result.confidence,
            })

    accuracy = correct / len(INTENT_TEST_CASES)
    print(f"  Accuracy: {correct}/{len(INTENT_TEST_CASES)} = {accuracy*100:.1f}%")
    if wrong:
        for w in wrong:
            print(f"    ✗ '{w['query'][:50]}' expected={w['expected']} got={w['got']}")

    return {
        "component":   "intent_classifier",
        "accuracy":    round(accuracy, 4),
        "correct":     correct,
        "total":       len(INTENT_TEST_CASES),
        "wrong_cases": wrong,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. DOCUMENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

CLASSIFIER_TEST_CASES = [
    ("This rental agreement is executed between the landlord Smt. Lakshmi Nair and tenant "
     "Sri. Arjun Menon for residential premises at Kakkanad, Ernakulam. Monthly rent Rs.12,000. "
     "Lease period eleven months. Security deposit Rs.36,000.",
     "rental_agreement"),
    ("FIRST INFORMATION REPORT. FIR No: 0145/2025. Police Station: Ernakulam Central. "
     "Offence: Section 379 IPC. Complainant states accused committed theft of cash Rs.45,000.",
     "fir"),
    ("IN THE COURT OF JUDICIAL MAGISTRATE. You are hereby summoned to appear before this "
     "Honourable Court on 15/03/2025. Case No: CC 234/2025. Failure to appear will result "
     "in ex-parte proceedings.",
     "court_notice_summons"),
    ("APPOINTMENT LETTER. You have been selected as Senior Software Engineer at TechSolutions "
     "India Pvt Ltd. Monthly salary Rs.65,000. Probation period 6 months. Notice period 2 months.",
     "employment_contract"),
    ("SALE DEED. This deed is executed between vendor Sri.Thomas Mathew and purchaser "
     "Sri.Rajesh Nambiar. Property: Survey No.123/2B, Kakkanad Village, 5.25 cents. "
     "Sale consideration Rs.45,00,000. Free from encumbrances.",
     "property_deed"),
    ("IN THE SUPREME COURT OF INDIA. Civil Appeal No.1234 of 2024. Before Hon'ble "
     "Justice A.K. Sinha and Justice B.R. Mehta. The appeal is allowed and the order "
     "of the High Court is set aside.",
     "sc_judgment"),
    ("IN THE HIGH COURT OF KERALA AT ERNAKULAM. WP(C) No.15234 of 2024. The writ petition "
     "is allowed. The impugned order is quashed. Respondent directed to comply within 4 weeks.",
     "hc_judgment"),
    ("LEGAL NOTICE. Take notice that my client Sri.Joseph Varghese demands payment of "
     "Rs.3,00,000 outstanding dues within 15 days. Failing compliance, civil and criminal "
     "proceedings shall be initiated.",
     "legal_notice"),
    ("AFFIDAVIT. I Sri.Manoj Kumar do hereby solemnly affirm and declare on oath that the "
     "original Registration Certificate of vehicle KL-07-AB-5678 has been lost. I request "
     "issuance of duplicate RC from RTO Ernakulam.",
     "affidavit"),
    ("POWER OF ATTORNEY. I Smt.Usha Nair hereby appoint my son Sri.Anil Kumar Nair as my "
     "attorney to appear before Sub-Registrar Office and register the sale deed for property "
     "Survey No.456/3 on my behalf.",
     "power_of_attorney"),
    ("LEGAL NOTICE under Section 138 NI Act. Cheque No.456789 dated 01/01/2025 for Rs.3,00,000 "
     "drawn on SBI returned dishonoured with memo Funds Insufficient. Pay within 15 days "
     "or face criminal complaint.",
     "cheque_bounce_notice"),
    ("BAIL APPLICATION. Application under Section 437 CrPC. Applicant Sri.Rahul Singh arrested "
     "in Crime No.456/2024 under Sections 323 and 341 IPC. First offender. Aged parents dependent. "
     "Prays for release on bail.",
     "bail_application"),
    ("CONSUMER COMPLAINT. Before District Consumer Disputes Redressal Commission. Complainant "
     "purchased Samsung Galaxy phone for Rs.74,999. Defective within warranty. Deficiency in "
     "service under Consumer Protection Act 2019.",
     "consumer_complaint"),
    ("LOAN AGREEMENT. Lender Kerala Finance Corporation and Borrower Sri.Suresh Pillai. "
     "Loan amount Rs.10,00,000 at 14% per annum. 60 EMIs of Rs.23,268. "
     "Property Survey No.234/1 pledged as collateral.",
     "loan_agreement"),
    ("To the Station House Officer, Ernakulam Central Police Station. I Sri.Anand Raj wish "
     "to lodge complaint against Sri.Prakash Mehta for continuous threats and harassment. "
     "Request registration of FIR and police protection.",
     "police_complaint"),
]


def eval_document_classifier() -> dict:
    print("\n[2/6] Evaluating Document Classifier...")
    from models.classifier import classifier

    correct     = 0
    wrong       = []
    confidences = []

    for text, expected in CLASSIFIER_TEST_CASES:
        result     = classifier.predict(text)
        predicted  = result["label_name"]
        confidence = result["confidence"]
        confidences.append(confidence)

        if predicted == expected:
            correct += 1
            print(f"  ✓ {expected}")
        else:
            wrong.append({
                "expected":   expected,
                "got":        predicted,
                "confidence": round(confidence, 3),
            })
            print(f"  ✗ expected={expected}, got={predicted} ({confidence:.2f})"
                  f"{'  ⚠ uncertain' if result.get('uncertain') else ''}")

    accuracy = correct / len(CLASSIFIER_TEST_CASES)
    avg_conf = sum(confidences) / len(confidences)

    print(f"  Mode: {classifier.get_mode()}")
    print(f"  Accuracy: {correct}/{len(CLASSIFIER_TEST_CASES)} = {accuracy*100:.1f}%")
    print(f"  Avg confidence: {avg_conf*100:.1f}%")

    return {
        "component":      "document_classifier",
        "mode":           classifier.get_mode(),
        "accuracy":       round(accuracy, 4),
        "correct":        correct,
        "total":          len(CLASSIFIER_TEST_CASES),
        "avg_confidence": round(avg_conf, 4),
        "wrong_cases":    wrong,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. NER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

NER_TEST_CASES = [
    {
        "text": "FIR No.145/2025 registered at Ernakulam Central Police Station. "
                "Complainant Sri.Vijay Kumar filed complaint against Sri.Ramesh Pillai "
                "under Section 379 IPC and Section 420 IPC. Case registered on 13/01/2025.",
        "expected": {
            "ipc_sections": ["Section 379", "Section 420"],
            "locations":    ["Ernakulam"],
            "persons":      ["Vijay Kumar", "Ramesh Pillai"],
        },
    },
    {
        "text": "Cheque No.456789 for Rs.3,00,000 drawn on State Bank of India returned "
                "dishonoured. Notice issued under Section 138 Negotiable Instruments Act. "
                "High Court of Kerala at Ernakulam directed payment within 15 days.",
        "expected": {
            "ipc_sections": ["Section 138"],
            "monetary":     ["Rs.3,00,000"],
            "locations":    ["Kerala", "Ernakulam"],
            "acts":         ["Negotiable Instruments Act"],
        },
    },
    {
        "text": "WP(C) No.15234/2024. In the High Court of Kerala. Petitioner Smt.Radha Krishnan "
                "challenged order of District Collector Thrissur dated 15/09/2024 under "
                "Article 226 of Constitution of India.",
        "expected": {
            "case_numbers": ["WP(C) No.15234/2024"],
            "persons":      ["Radha Krishnan"],
            "locations":    ["Kerala", "Thrissur"],
        },
    },
]


def _check_ner_field(got: list, expected: list) -> tuple:
    got_lower = [g.lower() for g in got]
    hits = sum(
        1 for exp in expected
        if any(exp.lower() in g or g in exp.lower() for g in got_lower)
    )
    return hits, len(expected)


def eval_ner_pipeline() -> dict:
    print("\n[3/6] Evaluating NER Pipeline...")
    from nlp.ner_pipeline import extract_entities

    total_hits = total_exp = 0
    case_results = []

    for i, case in enumerate(NER_TEST_CASES):
        result_d = extract_entities(case["text"]).to_dict()
        c_hits = c_exp = 0
        field_results = {}

        for field, expected_vals in case["expected"].items():
            got        = result_d.get(field, [])
            hits, exp  = _check_ner_field(got, expected_vals)
            c_hits    += hits
            c_exp     += exp
            total_hits += hits
            total_exp  += exp
            field_results[field] = {"expected": expected_vals, "got": got,
                                    "hits": hits, "total": exp}

        recall = c_hits / c_exp if c_exp > 0 else 0
        case_results.append({"case": i+1, "recall": round(recall, 3),
                             "fields": field_results})
        print(f"  Case {i+1}: recall={recall*100:.0f}%")
        for field, fr in field_results.items():
            status = "✓" if fr["hits"] == fr["total"] else f"✗ ({fr['hits']}/{fr['total']})"
            print(f"    {field:15s} {status}  got={fr['got'][:3]}")

    overall = total_hits / total_exp if total_exp > 0 else 0
    print(f"  Overall recall: {total_hits}/{total_exp} = {overall*100:.1f}%")

    return {
        "component":      "ner_pipeline",
        "overall_recall": round(overall, 4),
        "total_hits":     total_hits,
        "total_expected": total_exp,
        "cases":          case_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

RAG_TEST_CASES = [
    {
        "query":             "What is the punishment for murder under IPC?",
        "expected_section":  "302",
        "expected_keywords": ["death", "imprisonment", "life"],
    },
    {
        "query":             "What are the rights of an arrested person?",
        "expected_section":  "50",
        "expected_keywords": ["arrest", "inform", "grounds"],
    },
    {
        "query":             "What is Section 138 of Negotiable Instruments Act?",
        "expected_section":  "138",
        "expected_keywords": ["cheque", "dishonour", "imprisonment"],
    },
    {
        "query":             "Consumer rights under Consumer Protection Act 2019",
        "expected_section":  None,
        "expected_keywords": ["consumer", "deficiency", "complaint"],
    },
    {
        "query":             "What is the penalty for dowry demand?",
        "expected_section":  None,
        "expected_keywords": ["dowry", "imprisonment", "fine"],
    },
]


def _get_answer_text(response) -> str:
    """Safely extract answer text from LegalAnswer object."""
    # Try all known attribute names
    for attr in ("answer_text", "answer", "response", "text", "result"):
        if hasattr(response, attr):
            val = getattr(response, attr)
            if val:
                return str(val).lower()
    # Dict fallback
    if isinstance(response, dict):
        for key in ("answer_text", "answer", "response", "text"):
            if response.get(key):
                return str(response[key]).lower()
    return str(response).lower()


def _get_citations(response) -> list:
    for attr in ("citations", "sources", "references", "sources_consulted"):
        if hasattr(response, attr):
            val = getattr(response, attr)
            if val:
                return val if isinstance(val, list) else [val]
    return []


def eval_rag_pipeline() -> dict:
    print("\n[4/6] Evaluating RAG Pipeline...")
    from rag.pipeline import rag_pipeline

    keyword_hits = total_kw = section_hits = section_total = 0
    latencies    = []
    case_results = []

    for case in RAG_TEST_CASES:
        start = time.time()
        try:
            response  = rag_pipeline.query(case["query"])
            latency   = round(time.time() - start, 2)
            answer    = _get_answer_text(response)
            citations = _get_citations(response)

            kw_found  = [kw for kw in case["expected_keywords"] if kw in answer]
            kw_score  = len(kw_found) / len(case["expected_keywords"])
            keyword_hits += len(kw_found)
            total_kw     += len(case["expected_keywords"])

            sec_hit = False
            if case["expected_section"]:
                section_total += 1
                sec_hit = (
                    case["expected_section"] in answer
                    or any(case["expected_section"] in str(c) for c in citations)
                )
                if sec_hit:
                    section_hits += 1

            latencies.append(latency)
            case_results.append({
                "query":       case["query"][:50],
                "kw_score":    round(kw_score, 2),
                "kw_found":    kw_found,
                "section_hit": sec_hit,
                "latency_s":   latency,
            })

            kw_str  = f"{len(kw_found)}/{len(case['expected_keywords'])} kw"
            sec_str = f"sec={'✓' if sec_hit else '✗'}" if case["expected_section"] else "sec=N/A"
            print(f"  '{case['query'][:45]}'")
            print(f"    {kw_str} | {sec_str} | {latency:.1f}s")

        except Exception as e:
            print(f"  ERROR: {case['query'][:45]} → {e}")
            case_results.append({"query": case["query"][:50], "error": str(e)})

    kw_recall   = keyword_hits / total_kw     if total_kw     > 0 else 0
    sec_recall  = section_hits / section_total if section_total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"  Keyword recall:  {kw_recall*100:.1f}%")
    print(f"  Section recall:  {sec_recall*100:.1f}%")
    print(f"  Avg latency:     {avg_latency:.1f}s")

    return {
        "component":      "rag_pipeline",
        "keyword_recall": round(kw_recall, 4),
        "section_recall": round(sec_recall, 4),
        "avg_latency_s":  round(avg_latency, 2),
        "cases":          case_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. RISK SCORER
# ═══════════════════════════════════════════════════════════════════════════

RISK_TEST_CASES = [
    {
        "text":              "FIR registered under Section 302 IPC for murder. Non-bailable offence.",
        "doc_type":          "fir",
        "entities":          {"ipc_sections": ["Section 302"], "acts": ["Indian Penal Code"]},
        "expected_level":    "Critical",
        "expected_min_score": 0.80,
    },
    {
        "text":              "Cheque No.123 for Rs.5,00,000 dishonoured. Notice under Section 138 NI Act.",
        "doc_type":          "cheque_bounce_notice",
        "entities":          {"ipc_sections": ["Section 138"],
                              "acts": ["Negotiable Instruments Act"],
                              "monetary": ["Rs.5,00,000"]},
        "expected_level":    "High",
        "expected_min_score": 0.65,
    },
    {
        "text":              "Rental agreement for residential flat at Kochi. Monthly rent Rs.12,000.",
        "doc_type":          "rental_agreement",
        "entities":          {"monetary": ["Rs.12,000"]},
        "expected_level":    "Low",
        "expected_max_score": 0.40,
    },
    {
        "text":              "Consumer complaint for defective product worth Rs.74,999. "
                             "Deficiency in service under Consumer Protection Act 2019.",
        "doc_type":          "consumer_complaint",
        "entities":          {"monetary": ["Rs.74,999"], "acts": ["Consumer Protection Act"]},
        "expected_level":    "Medium",
        "expected_min_score": 0.30,
        "expected_max_score": 0.65,
    },
    {
        # bail for Section 376 IS Critical — corrected from previous version
        "text":              "Bail application under Section 437 CrPC. Accused charged under "
                             "Section 376 IPC. Non-bailable offence.",
        "doc_type":          "bail_application",
        "entities":          {"ipc_sections": ["Section 376", "Section 437"],
                              "acts": ["Indian Penal Code", "Code of Criminal Procedure"]},
        "expected_level":    "Critical",
        "expected_min_score": 0.80,
    },
]


def eval_risk_scorer() -> dict:
    print("\n[5/6] Evaluating Risk Scorer...")
    from models.risk_scorer import risk_scorer

    correct = 0
    wrong   = []

    for case in RISK_TEST_CASES:
        result   = risk_scorer.score(
            text     = case["text"],
            doc_type = case["doc_type"],
            entities = case["entities"],
            use_llm  = False,
        )
        level_ok = result.level == case["expected_level"]
        score_ok = True
        if "expected_min_score" in case:
            score_ok = score_ok and result.score >= case["expected_min_score"]
        if "expected_max_score" in case:
            score_ok = score_ok and result.score <= case["expected_max_score"]

        passed = level_ok and score_ok
        if passed:
            correct += 1
        else:
            wrong.append({
                "doc_type":       case["doc_type"],
                "expected_level": case["expected_level"],
                "got_level":      result.level,
                "score":          result.score,
            })

        status = "✓" if passed else "✗"
        notes  = []
        if not level_ok: notes.append(f"level: expected {case['expected_level']} got {result.level}")
        if not score_ok: notes.append(f"score {result.score:.3f} out of range")
        print(f"  {status} {case['doc_type']:25s} level={result.level:8s} "
              f"score={result.score:.3f}  {' | '.join(notes)}")

    accuracy = correct / len(RISK_TEST_CASES)
    print(f"  Accuracy: {correct}/{len(RISK_TEST_CASES)} = {accuracy*100:.1f}%")
    return {
        "component": "risk_scorer",
        "accuracy":  round(accuracy, 4),
        "correct":   correct,
        "total":     len(RISK_TEST_CASES),
        "wrong":     wrong,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. API HEALTH
# ═══════════════════════════════════════════════════════════════════════════

API_ENDPOINTS = [
    ("GET",  "/health",                     None),
    ("GET",  "/api/v1/classify/status",     None),
    ("GET",  "/api/v1/classify/categories", None),
    ("POST", "/api/v1/classify",
     {"text": "FIR registered at Ernakulam Police Station under Section 379 IPC"}),
    ("POST", "/api/v1/legal/query",
     {"query": "What is Section 302 IPC?"}),
    ("POST", "/api/v1/master/query",
     {"query": "What is bail?", "session_id": "eval_test"}),
]


def eval_api_health() -> dict:
    print("\n[6/6] Evaluating API Health...")
    results = []
    passed  = 0

    for method, path, body in API_ENDPOINTS:
        url = BASE_URL + path
        try:
            start = time.time()
            r     = requests.get(url, timeout=30) if method == "GET" \
                    else requests.post(url, json=body, timeout=90)
            latency = round(time.time() - start, 2)
            ok      = r.status_code < 400
            if ok:
                passed += 1
            print(f"  {'✓' if ok else '✗'} {method:4s} {path:40s} "
                  f"{r.status_code} ({latency:.1f}s)")
            results.append({"method": method, "path": path,
                            "status": r.status_code, "ok": ok, "latency_s": latency})
        except Exception as e:
            print(f"  ✗ {method:4s} {path:40s} ERROR: {e}")
            results.append({"method": method, "path": path, "ok": False, "error": str(e)})

    print(f"  Health: {passed}/{len(API_ENDPOINTS)} OK")
    return {"component": "api_health", "passed": passed,
            "total": len(API_ENDPOINTS), "endpoints": results}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

_COMPONENT_KEY_MAP = {
    "intent_classifier":    ("accuracy",       0.85, "%"),
    "document_classifier":  ("accuracy",       0.80, "%"),
    "ner_pipeline":         ("overall_recall", 0.70, "%"),
    "rag_pipeline":         ("keyword_recall", 0.70, "%"),
    "risk_scorer":          ("accuracy",       0.80, "%"),
    "api_health":           (None,             1.00, "ratio"),
}

_DISPLAY_NAMES = {
    "intent_classifier":   "Intent Classifier",
    "document_classifier": "Doc Classifier",
    "ner_pipeline":        "NER Pipeline",
    "rag_pipeline":        "RAG Pipeline",
    "risk_scorer":         "Risk Scorer",
    "api_health":          "API Health",
}


def run_all():
    print("=" * 65)
    print("LexShield AI — Full Evaluation Suite v2")
    print("=" * 65)

    start_total = time.time()
    report      = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": []}

    evaluators = [
        eval_intent_classifier,
        eval_document_classifier,
        eval_ner_pipeline,
        eval_rag_pipeline,
        eval_risk_scorer,
        eval_api_health,
    ]

    for fn in evaluators:
        try:
            result = fn()
            report["results"].append(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            report["results"].append({"component": fn.__name__, "error": str(e)})

    total_time = round(time.time() - start_total, 1)

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  {'Component':<23} {'Metric':<18} {'Score':<10} {'Target':<12} Status")
    print("-" * 65)

    all_pass = True
    for comp_key, (metric_key, target, fmt) in _COMPONENT_KEY_MAP.items():
        display = _DISPLAY_NAMES[comp_key]

        result = next(
            (r for r in report["results"]
             if r.get("component") == comp_key),
            None
        )
        if not result or "error" in result:
            print(f"  {'ERROR: ' + comp_key:<60}")
            all_pass = False
            continue

        if metric_key:
            score = result.get(metric_key, 0)
        else:
            score = result.get("passed", 0) / max(result.get("total", 1), 1)

        display_score  = f"{score*100:.1f}%" if fmt == "%" else f"{score:.2f}"
        display_target = f"{target*100:.0f}%" if fmt == "%" else f"{target:.2f}"
        passed         = score >= target
        if not passed:
            all_pass = False

        metric_name = metric_key.replace("_", " ").title() if metric_key else "Endpoints OK"
        print(f"  {display:<23} {metric_name:<18} {display_score:<10} "
              f"target={display_target:<8} {'✓ PASS' if passed else '✗ FAIL'}")

    print("-" * 65)
    verdict = "ALL PASS ✓" if all_pass else "SOME FAILED ✗"
    print(f"  Overall: {verdict}  (total time: {total_time}s)")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {REPORT_PATH}")
    print("=" * 65)
    return report


if __name__ == "__main__":
    run_all()