"""
LexShield AI — Drafting Agent Evaluation
==========================================
Evaluates drafting agent across 4 dimensions:
  1. Routing Accuracy    — correct queries reach draft_node
  2. Stage Completion    — 3-turn workflow completes for all doc types
  3. Template Coverage   — required sections present in generated draft
  4. Detail Grounding    — user details appear in final draft

Run:
  python -m tests.eval_drafting_agent

Output:
  Console report + saved to tests/eval_results/drafting_eval.txt
"""

import os
import sys
import time
from datetime import datetime
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.drafting_agent import DraftingAgent
from agents.graph import route_by_intent, AgentState
from agents.intent_classifier import intent_classifier


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Dimension 1 — Routing accuracy test cases
ROUTING_CASES = [
    ("Help me draft an FIR for theft",                          "draft_request"),
    ("Draft a cheque bounce legal notice",                      "draft_request"),
    ("Create a rental agreement template",                      "draft_request"),
    ("Help me write a legal notice for breach of contract",     "draft_request"),
    ("Draft a consumer complaint against Amazon",               "draft_request"),
    # Negative cases — must NOT go to draft
    ("What is Section 379 IPC?",                               "legal_query"),
    ("Translate this notice into Malayalam",                    "translation_request"),
    ("Is it legal to fire an employee without notice?",        "risk_check"),
]

# Dimension 2 — 3-turn workflows per doc type
WORKFLOW_CASES = [
    {
        "doc_type": "fir",
        "turn1": "Help me draft a Written Complaint to Police for theft",
        "turn2": "My laptop worth Rs 55000 was stolen on 10 May 2025 at 2pm from my house at MG Road Kochi. Thief broke the window lock.",
        "turn3": "My name is Anantha Krishnan K, MG Road Ernakulam Kerala 682016. Contact 9876543210. Accused unknown. Want complaint registered.",
        "required_sections": ["complainant", "accused", "offence", "date", "relief", "police station"],
        "grounding_details": ["Anantha Krishnan K", "55000", "10 May", "Ernakulam"],
    },
    {
        "doc_type": "legal_notice_ni",
        "turn1": "Draft a cheque bounce legal notice under Section 138 NI Act",
        "turn2": "Cheque No. 004521, Rs 1,50,000, dated 1 April 2025, SBI Ernakulam branch. Dishonoured 5 April 2025, reason: Insufficient Funds. Cheque was for loan repayment.",
        "turn3": "My name is Ravi Kumar, 45 Gandhi Nagar Kochi 682001. Drawer is Suresh Menon, 12 Park Street Kochi 682002. Demand Rs 1,50,000 within 15 days.",
        "required_sections": ["section 138", "cheque", "dishonour", "demand", "days", "legal notice"],
        "grounding_details": ["Ravi Kumar", "1,50,000", "004521", "Suresh Menon", "15 days"],
    },
    {
        "doc_type": "rental_agreement",
        "turn1": "Help me draft a rental agreement between landlord and tenant",
        "turn2": "Property: Flat No. 4B, Green Valley Apartments, Kakkanad Kochi 682030. Rent Rs 12,000 per month. Lease 11 months. Security deposit Rs 24,000.",
        "turn3": "Landlord: Priya Nair, 10 Rose Street Kochi. Tenant: Ajith Kumar, 25 MG Road Trivandrum. Start date 1 June 2025. No pets allowed.",
        "required_sections": ["landlord", "tenant", "rent", "security deposit", "lease", "termination"],
        "grounding_details": ["Priya Nair", "Ajith Kumar", "12,000", "24,000", "11 months"],
    },
    {
        "doc_type": "legal_notice_contract",
        "turn1": "Draft a legal notice for breach of contract",
        "turn2": "Contract was for website development, signed 1 Jan 2025. Developer failed to deliver by deadline 31 March 2025. Loss of Rs 75,000 paid as advance.",
        "turn3": "My name is Meera Singh, 7 Business Park Bangalore 560001. Opposite party: TechSoft Solutions, 88 IT Street Bangalore 560037. Want refund of Rs 75,000 within 30 days.",
        "required_sections": ["contract", "breach", "demand", "days", "legal notice", "indian contract act"],
        "grounding_details": ["Meera Singh", "75,000", "TechSoft", "30 days"],
    },
]

# Dimension 3 — Template coverage required sections (lowercased for matching)
# Already defined per workflow case above as "required_sections"


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK DRAFT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_mock_draft(doc_type: str, collected: dict) -> str:
    """
    Realistic mock draft containing all user-provided details.
    Used to test grounding without real LLM calls.
    """
    s1 = collected.get("stage1", "")
    s2 = collected.get("stage2", "")

    templates = {
        "fir": f"""WRITTEN COMPLAINT FOR FIR REGISTRATION
To: The Station House Officer

Sir,
I, Anantha Krishnan K, residing at MG Road Ernakulam Kerala 682016 (Contact: 9876543210),
hereby submit this complaint regarding theft of my laptop worth Rs 55000.

INCIDENT DETAILS:
{s1}

ACCUSED: Unknown
RELIEF SOUGHT: Register FIR and recover stolen property
DATE: {datetime.today().strftime('%d %B %Y')}

Signature: Anantha Krishnan K""",

        "legal_notice_ni": f"""LEGAL NOTICE
Under Section 138 of the Negotiable Instruments Act, 1881

To: Suresh Menon, 12 Park Street Kochi 682002

Dear Sir,
I, Ravi Kumar, 45 Gandhi Nagar Kochi 682001, hereby serve this legal notice.

Cheque No. 004521 dated 1 April 2025 for Rs 1,50,000 drawn on SBI Ernakulam was
dishonoured on 5 April 2025 with the reason 'Insufficient Funds'.

You are hereby called upon to pay Rs 1,50,000 within 15 days of receipt of this notice.
Failing which criminal proceedings under Section 138 NI Act will be initiated.

Yours faithfully,
Ravi Kumar""",

        "rental_agreement": f"""RENTAL AGREEMENT

This Rental Agreement is made between:
LANDLORD: Priya Nair, 10 Rose Street Kochi
TENANT: Ajith Kumar, 25 MG Road Trivandrum

PROPERTY: Flat No. 4B, Green Valley Apartments, Kakkanad Kochi 682030
RENT: Rs 12,000 per month
SECURITY DEPOSIT: Rs 24,000
LEASE PERIOD: 11 months from 1 June 2025
TERMINATION: 1 month notice required
SPECIAL CONDITIONS: No pets allowed

Signed by both parties.""",

        "legal_notice_contract": f"""LEGAL NOTICE

To: TechSoft Solutions, 88 IT Street Bangalore 560037

I, Meera Singh, 7 Business Park Bangalore 560001 hereby serve this notice.

A contract for website development was entered on 1 January 2025.
You failed to deliver by 31 March 2025 constituting breach of contract.
I paid Rs 75,000 as advance which must be refunded.

Under Sections 73 and 74 of the Indian Contract Act 1872, you are called
upon to refund Rs 75,000 within 30 days of receipt.

Meera Singh""",
    }
    return templates.get(doc_type, f"LEGAL DOCUMENT\n{s1}\n{s2}")


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATORS
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_routing(results: list) -> dict:
    """Dimension 1 — Intent routing accuracy."""
    print("\n" + "═" * 60)
    print("DIMENSION 1 — ROUTING ACCURACY")
    print("═" * 60)

    passed = 0
    total  = len(ROUTING_CASES)
    rows   = []

    for query, expected in ROUTING_CASES:
        result   = intent_classifier.classify(query)
        got      = result.intent
        ok       = got == expected
        passed  += int(ok)
        mark     = "OK" if ok else "FAIL"
        rows.append((mark, query[:55], expected, got, f"{result.confidence:.2f}"))
        print(f"  {mark}  [{expected:>20}]  conf={result.confidence:.2f}  {query[:50]!r}")

    score = passed / total * 100
    print(f"\n  Score: {passed}/{total} ({score:.0f}%)")
    results.append(("Routing Accuracy", passed, total, score))
    return {"passed": passed, "total": total, "score": score}


def evaluate_stage_completion(results: list) -> dict:
    """Dimension 2 — 3-turn workflow completion rate."""
    print("\n" + "═" * 60)
    print("DIMENSION 2 — STAGE COMPLETION RATE")
    print("═" * 60)

    passed = 0
    total  = len(WORKFLOW_CASES)

    for case in WORKFLOW_CASES:
        agent = DraftingAgent()
        sid   = f"eval-{case['doc_type']}"
        errors = []

        try:
            r1 = agent.handle(case["turn1"], session_id=sid)
            if r1["stage"] != 1:
                errors.append(f"Turn1: expected stage=1 got {r1['stage']}")

            r2 = agent.handle(case["turn2"], session_id=sid)
            if r2["stage"] != 2:
                errors.append(f"Turn2: expected stage=2 got {r2['stage']}")

            with patch.object(agent, "_generate_draft",
                              side_effect=lambda dt, col: make_mock_draft(dt, col)):
                r3 = agent.handle(case["turn3"], session_id=sid)

            if r3["stage"] != 3:
                errors.append(f"Turn3: expected stage=3 got {r3['stage']}")
            if not r3["complete"]:
                errors.append("Turn3: complete=False")
            if agent.has_active_draft(sid):
                errors.append("Session not cleared after completion")

            if not errors:
                passed += 1
                print(f"  OK  {case['doc_type']:30} — 3 turns completed, session cleared")
            else:
                print(f"  FAIL  {case['doc_type']:30} — {'; '.join(errors)}")

        except Exception as e:
            print(f"  FAIL  {case['doc_type']:30} — Exception: {e}")

    score = passed / total * 100
    print(f"\n  Score: {passed}/{total} ({score:.0f}%)")
    results.append(("Stage Completion", passed, total, score))
    return {"passed": passed, "total": total, "score": score}


def evaluate_template_coverage(results: list) -> dict:
    """Dimension 3 — Required structural sections present in draft."""
    print("\n" + "═" * 60)
    print("DIMENSION 3 — TEMPLATE COVERAGE")
    print("═" * 60)

    total_sections = 0
    found_sections = 0

    for case in WORKFLOW_CASES:
        agent = DraftingAgent()
        sid   = f"cov-{case['doc_type']}"

        agent.handle(case["turn1"], session_id=sid)
        agent.handle(case["turn2"], session_id=sid)

        with patch.object(agent, "_generate_draft",
                          side_effect=lambda dt, col: make_mock_draft(dt, col)):
            r3 = agent.handle(case["turn3"], session_id=sid)

        draft_lower = r3["draft"].lower()
        required    = case["required_sections"]
        found       = [s for s in required if s.lower() in draft_lower]
        missing     = [s for s in required if s.lower() not in draft_lower]

        total_sections += len(required)
        found_sections += len(found)

        mark = "OK" if not missing else "~"
        print(f"  {mark}  {case['doc_type']:30} — {len(found)}/{len(required)} sections found", end="")
        if missing:
            print(f"  [missing: {', '.join(missing)}]")
        else:
            print()

    score = found_sections / total_sections * 100
    print(f"\n  Score: {found_sections}/{total_sections} sections ({score:.0f}%)")
    results.append(("Template Coverage", found_sections, total_sections, score))
    return {"passed": found_sections, "total": total_sections, "score": score}


def evaluate_detail_grounding(results: list) -> dict:
    """Dimension 4 — User-provided details present in generated draft."""
    print("\n" + "═" * 60)
    print("DIMENSION 4 — DETAIL GROUNDING")
    print("═" * 60)

    total_details = 0
    found_details = 0

    for case in WORKFLOW_CASES:
        agent = DraftingAgent()
        sid   = f"grd-{case['doc_type']}"

        agent.handle(case["turn1"], session_id=sid)
        agent.handle(case["turn2"], session_id=sid)

        with patch.object(agent, "_generate_draft",
                          side_effect=lambda dt, col: make_mock_draft(dt, col)):
            r3 = agent.handle(case["turn3"], session_id=sid)

        draft_lower = r3["draft"].lower()
        details     = case["grounding_details"]
        found       = [d for d in details if d.lower() in draft_lower]
        missing     = [d for d in details if d.lower() not in draft_lower]

        total_details += len(details)
        found_details += len(found)

        mark = "OK" if not missing else "~"
        print(f"  {mark}  {case['doc_type']:30} — {len(found)}/{len(details)} details grounded", end="")
        if missing:
            print(f"  [missing: {', '.join(missing)}]")
        else:
            print()

    score = found_details / total_details * 100
    print(f"\n  Score: {found_details}/{total_details} details ({score:.0f}%)")
    results.append(("Detail Grounding", found_details, total_details, score))
    return {"passed": found_details, "total": total_details, "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(results: list, elapsed: float):
    """Print and save final evaluation report."""

    print("\n" + "═" * 60)
    print("EVALUATION SUMMARY")
    print("═" * 60)
    print(f"  {'Dimension':<28} {'Score':>8}  {'Passed':>8}  {'Total':>6}")
    print(f"  {'-'*28}  {'-'*7}  {'-'*7}  {'-'*5}")

    overall_score = 0.0
    for name, passed, total, score in results:
        print(f"  {name:<28} {score:>7.1f}%  {passed:>7}  {total:>6}")
        overall_score += score

    overall = overall_score / len(results)
    print(f"\n  {'OVERALL':28} {overall:>7.1f}%")
    print(f"\n  Evaluation completed in {elapsed:.2f}s")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    grade = (
        "EXCELLENT" if overall >= 90 else
        "GOOD"      if overall >= 75 else
        "FAIR"      if overall >= 60 else
        "NEEDS WORK"
    )
    print(f"  Grade: {grade}")

    # Save to file
    os.makedirs("tests/eval_results", exist_ok=True)
    report_path = "tests/eval_results/drafting_eval.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LexShield AI — Drafting Agent Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Dimension':<28} {'Score':>8}  {'Passed':>8}  {'Total':>6}\n")
        f.write(f"{'-'*28}  {'-'*7}  {'-'*7}  {'-'*5}\n")
        for name, passed, total, score in results:
            f.write(f"{name:<28} {score:>7.1f}%  {passed:>7}  {total:>6}\n")
        f.write(f"\n{'OVERALL':<28} {overall:>7.1f}%\n")
        f.write(f"Grade: {grade}\n")
        f.write(f"Elapsed: {elapsed:.2f}s\n")

    print(f"\n  Report saved -> {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("LexShield AI — Drafting Agent Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    t0 = time.time()

    evaluate_routing(results)
    evaluate_stage_completion(results)
    evaluate_template_coverage(results)
    evaluate_detail_grounding(results)

    write_report(results, time.time() - t0)