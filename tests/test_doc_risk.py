"""
LexShield AI — Day 5 Test Suite
================================
Tests:
  1. Classifier accuracy on held-out samples (target >85%)
  2. Risk scorer flags correct clauses for each document type
  3. Full pipeline integration (classify + NER + risk in sequence)

Usage:
    python tests/test_day5.py
    python tests/test_day5.py --skip-train   (if model already trained)
    python tests/test_day5.py --verbose
"""

import sys
import argparse

# ── Sample documents for integration test ────────────────────────────────────

RENTAL_WITH_RISKS = """
RENTAL AGREEMENT

This agreement is between SURESH NAIR (Landlord) and MEERA KRISHNAN (Tenant)
for premises at Green Valley Layout, Thiruvananthapuram, Kerala.

1. RENT: Monthly rent of Rs. 12,000 shall be paid by 5th of every month.

2. DEPOSIT: A security deposit of ₹72,000 is paid. The deposit shall be 
   NON-REFUNDABLE and shall be forfeited in case of early termination.

3. RENT INCREASE: Rent shall increase by 20% every year automatically.

4. ENTRY: The Landlord may enter the premises at any time without prior 
   notice to inspect the property condition.

5. SUBLETTING: The Tenant shall not sublet or assign the premises.

6. DISPUTES: All disputes shall be subject to exclusive jurisdiction of 
   courts in Mumbai only.

7. AUTO RENEWAL: This agreement shall be automatically renewed for another 
   11 months unless terminated 30 days prior.

8. TERMINATION: The landlord shall have the right to terminate this agreement
   summarily without cause or notice.

Signed at Thiruvananthapuram on 1st January 2024.
SURESH NAIR (Landlord)    MEERA KRISHNAN (Tenant)
"""

EMPLOYMENT_WITH_RISKS = """
EMPLOYMENT AGREEMENT

This Employment Agreement is made on 15th February 2024 between:

M/S TechSolutions Pvt Ltd, incorporated under the Companies Act, 2013,
having its registered office at Cyberpark, Bengaluru, Karnataka,
(hereinafter referred to as "Employer")

AND

ANAND PILLAI, residing at 142, Bengaluru, Karnataka,
(hereinafter referred to as "Employee").

TERMS OF EMPLOYMENT:

1. Designation: Software Engineer\nEmployer: TechCo Pvt Ltd\nCTC: ₹50,000 per annum\nProbation: 3 months\nPF and ESI applicable.
2. COMMENCEMENT: The employment shall commence from 1st March 2024.
3. COMPENSATION: Monthly CTC: Rs. 65,000
4. NON-COMPETE: The Employee shall not work with any competitor company 
   for a period of 3 years after leaving employment in entire India.
5. WAGE DEDUCTION: The Employer may deduct salary without notice at 
   employer's discretion for performance issues.
6. COURT ACCESS: The Employee waives any right to approach courts for 
   employment disputes.

For TechSolutions Pvt Ltd

Rajesh Kumar                              ANAND PILLAI
(Authorized Signatory)               (Employee)
"""

CLEAN_COURT_NOTICE = """
IN THE HIGH COURT OF KERALA AT ERNAKULAM

W.P.(C) No. 5621/2023

MOHAMMED IBRAHIM ... Petitioner
VERSUS
State of Kerala ... Respondent

NOTICE

The above petition has been filed under Article 226 challenging the 
order dated 15th March 2023 passed by the District Collector, Kozhikode.

You are directed to appear before this Court on 14th September 2023.

By Order of the Court,
REGISTRAR GENERAL, High Court of Kerala
"""

# ── Test runner ───────────────────────────────────────────────────────────────

def test_classifier(verbose: bool = False) -> bool:
    print("\n" + "=" * 60)
    print("TEST 1: Document Classifier")
    print("=" * 60)

    from models.classifier import classifier

    if not classifier.is_ready():
        print("  ✗ Classifier not loaded. Run training first.")
        return False

    # Quick sanity check on 5 known document types
    test_cases = [
        (RENTAL_WITH_RISKS,     "rental_agreement"),
        (EMPLOYMENT_WITH_RISKS, "employment_contract"),
        (CLEAN_COURT_NOTICE,    "court_notice"),
    ]

    passed = 0
    for text, expected in test_cases:
        result    = classifier.predict(text)
        predicted = result["label_name"]
        confidence= result["confidence"]
        ok        = predicted == expected
        passed   += int(ok)
        mark      = "✓" if ok else "✗"
        print(f"  {mark} Expected: {expected:25s} | Got: {predicted:25s} | conf={confidence:.2f}")
        if verbose and not ok:
            print(f"    All scores: {result['all_scores']}")

    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_risk_scorer(verbose: bool = False) -> bool:
    print("\n" + "=" * 60)
    print("TEST 2: Risk Scorer")
    print("=" * 60)

    from models.risk_scorer import risk_scorer

    # Test 1: Rental agreement with risks
    print("\n  [2a] Rental agreement with known risks:")
    result = risk_scorer.score(RENTAL_WITH_RISKS, doc_type="rental_agreement")

    expected_flags = {
        "NON_REFUNDABLE_DEPOSIT",
        "ENTRY_WITHOUT_NOTICE",
        "EXCESSIVE_RENT_INCREASE",
        "AUTO_RENEWAL",
    }
    found_flags = {flag for cr in result.clause_risks for flag in cr.flags}

    print(f"  Overall score  : {result.overall_score} ({result.risk_level})")
    print(f"  High risk count: {result.high_risk_count}")
    print(f"  Summary        : {result.summary[:80]}...")
    print(f"  Flags found    : {found_flags}")

    missing = expected_flags - found_flags
    if missing:
        print(f"  ✗ Missing expected flags: {missing}")
    else:
        print(f"  ✓ All expected flags detected")

    if verbose:
        for cr in result.clause_risks:
            if cr.score > 0:
                print(f"\n    Clause {cr.clause_number} (score={cr.score} {cr.risk_level}):")
                print(f"    Text: {cr.clause_text[:80]}...")
                print(f"    Flags: {cr.flags}")

    passed_a = len(missing) == 0 and result.overall_score >= 50

    # Test 2: Employment contract with risks
    print(f"\n  [2b] Employment contract with risks:")
    result2 = risk_scorer.score(EMPLOYMENT_WITH_RISKS, doc_type="employment_contract")
    flags2  = {flag for cr in result2.clause_risks for flag in cr.flags}

    expected_flags2 = {
        "NON_COMPETE_CLAUSE",
        "WAIVER_OF_COURT_ACCESS",
        "UNLAWFUL_WAGE_DEDUCTION",
    }
    missing2 = expected_flags2 - flags2
    print(f"  Overall score  : {result2.overall_score} ({result2.risk_level})")
    print(f"  Flags found    : {flags2}")
    if missing2:
        print(f"  ✗ Missing flags: {missing2}")
    else:
        print(f"  ✓ All expected flags detected")

    passed_b = len(missing2) == 0

    # Test 3: Court notice should have LOW risk
    print(f"\n  [2c] Court notice (should be low risk):")
    result3 = risk_scorer.score(CLEAN_COURT_NOTICE, doc_type="court_notice")
    print(f"  Overall score  : {result3.overall_score} ({result3.risk_level})")
    passed_c = result3.overall_score < 40
    print(f"  {'✓' if passed_c else '✗'} Score < 40 (expected for court notice)")

    return passed_a and passed_b and passed_c


def test_full_pipeline(verbose: bool = False) -> bool:
    print("\n" + "=" * 60)
    print("TEST 3: Full Pipeline Integration (classify + NER + risk)")
    print("=" * 60)

    from models.classifier  import classifier
    from nlp.ner_pipeline   import extract_entities
    from models.risk_scorer import risk_scorer

    text    = RENTAL_WITH_RISKS
    checks  = {}

    # Classify
    clf     = classifier.predict(text)
    checks["classification_correct"] = clf["label_name"] == "rental_agreement"
    print(f"\n  Classification: {clf['label_name']} (conf={clf['confidence']:.2f})")
    print(f"  {'✓' if checks['classification_correct'] else '✗'} Expected: rental_agreement")

    # NER
    ents    = extract_entities(text)
    ed      = ents.to_dict()
    checks["persons_found"]   = len(ed["persons"]) > 0
    checks["locations_found"] = len(ed["locations"]) > 0
    checks["monetary_found"]  = len(ed["monetary"]) > 0

    print(f"\n  NER results:")
    print(f"  {'✓' if checks['persons_found']   else '✗'} Persons:   {ed['persons'][:3]}")
    print(f"  {'✓' if checks['locations_found'] else '✗'} Locations: {ed['locations'][:3]}")
    print(f"  {'✓' if checks['monetary_found']  else '✗'} Monetary:  {ed['monetary'][:3]}")

    # Risk
    risk    = risk_scorer.score(text, doc_type=clf["label_name"])
    checks["risk_nonzero"]    = risk.overall_score > 0
    checks["clauses_flagged"] = risk.high_risk_count > 0

    print(f"\n  Risk scoring:")
    print(f"  {'✓' if checks['risk_nonzero']    else '✗'} Overall score: {risk.overall_score}")
    print(f"  {'✓' if checks['clauses_flagged'] else '✗'} High-risk clauses: {risk.high_risk_count}")

    passed = all(checks.values())
    print(f"\n  Pipeline: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training step (use existing saved model)")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if not args.skip_train:
        print("\n[Pre-test] Retraining classifier with current training_data.py...")
        from models.classifier import train, classifier
        metrics = train(samples_per_class=80)

        print(f"\n  CV Val  Accuracy : {metrics['cv_val_accuracy']  * 100:.1f}%")
        print(f"  CV Train Accuracy: {metrics['cv_train_accuracy'] * 100:.1f}%")
        print(f"  Overfitting gap  : {metrics['overfit_gap']       * 100:.1f}%")
        print(f"  CV Val F1-Macro  : {metrics['cv_val_f1_macro']   * 100:.1f}%")

        # ── Hot-swap the singleton to the freshly saved model ─────────────────
        print("\n  Reloading classifier singleton from new .pkl...")
        ok = classifier.reload()
        print(f"  {'✓ Reload succeeded' if ok else '✗ Reload FAILED — check disk write'}")
        if not ok:
            print("  Aborting tests.")
            sys.exit(1)

        if not metrics["target_met"]:
            print(f"\n  ⚠ WARNING: CV val accuracy {metrics['cv_val_accuracy']*100:.1f}% "
                  f"< 85% target.")
        else:
            print(f"\n  ✓ Accuracy target met.")
    else:
        print("\n[Pre-test] Skipping training — using existing saved model.")

    r1 = test_classifier(args.verbose)
    r2 = test_risk_scorer(args.verbose)
    r3 = test_full_pipeline(args.verbose)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test 1 — Classifier   : {'✓ PASS' if r1 else '✗ FAIL'}")
    print(f"  Test 2 — Risk Scorer  : {'✓ PASS' if r2 else '✗ FAIL'}")
    print(f"  Test 3 — Full Pipeline: {'✓ PASS' if r3 else '✗ FAIL'}")

    all_pass = r1 and r2 and r3
    print(f"\n{'✓ DAY 5 CHECKPOINT COMPLETE' if all_pass else '✗ SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)