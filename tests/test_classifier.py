# tests/test_classifier_real.py
# Run: python tests/test_classifier_real.py

from models.classifier import classifier

# Real-world samples — short, varied, realistic
# Deliberately written differently from synthetic training data

REAL_TESTS = [
    # (text, expected_label)

    # Deliberately terse — not generator-style verbose
    ("Flat No 4B, second floor, leased to tenant @ Rs 8500 per month. "
     "Deposit 25000. Period 11 months from March 2024. "
     "No subletting. One month notice required.",
     "rental_agreement"),

    # All-caps FIR style (common in real scanned docs)
    ("FIR NO 234/2024 PS KOZHIKODE NORTH. COMPLAINANT ARUN KUMAR. "
     "ACCUSED UNKNOWN. OFFENCE U/S 379 IPC. THEFT OF MOBILE PHONE "
     "VALUED RS 18000 FROM POCKET AT RAILWAY STATION.",
     "fir"),

    # Lawyer-drafted notice, terse
    ("Your cheque no 456123 dt 01.03.2024 for Rs 2,50,000 drawn on "
     "SBI Thrissur returned unpaid. Pay within 15 days. "
     "Else Section 138 NI Act complaint filed.",
     "cheque_bounce_notice"),

    # Real HC order style — very terse
    ("WP(C) 4521/2024. Heard. Counter filed. "
     "Petitioner challenges GO(P) 45/2024 Revenue. "
     "Respondents directed to file reply in 3 weeks. Post 6 weeks.",
     "hc_judgment"),

    # Casual employment letter style
    ("Dear Rahul, pleased to offer you position of Junior Developer "
     "at Rs 35000 per month. Join by 1st April. "
     "3 month probation. 1 month notice period.",
     "employment_contract"),

    # Consumer complaint — very plain
    ("I bought Samsung TV from Croma Kochi on 15 Jan 2024 for Rs 45000. "
     "Screen stopped working after 2 months. Service centre says not covered. "
     "Seeking replacement or refund.",
     "consumer_complaint"),

    # Bail app — short, point form
    ("Application for bail. Accused Suresh in custody since 10.01.2024. "
     "Crime No 45/2024 PS Ernakulam. Sections 323 324 IPC. "
     "First offender. Permanent resident. Willing to cooperate.",
     "bail_application"),

    # Affidavit — minimal
    ("I Meena Kumari aged 45 years do solemnly affirm that "
     "my original birth certificate is lost and cannot be traced. "
     "Sworn before notary Ernakulam 15.02.2024.",
     "affidavit"),
]

print("Real-World Classifier Test")
print("=" * 55)
correct = 0
for text, expected in REAL_TESTS:
    r = classifier.predict(text)
    got   = r["label_name"]
    conf  = r["confidence"]
    ok    = "✓" if got == expected else "✗"
    if got == expected:
        correct += 1
    print(f"{ok} Expected: {expected:<25} Got: {got:<25} Conf: {conf:.2f}")
    if got != expected:
        # Show top 3 scores for wrong predictions
        top3 = sorted(r["all_scores"].items(), key=lambda x: -x[1])[:3]
        print(f"  Top 3: {top3}")

print("=" * 55)
print(f"Real-world accuracy: {correct}/{len(REAL_TESTS)} = {correct/len(REAL_TESTS)*100:.0f}%")
print()
print("Interpretation:")
print("  ≥87% → Model generalized well, synthetic data was high quality")
print("  70–87% → Mild overfit, acceptable for portfolio")
print("  <70%  → Significant overfit, need more diverse training data")