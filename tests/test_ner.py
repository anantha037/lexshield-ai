"""
LexShield AI — Day 4 NER Test Suite
=====================================
Tests NER extraction on 3 document types:
  1. Rental Agreement
  2. Court Notice
  3. Employment Contract

For each document, verifies:
  - Persons are detected
  - Organizations are detected
  - Dates are detected
  - Monetary amounts are detected (where applicable)
  - IPC sections are detected (where applicable)
  - Case numbers are detected (where applicable)
  - Acts are detected

Usage:
    python tests/test_ner.py
    python tests/test_ner.py --verbose     (show all extracted entities)
    python tests/test_ner.py --doc 2       (run only document type 2)
"""

import sys
import argparse
from dataclasses import dataclass

# ── Sample documents ──────────────────────────────────────────────────────────
# These are representative Indian legal document samples.
# In production, these come from the CV pipeline (OCR of uploaded files).

RENTAL_AGREEMENT = """
RENTAL AGREEMENT

This Rental Agreement is entered into on this 1st day of January 2024,
between RAJESH KUMAR MENON, son of Gopinath Menon, residing at
Flat No. 4B, Sunrise Apartments, Pattom, Thiruvananthapuram, Kerala - 695004,
hereinafter referred to as the "Landlord",

AND

PRIYA SHARMA, daughter of Ramesh Sharma, residing at
12, MG Road, Ernakulam, Kerala - 682011,
hereinafter referred to as the "Tenant".

WHEREAS the Landlord is the owner of the premises situated at
Plot No. 14, Green Valley Layout, Kowdiar, Thiruvananthapuram, Kerala.

NOW THEREFORE, in consideration of the mutual covenants, the parties agree:

1. RENT: The Tenant agrees to pay a monthly rent of Rs. 15,000 (Rupees Fifteen
   Thousand only) on or before the 5th of every month.

2. DEPOSIT: The Tenant has paid a security deposit of ₹90,000 (Rupees Ninety
   Thousand only) which shall be refunded at the time of vacating the premises.

3. TENURE: The tenancy shall commence from 1st January 2024 and shall continue
   for a period of 11 months ending on 30th November 2024.

4. This agreement is governed by the Kerala Buildings (Lease and Rent Control)
   Act and the Indian Contract Act, 1872.

Signed on 1st January 2024 at Thiruvananthapuram.

RAJESH KUMAR MENON                    PRIYA SHARMA
(Landlord)                            (Tenant)

Witness 1: SURESH NAIR
Witness 2: ANITHA KRISHNAN
"""

COURT_NOTICE = """
IN THE HIGH COURT OF KERALA AT ERNAKULAM

W.P.(C) No. 4521/2023

IN THE MATTER OF:

MOHAMMED IBRAHIM, aged 45 years, son of Abdul Kareem,
residing at House No. 23, Beach Road, Kozhikode, Kerala - 673001.
                                                    ... Petitioner

VERSUS

STATE OF KERALA represented by its Chief Secretary,
Government Secretariat, Thiruvananthapuram, Kerala - 695001,

AND

THE DISTRICT COLLECTOR, Kozhikode District,
Mini Civil Station, Kozhikode, Kerala - 673020.
                                                    ... Respondents

NOTICE

WHEREAS the above Writ Petition has been filed by the Petitioner
challenging the order dated 15th March 2023 passed by the District Collector,
Kozhikode, under Section 144 of the Code of Criminal Procedure (CrPC).

AND WHEREAS the Petitioner alleges violation of his fundamental rights
guaranteed under Article 21 and Article 19(1)(g) of the Constitution of India.

YOU ARE HEREBY DIRECTED to appear before this Hon'ble Court on
14th September 2023 at 10:30 AM in Court Hall No. 5.

The Petitioner is represented by Advocate JOSEPH MATHEW,
enrolled with the Bar Council of Kerala.

Dated this 1st day of September 2023.

By Order of the Court,
REGISTRAR GENERAL
High Court of Kerala

Also see: Crl.A. 1892/2022 (related matter)
Fine imposed: Rs. 50,000 under Section 188 IPC.
"""

EMPLOYMENT_CONTRACT = """
EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is made on 15th February 2024,
between:

M/S TECHVISION SOLUTIONS PRIVATE LIMITED, a company incorporated under
the Companies Act, 2013, having its registered office at
4th Floor, Infopark Cherthala, Alappuzha, Kerala - 688539,
(hereinafter referred to as "Employer")

AND

ANANYA VIJAYAKUMAR, daughter of P. Vijayakumar,
residing at TC 12/445, Ambalamukku, Thiruvananthapuram, Kerala - 695005,
(hereinafter referred to as "Employee").

TERMS OF EMPLOYMENT:

1. DESIGNATION: Software Engineer (AI/ML Division)

2. COMMENCEMENT: The employment shall commence from 1st March 2024.

3. COMPENSATION:
   - Monthly CTC: Rs. 85,000 (Rupees Eighty Five Thousand only)
   - Annual CTC: ₹10,20,000 (Rupees Ten Lakhs Twenty Thousand only)
   - Performance Bonus: up to 15% of annual CTC

4. PROBATION: The Employee shall be on probation for 6 months from
   the date of joining, i.e., until 31st August 2024.

5. NOTICE PERIOD: Either party may terminate this agreement by giving
   60 days written notice. This is subject to the Code on Wages, 2019
   and the Industrial Disputes Act, 1947.

6. CONFIDENTIALITY: The Employee agrees not to disclose any proprietary
   information. Breach may invite action under Section 408 and Section 420
   of the Indian Penal Code, 1860.

7. GOVERNING LAW: This Agreement shall be governed by the laws of India
   including the Indian Contract Act, 1872 and the Specific Relief Act, 1963.

Signed at Thiruvananthapuram on 15th February 2024.

For M/S TECHVISION SOLUTIONS PRIVATE LIMITED

ROHIT AGARWAL                         ANANYA VIJAYAKUMAR
(HR Director)                         (Employee)
"""

# ── Test spec ─────────────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "id":          1,
        "name":        "Rental Agreement",
        "text":        RENTAL_AGREEMENT,
        "must_have": {
            "persons":       ["Rajesh", "Priya"],       # partial match OK
            "organizations": [],
            "dates":         ["2024"],
            "locations":     ["Thiruvananthapuram", "Kerala"],
            "monetary":      ["15,000", "90,000"],
            "ipc_sections":  [],
            "case_numbers":  [],
            "acts":          ["Kerala Buildings", "Indian Contract Act"],
        },
    },
    {
        "id":          2,
        "name":        "Court Notice",
        "text":        COURT_NOTICE,
        "must_have": {
            "persons":       ["Mohammed Ibrahim", "Joseph"],
            "organizations": ["High Court", "State of Kerala"],
            "dates":         ["2023"],
            "locations":     ["Kozhikode", "Kerala"],
            "monetary":      ["50,000"],
            "ipc_sections":  ["Section 144", "Section 188"],
            "case_numbers":  ["4521/2023"],
            "acts":          ["Code of Criminal Procedure"],
        },
    },
    {
        "id":          3,
        "name":        "Employment Contract",
        "text":        EMPLOYMENT_CONTRACT,
        "must_have": {
            "persons":       ["Ananya", "Rohit"],
            "organizations": ["Techvision", "TechVision"],
            "dates":         ["2024"],
            "locations":     ["Thiruvananthapuram", "Kerala"],
            "monetary":      ["85,000", "10,20,000"],
            "ipc_sections":  ["Section 408", "Section 420"],
            "case_numbers":  [],
            "acts":          ["Indian Penal Code", "Indian Contract Act", "Code on Wages"],
        },
    },
]


# ── Check helpers ─────────────────────────────────────────────────────────────

def _any_contains(extracted: list[str], keywords: list[str]) -> tuple[bool, list[str]]:
    """
    Returns (all_found, missing_keywords).
    A keyword is "found" if it appears (case-insensitive) in any extracted value.
    """
    missing = []
    for kw in keywords:
        found = any(kw.lower() in val.lower() for val in extracted)
        if not found:
            missing.append(kw)
    return len(missing) == 0, missing


@dataclass
class DocResult:
    doc_id:    int
    doc_name:  str
    passed:    bool
    checks:    dict[str, bool]
    missing:   dict[str, list[str]]
    extracted: dict[str, list[str]]


# ── Test runner ───────────────────────────────────────────────────────────────

def run_document(doc: dict, verbose: bool = False) -> DocResult:
    from nlp.ner_pipeline import extract_entities

    print(f"  [{doc['id']}] {doc['name']}", end=" ... ", flush=True)

    result   = extract_entities(doc["text"])
    ext_dict = result.to_dict()

    checks:  dict[str, bool]        = {}
    missing: dict[str, list[str]]   = {}

    for entity_type, expected_keywords in doc["must_have"].items():
        if not expected_keywords:
            checks[entity_type]  = True
            missing[entity_type] = []
            continue
        extracted = ext_dict.get(entity_type, [])
        ok, miss  = _any_contains(extracted, expected_keywords)
        checks[entity_type]  = ok
        missing[entity_type] = miss

    passed = all(checks.values())
    print("PASS ✓" if passed else "FAIL ✗")

    if verbose:
        print(f"\n  Extracted entities for '{doc['name']}':")
        for etype, vals in ext_dict.items():
            if etype == "entity_counts":
                continue
            if vals:
                print(f"    {etype:15s}: {vals}")
        print()

    return DocResult(
        doc_id    = doc["id"],
        doc_name  = doc["name"],
        passed    = passed,
        checks    = checks,
        missing   = missing,
        extracted = ext_dict,
    )


def print_summary(results: list[DocResult]) -> None:
    ENTITY_TYPES = [
        "persons", "organizations", "dates", "locations",
        "monetary", "ipc_sections", "case_numbers", "acts",
    ]
    COL = 14

    print("\n" + "=" * 90)
    print(f"{'DOC':<22}  "
          + "  ".join(f"{t[:COL]:^{COL}}" for t in ENTITY_TYPES)
          + f"  {'PASS':^5}")
    print("=" * 90)

    for r in results:
        marks = [
            ("✓" if r.checks.get(t, True) else "✗")
            for t in ENTITY_TYPES
        ]
        status = "✓ YES" if r.passed else "✗ NO "
        print(f"{r.doc_name:<22}  "
              + "  ".join(f"{m:^{COL}}" for m in marks)
              + f"  {status}")

    print("=" * 90)

    passed_total = sum(1 for r in results if r.passed)
    print(f"\nRESULT: {passed_total} / {len(results)} document types passed all checks.")

    failures = [r for r in results if not r.passed]
    if failures:
        print("\nFAILING CHECKS — detail:")
        for r in failures:
            print(f"\n  {r.doc_name}:")
            for etype, ok in r.checks.items():
                if not ok:
                    print(f"    ✗ {etype}: expected keywords {r.missing[etype]} "
                          f"not found in {r.extracted.get(etype, [])}")
    else:
        print("\n✓ All document types passed. Day 4 NER checkpoint complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Print all extracted entities per document")
    parser.add_argument("--doc",     type=int, default=None,
                        help="Run only this document number (1, 2, or 3)")
    args = parser.parse_args()

    docs_to_run = DOCUMENTS
    if args.doc:
        docs_to_run = [d for d in DOCUMENTS if d["id"] == args.doc]
        if not docs_to_run:
            print(f"Document {args.doc} not found. Valid: 1, 2, 3")
            sys.exit(1)

    print("\nLexShield AI — Day 4 NER Test")
    print(f"Running NER on {len(docs_to_run)} document type(s)...\n")

    results = [run_document(d, verbose=args.verbose) for d in docs_to_run]
    print_summary(results)