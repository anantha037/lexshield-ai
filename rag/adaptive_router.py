"""
LexShield AI — Adaptive RAG Query Router
==========================================
Rule-based query complexity classifier.
No model, no memory cost — pure regex + keyword scoring.

Public function:
    classify_query_complexity(query) -> Literal["simple", "moderate", "complex"]

Complexity tiers and their pipeline implications:
─────────────────────────────────────────────────
simple:
  • Single act, single section, no comparison intent
  • Examples: "What is Section 302 IPC?" / "Define bail"
  • Pipeline: ChromaDB section fast-path only — skip BM25, skip
    query rewriter, skip CRAG
  • Saves ~3 Groq calls on simple lookups

moderate:
  • Multi-section OR single act with procedural/definitional angle
  • Examples: "Punishment and procedure for cheque bounce Section 138 NI Act"
  • Pipeline: BM25 + vector + reranker — skip query rewriter

complex:
  • Multiple acts OR explicit comparison OR cross-era (IPC vs BNS) OR
    multi-hop reasoning required
  • Examples:
    "Compare IPC Section 420 vs BNS Section 318 and which court to file in"
    "Cheque bounce under NI Act vs IPC Section 420 fraud — difference"
  • Pipeline: full — query rewriter + CRAG + multi-hop decomposition

Detection logic
───────────────
1. Count distinct act keywords mentioned
2. Count section references  (regex: Section \d+ / u/s \d+)
3. Detect comparison/cross-era keywords
4. Apply thresholds:
     complex:  acts >= 2  OR  comparison keywords found
     simple:   acts <= 1  AND  sections <= 1  AND  no comparison keywords
     moderate: everything else
"""

import re
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════════════
# ACT KEYWORD REGISTRY
# Each tuple: (regex_pattern, canonical_act_name)
# ═══════════════════════════════════════════════════════════════════════════════

_ACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b(indian\s+penal\s+code|ipc)\b',                     re.I), "IPC"),
    (re.compile(r'\b(bharatiya\s+nyaya\s+sanhita|bns)\b',               re.I), "BNS"),
    (re.compile(r'\b(code\s+of\s+criminal\s+procedure|crpc)\b',         re.I), "CrPC"),
    (re.compile(r'\b(bharatiya\s+nagarik\s+suraksha\s+sanhita|bnss)\b', re.I), "BNSS"),
    (re.compile(r'\b(indian\s+evidence\s+act|evidence\s+act)\b',        re.I), "Evidence Act"),
    (re.compile(r'\b(bharatiya\s+sakshya\s+adhiniyam|bsa)\b',           re.I), "BSA"),
    (re.compile(r'\b(negotiable\s+instruments\s+act|ni\s?act)\b',       re.I), "NI Act"),
    (re.compile(r'\b(consumer\s+protection\s+act|consumer\s+act)\b',    re.I), "Consumer Act"),
    (re.compile(r'\bpocso\b',                                            re.I), "POCSO"),
    (re.compile(r'\b(prevention\s+of\s+money\s+laundering|pmla)\b',     re.I), "PMLA"),
    (re.compile(r'\b(ndps|narcotic\s+drugs)\b',                         re.I), "NDPS"),
    (re.compile(r'\b(uapa|unlawful\s+activities)\b',                    re.I), "UAPA"),
    (re.compile(r'\b(rera|real\s+estate\s+regulation)\b',               re.I), "RERA"),
    (re.compile(r'\b(information\s+technology\s+act|it\s?act)\b',       re.I), "IT Act"),
    (re.compile(r'\b(code\s+of\s+civil\s+procedure|cpc)\b',             re.I), "CPC"),
    (re.compile(r'\b(insolvency\s+and\s+bankruptcy|ibc)\b',             re.I), "IBC"),
    (re.compile(r'\b(fema|foreign\s+exchange\s+management)\b',          re.I), "FEMA"),
    (re.compile(r'\b(sebi\s+act|securities\s+and\s+exchange\s+board)\b',re.I), "SEBI"),
    (re.compile(r'\b(motor\s+vehicles?\s+act|mv\s?act|mva)\b',          re.I), "MVA"),
    (re.compile(r'\b(domestic\s+violence\s+act|dv\s+act|pwdv)\b',       re.I), "DV Act"),
    (re.compile(r'\b(right\s+to\s+information|rti)\b',                  re.I), "RTI"),
    (re.compile(r'\b(dpdp|digital\s+personal\s+data\s+protection)\b',   re.I), "DPDP"),
    (re.compile(r'\b(transfer\s+of\s+property\s+act|topa?)\b',          re.I), "TP Act"),
    (re.compile(r'\b(prevention\s+of\s+corruption|pca)\b',              re.I), "PCA"),
    (re.compile(r'\b(posh|sexual\s+harassment\s+of\s+women\s+at\s+workplace)\b', re.I), "POSH"),
    (re.compile(r'\b(companies\s+act)\b',                               re.I), "Companies Act"),
    (re.compile(r'\b(arbitration\s+and\s+conciliation\s+act|arbitration\s+act)\b', re.I), "Arbitration Act"),
]

# ── Section reference detector ─────────────────────────────────────────────────
_SECTION_RE = re.compile(
    r'\b(?:section|sec\.?|u/s|under\s+section)\s*\d{1,4}[A-Z]?'
    r'|\bArticle\s+\d{1,3}[A-Z]?'
    r'|\bClause\s+\d{1,3}',
    re.IGNORECASE,
)

# ── Comparison / cross-era keywords ───────────────────────────────────────────
_COMPARISON_KEYWORDS: list[re.Pattern] = [
    re.compile(r'\bvs\.?\b|\bversus\b',                          re.I),
    re.compile(r'\bcompare\b|\bcomparison\b|\bdifference\b',     re.I),
    re.compile(r'\bunder\s+both\b|\bboth\s+acts?\b',             re.I),
    re.compile(r'\bwhich\s+(applies?|act|law)\b',                re.I),
    re.compile(r'\bold\s+law\b|\bnew\s+law\b',                   re.I),
    re.compile(r'\bbefore\s+2024\b|\bafter\s+2024\b',            re.I),
    re.compile(r'\breplaced\s+by\b|\bsuperseded\b',              re.I),
    re.compile(r'\band\s+also\b|\bboth\b.{0,30}\b(act|ipc|bns|crpc|bnss)\b', re.I),
    re.compile(r'\bhow\s+does\b.{0,40}\bdiffer\b',               re.I),
    re.compile(r'\bpre[-\s]2024\b|\bpost[-\s]2024\b',            re.I),
]

# ── Multi-hop / complex reasoning markers ─────────────────────────────────────
_MULTIHOP_KEYWORDS: list[re.Pattern] = [
    re.compile(r'\bwhich\s+court\b',                             re.I),
    re.compile(r'\bsteps?\s+to\s+file\b|\bhow\s+to\s+file\b',   re.I),
    re.compile(r'\bprocedure\s+and\b|\band\s+procedure\b',       re.I),
    re.compile(r'\bpunishment\s+and\s+(bail|procedure|appeal)\b',re.I),
    re.compile(r'\bremedies?\s+available\b',                     re.I),
    re.compile(r'\bwhat\s+happens\s+if\b',                       re.I),
    re.compile(r'\bcan\s+i\b.{0,30}\band\b.{0,30}\b(also|then)\b', re.I),
]


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_query_complexity(
    query: str,
) -> Literal["simple", "moderate", "complex"]:
    """
    Rule-based query complexity classifier.
    Returns "simple", "moderate", or "complex".

    No model — pure regex + keyword counting.
    Zero memory cost, runs in < 1 ms.

    Args:
        query: Raw or preprocessed user query string.

    Returns:
        "simple"   — single act + single section, no comparison
        "moderate" — multi-section OR procedural but single act
        "complex"  — multi-act OR comparison OR multi-hop
    """
    q = query.strip()

    # ── Count distinct acts mentioned ─────────────────────────────────────────
    matched_acts: set[str] = set()
    for pattern, act_name in _ACT_PATTERNS:
        if pattern.search(q):
            matched_acts.add(act_name)
    act_count = len(matched_acts)

    # ── Count section references ───────────────────────────────────────────────
    section_refs = _SECTION_RE.findall(q)
    section_count = len(section_refs)

    # ── Check comparison / cross-era keywords ─────────────────────────────────
    has_comparison = any(p.search(q) for p in _COMPARISON_KEYWORDS)

    # ── Check multi-hop markers ───────────────────────────────────────────────
    has_multihop = any(p.search(q) for p in _MULTIHOP_KEYWORDS)

    # ── Classification thresholds ─────────────────────────────────────────────
    if act_count >= 2 or has_comparison:
        complexity = "complex"
    elif act_count <= 1 and section_count <= 1 and not has_multihop:
        complexity = "simple"
    else:
        complexity = "moderate"

    print(
        f"[AdaptiveRouter] acts={act_count} sections={section_count} "
        f"comparison={has_comparison} multihop={has_multihop} "
        f"→ complexity={complexity!r}"
    )
    return complexity


# ── Debug helper ───────────────────────────────────────────────────────────────

def explain_complexity(query: str) -> None:
    """Prints a detailed breakdown. Use in REPL for debugging."""
    matched_acts = {name for pat, name in _ACT_PATTERNS if pat.search(query)}
    section_refs = _SECTION_RE.findall(query)
    comparisons  = [p.pattern for p in _COMPARISON_KEYWORDS if p.search(query)]
    multihops    = [p.pattern for p in _MULTIHOP_KEYWORDS   if p.search(query)]
    result       = classify_query_complexity(query)

    print(f"\n[AdaptiveRouter] Query: {query!r}")
    print(f"  Acts found    : {sorted(matched_acts) or 'none'}")
    print(f"  Sections found: {section_refs or 'none'}")
    print(f"  Comparisons   : {comparisons or 'none'}")
    print(f"  Multi-hop     : {multihops or 'none'}")
    print(f"  → Complexity  : {result!r}\n")