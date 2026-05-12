"""
Day 4 Checkpoint — Legal Knowledge Graph Tests
===============================================
Tests graph construction, section lookup, 2-hop traversal,
context formatting, and pipeline KG injection.

Run: pytest tests/test_knowledge_graph.py -v
"""

import pytest
from unittest.mock import patch
from rag.knowledge_graph import (
    LegalKnowledgeGraph,
    normalize_source,
    legal_kg,
    get_kg,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURE — manual-edges-only KG (no chunks.json needed, fast)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def kg():
    g = LegalKnowledgeGraph()
    with patch.object(g, "_auto_populate", return_value=None):
        g.build()
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — normalize_source
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("source,expected", [
    ("Indian Penal Code (IPC) 1860",                    "IPC"),
    ("Bharatiya Nyaya Sanhita (BNS) 2023",              "BNS"),
    ("Code of Criminal Procedure (CrPC) 1973",          "CrPC"),
    ("Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023",  "BNSS"),
    ("Indian Evidence Act 1872",                        "IEA"),
    ("Negotiable Instruments Act 1881",                 "NI"),
    ("Consumer Protection Act 2019",                    "CPA"),
])
def test_normalize_source(source, expected):
    result = normalize_source(source)
    assert result == expected, f"normalize_source({source!r}) → {result!r}, expected {expected!r}"
    print(f"\n✓ {source!r} → {expected!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Graph builds correctly
# ═══════════════════════════════════════════════════════════════════════════════

def test_graph_builds(kg):
    stats = kg.stats()
    assert stats["built"]                          == True
    assert stats["nodes"]                          >  50
    assert stats["edges"]                          >  40
    assert "section"  in stats["node_types"]
    assert "concept"  in stats["node_types"]
    assert "paired_act" in stats["edge_types"]
    assert "relates_to" in stats["edge_types"]

    print(f"\n✓ Graph: {stats['nodes']} nodes | {stats['edges']} edges")
    print(f"   node_types : {stats['node_types']}")
    print(f"   edge_types : {stats['edge_types']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Section 420 IPC (Day 4 spec checkpoint)
# Must return: 415 IPC, 417 IPC, 318 BNS, concept:cheating, concept:fraud
# ═══════════════════════════════════════════════════════════════════════════════

def test_section_420_ipc(kg):
    related = kg.query_related_sections("420", source_hint="Indian Penal Code")
    assert len(related) > 0, "No related nodes for Section 420 IPC"

    sections = {(r["act_key"], r["section"]) for r in related if r["node_type"] == "section"}
    concepts = {r["concept"] for r in related if r["node_type"] == "concept"}

    assert ("IPC", "415") in sections, f"415 IPC missing — got {sections}"
    assert ("IPC", "417") in sections, f"417 IPC missing — got {sections}"
    assert ("BNS", "318") in sections, f"318 BNS missing — got {sections}"
    assert "cheating"     in concepts, f"concept:cheating missing — got {concepts}"
    assert "fraud"        in concepts, f"concept:fraud missing (via concept hop) — got {concepts}"

    print(f"\n✓ Section 420 IPC → {len(related)} related nodes:")
    for r in related:
        print(f"   {r['node_id']:38} [{r['edge_type']}]  {r['edge_label']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Section 302 IPC → murder, BNS 101
# ═══════════════════════════════════════════════════════════════════════════════

def test_section_302_ipc(kg):
    related  = kg.query_related_sections("302", source_hint="Indian Penal Code (IPC) 1860")
    sections = {(r["act_key"], r["section"]) for r in related if r["node_type"] == "section"}
    concepts = {r["concept"] for r in related if r["node_type"] == "concept"}

    assert ("BNS", "101") in sections, f"BNS 101 missing — got {sections}"
    assert "murder"        in concepts, f"concept:murder missing — got {concepts}"

    print(f"\n✓ Section 302 IPC → BNS 101 (paired_act) + concept:murder")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Section 138 NI Act (cheque bounce cluster)
# ═══════════════════════════════════════════════════════════════════════════════

def test_section_138_ni(kg):
    related  = kg.query_related_sections("138", source_hint="Negotiable Instruments Act")
    sections = {(r["act_key"], r["section"]) for r in related if r["node_type"] == "section"}
    concepts = {r["concept"] for r in related if r["node_type"] == "concept"}

    assert ("NI", "139") in sections or ("NI", "141") in sections, \
        f"NI 139/141 missing — got {sections}"
    assert "cheque_bounce" in concepts, f"concept:cheque_bounce missing — got {concepts}"

    print(f"\n✓ Section 138 NI Act → cheque bounce cluster confirmed")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6 — CrPC 154 ↔ BNSS 173 (FIR paired_act)
# ═══════════════════════════════════════════════════════════════════════════════

def test_crpc_154_paired_bnss(kg):
    related  = kg.query_related_sections("154", source_hint="Code of Criminal Procedure")
    sections = {(r["act_key"], r["section"]) for r in related if r["node_type"] == "section"}
    concepts = {r["concept"] for r in related if r["node_type"] == "concept"}

    assert ("BNSS", "173") in sections, f"BNSS 173 missing — got {sections}"
    assert "fir"            in concepts, f"concept:fir missing — got {concepts}"

    print(f"\n✓ CrPC 154 → BNSS 173 (paired_act) + concept:fir")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7 — format_context produces valid prompt string
# ═══════════════════════════════════════════════════════════════════════════════

def test_format_context(kg):
    related = kg.query_related_sections("420", source_hint="Indian Penal Code")
    context = kg.format_context("420", "Indian Penal Code", related)

    assert "[Knowledge Graph]" in context
    assert "420"               in context
    assert "IPC"               in context
    assert "Section"           in context
    assert len(context)        > 80

    print(f"\n✓ format_context output:\n{context}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8 — Unknown section returns empty list
# ═══════════════════════════════════════════════════════════════════════════════

def test_unknown_section_graceful(kg):
    result = kg.query_related_sections("99999", source_hint="Indian Penal Code")
    assert result == []
    print(f"\n✓ Unknown section → [] gracefully")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9 — Singleton build with real chunks.json
# ═══════════════════════════════════════════════════════════════════════════════

def test_singleton_with_chunks():
    """
    Build singleton KG against real chunks.json.
    Verifies auto-populate creates statute + section nodes from corpus.
    """
    kg_instance = get_kg()
    stats       = kg_instance.stats()

    assert stats["built"]                              == True
    assert stats["nodes"]                              >  200
    assert stats["node_types"].get("statute", 0)       >  0
    assert stats["node_types"].get("section", 0)       >  0
    assert stats["node_types"].get("concept", 0)       >  0

    print(f"\n✓ Singleton KG (real corpus): {stats['nodes']} nodes | {stats['edges']} edges")
    print(f"   {stats['node_types']}")