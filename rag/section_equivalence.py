"""
LexShield AI — Section Equivalence Lookup
==========================================
Provides a structured, auditable view of IPC↔BNS (and CrPC↔BNSS) section
equivalences stored in data/legal_graph.json.

Reuses rag.knowledge_graph.load_graph() — the lru_cache singleton — so the
JSON file is never opened a second time at runtime.

Public API
----------
    lookup_equivalent(act: str, section: str) -> dict | None

    act     — one of: IPC, BNS, CrPC, BNSS,
                      "Evidence Act", "Indian Evidence Act" (maps to IEA),
                      BSA
    section — raw section string, e.g. "302" or "304B"

    Returns a dict on success (see _RETURN_SHAPE below), or None when:
      • the node is not in the graph, or
      • the node's bns_equiv field is null / absent.

Return shape
------------
    {
        "source": {
            "act":     str,   # e.g. "IPC"
            "section": str,   # e.g. "302"
            "label":   str | None,
        },
        "target": {
            "act":     str,   # e.g. "BNS"
            "section": str,   # e.g. "103"
            "label":   str | None,
        },
        "also_merged_from": [          # may be empty
            {"act": str, "section": str, "label": str | None},
            ...
        ],
        "status": "verified" | "unverified",
    }

also_merged_from
----------------
For many-to-one cases (e.g. IPC 415 / 417 / 418 / 420 all merged into
BNS 318) the target node's bns_equiv points at one canonical source, but
multiple source sections share that same target.  We reverse-scan the
entire graph to find every node whose bns_equiv equals the target node ID,
then include them (excluding the source node itself) in also_merged_from.

status
------
Taken from the *source* node's optional "equiv_status" field.
  • "verified"   — field absent (default assumption for confirmed pairs)
  • "unverified" — field == "unverified_label" (label not yet confirmed)
"""

from __future__ import annotations

import logging
from typing import Optional

# Re-use the singleton loader — never opens the JSON a second time.
from rag.knowledge_graph import load_graph

logger = logging.getLogger(__name__)

# ── Act-name → JSON key prefix mapping ────────────────────────────────────────

_ACT_PREFIX: dict[str, str] = {
    # Substantive criminal law
    "IPC":                  "IPC",
    "BNS":                  "BNS",
    # Procedural law
    "CrPC":                 "CrPC",
    "BNSS":                 "BNSS",
    # Evidence law
    "Evidence Act":         "IEA",
    "Indian Evidence Act":  "IEA",
    "IEA":                  "IEA",
    "BSA":                  "BSA",
    # Negotiable Instruments (no BNS equiv, but accept for graceful None)
    "NI":                   "NI",
    "NI Act":               "NI",
    "Negotiable Instruments Act": "NI",
    # Consumer Protection (no BNS equiv)
    "CPA":                  "CPA",
}


def _act_to_prefix(act: str) -> Optional[str]:
    """
    Map a caller-supplied act name to the JSON key prefix used in legal_graph.json.
    Case-insensitive, strips extra whitespace.
    Returns None if the act is not recognised.
    """
    normalised = act.strip()
    # Exact match first (case-sensitive keys are checked first)
    if normalised in _ACT_PREFIX:
        return _ACT_PREFIX[normalised]
    # Case-insensitive fallback
    lower = normalised.lower()
    for key, prefix in _ACT_PREFIX.items():
        if key.lower() == lower:
            return prefix
    return None


def _node_to_stub(node_id: str, node: dict) -> dict:
    """
    Build a compact {act, section, label} dict from a node_id and node payload.
    e.g. "IPC_302" -> {"act": "IPC", "section": "302", "label": "..."}
    """
    parts   = node_id.split("_", 1)
    act     = parts[0]      if len(parts) == 2 else node_id
    section = parts[1]      if len(parts) == 2 else ""
    return {
        "act":     act,
        "section": section,
        "label":   node.get("label"),   # may be None (unverified nodes)
    }


# ── Main public function ───────────────────────────────────────────────────────

def lookup_equivalent(act: str, section: str) -> Optional[dict]:
    """
    Look up the BNS/BNSS equivalent for a legacy (or current) section.

    Args:
        act:     Act name — one of IPC, BNS, CrPC, BNSS,
                 "Evidence Act", "Indian Evidence Act", BSA, NI, CPA, …
        section: Raw section number string, e.g. "302" or "304B".

    Returns:
        Result dict (see module docstring) on success, None otherwise.

    Examples:
        lookup_equivalent("IPC", "302")
        -> {"source": {"act": "IPC", "section": "302", "label": "..."},
            "target": {"act": "BNS", "section": "103", "label": "..."},
            "also_merged_from": [],
            "status": "verified"}

        lookup_equivalent("IPC", "420")
        -> {"source": {...}, "target": {...},
            "also_merged_from": [
                {"act": "IPC", "section": "415", ...},
                {"act": "IPC", "section": "417", ...},
                {"act": "IPC", "section": "418", ...},
            ],
            "status": "verified"}

        lookup_equivalent("NI", "138")   -> None  (no bns_equiv)
        lookup_equivalent("IPC", "9999") -> None  (node absent)
    """
    graph = load_graph()

    # ── 1. Resolve act name → prefix ──────────────────────────────────────────
    prefix = _act_to_prefix(act)
    if prefix is None:
        logger.warning("[SE] lookup_equivalent: unrecognised act %r", act)
        return None

    # ── 2. Build the node ID and look it up ───────────────────────────────────
    node_id = f"{prefix}_{section}"
    source_node = graph.get(node_id)
    if source_node is None:
        logger.debug("[SE] lookup_equivalent: node %r not in graph", node_id)
        return None

    # ── 3. Read bns_equiv from the source node ─────────────────────────────────
    target_id: Optional[str] = source_node.get("bns_equiv")
    if not target_id:
        # null or absent → no known equivalent
        return None

    target_node = graph.get(target_id)
    if target_node is None:
        # bns_equiv points at a node that doesn't exist in the graph yet
        logger.warning(
            "[SE] lookup_equivalent: target node %r (bns_equiv of %r) is missing from graph",
            target_id,
            node_id,
        )
        return None

    # ── 4. Reverse-scan for also_merged_from ──────────────────────────────────
    # Find every OTHER node in the graph whose bns_equiv == target_id.
    also_merged: list[dict] = []
    for nid, ndata in graph.items():
        if nid == node_id:
            continue                        # skip the source itself
        if ndata.get("bns_equiv") == target_id:
            also_merged.append(_node_to_stub(nid, ndata))

    # ── 5. Determine status ────────────────────────────────────────────────────
    raw_status = source_node.get("equiv_status", "")
    status = "verified" if raw_status == "verified" else "unverified"

    # ── 6. Assemble result ────────────────────────────────────────────────────
    return {
        "source":           _node_to_stub(node_id,  source_node),
        "target":           _node_to_stub(target_id, target_node),
        "also_merged_from": also_merged,
        "status":           status,
    }


# ── is_equivalence_query ──────────────────────────────────────────────────────

# Keywords that signal "what maps to what?" intent.
_EQUIV_KEYWORDS: tuple[str, ...] = (
    "equivalent",
    "corresponds to",
    "correspond to",
    "same as",
    "renamed to",
    "now called",
    "used to be",
    "old section",
    "replaced by",
    "maps to",
    "bns equivalent",
    "ipc equivalent",
)

# Phrases that require the word "now" elsewhere in the query to count.
_EQUIV_KEYWORDS_REQUIRE_NOW: tuple[str, ...] = (
    "what was",
)

# Maps full source-hint strings (returned by extract_sections_and_sources)
# → a canonical short act name used for distinctness comparison.
# Two entries with DIFFERENT act names in the same query trigger condition 2.
# IPC and BNS are intentionally distinct — that is the point of this function.
_SOURCE_HINT_TO_ACT: dict[str, str] = {
    "Indian Penal Code":                  "IPC",
    "Bharatiya Nyaya Sanhita":            "BNS",
    "Code of Criminal Procedure":         "CrPC",
    "Bharatiya Nagarik Suraksha Sanhita": "BNSS",
    "Indian Evidence Act":                "Evidence Act",
    "Bharatiya Sakshya Adhiniyam":        "BSA",
}

# Bare abbreviation fallback when source_hint is None.
_TOKEN_TO_ACT: dict[str, str] = {
    "ipc":          "IPC",
    "bns":          "BNS",
    "crpc":         "CrPC",
    "bnss":         "BNSS",
    "evidence act": "Evidence Act",
    "bsa":          "BSA",
}


def source_hint_to_act_name(source_hint: Optional[str], query_lower: str) -> Optional[str]:
    """
    Resolve a source_hint string (or bare query tokens when hint is None)
    to a canonical short act name (e.g. "IPC", "BNS", "CrPC").
    Returns None when the act cannot be identified.
    """
    if source_hint:
        for key, act_name in _SOURCE_HINT_TO_ACT.items():
            if key.lower() in source_hint.lower():
                return act_name
    else:
        for token, act_name in _TOKEN_TO_ACT.items():
            if token in query_lower:
                return act_name
    return None


def is_equivalence_query(query: str) -> tuple[bool, list[tuple[str, str]]]:
    """
    Detect whether a query is asking for a cross-act section equivalence.

    Args:
        query: The raw user query string.

    Returns:
        (True,  [(act, section), ...])  — equivalence query detected.
        (False, [])                     — not an equivalence query.

    Trigger conditions (OR):

    1. Query contains any of the _EQUIV_KEYWORDS phrases (case-insensitive).
       Phrases in _EQUIV_KEYWORDS_REQUIRE_NOW only count when "now" also
       appears in the query (handles "what was … now").

    2. extract_sections_and_sources() returns results whose source_hints
       resolve to 2+ DISTINCT act names (IPC, BNS, CrPC, BNSS, Evidence
       Act, BSA).  IPC and BNS are distinct names and will trigger together.
       A query with only a single act name, or with no identifiable act,
       does NOT trigger condition 2.

    NOTE: extract_sections_and_sources is imported lazily to avoid pulling
    in ChromaDB / BM25 at module-import time.
    """
    q_lower = query.lower()

    # ── Condition 1: equivalence keywords ─────────────────────────────────────
    for kw in _EQUIV_KEYWORDS_REQUIRE_NOW:
        if kw in q_lower and "now" in q_lower:
            return True, []

    for kw in _EQUIV_KEYWORDS:
        if kw in q_lower:
            return True, []

    # ── Condition 2: sections from 2+ distinct act names ──────────────────────
    try:
        from rag.hybrid_search import extract_sections_and_sources
        pairs = extract_sections_and_sources(query)
    except Exception:
        # Vector-store stack not running — skip condition 2 gracefully.
        return False, []

    if len(pairs) < 2:
        return False, []

    acts_seen: set[str] = set()
    act_section_pairs: list[tuple[str, str]] = []

    for section_number, source_hint in pairs:
        act_name = source_hint_to_act_name(source_hint, q_lower)
        if act_name is not None:
            acts_seen.add(act_name)
            act_section_pairs.append((act_name, section_number))

    if len(acts_seen) >= 2:
        return True, act_section_pairs

    return False, []
