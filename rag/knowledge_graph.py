"""
LexShield AI — Legal Knowledge Graph (Session 3 — Full Implementation)
========================================================================
GraphRAG layer over data/legal_graph.json.

Provides:
  load_graph()               → loads JSON once, cached as module singleton
  get_related_sections()     → BFS traversal up to N hops
  get_bns_equivalent()       → IPC→BNS or BNS→IPC lookup
  get_era()                  → "legacy" | "current" | "unknown"
  enrich_retrieval()         → augments chunk pool with graph-connected sections

Wire-in point (agents/graph.py → legal_rag_node):
  After NER extracts section IDs from query, call enrich_retrieval() to
  add graph-connected chunks to the pool before reranking.

Wire-in point (rag/pipeline.py → _inject_kg):
  Already uses rag/knowledge_graph.py (the NetworkX version).
  This module (data/legal_graph.json based) is the NEW implementation —
  lighter, no NetworkX dependency, works on Windows i5/8GB.

Backward-compatible singleton:
  from rag.knowledge_graph import get_kg
  still works — get_kg() returns the LegalKnowledgeGraph (NetworkX) object.
  The new flat-JSON functions are imported directly:
  from rag.knowledge_graph import load_graph, get_related_sections, enrich_retrieval
"""

import json
import os
from functools import lru_cache
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GRAPH_JSON     = os.path.join(_PROJECT_ROOT, "data", "legal_graph.json")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD GRAPH  (cached after first call)
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_graph() -> dict:
    """
    Load data/legal_graph.json once and cache it.

    Returns:
        dict keyed by node_id (e.g. "IPC_302"), each value is a node dict:
        {
            "label":       str,
            "bns_equiv":   str | null,
            "related":     [str, ...],
            "parent_act":  str,
            "category":    str,
            "era":         str,   # "legacy" | "current"
            "risk":        str    # "low" | "medium" | "high" | "critical"
        }
    """
    if not os.path.exists(_GRAPH_JSON):
        print(f"[KG] WARNING: {_GRAPH_JSON} not found — returning empty graph")
        return {}

    with open(_GRAPH_JSON, encoding="utf-8") as f:
        graph = json.load(f)

    print(f"[KG] Loaded legal_graph.json: {len(graph)} nodes")
    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISE NODE IDS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise(node_id: str) -> str:
    """
    Accepts various formats and maps to the JSON key format (ACT_SECTION).

    Examples:
        "IPC 302"   → "IPC_302"
        "ipc_302"   → "IPC_302"
        "Section 302 IPC" → "IPC_302"
        "NI Act 138"      → "NI_138"
        "BNS 85"          → "BNS_85"
        "CrPC 154"        → "CrPC_154"
        "BNSS 173"        → "BNSS_173"
    """
    s = node_id.strip()

    # Handle "Section 302 IPC" or "Section 138 NI Act"
    import re
    m = re.match(
        r'[Ss]ection\s+(\d+[A-Za-z_]*)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
        s
    )
    if m:
        sec = m.group(1)
        act = m.group(2).upper().replace(" ", "").replace("ACT", "").strip("_")
        return f"{act}_{sec}".upper()

    # Handle "IPC 302", "NI Act 138", "CrPC 154"
    m2 = re.match(
        r'([A-Za-z]+(?:\s*[A-Za-z]+)?)\s+(\d+[A-Za-z_]*)',
        s
    )
    if m2:
        act = m2.group(1).upper().replace(" ", "").replace("ACT", "")
        sec = m2.group(2)
        return f"{act}_{sec}".upper()

    # Already in correct format or close ("IPC_302", "ipc_302")
    return s.upper().replace(" ", "_")


def _lookup(node_id: str, graph: dict) -> Optional[dict]:
    """Try to find a node by its raw or normalised ID."""
    if node_id in graph:
        return graph[node_id]
    norm = _normalise(node_id)
    if norm in graph:
        return graph[norm]
    return None


def _resolve_id(node_id: str, graph: dict) -> Optional[str]:
    """Return the actual key present in graph dict, or None."""
    if node_id in graph:
        return node_id
    norm = _normalise(node_id)
    if norm in graph:
        return norm
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def get_related_sections(section_id: str, hops: int = 1) -> list[str]:
    """
    BFS traversal of the legal graph starting from section_id.

    Args:
        section_id: node ID in any supported format (e.g. "IPC_302", "Section 302 IPC")
        hops:       traversal depth (default 1, max 3 for performance)

    Returns:
        Ordered list of related node IDs (strings), closest first.
        Excludes the starting node itself.
        Returns [] if section_id not found in graph.

    Example:
        get_related_sections("IPC_420", hops=1)
        → ["IPC_415", "IPC_417", "IPC_406", "IPC_471", "BNS_318"]
    """
    graph  = load_graph()
    hops   = min(hops, 3)  # Safety cap for performance on i5/8GB

    start  = _resolve_id(section_id, graph)
    if start is None:
        return []

    visited: dict[str, int] = {start: 0}   # node_id → hop_distance
    queue:   list[tuple[str, int]] = [(start, 0)]

    while queue:
        current, depth = queue.pop(0)
        if depth >= hops:
            continue

        node = graph.get(current, {})

        # Traverse "related" edges
        for neighbour in node.get("related", []):
            n_resolved = _resolve_id(neighbour, graph)
            target     = n_resolved if n_resolved else neighbour
            if target not in visited:
                visited[target] = depth + 1
                queue.append((target, depth + 1))

        # Traverse "bns_equiv" edge (bidirectional)
        equiv = node.get("bns_equiv")
        if equiv:
            e_resolved = _resolve_id(equiv, graph)
            target     = e_resolved if e_resolved else equiv
            if target and target not in visited:
                visited[target] = depth + 1
                queue.append((target, depth + 1))

    # Return sorted by hop distance, excluding start node
    result = sorted(
        [(nid, d) for nid, d in visited.items() if nid != start],
        key=lambda x: x[1]
    )
    return [nid for nid, _ in result]


def get_bns_equivalent(section_id: str) -> Optional[str]:
    """
    Return the BNS/BNSS equivalent of a legacy IPC/CrPC section, or vice versa.

    Args:
        section_id: e.g. "IPC_302", "CrPC_154", "BNS_101"

    Returns:
        Equivalent node ID string, or None if no mapping exists.

    Examples:
        get_bns_equivalent("IPC_302")  → "BNS_101"
        get_bns_equivalent("BNS_101")  → "IPC_302"
        get_bns_equivalent("CrPC_154") → "BNSS_173"
    """
    graph = load_graph()
    node  = _lookup(section_id, graph)
    if node is None:
        return None
    equiv = node.get("bns_equiv")
    if not equiv:
        return None
    # Resolve to actual key if possible
    resolved = _resolve_id(equiv, graph)
    return resolved if resolved else equiv


def get_era(section_id: str) -> str:
    """
    Return "legacy" (IPC/CrPC/IEA era) or "current" (BNS/BNSS/BSA era).

    Returns "unknown" if section not found in graph.
    """
    graph = load_graph()
    node  = _lookup(section_id, graph)
    if node is None:
        return "unknown"
    return node.get("era", "unknown")


def get_node_info(section_id: str) -> Optional[dict]:
    """
    Return the full node dict for a section ID, or None if not found.

    Useful for building context strings in prompts.
    """
    graph = load_graph()
    return _lookup(section_id, graph)


def get_risk_level(section_id: str) -> str:
    """
    Return the risk level string for a section: "low" | "medium" | "high" | "critical".
    Returns "unknown" if not found.
    """
    graph = load_graph()
    node  = _lookup(section_id, graph)
    if node is None:
        return "unknown"
    return node.get("risk", "unknown")


def format_related_context(section_id: str, hops: int = 1) -> str:
    """
    Build a compact knowledge-graph context string for prompt injection.

    Example output:
        [KG] IPC_420 related sections (hop=1):
          • IPC_415 — IPC Section 415 — Cheating (definition) [related, legacy]
          • BNS_318 — BNS Section 318 — Cheating (definition and punishment) [bns_equiv, current]
    """
    graph   = load_graph()
    related = get_related_sections(section_id, hops=hops)
    if not related:
        return ""

    start_node = _lookup(section_id, graph)
    start_label = start_node.get("label", section_id) if start_node else section_id

    lines = [f"[Knowledge Graph] {section_id} — {start_label}"]
    lines.append(f"  Related sections (within {hops} hop{'s' if hops > 1 else ''}):")

    start_resolved = _resolve_id(section_id, graph)
    start_info     = graph.get(start_resolved, {}) if start_resolved else {}
    direct_related = set(start_info.get("related", []))
    equiv          = start_info.get("bns_equiv", "")

    for nid in related[:10]:  # Cap at 10 for prompt length
        node = _lookup(nid, graph)
        if node is None:
            continue
        label    = node.get("label", nid)
        era      = node.get("era", "?")
        resolved = _resolve_id(nid, graph) or nid

        if resolved == equiv or nid == equiv:
            rel_type = "bns_equiv"
        elif resolved in direct_related or nid in direct_related:
            rel_type = "related"
        else:
            rel_type = "2nd-hop"

        lines.append(f"    • {resolved} — {label} [{rel_type}, {era}]")

    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────────────────
# ACT PAIRING HELPER  (BUG FIX 2)
# ───────────────────────────────────────────────────────────────────────────────

_PAIRED_ACTS = [
    ('Indian Penal Code',          'Bharatiya Nyaya Sanhita'),
    ('Code of Criminal Procedure', 'Bharatiya Nagarik Suraksha Sanhita'),
    ('Indian Evidence Act',        'Bharatiya Sakshya Adhiniyam'),
]


def _is_paired_act(act1: str, act2: str) -> bool:
    """
    IPC↔BNS and CrPC↔BNSS and IEA↔BSA are valid substitution pairs.
    All other cross-act combinations are NOT valid pairs.
    Used by enrich_retrieval to allow BNS equivalent injection when
    querying IPC sections (same crime, different era), while blocking
    CrPC sections appearing when querying IPC sections.
    """
    a1 = act1.lower()
    a2 = act2.lower()
    for p1, p2 in _PAIRED_ACTS:
        if (p1.lower() in a1 and p2.lower() in a2) or \
           (p2.lower() in a1 and p1.lower() in a2):
            return True
    return False


# ───────────────────────────────────────────────────────────────────────────────
# ENRICH RETRIEVAL  (main integration point with RAG pipeline)
# ───────────────────────────────────────────────────────────────────────────────

def enrich_retrieval(
    ner_sections:        list[str],
    chunk_pool:          list[dict],
    chroma_client=None,
    bypass_score_filter: bool = True,
    act_hint:            Optional[str] = None,   # BUG FIX 2
) -> list[dict]:
    """
    For each NER-extracted section, traverse the graph to find related sections,
    then fetch their chunks from the vector store and add to the pool.

    This implements GraphRAG: instead of relying purely on semantic similarity,
    we use structural legal relationships (IPC→BNS equivalents, definitional
    chains, penalty↔definition pairs) to ensure completeness of retrieval.

    Args:
        ner_sections:        list of section IDs extracted by NER from the query
                             (e.g. ["IPC_420", "Section 138 NI Act"])
        chunk_pool:          existing list of retrieved chunks (dicts with chunk_id,
                             text, source, section, etc.)
        chroma_client:       ChromaDB client (optional — uses vectorstore singleton
                             if not provided)
        bypass_score_filter: if True, graph-connected chunks are added regardless
                             of their similarity score (they are structurally relevant)

    Returns:
        Augmented chunk_pool list with graph-connected chunks appended.
        New chunks are tagged with retrieval_source="knowledge_graph".
        Deduplication is applied (no chunk_id appears twice).

    Example:
        User asks about "IPC 420 cheating".
        NER extracts: ["IPC_420"]
        Graph traversal finds: ["IPC_415", "IPC_417", "BNS_318", "IPC_406"]
        enrich_retrieval fetches chunks for all of these and adds them,
        ensuring the synthesiser sees both the definition (415) and the
        BNS equivalent (318) alongside the punishment section (420).
    """
    if not ner_sections:
        return chunk_pool

    graph = load_graph()
    if not graph:
        return chunk_pool

    # Collect all related section IDs across all NER hits
    all_related: dict[str, str] = {}  # section_id → source_ner_section

    for raw_id in ner_sections:
        resolved = _resolve_id(raw_id, graph)
        if resolved is None:
            continue

        related_all = get_related_sections(resolved, hops=1)

        # BUG FIX 2: when act_hint is known, filter related nodes to same act family
        if act_hint:
            act_kw = act_hint.split()[0].lower()
            related: list[str] = []
            for rel_id in related_all:
                node = graph.get(rel_id, {})
                node_act = node.get('parent_act', '')
                # Accept: same act keyword in parent_act OR acts are a valid pair
                same_family = (
                    act_kw in node_act.lower() or
                    _is_paired_act(act_hint, node_act)
                )
                if same_family:
                    related.append(rel_id)
            if related_all and not related:
                # Fallback: keep all if filtering removed everything
                # (graph may not have parent_act populated for all nodes)
                related = related_all
                print(f"[KG] act_hint filter removed all related for {resolved!r}; using unfiltered")
            else:
                if len(related) < len(related_all):
                    print(f"[KG] act_hint={act_hint!r}: {resolved!r} "
                          f"{len(related_all)}→{len(related)} related after act filter")
        else:
            related = related_all

        for rel_id in related:
            if rel_id not in all_related:
                all_related[rel_id] = resolved

        # Always include the BNS/IPC equivalent (paired acts are allowed)
        equiv = get_bns_equivalent(resolved)
        if equiv and equiv not in all_related:
            # But only if it's from a paired act (not a random same-numbered section)
            if act_hint is None or _is_paired_act(act_hint, graph.get(equiv, {}).get('parent_act', '')):
                all_related[equiv] = resolved

    if not all_related:
        return chunk_pool

    # Build set of already-present chunk IDs (for dedup)
    existing_ids: set[str] = {c.get("chunk_id", "") for c in chunk_pool}

    # Fetch chunks from vector store for each related section
    try:
        if chroma_client is not None:
            new_chunks = _fetch_via_chroma_client(
                all_related, existing_ids, graph, chroma_client
            )
        else:
            new_chunks = _fetch_via_vectorstore_singleton(
                all_related, existing_ids, graph
            )
    except Exception as e:
        print(f"[KG] enrich_retrieval fetch error (non-fatal): {e}")
        return chunk_pool

    if new_chunks:
        print(f"[KG] enrich_retrieval: added {len(new_chunks)} graph-connected chunks "
              f"for {len(ner_sections)} NER section(s)")

    return chunk_pool + new_chunks


def _fetch_via_vectorstore_singleton(
    all_related: dict[str, str],
    existing_ids: set[str],
    graph: dict,
) -> list[dict]:
    """
    Use rag.vectorstore singleton to fetch chunks by section ID.
    Falls back gracefully if vectorstore is unavailable.
    """
    try:
        from rag.vectorstore import vectorstore
    except ImportError:
        return []

    new_chunks: list[dict] = []

    for rel_id, source_section in all_related.items():
        node = _lookup(rel_id, graph)
        if node is None:
            continue

        # Parse act_key and section number from rel_id (e.g. "IPC_302" → act="IPC", sec="302")
        parts      = rel_id.split("_", 1)
        act_key    = parts[0] if len(parts) == 2 else ""
        sec_number = parts[1] if len(parts) == 2 else rel_id

        # Map act_key to partial source name for vectorstore lookup
        act_source_map = {
            "IPC":  "Indian Penal Code",
            "BNS":  "Bharatiya Nyaya Sanhita",
            "CrPC": "Code of Criminal Procedure",
            "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
            "NI":   "Negotiable Instruments",
            "CPA":  "Consumer Protection",
            "PWA":  "Payment of Wages",
            "IDA":  "Industrial Disputes",
            "TPA":  "Transfer of Property",
            "IEA":  "Indian Evidence",
            "BSA":  "Bharatiya Sakshya",
        }
        source_partial = act_source_map.get(act_key, "")

        try:
            if source_partial:
                hits = vectorstore.get_by_section(sec_number, source_hint=source_partial)
            else:
                hits = vectorstore.get_by_section(sec_number, source_hint=None)
        except Exception:
            hits = []

        for chunk in hits:
            cid = chunk.get("chunk_id", "")
            if cid and cid not in existing_ids:
                chunk["retrieval_source"] = "knowledge_graph"
                chunk["hybrid_score"]     = chunk.get("hybrid_score", 0.6)
                chunk["kg_source_section"] = source_section
                chunk["kg_related_id"]     = rel_id
                new_chunks.append(chunk)
                existing_ids.add(cid)

    return new_chunks


def _fetch_via_chroma_client(
    all_related: dict[str, str],
    existing_ids: set[str],
    graph: dict,
    chroma_client,
) -> list[dict]:
    """
    Fetch chunks directly via a ChromaDB client object.
    Uses metadata filter on 'section' field.
    """
    new_chunks: list[dict] = []

    try:
        collection = chroma_client.get_collection("legal_chunks")
    except Exception:
        return []

    for rel_id, source_section in all_related.items():
        parts      = rel_id.split("_", 1)
        sec_number = parts[1] if len(parts) == 2 else rel_id

        try:
            results = collection.get(
                where={"section": {"$eq": sec_number}},
                limit=3,
            )
            docs      = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            ids       = results.get("ids", [])

            for doc, meta, cid in zip(docs, metadatas, ids):
                if cid not in existing_ids:
                    chunk = {
                        "chunk_id":          cid,
                        "text":              doc,
                        "source":            meta.get("source", ""),
                        "section":           meta.get("section", sec_number),
                        "section_title":     meta.get("section_title", ""),
                        "category":          meta.get("category", ""),
                        "era":               meta.get("era", ""),
                        "hybrid_score":      0.6,
                        "retrieval_source":  "knowledge_graph",
                        "kg_source_section": source_section,
                        "kg_related_id":     rel_id,
                    }
                    new_chunks.append(chunk)
                    existing_ids.add(cid)
        except Exception:
            continue

    return new_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH STATS  (for /api/kg/stats endpoint or debugging)
# ═══════════════════════════════════════════════════════════════════════════════

def graph_stats() -> dict:
    """Return summary statistics about the loaded graph."""
    graph = load_graph()
    if not graph:
        return {"loaded": False, "nodes": 0}

    era_counts:      dict[str, int] = {}
    category_counts: dict[str, int] = {}
    risk_counts:     dict[str, int] = {}
    act_counts:      dict[str, int] = {}
    total_edges = 0

    for node_id, node in graph.items():
        era      = node.get("era",        "unknown")
        category = node.get("category",   "unknown")
        risk     = node.get("risk",        "unknown")
        act      = node.get("parent_act", "unknown")

        era_counts[era]          = era_counts.get(era, 0)          + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        risk_counts[risk]        = risk_counts.get(risk, 0)        + 1
        act_counts[act]          = act_counts.get(act, 0)          + 1

        # Count edges: related list + bns_equiv (if present)
        total_edges += len(node.get("related", []))
        if node.get("bns_equiv"):
            total_edges += 1

    return {
        "loaded":           True,
        "nodes":            len(graph),
        "edges_approx":     total_edges,
        "by_era":           era_counts,
        "by_category":      category_counts,
        "by_risk":          risk_counts,
        "by_act":           act_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY  — keep existing NetworkX-based class usable
# ═══════════════════════════════════════════════════════════════════════════════
# The file rag/knowledge_graph.py already contains the NetworkX LegalKnowledgeGraph
# class and get_kg() function.  This module (also at rag/knowledge_graph.py after
# replacement) now replaces that file entirely while preserving the get_kg() interface.
#
# If you are keeping BOTH files (old NetworkX + this new flat-JSON version),
# import this module as:
#     from rag.knowledge_graph import load_graph, get_related_sections, enrich_retrieval
# and keep the old NetworkX class under a different name.
#
# The code below re-exports get_kg() pointing to the flat-JSON loader so that
# existing call sites (pipeline.py _inject_kg) continue to work.

class _FlatGraphAdapter:
    """
    Thin adapter that wraps the flat JSON graph behind the interface that
    rag/pipeline.py's _inject_kg() expects from the old NetworkX-based KG.

    Methods implemented:
        query_related_sections(section, source_hint, hops) → list[dict]
        format_context(section, source_hint, related)      → str
        _built: bool
        build()
    """

    def __init__(self):
        self._built = False

    def build(self, *args, **kwargs) -> None:
        load_graph()  # Triggers lru_cache
        self._built = True

    def query_related_sections(
        self,
        section:     str,
        source_hint: Optional[str] = None,
        hops:        int           = 2,
    ) -> list[dict]:
        """
        Mimics the NetworkX KG interface.
        Returns list of dicts with keys: node_id, node_type, label, section,
        act_key, section_title, concept, edge_type, edge_label.
        """
        graph = load_graph()

        # Resolve section + source_hint → node_id
        if source_hint:
            # Try to derive act_key from source_hint
            import re
            acronym = re.search(r'\(([A-Z]{2,6})\)', source_hint)
            if acronym:
                act_key = acronym.group(1)
            else:
                hint_lower = source_hint.lower()
                hint_map = {
                    "indian penal code": "IPC",
                    "bharatiya nyaya": "BNS",
                    "code of criminal": "CrPC",
                    "bharatiya nagarik": "BNSS",
                    "negotiable": "NI",
                    "consumer protection": "CPA",
                    "payment of wages": "PWA",
                    "industrial disputes": "IDA",
                    "transfer of property": "TPA",
                }
                act_key = next(
                    (v for k, v in hint_map.items() if k in hint_lower), ""
                )
            candidate = f"{act_key}_{section}" if act_key else section
        else:
            candidate = section

        resolved = _resolve_id(candidate, graph)
        if resolved is None:
            # Try without act_key
            for key in graph:
                if key.endswith(f"_{section}"):
                    resolved = key
                    break

        if resolved is None:
            return []

        related_ids = get_related_sections(resolved, hops=hops)

        results = []
        start_node = graph.get(resolved, {})
        direct_related = set(start_node.get("related", []))
        equiv          = start_node.get("bns_equiv", "")

        for nid in related_ids:
            node = graph.get(nid, {})
            if not node:
                continue

            parts    = nid.split("_", 1)
            act_key  = parts[0] if len(parts) == 2 else ""
            sec_num  = parts[1] if len(parts) == 2 else nid

            if nid == equiv or nid == _resolve_id(equiv or "", graph):
                edge_type = "paired_act"
            elif nid in direct_related or any(
                r == nid for r in direct_related
            ):
                edge_type = "related"
            else:
                edge_type = "related"

            results.append({
                "node_id":       nid,
                "node_type":     "section",
                "label":         node.get("label", nid),
                "section":       sec_num,
                "act_key":       act_key,
                "section_title": node.get("label", "").split("—")[-1].strip()[:60]
                                 if "—" in node.get("label", "") else "",
                "concept":       node.get("category", ""),
                "edge_type":     edge_type,
                "edge_label":    node.get("label", ""),
            })

        return results

    def format_context(
        self,
        section:     str,
        source_hint: Optional[str],
        related:     list[dict],
    ) -> str:
        """Mimics NetworkX KG format_context output."""
        if not related:
            return ""

        hint_short = ""
        if source_hint:
            import re
            acr = re.search(r'\(([A-Z]{2,6})\)', source_hint)
            hint_short = acr.group(1) if acr else source_hint.split()[0].upper()

        lines = [f"[Knowledge Graph] Section {section} {hint_short} related nodes:"]
        for node in related[:8]:
            sec   = node.get("section", "")
            act   = node.get("act_key", "")
            etype = node.get("edge_type", "")
            label = (node.get("section_title") or node.get("label", ""))[:60]
            desc  = f" — {label}" if label else ""
            lines.append(f"  • Section {sec} {act}{desc} [{etype}]")

        return "\n".join(lines)


# Module-level singleton of the adapter
_flat_kg_singleton = _FlatGraphAdapter()


def get_kg() -> _FlatGraphAdapter:
    """
    Drop-in replacement for the old get_kg() that returned the NetworkX KG.
    Returns the flat-JSON adapter which implements the same interface.
    """
    if not _flat_kg_singleton._built:
        _flat_kg_singleton.build()
    return _flat_kg_singleton