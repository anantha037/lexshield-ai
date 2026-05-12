"""
LexShield AI — Legal Knowledge Graph
=======================================
NetworkX-based knowledge graph over Indian statutes.

Node types:
  statute  — an Act (e.g. Indian Penal Code 1860)
  section  — a section within an Act
  concept  — legal concept (murder, theft, cheating, etc.)

Edge types:
  belongs_to — section → statute
  related    — section → section (same act, definitional chain)
  paired_act — section → section (legacy ↔ current act equivalents)
  relates_to — section → concept

Population:
  Auto:   from data/processed/chunks.json (section + statute nodes)
  Manual: hardcoded key relationships for major IPC/BNS/NI sections

Usage:
  from rag.knowledge_graph import get_kg
  kg      = get_kg()
  related = kg.query_related_sections("420", source_hint="Indian Penal Code")
  context = kg.format_context("420", "Indian Penal Code", related)
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import networkx as nx


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════════

_ACRONYM_RE = re.compile(r'\(([A-Z]{2,6})\)')

_FALLBACK_MAP = {
    "indian penal code":                  "IPC",
    "bharatiya nyaya sanhita":            "BNS",
    "code of criminal procedure":         "CrPC",
    "bharatiya nagarik suraksha sanhita": "BNSS",
    "indian evidence act":                "IEA",
    "bharatiya sakshya adhiniyam":        "BSA",
    "negotiable instruments act":         "NI",
    "protection of children":             "POCSO",
    "consumer protection act":            "CPA",
    "information technology act":         "ITA",
    "prevention of corruption":           "PCA",
    "prevention of money laundering":     "PMLA",
    "narcotic drugs":                     "NDPS",
    "unlawful activities":                "UAPA",
    "right to information":               "RTI",
    "motor vehicles act":                 "MVA",
    "transfer of property act":           "TPA",
    "indian contract act":                "ICA",
    "code of civil procedure":            "CPC",
    "companies act":                      "CA",
    "real estate":                        "RERA",
    "insolvency and bankruptcy":          "IBC",
    "foreign exchange":                   "FEMA",
    "securities and exchange":            "SEBI",
    "sexual harassment":                  "POSH",
    "domestic violence":                  "DVPA",
    "hindu marriage act":                 "HMA",
    "special marriage act":               "SMA",
    "goods and services tax":             "GST",
}


def normalize_source(source: str) -> str:
    """Extract short acronym key from a source string."""
    m = _ACRONYM_RE.search(source)
    if m:
        return m.group(1)
    src_lower = source.lower()
    for keyword, key in _FALLBACK_MAP.items():
        if keyword in src_lower:
            return key
    # Last resort: initials of first 3 words
    words = source.split()
    return "".join(w[0] for w in words[:3] if w).upper()


def _section_id(act_key: str, section: str) -> str:
    return f"section:{act_key}:{section}"


def _statute_id(source: str) -> str:
    return f"statute:{source}"


def _concept_id(name: str) -> str:
    return f"concept:{name}"


# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL EDGES
# (from_node_id, to_node_id, edge_type, label)
# ═══════════════════════════════════════════════════════════════════════════════

_MANUAL_EDGES: list[tuple[str, str, str, str]] = [

    # ── IPC ↔ BNS paired sections ─────────────────────────────────────────────
    ("section:IPC:34",    "section:BNS:3",    "paired_act", "Common intention"),
    ("section:IPC:107",   "section:BNS:45",   "paired_act", "Abetment"),
    ("section:IPC:120B",  "section:BNS:61",   "paired_act", "Criminal conspiracy"),
    ("section:IPC:299",   "section:BNS:99",   "paired_act", "Culpable homicide definition"),
    ("section:IPC:300",   "section:BNS:100",  "paired_act", "Murder definition"),
    ("section:IPC:302",   "section:BNS:101",  "paired_act", "Murder punishment"),
    ("section:IPC:304",   "section:BNS:105",  "paired_act", "Culpable homicide punishment"),
    ("section:IPC:307",   "section:BNS:109",  "paired_act", "Attempt to murder"),
    ("section:IPC:354",   "section:BNS:74",   "paired_act", "Assault on woman"),
    ("section:IPC:376",   "section:BNS:63",   "paired_act", "Rape"),
    ("section:IPC:378",   "section:BNS:302",  "paired_act", "Theft definition"),
    ("section:IPC:379",   "section:BNS:303",  "paired_act", "Theft punishment"),
    ("section:IPC:380",   "section:BNS:305",  "paired_act", "Theft in dwelling"),
    ("section:IPC:390",   "section:BNS:308",  "paired_act", "Robbery definition"),
    ("section:IPC:392",   "section:BNS:309",  "paired_act", "Robbery punishment"),
    ("section:IPC:395",   "section:BNS:310",  "paired_act", "Dacoity"),
    ("section:IPC:406",   "section:BNS:316",  "paired_act", "Criminal breach of trust"),
    ("section:IPC:415",   "section:BNS:318",  "paired_act", "Cheating definition"),
    ("section:IPC:417",   "section:BNS:318",  "related",    "Cheating punishment"),
    ("section:IPC:420",   "section:BNS:318",  "paired_act", "Cheating aggravated"),
    ("section:IPC:498A",  "section:BNS:85",   "paired_act", "Cruelty by husband"),
    ("section:IPC:503",   "section:BNS:351",  "paired_act", "Criminal intimidation"),
    ("section:IPC:506",   "section:BNS:351",  "related",    "Criminal intimidation punishment"),

    # ── IPC internal chains ───────────────────────────────────────────────────
    ("section:IPC:299",   "section:IPC:300",  "related", "Culpable homicide→murder boundary"),
    ("section:IPC:300",   "section:IPC:302",  "related", "Murder definition→punishment"),
    ("section:IPC:307",   "section:IPC:302",  "related", "Attempt→completed offence"),
    ("section:IPC:378",   "section:IPC:379",  "related", "Theft definition→punishment"),
    ("section:IPC:378",   "section:IPC:380",  "related", "Theft→aggravated forms"),
    ("section:IPC:379",   "section:IPC:380",  "related", "Theft variants"),
    ("section:IPC:390",   "section:IPC:392",  "related", "Robbery definition→punishment"),
    ("section:IPC:415",   "section:IPC:417",  "related", "Cheating definition→punishment"),
    ("section:IPC:415",   "section:IPC:420",  "related", "Cheating→aggravated cheating"),
    ("section:IPC:417",   "section:IPC:420",  "related", "Simple→aggravated cheating"),
    ("section:IPC:403",   "section:IPC:406",  "related", "Misappropriation→breach of trust"),

    # ── BNS internal chains ───────────────────────────────────────────────────
    ("section:BNS:99",    "section:BNS:100",  "related", "Culpable homicide→murder"),
    ("section:BNS:100",   "section:BNS:101",  "related", "Murder definition→punishment"),
    ("section:BNS:302",   "section:BNS:303",  "related", "Theft definition→punishment"),

    # ── CrPC ↔ BNSS ──────────────────────────────────────────────────────────
    ("section:CrPC:154",  "section:BNSS:173", "paired_act", "FIR registration"),
    ("section:CrPC:161",  "section:BNSS:180", "paired_act", "Police examination of witnesses"),
    ("section:CrPC:437",  "section:BNSS:480", "paired_act", "Bail in non-bailable offences"),
    ("section:CrPC:438",  "section:BNSS:482", "paired_act", "Anticipatory bail"),
    ("section:CrPC:482",  "section:BNSS:528", "paired_act", "High Court inherent powers"),

    # ── NI Act internal ───────────────────────────────────────────────────────
    ("section:NI:138",    "section:NI:139",   "related", "Cheque bounce→presumption of liability"),
    ("section:NI:138",    "section:NI:141",   "related", "Cheque bounce→company/director liability"),
    ("section:NI:138",    "section:NI:142",   "related", "Cheque bounce→cognizance conditions"),

    # ── Concept links from sections ───────────────────────────────────────────
    ("section:IPC:302",   "concept:murder",           "relates_to", ""),
    ("section:IPC:300",   "concept:murder",           "relates_to", ""),
    ("section:BNS:101",   "concept:murder",           "relates_to", ""),
    ("section:BNS:100",   "concept:murder",           "relates_to", ""),
    ("section:IPC:420",   "concept:cheating",         "relates_to", ""),
    ("section:IPC:415",   "concept:cheating",         "relates_to", ""),
    ("section:IPC:417",   "concept:cheating",         "relates_to", ""),
    ("section:BNS:318",   "concept:cheating",         "relates_to", ""),
    ("section:IPC:379",   "concept:theft",            "relates_to", ""),
    ("section:IPC:378",   "concept:theft",            "relates_to", ""),
    ("section:BNS:303",   "concept:theft",            "relates_to", ""),
    ("section:IPC:376",   "concept:rape",             "relates_to", ""),
    ("section:BNS:63",    "concept:rape",             "relates_to", ""),
    ("section:IPC:498A",  "concept:cruelty",          "relates_to", ""),
    ("section:BNS:85",    "concept:cruelty",          "relates_to", ""),
    ("section:IPC:406",   "concept:breach_of_trust",  "relates_to", ""),
    ("section:BNS:316",   "concept:breach_of_trust",  "relates_to", ""),
    ("section:IPC:392",   "concept:robbery",          "relates_to", ""),
    ("section:BNS:309",   "concept:robbery",          "relates_to", ""),
    ("section:IPC:395",   "concept:dacoity",          "relates_to", ""),
    ("section:BNS:310",   "concept:dacoity",          "relates_to", ""),
    ("section:NI:138",    "concept:cheque_bounce",    "relates_to", ""),
    ("section:NI:139",    "concept:cheque_bounce",    "relates_to", ""),
    ("section:NI:141",    "concept:cheque_bounce",    "relates_to", ""),
    ("section:CrPC:154",  "concept:fir",              "relates_to", ""),
    ("section:BNSS:173",  "concept:fir",              "relates_to", ""),
    ("section:CrPC:437",  "concept:bail",             "relates_to", ""),
    ("section:CrPC:438",  "concept:bail",             "relates_to", ""),
    ("section:BNSS:480",  "concept:bail",             "relates_to", ""),
    ("section:BNSS:482",  "concept:bail",             "relates_to", ""),
    ("section:IPC:307",   "concept:attempt_to_murder","relates_to", ""),
    ("section:BNS:109",   "concept:attempt_to_murder","relates_to", ""),
    ("section:IPC:354",   "concept:assault_on_woman", "relates_to", ""),
    ("section:BNS:74",    "concept:assault_on_woman", "relates_to", ""),

    # ── Concept–concept links ─────────────────────────────────────────────────
    ("concept:cheating",  "concept:fraud",            "related", ""),
    ("concept:murder",    "concept:culpable_homicide","related", ""),
    ("concept:theft",     "concept:robbery",          "related", ""),
    ("concept:robbery",   "concept:dacoity",          "related", ""),
]

_CONCEPT_LABELS: dict[str, str] = {
    "murder":            "Murder",
    "cheating":          "Cheating / Fraud",
    "theft":             "Theft",
    "rape":              "Rape / Sexual Assault",
    "cruelty":           "Cruelty by Husband / Relatives",
    "breach_of_trust":   "Criminal Breach of Trust",
    "robbery":           "Robbery",
    "dacoity":           "Dacoity",
    "cheque_bounce":     "Cheque Bounce",
    "fir":               "FIR Registration",
    "bail":              "Bail",
    "fraud":             "Fraud",
    "culpable_homicide": "Culpable Homicide",
    "attempt_to_murder": "Attempt to Murder",
    "assault_on_woman":  "Assault on Woman",
}


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class LegalKnowledgeGraph:

    def __init__(self):
        self.graph  = nx.Graph()
        self._built = False

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks_path: str = "data/processed/chunks.json") -> None:
        """Build graph from chunks.json + manual edges. Call once at startup."""
        print("[KnowledgeGraph] Building legal knowledge graph...")
        t0 = time.time()

        self._auto_populate(chunks_path)
        self._add_manual_edges()
        self._built = True

        print(
            f"[KnowledgeGraph] Ready: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges ({time.time() - t0:.2f}s)"
        )

    def _auto_populate(self, chunks_path: str) -> None:
        """Create section + statute nodes from chunks.json (chunk_type='section' only)."""
        if not Path(chunks_path).exists():
            print(f"[KnowledgeGraph] Warning: {chunks_path} not found — skipping auto-populate")
            return

        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)

        seen_statutes: set[str] = set()

        for chunk in chunks:
            if chunk.get("chunk_type") != "section":
                continue

            source  = chunk.get("source", "").strip()
            section = chunk.get("section", "").strip()
            if not source or not section:
                continue

            act_key    = normalize_source(source)
            stat_id    = _statute_id(source)
            sect_id    = _section_id(act_key, section)

            # Statute node (once per source)
            if stat_id not in seen_statutes:
                self.graph.add_node(
                    stat_id,
                    node_type = "statute",
                    label     = source,
                    act_key   = act_key,
                )
                seen_statutes.add(stat_id)

            # Section node
            if not self.graph.has_node(sect_id):
                self.graph.add_node(
                    sect_id,
                    node_type     = "section",
                    section       = section,
                    act_key       = act_key,
                    source        = source,
                    section_title = chunk.get("section_title", ""),
                    category      = chunk.get("category", ""),
                    era           = chunk.get("era", ""),
                    label         = f"Section {section} {act_key}",
                )

            # belongs_to edge
            if not self.graph.has_edge(sect_id, stat_id):
                self.graph.add_edge(sect_id, stat_id, edge_type="belongs_to")

    def _add_manual_edges(self) -> None:
        """Add hardcoded relationships and concept nodes."""
        for src, dst, edge_type, label in _MANUAL_EDGES:
            for node_id in (src, dst):
                if not self.graph.has_node(node_id):
                    parts     = node_id.split(":", 2)
                    node_type = parts[0]

                    if node_type == "section" and len(parts) == 3:
                        self.graph.add_node(
                            node_id,
                            node_type     = "section",
                            act_key       = parts[1],
                            section       = parts[2],
                            section_title = "",
                            source        = "",
                            label         = f"Section {parts[2]} {parts[1]}",
                        )
                    elif node_type == "concept" and len(parts) >= 2:
                        cname = parts[1]
                        self.graph.add_node(
                            node_id,
                            node_type = "concept",
                            name      = cname,
                            label     = _CONCEPT_LABELS.get(cname, cname.replace("_", " ").title()),
                        )

            if not self.graph.has_edge(src, dst):
                self.graph.add_edge(src, dst, edge_type=edge_type, label=label)

    # ── Query ──────────────────────────────────────────────────────────────────

    def query_related_sections(
        self,
        section:     str,
        source_hint: Optional[str] = None,
        hops:        int           = 2,
    ) -> list[dict]:
        """
        Return nodes reachable within `hops` from the given section.
        Filters to section and concept nodes only (excludes statute nodes).

        Args:
            section:     section number string e.g. "420"
            source_hint: partial source name e.g. "Indian Penal Code"
            hops:        traversal depth (default 2)

        Returns:
            List of node dicts sorted by edge_type priority.
        """
        if not self._built:
            return []

        node_id = self._resolve_node(section, source_hint)
        if node_id is None or not self.graph.has_node(node_id):
            return []

        try:
            subgraph = nx.ego_graph(self.graph, node_id, radius=hops, undirected=True)
        except Exception:
            return []

        results = []
        for nid, attrs in subgraph.nodes(data=True):
            if nid == node_id:
                continue
            ntype = attrs.get("node_type", "")
            if ntype not in ("section", "concept"):
                continue

            edge_data = self.graph.get_edge_data(node_id, nid) or {}
            results.append({
                "node_id":       nid,
                "node_type":     ntype,
                "label":         attrs.get("label", nid),
                "section":       attrs.get("section", ""),
                "act_key":       attrs.get("act_key", ""),
                "section_title": attrs.get("section_title", ""),
                "concept":       attrs.get("name", ""),
                "edge_type":     edge_data.get("edge_type", ""),
                "edge_label":    edge_data.get("label", ""),
            })

        _order = {"paired_act": 0, "related": 1, "relates_to": 2, "belongs_to": 9}
        results.sort(key=lambda x: (_order.get(x["edge_type"], 5), x["label"]))
        return results

    def _resolve_node(self, section: str, source_hint: Optional[str]) -> Optional[str]:
        """Find the best matching node_id for a section + optional source hint."""
        section = section.strip()

        if source_hint:
            act_key = normalize_source(source_hint)
            nid     = _section_id(act_key, section)
            if self.graph.has_node(nid):
                return nid

        candidates = [
            nid for nid, attrs in self.graph.nodes(data=True)
            if attrs.get("node_type") == "section"
            and attrs.get("section") == section
        ]

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            for key in ["IPC", "BNS", "CrPC", "BNSS", "NI"]:
                pref = _section_id(key, section)
                if pref in candidates:
                    return pref
            return candidates[0]

        return None

    def format_context(
        self,
        section:     str,
        source_hint: Optional[str],
        related:     list[dict],
    ) -> str:
        """
        Format related nodes as a compact string for prompt injection.

        Example:
          [Knowledge Graph] Section 420 IPC related nodes:
            • Section 415 IPC — Cheating definition→punishment [related]
            • Section 318 BNS — Cheating aggravated [paired_act]
            • Concept: Cheating / Fraud [relates_to]
        """
        if not related:
            return ""

        act_hint = normalize_source(source_hint) if source_hint else "?"
        lines    = [f"[Knowledge Graph] Section {section} {act_hint} related nodes:"]

        for node in related[:8]:
            ntype = node["node_type"]
            label = node["edge_label"] or node["section_title"] or ""

            if ntype == "section":
                desc = f" — {label}" if label else ""
                lines.append(
                    f"  • Section {node['section']} {node['act_key']}{desc} [{node['edge_type']}]"
                )
            elif ntype == "concept":
                lines.append(f"  • Concept: {node['label']} [{node['edge_type']}]")

        return "\n".join(lines)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._built:
            return {"built": False}

        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}

        for _, attrs in self.graph.nodes(data=True):
            t = attrs.get("node_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        for _, _, attrs in self.graph.edges(data=True):
            t = attrs.get("edge_type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1

        return {
            "built":      True,
            "nodes":      self.graph.number_of_nodes(),
            "edges":      self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
legal_kg = LegalKnowledgeGraph()


def get_kg() -> LegalKnowledgeGraph:
    """Return the singleton KG, building it if not yet built."""
    if not legal_kg._built:
        legal_kg.build()
    return legal_kg