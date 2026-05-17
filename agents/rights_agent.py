"""
LexShield AI — Rights Agent
=============================
Serves structured Know Your Rights guides from data/rights_guide.json,
optionally enriched with live RAG pipeline context for deeper coverage.

5 categories:
  tenant    — Transfer of Property Act, State Rent Control Acts
  employee  — Payment of Wages Act, Industrial Disputes Act, EPF Act
  consumer  — Consumer Protection Act 2019
  women     — PWDVA 2005, Dowry Prohibition Act, Hindu Succession Act
  bail      — Constitution Art.22, BNSS/CrPC — arrested person's rights

Public API:
  get_rights(category)                          → dict
  format_rights_response(rights_dict)           → str
  get_rights_with_rag_enrichment(category, rag) → dict
  get_all_categories()                          → list[dict]  (for UI tab)
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT      = Path(__file__).resolve().parent.parent
_RIGHTS_GUIDE_PATH = _PROJECT_ROOT / "data" / "rights_guide.json"

VALID_CATEGORIES = ["tenant", "employee", "consumer", "women", "bail"]

_CATEGORY_DISPLAY: dict[str, str] = {
    "tenant":   "Tenant Rights",
    "employee": "Employee Rights",
    "consumer": "Consumer Rights",
    "women":    "Women's Legal Rights",
    "bail":     "Rights of Arrested Person / Bail Rights",
}

_CATEGORY_ICONS: dict[str, str] = {
    "tenant":   "🏠",
    "employee": "👷",
    "consumer": "🛒",
    "women":    "⚖️",
    "bail":     "🔒",
}

# ── Query used to enrich each category from RAG pipeline ─────────────────────
_CATEGORY_RAG_QUERIES: dict[str, str] = {
    "tenant":   "tenant rights illegal eviction rent control India applicable sections",
    "employee": "employee rights salary termination gratuity provident fund India law",
    "consumer": "consumer rights defective goods refund Consumer Protection Act 2019 India",
    "women":    "women rights domestic violence dowry protection India PWDVA applicable sections",
    "bail":     "bail rights arrested person BNSS CrPC default bail 60 90 days India",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_rights_guide() -> dict:
    """Load and cache rights_guide.json. Cached after first load."""
    if not _RIGHTS_GUIDE_PATH.exists():
        raise FileNotFoundError(
            f"rights_guide.json not found at {_RIGHTS_GUIDE_PATH}. "
            "Ensure data/rights_guide.json is present in the project root."
        )
    with open(_RIGHTS_GUIDE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"[RightsAgent] Loaded rights guide: {list(data.keys())}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: GET RIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def get_rights(category: str) -> dict:
    """
    Return the rights guide for a given category.

    Args:
        category: One of "tenant", "employee", "consumer", "women", "bail".
                  Case-insensitive. Partial matches supported:
                  "worker" → "employee", "arrest" → "bail", "woman" → "women".

    Returns:
        Full category dict from rights_guide.json, or an error dict with
        valid category list if category is unrecognised.

    Example:
        rights = get_rights("tenant")
        # {"title": "Tenant Rights in India", "applicable_acts": [...], "rights": [...]}
    """
    normalised = _normalise_category(category)
    if normalised is None:
        return {
            "error": True,
            "message": (
                f"Category '{category}' not found in rights guide. "
                f"Valid categories: {', '.join(VALID_CATEGORIES)}"
            ),
            "valid_categories": [
                {
                    "key":         cat,
                    "display":     _CATEGORY_DISPLAY[cat],
                    "icon":        _CATEGORY_ICONS[cat],
                }
                for cat in VALID_CATEGORIES
            ],
        }

    try:
        guide = _load_rights_guide()
        return guide.get(normalised, {})
    except FileNotFoundError as e:
        logger.error(f"[RightsAgent] {e}")
        return {"error": True, "message": str(e)}
    except Exception as e:
        logger.error(f"[RightsAgent] Failed to load rights for '{normalised}': {e}")
        return {"error": True, "message": f"Failed to load rights guide: {e}"}


def _normalise_category(raw: str) -> str | None:
    """
    Resolve user input to a valid category key.

    Handles:
      - Exact match: "tenant" → "tenant"
      - Alias match: "worker" → "employee", "arrested" → "bail"
      - Partial match: "employ" → "employee"
      - Case-insensitive: "TENANT" → "tenant"
    """
    if not raw:
        return None

    raw_lower = raw.strip().lower()

    # Exact match
    if raw_lower in VALID_CATEGORIES:
        return raw_lower

    # Alias map
    _ALIASES: dict[str, str] = {
        "workers":     "employee",
        "worker":      "employee",
        "employees":   "employee",
        "labour":      "employee",
        "labor":       "employee",
        "tenants":     "tenant",
        "renter":      "tenant",
        "renters":     "tenant",
        "woman":       "women",
        "wife":        "women",
        "domestic":    "women",
        "arrested":    "bail",
        "arrest":      "bail",
        "detention":   "bail",
        "detainee":    "bail",
        "accused":     "bail",
        "prison":      "bail",
        "consumers":   "consumer",
        "buyer":       "consumer",
        "customer":    "customer",
    }
    if raw_lower in _ALIASES:
        return _ALIASES[raw_lower]

    # Partial match (prefix)
    for cat in VALID_CATEGORIES:
        if cat.startswith(raw_lower) or raw_lower.startswith(cat[:4]):
            return cat

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_rights_response(rights_dict: dict) -> str:
    """
    Format a rights category dict into a structured, readable markdown string.

    Output structure:
      🏠 Tenant Rights in India
      ═══════════════════════════════
      Applicable Laws:
        • Transfer of Property Act 1882
        • ...

      YOUR RIGHTS:
      ──────────────────────────────
      1. Right to written rent agreement
         Section: Transfer of Property Act 1882
         ...
         What to do: ...

    Args:
        rights_dict: Dict from get_rights(). If it contains "error": True,
                     returns a user-friendly error message instead.

    Returns:
        Formatted markdown string for display in Streamlit or API response.
    """
    if not rights_dict:
        return "No rights information available."

    if rights_dict.get("error"):
        msg = rights_dict.get("message", "Category not found.")
        cats = rights_dict.get("valid_categories", [])
        lines = [f"⚠️  {msg}", ""]
        if cats:
            lines.append("**Available categories:**")
            for c in cats:
                lines.append(f"  {c['icon']}  **{c['key']}** — {c['display']}")
        return "\n".join(lines)

    title = rights_dict.get("title", "Legal Rights in India")
    acts  = rights_dict.get("applicable_acts", [])
    rights = rights_dict.get("rights", [])

    # Infer icon from title
    icon = "⚖️"
    for cat, ic in _CATEGORY_ICONS.items():
        if _CATEGORY_DISPLAY.get(cat, "").lower() in title.lower():
            icon = ic
            break

    lines = [
        f"{icon} **{title}**",
        "═" * 55,
        "",
    ]

    if acts:
        lines.append("**Applicable Laws:**")
        for act in acts:
            lines.append(f"  • {act}")
        lines.append("")

    lines.append("─" * 55)
    lines.append(f"**YOUR RIGHTS ({len(rights)} rights protected by law):**")
    lines.append("─" * 55)
    lines.append("")

    for i, r in enumerate(rights, 1):
        right_name   = r.get("right", "")
        section      = r.get("section", "")
        description  = r.get("description", "")
        remedy       = r.get("remedy", "")

        lines.append(f"**{i}. {right_name}**")
        if section:
            lines.append(f"   📖 *Legal basis: {section}*")
        if description:
            lines.append(f"   {description}")
        if remedy:
            lines.append(f"   ✅ **What to do:** {remedy}")
        lines.append("")

    lines.append("─" * 55)
    lines.append(
        "> *This information is based on current Indian law as of 2024. "
        "Laws may vary by state. For specific legal advice, consult a qualified advocate "
        "or contact your District Legal Services Authority (DLSA) for free legal aid.*"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# RAG ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_rights_with_rag_enrichment(category: str, rag_pipeline) -> dict:
    """
    Fetch base rights guide + enrich with top RAG chunks for deeper coverage.

    Enrichment strategy:
      1. Load base rights dict from rights_guide.json
      2. Query RAG pipeline with category-specific legal query
      3. Parse RAG response for any sections/acts not already in the guide
      4. Append found additional context as "Additional Legal Context" to the dict
      5. Return enriched dict (original rights preserved, extra sections appended)

    RAG enrichment is non-blocking — any failure falls back to base rights silently.

    Args:
        category:     Category key ("tenant", "employee", etc.)
        rag_pipeline: rag.pipeline.rag_pipeline singleton

    Returns:
        Enriched rights dict with optional "rag_enrichment" key added.
    """
    base_rights = get_rights(category)

    if base_rights.get("error"):
        return base_rights

    normalised = _normalise_category(category)
    if normalised is None:
        return base_rights

    rag_query = _CATEGORY_RAG_QUERIES.get(normalised, f"rights of {category} in India")

    try:
        rag_result        = rag_pipeline.query(rag_query)
        rag_answer        = rag_result.answer_text if hasattr(rag_result, "answer_text") else str(rag_result)
        sources_consulted = getattr(rag_result, "sources_consulted", 0)

        # Extract existing section references already in the guide to avoid duplication
        existing_sections = set()
        for r in base_rights.get("rights", []):
            existing_sections.add(r.get("section", "").lower())

        # Add RAG context as additional legal context block
        enriched = dict(base_rights)
        enriched["rag_enrichment"] = {
            "additional_context":  rag_answer,
            "sources_consulted":   sources_consulted,
            "query_used":          rag_query,
            "note": (
                "Additional context from LexShield knowledge base — "
                "includes relevant case law, state-specific variations, and procedural details."
            ),
        }

        logger.info(
            f"[RightsAgent] Enriched '{normalised}' rights with RAG "
            f"({sources_consulted} source(s))"
        )
        return enriched

    except Exception as e:
        logger.warning(
            f"[RightsAgent] RAG enrichment failed for '{normalised}' (non-fatal): {e}"
        )
        return base_rights  # Return base rights unchanged — enrichment is optional


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPER: ALL CATEGORIES (for Streamlit Tab)
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_categories() -> list[dict]:
    """
    Return metadata for all 5 rights categories for UI display.
    Used to populate the Know Your Rights tab selector.

    Returns:
        List of dicts with: key, display, icon, num_rights, applicable_acts_count
    """
    try:
        guide = _load_rights_guide()
    except Exception:
        return []

    result = []
    for cat in VALID_CATEGORIES:
        data = guide.get(cat, {})
        result.append({
            "key":                 cat,
            "display":             _CATEGORY_DISPLAY.get(cat, cat.title()),
            "icon":                _CATEGORY_ICONS.get(cat, "⚖️"),
            "num_rights":          len(data.get("rights", [])),
            "applicable_acts":     data.get("applicable_acts", []),
            "applicable_acts_count": len(data.get("applicable_acts", [])),
            "title":               data.get("title", ""),
        })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK-SEARCH: Find rights mentioning a keyword (for RAG post-processing)
# ═══════════════════════════════════════════════════════════════════════════════

def search_rights(keyword: str) -> list[dict]:
    """
    Search all categories for rights entries mentioning a keyword.
    Used internally when NER detects a legal concept in the query.

    Args:
        keyword: Search term (e.g. "gratuity", "bail", "FIR")

    Returns:
        List of matching rights entries with their category added.
    """
    try:
        guide   = _load_rights_guide()
        kw      = keyword.lower().strip()
        matches = []

        for cat, data in guide.items():
            for r in data.get("rights", []):
                searchable = (
                    r.get("right", "").lower() + " " +
                    r.get("description", "").lower() + " " +
                    r.get("section", "").lower()
                )
                if kw in searchable:
                    matches.append({
                        **r,
                        "category":      cat,
                        "category_title": data.get("title", ""),
                    })

        return matches

    except Exception as e:
        logger.warning(f"[RightsAgent] search_rights failed: {e}")
        return []