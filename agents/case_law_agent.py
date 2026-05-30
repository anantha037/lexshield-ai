"""
LexShield AI — Case Law Search Agent
=======================================
Searches Indian Kanoon for real Supreme Court and High Court judgments.
Provides cited, summarized case law to supplement RAG answers — the
feature that most clearly separates LexShield from generic legal chatbots.

API: api.indiankanoon.org (free non-commercial tier)
  Register: https://indiankanoon.org/api/
  .env key: INDIANKANOON_API_KEY

Usage:
  from agents.case_law_agent import search_and_summarize, format_case_law_response
  result   = search_and_summarize("Section 302 IPC murder judgment", groq_client=llm)
  response = format_case_law_response(result)

Optional enrichment flag (default: true):
  ENABLE_CASE_LAW_ENRICHMENT=true   # in .env
  When true, legal_rag_node appends top 2 judgments to every RAG response
  that mentions specific IPC/BNS section numbers.

Architecture note:
  search_cases()          -> raw API call -> list of case dicts
  summarize_case()        -> Groq 2-sentence summary of one case
  search_and_summarize()  -> search + summarize all results -> structured dict
  format_case_law_response() -> dict -> formatted markdown string for state["response"]
"""

import os
import re
import time
import asyncio
import logging
from datetime import datetime
from typing import Optional
import requests
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

_IK_API_KEY          = os.getenv("IK_API_TOKEN", "")
_IK_BASE_URL         = "https://api.indiankanoon.org"
_REQUEST_TIMEOUT     = 10      # seconds — hard cap per API call
_GROQ_DELAY_SECONDS  = 0.3     # rate-limit buffer between Groq summarise calls
ENABLE_CASE_LAW_ENRICHMENT = (
    os.getenv("ENABLE_CASE_LAW_ENRICHMENT", "true").lower() == "true"
)

# ── Court name normalisation map ───────────────────────────────────────────────
# Indian Kanoon's docsource field uses lowercase shortcodes.
# We map them to human-readable names used in Indian legal citations.
_COURT_DISPLAY: dict[str, str] = {
    "supremecourt":      "Supreme Court of India",
    "allahabad":         "Allahabad High Court",
    "bombay":            "Bombay High Court",
    "calcutta":          "Calcutta High Court",
    "delhi":             "Delhi High Court",
    "kerala":            "Kerala High Court",
    "madras":            "Madras High Court",
    "karnataka":         "Karnataka High Court",
    "gujarat":           "Gujarat High Court",
    "rajasthan":         "Rajasthan High Court",
    "punjabharyana":     "Punjab & Haryana High Court",
    "andhra":            "Andhra Pradesh High Court",
    "telangana":         "Telangana High Court",
    "patna":             "Patna High Court",
    "gauhati":           "Gauhati High Court",
    "himachalpradesh":   "Himachal Pradesh High Court",
    "jharkhand":         "Jharkhand High Court",
    "chhattisgarh":      "Chhattisgarh High Court",
    "uttarakhand":       "Uttarakhand High Court",
    "manipur":           "Manipur High Court",
    "meghalaya":         "Meghalaya High Court",
    "orissa":            "Orissa High Court",
    "sikkim":            "Sikkim High Court",
    "tripura":           "Tripura High Court",
    "nclat":             "NCLAT (National Company Law Appellate Tribunal)",
    "nclt":              "NCLT (National Company Law Tribunal)",
    "itat":              "Income Tax Appellate Tribunal",
    "drat":              "Debt Recovery Appellate Tribunal",
    "drt":               "Debt Recovery Tribunal",
    "ncdrc":             "NCDRC (National Consumer Disputes Redressal Commission)",
    "cestat":            "CESTAT (Customs, Excise & Service Tax Appellate Tribunal)",
    "aft":               "Armed Forces Tribunal",
    "cat":               "Central Administrative Tribunal",
}

# ── HTML entity cleanup regex ──────────────────────────────────────────────────
_HTML_TAG_RE    = re.compile(r'<[^>]+>')
_HTML_ENTITY_RE = re.compile(r'&(?:amp|lt|gt|quot|apos|nbsp);')
_ENTITY_MAP     = {"&amp;": "&", "&lt;": "<", "&gt;": ">",
                   "&quot;": '"', "&apos;": "'", "&nbsp;": " "}


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_html(raw: str) -> str:
    """Strip HTML tags and decode common entities from Indian Kanoon response."""
    text = _HTML_TAG_RE.sub("", raw or "")
    text = _HTML_ENTITY_RE.sub(lambda m: _ENTITY_MAP.get(m.group(0), m.group(0)), text)
    return " ".join(text.split())  # collapse whitespace


def _normalize_court(docsource: str) -> str:
    """Convert Indian Kanoon docsource shortcode to human-readable court name."""
    if not docsource:
        return "Indian Court"
    key = re.sub(r'[\s\-_]', '', docsource.lower())
    for partial_key, display_name in _COURT_DISPLAY.items():
        if partial_key in key:
            return display_name
    # Fallback: capitalise raw value
    return re.sub(r'[_\-]', ' ', docsource).title()


def _format_date(raw: str) -> str:
    """
    Normalise date string to human-readable format (e.g. "3 Jan 2022").
    Indian Kanoon returns dates as YYYY-MM-DD.
    Cross-platform safe (no %-d Linux-only format specifier).
    """
    if not raw:
        return "Date unknown"
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            # Zero-strip the day number cross-platform
            formatted = dt.strftime("%d %b %Y").lstrip("0")
            return formatted if formatted else dt.strftime("%d %b %Y")
        except ValueError:
            continue
    return raw.strip()


def _build_citation(title: str, court: str, date: str) -> str:
    """Construct a best-effort citation when Indian Kanoon doesn't provide one."""
    year = ""
    if date and len(date) >= 4:
        # Extract 4-digit year from whatever date format we have
        m = re.search(r'\b(\d{4})\b', date)
        year = m.group(1) if m else ""

    if "Supreme Court" in court:
        return f"{title} ({year})" if year else title
    short_court = court.split("(")[0].strip()  # strip parenthetical acronym
    return f"{title}, {short_court} ({year})" if year else f"{title}, {short_court}"


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def search_cases(query: str, max_results: int = 3) -> list[dict]:
    """
    Search Indian Kanoon for court judgments matching the query.

    Endpoint:
      POST https://api.indiankanoon.org/search/
      Form data: formInput=<query>&pagenum=0
      Header:    Authorization: Token <INDIANKANOON_API_KEY>

    Error handling:
      Timeout, connection errors, HTTP errors -> log warning, return [].
      API returns empty docs array -> return [].
      Missing INDIANKANOON_API_KEY -> log warning, return [].

    Args:
        query:       English-language legal search query.
                     Best results with: section number + act name + legal issue
                     e.g. "Section 138 NI Act cheque bounce conviction"
        max_results: Maximum cases to return (default 3, max 10 meaningful)

    Returns:
        List of case dicts, each with:
          title    (str) — full case title, e.g. "M.S. Dhoni v. State of Jharkhand"
          court    (str) — court name, e.g. "Supreme Court of India"
          date     (str) — judgment date, e.g. "3 Jan 2022"
          citation (str) — citation string if available
          headline (str) — excerpt/snippet from judgment (≤300 chars)
          doc_id   (str) — Indian Kanoon document ID (tid)
          url      (str) — full URL to judgment on indiankanoon.org
    """
    if not _IK_API_KEY:
        logger.warning(
            "[CaseLawAgent] INDIANKANOON_API_KEY not set in .env — "
            "case law search disabled. Register at: https://indiankanoon.org/api/"
        )
        return []

    try:
        resp = requests.post(
            f"{_IK_BASE_URL}/search/",
            data    = {"formInput": query, "pagenum": 0},
            headers = {
                "Authorization": f"Token {_IK_API_KEY}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            timeout = _REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.Timeout:
        logger.warning(
            f"[CaseLawAgent] Timeout ({_REQUEST_TIMEOUT}s) searching Indian Kanoon "
            f"for: {query[:60]!r}"
        )
        return []
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"[CaseLawAgent] Connection error — Indian Kanoon unreachable. Exception: {e}")
        return []
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.warning(f"[CaseLawAgent] HTTP {status} error from Indian Kanoon: {e}")
        return []
    except ValueError:
        logger.warning("[CaseLawAgent] Non-JSON response from Indian Kanoon")
        return []
    except Exception as e:
        logger.warning(f"[CaseLawAgent] Unexpected error: {e}")
        return []

    docs = data.get("docs", [])
    if not docs:
        logger.info(f"[CaseLawAgent] No results for: {query[:60]!r}")
        return []

    results: list[dict] = []
    for doc in docs[:max_results]:
        tid = str(doc.get("tid", "")).strip()
        if not tid:
            continue  # malformed entry — skip

        title    = _clean_html(doc.get("title", "Untitled"))
        court    = _normalize_court(doc.get("docsource", ""))
        date     = _format_date(doc.get("publishdate", ""))
        citation = doc.get("citation", "").strip() or _build_citation(title, court, date)
        headline = _clean_html(doc.get("headline", ""))[:300]
        url      = f"https://indiankanoon.org/doc/{tid}/"

        results.append({
            "title":    title,
            "court":    court,
            "date":     date,
            "citation": citation,
            "headline": headline,
            "doc_id":   tid,
            "url":      url,
        })

    print(f"[CaseLawAgent] {len(results)} case(s) found for: {query[:55]!r}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARISE
# ═══════════════════════════════════════════════════════════════════════════════

_SUMMARISE_SYSTEM = (
    "You are a senior Indian advocate preparing a legal precedent digest. "
    "Write precise, factual, authoritative case summaries. "
    "Never speculate beyond the information provided. "
    "Use formal legal English. Two sentences only — no more, no less."
)


async def summarize_case(case: dict, groq_client) -> str:
    """
    Generate a precise 2-sentence precedent summary of an Indian court case.

    Sentence 1 — The holding:
      What the legal dispute was about + what the court decided/held.
    Sentence 2 — The precedent:
      What legal principle this judgment established and why it matters
      in Indian law (which future courts now follow this rule).

    Args:
        case:        Case dict from search_cases()
        groq_client: rag.llm.llm singleton (Groq LLaMA 3.3 70B)

    Returns:
        2-sentence summary string. Falls back to a minimal description on error.
    """
    title    = case.get("title",    "Unknown case")
    court    = case.get("court",    "Indian Court")
    date     = case.get("date",     "")
    citation = case.get("citation", "")
    headline = case.get("headline", "")

    prompt = (
        f"Write a 2-sentence precedent summary for the following Indian court judgment.\n\n"
        f"Case title:  {title}\n"
        f"Court:       {court}\n"
        f"Date:        {date}\n"
        f"Citation:    {citation}\n"
        f"Excerpt:     {headline or '(no excerpt available)'}\n\n"
        f"FORMAT — exactly 2 sentences:\n"
        f"Sentence 1: Begin with the case name or 'The {court}'. "
        f"State precisely (a) what the legal dispute was about — parties, "
        f"legal issue, applicable law — and (b) what the court held or decided.\n"
        f"Sentence 2: Begin with 'This judgment established...' or "
        f"'This ruling is significant because...'. "
        f"State the legal precedent or principle the judgment created "
        f"and its ongoing importance in Indian courts.\n\n"
        f"Write only the 2 sentences. No headings. No bullet points. No preamble."
    )

    try:
        result = await asyncio.to_thread(
            groq_client.generate,
            prompt        = prompt,
            system_prompt = _SUMMARISE_SYSTEM,
            max_tokens    = 220,
            temperature   = 0.1,
        )
        result = result.strip()

        if not result:
            raise ValueError("Empty response from LLM")

        return result

    except Exception as e:
        logger.warning(f"[CaseLawAgent] summarize_case failed for {title[:50]!r}: {e}")
        # Minimal deterministic fallback — never returns empty string
        return (
            f"The {court} decided {title} on {date}. "
            f"This judgment is available on Indian Kanoon for reference."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED: SEARCH + SUMMARISE
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_case_results(results: list[dict]) -> list[dict]:
    valid_results = []
    for i, item in enumerate(results):
        case = item.get("case", {})
        summary = item.get("summary", "")
        
        reasons = []
        if not case.get("title"): reasons.append("missing title")
        if not case.get("citation"): reasons.append("missing citation")
        if not case.get("court"): reasons.append("missing court")
        if not case.get("date"): reasons.append("missing date")
        if not case.get("url"): 
            reasons.append("missing url")
        elif not str(case.get("url")).startswith("https://"):
            reasons.append("invalid url format")
            
        if not summary or len(summary) < 30:
            reasons.append(f"summary too short ({len(summary) if summary else 0} chars)")
            
        if reasons:
            print(f"[DEBUG CaseLaw] Case {i} failed. reasons={reasons}")
            print(f"  title: {case.get('title')}")
            print(f"  citation: {case.get('citation')}")
            print(f"  court: {case.get('court')}")
            print(f"  date: {case.get('date')}")
            print(f"  url: {case.get('url')}")
            print(f"  summary len: {len(summary) if summary else 0}")
            logger.warning(f"[CaseLawAgent] Case {i} failed validation: {', '.join(reasons)}")
        else:
            valid_results.append(item)
            
    return valid_results


@traceable(name="case_law.search_and_summarize", run_type="chain")
async def search_and_summarize(
    query:       str,
    groq_client,
    max_results: int = 3,
) -> dict:
    """
    End-to-end case law search + summarisation pipeline.

    Calls search_cases() -> summarize_case() for each result concurrently.

    Args:
        query:       Legal search query (English)
        groq_client: Groq LLM client
        max_results: Max cases to fetch and summarise (default 3)

    Returns:
        {
          "query":       str,                           # original query
          "results":     [{"case": dict, "summary": str}],  # enriched cases
          "total_found": int,                           # number of results
        }

    Example:
        result = await search_and_summarize(
            "Section 138 NI Act cheque bounce conviction precedent",
            groq_client = llm,
        )
        # result["results"][0]["case"]["title"] -> "MMTC Ltd. v. Medchi Chemicals..."
        # result["results"][0]["summary"]       -> "The Supreme Court held that..."
    """
    cases = search_cases(query, max_results=max_results)

    enriched: list[dict] = []
    if cases:
        summaries = await asyncio.gather(
            *[summarize_case(case, groq_client) for case in cases],
            return_exceptions=True
        )

        for i, (case, summary) in enumerate(zip(cases, summaries)):
            if isinstance(summary, Exception):
                logger.error(f"[CaseLawAgent] Summarization failed for case {i}: {summary}")
                summary_text = (
                    f"The {case.get('court', 'Indian Court')} decided {case.get('title', 'Unknown case')} on {case.get('date', '')}. "
                    f"This judgment is available on Indian Kanoon for reference."
                )
            else:
                summary_text = summary
                
            enriched.append({"case": case, "summary": summary_text})
            
        # Validate structural constraints
        enriched = _validate_case_results(enriched)

    rt = get_current_run_tree()
    if rt:
        rt.add_metadata({
            "cases_found": len(enriched),
            "enriched_query": query
        })

    return {
        "query":       query,
        "results":     enriched,
        "total_found": len(enriched),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_case_law_response(search_result: dict) -> str:
    """
    Format search_and_summarize() output into a markdown-compatible string
    for state["response"] in the LangGraph case_law_node.

    Includes: case title, court, date, citation, 2-sentence summary, clickable URL.
    Ends with a source disclaimer (standard legal information hygiene).

    Args:
        search_result: dict from search_and_summarize()

    Returns:
        Formatted string ready to display in the LexShield frontend.
    """
    query   = search_result.get("query", "")
    results = search_result.get("results", [])

    if not results:
        return (
            f"**No judgments found** on Indian Kanoon for: *{query}*\n\n"
            "Possible reasons:\n"
            "- The `INDIANKANOON_API_KEY` is not configured in `.env`\n"
            "- The search query is too broad or uses non-standard terminology\n"
            "- No matching judgments exist in the Indian Kanoon database\n\n"
            "**Suggested approach:** Try a more specific query using the section number, "
            "act name, and key legal issue. "
            "Example: *\"Section 302 IPC murder culpable homicide not amounting to murder\"*"
        )

    lines = [
        f"## 📋 Case Law Search Results",
        f"",
        f"**Query:** {query}",
        f"**Source:** Indian Kanoon (live judgment database)",
        f"**Results:** {len(results)} judgment(s) found",
        f"",
        "---",
    ]

    for i, item in enumerate(results, 1):
        case    = item["case"]
        summary = item["summary"]
        title   = case.get("title",    "Untitled")
        court   = case.get("court",    "")
        date    = case.get("date",     "")
        cite    = case.get("citation", "")
        url     = case.get("url",      "")

        lines.append(f"")
        lines.append(f"### {i}. {title}")

        meta_parts = [p for p in [court, date] if p]
        if meta_parts:
            lines.append(f"**{'  ·  '.join(meta_parts)}**")

        if cite:
            lines.append(f"*Citation: {cite}*")

        lines.append(f"")
        lines.append(summary)

        if url:
            lines.append(f"")
            lines.append(f"🔗 [Read full judgment on Indian Kanoon]({url})")

        lines.append("")
        lines.append("---")

    lines.append("")
    lines.append(
        "> *Source: Indian Kanoon (indiankanoon.org). "
        "Case summaries are AI-generated for reference only. "
        "Always verify judgments directly from the official source "
        "before relying on them in legal proceedings.*"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# RAG ENRICHMENT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_rag_response_with_case_law(
    rag_answer_text: str,
    ner_sections:    list[str],
    groq_client,
    max_cases_per_section: int = 2,
) -> str:
    """
    Append relevant judgments to a RAG-generated answer.

    Called from legal_rag_node when ENABLE_CASE_LAW_ENRICHMENT=true.
    Only runs when NER found specific IPC/BNS/NI Act section numbers.
    Fetches top 2 judgments per detected section (capped at first 2 sections).

    Args:
        rag_answer_text:       The existing RAG-generated legal answer
        ner_sections:          List of section identifiers from NER,
                               e.g. ["IPC_302", "IPC_304", "BNS_103"]
        groq_client:           Groq LLM client
        max_cases_per_section: Cases to fetch per section (default 2)

    Returns:
        Original answer + "Relevant Judgments:" section appended.
        Returns original answer unchanged if no cases found or on error.
    """
    if not ENABLE_CASE_LAW_ENRICHMENT or not ner_sections or not _IK_API_KEY:
        return rag_answer_text

    # Use at most 2 sections for enrichment (avoid too many API calls on free tier)
    target_sections = ner_sections[:2]
    all_cases: list[dict] = []

    for sec_id in target_sections:
        # Build a targeted search query from section identifier
        # sec_id format: "IPC_302", "BNS_103", "NI_138", or bare "302"
        parts = sec_id.split("_")
        if len(parts) >= 2:
            act_name, section_num = parts[0], parts[1]
            search_query = (
                f"Section {section_num} {act_name} India Supreme Court "
                f"High Court judgment precedent"
            )
        else:
            search_query = f"Section {sec_id} India court judgment"

        try:
            cases = search_cases(search_query, max_results=max_cases_per_section)
            all_cases.extend(cases)
        except Exception as e:
            logger.warning(f"[CaseLawAgent] Enrichment search failed for {sec_id}: {e}")
            continue

    if not all_cases:
        return rag_answer_text

    # De-duplicate by doc_id
    seen: set[str] = set()
    unique_cases: list[dict] = []
    for c in all_cases:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            unique_cases.append(c)

    # Build enrichment section
    judgment_lines = [
        "",
        "---",
        "**Relevant Judgments:**",
        "",
    ]

    async def _batch_summarize():
        return await asyncio.gather(
            *[summarize_case(case, groq_client) for case in unique_cases],
            return_exceptions=True
        )

    summaries = asyncio.run(_batch_summarize())

    for i, (case, summary) in enumerate(zip(unique_cases, summaries)):
        if isinstance(summary, Exception):
            logger.error(f"[CaseLawAgent] Summarization failed for case {i}: {summary}")
            summary_text = f"Judgment by {case.get('court', 'Indian Court')} ({case.get('date', '')})."
        else:
            summary_text = summary

        judgment_lines.append(
            f"• **{case.get('title', 'Untitled')}** — {case.get('court', 'Indian Court')}, {case.get('date', '')}"
        )
        if case.get("citation"):
            judgment_lines.append(f"  *{case['citation']}*")
        judgment_lines.append(f"  {summary_text}")
        if case.get("url"):
            judgment_lines.append(f"  🔗 [{case['url']}]({case['url']})")
        judgment_lines.append("")

    judgment_lines.append(
        "> *Judgments sourced from Indian Kanoon. Verify before use in legal proceedings.*"
    )

    return rag_answer_text + "\n" + "\n".join(judgment_lines)


# ── Public exports ─────────────────────────────────────────────────────────────
__all__ = [
    "search_cases",
    "summarize_case",
    "search_and_summarize",
    "format_case_law_response",
    "enrich_rag_response_with_case_law",
    "ENABLE_CASE_LAW_ENRICHMENT",
]