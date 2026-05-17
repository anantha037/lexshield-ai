"""
LexShield AI — Multilingual Agent
===================================
Auto-detects non-English queries and orchestrates the full pipeline:

  1. Language detection (Unicode script fast-path → langdetect fallback)
  2. Query translation → English (Groq LLaMA 3.3 70B)
  3. Full RAG pipeline on English query
  4. Response translation → source language (Groq)

This module handles AUTOMATIC multilingual detection — it is separate from
translation_agent.py which handles EXPLICIT translation requests
("translate this to Malayalam", "explain in Hindi").

Supported languages:
  Malayalam (ml), Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn),
  Marathi (mr), Bengali (bn), Gujarati (gu), Punjabi (pa), Odia (or), Urdu (ur)

Detection strategy:
  Unicode script range check BEFORE langdetect — reliable for Indian scripts
  even in mixed queries like "Section 302 IPC ൽ ശിക്ഷ എന്ത്?" (mostly English
  characters with a few Malayalam words appended).

Install:
  pip install langdetect
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── langdetect: lightweight, CPU-only, 2 MB ───────────────────────────────────
try:
    from langdetect import detect as _ld_detect
    from langdetect.lang_detect_exception import LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    logger.warning(
        "[MultilingualAgent] langdetect not installed. "
        "Run: pip install langdetect — falling back to Unicode-only detection."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE METADATA
# ═══════════════════════════════════════════════════════════════════════════════

# ISO 639-1 code → human-readable display name
LANG_NAMES: dict[str, str] = {
    "ml": "Malayalam",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
    "en": "English",
}

# Unicode script ranges → ISO 639-1 code
# Listed from most-specific to least-specific.
# Devanagari (hi/mr) is last because it is shared — we default to "hi".
# The count-based approach below handles ambiguity when multiple scripts appear.
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0C80, 0x0CFF, "kn"),   # Kannada
    (0x0980, 0x09FF, "bn"),   # Bengali
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    (0x0A00, 0x0A7F, "pa"),   # Punjabi / Gurmukhi
    (0x0B00, 0x0B7F, "or"),   # Odia
    (0x0600, 0x06FF, "ur"),   # Urdu (Arabic script)
    (0x0900, 0x097F, "hi"),   # Devanagari → Hindi (and Marathi)
]

SUPPORTED_NON_ENGLISH = {code for code in LANG_NAMES if code != "en"}


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Detect the dominant language of an input string.

    Detection pipeline:
      1. Unicode script range count — reliable even for mixed queries
         (e.g. "Section 302 IPC ൽ ശിക്ഷ?" has 6 Malayalam chars → "ml").
         Runs in O(n) string scan, zero network/disk cost.
      2. langdetect — covers Latin-script Indian languages and pure-English
         ambiguity; wrapped in try/except for LangDetectException.
      3. Fallback → "en"

    Args:
        text: Raw user query (any language, any script mix)

    Returns:
        ISO 639-1 code: "ml", "hi", "ta", "te", "kn", "mr",
                        "bn", "gu", "pa", "or", "ur", or "en"
    """
    if not text or not text.strip():
        return "en"

    # ── Step 1: Unicode script count (fast-path) ──────────────────────────────
    script_counts: dict[str, int] = {}
    for char in text:
        cp = ord(char)
        for start, end, iso_code in _SCRIPT_RANGES:
            if start <= cp <= end:
                script_counts[iso_code] = script_counts.get(iso_code, 0) + 1
                break  # each char belongs to at most one script

    if script_counts:
        # Pick the script with the highest character count.
        # Even 2–3 Malayalam chars in an otherwise English query is a strong signal.
        dominant = max(script_counts, key=lambda k: script_counts[k])
        logger.debug(
            f"[MultilingualAgent] Unicode fast-path → {dominant!r} "
            f"(counts={script_counts})"
        )
        return dominant

    # ── Step 2: langdetect for pure-ASCII text ────────────────────────────────
    if _LANGDETECT_AVAILABLE:
        try:
            raw_code = _ld_detect(text)
            # langdetect can return compound codes like "zh-cn" → normalise
            iso_code = raw_code.split("-")[0].lower()
            if iso_code in LANG_NAMES:
                logger.debug(f"[MultilingualAgent] langdetect → {iso_code!r}")
                return iso_code
        except LangDetectException:
            # Raised for text that is too short or ambiguous (e.g. single word)
            pass
        except Exception as e:
            logger.warning(f"[MultilingualAgent] langdetect error: {e}")

    return "en"


def get_language_name(iso_code: str) -> str:
    """Return the display name for a given ISO 639-1 code. Defaults to 'English'."""
    return LANG_NAMES.get(iso_code, "English")


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION — TO ENGLISH
# ═══════════════════════════════════════════════════════════════════════════════

_TRANSLATE_SYSTEM = (
    "You are a professional Indian legal translator with deep expertise in "
    "Malayalam, Hindi, Tamil, Telugu, Kannada, Marathi, Bengali, Gujarati, "
    "Punjabi, Odia, and Urdu legal vocabulary. "
    "Your translations are precise, natural, and preserve the full legal meaning. "
    "You return ONLY the translated text — no preamble, no explanation."
)


def translate_to_english(text: str, source_language: str, groq_client) -> str:
    """
    Translate a non-English legal query to English using Groq LLaMA 3.3 70B.

    Key behaviour:
    - Legal identifiers are NOT translated: IPC, BNS, CrPC, BNSS, NI Act,
      POCSO, PMLA, NDPS, RTI, RERA, section numbers (e.g. Section 302),
      court names (Supreme Court, High Court, District Court, etc.)
    - Only surrounding natural-language text is translated.
    - Temperature = 0.05 → deterministic, minimal hallucination risk.

    Args:
        text:            Input text in source language (may be mixed-script)
        source_language: ISO 639-1 code (e.g. "ml", "hi", "ta")
        groq_client:     rag.llm.llm singleton

    Returns:
        English translation string. Falls back to original text on any error.
    """
    if source_language == "en":
        return text

    lang_name = get_language_name(source_language)

    prompt = (
        f"Translate the following {lang_name} legal query to English.\n\n"
        f"STRICT RULES — follow exactly:\n"
        f"1. Do NOT translate these identifiers — keep them exactly as written:\n"
        f"   • Act names: IPC, BNS, CrPC, BNSS, NI Act, POCSO, PMLA, NDPS, "
        f"RTI Act, RERA, Consumer Protection Act, Payment of Wages Act, SARFAESI, DV Act\n"
        f"   • Section numbers: e.g. Section 302, Section 138, Section 498A, धारा 302\n"
        f"   • Court names: Supreme Court, High Court, District Court, "
        f"Sessions Court, NCDRC, Labour Court, Consumer Forum\n"
        f"   • Legal procedure terms: FIR, bail, chargesheet, cognizable, "
        f"non-cognizable, bailable, warrant, summons, affidavit\n"
        f"2. Translate only the surrounding natural language (the question words, "
        f"verbs, nouns, and context).\n"
        f"3. Return ONLY the translated text. No preamble. No explanation.\n\n"
        f"Query to translate:\n{text}\n\n"
        f"English translation:"
    )

    try:
        result = groq_client.generate(
            prompt        = prompt,
            system_prompt = _TRANSLATE_SYSTEM,
            max_tokens    = 400,
            temperature   = 0.05,
        ).strip()

        result = _strip_llm_preamble(result)

        if not result:
            logger.warning(f"[MultilingualAgent] Empty translation result for {lang_name}→EN")
            return text

        logger.info(f"[MultilingualAgent] {lang_name}→EN: {result[:80]!r}")
        return result

    except Exception as e:
        logger.error(f"[MultilingualAgent] translate_to_english failed ({lang_name}→EN): {e}")
        return text  # safe fallback — raw query will still work for English RAG


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION — FROM ENGLISH
# ═══════════════════════════════════════════════════════════════════════════════

def translate_to_source(text: str, target_language: str, groq_client) -> str:
    """
    Translate an English legal explanation into the target Indian language.

    Legal identifiers (section numbers, act names, court names, procedural
    terms) are kept in English because:
    - They are universally understood across Indian languages in legal contexts.
    - Translating "Supreme Court" to Malayalam "സുപ്രീം കോടതി" can cause
      confusion in formal legal documents.
    - Indian lawyers and citizens read legal notices with English identifiers.

    Args:
        text:            English legal explanation from RAG pipeline
        target_language: ISO 639-1 code (e.g. "ml", "hi")
        groq_client:     rag.llm.llm singleton

    Returns:
        Translated string in target language. Falls back to English on error.
    """
    if target_language == "en":
        return text

    lang_name = get_language_name(target_language)

    prompt = (
        f"Translate the following English legal explanation to {lang_name}.\n\n"
        f"STRICT RULES — follow exactly:\n"
        f"1. Keep ALL of the following in English (do NOT translate them):\n"
        f"   • Section numbers: Section 302, Section 138, Section 498A, "
        f"Section 420, Section 304B, etc.\n"
        f"   • Act names: IPC, BNS, CrPC, BNSS, NI Act, POCSO, PMLA, NDPS, "
        f"Consumer Protection Act, Payment of Wages Act, RERA, RTI Act, "
        f"DV Act (Protection of Women from Domestic Violence Act), SARFAESI\n"
        f"   • Court names: Supreme Court, High Court, District Court, "
        f"Sessions Court, Chief Judicial Magistrate, NCDRC, SCDRC, "
        f"Labour Court, Consumer Forum, Debt Recovery Tribunal\n"
        f"   • Procedural terms: FIR, bail, chargesheet, cognizable, "
        f"non-cognizable, bailable, warrant, summons, affidavit, pleadings, "
        f"complainant, accused, respondent, petitioner\n"
        f"   • Penalties/measures: imprisonment, fine, rigorous imprisonment, "
        f"life imprisonment\n"
        f"2. Translate all other explanatory text to natural, fluent {lang_name} "
        f"that a layperson can understand.\n"
        f"3. Preserve paragraph structure and formatting.\n"
        f"4. Return ONLY the translated text. No preamble. No English label.\n\n"
        f"English text:\n{text}\n\n"
        f"{lang_name} translation:"
    )

    try:
        result = groq_client.generate(
            prompt        = prompt,
            system_prompt = _TRANSLATE_SYSTEM,
            max_tokens    = 1500,
            temperature   = 0.05,
        ).strip()

        result = _strip_llm_preamble(result)

        if not result:
            logger.warning(f"[MultilingualAgent] Empty translation result EN→{lang_name}")
            return text

        logger.info(f"[MultilingualAgent] EN→{lang_name}: {result[:80]!r}")
        return result

    except Exception as e:
        logger.error(f"[MultilingualAgent] translate_to_source failed (EN→{lang_name}): {e}")
        return text  # return English as fallback rather than empty string


def _strip_llm_preamble(text: str) -> str:
    """
    Remove common LLM preamble phrases injected before the actual translation.

    Examples of preambles that get stripped:
      "Here is the translation: ..."
      "Translation: ..."
      "Sure! Here is the ..."
      "Certainly, here is the Malayalam translation: ..."
    """
    patterns = [
        re.compile(
            r'^(?:here\s+is|here\'s)\s+(?:the\s+)?(?:\w+\s+)?translation\s*[:\-—]\s*',
            re.IGNORECASE,
        ),
        re.compile(r'^translation\s*[:\-—]\s*', re.IGNORECASE),
        re.compile(r'^translated\s+text\s*[:\-—]\s*', re.IGNORECASE),
        re.compile(r'^(?:sure|certainly|of\s+course)[!,.]?\s*(?:here\s+is.{0,40})?\s*', re.IGNORECASE),
    ]
    result = text.strip()
    for pat in patterns:
        result = pat.sub("", result).strip()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def process_multilingual_query(
    query:        str,
    session_id:   str,
    rag_pipeline,
    groq_client,
) -> dict:
    """
    Full multilingual query processing pipeline.

    End-to-end flow:
      1. detect_language(query)          → ISO code
      2. translate_to_english(query)     → English query (if non-English)
      3. rag_pipeline.query(en_query)    → English legal answer
      4. translate_to_source(en_answer)  → Source language answer (if non-English)

    Performance notes:
      - Steps 2 and 4 are each one Groq API call (~0.5–1s each on free tier).
      - Step 3 is the full RAG pipeline (~2–5s depending on complexity routing).
      - Total latency for a multilingual query: ~4–7s.

    Args:
        query:        Raw user query in any language
        session_id:   LangGraph session ID (for logging)
        rag_pipeline: rag.pipeline.rag_pipeline singleton
        groq_client:  rag.llm.llm singleton

    Returns:
        {
          "response":           str,   # Final answer in user's own language
          "detected_language":  str,   # ISO 639-1 code, e.g. "ml"
          "original_language":  str,   # Display name, e.g. "Malayalam"
          "translated_query":   str,   # English query used for RAG (transparency)
          "english_response":   str,   # Raw English RAG answer (for debug/logging)
          "sources_consulted":  int,   # From RAG pipeline
          "synthesis_note":     str,
          "grounding_warning":  str,
          "rewritten_queries":  list[str],
          "reranker_used":      bool,
          "mode":               str,   # e.g. "multilingual_ml"
        }
    """
    detected_lang = detect_language(query)
    lang_name     = get_language_name(detected_lang)

    print(
        f"[MultilingualAgent] session={session_id[:8] if session_id else '?'}… "
        f"detected={detected_lang!r} ({lang_name}) | query={query[:60]!r}"
    )

    # ── Step 1: Translate query to English ────────────────────────────────────
    if detected_lang != "en":
        english_query = translate_to_english(query, detected_lang, groq_client)
        print(f"[MultilingualAgent] EN query: {english_query[:80]!r}")
    else:
        english_query = query

    # ── Step 2: RAG pipeline on English query ─────────────────────────────────
    rag_meta: dict = {
        "sources_consulted": 0,
        "synthesis_note":    "",
        "grounding_warning": "",
        "rewritten_queries": [],
        "reranker_used":     False,
    }
    try:
        rag_answer       = rag_pipeline.query(english_query)
        english_response = rag_answer.answer_text
        rag_meta = {
            "sources_consulted": rag_answer.sources_consulted,
            "synthesis_note":    rag_answer.synthesis_note    or "",
            "grounding_warning": rag_answer.grounding_warning or "",
            "rewritten_queries": rag_answer.rewritten_queries or [],
            "reranker_used":     rag_answer.reranker_used,
        }
        print(f"[MultilingualAgent] RAG complete — {rag_meta['sources_consulted']} source(s)")
    except Exception as e:
        logger.error(f"[MultilingualAgent] RAG pipeline error: {e}")
        english_response = (
            "I encountered an error processing your legal query. "
            "Please try again."
        )

    # ── Step 3: Translate response back to source language ────────────────────
    if detected_lang != "en":
        final_response = translate_to_source(english_response, detected_lang, groq_client)
    else:
        final_response = english_response

    return {
        "response":          final_response,
        "detected_language": detected_lang,
        "original_language": lang_name,
        "translated_query":  english_query,
        "english_response":  english_response,
        **rag_meta,
        "mode": f"multilingual_{detected_lang}",
    }