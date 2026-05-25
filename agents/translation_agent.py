"""
LexShield AI — Translation Agent
===================================
Detects non-English queries and performs:
  1. Translate query -> English
  2. Run RAG on English query
  3. Translate answer -> original language

Language detection:
  Script-based (Unicode range check) + keyword matching.
  Reliable, zero dependencies, works offline.

Translation:
  Groq LLaMA 3.3 70B via prompt — same API key already configured.
  AI4Bharat skipped — Windows install unreliable.

Supported languages:
  Malayalam, Hindi, Tamil, Telugu, Kannada, Marathi,
  Bengali, Gujarati, Punjabi, Odia, Urdu
"""

import re
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Unicode script ranges for Indian languages
_UNICODE_RANGES: list[tuple[int, int, str]] = [
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0900, 0x097F, "Hindi"),       # Devanagari (Hindi + Marathi)
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0A00, 0x0A7F, "Punjabi"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0600, 0x06FF, "Urdu"),
]

# English keyword triggers for translation intent
_TRANSLATION_TRIGGERS: dict[str, str] = {
    "malayalam":  "Malayalam",
    "hindi":      "Hindi",
    "tamil":      "Tamil",
    "telugu":     "Telugu",
    "kannada":    "Kannada",
    "marathi":    "Marathi",
    "bengali":    "Bengali",
    "gujarati":   "Gujarati",
    "punjabi":    "Punjabi",
    "odia":       "Odia",
    "urdu":       "Urdu",
}

_TRANSLATE_RE = re.compile(
    r'\b(?:translate|explain|say|write|convert)\b.{0,60}'
    r'\b(?:in|into|to)\s+(malayalam|hindi|tamil|telugu|kannada|marathi'
    r'|bengali|gujarati|punjabi|odia|urdu)\b',
    re.IGNORECASE,
)

_EXPLAIN_IN_RE = re.compile(
    r'\bexplain\s+(?:this\s+)?in\s+(malayalam|hindi|tamil|telugu|kannada|marathi)\b',
    re.IGNORECASE,
)


@dataclass
class LanguageDetectionResult:
    is_english:      bool
    detected_script: Optional[str]   # e.g. "Malayalam"
    target_language: Optional[str]   # explicitly requested output language
    original_query:  str


def detect_language(query: str) -> LanguageDetectionResult:
    """
    Detect if query contains non-English script or requests a specific language.

    Returns:
      is_english:      True if query appears to be English
      detected_script: script found in query text (for non-English input)
      target_language: output language requested (from English query like "translate to Malayalam")
    """
    # Check for non-ASCII Indian script characters
    detected_script = None
    for char in query:
        cp = ord(char)
        for start, end, lang in _UNICODE_RANGES:
            if start <= cp <= end:
                detected_script = lang
                break
        if detected_script:
            break

    # Check for explicit language request in English query
    target_language = None
    m = _TRANSLATE_RE.search(query) or _EXPLAIN_IN_RE.search(query)
    if m:
        target_language = _TRANSLATION_TRIGGERS.get(m.group(1).lower(), m.group(1).title())

    # Also check plain keyword mention at end of query
    if not target_language:
        q_lower = query.lower()
        for kw, lang in _TRANSLATION_TRIGGERS.items():
            if kw in q_lower:
                target_language = lang
                break

    is_english = (detected_script is None)

    return LanguageDetectionResult(
        is_english      = is_english,
        detected_script = detected_script,
        target_language = target_language,
        original_query  = query,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_TRANSLATE = (
    "You are a professional Indian legal translator. "
    "Translate text accurately, preserving all legal terminology. "
    "For section numbers and act names, keep them in their original form. "
    "Translate only — do not add explanations or summaries."
)


def _translate_to_english_prompt(text: str, source_lang: str) -> str:
    return (
        f"Translate the following {source_lang} text to English.\n"
        f"Keep all legal terms, section numbers, and act names unchanged.\n\n"
        f"Text:\n{text}\n\n"
        f"English translation:"
    )


def _translate_from_english_prompt(text: str, target_lang: str) -> str:
    return (
        f"Translate the following English legal text to {target_lang}.\n"
        f"Keep all section numbers (e.g. Section 302), act names (e.g. Indian Penal Code), "
        f"and legal terms in their original English form — only translate surrounding text.\n\n"
        f"Text:\n{text}\n\n"
        f"{target_lang} translation:"
    )


def _strip_legal_content(query: str, target_lang: str) -> str:
    """
    Remove the translation instruction from query to extract the actual legal question.
    e.g. "Translate this into Malayalam: What is Section 302 IPC?"
         -> "What is Section 302 IPC?"
    """
    patterns = [
        re.compile(
            r'(?:translate|convert|explain|say|write)\s+(?:this\s+)?'
            r'(?:into|in|to)\s+' + re.escape(target_lang) + r'\s*[:\-—]?\s*',
            re.IGNORECASE,
        ),
        re.compile(
            r'(?:in|into)\s+' + re.escape(target_lang) + r'\s*[:\-—]?\s*',
            re.IGNORECASE,
        ),
    ]
    clean = query
    for pat in patterns:
        clean = pat.sub("", clean).strip()
    return clean or query


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class TranslationAgent:

    @staticmethod
    def _get_llm():
        from rag.llm import llm
        return llm

    @staticmethod
    def _get_rag():
        from rag.pipeline import rag_pipeline
        return rag_pipeline

    def handle(self, query: str, session_id: str = "") -> dict:
        """
        Main entry point for translation_request intent.

        Workflow:
          Case A — Non-English input (e.g. Malayalam query):
            1. Detect script -> translate query to English
            2. Run RAG on English query
            3. Translate RAG answer back to detected language

          Case B — English query requesting translation (e.g. "explain in Malayalam"):
            1. Extract actual legal question from query
            2. Run RAG on English question
            3. Translate RAG answer to requested language

          Case C — Translation request with no clear legal question:
            -> Direct LLM translation of provided text

        Returns:
          answer:            translated answer string
          source_language:   detected input language
          target_language:   output language
          english_query:     the English query used for RAG (for transparency)
          sources_consulted: from RAG
          mode:              translation path taken
        """
        lang = detect_language(query)

        # ── Case A — Non-English script in query ──────────────────────────────
        if not lang.is_english and lang.detected_script:
            return self._handle_script_query(query, lang.detected_script)

        # ── Case B — English query requesting translation ──────────────────────
        if lang.target_language:
            return self._handle_translation_request(query, lang.target_language)

        # ── Case C — Fallback: translate provided text ─────────────────────────
        return self._handle_direct_translation(query)

    def _handle_script_query(self, query: str, source_lang: str) -> dict:
        """Non-English query -> translate -> RAG -> translate back."""
        llm = self._get_llm()
        rag = self._get_rag()

        print(f"[TranslationAgent] Detected {source_lang} script -> translating to English")

        # Step 1: Translate to English
        try:
            english_query = llm.generate(
                prompt        = _translate_to_english_prompt(query, source_lang),
                system_prompt = _SYSTEM_TRANSLATE,
                max_tokens    = 300,
                temperature   = 0.1,
            ).strip()
        except Exception as e:
            print(f"[TranslationAgent] Translation to English failed: {e}")
            english_query = query  # fallback

        # Step 2: RAG on English query
        legal_answer = rag.query(english_query)
        english_ans  = legal_answer.answer_text

        # Step 3: Translate answer back
        try:
            translated_answer = llm.generate(
                prompt        = _translate_from_english_prompt(english_ans, source_lang),
                system_prompt = _SYSTEM_TRANSLATE,
                max_tokens    = 800,
                temperature   = 0.1,
            ).strip()
        except Exception as e:
            print(f"[TranslationAgent] Translation back to {source_lang} failed: {e}")
            translated_answer = english_ans

        return {
            "answer":            translated_answer,
            "source_language":   source_lang,
            "target_language":   source_lang,
            "english_query":     english_query,
            "sources_consulted": legal_answer.sources_consulted,
            "synthesis_note":    legal_answer.synthesis_note or "",
            "grounding_warning": legal_answer.grounding_warning or "",
            "rewritten_queries": legal_answer.rewritten_queries or [],
            "reranker_used":     legal_answer.reranker_used,
            "mode":              "script_query_translation",
        }

    def _handle_translation_request(self, query: str, target_lang: str) -> dict:
        """English query asking for answer in target language."""
        llm = self._get_llm()
        rag = self._get_rag()

        print(f"[TranslationAgent] English query -> answer in {target_lang}")

        # Extract the actual legal question
        english_query = _strip_legal_content(query, target_lang)

        # RAG on English question
        legal_answer = rag.query(english_query)
        english_ans  = legal_answer.answer_text

        # Translate answer to target language
        try:
            translated_answer = llm.generate(
                prompt        = _translate_from_english_prompt(english_ans, target_lang),
                system_prompt = _SYSTEM_TRANSLATE,
                max_tokens    = 800,
                temperature   = 0.1,
            ).strip()

            # Prepend English answer too for reference
            final_answer = (
                f"[English]\n{english_ans}\n\n"
                f"[{target_lang}]\n{translated_answer}"
            )
        except Exception as e:
            print(f"[TranslationAgent] Translation to {target_lang} failed: {e}")
            final_answer = english_ans

        return {
            "answer":            final_answer,
            "source_language":   "English",
            "target_language":   target_lang,
            "english_query":     english_query,
            "sources_consulted": legal_answer.sources_consulted,
            "synthesis_note":    legal_answer.synthesis_note or "",
            "grounding_warning": legal_answer.grounding_warning or "",
            "rewritten_queries": legal_answer.rewritten_queries or [],
            "reranker_used":     legal_answer.reranker_used,
            "mode":              "english_to_target_translation",
        }

    def _handle_direct_translation(self, query: str) -> dict:
        """Fallback — no clear legal question found, translate text directly."""
        llm = self._get_llm()

        print(f"[TranslationAgent] Direct translation fallback")

        answer = llm.generate(
            prompt = (
                f"The user has a translation request. Help them as best as you can.\n\n"
                f"Request: {query}"
            ),
            system_prompt = _SYSTEM_TRANSLATE,
            max_tokens    = 600,
            temperature   = 0.1,
        )

        return {
            "answer":            answer,
            "source_language":   "Unknown",
            "target_language":   "Unknown",
            "english_query":     query,
            "sources_consulted": 0,
            "synthesis_note":    "Direct translation — no RAG",
            "grounding_warning": "",
            "rewritten_queries": [],
            "reranker_used":     False,
            "mode":              "direct_translation",
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
translation_agent = TranslationAgent()