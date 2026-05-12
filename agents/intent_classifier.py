"""
LexShield AI — Intent Classifier
==================================
Rule-based classifier. No ML needed.

6 intents:
  legal_query        → RAG pipeline
  document_analysis  → CV pipeline + RAG
  draft_request      → DraftingAgent (Week 3 Day 2-3)
  risk_check         → RAG pipeline + risk prompt modifier
  translation_request→ MultilingualAgent (Week 3 Day 4-5)
  general            → Direct LLM

Method:
  keyword match = 1 pt each
  regex pattern match = 2 pts each
  Highest total score wins.
  Confidence = min(raw_score / 10, 1.0), +0.2 bonus if any pattern matched.
"""

import re
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

INTENTS = [
    "legal_query",
    "document_analysis",
    "draft_request",
    "risk_check",
    "translation_request",
    "general",
]

# ── Keywords (1 pt each) ───────────────────────────────────────────────────────

_KEYWORDS: dict[str, list[str]] = {
    "legal_query": [
        "section", "article", "provision", "act", "ipc", "bns", "crpc", "bnss",
        "bsa", "evidence", "court", "judgment", "bail", "arrest", "fir", "charge",
        "offense", "offence", "punishment", "penalty", "conviction", "acquittal",
        "warrant", "custody", "rights", "legal", "law", "constitution", "writ",
        "petition", "appeal", "tribunal", "magistrate", "sessions", "high court",
        "supreme court", "divorce", "maintenance", "property", "inheritance",
        "consumer", "cheque", "bounce", "pocso", "ndps", "pmla", "uapa",
        "what", "explain", "define", "meaning", "scope", "applicability",
    ],
    "document_analysis": [
        "document", "file", "pdf", "image", "scan", "ocr", "upload", "attached",
        "extract", "read", "text", "contract", "deed", "agreement", "notice",
        "this document", "the document", "check document", "review document",
        "analyze document", "analyse document",
    ],
    "draft_request": [
        "draft", "write", "create", "generate", "prepare", "compose", "format",
        "template", "sample", "make", "help me write", "help me draft",
        "legal notice", "rental agreement", "employment contract", "affidavit",
        "power of attorney", "bail application", "fir complaint", "police complaint",
        "cheque bounce notice", "consumer complaint", "loan agreement",
    ],
    "risk_check": [
        "risk", "risky", "safe", "dangerous", "liable", "liability", "exposure",
        "consequence", "consequences", "breach", "violation", "penalty", "enforceable",
        "valid", "binding", "legal standing", "what happens if", "can i be",
        "will i be", "am i liable", "legal risk", "is it legal", "is it safe",
    ],
    "translation_request": [
        "translate", "translation", "malayalam", "hindi", "tamil", "telugu",
        "kannada", "marathi", "bengali", "gujarati", "punjabi", "odia",
        "regional language", "vernacular", "in malayalam", "in hindi",
        "explain in", "say in", "convert to",
    ],
    "general": [
        "hello", "hi", "hey", "help", "what can you do", "about you",
        "who are you", "tell me", "thanks", "thank you", "good morning",
        "good evening", "how are you", "what is lexshield", "features",
        "capabilities", "bye", "goodbye",
    ],
}

# ── Regex patterns (2 pts each) ────────────────────────────────────────────────

_PATTERNS: dict[str, list[re.Pattern]] = {
    "legal_query": [
        re.compile(r'\b(section|article|clause)\s+\d+[A-Z]?\b', re.IGNORECASE),
        re.compile(r'\b(ipc|bns|crpc|bnss|bsa|pocso|pmla|ndps|uapa|rti|rera|ni\s?act)\b', re.IGNORECASE),
        re.compile(r'\bwhat\s+(is|are|does)\b.{0,40}\b(law|act|section|legal|offence|offense|punishment)\b', re.IGNORECASE),
        re.compile(r'\b(punishment|penalty|sentence|imprisonment)\s+for\b', re.IGNORECASE),
        re.compile(r'\b(high\s+court|supreme\s+court|district\s+court|sessions\s+court)\b', re.IGNORECASE),
        re.compile(r'\bhow\s+(do|can|does|to)\b.{0,40}\b(file|register|apply|get|obtain)\b.{0,40}\b(fir|bail|writ|petition|complaint)\b', re.IGNORECASE),
    ],
    "document_analysis": [
        re.compile(r'\b(analyze|analyse|review|check|read|scan|extract)\s+(this\s+)?(document|file|pdf|image|contract|deed|agreement|notice)\b', re.IGNORECASE),
        re.compile(r'\b(what\s+does\s+this|summarize\s+this|tell\s+me\s+about\s+this)\b.{0,30}\b(document|file|contract|agreement)\b', re.IGNORECASE),
        re.compile(r'\b(extract|pull\s+out)\s+(text|clauses|sections|content)\s+(from|in)\b', re.IGNORECASE),
    ],
    "draft_request": [
        re.compile(r'\b(draft|write|create|generate|prepare|compose)\b.{0,40}\b(notice|agreement|contract|complaint|petition|affidavit|deed|letter|application)\b', re.IGNORECASE),
        re.compile(r'\b(template|format|sample|boilerplate)\b.{0,40}\b(legal|agreement|notice|contract|fir|deed)\b', re.IGNORECASE),
        re.compile(r'\bhelp\s+me\s+(write|draft|create|prepare)\b', re.IGNORECASE),
        re.compile(r'\b(fir|first\s+information\s+report)\b.{0,30}\b(draft|write|create|file|format|template)\b', re.IGNORECASE),
    ],
    "risk_check": [
        re.compile(r'\b(legal\s+risk|risk\s+of|am\s+i\s+(liable|at\s+risk)|is\s+(it|this)\s+(legal|safe|risky|valid))\b', re.IGNORECASE),
        re.compile(r'\b(what\s+(happens|are\s+the\s+consequences)|consequences\s+of|liable\s+for)\b', re.IGNORECASE),
        re.compile(r'\b(breach\s+of\s+contract|violation\s+of|penalty\s+for\s+not)\b', re.IGNORECASE),
        re.compile(r'\b(enforceable|legally\s+binding|legal\s+standing|can\s+i\s+be\s+(sued|arrested|charged))\b', re.IGNORECASE),
    ],
    "translation_request": [
        re.compile(r'\b(translate|translation)\b.{0,40}\b(malayalam|hindi|tamil|telugu|kannada|marathi|bengali|gujarati)\b', re.IGNORECASE),
        re.compile(r'\bexplain\s+(this\s+)?(in|to\s+me\s+in)\s+(malayalam|hindi|tamil|telugu|kannada|marathi)\b', re.IGNORECASE),
        re.compile(r'\b(in|into)\s+(malayalam|hindi|tamil|telugu|kannada|marathi|bengali|gujarati|punjabi|odia)\b', re.IGNORECASE),
        re.compile(r'\b(convert|say|write)\s+(this\s+)?in\s+(malayalam|hindi|tamil|telugu|kannada)\b', re.IGNORECASE),
    ],
    "general": [
        re.compile(r'^(hi|hello|hey|good\s+(morning|evening|afternoon|night))[\s!?.]*$', re.IGNORECASE),
        re.compile(r'\b(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+lexshield|tell\s+me\s+about\s+yourself)\b', re.IGNORECASE),
        re.compile(r'^(thanks?|thank\s+you|bye|goodbye|ok|okay|sure)[\s!.]*$', re.IGNORECASE),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentResult:
    intent: str
    confidence: float
    scores: dict[str, float]
    pattern_matched: bool
    matched_patterns: list[str]


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:

    def classify(self, text: str) -> IntentResult:
        """
        Classify input text into one of 6 intents.

        Args:
            text: raw user query string

        Returns:
            IntentResult with intent name, confidence, and debug info
        """
        text_lower = text.lower().strip()
        scores: dict[str, float] = {intent: 0.0 for intent in INTENTS}
        matched_patterns: list[str] = []
        pattern_hit = False

        # ── Keyword scoring (1 pt each) ────────────────────────────────────────
        _DRAFT_OVERRIDE = re.compile(
            r'\bhelp\s+me\s+(draft|write|create|prepare)\b'
            r'|\b(draft|write|create|prepare)\s+a\s+\w+\s+(notice|agreement|contract|complaint|petition|affidavit|deed|application|letter)\b',
            re.IGNORECASE,
        )
        _TRANSLATION_OVERRIDE = re.compile(
            r'\btranslate\b.{0,60}\b(into|to|in)\s+(malayalam|hindi|tamil|telugu|kannada|marathi|bengali|gujarati)\b'
            r'|\bexplain\s+(this\s+)?in\s+(malayalam|hindi|tamil|telugu|kannada|marathi)\b',
            re.IGNORECASE,
        )
        if _DRAFT_OVERRIDE.search(text):
            scores["draft_request"] += 5.0
        if _TRANSLATION_OVERRIDE.search(text):
            scores["translation_request"] += 5.0

        # ── Keyword scoring (1 pt each) ────────────────────────────────────────
        for intent, keywords in _KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[intent] += 1.0

        # ── Pattern scoring (2 pts each) ───────────────────────────────────────
        for intent, patterns in _PATTERNS.items():
            for pat in patterns:
                if pat.search(text):
                    scores[intent] += 2.0
                    pattern_hit = True
                    matched_patterns.append(f"{intent}:{pat.pattern[:40]}")

        # ── Pick winner ────────────────────────────────────────────────────────
        best_intent = max(scores, key=lambda k: scores[k])
        raw_score   = scores[best_intent]

        # Fallback to general if all scores are 0
        if raw_score == 0.0:
            best_intent = "general"

        # Confidence: scale raw score, bonus if pattern matched
        confidence = min(raw_score / 10.0, 1.0)
        if pattern_hit and scores[best_intent] > 0:
            confidence = min(confidence + 0.2, 1.0)

        return IntentResult(
            intent=best_intent,
            confidence=round(confidence, 3),
            scores=scores,
            pattern_matched=pattern_hit,
            matched_patterns=matched_patterns,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
intent_classifier = IntentClassifier()