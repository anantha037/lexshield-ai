"""
LexShield AI — Intent Classifier  (Week 3, Day 2 Session 2 — Updated)
=======================================================================
Changes from previous session:
  - rights_check added as 8th intent → rights_node
    Triggered by: "my rights", "rights as a tenant/employee/consumer",
    "know my rights", "what are my rights", bail rights, women rights.

8 intents:
  legal_query         → RAG pipeline (explain/define Indian law)
  document_analysis   → CV pipeline + RAG (uploaded documents)
  draft_request       → DraftingAgent, 8 complaint categories
  risk_check          → RAG pipeline + risk_scorer modifier
  translation_request → TranslationAgent (explicit: "explain in Malayalam")
  case_law_search     → CaseLawAgent → Indian Kanoon live judgments
  rights_check        → RightsAgent → structured rights guide + RAG enrichment
  general             → Direct LLM (greetings, capability questions)

Scoring:
  keyword match  = +1 pt
  regex pattern  = +2 pts
  hard override  = +5 pts (applied before scoring)
  Winner:        max total score
  Confidence:    min(score / 10, 1.0) + 0.2 bonus if any regex matched

Design rationale for rights_check vs legal_query disambiguation:
  "What is Section 17 PWDVA?" → legal_query (specific section explanation)
  "What are women's rights under PWDVA?" → rights_check (structured guide)
  "What are my rights as a tenant?" → rights_check (strong override trigger)
  The _RIGHTS_OVERRIDE hard override (+5 pts) fires before DRAFT_OVERRIDE so
  "tenant rights" routes to rights_check, not draft_request.
"""

import re
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

INTENTS = [
    "legal_query",
    "document_analysis",
    "draft_request",
    "risk_check",
    "translation_request",
    "case_law_search",
    "rights_check",
    "general",
]

# ── Keywords (1 pt each) ───────────────────────────────────────────────────────

_KEYWORDS: dict[str, list[str]] = {

    "legal_query": [
        "section", "article", "provision", "act", "ipc", "bns", "crpc", "bnss",
        "bsa", "evidence", "court", "bail", "arrest", "fir", "charge",
        "offense", "offence", "punishment", "penalty", "conviction", "acquittal",
        "warrant", "custody", "legal", "law", "constitution", "writ",
        "petition", "appeal", "tribunal", "magistrate", "sessions", "high court",
        "supreme court", "divorce", "maintenance", "property", "inheritance",
        "consumer", "cheque", "bounce", "pocso", "ndps", "pmla", "uapa",
        "what", "explain", "define", "meaning", "scope", "applicability",
        "rera", "rti", "income tax", "gst", "companies act", "sebi",
        "specific relief", "transfer of property", "hindu marriage",
        "limitation act", "arbitration", "insolvency", "bankruptcy",
    ],

    "document_analysis": [
        "document", "file", "pdf", "image", "scan", "ocr", "upload", "attached",
        "extract", "read", "text", "contract", "deed", "agreement", "notice",
        "this document", "the document", "check document", "review document",
        "analyze document", "analyse document", "attached file",
    ],

    "draft_request": [
        "draft", "write", "create", "generate", "prepare", "compose", "format",
        "template", "sample", "make", "help me write", "help me draft",
        "legal notice", "rental agreement", "employment contract", "affidavit",
        "power of attorney", "bail application",
        "fir complaint", "police complaint", "complaint to police",
        "complaint against", "file a complaint", "file complaint",
        "help me file", "write a complaint",
        "salary not paid", "salary not received", "wage theft", "unpaid salary",
        "unpaid wages", "salary complaint", "labour complaint",
        "payment of wages", "employer not paying",
        "eviction", "evicted", "landlord complaint", "illegal eviction",
        "thrown out", "locked out", "dispossessed",
        "cheque bounce complaint", "138 ni act", "cheque dishonour",
        "cheque bounce notice", "demand notice cheque",
        "consumer complaint", "consumer forum", "consumer court",
        "defective product", "deficiency in service",
        "domestic violence", "498a complaint", "cruelty complaint",
        "dv act complaint", "protection order",
        "wrongful termination", "illegal termination", "unfair dismissal",
        "termination complaint", "labour court complaint",
        "loan complaint", "bank complaint", "rbi complaint",
        "loan harassment", "recovery agent harassment",
    ],

    "risk_check": [
        "risk", "risky", "safe", "dangerous", "liable", "liability", "exposure",
        "consequence", "consequences", "breach", "violation", "enforceable",
        "valid", "binding", "legal standing", "what happens if", "can i be",
        "will i be", "am i liable", "legal risk", "is it legal", "is it safe",
        "penalty for", "what if i", "am i at risk",
    ],

    "translation_request": [
        "translate", "translation", "malayalam", "hindi", "tamil", "telugu",
        "kannada", "marathi", "bengali", "gujarati", "punjabi", "odia",
        "regional language", "vernacular", "in malayalam", "in hindi",
        "explain in", "say in", "convert to",
    ],

    "case_law_search": [
        "case law", "judgment", "judgement", "landmark", "precedent", "ruling",
        "verdict", "court held", "held that", "bench", "division bench",
        "constitution bench", "full bench", "PIL", "public interest litigation",
        "AIR", "SCC", "SCR", "CrLJ", "DLT", "BomCR", "KerLT", "MLJ",
        "SC order", "HC order", "apex court", "top court",
        "case of", "in the matter of", "versus", "v.", "vs",
        "cited", "overruled", "upheld", "quashed", "dismissed", "allowed",
        "set aside", "remanded", "observed", "directed", "declared",
    ],

    "rights_check": [
        "my rights", "our rights", "your rights", "know my rights",
        "what are my rights", "what are the rights", "rights as",
        "legal rights", "fundamental rights", "constitutional rights",
        "right to", "entitled to", "protected by law",
        "tenant rights", "renter rights",
        "employee rights", "worker rights", "labour rights", "workers rights",
        "consumer rights", "buyer rights",
        "women rights", "woman rights", "wife rights",
        "bail rights", "rights of arrested", "rights of accused",
        "arrest rights", "detention rights",
        "know your rights", "understand my rights", "explain my rights",
        "am i protected", "can they do this", "is this allowed",
    ],

    "general": [
        "hello", "hi", "hey", "help", "what can you do", "about you",
        "who are you", "tell me", "thanks", "thank you", "good morning",
        "good evening", "how are you", "what is lexshield", "features",
        "capabilities", "bye", "goodbye", "welcome", "start",
    ],
}

# ── Regex patterns (2 pts each) ────────────────────────────────────────────────

_PATTERNS: dict[str, list[re.Pattern]] = {

    "legal_query": [
        re.compile(r'\b(section|article|clause)\s+\d+[A-Za-z]?\b', re.IGNORECASE),
        re.compile(
            r'\b(ipc|bns|crpc|bnss|bsa|pocso|pmla|ndps|uapa|rti|rera|'
            r'ni\s?act|companies\s+act|sebi\s+act)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\bwhat\s+(is|are|does)\b.{0,40}\b(law|act|section|legal|offence|'
            r'offense|punishment|provision)\b',
            re.IGNORECASE,
        ),
        re.compile(r'\b(punishment|penalty|sentence|imprisonment)\s+for\b', re.IGNORECASE),
        re.compile(
            r'\b(high\s+court|supreme\s+court|district\s+court|sessions\s+court|'
            r'magistrate\s+court|family\s+court)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\bhow\s+(do|can|does|to)\b.{0,40}\b(file|register|apply|get|obtain)\b'
            r'.{0,40}\b(fir|bail|writ|petition|complaint)\b',
            re.IGNORECASE,
        ),
        re.compile(r'\b(bailable|non.bailable|cognizable|non.cognizable)\b', re.IGNORECASE),
        re.compile(r'\b(anticipatory\s+bail|regular\s+bail|default\s+bail)\b', re.IGNORECASE),
    ],

    "document_analysis": [
        re.compile(
            r'\b(analyze|analyse|review|check|read|scan|extract)\s+(this\s+)?'
            r'(document|file|pdf|image|contract|deed|agreement|notice)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(what\s+does\s+this|summarize\s+this|tell\s+me\s+about\s+this)\b'
            r'.{0,30}\b(document|file|contract|agreement)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(extract|pull\s+out)\s+(text|clauses|sections|content)\s+(from|in)\b',
            re.IGNORECASE,
        ),
    ],

    "draft_request": [
        re.compile(
            r'\b(draft|write|create|generate|prepare|compose)\b.{0,40}'
            r'\b(notice|agreement|contract|complaint|petition|affidavit|deed|'
            r'letter|application|reply|response)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(template|format|sample|boilerplate)\b.{0,40}'
            r'\b(legal|agreement|notice|contract|fir|deed)\b',
            re.IGNORECASE,
        ),
        re.compile(r'\bhelp\s+me\s+(write|draft|create|prepare)\b', re.IGNORECASE),
        re.compile(
            r'\b(fir|first\s+information\s+report)\b.{0,30}'
            r'\b(draft|write|create|file|format|template)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(salary|wages?)\s+(not\s+paid|withheld|pending|dues|not\s+received|stolen)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(employer|company)\s+.{0,20}\b(not\s+paying|cheating|fraud|absconding)\b',
            re.IGNORECASE,
        ),
        re.compile(r'\blabour\s+commissioner\b', re.IGNORECASE),
        re.compile(
            r'\b(illegally?\s+evict(ed|ion)?|forcible\s+eviction|unlawful\s+eviction|'
            r'thrown\s+out|locked\s+out|dispossess(ed|ion)?)\b',
            re.IGNORECASE,
        ),
        re.compile(r'\b(landlord).{0,30}\b(evict|lock|threaten|harass)\b', re.IGNORECASE),
        re.compile(
            r'\b(cheque\s+bounce|cheque\s+dishonour(ed)?|section\s+138|'
            r'ni\s+act\s+complaint|demand\s+notice.{0,20}cheque)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(consumer\s+(complaint|forum|court|commission)|defective\s+product|'
            r'deficiency\s+in\s+service|refund\s+complaint)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(write\s+a?\s*complaint\s+to\s+police|complaint\s+against.{0,30}'
            r'(theft|assault|fraud|cheating|robbery)|help\s+me\s+file.{0,20}fir)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(complaint\s+against|file\s+complaint\s+against|complaint\s+to\s+'
            r'(police|court|forum|commission|tribunal))\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(domestic\s+violence\s+complaint|498[aA]\s+complaint|dv\s+act|'
            r'protection\s+order|cruelty\s+by\s+husband)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(wrongful\s+termination|illegal\s+termination|unfair\s+dismissal|'
            r'terminated.{0,20}complaint|labour\s+court.{0,20}(complaint|application))\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(loan\s+(harassment|complaint)|rbi\s+ombudsman|recovery\s+agent.{0,20}'
            r'(harassment|complaint)|sarfaesi|drt\s+complaint)\b',
            re.IGNORECASE,
        ),
    ],

    "risk_check": [
        re.compile(
            r'\b(legal\s+risk|risk\s+of|am\s+i\s+(liable|at\s+risk)|'
            r'is\s+(it|this)\s+(legal|safe|risky|valid))\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(what\s+(happens|are\s+the\s+consequences)|consequences\s+of|liable\s+for)\b',
            re.IGNORECASE,
        ),
        re.compile(r'\b(breach\s+of\s+contract|violation\s+of|penalty\s+for\s+not)\b', re.IGNORECASE),
        re.compile(
            r'\b(enforceable|legally\s+binding|legal\s+standing|'
            r'can\s+i\s+be\s+(sued|arrested|charged))\b',
            re.IGNORECASE,
        ),
    ],

    "translation_request": [
        re.compile(
            r'\b(translate|translation)\b.{0,40}'
            r'\b(malayalam|hindi|tamil|telugu|kannada|marathi|bengali|gujarati)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\bexplain\s+(this\s+)?(in|to\s+me\s+in)\s+'
            r'(malayalam|hindi|tamil|telugu|kannada|marathi)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(in|into)\s+(malayalam|hindi|tamil|telugu|kannada|marathi|'
            r'bengali|gujarati|punjabi|odia)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(convert|say|write)\s+(this\s+)?in\s+'
            r'(malayalam|hindi|tamil|telugu|kannada)\b',
            re.IGNORECASE,
        ),
    ],

    "case_law_search": [
        re.compile(
            r'\b\d{4}\s+(?:SCC|AIR|SCR|CrLJ|DLT|BomCR|KerLT|MLJ|'
            r'SCALE|SLT|GLR|KLT|RCR)\s*\(?(?:Cri|Crl|Criminal)?\)?\s*\d+\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(supreme\s+court|high\s+court|apex\s+court)\s+'
            r'.{0,30}\b(held|ruled|decided|observed|stated|declared|directed|'
            r'quashed|upheld|dismissed|allowed|set\s+aside)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(landmark|leading|important|significant|notable)\s+'
            r'(case|judgment|judgement|ruling|verdict|decision)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(?:state|union\s+of\s+india|cbi)\s+(?:v\.?|vs\.?|versus)\s+\w+',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b\w+\s+(?:v\.?|vs\.?|versus)\s+(?:state|union\s+of\s+india|cbi|cci)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(find|search|show|give)\s+(me\s+)?(cases?|judgments?|judgements?|'
            r'precedents?|rulings?)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\bwhat\s+(did|has)\s+.{0,30}(court|bench)\s+'
            r'(held?|ruled?|decided?|said?)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(pil|public\s+interest\s+litigation)\b.{0,30}'
            r'\b(filed?|decided?|case|judgment)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(kesavananda\s+bharati|maneka\s+gandhi|vishaka|'
            r'mohd\.\s+ahmed\s+khan|shah\s+bano|indira\s+gandhi|'
            r'bandhua\s+mukti\s+morcha|olga\s+tellis|navtej\s+johar|'
            r'puttaswamy|triple\s+talaq|sabarimala|ayodhya)\b',
            re.IGNORECASE,
        ),
    ],

    "rights_check": [
        # "rights of/as/for <role>" — strongest signal
        re.compile(
            r'\brights?\s+(of|as|for)\s+'
            r'(tenant|tenants?|renter|employee|employees?|worker|workers?|'
            r'labour|consumer|consumers?|woman|women|wife|arrested|accused|detainee)\b',
            re.IGNORECASE,
        ),
        # "my/our/your rights"
        re.compile(r'\b(my|our|your|their)\s+rights?\b', re.IGNORECASE),
        # "know/understand my rights"
        re.compile(
            r'\b(know|understand|explain|tell\s+me)\s+(my|your|our)\s+rights?\b',
            re.IGNORECASE,
        ),
        # "what are my rights"
        re.compile(r'\bwhat\s+are\s+(my|the|our)\s+rights?\b', re.IGNORECASE),
        # "<role> rights"
        re.compile(
            r'\b(tenant|employee|consumer|women?|worker|labour|bail|arrested)\s+rights?\b',
            re.IGNORECASE,
        ),
        # "am I protected/entitled under law"
        re.compile(
            r'\b(am\s+i|are\s+we)\s+(protected|entitled|allowed|permitted)\b.{0,40}'
            r'\b(law|legally|act|rights?)\b',
            re.IGNORECASE,
        ),
        # "can landlord/employer/police do this legally?"
        re.compile(
            r'\bcan\s+(landlord|employer|police|company)\s+.{0,30}'
            r'\b(legally|lawfully|do\s+this|allowed)\b',
            re.IGNORECASE,
        ),
    ],

    "general": [
        re.compile(
            r'^(hi|hello|hey|good\s+(morning|evening|afternoon|night))[\s!?.]*$',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+lexshield|'
            r'tell\s+me\s+about\s+yourself|your\s+capabilities)\b',
            re.IGNORECASE,
        ),
        re.compile(r'^(thanks?|thank\s+you|bye|goodbye|ok|okay|sure)[\s!.]*$', re.IGNORECASE),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentResult:
    intent:           str
    confidence:       float
    scores:           dict[str, float]
    pattern_matched:  bool
    matched_patterns: list[str]


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    Keyword + regex scorer for 8 LexShield intents.

    Hard overrides (+5 pts) are applied in priority order:
      RIGHTS_OVERRIDE fires first — prevents "tenant rights" routing to draft_request.
      DRAFT_OVERRIDE  fires second.
      TRANSLATION_OVERRIDE and CASE_LAW_OVERRIDE fire for their respective intents.
    """

    _RIGHTS_OVERRIDE = re.compile(
        r'\b(my|our)\s+rights?\b'
        r'|\brights?\s+(of|as|for)\s+'
        r'(tenant|employee|worker|consumer|women?|wife|arrested|accused|detainee)\b'
        r'|\b(know|understand)\s+(my|your|our)\s+rights?\b'
        r'|\bwhat\s+are\s+(my|the\s+tenant|the\s+employee|the\s+consumer|'
        r'women\'?s?|bail)\s+rights?\b'
        r'|\b(tenant|employee|consumer|workers?|bail|women?)\s+rights?\b'
        r'|\bcan\s+(my\s+)?(landlord|employer|police|company)\s+.{0,25}'
        r'\b(legally|lawfully|do\s+this|is\s+this\s+allowed)\b',
        re.IGNORECASE,
    )

    _DRAFT_OVERRIDE = re.compile(
        r'\bhelp\s+me\s+(draft|write|create|prepare)\b'
        r'|\b(draft|write|create|prepare)\s+a\s+\w+\s+'
        r'(notice|agreement|contract|complaint|petition|affidavit|deed|application|letter)\b'
        r'|\b(salary|wages?)\s+(not\s+paid|withheld|not\s+received)\b'
        r'|\b(cheque\s+bounce|cheque\s+dishonour)\s+complaint\b'
        r'|\billegal(ly)?\s+evict(ed|ion)?\b'
        r'|\bdomestic\s+violence\s+complaint\b'
        r'|\bwrongful\s+termination\b'
        r'|\bwrite\s+(a\s+)?complaint\b'
        r'|\bfile\s+(a\s+)?complaint\b'
        r'|\bhelp\s+me\s+file\b'
        r'|\bsalary\s+not\s+(paid|received)\b'
        r'|\bwrongfully?\s+terminat(ed|ion)\b',
        re.IGNORECASE,
    )

    _TRANSLATION_OVERRIDE = re.compile(
        r'\btranslate\b.{0,60}\b(into|to|in)\s+'
        r'(malayalam|hindi|tamil|telugu|kannada|marathi|bengali|gujarati)\b'
        r'|\bexplain\s+(this\s+)?in\s+'
        r'(malayalam|hindi|tamil|telugu|kannada|marathi)\b',
        re.IGNORECASE,
    )

    _CASE_LAW_OVERRIDE = re.compile(
        r'\b\d{4}\s+(?:SCC|AIR|SCR|CrLJ|DLT|BomCR|KerLT|MLJ|SCALE)\b'
        r'|\b(kesavananda\s+bharati|maneka\s+gandhi|vishaka|shah\s+bano|'
        r'navtej\s+johar|puttaswamy|triple\s+talaq)\b'
        r'|\bwhat\s+did\s+.{0,30}\s+(court|bench)\s+(hold|rule|decide|say)\b'
        r'|\b(find|show|search)\s+(me\s+)?(case|cases|judgment|judgments)\b',
        re.IGNORECASE,
    )

    def classify(self, text: str) -> IntentResult:
        """
        Classify input text into one of 8 intents.

        Args:
            text: Raw user query (any language — non-English auto-detection
                  runs separately in classify_intent_node via detect_language)

        Returns:
            IntentResult with intent, confidence [0.0-1.0], all scores, debug info.
        """
        text_lower        = text.lower().strip()
        scores: dict[str, float] = {intent: 0.0 for intent in INTENTS}
        matched_patterns: list[str] = []
        pattern_hit       = False

        # ── Hard overrides (priority order matters) ────────────────────────────
        if self._RIGHTS_OVERRIDE.search(text):
            scores["rights_check"] += 5.0

        if self._DRAFT_OVERRIDE.search(text):
            scores["draft_request"] += 5.0

        if self._TRANSLATION_OVERRIDE.search(text):
            scores["translation_request"] += 5.0

        if self._CASE_LAW_OVERRIDE.search(text):
            scores["case_law_search"] += 5.0

        # ── Keyword scoring (+1 each) ──────────────────────────────────────────
        for intent, keywords in _KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[intent] += 1.0

        # ── Pattern scoring (+2 each) ──────────────────────────────────────────
        for intent, patterns in _PATTERNS.items():
            for pat in patterns:
                if pat.search(text):
                    scores[intent] += 2.0
                    pattern_hit = True
                    matched_patterns.append(f"{intent}:{pat.pattern[:55]}")

        # ── Winner ─────────────────────────────────────────────────────────────
        best_intent = max(scores, key=lambda k: scores[k])
        raw_score   = scores[best_intent]

        if raw_score == 0.0:
            best_intent = "general"

        confidence = min(raw_score / 10.0, 1.0)
        if pattern_hit and scores.get(best_intent, 0) > 0:
            confidence = min(confidence + 0.2, 1.0)

        return IntentResult(
            intent           = best_intent,
            confidence       = round(confidence, 3),
            scores           = scores,
            pattern_matched  = pattern_hit,
            matched_patterns = matched_patterns,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
intent_classifier = IntentClassifier()