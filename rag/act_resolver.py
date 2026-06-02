"""
LexShield AI — Act Name Resolver
==================================
Resolves an act name from a query string to a ChromaDB source partial string.

Problem solved:
  "Limited Liability Partnership Act section 24"
  -> old code: dict iteration hit "partnership" -> "Indian Partnership Act" ❌
  -> this module: sorts all act names longest-first, "limited liability partnership"
    (31 chars) matches before "partnership" (11 chars) -> correct source ✅

Design:
  - ACT_REGISTRY: complete list of (match_phrase, source_partial) pairs
  - Sorted by length DESCENDING at module load
  - resolve_act(text) returns the FIRST (longest) match — most specific wins
  - All comparisons are lowercase

Usage:
    from rag.act_resolver import act_resolver
    source = act_resolver.resolve_act("limited liability partnership act section 24")
    # -> "Limited Liability Partnership"
"""

import re
from typing import Optional

import logging

logger = logging.getLogger(__name__)

# ── Act registry ──────────────────────────────────────────────────────────────
# Format: (match_phrase_lowercase, source_partial_string)
# source_partial must be a substring of the `source` field in ChromaDB metadata.
# Add entries when new acts are ingested.
# ORDER DOES NOT MATTER HERE — sorted at build time.

_RAW_REGISTRY: list[tuple[str, str]] = [

    # ── Criminal — new codes (longer names first keeps collision safe) ─────────
    ("bharatiya nagarik suraksha sanhita", "Bharatiya Nagarik Suraksha Sanhita"),
    ("bharatiya nyaya sanhita",            "Bharatiya Nyaya Sanhita"),
    ("bharatiya sakshya adhiniyam",        "Bharatiya Sakshya Adhiniyam"),
    ("bnss",                               "Bharatiya Nagarik Suraksha Sanhita"),
    ("bns",                                "Bharatiya Nyaya Sanhita"),
    ("bsa",                                "Bharatiya Sakshya Adhiniyam"),

    # ── Criminal — old codes ──────────────────────────────────────────────────
    ("code of criminal procedure",         "Code of Criminal Procedure"),
    ("indian penal code",                  "Indian Penal Code"),
    ("indian evidence act",                "Indian Evidence Act"),
    ("crpc",                               "Code of Criminal Procedure"),
    ("ipc",                                "Indian Penal Code"),

    # ── Specific criminal acts ────────────────────────────────────────────────
    ("protection of children from sexual offences", "Protection of Children from Sexual Offences"),
    ("prevention of money laundering",     "Prevention of Money Laundering"),
    ("narcotic drugs and psychotropic substances", "Narcotic Drugs"),
    ("unlawful activities prevention",     "Unlawful Activities"),
    ("prevention of corruption",           "Prevention of Corruption"),
    ("juvenile justice",                   "Juvenile Justice"),
    ("pocso",                              "Protection of Children from Sexual Offences"),
    ("pmla",                               "Prevention of Money Laundering"),
    ("ndps",                               "Narcotic Drugs"),
    ("uapa",                               "Unlawful Activities"),

    # ── Family ────────────────────────────────────────────────────────────────
    ("protection of women from domestic violence", "Protection of Women from Domestic Violence"),
    ("maintenance and welfare of parents", "Maintenance and Welfare of Parents"),
    ("muslim personal law",                "Muslim Personal Law"),
    ("hindu succession",                   "Hindu Succession Act"),
    ("hindu marriage",                     "Hindu Marriage Act"),
    ("special marriage",                   "Special Marriage Act"),
    ("indian succession",                  "Indian Succession Act"),
    ("family courts",                      "Family Courts Act"),
    ("domestic violence",                  "Protection of Women from Domestic Violence"),
    ("dv act",                             "Protection of Women from Domestic Violence"),

    # ── Corporate — CRITICAL: longer names before shorter substrings ──────────
    ("micro small and medium enterprises development", "Micro, Small and Medium Enterprises"),
    ("limited liability partnership",      "Limited Liability Partnership"),
    ("arbitration and conciliation",       "Arbitration and Conciliation Act"),
    ("insolvency and bankruptcy",          "Insolvency and Bankruptcy Code"),
    ("negotiable instruments",             "Negotiable Instruments Act"),
    ("indian partnership",                 "Indian Partnership Act"),
    ("indian contract",                    "Indian Contract Act"),
    ("specific relief",                    "Specific Relief Act"),
    ("competition act",                    "Competition Act"),
    ("companies act",                      "Companies Act"),
    ("msmed",                              "Micro, Small and Medium Enterprises"),
    ("msme",                               "Micro, Small and Medium Enterprises"),
    ("ibc",                                "Insolvency and Bankruptcy Code"),
    ("llp",                                "Limited Liability Partnership"),
    ("ni act",                             "Negotiable Instruments Act"),

    # ── Taxation ──────────────────────────────────────────────────────────────
    ("central goods and services tax",     "Central Goods and Services Tax"),
    ("integrated goods and services tax",  "Integrated Goods and Services Tax"),
    ("securities and exchange board",      "Securities and Exchange Board of India"),
    ("foreign exchange management",        "Foreign Exchange Management"),
    ("banking regulation",                 "Banking Regulation Act"),
    ("income tax",                         "Income Tax Act"),
    ("customs act",                        "Customs Act"),
    ("cgst",                               "Central Goods and Services Tax"),
    ("igst",                               "Integrated Goods and Services Tax"),
    ("fema",                               "Foreign Exchange Management"),
    ("sebi",                               "Securities and Exchange Board of India"),

    # ── Property ──────────────────────────────────────────────────────────────
    ("real estate regulation and development", "Real Estate"),
    ("kerala buildings lease and rent",    "Kerala Buildings"),
    ("transfer of property",               "Transfer of Property Act"),
    ("registration act",                   "Registration Act"),
    ("rera",                               "Real Estate"),
    ("kerala rent",                        "Kerala Buildings"),

    # ── Labour ────────────────────────────────────────────────────────────────
    ("sexual harassment of women at workplace", "Sexual Harassment of Women at Workplace"),
    ("occupational safety health and working conditions", "Occupational Safety"),
    ("industrial relations code",          "Industrial Relations Code"),
    ("code on social security",            "Code on Social Security"),
    ("code on wages",                      "Code on Wages"),
    ("posh act",                           "Sexual Harassment of Women at Workplace"),
    ("posh",                               "Sexual Harassment of Women at Workplace"),

    # ── Health ────────────────────────────────────────────────────────────────
    ("food safety and standards",          "Food Safety and Standards"),
    ("drugs and cosmetics",                "Drugs and Cosmetics Act"),
    ("clinical establishments",            "Clinical Establishments"),
    ("right of children to free and compulsory education", "Right of Children to Free and Compulsory Education"),
    ("fssai",                              "Food Safety and Standards"),
    ("rte act",                            "Right of Children to Free and Compulsory Education"),

    # ── Environment ───────────────────────────────────────────────────────────
    ("water prevention and control of pollution", "Water (Prevention and Control of Pollution)"),
    ("air prevention and control of pollution",   "Air (Prevention and Control of Pollution)"),
    ("wildlife protection",                "Wildlife (Protection) Act"),
    ("forest conservation",                "Forest (Conservation) Act"),
    ("environment protection",             "Environment (Protection) Act"),
    ("indian forest",                      "Indian Forest Act"),
    ("epa",                                "Environment (Protection) Act"),

    # ── Technology ────────────────────────────────────────────────────────────
    ("digital personal data protection",   "Digital Personal Data Protection"),
    ("information technology",             "Information Technology"),
    ("trade marks",                        "Trade Marks Act"),
    ("trademark",                          "Trade Marks Act"),
    ("patents act",                        "Patents Act"),
    ("copyright act",                      "Copyright Act"),
    ("it act",                             "Information Technology"),
    ("dpdp",                               "Digital Personal Data Protection"),

    # ── Civil ─────────────────────────────────────────────────────────────────
    ("code of civil procedure",            "Code of Civil Procedure"),
    ("aadhaar targeted delivery",          "Aadhaar"),
    ("right to information",               "Right to Information"),
    ("motor vehicles act",                 "Motor Vehicles Act"),
    ("consumer protection",                "Consumer Protection"),
    ("constitution of india",              "Constitution of India"),
    ("motor vehicle act",                  "Motor Vehicles Act"),   # singular
    ("mv act",                             "Motor Vehicles Act"),
    ("rti",                                "Right to Information"),
    ("cpc",                                "Code of Civil Procedure"),
    ("aadhaar",                            "Aadhaar"),
]

# Sort LONGEST MATCH FIRST — critical for collision prevention
_REGISTRY: list[tuple[str, str]] = sorted(_RAW_REGISTRY, key=lambda x: -len(x[0]))


class ActResolver:
    """
    Resolves act name from query text using longest-match-first ordering.

    Longest match wins — prevents "partnership" from matching
    "Limited Liability Partnership Act" before the full name does.
    """

    def resolve_act(self, text: str) -> Optional[str]:
        """
        Return source_partial for the most specific act name found in text.
        Returns None if no act name found.

        text should be the expanded query string (lowercase-safe).
        """
        t = text.lower()
        # Remove punctuation except spaces
        t = re.sub(r'[^\w\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()

        for phrase, source_partial in _REGISTRY:
            if phrase in t:
                return source_partial

        return None

    def resolve_all_acts(self, text: str) -> list[str]:
        """
        Return all source_partials found in text (longest match per overlap).
        Used when a query mentions multiple acts.
        """
        t = text.lower()
        t = re.sub(r'[^\w\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()

        found:   list[str] = []
        matched_spans: list[tuple[int, int]] = []

        for phrase, source_partial in _REGISTRY:
            idx = t.find(phrase)
            if idx == -1:
                continue
            end = idx + len(phrase)
            # Skip if this span overlaps with an already-matched longer phrase
            overlaps = any(
                not (end <= ms or idx >= me)
                for ms, me in matched_spans
            )
            if not overlaps:
                found.append(source_partial)
                matched_spans.append((idx, end))

        return found

    def resolve_section_source(
        self,
        query:         str,
        section_number: str,
        window: int = 120,
    ) -> Optional[str]:
        """
        Resolve source_partial for a specific section reference in query.

        Searches ±window chars around the section number match for act name.
        Falls back to whole-query search.

        Example:
          query = "Limited Liability Partnership Act section 24"
          section_number = "24"
          -> scans local context -> finds "limited liability partnership act"
          -> returns "Limited Liability Partnership"
        """
        # Find section number position in query
        pattern = re.compile(
            r'\b[Ss]ections?\s*\.?\s*' + re.escape(section_number) + r'\b'
            r'|'
            r'\b' + re.escape(section_number) + r'\s+(?:of|under|in)\b',
            re.IGNORECASE,
        )
        m = pattern.search(query)

        if m:
            start = max(0, m.start() - window)
            end   = min(len(query), m.end() + window)
            local = query[start:end]
        else:
            local = query

        result = self.resolve_act(local)
        if result is None:
            result = self.resolve_act(query)  # whole query fallback
        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
act_resolver = ActResolver()