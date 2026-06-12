"""
LexShield AI — LangChain Tool Definitions for LLM Tool-Calling Routing
========================================================================
Defines 8 @tool decorated functions, one per intent bucket.

These tools are used by the orchestrator's bound LLM (llm.bind_tools) to
select the correct agent via tool-calling.  The LLM reads each tool's
docstring to decide which one to invoke for a given query.

IMPORTANT: These functions do NOT execute any agent logic — they return
a routing signal string ("ROUTE:<intent>") that classify_with_tool_calls()
parses to extract the intent name.  Actual execution still flows through
classify_intent_node -> route_by_intent() -> agent node -> END.

Usage:
    from agents.tools import ALL_TOOLS
    bound_llm = langchain_llm.bind_tools(ALL_TOOLS)
"""

from langchain_core.tools import tool


@tool
def tool_legal_query(query: str) -> str:
    """Answer questions about Indian law — sections, acts, articles, legal
    definitions, procedures, punishments, bail provisions, FIR process,
    court hierarchies, or any specific legal concept under Indian statutes
    such as IPC/BNS, CrPC/BNSS, Evidence Act/BSA, POCSO, NDPS, PMLA,
    UAPA, RERA, RTI, Companies Act, SEBI Act, Transfer of Property Act,
    Indian Contract Act, Hindu Marriage Act, Limitation Act, or the
    Constitution of India.

    Examples:
    - "What is Section 302 IPC?"
    - "Explain anticipatory bail under CrPC"
    - "What is the punishment for cheque bounce under NI Act?"
    - "Difference between cognizable and non-cognizable offences"
    - "धारा 302 IPC क्या है?" (legal question asked in Hindi)
    - "Article 21 के बारे में बताएं"
    """
    return "ROUTE:legal_query"


@tool
def tool_document_analysis(query: str) -> str:
    """Analyse, review, summarise, or extract information from an uploaded
    or pasted legal document such as a contract, deed, agreement, court
    notice, FIR copy, or legal letter.

    Use this when the user has attached, uploaded, or pasted document text
    and wants it reviewed, summarised, or explained.

    Examples:
    - "Analyse this rental agreement"
    - "Summarise this court notice"
    - "What does this contract clause mean?"
    - "Review the attached legal document"
    """
    return "ROUTE:document_analysis"


@tool
def tool_draft_request(query: str) -> str:
    """Draft, write, create, or prepare a legal document, complaint, notice,
    application, or letter for the user.  Covers 8 complaint categories:
    FIR/police complaint, salary/wage complaint, illegal eviction complaint,
    cheque bounce notice (Section 138 NI Act), consumer complaint, domestic
    violence complaint (498A/DV Act), wrongful termination complaint, and
    loan/bank harassment complaint (RBI ombudsman).

    Also covers: legal notices, rental agreements, employment contracts,
    affidavits, power of attorney, bail applications, and petitions.

    Use this ONLY when the user explicitly asks to draft, write, create,
    prepare, compose, or generate a document — not when they are asking
    about their rights or seeking legal information.

    Examples:
    - "Help me draft a legal notice to my landlord"
    - "Write a complaint about salary not paid"
    - "Create a bail application"
    - "Prepare an FIR complaint for theft"
    """
    return "ROUTE:draft_request"


@tool
def tool_risk_check(query: str) -> str:
    """Assess legal risk, liability, consequences, or whether a specific
    action or situation is legal or safe under Indian law.

    Use this when the user asks "Am I liable?", "Is this legal?",
    "What happens if I...?", "Can I be sued/arrested/charged?", or wants
    to understand the legal exposure or consequences of an action.

    Examples:
    - "Am I liable if my employee gets injured at the site?"
    - "Is it legal to record phone calls in India?"
    - "What are the consequences of breaking a rental agreement?"
    - "Can I be arrested for a civil dispute?"
    """
    return "ROUTE:risk_check"


@tool
def tool_translation_request(query: str) -> str:
    """IMPORTANT: A question asked IN Hindi/regional language about Indian law
    (e.g. "धारा 144 क्या है?") is legal_query, NOT translation_request.
    Use this ONLY when the user explicitly asks to TRANSLATE or CONVERT content.

    Translate or explain legal content in an Indian regional language.
    Supported languages: Malayalam, Hindi, Tamil, Telugu, Kannada, Marathi,
    Bengali, Gujarati, Punjabi, and Odia.

    Use this ONLY when the user explicitly asks to translate, convert, or
    explain something in a specific Indian language.

    Examples:
    - "Translate this in Malayalam"
    - "Explain Section 420 IPC in Hindi"
    - "Convert this legal notice to Tamil"
    - "Say this in Telugu"
    """
    return "ROUTE:translation_request"


@tool
def tool_case_law_search(query: str) -> str:
    """Search for Indian court judgments, case law, precedents, rulings, or
    verdicts from the Supreme Court, High Courts, and tribunals.  Handles
    citation lookups (e.g. "2023 SCC 456"), landmark case queries, and
    requests to find cases on a specific legal topic.

    Use this when the user asks about judgments, verdicts, court rulings,
    specific case names (e.g. "Kesavananda Bharati", "Maneka Gandhi"),
    or wants to search for precedents.

    Examples:
    - "Show me Supreme Court judgments on anticipatory bail"
    - "What did the court hold in Vishaka vs State of Rajasthan?"
    - "Find landmark cases on right to privacy"
    - "2019 SCC 438"
    """
    return "ROUTE:case_law_search"


@tool
def tool_rights_check(query: str) -> str:
    """Explain the user's legal rights, entitlements, or protections under
    Indian law for a specific role or situation.  Covers: tenant/renter
    rights, employee/worker rights, consumer/buyer rights, women's rights
    (domestic violence, dowry, workplace harassment), and bail/arrest rights
    (rights of the arrested or accused person).

    Use this when the user asks about "my rights", "rights as a tenant/
    employee/consumer", "what can I do" in a situation where they feel
    wronged, or "can my landlord/employer/police do this legally?".

    IMPORTANT DISAMBIGUATION:
    - "I got fired unfairly, what can I do?" -> rights_check (seeking guidance)
    - "Help me write a complaint about wrongful termination" -> draft_request
    - "What is Section 25 of the Industrial Disputes Act?" -> legal_query

    Examples:
    - "What are my rights as a tenant?"
    - "I was terminated without notice, what can I do?"
    - "Can my landlord evict me without a court order?"
    - "What are women's rights under the DV Act?"
    - "Check the risks in this NDA / agreement / contract"
    - "What are the legal risks of this clause?"
    """
    return "ROUTE:rights_check"


@tool
def tool_general(query: str) -> str:
    """Handle greetings, general chit-chat, capability questions, or queries
    that do not fall into any specific Indian legal category.

    Use this for: "hello", "hi", "what can you do?", "who are you?",
    "thank you", "goodbye", or any non-legal conversation.

    Examples:
    - "Hello"
    - "What can LexShield do?"
    - "Thanks for your help"
    - "Good morning"
    """
    return "ROUTE:general"


# ── Exported list of all tools ─────────────────────────────────────────────────

ALL_TOOLS = [
    tool_legal_query,
    tool_document_analysis,
    tool_draft_request,
    tool_risk_check,
    tool_translation_request,
    tool_case_law_search,
    tool_rights_check,
    tool_general,
]
