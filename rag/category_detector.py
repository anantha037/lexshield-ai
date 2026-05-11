"""
LexShield AI — Hybrid Category Detector
=========================================
Two-signal category prediction:
 
  Signal 1 — Keyword scoring (existing, unchanged)
    Fast rule-based matching of legal terms → raw score per category
 
  Signal 2 — Semantic scoring (NEW)
    Embeds the query using the existing all-MiniLM-L6-v2 embedder.
    Compares against pre-computed embeddings of category description strings.
    Cosine similarity → semantic score per category.
    Category embeddings are computed ONCE on first call and cached in memory.
    Uses the embedder singleton already loaded — zero additional model load.
 
  Fusion (NEW)
    final_score = KEYWORD_WEIGHT * keyword_norm + SEMANTIC_WEIGHT * semantic_score
    KEYWORD_WEIGHT  = 0.55   (keywords are precise for legal domain)
    SEMANTIC_WEIGHT = 0.45   (semantic catches rephrased / indirect queries)
 
    confidence = top_category_final_score / sum(all_final_scores)
 
Thresholds:
  >= CONFIDENCE_HIGH (0.55) → use as ChromaDB filter
  >= CONFIDENCE_MED  (0.35) → dual search
  <  CONFIDENCE_MED         → global search
 
Fallback:
  If embedder unavailable (import error, cold start), falls back to
  keyword-only mode silently. No pipeline disruption.
"""
 
import re
import numpy as np
from typing import Optional
 
CONFIDENCE_HIGH  = 0.55
CONFIDENCE_MED   = 0.35
KEYWORD_WEIGHT   = 0.55
SEMANTIC_WEIGHT  = 0.45
 
# ── Category description strings for semantic matching ────────────────────────
# These are written to maximally cover the vocabulary users actually query with.
# Each string is embedded once and cached.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "criminal": (
        "murder culpable homicide rape robbery theft cheating fraud forgery assault "
        "kidnapping abduction extortion dacoity criminal offence punishment imprisonment "
        "fine bail arrest warrant FIR first information report police investigation "
        "trial conviction acquittal sentence IPC BNS Indian Penal Code Bharatiya Nyaya "
        "Sanhita CRPC BNSS criminal procedure magistrate sessions court cognizable "
        "non-cognizable bailable non-bailable POCSO NDPS PMLA UAPA money laundering "
        "narcotics terrorism juvenile justice drunk driving penal code"
    ),
    "family": (
        "divorce marriage matrimonial alimony maintenance custody child guardianship "
        "adoption succession inheritance will testamentary widow spouse husband wife "
        "dowry domestic violence protection order shared household aggrieved person "
        "family court personal law Hindu Marriage Muslim law Shariat talaq nikah "
        "judicial separation annulment cruelty desertion adultery restitution conjugal "
        "rights senior citizen parents domestic relationship live-in relationship "
        "cohabitation child custody visitation rights child support stridhan"
    ),
    "corporate": (
        "company director shareholder board meeting quorum annual general meeting AGM "
        "incorporation memorandum articles of association prospectus debenture dividend "
        "insolvency bankruptcy liquidation winding up IBC NCLT resolution professional "
        "committee of creditors cheque bounce dishonour section 138 negotiable instrument "
        "promissory note bill of exchange partnership LLP limited liability MSME MSMED "
        "delayed payment supplier contract breach agreement arbitration specific "
        "performance injunction creditor debtor moratorium competition monopoly merger "
        "acquisition amalgamation takeover audit auditor statutory compliance frequency "
        "meetings notice minutes register"
    ),
    "taxation": (
        "income tax GST CGST IGST SGST goods services tax TDS TCS advance tax capital "
        "gains deduction exemption refund assessment reassessment scrutiny penalty tax "
        "evasion customs duty import export excise SEBI securities stock market shares "
        "IPO insider trading banking RBI reserve bank FEMA foreign exchange NBFC loan "
        "interest rate financial black money ITR return filing input tax credit"
    ),
    "property": (
        "property land flat apartment house building lease rent landlord tenant eviction "
        "tenancy rent control sale deed title ownership possession mortgage registration "
        "RERA real estate builder developer construction encumbrance easement transfer "
        "conveyance gift deed partition mutation adverse possession stamp duty Kerala "
        "rent buildings lease agreement notice vacate immovable property"
    ),
    "labour": (
        "employee employer worker workman labour wages salary minimum wage overtime "
        "termination dismissal retrenchment layoff strike lockout trade union collective "
        "bargaining industrial dispute provident fund EPF gratuity bonus ESI maternity "
        "leave paternity working hours sexual harassment POSH internal complaints "
        "committee workplace occupational safety workmen compensation contract labour "
        "apprentice social security industrial relations labour code"
    ),
    "health": (
        "food safety adulteration FSSAI food standard drug medicine hospital clinic "
        "doctor patient medical clinical establishment healthcare pharmaceutical "
        "cosmetic ayurvedic nutraceutical health supplement pharmacy drug license "
        "nursing home right to education RTE school compulsory education elementary "
        "mid day meal medical negligence vaccination genetically modified food"
    ),
    "environment": (
        "environment pollution air pollution water pollution noise emission effluent "
        "discharge hazardous waste forest deforestation tree wildlife poaching "
        "biodiversity national park sanctuary protected area mining quarrying "
        "environmental impact assessment EPA CPCB SPCB climate carbon green"
    ),
    "technology": (
        "cyber cybercrime hacking unauthorised access computer data breach data privacy "
        "personal data protection digital electronic digital signature internet social "
        "media IT Act DPDP information technology intermediary encryption phishing "
        "online fraud cyberbullying copyright patent trademark intellectual property "
        "software domain name cyber security"
    ),
    "civil": (
        "constitution fundamental rights right to life article 21 directive principles "
        "motor vehicle road accident traffic driving license vehicle registration "
        "insurance accident compensation RTI right to information public information "
        "officer Aadhaar consumer protection consumer forum deficiency in service "
        "unfair trade practice civil procedure CPC decree writ habeas corpus mandamus "
        "certiorari PIL public interest litigation citizenship election"
    ),
}

# Format: "keyword": ("category", weight)
# weight 2 = specific/multi-word phrase, weight 1 = general term
# Sorted longest-first at runtime to prevent sub-word false matches.

_KEYWORDS: dict[str, tuple[str, int]] = {

    # ═══════════════════════════════════════════════════════════════
    # CRIMINAL
    # ═══════════════════════════════════════════════════════════════
    # IPC / BNS offences
    "murder":                       ("criminal", 2),
    "culpable homicide":            ("criminal", 2),
    "homicide":                     ("criminal", 2),
    "attempt to murder":            ("criminal", 2),
    "manslaughter":                 ("criminal", 2),
    "rape":                         ("criminal", 2),
    "sexual assault":               ("criminal", 2),
    "sexual abuse":                 ("criminal", 2),
    "molestation":                  ("criminal", 2),
    "outrage modesty":              ("criminal", 2),
    "robbery":                      ("criminal", 2),
    "dacoity":                      ("criminal", 2),
    "theft":                        ("criminal", 1),
    "extortion":                    ("criminal", 2),
    "kidnapping":                   ("criminal", 2),
    "abduction":                    ("criminal", 2),
    "wrongful confinement":         ("criminal", 2),
    "wrongful restraint":           ("criminal", 2),
    "hurt":                         ("criminal", 1),
    "grievous hurt":                ("criminal", 2),
    "assault":                      ("criminal", 1),
    "criminal force":               ("criminal", 2),
    "cheating":                     ("criminal", 1),
    "fraud":                        ("criminal", 1),
    "forgery":                      ("criminal", 1),
    "counterfeit":                  ("criminal", 1),
    "sedition":                     ("criminal", 2),
    "defamation":                   ("criminal", 1),
    "abetment":                     ("criminal", 2),
    "conspiracy":                   ("criminal", 1),
    "criminal conspiracy":          ("criminal", 2),
    "rioting":                      ("criminal", 1),
    "affray":                       ("criminal", 2),
    "trespass":                     ("criminal", 1),
    "criminal trespass":            ("criminal", 2),
    "bribery":                      ("criminal", 2),
    "corruption":                   ("criminal", 1),
    "mischief":                     ("criminal", 1),
    "arson":                        ("criminal", 2),
    "poisoning":                    ("criminal", 2),
    "stalking":                     ("criminal", 1),
    "voyeurism":                    ("criminal", 2),
    "acid attack":                  ("criminal", 2),
    "trafficking":                  ("criminal", 2),
    "human trafficking":            ("criminal", 2),
    "obscenity":                    ("criminal", 1),
    "obscene":                      ("criminal", 1),
    "counterfeit currency":         ("criminal", 2),
    "waging war":                   ("criminal", 2),
    "terrorism":                    ("criminal", 2),
    "terrorist":                    ("criminal", 2),
    "illegal purchase":             ("criminal", 2),
    "illegal sale":                 ("criminal", 2),
    "public servant":               ("criminal", 1),
    "disobey":                      ("criminal", 1),
    # Procedure / enforcement
    "ipc":                          ("criminal", 2),
    "bns":                          ("criminal", 2),
    "crpc":                         ("criminal", 2),
    "bnss":                         ("criminal", 2),
    "bsa":                          ("criminal", 2),
    "penal code":                   ("criminal", 2),
    "bharatiya nyaya":              ("criminal", 2),
    "bharatiya nagarik":            ("criminal", 2),
    "bharatiya sakshya":            ("criminal", 2),
    "fir":                          ("criminal", 1),
    "first information report":     ("criminal", 2),
    "arrest":                       ("criminal", 1),
    "bail":                         ("criminal", 1),
    "anticipatory bail":            ("criminal", 2),
    "custody":                      ("criminal", 1),
    "remand":                       ("criminal", 1),
    "police custody":               ("criminal", 2),
    "judicial custody":             ("criminal", 2),
    "bailable":                     ("criminal", 1),
    "non bailable":                 ("criminal", 2),
    "cognizable":                   ("criminal", 1),
    "non cognizable":               ("criminal", 2),
    "chargesheet":                  ("criminal", 1),
    "charge sheet":                 ("criminal", 2),
    "acquittal":                    ("criminal", 1),
    "conviction":                   ("criminal", 1),
    "sentence":                     ("criminal", 1),
    "imprisonment":                 ("criminal", 1),
    "life imprisonment":            ("criminal", 2),
    "death penalty":                ("criminal", 2),
    "capital punishment":           ("criminal", 2),
    "fine":                         ("criminal", 1),
    "punishment":                   ("criminal", 1),
    "offence":                      ("criminal", 1),
    "offense":                      ("criminal", 1),
    "warrant":                      ("criminal", 1),
    "search warrant":               ("criminal", 2),
    "summons":                      ("criminal", 1),
    "magistrate":                   ("criminal", 1),
    "sessions court":               ("criminal", 2),
    "sessions judge":               ("criminal", 2),
    "police":                       ("criminal", 1),
    "investigation":                ("criminal", 1),
    "challan":                      ("criminal", 1),
    "cognizance":                   ("criminal", 1),
    "trial":                        ("criminal", 1),
    "criminal trial":               ("criminal", 2),
    "criminal procedure":           ("criminal", 2),
    "evidence":                     ("criminal", 1),
    "witness":                      ("criminal", 1),
    "confession":                   ("criminal", 1),
    "pocso":                        ("criminal", 2),
    "child sexual abuse":           ("criminal", 2),
    "ndps":                         ("criminal", 2),
    "narcotics":                    ("criminal", 2),
    "narcotic":                     ("criminal", 2),
    "psychotropic":                 ("criminal", 2),
    "drug trafficking":             ("criminal", 2),
    "uapa":                         ("criminal", 2),
    "unlawful activities":          ("criminal", 2),
    "pmla":                         ("criminal", 2),
    "money laundering":             ("criminal", 2),
    "proceeds of crime":            ("criminal", 2),
    "juvenile":                     ("criminal", 1),
    "juvenile justice":             ("criminal", 2),
    "delinquent":                   ("criminal", 1),
    "drunk driving":                ("criminal", 2),
    "drunken driving":              ("criminal", 2),
    "driving under influence":      ("criminal", 2),
    "under influence":              ("criminal", 1),
    "criminal":                     ("criminal", 1),
    "penal":                        ("criminal", 1),

    # ═══════════════════════════════════════════════════════════════
    # FAMILY
    # ═══════════════════════════════════════════════════════════════
    "divorce":                      ("family", 2),
    "divorce petition":             ("family", 2),
    "dissolution of marriage":      ("family", 2),
    "marriage":                     ("family", 1),
    "matrimonial":                  ("family", 2),
    "matrimonial dispute":          ("family", 2),
    "matrimonial home":             ("family", 2),
    "alimony":                      ("family", 2),
    "maintenance":                  ("family", 1),
    "maintenance order":            ("family", 2),
    "child support":                ("family", 2),
    "custody":                      ("family", 1),
    "child custody":                ("family", 2),
    "visitation rights":            ("family", 2),
    "guardianship":                 ("family", 2),
    "guardian":                     ("family", 1),
    "adoption":                     ("family", 2),
    "succession":                   ("family", 1),
    "inheritance":                  ("family", 1),
    "will":                         ("family", 1),
    "testamentary":                 ("family", 2),
    "intestate":                    ("family", 2),
    "widow":                        ("family", 1),
    "widower":                      ("family", 1),
    "spouse":                       ("family", 1),
    "husband":                      ("family", 1),
    "wife":                         ("family", 1),
    "dowry":                        ("family", 2),
    "dowry death":                  ("family", 2),
    "dowry harassment":             ("family", 2),
    "stridhan":                     ("family", 2),
    "domestic violence":            ("family", 2),
    "domestic relationship":        ("family", 2),
    "domestic abuse":               ("family", 2),
    "domestic":                     ("family", 1),
    "aggrieved person":             ("family", 2),
    "aggrieved woman":              ("family", 2),
    "protection order":             ("family", 2),
    "residence order":              ("family", 2),
    "monetary relief":              ("family", 2),
    "shared household":             ("family", 2),
    "family court":                 ("family", 2),
    "family dispute":               ("family", 2),
    "conjugal rights":              ("family", 2),
    "conjugal":                     ("family", 1),
    "judicial separation":          ("family", 2),
    "annulment":                    ("family", 2),
    "void marriage":                ("family", 2),
    "voidable marriage":            ("family", 2),
    "personal law":                 ("family", 2),
    "hindu marriage":               ("family", 2),
    "christian marriage":           ("family", 2),
    "muslim marriage":              ("family", 2),
    "muslim law":                   ("family", 2),
    "shariat":                      ("family", 2),
    "nikah":                        ("family", 2),
    "talaq":                        ("family", 2),
    "triple talaq":                 ("family", 2),
    "mehr":                         ("family", 2),
    "restitution of conjugal rights": ("family", 2),
    "restitution":                  ("family", 1),
    "cruelty":                      ("family", 1),
    "mental cruelty":               ("family", 2),
    "desertion":                    ("family", 2),
    "adultery":                     ("family", 2),
    "senior citizen":               ("family", 1),
    "senior citizens":              ("family", 2),
    "parent":                       ("family", 1),
    "parents":                      ("family", 1),
    "welfare of child":             ("family", 2),
    "minor child":                  ("family", 2),
    "child marriage":               ("family", 2),
    "special marriage":             ("family", 2),
    "inter religion marriage":      ("family", 2),
    "live in":                      ("family", 2),
    "live-in":                      ("family", 2),
    "cohabitation":                 ("family", 2),
    "relationship in nature of marriage": ("family", 2),

    # ═══════════════════════════════════════════════════════════════
    # CORPORATE
    # ═══════════════════════════════════════════════════════════════
    "company":                      ("corporate", 1),
    "companies act":                ("corporate", 2),
    "corporation":                  ("corporate", 1),
    "director":                     ("corporate", 1),
    "managing director":            ("corporate", 2),
    "board of directors":           ("corporate", 2),
    "shareholder":                  ("corporate", 1),
    "shareholder rights":           ("corporate", 2),
    "share":                        ("corporate", 1),
    "equity":                       ("corporate", 1),
    "incorporation":                ("corporate", 2),
    "memorandum":                   ("corporate", 1),
    "articles of association":      ("corporate", 2),
    "prospectus":                   ("corporate", 2),
    "debenture":                    ("corporate", 2),
    "dividend":                     ("corporate", 1),
    "annual general meeting":       ("corporate", 2),
    "agm":                          ("corporate", 2),
    "extraordinary general meeting": ("corporate", 2),
    "egm":                          ("corporate", 2),
    "board meeting":                ("corporate", 2),
    "meeting":                      ("corporate", 1),
    "meetings":                     ("corporate", 1),
    "quorum":                       ("corporate", 2),
    "resolution":                   ("corporate", 1),
    "ordinary resolution":          ("corporate", 2),
    "special resolution":           ("corporate", 2),
    "frequency of meetings":        ("corporate", 2),
    "notice of meeting":            ("corporate", 2),
    "minute book":                  ("corporate", 2),
    "minutes":                      ("corporate", 1),
    "audit":                        ("corporate", 1),
    "auditor":                      ("corporate", 1),
    "statutory audit":              ("corporate", 2),
    "insolvency":                   ("corporate", 2),
    "bankruptcy":                   ("corporate", 2),
    "liquidation":                  ("corporate", 2),
    "winding up":                   ("corporate", 2),
    "ibc":                          ("corporate", 2),
    "nclt":                         ("corporate", 2),
    "nclat":                        ("corporate", 2),
    "corporate insolvency":         ("corporate", 2),
    "resolution professional":      ("corporate", 2),
    "committee of creditors":       ("corporate", 2),
    "cheque bounce":                ("corporate", 2),
    "dishonour of cheque":          ("corporate", 2),
    "cheque dishonour":             ("corporate", 2),
    "section 138":                  ("corporate", 2),
    "negotiable":                   ("corporate", 1),
    "negotiable instrument":        ("corporate", 2),
    "promissory note":              ("corporate", 2),
    "bill of exchange":             ("corporate", 2),
    "hundi":                        ("corporate", 2),
    "partnership":                  ("corporate", 1),
    "llp":                          ("corporate", 2),
    "limited liability":            ("corporate", 2),
    "msme":                         ("corporate", 2),
    "msmed":                        ("corporate", 2),
    "micro enterprise":             ("corporate", 2),
    "small enterprise":             ("corporate", 2),
    "medium enterprise":            ("corporate", 2),
    "delayed payment":              ("corporate", 2),
    "supplier payment":             ("corporate", 2),
    "contract":                     ("corporate", 1),
    "breach of contract":           ("corporate", 2),
    "agreement":                    ("corporate", 1),
    "indemnity":                    ("corporate", 1),
    "guarantee":                    ("corporate", 1),
    "surety":                       ("corporate", 1),
    "arbitration":                  ("corporate", 1),
    "arbitral":                     ("corporate", 1),
    "arbitral award":               ("corporate", 2),
    "specific performance":         ("corporate", 2),
    "injunction":                   ("corporate", 1),
    "creditor":                     ("corporate", 1),
    "debtor":                       ("corporate", 1),
    "moratorium":                   ("corporate", 2),
    "startup":                      ("corporate", 1),
    "business":                     ("corporate", 1),
    "trade":                        ("corporate", 1),
    "competition":                  ("corporate", 1),
    "monopoly":                     ("corporate", 2),
    "cartel":                       ("corporate", 2),
    "anti competitive":             ("corporate", 2),
    "merger":                       ("corporate", 1),
    "acquisition":                  ("corporate", 1),
    "takeover":                     ("corporate", 2),
    "amalgamation":                 ("corporate", 2),
    "intellectual property":        ("corporate", 1),
    "registered office":            ("corporate", 2),
    "fiduciary":                    ("corporate", 2),
    "statutory meeting":            ("corporate", 2),

    # ═══════════════════════════════════════════════════════════════
    # TAXATION
    # ═══════════════════════════════════════════════════════════════
    "tax":                          ("taxation", 1),
    "taxes":                        ("taxation", 1),
    "income tax":                   ("taxation", 2),
    "income tax return":            ("taxation", 2),
    "itr":                          ("taxation", 2),
    "taxable income":               ("taxation", 2),
    "tax liability":                ("taxation", 2),
    "tax deduction":                ("taxation", 2),
    "tds":                          ("taxation", 2),
    "tcs":                          ("taxation", 2),
    "advance tax":                  ("taxation", 2),
    "self assessment tax":          ("taxation", 2),
    "capital gains":                ("taxation", 2),
    "capital gain tax":             ("taxation", 2),
    "long term capital":            ("taxation", 2),
    "short term capital":           ("taxation", 2),
    "gst":                          ("taxation", 2),
    "cgst":                         ("taxation", 2),
    "igst":                         ("taxation", 2),
    "sgst":                         ("taxation", 2),
    "goods and services tax":       ("taxation", 2),
    "input tax credit":             ("taxation", 2),
    "itc":                          ("taxation", 2),
    "gst registration":             ("taxation", 2),
    "gst return":                   ("taxation", 2),
    "customs":                      ("taxation", 1),
    "customs duty":                 ("taxation", 2),
    "import duty":                  ("taxation", 2),
    "export duty":                  ("taxation", 2),
    "excise":                       ("taxation", 1),
    "duty":                         ("taxation", 1),
    "tax return":                   ("taxation", 2),
    "assessment":                   ("taxation", 1),
    "reassessment":                 ("taxation", 2),
    "scrutiny":                     ("taxation", 1),
    "tax refund":                   ("taxation", 2),
    "refund":                       ("taxation", 1),
    "exemption":                    ("taxation", 1),
    "deduction":                    ("taxation", 1),
    "standard deduction":           ("taxation", 2),
    "hra":                          ("taxation", 2),
    "house rent allowance":         ("taxation", 2),
    "surcharge":                    ("taxation", 1),
    "cess":                         ("taxation", 1),
    "penalty":                      ("taxation", 1),
    "tax evasion":                  ("taxation", 2),
    "tax avoidance":                ("taxation", 2),
    "black money":                  ("taxation", 2),
    "fema":                         ("taxation", 2),
    "foreign exchange":             ("taxation", 2),
    "forex":                        ("taxation", 2),
    "remittance":                   ("taxation", 2),
    "fdi":                          ("taxation", 2),
    "foreign direct investment":    ("taxation", 2),
    "sebi":                         ("taxation", 2),
    "securities":                   ("taxation", 1),
    "stock market":                 ("taxation", 2),
    "share market":                 ("taxation", 2),
    "stock exchange":               ("taxation", 2),
    "nse":                          ("taxation", 2),
    "bse":                          ("taxation", 2),
    "ipo":                          ("taxation", 2),
    "initial public offering":      ("taxation", 2),
    "insider trading":              ("taxation", 2),
    "banking":                      ("taxation", 1),
    "rbi":                          ("taxation", 2),
    "reserve bank":                 ("taxation", 2),
    "loan":                         ("taxation", 1),
    "interest rate":                ("taxation", 2),
    "bank":                         ("taxation", 1),
    "nbfc":                         ("taxation", 2),
    "financial":                    ("taxation", 1),

    # ═══════════════════════════════════════════════════════════════
    # PROPERTY
    # ═══════════════════════════════════════════════════════════════
    "property":                     ("property", 1),
    "immovable property":           ("property", 2),
    "movable property":             ("property", 2),
    "land":                         ("property", 1),
    "flat":                         ("property", 1),
    "apartment":                    ("property", 1),
    "house":                        ("property", 1),
    "building":                     ("property", 1),
    "premises":                     ("property", 1),
    "lease":                        ("property", 1),
    "lease deed":                   ("property", 2),
    "rent":                         ("property", 1),
    "rent control":                 ("property", 2),
    "landlord":                     ("property", 2),
    "tenant":                       ("property", 1),
    "tenancy":                      ("property", 2),
    "eviction":                     ("property", 2),
    "eviction notice":              ("property", 2),
    "notice to vacate":             ("property", 2),
    "sale deed":                    ("property", 2),
    "title deed":                   ("property", 2),
    "ownership":                    ("property", 1),
    "possession":                   ("property", 1),
    "encumbrance":                  ("property", 2),
    "rera":                         ("property", 2),
    "real estate":                  ("property", 2),
    "builder":                      ("property", 1),
    "developer":                    ("property", 1),
    "construction":                 ("property", 1),
    "flat buyer":                   ("property", 2),
    "allotment":                    ("property", 2),
    "easement":                     ("property", 2),
    "mortgage":                     ("property", 1),
    "mortgage deed":                ("property", 2),
    "charge":                       ("property", 1),
    "lien":                         ("property", 2),
    "registration":                 ("property", 1),
    "stamp duty":                   ("property", 2),
    "transfer of property":         ("property", 2),
    "conveyance":                   ("property", 2),
    "gift deed":                    ("property", 2),
    "partition":                    ("property", 2),
    "mutation":                     ("property", 2),
    "khata":                        ("property", 2),
    "patta":                        ("property", 2),
    "encroachment":                 ("property", 2),
    "adverse possession":           ("property", 2),
    "kerala rent":                  ("property", 2),
    "rent agreement":               ("property", 2),

    # ═══════════════════════════════════════════════════════════════
    # LABOUR
    # ═══════════════════════════════════════════════════════════════
    "employee":                     ("labour", 1),
    "employer":                     ("labour", 1),
    "worker":                       ("labour", 1),
    "workman":                      ("labour", 1),
    "labour":                       ("labour", 1),
    "labor":                        ("labour", 1),
    "employment":                   ("labour", 1),
    "job":                          ("labour", 1),
    "salary":                       ("labour", 1),
    "wages":                        ("labour", 1),
    "wage":                         ("labour", 1),
    "minimum wage":                 ("labour", 2),
    "minimum wages":                ("labour", 2),
    "floor wage":                   ("labour", 2),
    "overtime":                     ("labour", 2),
    "overtime pay":                 ("labour", 2),
    "termination":                  ("labour", 1),
    "termination of employment":    ("labour", 2),
    "dismissal":                    ("labour", 1),
    "retrenchment":                 ("labour", 2),
    "layoff":                       ("labour", 2),
    "voluntary retirement":         ("labour", 2),
    "vrs":                          ("labour", 2),
    "strike":                       ("labour", 1),
    "lockout":                      ("labour", 2),
    "trade union":                  ("labour", 2),
    "collective bargaining":        ("labour", 2),
    "industrial dispute":           ("labour", 2),
    "labour dispute":               ("labour", 2),
    "provident fund":               ("labour", 2),
    "epf":                          ("labour", 2),
    "pf":                           ("labour", 1),
    "employees provident fund":     ("labour", 2),
    "gratuity":                     ("labour", 2),
    "bonus":                        ("labour", 1),
    "esi":                          ("labour", 2),
    "esic":                         ("labour", 2),
    "maternity":                    ("labour", 2),
    "maternity leave":              ("labour", 2),
    "maternity benefit":            ("labour", 2),
    "paternity leave":              ("labour", 2),
    "leave":                        ("labour", 1),
    "paid leave":                   ("labour", 2),
    "working hours":                ("labour", 2),
    "shift":                        ("labour", 1),
    "weekly off":                   ("labour", 2),
    "sexual harassment":            ("labour", 2),
    "posh":                         ("labour", 2),
    "internal complaints committee": ("labour", 2),
    "icc":                          ("labour", 2),
    "workplace":                    ("labour", 1),
    "workplace harassment":         ("labour", 2),
    "occupational safety":          ("labour", 2),
    "osh":                          ("labour", 2),
    "workmen compensation":         ("labour", 2),
    "industrial accident":          ("labour", 2),
    "contract labour":              ("labour", 2),
    "contractor":                   ("labour", 1),
    "apprentice":                   ("labour", 2),
    "apprenticeship":               ("labour", 2),
    "social security":              ("labour", 2),
    "labour code":                  ("labour", 2),
    "industrial relations":         ("labour", 2),

    # ═══════════════════════════════════════════════════════════════
    # HEALTH
    # ═══════════════════════════════════════════════════════════════
    "food safety":                  ("health", 2),
    "food adulteration":            ("health", 2),
    "adulteration":                 ("health", 2),
    "fssai":                        ("health", 2),
    "food standard":                ("health", 2),
    "food label":                   ("health", 2),
    "food packaging":               ("health", 2),
    "restaurant":                   ("health", 1),
    "food business":                ("health", 2),
    "genetically modified":         ("health", 2),
    "gmo":                          ("health", 2),
    "organic food":                 ("health", 2),
    "drug":                         ("health", 1),
    "drugs":                        ("health", 1),
    "medicine":                     ("health", 1),
    "pharmaceutical":               ("health", 1),
    "hospital":                     ("health", 1),
    "clinic":                       ("health", 1),
    "doctor":                       ("health", 1),
    "physician":                    ("health", 1),
    "surgeon":                      ("health", 1),
    "patient":                      ("health", 1),
    "medical":                      ("health", 1),
    "clinical establishment":       ("health", 2),
    "clinical establishments":      ("health", 2),
    "healthcare":                   ("health", 1),
    "health":                       ("health", 1),
    "nutrition":                    ("health", 1),
    "cosmetic":                     ("health", 1),
    "cosmetics":                    ("health", 1),
    "ayurvedic":                    ("health", 1),
    "homeopathic":                  ("health", 1),
    "nutraceutical":                ("health", 2),
    "health supplement":            ("health", 2),
    "pharmacy":                     ("health", 1),
    "pharmacist":                   ("health", 1),
    "drug license":                 ("health", 2),
    "nursing home":                 ("health", 2),
    "right to education":           ("health", 2),
    "free education":               ("health", 2),
    "rte":                          ("health", 2),
    "compulsory education":         ("health", 2),
    "school":                       ("health", 1),
    "education":                    ("health", 1),
    "elementary education":         ("health", 2),
    "mid day meal":                 ("health", 2),
    "vaccination":                  ("health", 2),
    "medical negligence":           ("health", 2),

    # ═══════════════════════════════════════════════════════════════
    # ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════
    "environment":                  ("environment", 1),
    "environmental":                ("environment", 1),
    "pollution":                    ("environment", 2),
    "air pollution":                ("environment", 2),
    "water pollution":              ("environment", 2),
    "noise pollution":              ("environment", 2),
    "soil pollution":               ("environment", 2),
    "land pollution":               ("environment", 2),
    "emission":                     ("environment", 1),
    "effluent":                     ("environment", 2),
    "discharge":                    ("environment", 1),
    "hazardous waste":              ("environment", 2),
    "hazardous":                    ("environment", 1),
    "toxic":                        ("environment", 1),
    "waste disposal":               ("environment", 2),
    "solid waste":                  ("environment", 2),
    "e-waste":                      ("environment", 2),
    "forest":                       ("environment", 1),
    "deforestation":                ("environment", 2),
    "forest land":                  ("environment", 2),
    "tree cutting":                 ("environment", 2),
    "wildlife":                     ("environment", 2),
    "wild animal":                  ("environment", 2),
    "poaching":                     ("environment", 2),
    "biodiversity":                 ("environment", 2),
    "national park":                ("environment", 2),
    "sanctuary":                    ("environment", 2),
    "protected area":               ("environment", 2),
    "tiger reserve":                ("environment", 2),
    "mining":                       ("environment", 1),
    "quarrying":                    ("environment", 2),
    "eia":                          ("environment", 2),
    "environmental impact":         ("environment", 2),
    "epa":                          ("environment", 2),
    "climate":                      ("environment", 1),
    "carbon":                       ("environment", 1),
    "green":                        ("environment", 1),
    "pcb":                          ("environment", 2),
    "cpcb":                         ("environment", 2),
    "spcb":                         ("environment", 2),

    # ═══════════════════════════════════════════════════════════════
    # TECHNOLOGY
    # ═══════════════════════════════════════════════════════════════
    "cyber":                        ("technology", 2),
    "cybercrime":                   ("technology", 2),
    "cyber crime":                  ("technology", 2),
    "hacking":                      ("technology", 2),
    "unauthorised access":          ("technology", 2),
    "computer":                     ("technology", 1),
    "computer system":              ("technology", 2),
    "data breach":                  ("technology", 2),
    "data privacy":                 ("technology", 2),
    "privacy":                      ("technology", 1),
    "personal data":                ("technology", 2),
    "data protection":              ("technology", 2),
    "dpdp":                         ("technology", 2),
    "digital":                      ("technology", 1),
    "electronic":                   ("technology", 1),
    "electronic record":            ("technology", 2),
    "digital signature":            ("technology", 2),
    "internet":                     ("technology", 1),
    "social media":                 ("technology", 2),
    "online":                       ("technology", 1),
    "it act":                       ("technology", 2),
    "information technology":       ("technology", 2),
    "intermediary":                 ("technology", 2),
    "intermediary guidelines":      ("technology", 2),
    "encryption":                   ("technology", 2),
    "phishing":                     ("technology", 2),
    "online fraud":                 ("technology", 2),
    "cyberbullying":                ("technology", 2),
    "cyber harassment":             ("technology", 2),
    "stalking online":              ("technology", 2),
    "copyright":                    ("technology", 1),
    "copyright infringement":       ("technology", 2),
    "patent":                       ("technology", 1),
    "patent infringement":          ("technology", 2),
    "trademark":                    ("technology", 1),
    "trademark infringement":       ("technology", 2),
    "software":                     ("technology", 1),
    "source code":                  ("technology", 2),
    "domain name":                  ("technology", 2),
    "cyber fraud":                  ("technology", 2),
    "cyber security":               ("technology", 2),

    # ═══════════════════════════════════════════════════════════════
    # CIVIL
    # ═══════════════════════════════════════════════════════════════
    "constitution":                 ("civil", 2),
    "constitutional":               ("civil", 2),
    "fundamental rights":           ("civil", 2),
    "fundamental right":            ("civil", 2),
    "right to life":                ("civil", 2),
    "article 21":                   ("civil", 2),
    "article 14":                   ("civil", 2),
    "article 19":                   ("civil", 2),
    "directive principles":         ("civil", 2),
    "dpsp":                         ("civil", 2),
    "basic structure":              ("civil", 2),
    "constitutional amendment":     ("civil", 2),
    "motor vehicle":                ("civil", 2),
    "motor vehicles":               ("civil", 2),
    "road accident":                ("civil", 2),
    "vehicle accident":             ("civil", 2),
    "traffic":                      ("civil", 1),
    "driving license":              ("civil", 2),
    "vehicle registration":         ("civil", 2),
    "insurance":                    ("civil", 1),
    "motor insurance":              ("civil", 2),
    "accident compensation":        ("civil", 2),
    "hit and run":                  ("civil", 2),
    "mv act":                       ("civil", 2),
    "rti":                          ("civil", 2),
    "right to information":         ("civil", 2),
    "public information":           ("civil", 2),
    "pio":                          ("civil", 2),
    "cpio":                         ("civil", 2),
    "aadhaar":                      ("civil", 2),
    "uid":                          ("civil", 2),
    "consumer":                     ("civil", 1),
    "consumer protection":          ("civil", 2),
    "consumer forum":               ("civil", 2),
    "consumer court":               ("civil", 2),
    "deficiency in service":        ("civil", 2),
    "unfair trade practice":        ("civil", 2),
    "product liability":            ("civil", 2),
    "cpc":                          ("civil", 2),
    "civil procedure":              ("civil", 2),
    "civil court":                  ("civil", 2),
    "civil suit":                   ("civil", 2),
    "civil case":                   ("civil", 2),
    "decree":                       ("civil", 1),
    "decree execution":             ("civil", 2),
    "order":                        ("civil", 1),
    "writ":                         ("civil", 2),
    "writ petition":                ("civil", 2),
    "habeas corpus":                ("civil", 2),
    "mandamus":                     ("civil", 2),
    "certiorari":                   ("civil", 2),
    "prohibition":                  ("civil", 1),
    "quo warranto":                 ("civil", 2),
    "pil":                          ("civil", 2),
    "public interest litigation":   ("civil", 2),
    "public":                       ("civil", 1),
    "citizen":                      ("civil", 1),
    "citizenship":                  ("civil", 2),
    "election":                     ("civil", 1),
    "election law":                 ("civil", 2),
}

_SORTED_KEYWORDS = sorted(_KEYWORDS.items(), key=lambda x: -len(x[0]))
_CATEGORIES      = list(CATEGORY_DESCRIPTIONS.keys())
 
 
def _keyword_scores(query: str) -> dict[str, float]:
    """Raw keyword score per category."""
    q = query.lower()
    q = re.sub(r'[^\w\s]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
 
    scores: dict[str, float] = {}
    q_work = q
    for keyword, (category, weight) in _SORTED_KEYWORDS:
        if ' ' in keyword:
            if keyword in q_work:
                scores[category] = scores.get(category, 0.0) + weight
                q_work = q_work.replace(keyword, ' ', 1)
        else:
            if re.search(r'\b' + re.escape(keyword) + r'\b', q_work):
                scores[category] = scores.get(category, 0.0) + weight
    return scores
 
 
class CategoryDetector:
    """Hybrid keyword + semantic category detector."""
 
    def __init__(self) -> None:
        self._category_embeddings: Optional[dict[str, np.ndarray]] = None
        self._embedder_available = True
 
    def _ensure_embeddings(self) -> bool:
        """Lazy-load category embeddings on first semantic call."""
        if self._category_embeddings is not None:
            return True
        if not self._embedder_available:
            return False
        try:
            from rag.embedder import embedder
            texts = [CATEGORY_DESCRIPTIONS[cat] for cat in _CATEGORIES]
            vecs  = embedder.embed(texts)
            self._category_embeddings = {
                cat: np.array(vec, dtype=np.float32)
                for cat, vec in zip(_CATEGORIES, vecs)
            }
            print(f"[CategoryDetector] Category embeddings cached ({len(_CATEGORIES)} categories)")
            return True
        except Exception as e:
            print(f"[CategoryDetector] Semantic layer unavailable: {e} — using keyword only")
            self._embedder_available = False
            return False
 
    def _semantic_scores(self, query: str) -> dict[str, float]:
        """Cosine similarity of query embedding against category descriptions."""
        if not self._ensure_embeddings():
            return {}
        try:
            from rag.embedder import embedder
            q_vec = np.array(embedder.embed([query])[0], dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return {}
            q_vec = q_vec / q_norm
 
            scores: dict[str, float] = {}
            for cat, c_vec in self._category_embeddings.items():
                c_norm = np.linalg.norm(c_vec)
                if c_norm == 0:
                    scores[cat] = 0.0
                else:
                    scores[cat] = float(np.dot(q_vec, c_vec / c_norm))
            return scores
        except Exception:
            return {}
 
    def detect(self, query: str) -> tuple[Optional[str], float]:
        """
        Hybrid detection: keyword + semantic fusion.
 
        Returns (category, confidence) where confidence is the fused score
        share of the top category.
        """
        kw_raw  = _keyword_scores(query)
        sem_raw = self._semantic_scores(query)
 
        # Normalise keyword scores to [0, 1]
        kw_total = sum(kw_raw.values())
        kw_norm: dict[str, float] = (
            {cat: s / kw_total for cat, s in kw_raw.items()}
            if kw_total > 0 else {}
        )
 
        # Semantic scores are already cosine similarities ∈ [-1, 1]
        # Shift to [0, 1]: (s + 1) / 2, then normalise
        if sem_raw:
            sem_shifted = {cat: (s + 1.0) / 2.0 for cat, s in sem_raw.items()}
            sem_total   = sum(sem_shifted.values())
            sem_norm: dict[str, float] = (
                {cat: s / sem_total for cat, s in sem_shifted.items()}
                if sem_total > 0 else {}
            )
        else:
            sem_norm = {}
 
        # Fuse signals
        fused: dict[str, float] = {}
        for cat in _CATEGORIES:
            kw_score  = kw_norm.get(cat,  0.0)
            sem_score = sem_norm.get(cat, 0.0)
 
            if sem_norm:
                fused[cat] = KEYWORD_WEIGHT * kw_score + SEMANTIC_WEIGHT * sem_score
            else:
                # Semantic unavailable — keyword only
                fused[cat] = kw_score
 
        fused_total = sum(fused.values())
        if fused_total == 0:
            return None, 0.0
 
        top_cat    = max(fused, key=lambda c: fused[c])
        confidence = fused[top_cat] / fused_total
 
        if confidence < CONFIDENCE_MED:
            return None, round(confidence, 3)
 
        return top_cat, round(confidence, 3)
 
    def detect_all(self, query: str) -> dict[str, float]:
        """Return normalised fused scores for all categories. For debugging."""
        kw_raw  = _keyword_scores(query)
        sem_raw = self._semantic_scores(query)
 
        kw_total = sum(kw_raw.values())
        kw_norm  = {cat: s / kw_total for cat, s in kw_raw.items()} if kw_total > 0 else {}
 
        if sem_raw:
            sem_shifted = {cat: (s + 1.0) / 2.0 for cat, s in sem_raw.items()}
            sem_total   = sum(sem_shifted.values())
            sem_norm    = {cat: s / sem_total for cat, s in sem_shifted.items()} if sem_total > 0 else {}
        else:
            sem_norm = {}
 
        fused: dict[str, float] = {}
        for cat in _CATEGORIES:
            kw_score  = kw_norm.get(cat,  0.0)
            sem_score = sem_norm.get(cat, 0.0)
            fused[cat] = (
                KEYWORD_WEIGHT * kw_score + SEMANTIC_WEIGHT * sem_score
                if sem_norm else kw_score
            )
 
        fused_total = sum(fused.values())
        if fused_total == 0:
            return {}
        return {
            cat: round(s / fused_total, 3)
            for cat, s in sorted(fused.items(), key=lambda x: -x[1])
        }
 
 
# ── Singleton ─────────────────────────────────────────────────────────────────
category_detector = CategoryDetector()