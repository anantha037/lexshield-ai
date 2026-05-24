import random
import re

# ── Category index ────────────────────────────────────────────────────────────
CATEGORIES: dict[int, str] = {
    0: "rental_agreement",
    1: "fir",
    2: "court_notice",
    3: "employment_contract",
    4: "property_deed",
    5: "sc_judgment",
    6: "hc_judgment",
    7: "legal_notice",
}

# ── Name/place pools ──────────────────────────────────────────────────────────
MALE_NAMES   = ["Rajesh Kumar", "Mohammed Ibrahim", "Suresh Nair", "Arun Sharma",
                "Vikram Singh", "Gopal Menon", "Ravi Krishnan", "Sanjay Gupta",
                "Deepak Verma", "Anand Pillai", "Ramesh Patel", "Vijay Reddy"]
FEMALE_NAMES = ["Priya Sharma", "Anitha Krishnan", "Meera Nair", "Sunita Verma",
                "Lakshmi Devi", "Rekha Iyer", "Kavitha Menon", "Radha Pillai",
                "Sushma Rani", "Usha Kumari", "Geetha Nair", "Asha Varma"]
ALL_NAMES    = MALE_NAMES + FEMALE_NAMES

CITIES  = ["Thiruvananthapuram", "Ernakulam", "Kozhikode", "Thrissur", "Kollam",
           "Alappuzha", "Kottayam", "Palakkad", "Mumbai", "Delhi", "Bengaluru",
           "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad"]
STATES  = ["Kerala", "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana",
           "Uttar Pradesh", "Gujarat", "Rajasthan", "West Bengal"]
COURTS  = ["High Court of Kerala", "High Court of Bombay", "High Court of Madras",
           "High Court of Karnataka", "High Court of Allahabad",
           "High Court of Calcutta", "High Court of Delhi"]
AMOUNTS = ["₹10,000", "₹25,000", "₹50,000", "₹1,00,000", "₹2,50,000",
           "Rs. 15,000", "Rs. 75,000", "Rs. 5,00,000", "Rs. 10,00,000",
           "5 lakhs", "2 crores", "₹3,00,000"]
YEARS   = ["2020", "2021", "2022", "2023", "2024"]
MONTHS  = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]
DAYS    = ["1st", "5th", "10th", "15th", "20th", "25th", "28th"]

TECH_COMPANIES = ["TechVision", "InfoSoft", "DataSystems", "CodeBridge",
                  "Nexus Tech", "ClearLogic", "ByteWave", "PixelForge"]
EXEC_COMPANIES = ["Apex Industries", "Zenith Group", "Meridian Holdings",
                  "Pinnacle Corp", "Stratex Ltd", "Vantage Solutions"]
SMALL_COMPANIES= ["Sunrise Consulting", "Horizon Services", "Pioneer Solutions",
                  "BlueSky Advisors", "GreenLeaf LLP", "Crescent Associates"]

ROLES_TECH = ["Software Engineer", "Senior Developer", "Data Analyst",
              "DevOps Engineer", "Backend Engineer", "QA Engineer",
              "Frontend Developer", "Cloud Architect"]
ROLES_EXEC = ["Chief Operating Officer", "General Manager", "Vice President",
              "Director - Operations", "Head of Strategy", "CFO"]
ROLES_JUNIOR = ["HR Executive", "Accounts Manager", "Marketing Coordinator",
                "Legal Associate", "Operations Executive", "Sales Executive",
                "Admin Officer", "Business Analyst"]

def _rn() -> str:  return random.choice(ALL_NAMES)
def _rf() -> str:  return random.choice(FEMALE_NAMES)
def _rm() -> str:  return random.choice(MALE_NAMES)
def _rc() -> str:  return random.choice(CITIES)
def _rs() -> str:  return random.choice(STATES)
def _ra() -> str:  return random.choice(AMOUNTS)
def _ry() -> str:  return random.choice(YEARS)
def _rd() -> str:  return f"{random.choice(DAYS)} {random.choice(MONTHS)} {_ry()}"
def _rct()-> str:  return random.choice(COURTS)
def _rcn()-> str:  return f"W.P.(C) No. {random.randint(100,9999)}/{_ry()}"
def _rtc()-> str:  return random.choice(TECH_COMPANIES)
def _rec()-> str:  return random.choice(EXEC_COMPANIES)
def _rsc()-> str:  return random.choice(SMALL_COMPANIES)
def _rtr()-> str:  return random.choice(ROLES_TECH)
def _rer()-> str:  return random.choice(ROLES_EXEC)
def _rjr()-> str:  return random.choice(ROLES_JUNIOR)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 0 — RENTAL AGREEMENT  (4 structural variants)
# Variant D deliberately contains "incorporated under Companies Act" to give
# the model negative pressure — corporate language ≠ employment.
# ══════════════════════════════════════════════════════════════════════════════

def _rental_agreement_formal() -> str:
    landlord, tenant = _rn(), _rn()
    rent, deposit    = _ra(), _ra()
    city             = _rc()
    return f"""RENTAL AGREEMENT

This Rental Agreement is entered into on {_rd()} between {landlord},
residing at {random.randint(1,100)}, {random.choice(['MG Road','Beach Road','Gandhi Nagar'])},
{city}, {_rs()}, hereinafter referred to as the "Landlord",

AND

{tenant}, hereinafter referred to as the "Tenant".

WHEREAS the Landlord is the owner of the premises at Plot No. {random.randint(1,500)},
{random.choice(['Green Valley Layout','Sunrise Apartments','Palm Grove'])}, {city}.

1. MONTHLY RENT: {rent} payable on or before the 5th of each month.
2. SECURITY DEPOSIT: Refundable deposit of {deposit} paid by Tenant.
3. TENURE: 11 months commencing {_rd()}.
4. USE OF PREMISES: Residential purposes only.
5. MAINTENANCE: Tenant to keep premises in good condition.
6. UTILITIES: Electricity and water charges borne by Tenant.
7. SUBLETTING: Subletting without written consent of Landlord is prohibited.
8. TERMINATION: One month written notice required by either party.
9. GOVERNING LAW: Kerala Buildings (Lease and Rent Control) Act.

{landlord}                      {tenant}
(Landlord)                      (Tenant)
Witness 1: {_rn()}
Witness 2: {_rn()}
"""

def _rental_agreement_brief() -> str:
    landlord, tenant = _rn(), _rn()
    city             = _rc()
    return f"""LEASE AGREEMENT — BRIEF FORM

Date: {_rd()}
Lessor: {landlord}, {city}
Lessee: {tenant}, {_rc()}

Property: Flat No. {random.randint(1,50)}, {random.choice(['Horizon Towers','Skyline Residency','Metro Homes'])}, {city}
Monthly Rent: {_ra()} (inclusive of maintenance)
Lease Period: {random.choice(['6 months','11 months','12 months'])} from {_rd()}
Deposit Collected: {_ra()} (refundable)

Special Conditions:
- No pets allowed on the premises.
- Power backup charges at actuals.
- Lock-in period of {random.choice(['3','4','5'])} months; early exit forfeits deposit.
- Rent to be revised by {random.randint(5,15)}% on renewal.

The parties agree to be bound by the terms of this agreement and the
provisions of the Rent Control Act applicable in the state of {_rs()}.

Signed:
{landlord} (Lessor)          {tenant} (Lessee)
"""

def _rental_agreement_commercial() -> str:
    landlord = _rn()
    tenant   = f"M/S {random.choice(['Sunrise Traders','Metro Retail','City Mart'])} {random.choice(['Pvt Ltd','Enterprises'])}"
    city     = _rc()
    return f"""COMMERCIAL PREMISES RENTAL AGREEMENT

This agreement executed on {_rd()} at {city}.

OWNER: {landlord}, {city}, {_rs()} (hereinafter "Owner")
TENANT: {tenant}, {city} (hereinafter "Tenant")

Premises: Shop No. {random.randint(1,30)}, Ground Floor,
{random.choice(['Commercial Complex','Shopping Arcade','Business Centre'])}, {city}
Area: {random.randint(200,2000)} sq. ft.
Purpose: {random.choice(['Retail shop','Office space','Showroom','Warehouse'])}
Monthly Rent: {_ra()} + GST as applicable
Security Deposit: {_ra()} (interest-free, refundable)
Term: {random.randint(1,5)} year(s) from {_rd()}
Lock-in: {random.randint(6,18)} months

Tenant to obtain trade licence and all statutory permissions.
Owner not liable for any business losses of Tenant.
Governed by Indian Contract Act, 1872 and applicable Rent Control laws.

{landlord}                              {tenant}
(Owner)                                 (Tenant)
"""

def _rental_agreement_risky() -> str:
    """Variant D — RISK CLAUSE variant: mirrors vocabulary of real-world
    tenant-unfriendly agreements. Uses NON-REFUNDABLE, AUTO RENEWAL, ENTRY,
    SUMMARILY, exclusive jurisdiction — the exact tokens the test doc contains."""
    landlord, tenant = _rn(), _rn()
    city = _rc()
    return f"""RENTAL AGREEMENT

This agreement is between {landlord} (Landlord) and {tenant} (Tenant)
for premises at {random.choice(['Green Valley Layout','Palm Grove','Sunrise Apartments'])},
{city}, {_rs()}.

1. RENT: Monthly rent of {_ra()} shall be paid by {random.choice(['5th','7th','10th'])} of every month.

2. DEPOSIT: A security deposit of {_ra()} is paid. The deposit shall be
   NON-REFUNDABLE and shall be forfeited in case of early termination.

3. RENT INCREASE: Rent shall increase by {random.randint(10,25)}% every year automatically.

4. ENTRY: The Landlord may enter the premises at any time without prior
   notice to inspect the property condition.

5. SUBLETTING: The Tenant shall not sublet or assign the premises under
   any circumstances whatsoever.

6. DISPUTES: All disputes shall be subject to exclusive jurisdiction of
   courts in {_rc()} only.

7. AUTO RENEWAL: This agreement shall be automatically renewed for another
   11 months unless terminated {random.randint(15,60)} days prior.

8. TERMINATION: The landlord shall have the right to terminate this agreement
   summarily without cause or notice at landlord's discretion.

Signed at {city} on {_rd()}.
{landlord} (Landlord)    {tenant} (Tenant)
"""

def _rental_agreement_corporate_tenant() -> str:
    """Variant E — NEGATIVE PRESSURE: corporate entity as tenant.
    Contains 'incorporated under Companies Act 2013' and 'registered office'
    but is still a LEASE, not employment. Forces model to learn context."""
    landlord = _rn()
    company  = f"M/S {_rtc()} {random.choice(['Pvt Ltd','Private Limited','Technologies Ltd'])}"
    city     = _rc()
    return f"""LEASE DEED

This Lease Deed is executed on {_rd()} at {city} between:

LESSOR: {landlord}, {random.randint(1,200)}, {city}, {_rs()} ("Lessor")

AND

LESSEE: {company}, a company incorporated under the Companies Act, 2013,
having its registered office at {random.randint(1,10)}th Floor,
{random.choice(['Tech Park','Business Hub','Corporate Tower'])}, {city}, {_rs()} ("Lessee")

The Lessor hereby leases to the Lessee the premises described below:

LEASED PREMISES:
Floor No. {random.randint(1,15)}, {random.choice(['Infopark Tower','Technopark Block','Cyberpark Wing'])},
{city}, {_rs()}, admeasuring {random.randint(500,5000)} sq. ft.

RENT: {_ra()} per month payable in advance on the 1st of each month.
LEASE PERIOD: {random.randint(1,5)} year(s) from {_rd()}, renewable.
SECURITY DEPOSIT: {_ra()} (refundable, interest-free).
PURPOSE: Office / IT use only. Subletting strictly prohibited.

In case of default in rent for 2 consecutive months, Lessor may terminate
this lease with 30 days notice. Governed by Rent Control Act.

{landlord} (Lessor)            Authorised Signatory, {company} (Lessee)
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 1 — FIR  (3 structural variants)
# ══════════════════════════════════════════════════════════════════════════════

def _fir_standard() -> str:
    officer   = _rm()
    accused   = _rn()
    informant = _rn()
    city      = _rc()
    secs      = random.choice([
        "Section 420 IPC", "Section 302 IPC", "Section 379 IPC",
        "Section 498A IPC", "Sections 420 and 406 IPC", "Section 354 IPC",
    ])
    ps = f"{city} {random.choice(['North','South','East','West','Central'])} Police Station"
    return f"""FIRST INFORMATION REPORT (FIR)
Under Section 154 of the Code of Criminal Procedure

FIR No.: {random.randint(100,999)}/{_ry()}
Date: {_rd()}
Police Station: {ps}
District: {city}

1. Type of Information: Written
2. Complainant: {informant}, S/o {_rm()}, {random.randint(1,200)}, {_rc()}, {_rs()}
3. Accused: {accused}, {random.randint(1,200)}, {city}
4. Offence: {secs}
5. Date of Occurrence: {_rd()}
6. Place of Occurrence: {_rc()}, {_rs()}

BRIEF FACTS:
The complainant states that on {_rd()}, the accused {accused} committed
{random.choice(['cheating','theft','assault','fraud','criminal breach of trust'])}
causing loss of {_ra()}. Immediate action requested under CrPC.

Action: FIR registered. Investigation commenced.

Signature: {officer} ({random.choice(['Sub-Inspector','Inspector','DSP'])})
Station: {ps}
"""

def _fir_cybercrime() -> str:
    informant = _rn()
    city      = _rc()
    return f"""FIRST INFORMATION REPORT
Cyber Crime Police Station — {city}
FIR No.: CY-{random.randint(10,99)}/{_ry()}    Date: {_rd()}

Complainant: {informant}, {_rc()}, {_rs()}
Contact: 98{random.randint(10000000,99999999)}

Offence Under: Section 66C and 66D IT Act 2000, Section 419 IPC
Date of Incident: {_rd()}

COMPLAINT DETAILS:
The complainant received a call from an unknown number claiming to be
{random.choice(['bank official','KYC verification officer','police officer','insurance agent'])}.
The accused fraudulently obtained OTP and transferred {_ra()} from
the complainant's account ending {random.randint(1000,9999)}.

Online transaction reference: TXN{random.randint(100000000,999999999)}
Fraudulent account: {_rc()} bank branch.

Relief Sought: Recovery of {_ra()}, arrest of accused.

Signature of Complainant: {informant}
Received by: Sub-Inspector, Cyber Crime Cell, {city}
"""

def _fir_domestic() -> str:
    complainant = _rf()
    accused     = _rm()
    city        = _rc()
    return f"""FIRST INFORMATION REPORT
Police Station: {city} Women's Police Station
FIR No.: {random.randint(50,300)}/{_ry()}
Date: {_rd()}

Complainant: {complainant} (W/o {accused}), {random.randint(1,200)}, {city}
Accused: {accused} and family members, {city}, {_rs()}

Offences: Section 498A IPC (cruelty by husband/relatives),
          Section 406 IPC (criminal breach of trust — dowry articles),
          Section 3 and 4 of the Dowry Prohibition Act, 1961

STATEMENT:
The complainant states that she was subjected to physical and mental
cruelty by her husband and in-laws on account of dowry demands of {_ra()}.
She was beaten on {_rd()} and driven out of the matrimonial home.
Dowry articles worth {_ra()} have been retained by the accused.

Medical examination conducted at: {city} Government Hospital
Doctor: Dr. {_rm()}

Officer-in-Charge: {_rm()}, {city} Women's Police Station
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 2 — COURT NOTICE  (3 structural variants)
# ══════════════════════════════════════════════════════════════════════════════

def _court_notice_writ() -> str:
    petitioner = _rn()
    court      = _rct()
    case_no    = _rcn()
    sec        = random.choice(["Article 226", "Article 227", "Section 482 CrPC"])
    return f"""IN THE {court.upper()}

{case_no}

{petitioner} ... Petitioner

VERSUS

{random.choice(['State of Kerala','State of Maharashtra','Union of India'])} ... Respondent

NOTICE

WHEREAS the above petition has been filed under {sec},
challenging the order dated {_rd()} passed by the {random.choice(['District Collector','Magistrate'])}.

AND WHEREAS the Petitioner alleges violation of rights under Article 21
and Article 14 of the Constitution of India.

YOU ARE HEREBY DIRECTED to appear before this Hon'ble Court on {_rd()}
at {random.choice(['10:30 AM','11:00 AM','2:00 PM'])} in Court Hall No. {random.randint(1,20)}.

FAILURE to appear may result in ex-parte proceedings.

By Order of the Court,
REGISTRAR GENERAL
{court}
Date: {_rd()}
"""

def _court_notice_summons() -> str:
    plaintiff  = _rn()
    defendant  = _rn()
    city       = _rc()
    court_type = random.choice(["District Court","Civil Judge Court","Munsiff Court"])
    case_no    = f"O.S. No. {random.randint(100,999)}/{_ry()}"
    return f"""IN THE {court_type.upper()}, {city.upper()}

{case_no}

{plaintiff}                ... Plaintiff
versus
{defendant}                ... Defendant

SUMMONS TO DEFENDANT

To: {defendant}, {random.randint(1,200)}, {_rc()}, {_rs()}

WHEREAS a suit has been filed against you by the Plaintiff for recovery
of a sum of {_ra()} on account of {random.choice(['money due','breach of contract','damages'])}.

YOU ARE HEREBY SUMMONED to appear in person or through a duly authorised
pleader before this Court on {_rd()} at 10:30 AM.

TAKE NOTICE that if you fail to appear on the day fixed, the suit will
be heard and decided in your absence (ex-parte).

ISSUED under the hand and seal of this Court.

Date: {_rd()}
(Seal)
Civil Judge, {city}
"""

def _court_notice_contempt() -> str:
    respondent = _rn()
    court      = _rct()
    case_no    = f"Cont.Cas.(C) No. {random.randint(10,999)}/{_ry()}"
    return f"""IN THE {court.upper()}

{case_no}

IN THE MATTER OF: Contempt of Court

Contemnor: {respondent}

NOTICE TO SHOW CAUSE

WHEREAS this Court had passed an order on {_rd()} directing {respondent}
to {random.choice(['pay the decretal amount','vacate the premises','restore possession','comply with the directions'])},

AND WHEREAS the said order has not been complied with till date,

YOU {respondent} are hereby called upon to SHOW CAUSE on {_rd()} as to
why you should not be committed for contempt of court under the Contempt
of Courts Act, 1971 and sentenced to imprisonment and/or fine.

You may appear in person or through Advocate on the said date.
Non-appearance will be treated as contempt.

Registrar, {court}
Date: {_rd()}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 3 — EMPLOYMENT CONTRACT  (6 structural variants)
#
# KEY FIXES vs previous version:
# 1. Discriminative tokens (CTC, salary, designation, probation, PF, ESI)
#    appear in the FIRST 150 CHARACTERS of every variant — before any
#    shared corporate boilerplate.
# 2. Short-doc variants (C, D, E) are ≤ 300 chars of real content so the
#    model learns employment even from truncated/partially-uploaded docs.
# 3. No IPC citations — labour statutes only.
# 4. All variants contain at least one of: CTC / salary / designation /
#    probation / ESI / PF / gratuity in position [0:200].
# ══════════════════════════════════════════════════════════════════════════════

def _employment_contract_tech() -> str:
    """Variant A: Full tech sector agreement — discriminative tokens in line 1."""
    company  = f"{_rtc()} {random.choice(['Pvt Ltd','Private Limited','Technologies Ltd'])}"
    employee = _rn()
    city     = _rc()
    role     = _rtr()
    salary   = _ra()
    return f"""EMPLOYMENT AGREEMENT — {role.upper()} — CTC: {salary}

Employer: {company}, incorporated under Companies Act, 2013,
{random.choice(['Infopark','Technopark','Cyberpark'])}, {city}, {_rs()}

Employee: {employee}, {city}, {_rs()}

DESIGNATION: {role}
DATE OF JOINING: {_rd()}
MONTHLY CTC: {salary} (Cost to Company; breakup in Annexure A)
PROBATION PERIOD: 6 months; performance review at end of probation.
NOTICE PERIOD: 60 days written notice by either party.
WORK LOCATION: {city} (transferable with 30-day notice)
LEAVES: 18 earned leaves + 12 casual leaves per annum.
STATUTORY BENEFITS: PF deducted at 12% of basic; ESI applicable;
Gratuity payable under Payment of Gratuity Act, 1972.
Governed by Code on Wages, 2019 and Industrial Disputes Act, 1947.

CONFIDENTIALITY: Employee shall not disclose proprietary information.
NON-COMPETE: Employee shall not join competitors for 1 year post-separation.
INTELLECTUAL PROPERTY: All work product vests exclusively with Employer.

For {company}: {_rm()} (HR Head)          {employee} (Employee)
Date: {_rd()}
"""

def _employment_contract_executive() -> str:
    """Variant B: Senior executive — salary/designation first line."""
    company  = f"{_rec()} {random.choice(['Ltd','Group Ltd','Holdings Pvt Ltd'])}"
    employee = _rn()
    role     = _rer()
    city     = _rc()
    ctc      = _ra()
    return f"""EMPLOYMENT AGREEMENT — EXECUTIVE — DESIGNATION: {role} — FIXED CTC: {ctc}

Employer: {company}, {city}, {_rs()}
Employee: {employee}, {city}

TERMS OF EXECUTIVE EMPLOYMENT:

1. DESIGNATION      : {role}
2. DATE OF JOINING  : {_rd()}
3. FIXED MONTHLY CTC: {ctc}
4. PERFORMANCE BONUS: Up to {random.randint(10,30)}% of annual CTC
5. STOCK OPTIONS    : {random.randint(500,5000)} ESOPs vesting over 4 years
6. PROBATION        : Not applicable (executive appointment)
7. NOTICE PERIOD    : 90 days (either party); waivable by Employer
8. PF & GRATUITY    : As per EPF Act, 1952; Payment of Gratuity Act, 1972
9. MEDICAL INSURANCE: Self + family covered under group policy
10. HRA             : {random.randint(20,40)}% of basic salary
11. NON-SOLICITATION: No solicitation of clients/employees for 2 years post-exit
12. FIDUCIARY DUTY  : Employee owes fiduciary duty to the Board
13. POSH COMPLIANCE : Mandatory under POSH Act, 2013

Subject to background verification.

{_rm()} (HR Director, {company})     {employee} (Employee)
"""

def _employment_contract_offer() -> str:
    """Variant C: Offer letter — short, salary/role in first line."""
    company  = f"{_rsc()} LLP"
    employee = _rn()
    role     = _rjr()
    city     = _rc()
    ctc      = _ra()
    return f"""EMPLOYMENT OFFER LETTER — {role.upper()} — SALARY: {ctc} per annum

Ref: HR/{random.randint(100,999)}/{_ry()}     Date: {_rd()}
To: {employee}, {city}, {_rs()}

Dear {employee.split()[0]},

We are pleased to offer you employment as {role} at {company}, {city}.

EMPLOYMENT DETAILS:
- Designation       : {role}
- Date of Joining   : {_rd()}
- Monthly CTC       : {ctc} per annum (breakup attached)
- Probation Period  : 3 months from date of joining
- Notice Period     : 30 days during probation; 60 days post-confirmation
- Reporting To      : {_rm()} ({random.choice(['Senior Manager','Department Head','Team Lead'])})
- Work Location     : {city}, {_rs()}

STATUTORY DEDUCTIONS:
PF at 12% of basic, Professional Tax, and TDS as per Income Tax Act, 1961.
ESI applicable if gross salary ≤ ₹21,000/month.

LEAVE ENTITLEMENT: Casual Leave 12 days | Earned Leave 15 days | Sick Leave 7 days.
Governed by Shops and Establishments Act, {_rs()}.

Kindly sign and return by {_rd()}.

{_rm()} (HR Manager, {company})       {employee} (Acceptance)
"""

def _employment_contract_short_offer() -> str:
    """Variant D: MINIMAL offer letter — tests short-doc classification.
    Entire doc fits in ~250 chars; CTC and designation appear immediately."""
    company  = f"{_rtc()} {random.choice(['Pvt Ltd','Ltd'])}"
    employee = _rn()
    role     = random.choice(ROLES_TECH + ROLES_JUNIOR)
    city     = _rc()
    return f"""OFFER OF EMPLOYMENT

To: {employee}
Designation: {role}
Employer: {company}, {city}
CTC: {_ra()} per annum
Probation: {random.choice(['3 months','6 months'])}
Date of Joining: {_rd()}
PF and ESI applicable as per law.
Notice Period: {random.choice(['30 days','60 days'])}

Please confirm acceptance within 7 days.

{_rm()} (HR, {company})
"""

def _employment_contract_appointment() -> str:
    """Variant E: Appointment letter style — brief, salary first."""
    company  = f"{_rsc()}"
    employee = _rn()
    role     = _rjr()
    city     = _rc()
    return f"""APPOINTMENT LETTER

Date: {_rd()}
Employee: {employee}
Designation: {role}
Department: {random.choice(['Operations','Finance','Legal','HR','Sales'])}
CTC (annual): {_ra()}
Basic Salary: {_ra()} per month
HRA: {random.randint(20,40)}% of basic
Probation Period: {random.choice(['3 months','6 months'])} from {_rd()}
Confirmation subject to satisfactory performance review.
Provident Fund: 12% employee contribution deducted monthly.
Gratuity: As per Payment of Gratuity Act, 1972.
Working Hours: {random.choice(['9 AM – 6 PM','8:30 AM – 5:30 PM'])}, Monday–Saturday.
Leave: {random.randint(12,18)} days earned leave per annum.

This appointment is subject to the Company's HR policy and Code of Conduct.

{_rm()} (Authorised Signatory, {company})
"""

def _employment_contract_internship() -> str:
    """Variant F: Internship / trainee agreement — still employment, not rental."""
    company  = f"{_rtc()} {random.choice(['Pvt Ltd','Ltd'])}"
    intern   = _rn()
    city     = _rc()
    return f"""INTERNSHIP / TRAINEE EMPLOYMENT AGREEMENT

Employer: {company}, {city}, {_rs()}
Intern/Trainee: {intern}, {city}

ROLE        : {random.choice(['Software Trainee','Legal Intern','HR Trainee','Marketing Intern'])}
STIPEND     : {_ra()} per month (taxable as salary)
DURATION    : {random.randint(2,6)} months from {_rd()}
PROBATION   : Not applicable (fixed-term internship)
WORKING HOURS: {random.choice(['9 AM – 6 PM','10 AM – 7 PM'])}, {random.randint(5,6)} days/week
PF / ESI    : Applicable if stipend qualifies under Code on Wages, 2019

Intern shall maintain confidentiality of all proprietary information.
All work product created during internship shall vest with {company}.
Either party may terminate with 7 days written notice.

{_rm()} (HR, {company})           {intern} (Intern)
Date: {_rd()}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 4 — PROPERTY DEED  (3 structural variants)
# ══════════════════════════════════════════════════════════════════════════════

def _property_deed_sale() -> str:
    seller = _rn(); buyer = _rn(); city = _rc()
    return f"""SALE DEED

Executed on {_rd()} at {city}.

VENDOR: {seller}, {random.randint(1,200)}, {city}, {_rs()} ("Vendor")
PURCHASER: {buyer}, {random.randint(1,200)}, {_rc()}, {_rs()} ("Purchaser")

SCHEDULE OF PROPERTY:
Plot No. {random.randint(1,500)}, Survey No. {random.randint(100,999)},
{random.choice(['Green Valley Layout','Palm Grove Colony'])}, {city}, {_rs()},
measuring {random.randint(500,5000)} sq. ft.

CONSIDERATION: {_ra()} paid in full by Purchaser to Vendor.

The Vendor conveys all rights, title and interest to the Purchaser absolutely
under the Transfer of Property Act, 1882 and Registration Act, 1908.
The property is free from all encumbrances.

Stamp Duty Paid: {_ra()}   Sub-Registrar Office: {city}

{seller} (Vendor)          {buyer} (Purchaser)
Witnesses: 1. {_rn()}   2. {_rn()}
"""

def _property_deed_gift() -> str:
    donor  = _rn(); donee = _rn(); city = _rc()
    return f"""DEED OF GIFT

This Gift Deed is executed on {_rd()} at {city}.

DONOR: {donor}, {city}, {_rs()}
DONEE: {donee} ({random.choice(['son','daughter','nephew','niece'])} of Donor), {_rc()}, {_rs()}

OUT OF NATURAL LOVE AND AFFECTION, the Donor hereby gifts, transfers
and conveys the following property to the Donee:

PROPERTY: House/Flat No. {random.randint(1,100)},
{random.choice(['Coconut Grove','Riverside Colony','Sun City'])},
{city}, {_rs()}, Survey No. {random.randint(100,999)},
area {random.randint(800,3000)} sq. ft.

The gift is made voluntarily, without any monetary consideration,
out of natural love and affection for the Donee.

The Donee accepts the gift of the said property.
Governed by Transfer of Property Act, 1882 (Section 122).

Registered at Sub-Registrar's Office, {city}, on {_rd()}.
Stamp Duty: {_ra()}

{donor} (Donor)            {donee} (Donee)
Witnesses: 1. {_rn()}   2. {_rn()}
"""

def _property_deed_mortgage() -> str:
    mortgagor = _rn(); city = _rc()
    bank      = random.choice(["State Bank of India","Bank of Baroda","Federal Bank","HDFC Bank","Canara Bank"])
    return f"""MORTGAGE DEED

This Mortgage Deed is executed on {_rd()} at {city}.

MORTGAGOR: {mortgagor}, {random.randint(1,200)}, {city}, {_rs()}
MORTGAGEE: {bank}, {city} Branch, {_rs()}

LOAN AMOUNT: {_ra()} (Principal)
RATE OF INTEREST: {random.choice(['8.5%','9%','9.5%','10%'])} per annum (floating)
REPAYMENT: {random.randint(60,300)} monthly EMIs

MORTGAGED PROPERTY:
Plot/House No. {random.randint(1,500)}, Survey No. {random.randint(100,999)},
{city}, {_rs()}, area {random.randint(800,4000)} sq. ft.
Property value as per bank valuation: {_ra()}

The Mortgagor hereby creates an equitable mortgage over the above property
as security for the loan availed from the Mortgagee Bank.
In case of default, the Bank may invoke its rights under the SARFAESI Act, 2002.

Sub-Registrar Office: {city}   Registration Fee: {_ra()}

{mortgagor} (Mortgagor)        Authorised Signatory, {bank}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 5 — SC JUDGMENT  (3 structural variants)
# ══════════════════════════════════════════════════════════════════════════════

def _sc_judgment_criminal() -> str:
    appellant = _rn(); respondent = _rn()
    j1 = f"Justice {_rm()}"; j2 = f"Justice {_rm()}"
    secs = random.choice(["Section 302 IPC", "Section 376 IPC", "Section 420 IPC"])
    return f"""IN THE SUPREME COURT OF INDIA
Criminal Appellate Jurisdiction
Criminal Appeal No. {random.randint(1000,9999)}/{_ry()}

{appellant}                    ... Appellant
VERSUS
{respondent}                   ... Respondent

CORAM: {j1}  {j2}

JUDGMENT

{j1}, J.:
This appeal is directed against the judgment of the {_rct()} dated {_rd()}
convicting the Appellant under {secs} and sentencing him to
{random.choice(['life imprisonment','7 years rigorous imprisonment','5 years rigorous imprisonment'])}.

HELD:
After careful consideration of evidence and submissions of learned counsel,
this Court finds that the prosecution has {random.choice(['proved the charge beyond reasonable doubt',
'failed to establish the charge beyond reasonable doubt'])}.

The appeal is {random.choice(['dismissed','allowed'])}.
The conviction {random.choice(['upheld','set aside'])}.

Costs: {_ra()}. Bail bonds cancelled.

....{j1}          ....{j2}
New Delhi, {_rd()}
"""

def _sc_judgment_constitutional() -> str:
    appellant = _rn(); j1 = f"Justice {_rm()}"; j2 = f"Justice {_rm()}"
    return f"""IN THE SUPREME COURT OF INDIA
Civil Appellate Jurisdiction
Civil Appeal No. {random.randint(1000,9999)}/{_ry()}

{appellant}                    ... Appellant
Union of India & Ors.          ... Respondents

CORAM: {j1}  {j2}

J U D G M E N T

The core question in this appeal is whether the impugned legislation
violates Articles 14 and 21 of the Constitution of India.

BACKGROUND:
The High Court upheld the constitutionality of the impugned order.
The Appellant has challenged the same before this Court.

ANALYSIS:
The right to {random.choice(['livelihood','equality','free speech','fair trial'])} is a
fundamental right guaranteed under Part III of the Constitution.
The impugned order is {random.choice(['arbitrary and unreasonable','violative of natural justice',
'in excess of statutory authority'])}.

CONCLUSION:
The appeal is {random.choice(['allowed','dismissed'])}.
The impugned order stands {random.choice(['quashed','modified','upheld'])}.
Writ of {random.choice(['Mandamus','Certiorari','Prohibition'])} issued.

....{j1}          ....{j2}
Supreme Court of India, {_rd()}
"""

def _sc_judgment_civil() -> str:
    plaintiff = _rn(); defendant = _rn()
    j1 = f"Justice {_rm()}"
    return f"""IN THE SUPREME COURT OF INDIA
Civil Appellate Jurisdiction
Civil Appeal No. {random.randint(500,5000)}/{_ry()}

{plaintiff}                   ... Plaintiff-Appellant
VERSUS
{defendant}                   ... Defendant-Respondent

BEFORE: {j1}

The dispute relates to {random.choice(['title to immovable property','succession rights',
'specific performance of contract','partition of joint family property'])}.

FACTS:
The Trial Court decreed the suit in favour of Plaintiff.
The High Court reversed the decree by judgment dated {_rd()}.
Hence this appeal by Special Leave under Article 136.

ANALYSIS:
This Court has examined the evidence on record.
The High Court {random.choice(['correctly appreciated','misread','overlooked'])} the evidence.

ORDER:
Civil Appeal {random.choice(['allowed','dismissed'])}.
Decree of Trial Court {random.choice(['restored','set aside'])}.
Costs: {_ra()}.

{j1}
Supreme Court of India, {_rd()}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 6 — HC JUDGMENT  (3 structural variants)
# ══════════════════════════════════════════════════════════════════════════════

def _hc_judgment_writ() -> str:
    petitioner = _rn(); court = _rct(); judge = f"Justice {_rm()}"
    case_no    = _rcn()
    sec        = random.choice(["Article 226", "Article 227"])
    return f"""IN THE {court.upper()}

{case_no}

{petitioner}                     ... Petitioner
VERSUS
{random.choice(['State of Kerala','Union of India','Government of Maharashtra'])}  ... Respondent

BEFORE: {judge}
JUDGMENT DATED {_rd()}

This petition under {sec} of the Constitution challenges the order
dated {_rd()} of the learned {random.choice(['District Collector','Revenue Divisional Officer'])}.

ANALYSIS:
This Court finds the impugned order {random.choice(['violative of principles of natural justice',
'without jurisdiction','contrary to settled law','unsustainable'])}.

ORDER:
Petition is {random.choice(['allowed','partly allowed','dismissed'])}.
Impugned order is {random.choice(['set aside','quashed','modified'])}.
Costs: {_ra()}.

{judge}
{court}
"""

def _hc_judgment_bail() -> str:
    applicant = _rn(); court = _rct(); judge = f"Justice {_rm()}"
    case_no   = f"Bail Appl. No. {random.randint(100,9999)}/{_ry()}"
    offence   = random.choice(["Section 420 IPC","Section 302 IPC","Section 376 IPC","NDPS Act"])
    return f"""IN THE {court.upper()}

{case_no}

{applicant}                 ... Applicant/Accused
VERSUS
State                       ... Respondent

BEFORE: {judge}

ORDER ON BAIL APPLICATION

The applicant is accused of offence under {offence} and is in judicial
custody since {_rd()}.

Heard learned counsel for the applicant and learned Public Prosecutor.

CONSIDERATIONS:
- Gravity of offence: {random.choice(['serious','heinous','economic'])}
- Antecedents of accused: {random.choice(['clean','two prior FIRs'])}
- Flight risk: {random.choice(['low','cannot be ruled out'])}
- Likelihood of tampering with evidence: {random.choice(['present','not established'])}

ORDER:
Bail application is {random.choice(['allowed','rejected'])}.
{random.choice([f'Bail granted on surety of {_ra()}.', 'Accused to remain in judicial custody.'])}

{judge}
{court}, {_rd()}
"""

def _hc_judgment_revision() -> str:
    petitioner = _rn(); court = _rct(); judge = f"Justice {_rm()}"
    case_no    = f"Crl. Rev. Pet. No. {random.randint(100,9999)}/{_ry()}"
    return f"""IN THE {court.upper()}

{case_no}

{petitioner}              ... Revision Petitioner
VERSUS
State                     ... Respondent

BEFORE: {judge}

JUDGMENT

This Criminal Revision Petition is filed under Section 397 read with
Section 401 of the Code of Criminal Procedure, challenging the order
of the {random.choice(['Sessions Court','Magistrate Court'])} dated {_rd()}.

The learned Sessions Judge {random.choice(['convicted','acquitted'])} the accused and
sentenced him to {random.choice(['2 years SI','3 years RI','fine of ' + _ra()])}.

FINDINGS:
Perused the lower court records. The appreciation of evidence by the
lower court is {random.choice(['correct and requires no interference',
'perverse and requires to be set aside'])}.

ORDER:
Revision petition {random.choice(['dismissed','allowed'])}.
Lower court order {random.choice(['confirmed','reversed'])}.

{judge}
{court}, {_rd()}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 7 — LEGAL NOTICE  (3 structural variants)
# ══════════════════════════════════════════════════════════════════════════════

def _legal_notice_cheque() -> str:
    sender   = _rn(); advocate = _rn(); recipient = _rn(); city = _rc()
    return f"""LEGAL NOTICE UNDER SECTION 138 OF THE NEGOTIABLE INSTRUMENTS ACT, 1881

Sent by: Advocate {advocate} on behalf of {sender}
Date: {_rd()}

TO: {recipient}, {random.randint(1,200)}, {_rc()}, {_rs()}

My client {sender} states as follows:

1. You issued Cheque No. {random.randint(100000,999999)} dated {_rd()} for {_ra()}
   drawn on {random.choice(['State Bank of India','HDFC Bank','Federal Bank','Canara Bank'])},
   {city} Branch, bearing Account No. {random.randint(10000000000,99999999999)}.

2. The cheque was presented for encashment and was dishonoured with the
   remark "{random.choice(['Insufficient funds','Account closed','Payment stopped by drawer'])}".

3. You are hereby called upon to pay the cheque amount of {_ra()} along with
   interest and costs within 15 days of receipt of this notice.

4. FAILING WHICH, my client shall file a criminal complaint under
   Section 138 of the Negotiable Instruments Act, 1881, without further notice.

Advocate {advocate}
Enrolment No.: {random.randint(1000,9999)}/{_ry()}
"""

def _legal_notice_consumer() -> str:
    sender   = _rn(); advocate = _rn(); city = _rc()
    product  = random.choice(["washing machine","refrigerator","air conditioner","laptop","mobile phone"])
    return f"""LEGAL NOTICE UNDER CONSUMER PROTECTION ACT, 2019

From: Advocate {advocate} (for {sender})
Date: {_rd()}

TO: {random.choice(['M/S TechMart Pvt Ltd','Electronics Hub','Digi World'])}
    {city}, {_rs()}

Dear Sir/Madam,

1. My client purchased a {product} from you on {_rd()} vide Invoice No.
   INV/{random.randint(10000,99999)} for {_ra()}.

2. The {product} developed defects within the warranty period and despite
   {random.randint(2,6)} service complaints, no satisfactory remedy was provided.

3. This constitutes a "deficiency in service" and "unfair trade practice"
   under Section 2(11) and 2(47) of the Consumer Protection Act, 2019.

4. You are called upon to {random.choice(['replace the defective product','refund ' + _ra(),'repair free of cost'])}
   within 30 days of this notice.

5. Failing compliance, my client shall approach the District Consumer
   Disputes Redressal Commission under Section 34 of the Act.

Advocate {advocate}
Consumer Legal Forum, {city}
"""

def _legal_notice_rent() -> str:
    landlord = _rn(); tenant = _rn(); advocate = _rn(); city = _rc()
    return f"""LEGAL NOTICE

Sent by Advocate {advocate} on behalf of {landlord}
Address: {random.randint(1,200)}, {city}, {_rs()}
Date: {_rd()}

TO: {tenant}, occupying premises at Plot No. {random.randint(1,200)}, {city}

Subject: NOTICE TO QUIT AND VACATE PREMISES / RECOVERY OF RENT ARREARS

1. My client {landlord} is the owner of the premises occupied by you
   as a tenant under a rental agreement dated {_rd()}.

2. You have failed to pay rent from {random.choice(['last 3 months','last 6 months','last 4 months'])},
   totalling arrears of {_ra()}.

3. You are also in violation of the tenancy terms by
   {random.choice(['subletting without consent','causing structural damage','using for commercial purposes'])}.

4. You are hereby called upon to:
   (a) Pay arrears of {_ra()} within 7 days; and
   (b) Vacate and hand over vacant possession within 30 days.

5. Failing compliance, proceedings shall be initiated under the Kerala
   Buildings (Lease and Rent Control) Act, 1965 before the Rent Control Court.

Advocate {advocate}
{city}, {_rs()}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Generator pool
# Employment has 6 variants (double others) to fix its under-representation.
# Rental has 4 variants (one corporate-tenant) for negative pressure.
# ══════════════════════════════════════════════════════════════════════════════
GENERATORS: dict[int, list] = {
    0: [_rental_agreement_formal, _rental_agreement_brief,
        _rental_agreement_commercial, _rental_agreement_risky,
        _rental_agreement_corporate_tenant],
    1: [_fir_standard,            _fir_cybercrime,             _fir_domestic],
    2: [_court_notice_writ,       _court_notice_summons,       _court_notice_contempt],
    3: [_employment_contract_tech, _employment_contract_executive,
        _employment_contract_offer, _employment_contract_short_offer,
        _employment_contract_appointment, _employment_contract_internship],
    4: [_property_deed_sale,      _property_deed_gift,         _property_deed_mortgage],
    5: [_sc_judgment_criminal,    _sc_judgment_constitutional,  _sc_judgment_civil],
    6: [_hc_judgment_writ,        _hc_judgment_bail,           _hc_judgment_revision],
    7: [_legal_notice_cheque,     _legal_notice_consumer,      _legal_notice_rent],
}


def generate_dataset(samples_per_class: int = 150) -> list[dict]:
    """
    Generate training dataset.

    Default bumped 80 -> 150: more samples reduce sensitivity to
    template-level memorisation.

    Cycles round-robin through all variants so every structural form
    is represented regardless of samples_per_class.

    Returns list of {"text": str, "label": int, "label_name": str}
    """
    random.seed(42)
    dataset = []

    for label, variant_fns in GENERATORS.items():
        label_name = CATEGORIES[label]
        n_variants = len(variant_fns)

        for i in range(samples_per_class):
            gen_fn = variant_fns[i % n_variants]
            text   = gen_fn()
            text   = re.sub(r'\n{3,}', lambda _: '\n' * random.randint(1, 2), text)
            dataset.append({
                "text":       text.strip(),
                "label":      label,
                "label_name": label_name,
            })

        print(f"  Generated {samples_per_class} samples for: {label_name}")

    random.shuffle(dataset)
    return dataset