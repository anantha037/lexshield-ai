"""
LexShield AI — Synthetic Data Generator v3
===========================================
Changes from v2:
- Gemini: switched from deprecated google-generativeai to google-genai
- Per-provider sleep rates (Gemini needs 5s, Groq needs 2s)
- Smarter wait: on rate limit, waits parsed time then retries SAME provider
  before switching (daily limit vs RPM limit handled differently)
- OpenRouter added as third free provider

Setup:
    pip install google-genai groq

    .env:
        GROQ_API_KEY=...
        GEMINI_API_KEY=...       # https://aistudio.google.com/app/apikey
        OPENROUTER_API_KEY=...   # https://openrouter.ai (free, no card needed)

Usage:
    python -m models.data_generator --samples 50 --provider gemini
    python -m models.data_generator --samples 50 --category employment_contract
"""

import os
import json
import time
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)
load_dotenv()

OUTPUT_DIR   = Path("data/classifier_samples")
MAX_TOKENS   = 1200
GROQ_MODEL   = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"

# Per-provider sleep to respect RPM limits
# Gemini free: 15 RPM -> 4s min. Groq: no RPM limit listed -> 2s safe.
PROVIDER_SLEEP = {
    "gemini":     5.0,
    "groq":       2.0,
    "openrouter": 3.0,
}
# ── Category prompts ──────────────────────────────────────────────────────────
# Each prompt instructs Groq to generate a realistic Indian legal document.
# Variation instructions ensure diversity across samples.

CATEGORY_PROMPTS: dict[str, str] = {
    "rental_agreement": """Generate a realistic Indian residential rental agreement in English.
Include: parties (landlord and tenant with Indian names), property address in Kerala or another Indian state, 
monthly rent amount in rupees, lease duration (11 months typically), security deposit, 
termination clause, maintenance responsibilities, and witness signatures.
Make it legally worded but vary the rent amount, location, names, and specific clauses each time.
Output only the document text, no commentary.""",

    "fir": """Generate a realistic First Information Report (FIR) as used in Indian police stations.
Include: FIR number, police station name, date, complainant details, accused details (can be unknown),
nature of offence, relevant IPC/BNS sections, brief facts of the case, list of witnesses if any.
Vary the type of offence (theft, assault, cheating, domestic violence, cybercrime etc.) each time.
Output only the FIR document text, no commentary.""",

    "court_notice_summons": """Generate a realistic court notice or summons issued by an Indian court.
Include: court name (Magistrate/District/High Court), case number, parties, 
date of appearance required, purpose (show cause / evidence / judgment), 
consequences of non-appearance, court seal reference.
Vary the court level, case type, and jurisdiction each time.
Output only the notice text, no commentary.""",

    "employment_contract": """Generate a realistic Indian employment appointment letter or contract.
Include: employer company name, employee name, designation, joining date, 
salary (basic + allowances), probation period, notice period, 
confidentiality clause, leave policy, termination conditions.
Vary the industry (IT, manufacturing, healthcare, retail), salary levels, and specific clauses.
Output only the document text, no commentary.""",

    "property_deed": """Generate a realistic Indian property sale deed or conveyance deed.
Include: vendor and purchaser names, property description (plot number, survey number, area), 
location in an Indian state, sale consideration amount, payment terms, 
encumbrance certificate reference, registration details, witnesses.
Vary the property type (residential plot, flat, agricultural land), location, and amounts.
Output only the deed text, no commentary.""",

    "sc_judgment": """Generate a realistic Supreme Court of India judgment excerpt.
Include: case title, civil/criminal appeal number, bench composition (Justice names),
brief facts, legal issues framed, arguments of both sides, court's analysis,
references to previous judgments, final order (allowed/dismissed).
Vary the legal subject matter (constitutional, civil, criminal, tax) each time.
Output only the judgment text, no commentary.""",

    "hc_judgment": """Generate a realistic High Court of India judgment (Kerala, Delhi, Bombay, or Madras HC).
Include: writ petition or appeal number, court name, bench, parties,
brief facts, legal questions, reasoning, references to statutes and precedents, final order.
Vary the type (writ petition, criminal revision, civil appeal, contempt) each time.
Output only the judgment text, no commentary.""",

    "legal_notice": """Generate a realistic Indian legal notice sent by an advocate on behalf of a client.
Include: sender advocate name and address, recipient details, 
subject matter (recovery of dues, property dispute, defamation, breach of contract),
specific demands, timeline for compliance (15/30 days), 
warning of legal action, instruction to address reply to advocate.
Vary the subject matter and demands each time.
Output only the notice text, no commentary.""",

    "affidavit": """Generate a realistic Indian affidavit document.
Include: deponent's name, age, address, occupation, 
sworn statement content (lost document, character certificate, income declaration etc.),
notary/oath commissioner details, date, place.
Vary the purpose (lost document, address proof, income declaration, relationship proof) each time.
Output only the affidavit text, no commentary.""",

    "power_of_attorney": """Generate a realistic Indian Power of Attorney document.
Include: grantor and grantee details with Indian names and addresses,
specific powers granted (property transactions, banking, legal proceedings),
duration or revocation terms, witness details, notarization reference.
Vary the type (general POA, specific POA for property/banking) each time.
Output only the document text, no commentary.""",

    "cheque_bounce_notice": """Generate a realistic cheque bounce legal notice under Section 138 of the Negotiable Instruments Act.
Include: payee's advocate name, drawee's details, cheque details (number, amount, date, bank),
date of dishonour, bank's memo reference, demand for payment within 15 days,
warning of criminal complaint under Section 138 NI Act.
Vary the amount, bank names, and relationship between parties each time.
Output only the notice text, no commentary.""",

    "bail_application": """Generate a realistic bail application filed before an Indian court.
Include: applicant/accused details, FIR number and police station, 
offences charged (IPC sections), grounds for bail (first offender, family dependents, 
flight risk assessment, evidence not likely to be tampered), 
prayer clause, advocate's signature.
Vary the offence type and grounds each time.
Output only the application text, no commentary.""",

    "consumer_complaint": """Generate a realistic consumer complaint filed before a Consumer Disputes Redressal Commission in India.
Include: complainant details, opposite party (company/service provider),
nature of deficiency (defective product, service failure, unfair trade practice),
amount paid, relief sought (refund, compensation, replacement),
reference to Consumer Protection Act 2019.
Vary the product/service type each time.
Output only the complaint text, no commentary.""",

    "loan_agreement": """Generate a realistic Indian loan agreement between a lender and borrower.
Include: lender and borrower details, loan amount in rupees, 
interest rate (monthly/annual), repayment schedule (EMI details),
security/collateral if any, default clause, prepayment terms,
governing law (Indian Contract Act).
Vary loan amounts, purposes (personal/business/vehicle), and terms each time.
Output only the agreement text, no commentary.""",

    "police_complaint": """Generate a realistic written police complaint (not FIR) submitted by a citizen to a police station in India.
Include: complainant's name and address, date, addressed to Station House Officer,
nature of complaint (neighbour dispute, threat, minor assault, missing person),
request to take action, complainant's signature.
Note: this is a complaint requesting registration of FIR, not the FIR itself.
Vary the nature of complaint and details each time.
Output only the complaint text, no commentary.""",
}


# ── Providers ─────────────────────────────────────────────────────────────────

class GroqProvider:
    name  = "groq"
    sleep = PROVIDER_SLEEP["groq"]

    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = MAX_TOKENS,
            temperature = 0.85,
        )
        return r.choices[0].message.content.strip()


class GeminiProvider:
    name  = "gemini"
    sleep = PROVIDER_SLEEP["gemini"]

    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> str:
        from google.genai import types
        r = self.client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                max_output_tokens = MAX_TOKENS,
                temperature       = 0.85,
            ),
        )
        return r.text.strip()


class OpenRouterProvider:
    name  = "openrouter"
    sleep = PROVIDER_SLEEP["openrouter"]

    def __init__(self):
        import httpx
        self.client  = httpx
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        # Free model that handles long legal text well
        self.model   = "meta-llama/llama-3.1-8b-instruct"

    def generate(self, prompt: str) -> str:
        import httpx
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            json = {
                "model":       self.model,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  MAX_TOKENS,
                "temperature": 0.85,
            },
            timeout = 60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def _build_providers(preferred: str) -> list:
    providers = []
    errors    = []

    # Build in preferred order
    all_classes = {
        "groq":       (GroqProvider,       "GROQ_API_KEY"),
        "gemini":     (GeminiProvider,     "GEMINI_API_KEY"),
        "openrouter": (OpenRouterProvider, "OPENROUTER_API_KEY"),
    }

    # Put preferred first
    order = [preferred] + [k for k in all_classes if k != preferred]

    for name in order:
        cls, env_key = all_classes[name]
        if not os.getenv(env_key):
            errors.append(f"{name}: {env_key} not set in .env")
            continue
        try:
            providers.append(cls())
            logger.info(f"[Generator] OK {name} ready")
        except Exception as e:
            errors.append(f"{name}: {e}")
            logger.info(f"[Generator] FAIL {name} failed: {e}")

    if not providers:
        raise RuntimeError(
            "No providers available.\n" + "\n".join(errors) + "\n\n"
            "Get free keys:\n"
            "  Gemini:     https://aistudio.google.com/app/apikey\n"
            "  Groq:       https://console.groq.com\n"
            "  OpenRouter: https://openrouter.ai (no credit card)\n"
        )

    return providers


# ── Generator ─────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:

    def __init__(self, preferred_provider: str = "gemini"):
        self.providers = _build_providers(preferred_provider)
        self._idx      = 0

    @property
    def provider(self):
        return self.providers[self._idx]

    def _next_provider(self) -> bool:
        if self._idx + 1 < len(self.providers):
            self._idx += 1
            logger.info(f"\n  [Provider] Switched -> {self.provider.name}")
            return True
        return False

    def _reset_providers(self):
        self._idx = 0

    @staticmethod
    def _parse_wait_seconds(err_str: str) -> int:
        """Parse wait time from rate limit error. Returns seconds."""
        # "Please try again in 7m47.424s"
        m = re.search(r'(\d+)m(\d+)', err_str)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2)) + 10
        # "try again in 30s"
        m = re.search(r'in (\d+\.?\d*)s', err_str, re.IGNORECASE)
        if m:
            return int(float(m.group(1))) + 5
        return 65   # safe default: 65 seconds

    def generate_one(self, category: str, sample_idx: int) -> str | None:
        prompt = (
            CATEGORY_PROMPTS[category]
            + f"\n\nIMPORTANT: Sample {sample_idx} — vary names, amounts, "
              f"locations, dates significantly from previous samples."
        )

        providers_tried = 0
        while providers_tried < len(self.providers):
            try:
                result = self.provider.generate(prompt)
                time.sleep(self.provider.sleep)
                return result

            except Exception as e:
                err_str    = str(e)
                is_rpm     = "429" in err_str and "per_minute" in err_str.lower()
                is_daily   = "429" in err_str and (
                    "per_day" in err_str.lower()
                    or "tokens per day" in err_str.lower()
                    or "daily" in err_str.lower()
                )
                is_quota   = "quota" in err_str.lower()
                is_rate    = is_rpm or is_daily or is_quota or "rate_limit" in err_str.lower()

                if is_rpm:
                    # RPM limit — short wait, retry SAME provider
                    wait = self._parse_wait_seconds(err_str)
                    wait = min(wait, 70)   # RPM resets in <60s always
                    logger.warning(f"\n  [!] {self.provider.name} RPM limit. "
                          f"Waiting {wait}s then retrying...")
                    time.sleep(wait)
                    # Don't increment providers_tried — retry same provider
                    continue

                elif is_daily or is_quota:
                    # Daily limit hit — switch provider permanently
                    logger.info(f"\n  [!] {self.provider.name} daily limit exhausted.")
                    if self._next_provider():
                        providers_tried += 1
                        continue
                    else:
                        # All providers daily-limited — parse longest wait
                        wait = self._parse_wait_seconds(err_str)
                        logger.warning(f"  All providers exhausted. "
                              f"Waiting {wait}s for reset...")
                        time.sleep(wait)
                        self._reset_providers()
                        return None   # skip this sample, resume will catch it

                elif is_rate:
                    # Generic rate limit — try next provider
                    logger.info(f"\n  [!] {self.provider.name} rate limited.")
                    if self._next_provider():
                        providers_tried += 1
                        continue
                    else:
                        wait = self._parse_wait_seconds(err_str)
                        logger.info(f"  Waiting {wait}s...")
                        time.sleep(wait)
                        self._reset_providers()
                        return None

                else:
                    # Non-rate-limit error (auth, network, etc.)
                    logger.exception(f"{self.provider.name} failed")
                    if self._next_provider():
                        providers_tried += 1
                        continue
                    return None

        return None

    def generate_category(
        self,
        category:  str,
        n_samples: int,
        resume:    bool = True,
    ) -> int:
        cat_dir = OUTPUT_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        existing   = sorted(cat_dir.glob("synth_*.txt"))
        start_from = len(existing) if resume else 0

        if resume and start_from >= n_samples:
            logger.info(f"  [{category}] Already complete ({start_from}). Skipping.")
            return start_from

        logger.info(f"  [{category}] {start_from} done, need "
              f"{n_samples - start_from} more...")

        generated = start_from
        for i in range(start_from, n_samples):
            text = self.generate_one(category, i + 1)
            if text and len(text.strip()) > 100:
                (cat_dir / f"synth_{i+1:03d}.txt").write_text(
                    text, encoding="utf-8"
                )
                generated += 1
                logger.info(f"    {i+1}/{n_samples} OK "
                      f"({len(text)} chars) [{self.provider.name}]")
            else:
                logger.info(f"    {i+1}/{n_samples} FAIL skipped")

        return generated

    def generate_all(self, n_samples: int = 50, resume: bool = True) -> dict:
        logger.info("=" * 60)
        logger.info(f"LexShield AI — Synthetic Data Generator v3")
        logger.info(f"Target: {n_samples} × {len(CATEGORY_PROMPTS)} categories")
        logger.info(f"Output: {OUTPUT_DIR.resolve()}")
        logger.info("=" * 60)

        results = {}
        for category in CATEGORY_PROMPTS:
            logger.info(f"\n[{category}]")
            results[category] = self.generate_category(
                category, n_samples, resume
            )

        logger.info("\n" + "=" * 60)
        total = sum(results.values())
        logger.info(f"DONE. Total: {total}")
        for cat, c in results.items():
            logger.info(f"  {cat:30s} "
                  f"{'OK' if c >= n_samples else f'FAIL ({c}/{n_samples})'}")

        with open(OUTPUT_DIR / "manifest.json", "w") as f:
            json.dump({"total": total, "per_category": results}, f, indent=2)

        return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",   type=int, default=50)
    parser.add_argument("--category",  type=str, default=None)
    parser.add_argument("--provider",  type=str, default="gemini",
                        choices=["groq", "gemini", "openrouter"])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    gen = SyntheticDataGenerator(preferred_provider=args.provider)

    if args.category:
        gen.generate_category(
            args.category, args.samples, resume=not args.no_resume
        )
    else:
        gen.generate_all(n_samples=args.samples, resume=not args.no_resume)