"""
LexShield LLM Client
=====================
Production LLM client backed by MultiLLMRouter.

All existing callers continue to work unchanged:
  from rag.llm import llm
  response = llm.generate("What is Article 21 of the Indian Constitution?")

Provider priority (first available wins, proactively switches before rate limit):
  1. Groq — llama-3.3-70b-versatile (primary, fastest)
  2. OpenRouter — meta-llama/llama-3.3-70b-instruct:free
  3. OpenRouter — qwen/qwen3-235b-a22b:free
  4. Gemini — gemini-2.0-flash
  5. OpenRouter — nvidia/llama-3.1-nemotron-70b-instruct:free
  6. OpenRouter — deepseek/deepseek-r1:free
  7. OpenRouter — mistralai/mistral-7b-instruct:free (final fallback)

.env requirements:
  GROQ_API_KEY       — required (primary provider)
  OPENROUTER_API_KEY — strongly recommended (providers 2, 3, 5, 6, 7)
  GEMINI_API_KEY     — optional (provider 4)
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Keep LegalLLM as a reference implementation (not used in production) ──────
# Retained so any tests that import LegalLLM directly don't break.

class LegalLLM:
    """
    Original single-provider Groq client.
    Kept for backward compatibility and as a reference implementation.
    In production, MultiLLMRouter is used instead.
    """

    def __init__(self):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        self.client = Groq(api_key=api_key)
        self.model  = "llama-3.3-70b-versatile"
        print(f"LLM ready: {self.model} via Groq (single-provider mode)")

    def generate(
        self,
        prompt:        str,
        system_prompt: str   = "You are a knowledgeable Indian legal assistant.",
        max_tokens:    int   = 1024,
        temperature:   float = 0.1,
    ) -> str:
        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return response.choices[0].message.content.strip()


# ── Module-level singleton — MultiLLMRouter with LegalLLM fallback ────────────

def _create_llm():
    """
    Try to create a MultiLLMRouter. Falls back to LegalLLM (Groq-only)
    if the openai package is not installed or MultiLLMRouter fails to init.
    """
    try:
        from rag.multi_llm import MultiLLMRouter
        router = MultiLLMRouter()
        return router
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "[LLM] openai package not installed — falling back to single-provider LegalLLM. "
            "Run: pip install openai  to enable multi-provider failover."
        )
        return LegalLLM()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"[LLM] MultiLLMRouter init failed ({e}) — "
            "falling back to single-provider LegalLLM."
        )
        return LegalLLM()


llm = _create_llm()