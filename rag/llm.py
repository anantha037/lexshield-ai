"""
LexShield LLM Client
=====================
Connects to Groq's free API (LLaMA 3.3 70B).

Why Groq:
  - Free tier: generous rate limits
  - LLaMA 3.3 70B: strong legal reasoning
  - Fast inference: runs on Groq's hardware, not your laptop

Usage:
  from rag.llm import llm
  response = llm.generate("What is Article 21 of the Indian Constitution?")
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Model config
MODEL_NAME   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.1   # Low = more factual, less creative (correct for legal)


class LegalLLM:
    """
    Thin wrapper around Groq client for legal Q&A.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model  = MODEL_NAME
        print(f"LLM ready: {self.model} via Groq")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a knowledgeable Indian legal assistant.",
        max_tokens: int    = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> str:
        """
        Send a prompt to Groq and return the response text.

        Args:
            prompt      : the full user prompt
            system_prompt: sets the LLM's behaviour/role
            max_tokens  : max response length
            temperature : 0.0 = deterministic, 1.0 = creative

        Returns:
            Response string from the LLM.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# Module-level singleton
llm = LegalLLM()