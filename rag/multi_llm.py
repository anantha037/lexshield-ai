"""
LexShield AI — Multi-LLM Router
=================================
Production-grade LLM client with proactive provider failover.

Architecture:
  - Sliding window rate tracker per provider (avoids 429 before it happens)
  - Circuit breaker per provider (backs off after 3 consecutive failures)
  - Priority queue: Groq -> OpenRouter Llama -> OpenRouter Qwen -> Gemini ->
                    OpenRouter Nemotron -> OpenRouter DeepSeek -> OpenRouter Mistral
  - Identical generate() interface to LegalLLM — drop-in replacement
  - get_langchain_llm() returns a LangChain-compatible LLM for RAGAS eval

Why proactive switching matters:
  Reactive (wait for 429) wastes ~8–14s per failure on Groq's exponential retry.
  Proactive (pre-check sliding window) costs 0ms — no API call is ever wasted.

.env keys used (all optional except GROQ_API_KEY):
  GROQ_API_KEY       — primary provider
  OPENROUTER_API_KEY — providers 2, 3, 5, 6, 7
  GEMINI_API_KEY     — provider 4
"""

import os
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProviderConfig:
    """
    Immutable configuration for one LLM provider endpoint.
    All runtime state (timestamps, failures) lives in ProviderState.
    """
    name:           str
    model:          str
    api_key_env:    str          # .env variable name
    base_url:       str          # OpenAI-compatible endpoint
    rpm_limit:      int          # hard requests-per-minute cap
    priority:       int          # lower = higher priority
    # Buffer before the hard limit — switch when (requests_in_window >= rpm_limit - buffer)
    preempt_buffer: int  = 3


@dataclass
class ProviderState:
    """Mutable runtime state for one provider — tracked separately from config."""
    request_timestamps:   deque = field(default_factory=deque)
    consecutive_failures: int   = 0
    circuit_open_until:   float = 0.0    # epoch seconds; 0 = circuit closed
    total_requests:       int   = 0
    total_failures:       int   = 0

    @property
    def is_circuit_open(self) -> bool:
        return time.time() < self.circuit_open_until


# ── Provider registry (priority order) ────────────────────────────────────────

_PROVIDERS: list[ProviderConfig] = [
    # 1. Groq — correct (browser GET error is expected, POST works fine)
    ProviderConfig(
        name           = "groq-llama33-70b",
        model          = "llama-3.3-70b-versatile",
        api_key_env    = "GROQ_API_KEY",
        base_url       = "https://api.groq.com/openai/v1",
        rpm_limit      = 30,
        priority       = 1,
        preempt_buffer = 4,
    ),
    # 2. OpenRouter Llama 3.3 70B — correct (browser redirect is expected)
    ProviderConfig(
        name           = "openrouter-llama33-70b",
        model          = "meta-llama/llama-3.3-70b-instruct:free",
        api_key_env    = "OPENROUTER_API_KEY",
        base_url       = "https://openrouter.ai/api/v1",
        rpm_limit      = 20,
        priority       = 2,
        preempt_buffer = 3,
    ),
    # 3. DeepSeek V4 Flash — fast, free, strong on structured output
    ProviderConfig(
        name           = "openrouter-deepseek-v4",
        model          = "deepseek/deepseek-v4-flash:free",
        api_key_env    = "OPENROUTER_API_KEY",
        base_url       = "https://openrouter.ai/api/v1",
        rpm_limit      = 20,
        priority       = 3,
        preempt_buffer = 3,
    ),
    # 4. FIXED: trailing slash added. Without it the OpenAI client constructs
    #    ".../openaichat/completions" instead of ".../openai/chat/completions".
    #    Browser 404 was caused by missing slash + no API key in GET — not a
    #    wrong domain. In code this will work with your GEMINI_API_KEY.
    ProviderConfig(
        name           = "gemini-2.0-flash",
        model          = "gemini-2.0-flash",
        api_key_env    = "GEMINI_API_KEY",
        base_url       = "https://generativelanguage.googleapis.com/v1beta/openai/",
        rpm_limit      = 15,
        priority       = 4,
        preempt_buffer = 2,
    ),
    # 5. Google Gemma 4 31B — free, good general-purpose
    ProviderConfig(
        name           = "openrouter-gemma4-31b",
        model          = "google/gemma-4-31b-it:free",
        api_key_env    = "OPENROUTER_API_KEY",
        base_url       = "https://openrouter.ai/api/v1",
        rpm_limit      = 20,
        priority       = 5,
        preempt_buffer = 3,
    ),
    # 6. MiniMax M2.5 — free, reliable fallback
    ProviderConfig(
        name           = "openrouter-minimax-m25",
        model          = "minimax/minimax-m2.5:free",
        api_key_env    = "OPENROUTER_API_KEY",
        base_url       = "https://openrouter.ai/api/v1",
        rpm_limit      = 20,
        priority       = 6,
        preempt_buffer = 3,
    ),
    # 7. OpenRouter auto-free router — ultimate fallback, picks any free model
    ProviderConfig(
        name           = "openrouter-auto-free",
        model          = "openrouter/free",
        api_key_env    = "OPENROUTER_API_KEY",
        base_url       = "https://openrouter.ai/api/v1",
        rpm_limit      = 20,
        priority       = 7,
        preempt_buffer = 2,
    ),
]

# Circuit breaker config
_CIRCUIT_OPEN_SECONDS    = 300   # 5 minutes before retrying a failed provider
_MAX_CONSECUTIVE_FAILURES = 3    # open circuit after this many back-to-back failures


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-LLM ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLLMRouter:
    """
    Drop-in replacement for LegalLLM with multi-provider failover.

    Interface is identical to LegalLLM.generate() — all existing callers
    (rag/pipeline.py, agents/graph.py, etc.) work without any changes.

    Rate tracking strategy:
      Each provider maintains a deque of request timestamps.
      Before dispatching, we prune entries older than 60s and compare
      len(deque) against (rpm_limit - preempt_buffer).
      If near-limit -> skip to next provider without making an API call.
      Only if all providers are near-limit do we sleep and retry.

    Circuit breaker:
      3 consecutive failures -> circuit opens for 5 minutes.
      Prevents hammering a provider that is returning 5xx errors.
    """

    def __init__(self):
        self._configs: list[ProviderConfig] = sorted(_PROVIDERS, key=lambda p: p.priority)
        self._states:  dict[str, ProviderState] = {
            cfg.name: ProviderState() for cfg in self._configs
        }
        # Build OpenAI-compatible clients for available providers
        self._clients: dict[str, object] = {}
        self._available_providers: list[ProviderConfig] = []
        self._init_clients()

        if not self._available_providers:
            raise RuntimeError(
                "MultiLLMRouter: no providers could be initialised. "
                "At minimum, set GROQ_API_KEY in .env."
            )

        # Expose .model for backward compatibility with code that reads llm.model
        self.model = self._available_providers[0].model

        logger.info(
            f"[MultiLLMRouter] Initialised with {len(self._available_providers)} provider(s): "
            f"{[p.name for p in self._available_providers]}"
        )
        print(
            f"[MultiLLMRouter] {len(self._available_providers)} provider(s) ready: "
            f"{' -> '.join(p.name for p in self._available_providers)}"
        )

    def _init_clients(self):
        """
        Build one OpenAI-compatible client per available provider.
        Skips providers whose API key is missing from .env — no crash.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for MultiLLMRouter. "
                "Run: pip install openai"
            )

        seen_openrouter = False   # one client for all OpenRouter models
        openrouter_client = None

        for cfg in self._configs:
            api_key = os.getenv(cfg.api_key_env, "")
            if not api_key:
                logger.debug(f"[MultiLLMRouter] Skipping {cfg.name}: {cfg.api_key_env} not set")
                continue

            try:
                if "openrouter" in cfg.base_url:
                    # Reuse one OpenRouter client (same key, different model per call)
                    if not seen_openrouter:
                        openrouter_client = OpenAI(
                            api_key  = api_key,
                            base_url = cfg.base_url,
                            default_headers = {
                                "HTTP-Referer": "https://lexshield.ai",
                                "X-Title":      "LexShield AI",
                            },
                        )
                        seen_openrouter = True
                    self._clients[cfg.name] = openrouter_client
                else:
                    self._clients[cfg.name] = OpenAI(
                        api_key  = api_key,
                        base_url = cfg.base_url,
                    )

                self._available_providers.append(cfg)
                logger.debug(f"[MultiLLMRouter] Provider ready: {cfg.name}")

            except Exception as e:
                logger.warning(f"[MultiLLMRouter] Could not init {cfg.name}: {e}")

    # ── Rate tracking ──────────────────────────────────────────────────────────

    def _prune_window(self, state: ProviderState):
        """Remove timestamps older than 60 seconds from the sliding window."""
        cutoff = time.time() - 60.0
        while state.request_timestamps and state.request_timestamps[0] < cutoff:
            state.request_timestamps.popleft()

    def _is_near_limit(self, cfg: ProviderConfig, state: ProviderState) -> bool:
        """
        Returns True if sending one more request would approach the RPM limit.
        Prunes the window first for accurate count.
        """
        self._prune_window(state)
        return len(state.request_timestamps) >= (cfg.rpm_limit - cfg.preempt_buffer)

    def _record_request(self, state: ProviderState):
        """Log a request timestamp immediately before dispatch."""
        state.request_timestamps.append(time.time())
        state.total_requests += 1

    def _record_success(self, cfg: ProviderConfig, state: ProviderState):
        state.consecutive_failures = 0
        # Update .model to reflect active provider for external visibility
        self.model = cfg.model

    def _record_failure(self, cfg: ProviderConfig, state: ProviderState, exc: Exception):
        state.consecutive_failures += 1
        state.total_failures       += 1
        if state.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            state.circuit_open_until = time.time() + _CIRCUIT_OPEN_SECONDS
            logger.warning(
                f"[MultiLLMRouter] Circuit OPEN for {cfg.name} "
                f"({state.consecutive_failures} failures) — "
                f"resuming in {_CIRCUIT_OPEN_SECONDS}s"
            )

    # ── Provider selection ─────────────────────────────────────────────────────

    def _get_available_providers(self) -> list[tuple[ProviderConfig, ProviderState]]:
        """
        Return providers that are: initialised + circuit closed + not near rate limit.
        Falls back to returning rate-limited (but circuit-closed) providers if
        everything is saturated — caller will sleep and retry.
        """
        available = []
        for cfg in self._available_providers:
            if cfg.name not in self._clients:
                continue
            state = self._states[cfg.name]
            if state.is_circuit_open:
                continue
            available.append((cfg, state))

        if not available:
            return []

        # Prefer providers not near their rate limit
        not_limited  = [(c, s) for c, s in available if not self._is_near_limit(c, s)]
        near_limited = [(c, s) for c, s in available if self._is_near_limit(c, s)]

        return not_limited if not_limited else near_limited

    # ── Core generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:        str,
        system_prompt: str   = "You are a knowledgeable Indian legal assistant.",
        max_tokens:    int   = 1024,
        temperature:   float = 0.1,
    ) -> str:
        """
        Generate a response using the best available provider.

        Tries providers in priority order. On 429 or transient error, moves to
        next provider immediately. On sustained failure, opens circuit breaker.
        Only raises if ALL providers fail — which means something is critically wrong.

        Args:
            prompt:       User prompt text
            system_prompt: System role instruction
            max_tokens:   Maximum tokens in response
            temperature:  0.0 = deterministic, 1.0 = creative

        Returns:
            Response text string.

        Raises:
            RuntimeError: if every provider fails on this request.
        """
        max_attempts    = 3     # full cycles through provider list
        last_exception  = None

        for attempt in range(max_attempts):
            candidates = self._get_available_providers()

            if not candidates:
                # All circuits open — wait for fastest circuit to reset
                soonest = min(
                    self._states[cfg.name].circuit_open_until
                    for cfg in self._available_providers
                    if cfg.name in self._clients
                )
                wait = max(0.0, soonest - time.time()) + 1.0
                logger.warning(f"[MultiLLMRouter] All circuits open — waiting {wait:.0f}s")
                time.sleep(wait)
                continue

            for cfg, state in candidates:
                # Proactive rate check: if near limit, skip without calling API
                if self._is_near_limit(cfg, state):
                    logger.debug(f"[MultiLLMRouter] Preemptive skip {cfg.name} (near RPM limit)")
                    continue

                try:
                    self._record_request(state)
                    client = self._clients[cfg.name]

                    logger.debug(f"[MultiLLMRouter] Dispatching to {cfg.name}")
                    response = client.chat.completions.create(
                        model       = cfg.model,
                        messages    = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": prompt},
                        ],
                        max_tokens  = max_tokens,
                        temperature = temperature,
                    )
                    raw_content = response.choices[0].message.content
                    if raw_content is None or raw_content.strip() == "":
                        logger.warning(
                            f"[MultiLLMRouter] Provider {cfg.name} returned empty response — trying next"
                        )
                        self._record_failure(cfg, state, RuntimeError("empty response"))
                        continue
                    result = raw_content.strip()
                    self._record_success(cfg, state)
                    return result

                except Exception as exc:
                    err_str = str(exc)
                    last_exception = exc

                    if "429" in err_str or "rate_limit" in err_str.lower() or "too many" in err_str.lower():
                        # Rate limit hit despite proactive check — add to window, skip
                        logger.info(
                            f"[MultiLLMRouter] 429 on {cfg.name} — switching provider"
                        )
                        # Mark one more request in the window to prevent immediate retry
                        state.request_timestamps.append(time.time())
                        continue  # try next provider immediately

                    elif any(code in err_str for code in ("500", "502", "503", "504")):
                        # Server error — record failure, may open circuit
                        logger.warning(f"[MultiLLMRouter] Server error on {cfg.name}: {exc}")
                        self._record_failure(cfg, state, exc)
                        continue

                    else:
                        # Auth error, model not found, malformed request — not transient
                        logger.error(
                            f"[MultiLLMRouter] Non-retryable error on {cfg.name}: {exc}"
                        )
                        self._record_failure(cfg, state, exc)
                        continue

            # Exhausted all providers this cycle — brief sleep before retry
            if attempt < max_attempts - 1:
                sleep_time = 5 * (attempt + 1)   # 5s, 10s
                logger.info(
                    f"[MultiLLMRouter] All providers exhausted (attempt {attempt+1}/"
                    f"{max_attempts}) — sleeping {sleep_time}s"
                )
                time.sleep(sleep_time)

        raise RuntimeError(
            f"MultiLLMRouter: all {len(self._available_providers)} providers failed. "
            f"Last error: {last_exception}"
        )

    # ── LangChain adapter for RAGAS ────────────────────────────────────────────

    def get_langchain_llm(self, provider_priority: int = 2):
        """
        Return a LangChain-compatible LLM pointed at the provider at the
        given priority level. Used by RAGAS evaluation (run_evals.py).

        RAGAS scoring calls are batched separately from RAG pipeline calls,
        so pointing them at a secondary provider avoids competing for the
        same Groq rate limit.

        Args:
            provider_priority: 1 = Groq (default for production),
                               2 = OpenRouter Llama (default for eval scoring)

        Returns:
            LangchainLLMWrapper wrapping a ChatOpenAI pointed at the provider.
        """
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms       import LangchainLLMWrapper
        except ImportError as e:
            raise ImportError(
                f"langchain_openai and ragas required: {e}. "
                "Run: pip install langchain-openai ragas"
            )

        # Find the provider at the requested priority (or next available)
        target_cfg = None
        for cfg in sorted(self._available_providers, key=lambda c: c.priority):
            if cfg.priority >= provider_priority and cfg.name in self._clients:
                api_key = os.getenv(cfg.api_key_env, "")
                if api_key:
                    target_cfg = cfg
                    break

        if target_cfg is None:
            # Fallback: use whatever is available
            target_cfg = self._available_providers[0]

        api_key = os.getenv(target_cfg.api_key_env, "")

        # Build ChatOpenAI client pointed at this provider
        kwargs: dict = {
            "model":       target_cfg.model,
            "api_key":     api_key,
            "base_url":    target_cfg.base_url,
            "temperature": 0.0,
            "max_tokens":  1024,
        }
        if "openrouter" in target_cfg.base_url:
            kwargs["default_headers"] = {
                "HTTP-Referer": "https://lexshield.ai",
                "X-Title":      "LexShield AI",
            }

        chat_llm = ChatOpenAI(**kwargs)
        logger.info(
            f"[MultiLLMRouter] RAGAS eval LLM: {target_cfg.name} ({target_cfg.model})"
        )
        print(f"[MultiLLMRouter] RAGAS eval LLM: {target_cfg.name}")
        return LangchainLLMWrapper(chat_llm)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return current state of all providers. Useful for /health endpoint."""
        result = {}
        for cfg in self._available_providers:
            state = self._states[cfg.name]
            self._prune_window(state)
            result[cfg.name] = {
                "model":              cfg.model,
                "priority":           cfg.priority,
                "requests_in_window": len(state.request_timestamps),
                "rpm_limit":          cfg.rpm_limit,
                "utilisation_pct":    round(
                    len(state.request_timestamps) / cfg.rpm_limit * 100, 1
                ),
                "circuit_open":       state.is_circuit_open,
                "consecutive_failures": state.consecutive_failures,
                "total_requests":     state.total_requests,
                "total_failures":     state.total_failures,
                "available":          cfg.name in self._clients,
            }
        return result