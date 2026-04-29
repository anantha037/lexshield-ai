"""
LexShield AI — NVIDIA NIM Reranker  (Week 2, Day 2)
====================================================
Sends top-N retrieved chunks to NVIDIA NIM reranker API.
Gets back chunks reordered by true relevance to the original query.
Falls back gracefully to hybrid search order if API is unavailable.

Model used: nvidia/llama-3.2-nv-rerankqa-1b-v2
  • Free tier on build.nvidia.com
  • Specifically trained for passage reranking (not generation)
  • Returns a relevance score per passage, we re-sort by it

Fallback chain:
  NVIDIA API available  →  reranked order
  NVIDIA API down/slow  →  original hybrid order (no crash)
  API key missing       →  original hybrid order + warning logged
"""

import os
import time
import json
import urllib.request
import urllib.error
from typing import Optional
from dotenv import load_dotenv

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


load_dotenv()

NVIDIA_API_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
REQUEST_TIMEOUT  = 15    # seconds — fail fast, never block the pipeline
MAX_PASSAGE_LEN  = 500   # chars sent per chunk (reranker has token limits)


def _truncate(text: str, max_chars: int = MAX_PASSAGE_LEN) -> str:
    """Truncate chunk text for reranker API (keeps first max_chars chars)."""
    return text[:max_chars].strip() if len(text) > max_chars else text.strip()


# ── Reranker class ────────────────────────────────────────────────────────────

class NVIDIAReranker:
    """
    Wraps the NVIDIA NIM reranking API.

    Usage:
        from rag.reranker import reranker
        reranked = reranker.rerank(query, chunks, top_n=5)
    """

    def __init__(self):
        self.api_key   = NVIDIA_API_KEY
        self.available = bool(self.api_key)
        if not self.available:
            print("[Reranker] NVIDIA_API_KEY not set — reranker will use fallback mode.")
        else:
            print("[Reranker] NVIDIA NIM reranker ready.")

    # ── Core rerank ───────────────────────────────────────────────────────────

    def rerank(
        self,
        query:   str,
        chunks:  list[dict],
        top_n:   int = 5,
    ) -> tuple[list[dict], bool]:
        """
        Rerank chunks by relevance to query using NVIDIA NIM.

        Returns:
            (reranked_chunks, used_reranker)
            used_reranker = True  → NVIDIA API was called successfully
            used_reranker = False → fallback order used (first top_n from input)

        The returned chunks have a new field:
            rerank_score  (float) — relevance score from NVIDIA, or None on fallback
        """
        if not chunks:
            return [], False

        # Cap at top_n even on fallback
        if not self.available:
            return self._fallback(chunks, top_n, reason="no API key"), False

        try:
            return self._call_api(query, chunks, top_n)
        except Exception as e:
            print(f"[Reranker] API call failed ({type(e).__name__}: {e}) — using fallback.")
            return self._fallback(chunks, top_n, reason=str(e)), False

    # ── NVIDIA API call ───────────────────────────────────────────────────────

    def _call_api(
        self,
        query:  str,
        chunks: list[dict],
        top_n:  int,
    ) -> tuple[list[dict], bool]:
        """
        Calls NVIDIA NIM reranking endpoint.
        Endpoint expects:
          { "model": "...", "query": {"text": "..."}, "passages": [{"text": "..."}] }
        Returns sorted chunks with rerank_score attached.
        """
        # Build passages list (truncated for API limits)
        passages = [
            {"text": _truncate(c.get("text", ""))}
            for c in chunks
        ]

        payload = json.dumps({
            "model": "nv-rerank-qa-mistral-4b:1",
            "query":    {"text": query},
            "passages": passages,
        }).encode("utf-8")

        req = urllib.request.Request(
            NVIDIA_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        # body["rankings"] = [{"index": int, "logit": float}, ...]
        rankings = body.get("rankings", [])
        if not rankings:
            raise ValueError("Empty rankings in NVIDIA response.")

        # Map index → score
        score_map: dict[int, float] = {
            r["index"]: float(r.get("logit", 0.0))
            for r in rankings
        }

        # Attach rerank_score to each chunk, sort descending
        scored: list[dict] = []
        for i, chunk in enumerate(chunks):
            c = dict(chunk)                     # don't mutate original
            c["rerank_score"] = score_map.get(i, -999.0)
            scored.append(c)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n], True

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback(
        self,
        chunks: list[dict],
        top_n:  int,
        reason: str = "",
    ) -> list[dict]:
        """Returns first top_n chunks with rerank_score=None."""
        if reason:
            print(f"[Reranker] Fallback reason: {reason}")
        result = []
        for c in chunks[:top_n]:
            c2 = dict(c)
            c2["rerank_score"] = None
            result.append(c2)
        return result

    # ── Health check ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Quick liveness test. Returns True if API responds."""
        if not self.api_key:
            return False
        try:
            self._call_api(
                query="test",
                chunks=[{"text": "test passage", "chunk_id": "ping"}],
                top_n=1,
            )
            return True
        except Exception as e:
            print(f"[Reranker] Ping failed: {e}")
            return False


# ── Singleton ─────────────────────────────────────────────────────────────────
reranker = NVIDIAReranker()