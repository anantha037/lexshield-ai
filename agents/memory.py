"""
LexShield AI — Session Memory
==============================
In-memory conversation history per session.

Schema per turn:
  { role, content, intent, timestamp }

Limits:
  MAX_TURNS_STORED  = 20  (per session, FIFO)
  MAX_TURNS_INJECT  = 5   (last N turns injected into prompts)

Storage:
  Plain dict — replace with Redis for production.
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


MAX_TURNS_STORED = 20
MAX_TURNS_INJECT = 5


# ═══════════════════════════════════════════════════════════════════════════════
# TURN DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    role: str                    # "user" or "assistant"
    content: str
    intent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MEMORY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class SessionMemory:

    def __init__(self):
        # { session_id: [Turn, ...] }
        self._store: dict[str, list[Turn]] = {}

    # ── Session management ─────────────────────────────────────────────────────

    def create_session(self) -> str:
        """Generate a new session ID."""
        sid = str(uuid.uuid4())
        self._store[sid] = []
        return sid

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._store

    def ensure_session(self, session_id: str) -> str:
        """Use provided session_id if valid, else create new one."""
        if not session_id or session_id not in self._store:
            return self.create_session()
        return session_id

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    # ── Turn operations ────────────────────────────────────────────────────────

    def add_turn(self, session_id: str, role: str, content: str, intent: Optional[str] = None) -> None:
        """Append a turn. Trims to MAX_TURNS_STORED if exceeded."""
        if session_id not in self._store:
            self._store[session_id] = []

        turn = Turn(role=role, content=content, intent=intent)
        self._store[session_id].append(turn)

        # FIFO trim
        if len(self._store[session_id]) > MAX_TURNS_STORED:
            self._store[session_id] = self._store[session_id][-MAX_TURNS_STORED:]

    def get_history(self, session_id: str) -> list[dict]:
        """Return full stored history as list of dicts."""
        if session_id not in self._store:
            return []
        return [t.to_dict() for t in self._store[session_id]]

    def get_context_block(self, session_id: str) -> str:
        """
        Return last MAX_TURNS_INJECT turns formatted for prompt injection.
        Returns empty string if no history.

        Format:
          [CONVERSATION HISTORY]
          User: ...
          Assistant: ...
          [END HISTORY]
        """
        if session_id not in self._store:
            return ""

        recent = self._store[session_id][-MAX_TURNS_INJECT:]
        if not recent:
            return ""

        lines = ["[CONVERSATION HISTORY]"]
        for turn in recent:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")
        lines.append("[END HISTORY]")

        return "\n".join(lines)

    def turn_count(self, session_id: str) -> int:
        return len(self._store.get(session_id, []))


# ── Singleton ──────────────────────────────────────────────────────────────────
session_memory = SessionMemory()