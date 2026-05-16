"""
LexShield AI — Session Memory  (SQLite backend)
================================================
Replaces the in-memory dict with a persistent SQLite store.
Survives server restarts.  Same public interface as the old
in-memory version so orchestrator.py needs no changes.

Tables
------
sessions (session_id PK, created_ts)
turns    (id PK, session_id FK, role, content, intent, ts)

Uses the SAME data/sessions.db file as the LangGraph SqliteSaver
checkpointer — the table names do not overlap (LangGraph uses
checkpoints / checkpoint_blobs / checkpoint_writes).

Public interface (unchanged from in-memory version)
----------------------------------------------------
session_memory.ensure_session(sid)         → str
session_memory.session_exists(sid)         → bool
session_memory.create_session()            → str
session_memory.delete_session(sid)         → bool
session_memory.add_turn(sid, role, content, intent=None)
session_memory.get_history(sid)            → list[dict]
session_memory.get_context_block(sid)      → str
session_memory.turn_count(sid)             → int
"""

import os
import sqlite3
import threading
import time
import uuid
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH      = os.path.join(_PROJECT_ROOT, "data", "sessions.db")

# One connection, shared across threads.
# check_same_thread=False is safe here because all writes go through _lock.
_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_lock = threading.Lock()


def _init_db() -> None:
    """Create tables if they don't already exist."""
    with _lock:
        _conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT    PRIMARY KEY,
                created_ts  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,   -- "user" | "assistant"
                content     TEXT    NOT NULL,
                intent      TEXT,               -- nullable
                ts          REAL    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session_id
                ON turns (session_id, id);
        """)
        _conn.commit()


_init_db()

MAX_TURNS_STORED = 20   # hard cap per session (FIFO trim)
MAX_TURNS_INJECT = 5    # last N turns injected into prompts


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MEMORY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class SessionMemory:

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def create_session(self) -> str:
        """Generate a new UUID session and persist it."""
        sid = str(uuid.uuid4())
        with _lock:
            _conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_ts) VALUES (?, ?)",
                (sid, time.time()),
            )
            _conn.commit()
        return sid

    def session_exists(self, session_id: str) -> bool:
        """True if the session row exists in SQLite."""
        if not session_id:
            return False
        row = _conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row is not None

    def ensure_session(self, session_id: Optional[str]) -> str:
        """
        Return session_id if it exists, otherwise create a new one.
        Mirrors the old in-memory behaviour exactly.
        """
        if session_id and self.session_exists(session_id):
            return session_id
        return self.create_session()

    def delete_session(self, session_id: str) -> bool:
        """Delete all turns and the session row.  Returns False if not found."""
        if not self.session_exists(session_id):
            return False
        with _lock:
            _conn.execute("DELETE FROM turns    WHERE session_id = ?", (session_id,))
            _conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            _conn.commit()
        return True

    # ── Turn operations ────────────────────────────────────────────────────────

    def add_turn(
        self,
        session_id: str,
        role:       str,
        content:    str,
        intent:     Optional[str] = None,
    ) -> None:
        """
        Append a turn.  Auto-creates session row if missing.
        Trims to MAX_TURNS_STORED (FIFO) after insert.
        """
        with _lock:
            # Ensure session row exists (idempotent)
            _conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_ts) VALUES (?, ?)",
                (session_id, time.time()),
            )
            _conn.execute(
                "INSERT INTO turns (session_id, role, content, intent, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, intent, time.time()),
            )
            _conn.commit()

        self._trim(session_id)

    def _trim(self, session_id: str) -> None:
        """Delete oldest turns that exceed MAX_TURNS_STORED."""
        # Rows to delete = all rows except the newest MAX_TURNS_STORED
        rows = _conn.execute(
            """
            SELECT id FROM turns
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT -1 OFFSET ?
            """,
            (session_id, MAX_TURNS_STORED),
        ).fetchall()

        if rows:
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            with _lock:
                _conn.execute(
                    f"DELETE FROM turns WHERE id IN ({placeholders})", ids
                )
                _conn.commit()

    def get_history(self, session_id: str) -> list[dict]:
        """
        Return full stored history as list of dicts.
        Each dict: { role, content, intent, ts }
        """
        rows = _conn.execute(
            "SELECT role, content, intent, ts FROM turns "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

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
        rows = _conn.execute(
            "SELECT role, content FROM turns "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, MAX_TURNS_INJECT),
        ).fetchall()

        if not rows:
            return ""

        # Reverse to get chronological order
        lines = ["[CONVERSATION HISTORY]"]
        for row in reversed(rows):
            label = "User" if row["role"] == "user" else "Assistant"
            lines.append(f"{label}: {row['content']}")
        lines.append("[END HISTORY]")
        return "\n".join(lines)

    def turn_count(self, session_id: str) -> int:
        row = _conn.execute(
            "SELECT COUNT(*) AS c FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["c"]) if row else 0


# ── Singleton ──────────────────────────────────────────────────────────────────
session_memory = SessionMemory()