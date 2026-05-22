"""
LexShield AI — Session Memory  (SQLite backend)
================================================
Replaces the in-memory dict with a persistent SQLite store.
Survives server restarts.  Same public interface as the old
in-memory version so orchestrator.py needs no changes.

Tables
------
sessions (session_id PK, created_ts, user_id TEXT nullable)
turns    (id PK, session_id FK, role, content, intent, ts)

Uses the SAME data/sessions.db file as the LangGraph SqliteSaver
checkpointer — the table names do not overlap (LangGraph uses
checkpoints / checkpoint_blobs / checkpoint_writes).

Public interface (unchanged from in-memory version)
----------------------------------------------------
session_memory.ensure_session(sid)              → str
session_memory.session_exists(sid)              → bool
session_memory.create_session()                 → str
session_memory.delete_session(sid)              → bool
session_memory.add_turn(sid, role, content, intent=None)
session_memory.get_history(sid)                 → list[dict]  (ALL turns)
session_memory.get_context_block(sid)           → str
session_memory.turn_count(sid)                  → int

New in Session 6 (auth)
------------------------
session_memory.link_session_to_user(sid, uid)   → None
session_memory.get_user_sessions(uid)           → list[dict]
"""

import os
import sqlite3
import threading
import time
import uuid
import json
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
    """Create tables and run idempotent migrations."""
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

            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id       TEXT    NOT NULL,
                summary_text     TEXT    NOT NULL,
                created_at       REAL    NOT NULL,
                turn_range_start INTEGER NOT NULL,
                turn_range_end   INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_summaries_session_id
                ON session_summaries (session_id, created_at);
        """)
        _conn.commit()

        # Idempotent migration: add user_id column to sessions if not present
        cols = [
            row["name"]
            for row in _conn.execute("PRAGMA table_info(sessions)").fetchall()
        ]
        if "user_id" not in cols:
            _conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
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

    # ── User ↔ Session linking (Session 6) ────────────────────────────────────

    def link_session_to_user(self, session_id: str, user_id: str) -> None:
        """
        Associate a session with an authenticated user.
        Idempotent — safe to call on every authenticated request.
        """
        with _lock:
            _conn.execute(
                "UPDATE sessions SET user_id = ? WHERE session_id = ?",
                (user_id, session_id),
            )
            _conn.commit()

    def get_user_sessions(self, user_id: str) -> list[dict]:
        """
        Return all sessions owned by user_id, ordered by most-recently-active first.

        Each dict:
          session_id, created_at, last_active, turn_count, first_message

        first_message — first user turn content truncated to 60 chars.
        last_active   — timestamp of the most recent turn (or session creation).
        """
        rows = _conn.execute(
            """
            SELECT
                s.session_id,
                s.created_ts                                            AS created_at,
                COALESCE(MAX(t.ts), s.created_ts)                      AS last_active,
                COUNT(t.id)                                             AS turn_count,
                (
                    SELECT SUBSTR(t2.content, 1, 60)
                    FROM   turns t2
                    WHERE  t2.session_id = s.session_id
                    AND    t2.role = 'user'
                    ORDER  BY t2.id ASC
                    LIMIT  1
                )                                                       AS first_message
            FROM   sessions s
            LEFT JOIN turns t ON t.session_id = s.session_id
            WHERE  s.user_id = ?
            GROUP  BY s.session_id
            ORDER  BY last_active DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

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
        rows = _conn.execute(
            """
            SELECT id, role, content FROM turns
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT -1 OFFSET ?
            """,
            (session_id, MAX_TURNS_STORED),
        ).fetchall()

        if rows:
            ids = [r["id"] for r in rows]
            
            # Summarize before deleting
            # Reverse to get chronological order for the summary
            turns_to_summarize = [
                {"id": r["id"], "role": r["role"], "content": r["content"]} 
                for r in reversed(rows)
            ]
            
            threading.Thread(
                target=self._summarize_turns, 
                args=(session_id, turns_to_summarize),
                daemon=True
            ).start()

            placeholders = ",".join("?" * len(ids))
            with _lock:
                _conn.execute(
                    f"DELETE FROM turns WHERE id IN ({placeholders})", ids
                )
                _conn.commit()

    def _summarize_turns(self, session_id: str, turns: list[dict]) -> None:
        """Background thread to summarize older turns via Groq LLM."""
        from rag.llm import llm
        
        transcript = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)
        
        system_prompt = (
            "You are an expert legal AI memory summarizer. "
            "Summarize the following conversation into a single dense paragraph. "
            "CRITICAL INSTRUCTIONS:\n"
            "- Preserve all named legal acts (IPC, CrPC, BNS, etc.), section numbers, and document types.\n"
            "- Preserve the user's stated jurisdiction, context, and intent.\n"
            "- DO NOT omit specific factual details, case names, or entities."
        )
        
        try:
            summary = llm.generate(prompt=transcript, system_prompt=system_prompt, max_tokens=300)
            
            start_id = turns[0]['id']
            end_id = turns[-1]['id']
            
            with _lock:
                _conn.execute(
                    """
                    INSERT INTO session_summaries (session_id, summary_text, created_at, turn_range_start, turn_range_end)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, summary, time.time(), start_id, end_id)
                )
                _conn.commit()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"[SessionMemory] Failed to summarize turns: {e}")

    def get_summary(self, session_id: str) -> str:
        """Fetch all summaries for a session, concatenated chronologically."""
        rows = _conn.execute(
            """
            SELECT summary_text FROM session_summaries
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,)
        ).fetchall()
        
        if not rows:
            return ""
            
        return "\n\n".join(r["summary_text"] for r in rows)

    def get_history(self, session_id: str) -> list[dict]:
        """
        Return FULL stored history as list of dicts (no 5-turn cap).
        Used by frontend to restore complete chat history when a user
        reopens an old session.

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


# ═══════════════════════════════════════════════════════════════════════════════
# USER PROFILE MEMORY (Task 2)
# ═══════════════════════════════════════════════════════════════════════════════

class ProfileMemory:
    def __init__(self):
        with _lock:
            _conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    session_id           TEXT PRIMARY KEY,
                    preferred_domain     TEXT,
                    jurisdiction         TEXT,
                    frequent_acts        TEXT,
                    frequent_doc_types   TEXT,
                    act_frequencies      TEXT,
                    doc_type_frequencies TEXT,
                    last_updated         REAL
                );
            """)
            _conn.commit()

    def update_profile(self, session_id: str, intent: str, entities: dict):
        row = _conn.execute("SELECT * FROM user_profiles WHERE session_id = ?", (session_id,)).fetchone()
        
        if row:
            profile = dict(row)
            act_freq = json.loads(profile['act_frequencies'] or '{}')
            doc_freq = json.loads(profile['doc_type_frequencies'] or '{}')
            jurisdictions = set(json.loads(profile['jurisdiction'] or '[]'))
            domains = set(json.loads(profile['preferred_domain'] or '[]'))
        else:
            act_freq = {}
            doc_freq = {}
            jurisdictions = set()
            domains = set()

        for loc in entities.get("locations", []):
            jurisdictions.add(loc)
            
        for act in entities.get("acts", []):
            act_freq[act] = act_freq.get(act, 0) + 1
            
        for doc in entities.get("doc_types", []):
            doc_freq[doc] = doc_freq.get(doc, 0) + 1
            
        for dom in entities.get("domains", []):
            domains.add(dom)

        frequent_acts = [act for act, count in act_freq.items() if count >= 3]
        frequent_doc_types = [dt for dt, count in doc_freq.items() if count >= 3]

        with _lock:
            _conn.execute("""
                INSERT INTO user_profiles (
                    session_id, preferred_domain, jurisdiction, frequent_acts, 
                    frequent_doc_types, act_frequencies, doc_type_frequencies, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    preferred_domain=excluded.preferred_domain,
                    jurisdiction=excluded.jurisdiction,
                    frequent_acts=excluded.frequent_acts,
                    frequent_doc_types=excluded.frequent_doc_types,
                    act_frequencies=excluded.act_frequencies,
                    doc_type_frequencies=excluded.doc_type_frequencies,
                    last_updated=excluded.last_updated
            """, (
                session_id,
                json.dumps(list(domains)),
                json.dumps(list(jurisdictions)),
                json.dumps(frequent_acts),
                json.dumps(frequent_doc_types),
                json.dumps(act_freq),
                json.dumps(doc_freq),
                time.time()
            ))
            _conn.commit()

    def get_profile_block(self, session_id: str) -> str:
        row = _conn.execute("SELECT * FROM user_profiles WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return ""
            
        domains = json.loads(row['preferred_domain'] or '[]')
        jurisdictions = json.loads(row['jurisdiction'] or '[]')
        acts = json.loads(row['frequent_acts'] or '[]')
        docs = json.loads(row['frequent_doc_types'] or '[]')
        
        if not any([domains, jurisdictions, acts, docs]):
            return ""
            
        lines = ["[USER PROFILE]"]
        if domains: lines.append(f"Preferred Domains: {', '.join(domains)}")
        if jurisdictions: lines.append(f"Jurisdiction: {', '.join(jurisdictions)}")
        if acts: lines.append(f"Frequently Mentioned Acts: {', '.join(acts)}")
        if docs: lines.append(f"Frequently Used Document Types: {', '.join(docs)}")
        lines.append("[END PROFILE]")
        
        return "\n".join(lines)

profile_memory = ProfileMemory()