"""
LexShield AI — Session Memory  (PostgreSQL backend)
====================================================
Persistent session store backed by PostgreSQL via psycopg v3.
Survives server restarts.  Same public interface as the previous
SQLite version so orchestrator.py needs no changes.

Tables
------
sessions (session_id PK, created_ts, user_id TEXT nullable)
turns    (id PK SERIAL, session_id FK, role, content, intent, ts, citation_status)
session_summaries (session_id, summary_text, created_at, turn_range_start, turn_range_end)
user_profiles (session_id PK, preferred_domain, jurisdiction, ...)

Public interface (unchanged from previous version)
----------------------------------------------------
session_memory.ensure_session(sid)              -> str
session_memory.session_exists(sid)              -> bool
session_memory.create_session()                 -> str
session_memory.delete_session(sid)              -> bool
session_memory.add_turn(sid, role, content, intent=None, citation_status=None)
session_memory.get_history(sid)                 -> list[dict]  (ALL turns)
session_memory.get_context_block(sid)           -> str
session_memory.turn_count(sid)                  -> int

New in Session 6 (auth)
------------------------
session_memory.link_session_to_user(sid, uid)   -> None
session_memory.get_user_sessions(uid)           -> list[dict]
"""

import threading
import time
import uuid
import json
from contextvars import ContextVar
from typing import Optional
import logging
import psycopg

from agents.pg_sessions import get_conn
from rag.embedder import embedder

logger = logging.getLogger(__name__)

# ── Per-request session context carrier ───────────────────────────────────────
# Set by legal_rag_node before entering the RAG chain so that query_rewriter.py
# can read the active session_id without receiving it as a parameter.
# ContextVar is safe across asyncio tasks and native threads.
_active_session_id: ContextVar[str] = ContextVar("active_session_id", default="")


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def _init_db() -> None:
    """Create tables and run idempotent migrations."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT    PRIMARY KEY,
                created_ts  DOUBLE PRECISION NOT NULL,
                user_id     TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT    NOT NULL,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                intent          TEXT,
                ts              DOUBLE PRECISION NOT NULL,
                citation_status TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_turns_session_id
                ON turns (session_id, id)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id       TEXT    NOT NULL,
                summary_text     TEXT    NOT NULL,
                created_at       DOUBLE PRECISION NOT NULL,
                turn_range_start INTEGER NOT NULL,
                turn_range_end   INTEGER NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summaries_session_id
                ON session_summaries (session_id, created_at)
        """)

        # Idempotent migrations (no-ops on fresh DBs, safe on older schemas)
        conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id TEXT")
        conn.execute("ALTER TABLE turns ADD COLUMN IF NOT EXISTS citation_status TEXT")
        conn.execute("ALTER TABLE turns ADD COLUMN IF NOT EXISTS embedding vector(384)")


_init_db()

MAX_TURNS_STORED = 20   # hard cap per session (FIFO trim)
MAX_TURNS_INJECT = 5    # last N turns injected into prompts
GENERAL_INTENT_TURNS = 3  # reduced turn count for trivial/general queries


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MEMORY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

# One turn = one user→assistant round-trip.  last_act expires after this many.
_LAST_ACT_TTL_TURNS = 10


class SessionMemory:

    def __init__(self):
        self.sessions = {}

    # ── Per-request context carrier (for query_rewriter) ──────────────────────

    def set_session_context(self, session_id: str) -> None:
        """
        Bind session_id to the current execution context (ContextVar).
        Call this at the start of every legal_rag_node invocation before the
        RAG chain runs.  query_rewriter.py reads it via get_session_context().
        """
        logger.debug(f"Setting session context to {session_id}")
        _active_session_id.set(session_id)

    def get_session_context(self) -> str:
        """Return the session_id bound to the current execution context."""
        logger.debug("Getting session context")
        return _active_session_id.get()

    # ── Last-act / last-section persistence (in-memory, TTL-capped) ───────────

    def set_last_act(
        self,
        session_id: str,
        act:        str,
        section:    str = "",
    ) -> None:
        """
        Persist the most-recently resolved act (and optionally section) for a
        session.  Called by legal_rag_node after a successful RAG response.

        The entry expires automatically after _LAST_ACT_TTL_TURNS turns so
        stale context from an old topic does not bleed into unrelated follow-ups.
        The turn counter is reset whenever the caller explicitly provides a new
        act (i.e. call set_last_act again with the new act name).
        """
        if not session_id or not act:
            return
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id]["last_act"]          = act.strip()
        self.sessions[session_id]["last_section"]      = section.strip()
        self.sessions[session_id]["last_act_ttl"]      = _LAST_ACT_TTL_TURNS
        logger.debug(f"set_last_act session={session_id[:8]}... act={act!r} section={section!r}")

    def get_last_act(self, session_id: str) -> tuple[str, str]:
        """
        Return (last_act, last_section) for the session, decrementing the TTL
        counter on each read.  Returns ('', '') when expired or absent.
        """
        if not session_id:
            return "", ""
        sess = self.sessions.get(session_id, {})
        act  = sess.get("last_act", "")
        if not act:
            return "", ""
        ttl = sess.get("last_act_ttl", 0)
        if ttl <= 0:
            # Expired — purge
            sess.pop("last_act",     None)
            sess.pop("last_section", None)
            sess.pop("last_act_ttl", None)
            return "", ""
        # Decrement TTL on each read (each retrieval attempt = one turn usage)
        self.sessions[session_id]["last_act_ttl"] = ttl - 1
        return act, sess.get("last_section", "")

    def clear_last_act(self, session_id: str) -> None:
        """Explicitly clear last_act — called when the user switches topics."""
        sess = self.sessions.get(session_id, {})
        sess.pop("last_act",     None)
        sess.pop("last_section", None)
        sess.pop("last_act_ttl", None)

    # ── Entity scratchpad (pre-existing) ───────────────────────────────────────

    def store_last_entities(self, session_id: str, entities: list[str]) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id]["last_entities"] = entities

    def get_last_scratchpad_entities(self, session_id: str) -> list[str]:
        return self.sessions.get(session_id, {}).get("last_entities", [])

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def create_session(self) -> str:
        """Generate a new UUID session and persist it."""
        sid = str(uuid.uuid4())
        logger.info(f"Creating new session: {sid}")
        try:
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, created_ts) VALUES (%s, %s) "
                    "ON CONFLICT (session_id) DO NOTHING",
                    (sid, time.time()),
                )
        except psycopg.OperationalError:
            logger.warning("Database connection failed. Reopening connection and retrying...")
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, created_ts) VALUES (%s, %s) "
                    "ON CONFLICT (session_id) DO NOTHING",
                    (sid, time.time()),
                )
        return sid

    def session_exists(self, session_id: str) -> bool:
        """True if the session row exists in PostgreSQL."""
        if not session_id:
            return False
        with get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = %s",
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
        logger.info(f"Deleting session: {session_id}")
        if not self.session_exists(session_id):
            logger.warning(f"Session {session_id} not found for deletion")
            return False
        with get_conn() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM turns    WHERE session_id = %s", (session_id,))
                conn.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        return True

    # ── User ↔ Session linking (Session 6) ────────────────────────────────────

    def link_session_to_user(self, session_id: str, user_id: str) -> None:
        """
        Associate a session with an authenticated user.
        Idempotent — safe to call on every authenticated request.
        """
        with get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET user_id = %s WHERE session_id = %s",
                (user_id, session_id),
            )

    def get_user_sessions(self, user_id: str) -> list[dict]:
        """
        Return all sessions owned by user_id, ordered by most-recently-active first.

        Each dict:
          session_id, created_at, last_active, turn_count, first_message

        first_message — first user turn content truncated to 60 chars.
        last_active   — timestamp of the most recent turn (or session creation).
        """
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.created_ts                                            AS created_at,
                    COALESCE(MAX(t.ts), s.created_ts)                      AS last_active,
                    COUNT(t.id)                                             AS turn_count,
                    (
                        SELECT LEFT(t2.content, 60)
                        FROM   turns t2
                        WHERE  t2.session_id = s.session_id
                        AND    t2.role = 'user'
                        ORDER  BY t2.id ASC
                        LIMIT  1
                    )                                                       AS first_message
                FROM   sessions s
                LEFT JOIN turns t ON t.session_id = s.session_id
                WHERE  s.user_id = %s
                GROUP  BY s.session_id, s.created_ts
                ORDER  BY last_active DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Turn operations ────────────────────────────────────────────────────────

    def add_turn(
        self,
        session_id:      str,
        role:            str,
        content:         str,
        intent:          Optional[str] = None,
        citation_status: Optional[str] = None,  # FIX: new param
    ) -> None:
        """
        Append a turn.  Auto-creates session row if missing.
        Trims to MAX_TURNS_STORED (FIFO) after insert.
        """
        # Best-effort embedding — NULL on any failure
        emb_str = None
        try:
            if content and content.strip():
                vec = embedder.embed_single(content)
                emb_str = '[' + ','.join(map(str, vec)) + ']'
        except Exception:
            logger.warning("Failed to embed turn content; storing with embedding=NULL",
                           exc_info=True)

        with get_conn() as conn:
            # Ensure session row exists (idempotent)
            conn.execute(
                "INSERT INTO sessions (session_id, created_ts) VALUES (%s, %s) "
                "ON CONFLICT (session_id) DO NOTHING",
                (session_id, time.time()),
            )
            conn.execute(
                "INSERT INTO turns (session_id, role, content, intent, ts, citation_status, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (session_id, role, content, intent, time.time(), citation_status, emb_str),
            )

        self._trim(session_id)

    def _trim(self, session_id: str) -> None:
        """Delete oldest turns that exceed MAX_TURNS_STORED."""
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content FROM turns
                WHERE session_id = %s
                ORDER BY id DESC
                OFFSET %s
                """,
                (session_id, MAX_TURNS_STORED),
            ).fetchall()

        if rows:
            ids = [r["id"] for r in rows]

            # Summarize before deleting
            turns_to_summarize = [
                {"id": r["id"], "role": r["role"], "content": r["content"]}
                for r in reversed(rows)
            ]

            threading.Thread(
                target=self._summarize_turns,
                args=(session_id, turns_to_summarize),
                daemon=True
            ).start()

            with get_conn() as conn:
                conn.execute(
                    "DELETE FROM turns WHERE id = ANY(%s)", (ids,)
                )

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

            with get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO session_summaries (session_id, summary_text, created_at, turn_range_start, turn_range_end)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_id, summary, time.time(), start_id, end_id)
                )
        except Exception as e:
            logger.exception("Failed to summarize turns")

    def get_summary(self, session_id: str) -> str:
        """Fetch all summaries for a session, concatenated chronologically."""
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT summary_text FROM session_summaries
                WHERE session_id = %s
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

        Each dict: { role, content, intent, ts, citation_status }
        """
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, intent, ts, citation_status FROM turns "
                "WHERE session_id = %s ORDER BY id ASC",
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
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM turns "
                "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                (session_id, MAX_TURNS_INJECT),
            ).fetchall()

        if not rows:
            return ""

        lines = ["[CONVERSATION HISTORY]"]
        for row in reversed(rows):
            label = "User" if row["role"] == "user" else "Assistant"
            lines.append(f"{label}: {row['content']}")
        lines.append("[END HISTORY]")
        return "\n".join(lines)

    def get_relevant_context(
            self,
            session_id:    str,
            current_query: str,
            top_k:         int = 5,
        ) -> str:
        """
        Semantic context retrieval via pgvector cosine similarity.
        Falls back to BM25 keyword scoring on any failure.

        Fallback to recency-based context when:
          - query tokenizes to <=2 tokens (trivial / greeting)
          - any exception during pgvector or BM25 scoring
        """
        from rag.bm25_retriever import tokenize

        query_tokens = tokenize(current_query)
        if len(query_tokens) <= 2:
            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT role, content FROM turns "
                    "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                    (session_id, GENERAL_INTENT_TURNS),
                ).fetchall()
            if not rows:
                return ""
            lines = ["[CONVERSATION HISTORY]"]
            for row in reversed(rows):
                label = "User" if row["role"] == "user" else "Assistant"
                lines.append(f"{label}: {row['content']}")
            lines.append("[END HISTORY]")
            return "\n".join(lines)

        # ── Primary: pgvector cosine similarity ───────────────────────────
        # Rank by USER-turn distance only (not individual rows), then attach
        # each matched user turn's paired assistant reply (the next row by id
        # in the same session). This avoids stitching together unrelated
        # user/assistant rows from different exchanges into a single,
        # incoherent printed pair.
        try:
            query_vec = embedder.embed_single(current_query)
            query_emb_str = '[' + ','.join(map(str, query_vec)) + ']'

            with get_conn() as conn:
                user_rows = conn.execute(
                    "SELECT id, role, content FROM turns "
                    "WHERE session_id = %s AND role = 'user' AND embedding IS NOT NULL "
                    "ORDER BY embedding <=> %s "
                    "LIMIT %s",
                    (session_id, query_emb_str, top_k),
                ).fetchall()

                exchange_rows = []
                if user_rows:
                    matched_ids = [r["id"] for r in user_rows]
                    reply_ids = [i + 1 for i in matched_ids]
                    reply_rows = conn.execute(
                        "SELECT id, role, content FROM turns "
                        "WHERE session_id = %s AND id = ANY(%s) AND role = 'assistant'",
                        (session_id, reply_ids),
                    ).fetchall()
                    replies_by_id = {r["id"]: r for r in reply_rows}

                    for u in user_rows:
                        exchange_rows.append(u)
                        reply = replies_by_id.get(u["id"] + 1)
                        if reply:
                            exchange_rows.append(reply)

            if exchange_rows:
                # Re-sort chronologically (oldest first) to match BM25 output
                exchange_rows.sort(key=lambda r: r["id"])
                lines = ["[CONVERSATION HISTORY]"]
                for r in exchange_rows:
                    label = "User" if r["role"] == "user" else "Assistant"
                    lines.append(f"{label}: {r['content']}")
                lines.append("[END HISTORY]")
                return "\n".join(lines)
        except Exception:
            logger.debug("pgvector retrieval failed; falling back to BM25",
                         exc_info=True)

        # ── Fallback: BM25 keyword scoring (byte-identical to original) ──
        try:
            from rank_bm25 import BM25Okapi

            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT id, role, content FROM turns "
                    "WHERE session_id = %s ORDER BY id ASC",
                    (session_id,),
                ).fetchall()

            if len(rows) < top_k:
                return self.get_context_block(session_id)

            turn_data = [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]
            corpus_tokenized = [tokenize(t["content"]) for t in turn_data]
            bm25 = BM25Okapi(corpus_tokenized)

            scores = bm25.get_scores(query_tokens)

            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in indexed_scores[:top_k]]

            top_indices.sort()

            lines = ["[CONVERSATION HISTORY]"]
            for idx in top_indices:
                t = turn_data[idx]
                label = "User" if t["role"] == "user" else "Assistant"
                lines.append(f"{label}: {t['content']}")
            lines.append("[END HISTORY]")
            return "\n".join(lines)

        except Exception:
            return self.get_context_block(session_id)

    def turn_count(self, session_id: str) -> int:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM turns WHERE session_id = %s",
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
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    session_id           TEXT PRIMARY KEY,
                    preferred_domain     TEXT,
                    jurisdiction         TEXT,
                    frequent_acts        TEXT,
                    frequent_doc_types   TEXT,
                    act_frequencies      TEXT,
                    doc_type_frequencies TEXT,
                    last_updated         DOUBLE PRECISION
                )
            """)

    def update_profile(self, session_id: str, intent: str, entities: dict):
        logger.info(f"Updating profile for session: {session_id}")
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM user_profiles WHERE session_id = %s", (session_id,)).fetchone()

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

        with get_conn() as conn:
            conn.execute("""
                INSERT INTO user_profiles (
                    session_id, preferred_domain, jurisdiction, frequent_acts,
                    frequent_doc_types, act_frequencies, doc_type_frequencies, last_updated
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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

    def get_profile_block(self, session_id: str) -> str:
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM user_profiles WHERE session_id = %s", (session_id,)).fetchone()
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