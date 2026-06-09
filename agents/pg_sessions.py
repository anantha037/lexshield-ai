"""
LexShield AI — Shared PostgreSQL Connection Pool
=================================================
Central psycopg v3 connection pool for all session-related tables
(sessions, turns, session_summaries, user_profiles, drafts, users).

Every module that previously used a raw sqlite3 connection against
data/sessions.db now imports ``get_conn`` from here instead.

The pool is configured with ``autocommit=True`` and ``dict_row`` so
that:
  - Individual statements auto-commit (no manual ``conn.commit()``).
  - Multi-statement atomicity uses ``with conn.transaction():``.
  - Rows are returned as plain dicts (drop-in for ``sqlite3.Row``).

Connection string
-----------------
Read from ``DATABASE_URL`` env var (same var as models/database.py).
Any SQLAlchemy-style prefix (``postgresql+psycopg://``) is normalised
to standard libpq format (``postgresql://``).
"""

import os
import re
import logging

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexshield:lexshield@localhost:5432/lexshield_sessions",
)

# Normalise SQLAlchemy dialect prefixes to plain libpq format
_conninfo = re.sub(r"^postgresql\+\w+://", "postgresql://", DATABASE_URL)


pool = ConnectionPool(
    conninfo=_conninfo,
    min_size=2,
    max_size=10,
    reconnect_timeout=10,
    max_waiting=5,
    kwargs={"autocommit": True, "row_factory": dict_row},
)


def get_conn():
    """Return a pooled connection (use as a context manager).

    Usage::

        with get_conn() as conn:
            row = conn.execute("SELECT ...").fetchone()
    """
    logger.debug("Acquiring connection from PostgreSQL pool")
    return pool.connection()
