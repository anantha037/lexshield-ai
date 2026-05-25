"""
LexShield AI — Authentication Module  (Session 6 — Final)
===========================================================
Pure SQLite backend — uses data/sessions.db (same file as session_memory).
No SQLAlchemy, no ORM — consistent with the rest of the project.

Tables managed here
--------------------
users (
  id TEXT PRIMARY KEY,           -- UUID4
  email TEXT UNIQUE NOT NULL,
  hashed_password TEXT NOT NULL,
  full_name TEXT,
  created_at REAL,               -- Unix timestamp
  last_login REAL                -- Unix timestamp, nullable
)

The sessions table already exists in memory.py; we ADD a user_id column
via ALTER TABLE IF NOT EXISTS pattern (SQLite-safe idempotent migration).

Endpoints
---------
POST /api/v1/auth/register — {email, password, full_name}
POST /api/v1/auth/login    — {email, password}
GET  /api/v1/auth/me       — Bearer token required

Auth dependency
---------------
get_current_user(token) -> dict  — use as FastAPI Depends() in any endpoint.
Returns {"id": ..., "email": ..., "full_name": ..., "created_at": ...}
or raises HTTP 401 if token is missing / invalid / user not found.

Verification
------------
curl -s -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"anantha@lexshield.ai","password":"test1234","full_name":"Anantha Krishnan K"}' | python -m json.tool
"""

import os
import sqlite3
import threading
import time
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE — reuse sessions.db from memory.py
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH      = os.path.join(_PROJECT_ROOT, "data", "sessions.db")

_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_lock = threading.Lock()


def _init_auth_tables() -> None:
    """Create users table and add user_id column to sessions if missing."""
    with _lock:
        _conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id              TEXT    PRIMARY KEY,
                email           TEXT    UNIQUE NOT NULL,
                hashed_password TEXT    NOT NULL,
                full_name       TEXT,
                created_at      REAL    NOT NULL,
                last_login      REAL
            );
        """)
        _conn.commit()

        # Idempotent ALTER: add user_id to sessions table (SQLite-safe)
        cols = [
            row["name"]
            for row in _conn.execute("PRAGMA table_info(sessions)").fetchall()
        ]
        if "user_id" not in cols:
            _conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
            _conn.commit()


_init_auth_tables()


# ═══════════════════════════════════════════════════════════════════════════════
# JWT CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

from jose import JWTError, jwt
import bcrypt
from datetime import datetime, timedelta

_SECRET_KEY  = os.getenv("JWT_SECRET_KEY", "lexshield-dev-secret-please-change-me-in-prod")
_ALGORITHM   = "HS256"
_EXPIRY_DAYS = 7


# ── Password helpers (direct bcrypt — no passlib) ────────────────────────────

def _hash_password(plain: str) -> str:
    # bcrypt max input is 72 bytes — truncate to be safe
    pw = plain.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    pw = plain.encode("utf-8")[:72]
    return bcrypt.checkpw(pw, hashed.encode("utf-8"))


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _create_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow() + timedelta(days=_EXPIRY_DAYS)
    payload = {"sub": user_id, "email": email, "exp": expire}
    token = jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)
    return token if isinstance(token, str) else token.decode("utf-8")


def _decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
    except JWTError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# USER CRUD (raw SQLite — no ORM)
# ═══════════════════════════════════════════════════════════════════════════════

def _create_user(email: str, password: str, full_name: str) -> dict:
    """Insert a new user. Raises ValueError if email already exists."""
    existing = _conn.execute(
        "SELECT id FROM users WHERE email = ?", (email.lower(),)
    ).fetchone()
    if existing:
        raise ValueError(f"Email already registered: {email}")

    user_id = str(uuid.uuid4())
    now     = time.time()
    hashed  = _hash_password(password)

    with _lock:
        _conn.execute(
            "INSERT INTO users (id, email, hashed_password, full_name, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, email.lower(), hashed, full_name.strip(), now),
        )
        _conn.commit()

    return {
        "id":         user_id,
        "email":      email.lower(),
        "full_name":  full_name.strip(),
        "created_at": now,
    }


def _authenticate_user(email: str, password: str) -> Optional[dict]:
    """Return user dict on success, None on bad credentials."""
    row = _conn.execute(
        "SELECT id, email, hashed_password, full_name, created_at FROM users WHERE email = ?",
        (email.lower(),),
    ).fetchone()
    if not row:
        return None
    if not _verify_password(password, row["hashed_password"]):
        return None

    # Update last_login
    with _lock:
        _conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (time.time(), row["id"]),
        )
        _conn.commit()

    return {
        "id":         row["id"],
        "email":      row["email"],
        "full_name":  row["full_name"],
        "created_at": row["created_at"],
    }


def _get_user_by_id(user_id: str) -> Optional[dict]:
    row = _conn.execute(
        "SELECT id, email, full_name, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
_bearer = HTTPBearer(auto_error=False)


# ── Request / Response schemas ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:     EmailStr
    password:  str = Field(..., min_length=6,  description="Minimum 6 characters")
    full_name: str = Field(..., min_length=2,  description="Full name")


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         dict


class MeResponse(BaseModel):
    id:         str
    email:      str
    full_name:  str
    created_at: float


# ── Dependency: get current user from Bearer token ─────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    FastAPI dependency — decodes JWT, returns user dict.
    Raises HTTP 401 if token missing, invalid, or user deleted.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = _get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """
    Optional auth dependency — returns user dict if valid token provided,
    None for anonymous requests.  Never raises.
    """
    if not credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest):
    """
    Create a new user account.
    Returns a JWT access token (7-day expiry) immediately.

    curl -s -X POST http://localhost:8000/api/v1/auth/register \\
      -H "Content-Type: application/json" \\
      -d '{"email":"user@example.com","password":"secret123","full_name":"Test User"}' | python -m json.tool
    """
    try:
        user = _create_user(req.email, req.password, req.full_name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    token = _create_token(user["id"], user["email"])
    return AuthResponse(
        access_token=token,
        user={
            "id":        user["id"],
            "email":     user["email"],
            "full_name": user["full_name"],
        },
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """
    Authenticate with email + password.
    Returns a JWT access token.

    curl -s -X POST http://localhost:8000/api/v1/auth/login \\
      -H "Content-Type: application/json" \\
      -d '{"email":"user@example.com","password":"secret123"}' | python -m json.tool
    """
    user = _authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_token(user["id"], user["email"])
    return AuthResponse(
        access_token=token,
        user={
            "id":        user["id"],
            "email":     user["email"],
            "full_name": user["full_name"],
        },
    )


@router.get("/me", response_model=MeResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    """
    Returns the currently authenticated user's profile.
    Requires: Authorization: Bearer <token>

    curl -s http://localhost:8000/api/v1/auth/me \\
      -H "Authorization: Bearer <your_token>" | python -m json.tool
    """
    return MeResponse(
        id         = current_user["id"],
        email      = current_user["email"],
        full_name  = current_user["full_name"],
        created_at = current_user["created_at"],
    )