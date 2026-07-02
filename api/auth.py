"""
LexShield AI — Authentication Module  (Session 7 — Email Verification + Password Reset)
=========================================================================================
PostgreSQL backend — uses shared psycopg v3 pool from pg_sessions.

Tables managed here
--------------------
users (
  id                   TEXT             PRIMARY KEY,  -- UUID4
  email                TEXT             UNIQUE NOT NULL,
  hashed_password      TEXT             NOT NULL,
  full_name            TEXT,
  created_at           DOUBLE PRECISION NOT NULL,
  last_login           DOUBLE PRECISION,
  is_email_verified    BOOLEAN          DEFAULT FALSE,
  verification_token   TEXT,
  reset_token          TEXT,
  reset_token_expiry   DOUBLE PRECISION
)

Endpoints
---------
POST /api/v1/auth/register              {email, password, full_name}
POST /api/v1/auth/login                 {email, password}
GET  /api/v1/auth/me                    Bearer token required
POST /api/v1/auth/request-verification  Bearer token required
GET  /api/v1/auth/verify-email          ?token=<token>
POST /api/v1/auth/forgot-password       {email}  always generic success
POST /api/v1/auth/reset-password        {token, new_password}

Email sending
-------------
Controlled by EMAIL_SENDING_ENABLED env var (default: false).
  false/unset : links are written to the server log — no credentials needed.
  true        : emails delivered via Resend SDK; requires RESEND_API_KEY + EMAIL_FROM.
"""

import os
import time
import uuid
import secrets
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field

from agents.pg_sessions import get_conn

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE — uses shared PostgreSQL pool
# ═══════════════════════════════════════════════════════════════════════════════


def _init_auth_tables() -> None:
    """Create users table and add all required columns if missing (idempotent)."""
    logger.info("Initializing auth tables in database")
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              TEXT             PRIMARY KEY,
                email           TEXT             UNIQUE NOT NULL,
                hashed_password TEXT             NOT NULL,
                full_name       TEXT,
                created_at      DOUBLE PRECISION NOT NULL,
                last_login      DOUBLE PRECISION
            )
        """)

        # Idempotent: add user_id to sessions table if not present
        conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id TEXT")

        # ── Email verification & password reset columns (Session 7) ──────────
        conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
            "is_email_verified BOOLEAN DEFAULT FALSE"
        )
        conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
            "verification_token TEXT"
        )
        conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
            "reset_token TEXT"
        )
        conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
            "reset_token_expiry DOUBLE PRECISION"
        )

        # Backfill: mark all pre-existing users as already verified so no one
        # currently registered gets locked out when this ships.
        # UPDATE on already-TRUE rows is a no-op in PostgreSQL — safe to re-run.
        conn.execute(
            "UPDATE users SET is_email_verified = TRUE "
            "WHERE is_email_verified IS NULL OR is_email_verified = FALSE"
        )
        logger.info("Auth table migration complete — existing users backfilled as verified")


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


def _hash_password(plain: str) -> str:
    pw = plain.encode("utf-8")[:72]   # bcrypt max input is 72 bytes
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    pw = plain.encode("utf-8")[:72]
    return bcrypt.checkpw(pw, hashed.encode("utf-8"))


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
# EMAIL — Resend SDK, gated by EMAIL_SENDING_ENABLED
# ═══════════════════════════════════════════════════════════════════════════════

_EMAIL_SENDING_ENABLED = os.getenv("EMAIL_SENDING_ENABLED", "false").lower() == "true"
_RESEND_API_KEY        = os.getenv("RESEND_API_KEY", "")
_EMAIL_FROM            = os.getenv("EMAIL_FROM", "noreply@lexshield.co.in")
# Frontend origin (Firebase Hosting) — used for links the user clicks in a browser.
# e.g. https://lexshield.co.in
_APP_BASE_URL          = os.getenv("APP_BASE_URL", "http://localhost:5173")
# Backend origin (Cloud Run) — used for links that call the FastAPI verify-email endpoint directly.
# e.g. https://your-service-xyz.run.app
_API_BASE_URL          = os.getenv("API_BASE_URL", "http://localhost:8000")

_RESET_TOKEN_TTL = 3600  # seconds — 1 hour


def _send_email(to: str, subject: str, body: str) -> None:
    """
    Send a plain-text email via the Resend SDK.

    When EMAIL_SENDING_ENABLED is false/unset, no email is sent — the full
    content is written to the server log at INFO level so the verification/reset
    flow is fully testable locally and on Cloud Run without credentials.

    On Resend delivery failure the error is logged but NOT re-raised.
    The token is already persisted; the user can request a new one.
    """
    if not _EMAIL_SENDING_ENABLED:
        logger.info(
            "[EMAIL DISABLED — set EMAIL_SENDING_ENABLED=true to send real mail]\n"
            "  To      : %s\n"
            "  Subject : %s\n"
            "  Body    :\n%s",
            to, subject, body,
        )
        return

    try:
        import resend           # pip install resend
        resend.api_key = _RESEND_API_KEY
        resend.Emails.send({
            "from":    _EMAIL_FROM,
            "to":      to,
            "subject": subject,
            "text":    body,
        })
        logger.info("Email sent via Resend to %s — %s", to, subject)
    except Exception as exc:
        logger.error("Resend delivery failed to %s: %s", to, exc)


# ═══════════════════════════════════════════════════════════════════════════════
# USER CRUD (raw psycopg — no ORM)
# ═══════════════════════════════════════════════════════════════════════════════

def _create_user(email: str, password: str, full_name: str) -> dict:
    """Insert a new user with a pending verification token. Raises ValueError if duplicate."""
    logger.info("Creating new user: %s", email)
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = %s", (email.lower(),)
        ).fetchone()
    if existing:
        raise ValueError(f"Email already registered: {email}")

    user_id            = str(uuid.uuid4())
    now                = time.time()
    hashed             = _hash_password(password)
    verification_token = secrets.token_urlsafe(32)   # 256 bits

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users "
            "(id, email, hashed_password, full_name, created_at, "
            " is_email_verified, verification_token) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (user_id, email.lower(), hashed, full_name.strip(),
             now, False, verification_token),
        )

    return {
        "id":                 user_id,
        "email":              email.lower(),
        "full_name":          full_name.strip(),
        "created_at":         now,
        "is_email_verified":  False,
        "verification_token": verification_token,
    }


def _authenticate_user(email: str, password: str) -> Optional[dict]:
    """Return user dict on success, None on bad credentials."""
    logger.info("Authenticating user: %s", email)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, email, hashed_password, full_name, created_at, is_email_verified "
            "FROM users WHERE email = %s",
            (email.lower(),),
        ).fetchone()
    if not row:
        return None
    if not _verify_password(password, row["hashed_password"]):
        return None

    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = %s WHERE id = %s",
            (time.time(), row["id"]),
        )

    return {
        "id":                row["id"],
        "email":             row["email"],
        "full_name":         row["full_name"],
        "created_at":        row["created_at"],
        "is_email_verified": bool(row["is_email_verified"]),
    }


def _get_user_by_id(user_id: str) -> Optional[dict]:
    logger.debug("Fetching user by id: %s", user_id)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, email, full_name, created_at, is_email_verified "
            "FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id":                row["id"],
        "email":             row["email"],
        "full_name":         row["full_name"],
        "created_at":        row["created_at"],
        "is_email_verified": bool(row["is_email_verified"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
_bearer = HTTPBearer(auto_error=False)


# ── Request / Response schemas ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:     EmailStr
    password:  str = Field(..., min_length=6, description="Minimum 6 characters")
    full_name: str = Field(..., min_length=2, description="Full name")


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         dict


class MeResponse(BaseModel):
    id:                str
    email:             str
    full_name:         str
    created_at:        float
    is_email_verified: bool


class MessageResponse(BaseModel):
    message: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token:        str
    new_password: str = Field(..., min_length=6, description="Minimum 6 characters")


# ── Dependency: get current user from Bearer token ─────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """FastAPI dependency — decodes JWT, returns user dict. Raises 401 on any failure."""
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
    """Optional auth dependency — returns user dict or None. Never raises."""
    if not credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        logger.exception("HTTPException in get_optional_user")
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest):
    """
    Create a new user account.
    Returns a JWT access token (7-day expiry) immediately.
    A verification email is sent (or logged if EMAIL_SENDING_ENABLED=false).
    """
    try:
        logger.info("Registering user: %s", req.email)
        user = _create_user(req.email, req.password, req.full_name)
    except ValueError as exc:
        logger.warning("Registration rejected for %s: %s", req.email, exc)
        raise HTTPException(status_code=409, detail=str(exc))

    verification_link = (
        f"{_API_BASE_URL}/api/v1/auth/verify-email"
        f"?token={user['verification_token']}"
    )
    _send_email(
        to      = user["email"],
        subject = "Verify your LexShield email address",
        body    = (
            f"Hi {user['full_name']},\n\n"
            f"Welcome to LexShield AI! Please verify your email address "
            f"by clicking the link below:\n\n"
            f"  {verification_link}\n\n"
            f"This link does not expire.\n\n"
            f"If you did not create this account, you can safely ignore this email.\n\n"
            f"— The LexShield Team"
        ),
    )

    token = _create_token(user["id"], user["email"])
    return AuthResponse(
        access_token=token,
        user={
            "id":                user["id"],
            "email":             user["email"],
            "full_name":         user["full_name"],
            "is_email_verified": user["is_email_verified"],
        },
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """
    Authenticate with email + password. Returns a JWT access token.
    Unverified users are allowed to log in — is_email_verified is exposed
    in the response so the frontend can show a prompt without blocking access.
    """
    logger.info("Login request for: %s", req.email)
    user = _authenticate_user(req.email, req.password)
    if not user:
        logger.debug("Login failed: invalid credentials for %s", req.email)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_token(user["id"], user["email"])
    return AuthResponse(
        access_token=token,
        user={
            "id":                user["id"],
            "email":             user["email"],
            "full_name":         user["full_name"],
            "is_email_verified": user["is_email_verified"],
        },
    )


@router.get("/me", response_model=MeResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    """Returns the authenticated user's profile including email verification status."""
    logger.info("Fetching profile for: %s", current_user["email"])
    return MeResponse(
        id                = current_user["id"],
        email             = current_user["email"],
        full_name         = current_user["full_name"],
        created_at        = current_user["created_at"],
        is_email_verified = current_user["is_email_verified"],
    )


@router.post("/request-verification", response_model=MessageResponse)
def request_verification(current_user: dict = Depends(get_current_user)):
    """
    (Re-)send a verification email for the current user.
    Generates a fresh token on every call (previous token is invalidated).
    Returns success immediately if the user is already verified.
    Requires: Authorization: Bearer <token>
    """
    if current_user["is_email_verified"]:
        return MessageResponse(message="Email address is already verified.")

    new_token = secrets.token_urlsafe(32)
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET verification_token = %s WHERE id = %s",
            (new_token, current_user["id"]),
        )

    verification_link = (
        f"{_API_BASE_URL}/api/v1/auth/verify-email?token={new_token}"
    )
    _send_email(
        to      = current_user["email"],
        subject = "Verify your LexShield email address",
        body    = (
            f"Hi {current_user['full_name']},\n\n"
            f"Please verify your email address by clicking the link below:\n\n"
            f"  {verification_link}\n\n"
            f"This link does not expire.\n\n"
            f"If you did not request this, you can safely ignore this email.\n\n"
            f"— The LexShield Team"
        ),
    )
    return MessageResponse(message="Verification email sent. Please check your inbox.")


@router.get("/verify-email", response_model=MessageResponse)
def verify_email(token: str = Query(..., description="Email verification token")):
    """
    Confirm email ownership via the token from the verification email.
    Token is cleared after use — clicking the link a second time returns 400.
    GET /api/v1/auth/verify-email?token=<token>
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, is_email_verified FROM users WHERE verification_token = %s",
            (token,),
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=400,
            detail="Invalid or already-used verification token.",
        )

    if row["is_email_verified"]:
        # Already verified — clear stale token, return friendly message
        with get_conn() as conn:
            conn.execute(
                "UPDATE users SET verification_token = NULL WHERE id = %s",
                (row["id"],),
            )
        return MessageResponse(message="Email address is already verified.")

    with get_conn() as conn:
        conn.execute(
            "UPDATE users "
            "SET is_email_verified = TRUE, verification_token = NULL "
            "WHERE id = %s",
            (row["id"],),
        )

    logger.info("Email verified for user: %s", row["id"])
    return MessageResponse(message="Email address verified successfully. Thank you!")


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(req: ForgotPasswordRequest):
    """
    Request a password reset link.
    Always returns the same generic message regardless of whether the email
    is registered — prevents email enumeration attacks.
    Reset token expires in 1 hour and is single-use.
    """
    _GENERIC = MessageResponse(
        message=(
            "If an account with that email address exists, "
            "a password reset link has been sent."
        )
    )

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, full_name FROM users WHERE email = %s",
            (req.email.lower(),),
        ).fetchone()

    if not row:
        logger.info("forgot-password: no account for %s — generic response returned", req.email)
        return _GENERIC

    reset_token  = secrets.token_urlsafe(32)
    token_expiry = time.time() + _RESET_TOKEN_TTL

    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET reset_token = %s, reset_token_expiry = %s WHERE id = %s",
            (reset_token, token_expiry, row["id"]),
        )

    # The reset link points to the frontend page that collects the new password.
    # The frontend reads ?token= from the URL and POSTs it to /reset-password.
    reset_link = f"{_APP_BASE_URL}/reset-password?token={reset_token}"
    _send_email(
        to      = req.email.lower(),
        subject = "Reset your LexShield password",
        body    = (
            f"Hi {row['full_name']},\n\n"
            f"We received a request to reset your LexShield AI password. "
            f"Click the link below to choose a new one:\n\n"
            f"  {reset_link}\n\n"
            f"This link expires in 1 hour.\n\n"
            f"If you did not request a password reset, your password has NOT been "
            f"changed — you can safely ignore this email.\n\n"
            f"— The LexShield Team"
        ),
    )

    return _GENERIC


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(req: ResetPasswordRequest):
    """
    Complete a password reset using the token from the reset email.
    Token must be valid and unexpired (1-hour TTL).
    Invalidated immediately after use (single-use).
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, reset_token_expiry FROM users WHERE reset_token = %s",
            (req.token,),
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=400,
            detail="Invalid or already-used password reset token.",
        )

    if time.time() > (row["reset_token_expiry"] or 0):
        # Expired — wipe token so it cannot be retried
        with get_conn() as conn:
            conn.execute(
                "UPDATE users SET reset_token = NULL, reset_token_expiry = NULL "
                "WHERE id = %s",
                (row["id"],),
            )
        raise HTTPException(
            status_code=400,
            detail="Password reset token has expired. Please request a new one.",
        )

    new_hashed = _hash_password(req.new_password)
    with get_conn() as conn:
        conn.execute(
            "UPDATE users "
            "SET hashed_password = %s, reset_token = NULL, reset_token_expiry = NULL "
            "WHERE id = %s",
            (new_hashed, row["id"]),
        )

    logger.info("Password reset successfully for user: %s", row["id"])
    return MessageResponse(
        message="Password reset successfully. You can now log in with your new password."
    )