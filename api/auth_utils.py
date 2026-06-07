"""
LexShield AI — Auth Utilities
================================
Password hashing with bcrypt.
JWT token creation and verification.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
import logging

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY      = os.getenv("JWT_SECRET_KEY", "lexshield-jwt-secret-change-in-production")
ALGORITHM       = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ── Password hashing ───────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(plain: str) -> str:
    logger.debug("Hashing password")
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    logger.debug("Verifying password")
    return pwd_context.verify(plain, hashed)


# ── JWT ────────────────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    logger.info(f"Creating access token for data keys: {list(data.keys())}")
    to_encode = data.copy()
    expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded if isinstance(encoded, str) else encoded.decode("utf-8")


def decode_access_token(token: str) -> Optional[dict]:
    """Returns the payload dict, or None if invalid/expired."""
    logger.debug("Decoding access token")
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        logger.exception("Failed to decode JWT token")
        return None