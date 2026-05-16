"""
LexShield AI — Auth API Router
================================
POST /api/v1/auth/register  — create account
POST /api/v1/auth/login     — get JWT token
GET  /api/v1/auth/me        — get current user (requires token)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from models.database import get_db
from models.user     import User
from api.auth_utils  import hash_password, verify_password, create_access_token, decode_access_token

router  = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
bearer  = HTTPBearer()


# ── Request / Response schemas ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:     EmailStr
    password:  str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id:        int
    email:     str
    full_name: str
    is_active: bool


# ── Dependency: get current user from Bearer token ─────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    token   = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return user


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Create a new user account.
    Returns a JWT access token immediately (auto-login after register).
    """
    existing = db.query(User).filter(User.email == req.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    user = User(
        email           = req.email.lower(),
        full_name       = req.full_name.strip(),
        hashed_password = hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id), "email": user.email})

    return AuthResponse(
        access_token = token,
        user = {
            "id":        user.id,
            "email":     user.email,
            "full_name": user.full_name,
        },
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate with email + password.
    Returns a JWT access token.
    """
    user = db.query(User).filter(User.email == req.email.lower()).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    token = create_access_token({"sub": str(user.id), "email": user.email})

    return AuthResponse(
        access_token = token,
        user = {
            "id":        user.id,
            "email":     user.email,
            "full_name": user.full_name,
        },
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Returns the currently authenticated user's profile."""
    return UserResponse(
        id        = current_user.id,
        email     = current_user.email,
        full_name = current_user.full_name,
        is_active = current_user.is_active,
    )