"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DEPRECATED — DO NOT USE                                                    ║
║                                                                              ║
║  This file is legacy code from an early version of LexShield AI that used   ║
║  SQLAlchemy ORM.  Its schema is INCOMPATIBLE with the live PostgreSQL users  ║
║  table:                                                                      ║
║    • Uses Integer primary key — live schema uses TEXT (UUID4)                ║
║    • Defines is_active (Boolean) — not present in live schema                ║
║    • Uses DateTime columns — live schema uses DOUBLE PRECISION Unix stamps   ║
║                                                                              ║
║  The live users table is managed entirely by _init_auth_tables() in          ║
║  api/auth.py via raw psycopg v3 SQL.  Nothing imports this file.             ║
║  Kept for historical reference only — do not build on top of it.             ║
╚══════════════════════════════════════════════════════════════════════════════╝

LexShield AI — User Model  [DEPRECATED]
========================================
SQLAlchemy ORM model for the users table.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from models.database import Base

import logging

logger = logging.getLogger(__name__)


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String, unique=True, index=True, nullable=False)
    full_name  = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<User id={self.id} email={self.email!r}>"