"""
LexShield AI — Database Setup
================================
SQLAlchemy engine + session factory.
Switch databases by changing DATABASE_URL only.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lexshield.db")

# Ensure SQLAlchemy uses the psycopg v3 driver for PostgreSQL connections.
# Standard libpq URLs (postgresql://...) default to psycopg2 in SQLAlchemy;
# rewrite to postgresql+psycopg:// so the installed psycopg v3 is used instead.
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# For SQLite, we need check_same_thread=False
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session, closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Call once at startup to create all tables."""
    Base.metadata.create_all(bind=engine)