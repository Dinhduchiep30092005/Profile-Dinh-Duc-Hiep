"""
Database Engine & Session management for SEO Bot.
Uses SQLAlchemy ORM with PostgreSQL.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from core.models import Base


# ──────────────────────────────────────────────
# Database Engine & Session
# ──────────────────────────────────────────────

_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database.url, echo=False, pool_pre_ping=True)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


def init_db():
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("[DB] All tables created successfully.")


if __name__ == "__main__":
    init_db()
