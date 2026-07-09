"""
SQLAlchemy ORM Models for SEO Bot.
All database models and enums are defined here.
"""

from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Enum,
    JSON,
    Table,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    relationship,
)


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class IntentType(PyEnum):
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    COMPARISON = "comparison"
    TRANSACTIONAL = "transactional"


class ArticleStatus(PyEnum):
    DRAFT = "draft"
    OUTLINE = "outline"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    NEEDS_UPDATE = "needs_update"


class ClusterRole(PyEnum):
    PILLAR = "pillar"
    CLUSTER = "cluster"


# ──────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────
# Association tables
# ──────────────────────────────────────────────

article_internal_links = Table(
    "article_internal_links",
    Base.metadata,
    Column("source_article_id", Integer, ForeignKey("articles.id"), primary_key=True),
    Column("target_article_id", Integer, ForeignKey("articles.id"), primary_key=True),
    Column("anchor_text", String(500)),
)


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

class Keyword(Base):
    """Input keyword with SERP analysis results."""

    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String(500), nullable=False, unique=True, index=True)
    intent_type = Column(Enum(IntentType), nullable=True)
    search_volume = Column(Integer, nullable=True)
    difficulty = Column(Float, nullable=True)
    saturation_score = Column(Float, nullable=True)

    # SERP analysis data (stored as JSON)
    serp_data = Column(JSON, nullable=True)  # Top results, PAA, related searches
    serp_structure_map = Column(JSON, nullable=True)  # H2/H3 patterns
    serp_avg_word_count = Column(Integer, nullable=True)
    serp_dominant_intent = Column(String(100), nullable=True)
    serp_content_patterns = Column(JSON, nullable=True)

    # Intent gap analysis
    intent_gaps = Column(JSON, nullable=True)  # Untapped angles
    suggested_angles = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_serp_check = Column(DateTime, nullable=True)

    # Relationships
    articles = relationship("Article", back_populates="keyword")
    cluster_keywords = relationship("ClusterKeyword", back_populates="keyword")


class TopicCluster(Base):
    """Topic cluster containing pillar + cluster articles."""

    __tablename__ = "topic_clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    pillar_keyword = Column(String(500), nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    cluster_keywords = relationship("ClusterKeyword", back_populates="cluster")
    articles = relationship("Article", back_populates="cluster")


class ClusterKeyword(Base):
    """Keywords belonging to a topic cluster."""

    __tablename__ = "cluster_keywords"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey("topic_clusters.id"), nullable=False)
    keyword_id = Column(Integer, ForeignKey("keywords.id"), nullable=False)
    role = Column(Enum(ClusterRole), nullable=False)
    suggested_title = Column(String(500), nullable=True)
    anchor_text = Column(String(500), nullable=True)
    intent_differentiation = Column(Text, nullable=True)  # How this differs from others

    cluster = relationship("TopicCluster", back_populates="cluster_keywords")
    keyword = relationship("Keyword", back_populates="cluster_keywords")


class Article(Base):
    """Generated article with full content and metadata."""

    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword_id = Column(Integer, ForeignKey("keywords.id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("topic_clusters.id"), nullable=True)

    # Content
    title = Column(String(500), nullable=False)
    slug = Column(String(500), nullable=True)
    meta_description = Column(String(320), nullable=True)
    outline = Column(JSON, nullable=True)  # Structured outline with H2/H3
    content_html = Column(Text, nullable=True)
    content_markdown = Column(Text, nullable=True)
    word_count = Column(Integer, nullable=True)

    # Content versioning
    version = Column(Integer, default=1)
    previous_version_id = Column(Integer, ForeignKey("articles.id"), nullable=True)

    # Brand angle
    brand_angle = Column(Text, nullable=True)
    brand_perspective = Column(Text, nullable=True)

    # EEAT signals
    eeat_experience = Column(Text, nullable=True)  # Practical examples
    eeat_expertise = Column(Text, nullable=True)  # Technical explanations
    eeat_authority = Column(Text, nullable=True)  # Structured comparisons
    eeat_trust = Column(Text, nullable=True)  # Clear positioning

    # Schema / structured data
    faq_schema = Column(JSON, nullable=True)
    data_tables = Column(JSON, nullable=True)

    # Status
    status = Column(Enum(ArticleStatus), default=ArticleStatus.DRAFT)
    cluster_role = Column(Enum(ClusterRole), nullable=True)
    intent_explanation = Column(Text, nullable=True)
    angle_explanation = Column(Text, nullable=True)

    # Review
    reviewer_notes = Column(Text, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    published_url = Column(String(1000), nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    keyword = relationship("Keyword", back_populates="articles")
    cluster = relationship("TopicCluster", back_populates="articles")
    links_out = relationship(
        "Article",
        secondary=article_internal_links,
        primaryjoin=id == article_internal_links.c.source_article_id,
        secondaryjoin=id == article_internal_links.c.target_article_id,
        backref="links_in",
    )
    rank_snapshots = relationship("RankSnapshot", back_populates="article")
    previous_version = relationship("Article", remote_side=[id], uselist=False)


class RankSnapshot(Base):
    """Periodic ranking data for published articles."""

    __tablename__ = "rank_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    keyword_query = Column(String(500), nullable=False)
    position = Column(Integer, nullable=True)
    serp_features = Column(JSON, nullable=True)  # Featured snippet, PAA, etc.
    competitor_changes = Column(JSON, nullable=True)
    suggested_updates = Column(JSON, nullable=True)
    checked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    article = relationship("Article", back_populates="rank_snapshots")
