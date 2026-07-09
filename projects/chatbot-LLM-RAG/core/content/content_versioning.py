"""
Content Versioning - Tracks content versions and changes over time.
Enables content updates while preserving history.
"""

import logging
from datetime import datetime, timezone

from core.models import Article, ArticleStatus
from core.database import get_session

logger = logging.getLogger(__name__)


class ContentVersioning:
    """Manages content versions and update history."""

    def create_version(self, article_id: int, new_content: str, change_reason: str) -> dict:
        """
        Create a new version of an article while preserving the old one.

        Args:
            article_id: ID of the article to update
            new_content: The new Markdown content
            change_reason: Why the update was made

        Returns:
            New version article dict
        """
        session = get_session()
        try:
            original = session.query(Article).get(article_id)
            if not original:
                logger.error(f"[Versioning] Article {article_id} not found")
                return {"error": "Article not found"}

            # Create new version
            new_version = Article(
                keyword_id=original.keyword_id,
                title=original.title,
                slug=original.slug,
                meta_description=original.meta_description,
                outline=original.outline,
                content_markdown=new_content,
                word_count=len(new_content.split()),
                brand_angle=original.brand_angle,
                brand_perspective=original.brand_perspective,
                eeat_experience=original.eeat_experience,
                eeat_expertise=original.eeat_expertise,
                eeat_authority=original.eeat_authority,
                eeat_trust=original.eeat_trust,
                faq_schema=original.faq_schema,
                data_tables=original.data_tables,
                status=ArticleStatus.DRAFT,
                cluster_role=original.cluster_role,
                intent_explanation=original.intent_explanation,
                angle_explanation=original.angle_explanation,
                version=(original.version or 1) + 1,
                previous_version_id=original.id,
            )

            session.add(new_version)
            session.commit()

            logger.info(
                f"[Versioning] Created v{new_version.version} of '{original.title}' "
                f"(reason: {change_reason})"
            )

            return {
                "article_id": new_version.id,
                "title": new_version.title,
                "version": new_version.version,
                "previous_version_id": original.id,
                "word_count": new_version.word_count,
                "change_reason": change_reason,
            }

        except Exception as e:
            session.rollback()
            logger.error(f"[Versioning] Error: {e}")
            return {"error": str(e)}
        finally:
            session.close()

    def get_history(self, article_id: int) -> list[dict]:
        """Get version history for an article."""
        session = get_session()
        try:
            history = []
            current = session.query(Article).get(article_id)

            while current:
                history.append({
                    "id": current.id,
                    "version": current.version or 1,
                    "word_count": current.word_count,
                    "status": current.status.value if current.status else "unknown",
                    "created_at": current.created_at.isoformat() if current.created_at else None,
                })
                if current.previous_version_id:
                    current = session.query(Article).get(current.previous_version_id)
                else:
                    current = None

            return list(reversed(history))
        finally:
            session.close()
