"""
SERP Snapshot - Database persistence for SERP analysis results.
"""

import logging
from datetime import datetime, timezone

from core.models import Keyword
from core.database import get_session

logger = logging.getLogger(__name__)


class SerpSnapshot:
    """Handles saving and loading SERP analysis data from the database."""

    def save(self, keyword_text: str, analysis: dict):
        """Save SERP analysis results to database."""
        session = get_session()
        try:
            kw = session.query(Keyword).filter_by(keyword=keyword_text).first()
            if not kw:
                kw = Keyword(keyword=keyword_text)
                session.add(kw)

            kw.serp_data = {
                "organic": analysis["serp_results"],
                "paa": analysis["people_also_ask"],
                "related": analysis["related_searches"],
            }
            kw.serp_structure_map = analysis["structure_map"]
            kw.serp_avg_word_count = analysis["avg_word_count"]
            kw.serp_dominant_intent = analysis["dominant_intent"]
            kw.serp_content_patterns = analysis["content_patterns"]
            kw.saturation_score = analysis["saturation_score"]
            kw.last_serp_check = datetime.now(timezone.utc)

            session.commit()
            logger.info(f"[SERP] Saved analysis for '{keyword_text}' to database")
        except Exception as e:
            session.rollback()
            logger.error(f"[SERP] DB save error: {e}")
        finally:
            session.close()

    def load(self, keyword_text: str) -> dict | None:
        """Load SERP analysis from database for a keyword."""
        session = get_session()
        try:
            kw = session.query(Keyword).filter_by(keyword=keyword_text).first()
            if not kw or not kw.serp_data:
                return None

            return {
                "keyword": keyword_text,
                "serp_results": kw.serp_data.get("organic", []),
                "people_also_ask": kw.serp_data.get("paa", []),
                "related_searches": kw.serp_data.get("related", []),
                "structure_map": kw.serp_structure_map,
                "avg_word_count": kw.serp_avg_word_count,
                "dominant_intent": kw.serp_dominant_intent,
                "content_patterns": kw.serp_content_patterns,
                "saturation_score": kw.saturation_score,
                "last_checked": kw.last_serp_check.isoformat() if kw.last_serp_check else None,
            }
        finally:
            session.close()
