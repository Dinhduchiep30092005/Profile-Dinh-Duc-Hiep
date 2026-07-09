"""
[8] Rank Monitor + Update Engine
==================================
Periodic monitoring and update recommendation system.
Runs every 3 months (configurable).

Monitors:
- SERP position changes
- New SERP features
- Competitor content changes

Recommends:
- Content updates
- New sections
- Refreshed data
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.serp.serp_engine import SerpEngine
from core.models import (
    Article,
    ArticleStatus,
    RankSnapshot,
    Keyword,
)
from core.database import get_session
from core.monitoring.serp_update_checker import SerpUpdateChecker

logger = logging.getLogger(__name__)


class RankMonitor:
    """Rank Monitor + Update Engine - Tracks and maintains published content."""

    CHECK_INTERVAL_DAYS = 90  # 3 months

    def __init__(self):
        self.serp_engine = SerpEngine()
        self.update_checker = SerpUpdateChecker()

    def check_all_published(self) -> list[dict]:
        """Check all published articles for ranking changes."""
        session = get_session()
        try:
            articles = session.query(Article).filter_by(
                status=ArticleStatus.PUBLISHED
            ).all()

            results = []
            for article in articles:
                last_snapshot = (
                    session.query(RankSnapshot)
                    .filter_by(article_id=article.id)
                    .order_by(RankSnapshot.checked_at.desc())
                    .first()
                )

                if last_snapshot:
                    days_since = (datetime.now(timezone.utc) - last_snapshot.checked_at).days
                    if days_since < self.CHECK_INTERVAL_DAYS:
                        logger.info(
                            f"[Monitor] Skipping '{article.title}' — "
                            f"checked {days_since} days ago"
                        )
                        continue

                result = self.check_article(article.id)
                if result:
                    results.append(result)

            return results
        finally:
            session.close()

    def check_article(self, article_id: int) -> Optional[dict]:
        """
        Check a single published article for ranking changes.
        Returns monitoring report with update recommendations.
        """
        session = get_session()
        try:
            article = session.query(Article).get(article_id)
            if not article:
                logger.error(f"[Monitor] Article {article_id} not found")
                return None

            keyword = article.keyword.keyword if article.keyword else ""
            logger.info(f"[Monitor] Checking: '{article.title}' for '{keyword}'")

            # Step 1: Fresh SERP analysis
            current_serp = self.serp_engine.analyze(keyword)

            # Step 2: Get previous snapshot for comparison
            prev_snapshot = (
                session.query(RankSnapshot)
                .filter_by(article_id=article.id)
                .order_by(RankSnapshot.checked_at.desc())
                .first()
            )

            # Step 3: Find our position in SERP
            our_position = self._find_position(
                article.published_url or "",
                current_serp.get("serp_results", []),
            )

            # Step 4: Detect changes (delegated to SerpUpdateChecker)
            changes = self.update_checker.detect_changes(
                article, current_serp, prev_snapshot
            )

            # Step 5: Generate update recommendations
            recommendations = self._generate_recommendations(
                article, current_serp, changes
            )

            # Step 6: Save snapshot
            snapshot = RankSnapshot(
                article_id=article.id,
                keyword_query=keyword,
                position=our_position,
                serp_features=current_serp.get("content_patterns", []),
                competitor_changes=changes,
                suggested_updates=recommendations,
            )
            session.add(snapshot)

            # Step 7: Flag for update if needed
            if recommendations.get("needs_update", False):
                article.status = ArticleStatus.NEEDS_UPDATE
                logger.warning(f"[Monitor] '{article.title}' flagged for update!")

            session.commit()

            report = {
                "article_id": article.id,
                "title": article.title,
                "keyword": keyword,
                "current_position": our_position,
                "previous_position": prev_snapshot.position if prev_snapshot else None,
                "position_change": self._calc_position_change(
                    our_position, prev_snapshot.position if prev_snapshot else None
                ),
                "changes": changes,
                "recommendations": recommendations,
                "needs_update": recommendations.get("needs_update", False),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"[Monitor] '{article.title}': Position {our_position or '?'} | "
                f"Update needed: {report['needs_update']}"
            )

            return report

        except Exception as e:
            session.rollback()
            logger.error(f"[Monitor] Check error: {e}")
            return None
        finally:
            session.close()

    def _find_position(self, published_url: str, serp_results: list) -> Optional[int]:
        """Find our article's position in SERP results."""
        if not published_url:
            return None

        url_clean = published_url.rstrip("/").lower()

        for result in serp_results:
            result_url = result.get("link", "").rstrip("/").lower()
            if url_clean in result_url or result_url in url_clean:
                return result.get("position")

        return None

    def _generate_recommendations(
        self,
        article: Article,
        current_serp: dict,
        changes: dict,
    ) -> dict:
        """Use LLM to generate update recommendations."""

        user_prompt = f"""Our Article:
Title: {article.title}
Word Count: {article.word_count}
Published: {article.published_at}
Current Status: {article.status.value if article.status else 'unknown'}

Current SERP Data:
Average Word Count: {current_serp.get('avg_word_count', 'Unknown')}
Saturation Score: {current_serp.get('saturation_score', 'Unknown')}/100

Changes Detected:
- Word count shift: {changes.get('word_count_shift', 0):+d} words
- New competitors: {', '.join(changes.get('new_competitors', [])) or 'None'}
- New sections appearing in SERP: {', '.join(changes.get('new_serp_features', [])) or 'None'}

New PAA Questions:
{chr(10).join(f'- {q["question"]}' for q in current_serp.get('people_also_ask', [])[:5])}

Should this article be updated? What specific changes are recommended?"""

        return llm.chat_json(PromptTemplates.RANK_UPDATE_SYSTEM, user_prompt, temperature=0.3)

    @staticmethod
    def _calc_position_change(
        current: Optional[int], previous: Optional[int]
    ) -> Optional[int]:
        """Calculate position change (negative = improved, positive = dropped)."""
        if current is None or previous is None:
            return None
        return current - previous

    def get_update_queue(self) -> list[dict]:
        """Get all articles flagged for update."""
        session = get_session()
        try:
            articles = session.query(Article).filter_by(
                status=ArticleStatus.NEEDS_UPDATE
            ).all()

            queue = []
            for article in articles:
                latest = (
                    session.query(RankSnapshot)
                    .filter_by(article_id=article.id)
                    .order_by(RankSnapshot.checked_at.desc())
                    .first()
                )
                queue.append({
                    "id": article.id,
                    "title": article.title,
                    "keyword": article.keyword.keyword if article.keyword else "",
                    "current_position": latest.position if latest else None,
                    "recommendations": latest.suggested_updates if latest else {},
                    "last_checked": latest.checked_at.isoformat() if latest else None,
                })
            return queue
        finally:
            session.close()
