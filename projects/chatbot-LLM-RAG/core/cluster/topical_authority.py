"""
Topical Authority Analyzer
===========================
Assesses topical authority level based on existing content coverage.
Recommends new articles to build complete topical authority.
"""

import logging

from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.models import Article, TopicCluster, ArticleStatus
from core.database import get_session

logger = logging.getLogger(__name__)


class TopicalAuthority:
    """Analyzes and scores topical authority for a given topic cluster."""

    def assess(self, cluster_id: int) -> dict:
        """
        Assess current topical authority for a cluster.

        Returns:
        - authority_score: 0-100
        - covered_subtopics
        - missing_subtopics
        - recommended_new_articles
        """
        session = get_session()
        try:
            cluster = session.query(TopicCluster).get(cluster_id)
            if not cluster:
                logger.error(f"[Authority] Cluster {cluster_id} not found")
                return {"authority_score": 0, "error": "Cluster not found"}

            # Gather existing articles
            articles = (
                session.query(Article)
                .filter(
                    Article.status.in_([
                        ArticleStatus.PUBLISHED,
                        ArticleStatus.APPROVED,
                        ArticleStatus.REVIEW,
                    ])
                )
                .all()
            )

            article_summaries = [
                {
                    "title": a.title,
                    "keyword": a.keyword.keyword if a.keyword else "",
                    "word_count": a.word_count,
                    "status": a.status.value if a.status else "unknown",
                }
                for a in articles
            ]

            user_prompt = f"""Topic Cluster: "{cluster.name}"
Pillar Keyword: {cluster.pillar_keyword}
Description: {cluster.description}

Existing Content:
{chr(10).join(f'- [{a["status"]}] "{a["title"]}" ({a["keyword"]}, {a["word_count"]} words)' for a in article_summaries)}

Total articles: {len(article_summaries)}

Assess the topical authority and identify gaps in coverage."""

            result = llm.chat_json(
                PromptTemplates.TOPICAL_AUTHORITY_SYSTEM,
                user_prompt,
                temperature=0.4,
            )

            logger.info(
                f"[Authority] Cluster '{cluster.name}': "
                f"Score {result.get('authority_score', '?')}/100"
            )

            return result

        except Exception as e:
            logger.error(f"[Authority] Assessment error: {e}")
            return {"authority_score": 0, "error": str(e)}
        finally:
            session.close()
