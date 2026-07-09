"""
[6] Internal Link Automation Engine
======================================
Automates internal linking across the topic cluster.

Responsibilities:
- Identify pillar page links
- Insert natural anchor text
- Prevent over-optimization
- Ensure each article:
  - Links UP to pillar
  - Links ACROSS to related cluster articles
- Build real authority through proper linking structure
"""

import logging
import re
from typing import Optional

from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.models import Article, article_internal_links
from core.database import get_session

logger = logging.getLogger(__name__)


class LinkAutomation:
    """Internal Link Automation Engine - Manages cluster internal linking."""

    MAX_INTERNAL_LINKS_PER_ARTICLE = 8
    MIN_INTERNAL_LINKS_PER_ARTICLE = 2

    def process_cluster_links(
        self,
        articles: list[dict],
        linking_map: dict,
    ) -> list[dict]:
        """
        Process all articles in a cluster and insert internal links.

        Args:
            articles: List of article dicts from EEAT builder
            linking_map: Linking map from cluster architect

        Returns:
            Updated articles with internal links inserted
        """
        logger.info(f"[Link] Processing internal links for {len(articles)} articles")

        processed = []
        for article in articles:
            linked_article = self._insert_links(article, articles, linking_map)
            processed.append(linked_article)

        # Validate link structure
        self._validate_links(processed)

        logger.info(f"[Link] Completed internal linking for {len(processed)} articles")
        return processed

    def _insert_links(
        self,
        article: dict,
        all_articles: list[dict],
        linking_map: dict,
    ) -> dict:
        """Insert internal links into a single article's content."""

        content = article.get("content_markdown", "")
        title = article.get("title", "")

        # Step 1: Find existing placeholder links [INTERNAL_LINK:anchor:target]
        placeholder_pattern = r'\[INTERNAL_LINK:([^:]+):([^\]]+)\]'
        placeholders = re.findall(placeholder_pattern, content)

        # Step 2: Resolve placeholders to actual article links
        for anchor, target_kw in placeholders:
            target_article = self._find_target_article(target_kw, all_articles)
            if target_article:
                slug = target_article.get("slug", "")
                link_md = f"[{anchor}](/{slug})"
                content = content.replace(
                    f"[INTERNAL_LINK:{anchor}:{target_kw}]", link_md, 1
                )

        # Step 3: Use LLM to suggest additional natural link insertions
        current_link_count = len(re.findall(r'\[([^\]]+)\]\(/[^\)]+\)', content))
        if current_link_count < self.MIN_INTERNAL_LINKS_PER_ARTICLE:
            content = self._add_natural_links(
                content, article, all_articles, linking_map,
                needed=self.MIN_INTERNAL_LINKS_PER_ARTICLE - current_link_count
            )

        # Step 4: Check for over-optimization
        link_count = len(re.findall(r'\[([^\]]+)\]\(/[^\)]+\)', content))
        if link_count > self.MAX_INTERNAL_LINKS_PER_ARTICLE:
            logger.warning(
                f"[Link] '{title}' has {link_count} links "
                f"(max: {self.MAX_INTERNAL_LINKS_PER_ARTICLE}). Consider reducing."
            )

        article_copy = dict(article)
        article_copy["content_markdown"] = content
        article_copy["internal_link_count"] = link_count
        return article_copy

    def _find_target_article(self, target_keyword: str, articles: list[dict]) -> Optional[dict]:
        """Find a target article by keyword match."""
        target_lower = target_keyword.lower().strip()
        for article in articles:
            kw = article.get("keyword", "").lower().strip()
            title = article.get("title", "").lower().strip()
            if target_lower in kw or target_lower in title or kw in target_lower:
                return article
        return None

    def _add_natural_links(
        self,
        content: str,
        current_article: dict,
        all_articles: list[dict],
        linking_map: dict,
        needed: int,
    ) -> str:
        """Use LLM to suggest and insert natural internal links."""

        other_articles = [
            {"title": a["title"], "slug": a.get("slug", ""), "keyword": a.get("keyword", "")}
            for a in all_articles
            if a["title"] != current_article["title"]
        ]

        if not other_articles:
            return content

        content_preview = content[:2000]

        user_prompt = f"""Current Article: "{current_article.get('title', '')}"

Article Content (preview):
{content_preview}

Available Articles to Link To:
{chr(10).join(f'- "{a["title"]}" (slug: /{a["slug"]}, keyword: {a["keyword"]})' for a in other_articles[:8])}

Find {needed} natural places to insert internal links. Return exact text matches."""

        try:
            result = llm.chat_json(PromptTemplates.INTERNAL_LINK_SYSTEM, user_prompt, temperature=0.3)
            insertions = result.get("link_insertions", [])

            for ins in insertions[:needed]:
                find = ins.get("find_text", "")
                replace = ins.get("replace_text", "")
                if find and replace and find in content:
                    content = content.replace(find, replace, 1)
                    logger.info(f"[Link] Inserted link to /{ins.get('target_slug', '?')}")

        except Exception as e:
            logger.error(f"[Link] Failed to add natural links: {e}")

        return content

    def _validate_links(self, articles: list[dict]):
        """Validate the internal linking structure."""
        pillar = None
        clusters = []

        for a in articles:
            if a.get("cluster_role") == "pillar":
                pillar = a
            else:
                clusters.append(a)

        if not pillar:
            logger.warning("[Link] No pillar article found in cluster!")
            return

        pillar_slug = pillar.get("slug", "")
        for cluster in clusters:
            content = cluster.get("content_markdown", "")
            if pillar_slug and f"/{pillar_slug}" not in content:
                logger.warning(
                    f"[Link] '{cluster.get('title', '?')}' does not link to pillar!"
                )

        pillar_content = pillar.get("content_markdown", "")
        for cluster in clusters:
            slug = cluster.get("slug", "")
            if slug and f"/{slug}" not in pillar_content:
                logger.warning(
                    f"[Link] Pillar does not link to '{cluster.get('title', '?')}'!"
                )

    def save_links_to_db(self, articles: list[dict]):
        """Save internal link relationships to database."""
        session = get_session()
        try:
            for article in articles:
                db_article = session.query(Article).filter_by(
                    slug=article.get("slug", "")
                ).first()
                if not db_article:
                    continue

                db_article.content_markdown = article.get("content_markdown", "")

                content = article.get("content_markdown", "")
                link_pattern = r'\[([^\]]+)\]\(/([^\)]+)\)'
                links = re.findall(link_pattern, content)

                for anchor, slug in links:
                    target = session.query(Article).filter_by(slug=slug).first()
                    if target and target.id != db_article.id:
                        if target not in db_article.links_out:
                            db_article.links_out.append(target)

            session.commit()
            logger.info(f"[Link] Saved link relationships to database")
        except Exception as e:
            session.rollback()
            logger.error(f"[Link] DB save error: {e}")
        finally:
            session.close()
