"""
[5] EEAT Content Builder
==========================
Builds full articles with structured EEAT signals.
Orchestrates outline generation, content writing, FAQ, meta description, data tables.
"""

import logging
import json
import re
from typing import Optional

from config import settings
from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.models import Article, ArticleStatus, ClusterRole, Keyword
from core.database import get_session
from core.content.outline_generator import OutlineGenerator

logger = logging.getLogger(__name__)


class EEATContentBuilder:
    """EEAT Content Builder - Creates high-quality, structured SEO content."""

    def __init__(self):
        self.outline_gen = OutlineGenerator()

    def build_article(
        self,
        keyword: str,
        article_spec: dict,
        brand_data: dict,
        serp_analysis: dict,
        cluster_role: str = "cluster",
    ) -> dict:
        """
        Build a complete EEAT-optimized article.

        Pipeline:
        1. Generate structured outline
        2. Build content section by section
        3. Add EEAT signals
        4. Generate FAQ schema
        5. Create meta description
        6. Add data tables where relevant
        7. Insert internal link placeholders
        """
        logger.info(f"[EEAT] Building article for: '{keyword}'")

        title = article_spec.get("branded_title", article_spec.get("title", keyword))
        brand_angle = article_spec.get("brand_angle", "")
        target_intent = article_spec.get("target_intent", "informational")
        content_type = article_spec.get("content_type", "article")

        # Step 1: Generate detailed outline
        outline = self.outline_gen.generate(
            keyword, title, brand_angle, target_intent, content_type, serp_analysis
        )

        # Step 2: Build full content
        content = self._build_full_content(
            keyword, title, outline, brand_data, serp_analysis, target_intent
        )

        # Step 3: Generate EEAT signals
        eeat = self._generate_eeat_signals(keyword, title, content, brand_data)

        # Step 4: Generate FAQ schema
        faq_schema = self._generate_faq_schema(keyword, serp_analysis)

        # Step 5: Generate meta description
        meta_desc = self._generate_meta_description(keyword, title, brand_angle)

        # Step 6: Generate data tables
        data_tables = self._generate_data_tables(keyword, content_type, brand_data)

        # Step 7: Assemble final article
        article = {
            "keyword": keyword,
            "title": title,
            "slug": self._slugify(title),
            "meta_description": meta_desc,
            "outline": outline,
            "content_markdown": content,
            "word_count": len(content.split()),
            "brand_angle": brand_angle,
            "brand_perspective": article_spec.get("brand_hook", ""),
            "eeat_experience": eeat.get("experience", ""),
            "eeat_expertise": eeat.get("expertise", ""),
            "eeat_authority": eeat.get("authority", ""),
            "eeat_trust": eeat.get("trust", ""),
            "faq_schema": faq_schema,
            "data_tables": data_tables,
            "cluster_role": cluster_role,
            "target_intent": target_intent,
            "intent_explanation": article_spec.get("intent_differentiation", ""),
            "angle_explanation": article_spec.get("perspective_shift", brand_angle),
            "status": "draft",
        }

        # Step 8: Save to database
        self._save_to_db(article)

        logger.info(
            f"[EEAT] Article built: '{title}' | "
            f"{article['word_count']} words | Status: draft"
        )

        return article

    # ──────────────────────────────────────────
    # Full Content Generation
    # ──────────────────────────────────────────

    def _build_full_content(
        self,
        keyword: str,
        title: str,
        outline: dict,
        brand_data: dict,
        serp_analysis: dict,
        target_intent: str,
    ) -> str:
        """Build the full article content section by section."""

        sections = outline.get("sections", [])
        brand_examples = brand_data.get("brand_examples",
                                         brand_data.get("pillar", {}).get("brand_examples", []))

        system_prompt = PromptTemplates.content_writer_system(
            target_intent, brand_data, settings.content
        )

        outline_text = "\n".join(
            f"{'##' if s.get('level') == 'h2' else '###'} {s.get('heading', '')}\n"
            f"  Points: {', '.join(s.get('key_points', []))}\n"
            f"  Type: {s.get('content_type', 'text')}\n"
            f"  EEAT: {s.get('eeat_signal', 'expertise')}\n"
            f"  Words: ~{s.get('word_count_target', 200)}"
            for s in sections
        )

        user_prompt = f"""Title: "{title}"
Keyword: "{keyword}"

OUTLINE:
{outline_text}

Brand Examples to Include:
{chr(10).join(f'- {ex}' for ex in (brand_examples or ['Use practical examples relevant to the topic']))}

Write the complete article following the outline. Each section should be substantial
and provide real value. Include data tables in markdown format where the outline
specifies "comparison_table" or "data_table" content types."""

        return llm.chat_long(system_prompt, user_prompt, temperature=0.7)

    # ──────────────────────────────────────────
    # EEAT Signal Generation
    # ──────────────────────────────────────────

    def _generate_eeat_signals(
        self, keyword: str, title: str, content: str, brand_data: dict
    ) -> dict:
        """Generate explicit EEAT signal summaries for the article."""

        content_preview = content[:3000] + ("..." if len(content) > 3000 else "")

        user_prompt = f"""Title: "{title}"
Keyword: "{keyword}"

Content Preview:
{content_preview}

Analyze the EEAT signals in this content."""

        return llm.chat_json(PromptTemplates.EEAT_ANALYSIS_SYSTEM, user_prompt, temperature=0.3)

    # ──────────────────────────────────────────
    # FAQ Schema
    # ──────────────────────────────────────────

    def _generate_faq_schema(self, keyword: str, serp_analysis: dict) -> list:
        """Generate FAQ schema from PAA and content analysis."""

        paa_questions = [q["question"] for q in serp_analysis.get("people_also_ask", [])]

        user_prompt = f"""Keyword: "{keyword}"

People Also Ask questions from Google:
{chr(10).join(f'- {q}' for q in paa_questions)}

Generate FAQ schema entries. Include the PAA questions and add 2-3 more relevant FAQs."""

        result = llm.chat_json(PromptTemplates.FAQ_SCHEMA_SYSTEM, user_prompt, temperature=0.4)
        return result.get("faqs", [])

    # ──────────────────────────────────────────
    # Meta Description
    # ──────────────────────────────────────────

    def _generate_meta_description(
        self, keyword: str, title: str, brand_angle: str
    ) -> str:
        """Generate an optimized meta description (~155 chars)."""

        user_prompt = f"""Title: "{title}"
Keyword: "{keyword}"
Brand Angle: {brand_angle or 'None'}

Generate an optimized meta description."""

        result = llm.chat_json(PromptTemplates.META_DESCRIPTION_SYSTEM, user_prompt, temperature=0.5)
        return result.get("meta_description", "")

    # ──────────────────────────────────────────
    # Data Tables
    # ──────────────────────────────────────────

    def _generate_data_tables(
        self, keyword: str, content_type: str, brand_data: dict
    ) -> list:
        """Generate data/comparison tables for the article."""

        product = brand_data.get("pillar", {}).get("brand_examples", [])

        user_prompt = f"""Keyword: "{keyword}"
Content Type: {content_type}
Product Context: {', '.join(product) if product else 'General topic'}

Generate relevant data/comparison tables for this article."""

        result = llm.chat_json(PromptTemplates.DATA_TABLE_SYSTEM, user_prompt, temperature=0.5)
        return result.get("tables", [])

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    @staticmethod
    def _slugify(title: str) -> str:
        """Convert title to URL slug."""
        slug = title.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug.strip('-')

    def _save_to_db(self, article_data: dict):
        """Save article to database."""
        session = get_session()
        try:
            kw = session.query(Keyword).filter_by(keyword=article_data["keyword"]).first()
            if not kw:
                kw = Keyword(keyword=article_data["keyword"])
                session.add(kw)
                session.flush()

            role_map = {
                "pillar": ClusterRole.PILLAR,
                "cluster": ClusterRole.CLUSTER,
            }

            article = Article(
                keyword_id=kw.id,
                title=article_data["title"],
                slug=article_data["slug"],
                meta_description=article_data.get("meta_description", ""),
                outline=article_data.get("outline"),
                content_markdown=article_data.get("content_markdown", ""),
                word_count=article_data.get("word_count", 0),
                brand_angle=article_data.get("brand_angle", ""),
                brand_perspective=article_data.get("brand_perspective", ""),
                eeat_experience=article_data.get("eeat_experience", ""),
                eeat_expertise=article_data.get("eeat_expertise", ""),
                eeat_authority=article_data.get("eeat_authority", ""),
                eeat_trust=article_data.get("eeat_trust", ""),
                faq_schema=article_data.get("faq_schema"),
                data_tables=article_data.get("data_tables"),
                status=ArticleStatus.DRAFT,
                cluster_role=role_map.get(article_data.get("cluster_role"), ClusterRole.CLUSTER),
                intent_explanation=article_data.get("intent_explanation", ""),
                angle_explanation=article_data.get("angle_explanation", ""),
            )
            session.add(article)
            session.commit()
            logger.info(f"[EEAT] Saved article: '{article_data['title']}'")
        except Exception as e:
            session.rollback()
            logger.error(f"[EEAT] DB save error: {e}")
        finally:
            session.close()
