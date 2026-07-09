"""
[4] Brand Angle Generator (Critical Layer)
============================================
The most important differentiation layer.
Prevents "generic, no unique perspective" content.

Same keyword → Different perspective
"""

import logging
from typing import Optional

from config import settings, BrandConfig
from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class BrandAngleGenerator:
    """Brand Angle Generator - Injects unique brand perspective into content."""

    def __init__(self, brand_config: Optional[BrandConfig] = None):
        self.brand = brand_config or settings.brand

    def configure_brand(
        self,
        product_name: str,
        product_description: str,
        use_cases: list[str],
        competitive_advantages: list[str],
        target_audience: str = "",
        brand_voice: str = "",
    ):
        """Configure brand information for angle generation."""
        self.brand = BrandConfig(
            product_name=product_name,
            product_description=product_description,
            use_cases=use_cases,
            competitive_advantages=competitive_advantages,
            target_audience=target_audience,
            brand_voice=brand_voice or "professional, technical, authoritative",
        )
        logger.info(f"[Brand] Configured brand: {product_name}")

    def generate_angles(
        self,
        keyword: str,
        cluster_blueprint: dict,
        intent_analysis: dict,
    ) -> dict:
        """
        Generate brand-differentiated angles for all articles in a cluster.
        Transforms generic titles/angles into brand-specific perspectives.
        """
        if not self.brand.product_name:
            logger.warning("[Brand] No brand configured. Using generic angles.")
            return cluster_blueprint

        logger.info(f"[Brand] Generating brand angles for: '{keyword}'")

        # Step 1: Generate brand angles for pillar
        pillar = cluster_blueprint.get("pillar", {})
        branded_pillar = self._brand_angle_for_article(
            keyword=pillar.get("keyword", keyword),
            original_title=pillar.get("title", ""),
            content_type=pillar.get("content_type", "comprehensive guide"),
            target_intent=pillar.get("target_intent", "informational"),
            is_pillar=True,
        )

        # Step 2: Generate brand angles for each cluster article
        branded_clusters = []
        for cluster in cluster_blueprint.get("clusters", []):
            branded = self._brand_angle_for_article(
                keyword=cluster.get("keyword", ""),
                original_title=cluster.get("title", ""),
                content_type=cluster.get("content_type", ""),
                target_intent=cluster.get("target_intent", "informational"),
                is_pillar=False,
            )
            cluster_copy = dict(cluster)
            cluster_copy.update({
                "branded_title": branded.get("branded_title", cluster.get("title")),
                "brand_angle": branded.get("brand_angle", ""),
                "brand_hook": branded.get("brand_hook", ""),
                "brand_examples": branded.get("brand_examples", []),
                "brand_cta": branded.get("brand_cta", ""),
            })
            branded_clusters.append(cluster_copy)

        # Step 3: Generate overall brand positioning strategy
        strategy = self._generate_brand_strategy(keyword, intent_analysis)

        result = dict(cluster_blueprint)
        result["pillar"] = dict(pillar)
        result["pillar"].update({
            "branded_title": branded_pillar.get("branded_title", pillar.get("title")),
            "brand_angle": branded_pillar.get("brand_angle", ""),
            "brand_hook": branded_pillar.get("brand_hook", ""),
            "brand_examples": branded_pillar.get("brand_examples", []),
        })
        result["clusters"] = branded_clusters
        result["brand_strategy"] = strategy

        logger.info(f"[Brand] Generated {len(branded_clusters) + 1} branded angles")
        return result

    def _brand_angle_for_article(
        self,
        keyword: str,
        original_title: str,
        content_type: str,
        target_intent: str,
        is_pillar: bool,
    ) -> dict:
        """Generate a brand-differentiated angle for a single article."""

        system_prompt = PromptTemplates.brand_angle_system(self.brand)

        user_prompt = f"""Original Title: "{original_title}"
Keyword: "{keyword}"
Content Type: {content_type}
Target Intent: {target_intent}
Is Pillar Page: {is_pillar}

Transform this into a brand-differentiated angle for {self.brand.product_name}.
The result should target the same keyword but from a completely different perspective
that leverages the product's unique strengths."""

        return llm.chat_json(system_prompt, user_prompt, temperature=0.7)

    def _generate_brand_strategy(self, keyword: str, intent_analysis: dict) -> dict:
        """Generate overall brand content strategy for this keyword cluster."""

        system_prompt = PromptTemplates.brand_strategy_system(self.brand)

        gaps = intent_analysis.get("intent_gaps", [])
        user_prompt = f"""Keyword: "{keyword}"
Intent: {intent_analysis.get('intent_type', 'informational')}

Current Gaps in SERP:
{chr(10).join(f'- {g["gap"]}' for g in gaps)}

Suggested Angles:
{chr(10).join(f'- {a.get("title_suggestion", a.get("angle_description", str(a)))}' for a in intent_analysis.get('suggested_angles', []))}

Create a brand content strategy that leverages these gaps while authentically positioning {self.brand.product_name}."""

        return llm.chat_json(system_prompt, user_prompt, temperature=0.5)
