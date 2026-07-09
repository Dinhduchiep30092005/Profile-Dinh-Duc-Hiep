"""
Outline Generator - Creates detailed EEAT-optimized content outlines.
Extracted from EEAT Builder for modularity.
"""

import logging

from config import settings
from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.knowledge.retrieval import build_product_solution_context, has_knowledge_base

logger = logging.getLogger(__name__)


class OutlineGenerator:
    """Generates structured content outlines with EEAT signals."""

    def generate(
        self,
        keyword: str,
        title: str,
        brand_angle: str,
        target_intent: str,
        content_type: str,
        serp_analysis: dict,
    ) -> dict:
        """Generate a detailed content outline with EEAT structure."""

        common_h2 = serp_analysis.get("structure_map", {}).get("common_h2", [])
        features = serp_analysis.get("structure_map", {}).get("content_features", {})

        # Check if product should be mentioned as a solution
        product_instruction = ""
        if has_knowledge_base():
            product_context, product_meta = build_product_solution_context(
                keyword=keyword,
                title=title,
                intent_type=target_intent,
                relevance_threshold=0.35,
                max_tokens_approx=1000,
            )
            if product_meta.get("is_product_relevant") and product_context:
                product_names = ", ".join(product_meta.get("product_names", [])) or "sản phẩm"
                section_type = product_meta.get("recommended_section_type", "product_as_solution")
                product_instruction = f"""

{product_context}

PRODUCT INTEGRATION:
Bài viết này liên quan đến sản phẩm: {product_names}.
Hãy thêm 1-2 section giới thiệu sản phẩm như giải pháp ({section_type}).
- Dùng heading tự nhiên, không quảng cáo lộ liễu
- Sử dụng dữ liệu cụ thể từ tài liệu sản phẩm ở trên
- Đặt content_type: "product_solution" cho các section này
- Section nên nằm sau phần expertise, trước phần trust"""
                logger.info(
                    f"[Outline] Product mention enabled for: {product_names} "
                    f"(score: {product_meta.get('relevance_score', 0):.3f})"
                )

        user_prompt = f"""Title: "{title}"
Keyword: "{keyword}"
Content Type: {content_type}
Target Intent: {target_intent}
Brand Angle: {brand_angle}

Target Word Count: {settings.content.min_words}-{settings.content.max_words}
Average SERP Word Count: {serp_analysis.get('avg_word_count', 1500)}

Common H2 Headings in SERP:
{chr(10).join(f'- {h["text"]} ({h["frequency"]}x)' for h in common_h2[:10])}

Content Features in SERP:
- {features.get('pct_with_tables', 0):.0f}% have tables
- {features.get('pct_with_lists', 0):.0f}% have lists
- {features.get('pct_with_faq', 0):.0f}% have FAQ
{product_instruction}

Create a detailed outline that:
1. Covers the topic comprehensively with EEAT structure
2. Includes sections for data tables/comparisons if relevant
3. Has clear internal linking opportunities
4. Differs from generic SERP content via the brand angle
5. Targets {settings.content.min_words}-{settings.content.max_words} words
6. If product info is provided, naturally integrates product mention sections"""

        return llm.chat_json(
            PromptTemplates.OUTLINE_SYSTEM, user_prompt, temperature=0.5, max_tokens=4096
        )
