"""
Gap Detector - Detects over-exploited angles and gaps in SERP coverage.
Generates differentiated content angle suggestions.
"""

import logging

from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class GapDetector:
    """Detects content gaps and generates differentiated content angles."""

    def detect_gaps(self, keyword: str, serp_analysis: dict, classification: dict) -> dict:
        """Detect over-exploited angles and gaps in SERP coverage."""

        titles = [r["title"] for r in serp_analysis.get("serp_results", [])[:15]]
        patterns = serp_analysis.get("content_patterns", [])
        structure = serp_analysis.get("structure_map", {})

        user_prompt = f"""Keyword: "{keyword}"
Intent: {classification['intent_type']}

Current Top SERP Titles:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(titles))}

Content Patterns Found:
{chr(10).join(f'- {p}' for p in patterns)}

Common H2 Headings:
{chr(10).join(f'- {h["text"]}' for h in structure.get("common_h2", [])[:10])}

Avg Word Count: {serp_analysis.get('avg_word_count', 'unknown')}
Saturation Score: {serp_analysis.get('saturation_score', 'unknown')}/100

Analyze:
1. Which angles are OVER-exploited (everyone is writing the same thing)?
2. What perspectives/angles are MISSING from the current SERP?
3. Is there sufficient technical depth, or is everything surface-level?
4. What content formats are missing (tables, comparisons, case studies)?"""

        return llm.chat_json(PromptTemplates.GAP_DETECTION_SYSTEM, user_prompt, temperature=0.4)

    def generate_angles(self, keyword: str, serp_analysis: dict, gap_analysis: dict) -> list:
        """Generate differentiated content angles based on gaps."""

        gaps = gap_analysis.get("intent_gaps", [])
        exploited = gap_analysis.get("exploited_angles", [])

        user_prompt = f"""Keyword: "{keyword}"

Over-exploited angles (AVOID these):
{chr(10).join(f'- {a}' for a in exploited)}

Identified Gaps:
{chr(10).join(f'- {g["gap"]} (opportunity: {g["opportunity_score"]}/10)' for g in gaps)}

Technical Depth Missing: {gap_analysis.get('technical_depth_missing', False)}

Generate 3-5 differentiated content angles that fill these gaps.
Each angle should target the keyword but from a unique perspective that current SERP results don't cover."""

        result = llm.chat_json(PromptTemplates.ANGLE_GENERATION_SYSTEM, user_prompt, temperature=0.6)
        return result.get("angles", [])
