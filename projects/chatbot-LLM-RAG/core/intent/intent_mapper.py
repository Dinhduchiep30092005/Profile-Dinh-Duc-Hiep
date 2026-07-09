"""
[2] Intent Mapping & Gap Detection
====================================
Anti-duplicate intent layer.

Classifies keywords by:
- Informational
- Commercial
- Comparison
- Transactional

Output:
- Intent classification with confidence
- Differentiated angle suggestions
"""

import logging

from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.models import Keyword, IntentType
from core.database import get_session
from core.intent.gap_detector import GapDetector

logger = logging.getLogger(__name__)


class IntentMapper:
    """Intent Mapping & Gap Detection Engine."""

    def __init__(self):
        self.gap_detector = GapDetector()

    def analyze(self, keyword: str, serp_analysis: dict) -> dict:
        """
        Analyze intent and detect gaps for a keyword.

        Args:
            keyword: The target keyword
            serp_analysis: Output from SerpEngine.analyze()

        Returns dict with:
        - intent_type: classified intent
        - intent_confidence: confidence score (0-1)
        - exploited_angles: angles already heavily covered
        - intent_gaps: untapped angles/perspectives
        - suggested_angles: differentiated content angles
        - technical_depth_missing: bool
        """
        logger.info(f"[Intent] Mapping intent for: '{keyword}'")

        # Step 1: Classify intent using LLM
        classification = self._classify_intent(keyword, serp_analysis)

        # Step 2: Detect gaps (delegated to GapDetector)
        gap_analysis = self.gap_detector.detect_gaps(keyword, serp_analysis, classification)

        # Step 3: Generate differentiated angles (delegated to GapDetector)
        angles = self.gap_detector.generate_angles(keyword, serp_analysis, gap_analysis)

        result = {
            "keyword": keyword,
            "intent_type": classification["intent_type"],
            "intent_confidence": classification["confidence"],
            "sub_intents": classification.get("sub_intents", []),
            "exploited_angles": gap_analysis["exploited_angles"],
            "intent_gaps": gap_analysis["intent_gaps"],
            "technical_depth_missing": gap_analysis["technical_depth_missing"],
            "suggested_angles": angles,
        }

        # Save to DB
        self._save_to_db(keyword, result)

        logger.info(f"[Intent] '{keyword}' → {result['intent_type']} | "
                     f"Gaps found: {len(result['intent_gaps'])}")

        return result

    def _classify_intent(self, keyword: str, serp_analysis: dict) -> dict:
        """Classify the dominant search intent of a keyword."""

        serp_titles = [r["title"] for r in serp_analysis.get("serp_results", [])[:10]]
        paa = [q["question"] for q in serp_analysis.get("people_also_ask", [])]

        user_prompt = f"""Keyword: {keyword}

Top SERP Titles:
{chr(10).join(f'- {t}' for t in serp_titles)}

People Also Ask:
{chr(10).join(f'- {q}' for q in paa)}

Dominant Intent from SERP: {serp_analysis.get('dominant_intent', 'unknown')}

Classify this keyword's search intent."""

        return llm.chat_json(PromptTemplates.INTENT_CLASSIFY_SYSTEM, user_prompt, temperature=0.2)

    def _save_to_db(self, keyword_text: str, analysis: dict):
        """Save intent analysis to database."""
        session = get_session()
        try:
            kw = session.query(Keyword).filter_by(keyword=keyword_text).first()
            if kw:
                intent_map = {
                    "informational": IntentType.INFORMATIONAL,
                    "commercial": IntentType.COMMERCIAL,
                    "comparison": IntentType.COMPARISON,
                    "transactional": IntentType.TRANSACTIONAL,
                }
                kw.intent_type = intent_map.get(
                    analysis["intent_type"], IntentType.INFORMATIONAL
                )
                kw.intent_gaps = analysis["intent_gaps"]
                kw.suggested_angles = analysis["suggested_angles"]
                session.commit()
                logger.info(f"[Intent] Saved intent mapping for '{keyword_text}'")
        except Exception as e:
            session.rollback()
            logger.error(f"[Intent] DB save error: {e}")
        finally:
            session.close()
