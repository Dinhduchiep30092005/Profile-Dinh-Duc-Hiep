"""
ArticleController — Orchestrates the full article generation pipeline.

Flow:
    UI
    ↓
    ArticleController
    ↓
    PreValidationLayer   ← validates keyword, title, heading, API key
    ↓
    ArticleGenerator     ← wraps ArticleBuilder (17 phases)
    ↓
    QualityGate          ← automated quality score (non-blocking)
    ↓
    Renderer             ← final HTML cleanup
    ↓
    Save DB (draft)      ← handled by the caller (gsc_dashboard.py)
"""

import logging

logger = logging.getLogger(__name__)


class ArticleController:
    """
    Main orchestrator for the article generation pipeline.

    Called by API endpoints in gsc_dashboard.py.
    Keeps each step decoupled — swap any layer without touching the API.
    """

    def __init__(self):
        from core.content.pre_validation import PreValidationLayer
        from core.content.renderer import Renderer
        from core.quality.quality_gate import QualityGate

        self.validator = PreValidationLayer()
        self.renderer = Renderer()
        self.quality_gate = QualityGate()

    def generate(
        self,
        keyword: str,
        title: str,
        heading_structure: dict,
        language: str = "vi",
    ) -> dict:
        """
        Run the full article generation pipeline.

        Returns:
            dict with keys:
                success (bool)
                article_text, article_html, internal_links, word_count
                quality_check (dict)
                validation_warnings (list[str])
                status ("draft")
            On failure:
                success=False, error (str), validation_errors (list[str])
        """
        logger.info(
            f"[ArticleController] ═══ Start: '{title}' | lang={language} ═══"
        )

        # ── Step 1: PreValidationLayer ──
        validation = self.validator.validate(
            keyword=keyword,
            title=title,
            heading_structure=heading_structure,
            language=language,
        )
        if not validation["valid"]:
            return {
                "success": False,
                "error": " | ".join(validation["errors"]),
                "validation_errors": validation["errors"],
            }

        # ── Step 2: ArticleGenerator → ArticleBuilder (17 phases) ──
        from core.content.article_generator import ArticleGenerator

        generator = ArticleGenerator()
        result = generator.generate(
            keyword=keyword,
            title=title,
            heading_structure=heading_structure,
            language=language,
        )

        # ── Step 3: QualityGate (automated, non-blocking) ──
        quality_check = self.quality_gate.auto_check(result)
        result["quality_check"] = quality_check

        if quality_check["passed"]:
            logger.info(
                f"[ArticleController] QualityGate ✓ — "
                f"score={quality_check['score']:.1f}/10"
            )
        else:
            logger.warning(
                f"[ArticleController] QualityGate ✗ — "
                f"score={quality_check['score']:.1f}/10 | "
                f"issues: {quality_check['issues']}"
            )

        # ── Step 4: Renderer ──
        result["article_html"] = self.renderer.render(
            article_html=result.get("article_html", ""),
            article_text=result.get("article_text", ""),
            language=language,
        )

        # ── Step 5: Mark as draft (caller saves to DB) ──
        result["success"] = True
        result["status"] = "draft"
        result["validation_warnings"] = validation["warnings"]

        logger.info(
            f"[ArticleController] ═══ Complete: {result.get('word_count', 0)}w | "
            f"QC: {'✓' if quality_check['passed'] else '✗'} "
            f"{quality_check['score']:.1f}/10 ═══"
        )
        return result
