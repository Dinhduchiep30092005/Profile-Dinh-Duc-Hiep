"""
Renderer — Final HTML post-processing layer.

Receives article_html from ArticleBuilder (Phase 17) and applies
lightweight normalizations before the article is saved to DB.
"""

import logging
import re

logger = logging.getLogger(__name__)


class Renderer:
    """Post-processes generated article HTML for final output."""

    def render(
        self,
        article_html: str,
        article_text: str = "",
        language: str = "vi",
    ) -> str:
        """
        Final normalization pass on the generated HTML.

        Steps:
          1. Remove stray Markdown code-fence artifacts (``` ```)
          2. Collapse 3+ consecutive blank lines → 2
          3. Strip leading/trailing whitespace

        Args:
            article_html: Raw HTML from ArticleBuilder Phase 17
            article_text: Markdown source (for logging/debugging only)
            language: Article language code

        Returns:
            Cleaned HTML string ready for DB storage / WordPress.
        """
        if not article_html:
            logger.warning("[Renderer] Empty article_html — nothing to render")
            return article_html

        html = article_html

        # 1. Remove stray Markdown code fences
        html = re.sub(r"```[\w]*\n?", "", html)
        html = re.sub(r"```", "", html)

        # 2. Collapse excess blank lines
        html = re.sub(r"\n{3,}", "\n\n", html)

        # 3. Strip outer whitespace
        html = html.strip()

        char_delta = len(html) - len(article_html)
        logger.info(
            f"[Renderer] ✓ Render complete — {len(html):,} chars "
            f"({'−' if char_delta < 0 else '+'}{abs(char_delta)} from cleanup)"
        )
        return html
