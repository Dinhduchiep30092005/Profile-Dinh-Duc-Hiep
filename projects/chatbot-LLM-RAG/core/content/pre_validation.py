"""
PreValidationLayer — Validates all inputs before article generation.

Sits between ArticleController and ArticleGenerator.
Catches bad inputs early so the expensive LLM pipeline never fires on garbage.
"""

import logging
from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "vi", "en", "fr", "de", "es", "ja", "ko", "zh", "th", "id", "pt",
}


class PreValidationLayer:
    """Validates keyword, title, heading_structure, and API key before generation."""

    def validate(
        self,
        keyword: str,
        title: str,
        heading_structure: dict,
        language: str = "vi",
    ) -> dict:
        """
        Validate all inputs required for article generation.

        Returns:
            dict:
                valid (bool)      — False if any hard error is found
                errors (list[str])  — blocking issues
                warnings (list[str]) — non-blocking notices
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── API Key ──
        if not settings.openai.api_key:
            errors.append("Chưa cấu hình OpenAI API Key. Vào Cài đặt để nhập key.")

        # ── Keyword ──
        if not keyword or not keyword.strip():
            errors.append("Keyword không được để trống.")
        elif len(keyword.strip()) < 2:
            errors.append("Keyword quá ngắn (tối thiểu 2 ký tự).")
        elif len(keyword.strip()) > 300:
            warnings.append("Keyword quá dài — nên dưới 300 ký tự.")

        # ── Title ──
        if not title or not title.strip():
            errors.append("Title không được để trống.")
        elif len(title.strip()) < 5:
            warnings.append("Title rất ngắn — nên ít nhất 5 ký tự để SEO hiệu quả.")

        # ── Heading Structure ──
        if not heading_structure:
            errors.append("Chưa có heading structure.")
        elif not isinstance(heading_structure, dict):
            errors.append("Heading structure không hợp lệ (phải là dict).")
        elif not heading_structure.get("sections"):
            errors.append("Heading structure thiếu sections. Hãy tạo heading trước.")
        else:
            sections = heading_structure["sections"]
            if len(sections) < 2:
                warnings.append(
                    f"Heading chỉ có {len(sections)} section — bài viết có thể quá ngắn."
                )

        # ── Language ──
        if language not in SUPPORTED_LANGUAGES:
            warnings.append(
                f"Ngôn ngữ '{language}' chưa được kiểm thử đầy đủ. "
                f"Hỗ trợ tốt nhất: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
            )

        if warnings:
            for w in warnings:
                logger.warning(f"[PreValidation] ⚠ {w}")
        if errors:
            for e in errors:
                logger.error(f"[PreValidation] ✗ {e}")
        else:
            logger.info("[PreValidation] ✓ All inputs valid")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
