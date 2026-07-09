"""
EEAT Enforcer — Validates and enhances section content for EEAT compliance.

Layer 2 of the 3-layer article generation architecture:
    1. Analyze each generated section for EEAT gaps
    2. Enforce mandatory elements: practical examples, scenario-based explanations,
       technical reasoning, precise language
    3. Score each section and trigger re-generation if below threshold

Quality gates:
    - No vague/filler statements
    - Must have practical example
    - Must have "why it matters"
    - Must have technical reasoning
    - Must avoid generic language
"""

import logging
import re
from typing import Optional

from core.llm.llm_client import llm
from core.content.section_writer import _build_language_enforcement

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# EEAT validation prompt
# ──────────────────────────────────────────

EEAT_VALIDATOR_SYSTEM = """You are an editorial quality gate specializing in EEAT compliance.

Analyze the given article section and return a JSON evaluation.

SCORING CRITERIA (1-10 each):
- practical_example: Does it contain a concrete, real-world example or scenario with SPECIFIC details (names, numbers, outcomes)?
- technical_depth: Does it explain HOW/WHY, not just WHAT? Does it cover the mechanism or process?
- specificity: Does it use precise data points (numbers, percentages, timeframes, metrics)? Generic phrases like "nhiều" or "tốt" score low.
- no_filler: Is it free from vague statements, generic advice, and AI clichés?
- actionability: Can the reader take concrete action based on this section?
- comparison: If alternatives exist, are they compared with specific criteria? (rate N/A as 7)
- flow: Does the writing flow logically with smooth transitions?
- data_backed: Are claims supported by specific numbers, study references, or measurable outcomes? (NOT just "theo nghiên cứu" without details)
- counter_argument: Does the section acknowledge limitations, risks, or alternative viewpoints? (rate N/A as 7 if topic is purely factual)
- tone_variety: Is the writing varied in rhythm (mix of paragraph lengths, analysis/example alternation)? Or is every paragraph the same pattern?
- case_study_quality: Are case studies focused on STRATEGIC ANALYSIS (not statistical reporting)? Must NOT contain fabricated numbers (follower counts, percentages, revenue). Must NOT use real brand names without public sources. Should follow: Context → Strategic Move → Execution Model → Why It Worked → Takeaway. Use qualitative language for outcomes. Score 1-3 if contains fabricated specific numbers or unverified brand claims. (rate N/A as 7 if no case study present)

FLAG these specific problems if found:
- vague_statements: list of sentences that are too vague or generic
- filler_sentences: list of sentences that add no value
- missing_elements: list of what's missing (data point, counter-argument, risk discussion, practical scenario, etc.)
- unsupported_claims: list of claims that lack specific evidence or data
- repetitive_patterns: list of structural patterns that repeat across paragraphs (e.g., all paragraphs start the same way)
- unsourced_case_studies: list of case studies that contain fabricated specific numbers (follower counts, exact percentages, revenue figures) or use real brand names without citing public sources

Return JSON:
{
    "scores": {
        "practical_example": int,
        "technical_depth": int,
        "specificity": int,
        "no_filler": int,
        "actionability": int,
        "comparison": int,
        "flow": int,
        "data_backed": int,
        "counter_argument": int,
        "tone_variety": int,
        "case_study_quality": int
    },
    "overall_score": float (average of all scores),
    "pass": boolean (true if overall >= 7.0),
    "vague_statements": [],
    "filler_sentences": [],
    "missing_elements": [],
    "unsupported_claims": [],
    "repetitive_patterns": [],
    "unsourced_case_studies": [],
    "improvement_suggestions": []
}"""

EEAT_ENHANCER_SYSTEM = """You are an expert content editor specializing in RESTRUCTURING weak content into authoritative, data-backed writing.

You don't just "add signals" — you RESTRUCTURE the section to feel genuinely authoritative.

ABSOLUTE RULES (VIOLATION = FAILURE):
1. Keep ALL headings EXACTLY as they are — do NOT change, rephrase, shorten, expand, add, or remove ANY heading text
2. Keep the SAME heading levels (## stays ##, ### stays ###, #### stays ####)
3. Do NOT add new headings that were not in the original
4. Do NOT remove any heading from the original
5. Every heading from the original MUST appear in your output, character-for-character identical

RESTRUCTURING RULES:
6. Replace EVERY unsupported claim with a data-backed version: add specific numbers, percentages, timeframes, or measurable outcomes
7. For vague authority references ("theo chuyên gia", "nhiều nghiên cứu"), either ADD specific source details or REMOVE the reference entirely
8. Add counter-arguments or risk discussions where flagged — structure as: "However, [specific concern]. Data from [context] suggests [nuance]."
9. Replace repetitive paragraph patterns: if 3+ paragraphs follow the same structure, vary the opening and flow
10. Replace filler sentences with specific insights or remove them
11. Vary paragraph lengths: mix 2-sentence and 4-sentence paragraphs for rhythm
12. Add at least one "However" or "On the other hand" statement to show balanced thinking
13. Keep Markdown formatting
14. The enhanced version should be roughly the same length — replace weak content with strong content, don't just append

CASE STUDY QUALITY FIX (BẮT BUỘC):
15. Nếu phát hiện case study có SỐ LIỆU BỊA (đếm follower, lượt xem, %, doanh thu cụ thể không có nguồn):
    - XÓA số liệu cụ thể, thay bằng ngôn ngữ định tính ("tăng trưởng rõ rệt", "cải thiện đáng kể")
    - Chuyển sang format phân tích chiến lược: Bối cảnh → Quyết định chiến lược → Mô hình triển khai → Tại sao hiệu quả → Bài học
    - Nếu dùng tên brand thật không có nguồn → thay bằng mô tả chung ("một thương hiệu thời trang trên TikTok")
    - Ví dụ fix: Thay "Tăng từ 50.000 lên 300.000, tăng 500%" → "Trong vài tháng triển khai chiến lược influencer kết hợp nội dung xu hướng, thương hiệu ghi nhận sự tăng trưởng rõ rệt về mức độ nhận diện và tương tác."

DATA INTEGRATION:
- When adding data points, reference VERIFIABLE industry patterns or publicly available research
- Do NOT fabricate specific numbers for case studies (follower counts, exact percentages, revenue)
- Use qualitative outcomes: "đạt tăng trưởng rõ rệt", "cải thiện đáng kể", "mức độ nhận diện tăng mạnh"
- For non-case-study content: specific numbers/percentages are OK when citing industry benchmarks or publicly known data
- Replace vague authority ("theo chuyên gia") with industry context ("theo xu hướng ngành [domain] 2024-2025")
"""


class EEATEnforcer:
    """Validates and enhances article sections for EEAT compliance."""

    # Minimum score to pass without enhancement
    PASS_THRESHOLD = 7.0
    # Maximum enhancement attempts per section
    MAX_ENHANCE_ATTEMPTS = 1

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, always validate and enhance. If False, only log warnings.
        """
        self.strict_mode = strict_mode

    def validate_section(self, section_content: str, section_heading: str) -> dict:
        """Validate a section's EEAT quality.

        Args:
            section_content: The generated Markdown content for the section
            section_heading: The H2 heading (for logging)

        Returns:
            Validation result dict with scores, pass/fail, and suggestions
        """
        if not section_content or len(section_content.split()) < 30:
            return {
                "scores": {},
                "overall_score": 0,
                "pass": False,
                "missing_elements": ["Content too short"],
                "improvement_suggestions": ["Generate more substantive content"],
            }

        prompt = f"""SECTION HEADING: {section_heading}

SECTION CONTENT:
{section_content}

Evaluate this section's EEAT quality."""

        try:
            result = llm.chat_json(
                EEAT_VALIDATOR_SYSTEM, prompt,
                temperature=0.2, max_tokens=1024,
            )
            overall = result.get("overall_score", 0)
            passed = result.get("pass", overall >= self.PASS_THRESHOLD)

            logger.info(
                f"[EEATEnforcer] '{section_heading}' — score: {overall:.1f}/10 "
                f"{'✓ PASS' if passed else '✗ NEEDS IMPROVEMENT'}"
            )

            if not passed:
                missing = result.get("missing_elements", [])
                if missing:
                    logger.info(f"[EEATEnforcer] Missing: {', '.join(missing)}")

            return result

        except Exception as e:
            logger.warning(f"[EEATEnforcer] Validation failed for '{section_heading}': {e}")
            # On error, assume pass to not block the pipeline
            return {"scores": {}, "overall_score": 7.0, "pass": True}

    def enhance_section(
        self,
        section_content: str,
        section_heading: str,
        validation_result: dict,
        knowledge_context: str = "",
        product_context: str = "",
        language: str = "vi",
    ) -> str:
        """Enhance a section that failed EEAT validation.

        Args:
            section_content: Original section content
            section_heading: The H2 heading
            validation_result: Result from validate_section()
            knowledge_context: Optional RAG knowledge for enhancement
            product_context: Optional product data for enhancement
            language: Content language for output enforcement

        Returns:
            Enhanced section content
        """
        vague = validation_result.get("vague_statements", [])
        filler = validation_result.get("filler_sentences", [])
        missing = validation_result.get("missing_elements", [])
        suggestions = validation_result.get("improvement_suggestions", [])
        unsupported = validation_result.get("unsupported_claims", [])
        repetitive = validation_result.get("repetitive_patterns", [])

        feedback_parts = []
        if vague:
            feedback_parts.append(f"VAGUE STATEMENTS TO FIX:\n" + "\n".join(f"- \"{v}\"" for v in vague))
        if filler:
            feedback_parts.append(f"FILLER TO REMOVE:\n" + "\n".join(f"- \"{f}\"" for f in filler))
        if missing:
            feedback_parts.append(f"MISSING ELEMENTS TO ADD:\n" + "\n".join(f"- {m}" for m in missing))
        if unsupported:
            feedback_parts.append(f"UNSUPPORTED CLAIMS (must add specific data):\n" + "\n".join(f"- \"{u}\"" for u in unsupported))
        unsourced_cases = validation_result.get("unsourced_case_studies", [])
        if unsourced_cases:
            feedback_parts.append(
                f"CASE STUDIES WITH FABRICATED DATA (must fix):\n"
                + "\n".join(f"- \"{c}\"" for c in unsourced_cases)
                + "\n→ Xóa số liệu cụ thể, thay bằng ngôn ngữ định tính. Chuyển sang format: Bối cảnh → Chiến lược → Triển khai → Tại sao hiệu quả → Bài học."
            )
        if repetitive:
            feedback_parts.append(f"REPETITIVE PATTERNS TO VARY:\n" + "\n".join(f"- {r}" for r in repetitive))
        if suggestions:
            feedback_parts.append(f"IMPROVEMENT SUGGESTIONS:\n" + "\n".join(f"- {s}" for s in suggestions))

        feedback_text = "\n\n".join(feedback_parts)

        prompt = f"""SECTION: {section_heading}

CURRENT CONTENT:
{section_content}

═══ EDITORIAL FEEDBACK ═══
{feedback_text}
═══ END FEEDBACK ═══
"""
        if knowledge_context:
            prompt += f"\nAVAILABLE KNOWLEDGE (use to add specificity):\n{knowledge_context[:1500]}\n"
        if product_context:
            prompt += f"\nPRODUCT DATA (use if relevant):\n{product_context[:500]}\n"

        prompt += "\nRewrite the section addressing ALL the feedback. Keep the same headings."

        try:
            system = _build_language_enforcement(language) + EEAT_ENHANCER_SYSTEM
            enhanced = llm.chat(
                system, prompt,
                temperature=0.6, max_tokens=3000,
            )
            logger.info(
                f"[EEATEnforcer] Enhanced '{section_heading}' — "
                f"{len(enhanced.split())} words"
            )
            return enhanced
        except Exception as e:
            logger.warning(f"[EEATEnforcer] Enhancement failed for '{section_heading}': {e}")
            return section_content  # Return original on failure

    def enforce(
        self,
        section_content: str,
        section_heading: str,
        knowledge_context: str = "",
        product_context: str = "",
        language: str = "vi",
    ) -> tuple[str, dict]:
        """Full enforcement pipeline: validate → enhance if needed.

        Args:
            section_content: Generated section content
            section_heading: H2 heading
            knowledge_context: Optional RAG knowledge
            product_context: Optional product data
            language: Content language for output enforcement

        Returns:
            Tuple of (final_content, validation_result)
        """
        # Step 1: Validate
        validation = self.validate_section(section_content, section_heading)

        if validation.get("pass", False):
            return section_content, validation

        if not self.strict_mode:
            logger.warning(
                f"[EEATEnforcer] '{section_heading}' below threshold but strict_mode=False, skipping"
            )
            return section_content, validation

        # Step 2: Enhance
        enhanced = section_content
        for attempt in range(self.MAX_ENHANCE_ATTEMPTS):
            enhanced = self.enhance_section(
                section_content=enhanced,
                section_heading=section_heading,
                validation_result=validation,
                knowledge_context=knowledge_context,
                product_context=product_context,
                language=language,
            )

            # Re-validate
            validation = self.validate_section(enhanced, section_heading)
            if validation.get("pass", False):
                logger.info(f"[EEATEnforcer] '{section_heading}' passed after enhancement")
                break

        return enhanced, validation

    def analyze_global_eeat(self, article_text: str, keyword: str = "") -> dict:
        """Deep EEAT analysis of the full article for authority assessment.

        This runs BEFORE conclusion generation so the conclusion can reflect
        the article's authority level.

        Returns:
            dict with authority_signals, missing_authority, data_density,
                  counter_argument_count, tone_variation_score, and
                  authority_summary for the conclusion writer.
        """
        # Count data-backed sentences (contain numbers/percentages)
        data_pattern = re.compile(r'\d+[%,.\d]*\s*(%|phần trăm|lần|tháng|năm|ngày|triệu|nghìn|tỷ|USD|VND|\$)')
        sentences = re.split(r'(?<=[.!?])\s+', article_text)
        data_sentences = [s for s in sentences if data_pattern.search(s)]
        data_density = len(data_sentences) / max(len(sentences), 1) * 100

        # Count counter-arguments / balanced viewpoints
        counter_patterns = [
            r'tuy nhiên', r'however', r'mặt khác', r'on the other hand',
            r'ngược lại', r'rủi ro', r'risk', r'hạn chế', r'limitation',
            r'nhược điểm', r'downside', r'cần lưu ý', r'caution',
            r'không phải lúc nào', r'not always',
        ]
        counter_count = sum(
            1 for sent in sentences
            if any(re.search(p, sent, re.IGNORECASE) for p in counter_patterns)
        )

        # Analyze paragraph pattern variety
        paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip() and not p.strip().startswith('#')]
        para_lengths = [len(p.split()) for p in paragraphs]
        length_variance = 0
        if len(para_lengths) > 1:
            avg_len = sum(para_lengths) / len(para_lengths)
            length_variance = sum((l - avg_len) ** 2 for l in para_lengths) / len(para_lengths)

        # Check opening pattern variety
        opening_words = [p.split()[0].lower() if p.split() else '' for p in paragraphs]
        unique_openings = len(set(opening_words))
        opening_variety = unique_openings / max(len(opening_words), 1) * 100

        # Build authority summary for conclusion
        authority_points = []
        if data_density >= 15:
            authority_points.append(f"Article contains {len(data_sentences)} data-backed statements")
        if counter_count >= 2:
            authority_points.append(f"Acknowledges {counter_count} counter-arguments/limitations")
        if opening_variety >= 60:
            authority_points.append("Writing shows varied, non-formulaic structure")

        missing = []
        if data_density < 10:
            missing.append("Low data density — needs more specific numbers/metrics")
        if counter_count < 2:
            missing.append("Missing counter-arguments — add balanced viewpoints")
        if opening_variety < 40:
            missing.append("Repetitive paragraph patterns — vary opening structures")
        if length_variance < 100:
            missing.append("Uniform paragraph lengths — vary between short (2-sent) and longer (4-sent)")

        result = {
            "data_density_pct": round(data_density, 1),
            "data_backed_sentences": len(data_sentences),
            "counter_argument_count": counter_count,
            "paragraph_length_variance": round(length_variance, 1),
            "opening_variety_pct": round(opening_variety, 1),
            "authority_signals": authority_points,
            "missing_authority": missing,
            "authority_summary": (
                f"Bài viết có {len(data_sentences)} câu được hỗ trợ bằng dữ liệu cụ thể, "
                f"{counter_count} góc nhìn phản biện. "
                + ("Bài viết thể hiện authority tốt." if not missing else
                   f"Cần cải thiện: {'; '.join(missing[:2])}")
            ),
            "needs_improvement": len(missing) > 0,
        }

        logger.info(
            f"[EEATEnforcer] Global EEAT: data_density={data_density:.0f}%, "
            f"counter_args={counter_count}, opening_variety={opening_variety:.0f}%"
        )

        return result

    def inject_authority_signals(
        self, article_text: str, global_analysis: dict, knowledge_context: str = "",
        language: str = "vi",
    ) -> str:
        """Inject missing authority signals into the assembled article.

        Called when global EEAT analysis reveals gaps (low data density,
        missing counter-arguments, repetitive patterns).

        Args:
            article_text: Full assembled article
            global_analysis: Output from analyze_global_eeat()
            knowledge_context: Optional RAG knowledge for data enhancement
            language: Content language for output enforcement

        Returns:
            Enhanced article text with injected authority signals
        """
        missing = global_analysis.get("missing_authority", [])
        if not missing:
            return article_text

        prompt = f"""ARTICLE:
{article_text[:8000]}

═══ AUTHORITY ANALYSIS ═══
Data density: {global_analysis.get('data_density_pct', 0):.0f}% of sentences have specific data
Counter-arguments found: {global_analysis.get('counter_argument_count', 0)}
Opening variety: {global_analysis.get('opening_variety_pct', 0):.0f}%

MISSING AUTHORITY SIGNALS:
{chr(10).join(f'- {m}' for m in missing)}
"""
        if knowledge_context:
            prompt += f"\nKNOWLEDGE (use for adding data):\n{knowledge_context[:2000]}\n"

        prompt += """\nENHANCE the article by:
1. Replacing 3-5 vague claims with data-backed versions (specific numbers, percentages)
2. Adding 1-2 counter-argument/risk statements where the content is one-sided
3. Varying paragraph opening words if too many start the same way
4. Keep ALL headings EXACTLY as they are
5. Keep the same overall length — replace weak content, don't append

Return the FULL enhanced article."""

        try:
            system = _build_language_enforcement(language) + EEAT_ENHANCER_SYSTEM
            enhanced = llm.chat(
                system, prompt,
                temperature=0.5, max_tokens=8000,
            )
            logger.info("[EEATEnforcer] Authority signals injected into full article")
            return enhanced
        except Exception as e:
            logger.warning(f"[EEATEnforcer] Authority injection failed: {e}")
            return article_text

    def validate_full_article(self, article_text: str) -> dict:
        """Quick quality check on the assembled full article.

        Checks:
            - No duplicate paragraphs
            - Consistent heading hierarchy
            - Minimum word count per section
            - No orphan headings (heading without content)
            - Tone variation (paragraph pattern variety)

        Returns:
            dict with issues list and overall pass/fail
        """
        issues = []

        lines = article_text.split("\n")
        headings = []
        current_heading = None
        current_content_len = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("### ") or stripped.startswith("#### "):
                if current_heading and current_content_len < 20:
                    issues.append(f"Orphan heading (too little content): {current_heading}")
                current_heading = stripped
                current_content_len = 0
                headings.append(stripped)
            elif stripped:
                current_content_len += len(stripped.split())

        # Check for duplicate paragraphs
        paragraphs = [p.strip() for p in article_text.split("\n\n") if len(p.strip()) > 50]
        seen = set()
        for p in paragraphs:
            normalized = re.sub(r'\s+', ' ', p.lower().strip())
            if normalized in seen:
                issues.append(f"Duplicate paragraph found: {p[:80]}...")
            seen.add(normalized)

        # Check heading hierarchy
        for i, h in enumerate(headings):
            level = len(h.split(" ")[0])  # count #
            if i == 0 and level > 2:
                issues.append(f"First heading should be H2, got: {h}")

        # Check tone variation — flag repetitive opening patterns
        content_paragraphs = [
            p.strip() for p in article_text.split("\n\n")
            if p.strip() and not p.strip().startswith('#') and len(p.strip().split()) > 10
        ]
        if content_paragraphs:
            openings = [p.split()[0].lower() for p in content_paragraphs if p.split()]
            from collections import Counter
            opening_counts = Counter(openings)
            for word, count in opening_counts.items():
                if count >= 4:
                    issues.append(
                        f"Tone monotony: {count} paragraphs start with '{word}' — vary openings"
                    )

        word_count = len(article_text.split())

        return {
            "pass": len(issues) == 0,
            "issues": issues,
            "word_count": word_count,
            "heading_count": len(headings),
        }
