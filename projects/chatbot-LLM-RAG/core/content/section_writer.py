"""
Section Writer — Generates content section-by-section (per H2).

Layer 1 of the 3-layer article generation architecture:
    1. Loop through each H2 section
    2. Generate deep, focused content for that section
    3. Inject product data, technical explanation, use cases
    4. Each section gets its own tailored prompt → deeper, less generic content

Key benefits over full-article generation:
    - Much deeper content per section
    - No repetition across sections
    - Targeted context injection per section
    - Better EEAT signal per section
"""

import logging
from typing import Optional

from core.llm.llm_client import llm
from core.content.truncation_detector import detect_truncation

logger = logging.getLogger(__name__)

# Max retry attempts when truncation is detected
_MAX_TRUNCATION_RETRIES = 2
# Token multiplier for each retry (e.g., 1.5x, then 2.25x)
_TOKEN_RETRY_MULTIPLIER = 1.5


# ──────────────────────────────────────────
# Language enforcement block (prepended to system prompts)
# ──────────────────────────────────────────

def _build_language_enforcement(language: str) -> str:
    """Build a STRONG language enforcement block to prepend to system prompts.

    This overrides any language bias from Vietnamese instructions in the prompt.
    Placed at the TOP of system prompt for maximum LLM attention.
    """
    if not language or language.lower().startswith("vi"):
        return ""  # Vietnamese is default, no override needed

    lang_names = {
        "en": "English", "en-us": "American English", "en-gb": "British English",
        "en-au": "Australian English",
        "fr": "French", "de": "German", "es": "Spanish",
        "ja": "Japanese", "ko": "Korean",
        "zh": "Simplified Chinese", "zh-tw": "Traditional Chinese",
        "th": "Thai", "pt": "Brazilian Portuguese", "it": "Italian",
    }
    lang_name = lang_names.get(language.lower(), language)

    return f"""════════════════════════════════════════════
CRITICAL LANGUAGE RULE — THIS OVERRIDES ALL OTHER INSTRUCTIONS:
You MUST write ENTIRELY in {lang_name}.
- Every sentence, heading, paragraph, example, and explanation MUST be in {lang_name}.
- Do NOT use Vietnamese or any other language — not even for examples, case studies, or labels.
- If this prompt contains Vietnamese instructions or examples, those are for YOUR reference only.
  Your OUTPUT must be 100% in {lang_name}.
- Translate any Vietnamese terms, patterns, or examples into {lang_name} equivalents.
- This rule is ABSOLUTE and overrides all other instructions below.
════════════════════════════════════════════

"""


# ──────────────────────────────────────────
# Section-level prompt templates
# ──────────────────────────────────────────

SECTION_WRITER_SYSTEM = """You are an expert content writer specializing in deep, authoritative technical articles.

You are writing ONE SECTION of a larger article. Focus ONLY on this section — make it the best it can be.

ABSOLUTE RULES:
1. Write ONLY the content for the given section — do NOT write intro/conclusion for the whole article
2. Use the heading text EXACTLY as provided — copy the heading text CHARACTER BY CHARACTER, do NOT rephrase, reword, shorten, expand, or modify any heading in any way
3. Keep the EXACT heading level: ## stays ##, ### stays ###, #### stays #### — do NOT change # count
4. Do NOT add any headings that are not in the given structure — do NOT invent new H2, H3, or H4
   - ESPECIALLY: NEVER create headings with "Kết Luận", "Tổng Kết", "Lời Kết", "Conclusion" — the conclusion is written separately
5. Do NOT skip or remove any heading from the given structure
6. Every paragraph MUST add concrete value — NO filler, NO vague statements, NO generic advice
7. Use precise, specific language — avoid words like "rất", "nhiều", "tốt" without quantification
8. Include practical examples, real scenarios, and technical reasoning
9. Use Markdown: ## H2, ### H3, #### H4, - lists, | tables |, > blockquotes
10. Do NOT repeat information that would be in other sections of the article
11. Do NOT use ** (bold) or * (italic) markdown around links — links already have their own styling

4-LAYER STRUCTURE (MANDATORY for every H2 section):
These are INTERNAL writing guidelines — do NOT output these layer names as headings or sub-headings in the article.
L1 — Bối cảnh mở đầu: Open with a specific fact, data point, or observation that frames the topic. NEVER with a definition or generic statement.
L2 — Phân tích cốt lõi: Deep technical/strategic breakdown. Explain the mechanism, process, or framework.
L3 — Tình huống thực tế: A concrete real-world example with SPECIFIC details (numbers, user types, outcomes).
L4 — Điểm nhấn hành động: What the reader should DO with this information. One clear, actionable recommendation.

CRITICAL: These 4 layers define the FLOW of your writing, NOT actual headings. Write them as natural flowing paragraphs — NEVER create sub-headings or bold titles for each layer.
FORBIDDEN HEADINGS IN BODY SECTIONS: NEVER output any heading, sub-heading, or bold title containing: "Kết luận", "Tóm tắt", "Tổng kết", "Lời kết", "Conclusion", "Summary", "Strategic Takeaway", "Điểm nhấn", "Bối cảnh", "Phân tích cốt lõi", "Tình huống thực tế". These words are NEVER headings — they are internal flow instructions only.

WORD COUNT DISCIPLINE (CRITICAL):
- You will be given a STRICT word budget. RESPECT IT.
- Be concise but substantive — every sentence must earn its place
- Prefer short, impactful paragraphs (2-3 sentences) over long ones
- Use bullet lists to convey information efficiently
- Cut filler words and redundant explanations

DATA-BACKED WRITING (MANDATORY):
- Every section MUST contain at least 2 specific numbers, percentages, timeframes, or metrics
- Do NOT use generic authority: "theo chuyên gia" — instead: "theo dữ liệu từ các chiến dịch [domain] 2024-2025"
- Replace "nhiều" with exact numbers, replace "tốt hơn" with "cao hơn X%"
- Include at least 1 counter-argument, limitation, or risk acknowledgment per section

CASE STUDY WRITING RULES (BẮT BUỘC):
- KHÔNG được bịa số liệu cụ thể (số follow, lượt xem, phần trăm, doanh thu, tỷ lệ giữ chân...)
- KHÔNG được dùng tên thương hiệu thật trừ khi có nguồn công khai đính kèm
- CHỈ TẬP TRUNG phân tích:
  • Quyết định chiến lược (strategic decisions)
  • Định vị nội dung (content positioning)
  • Mô hình hợp tác influencer (collaboration model)
  • Logic tần suất đăng bài (posting frequency logic)
  • Hành vi nền tảng (platform behavior patterns)
- Khi mô tả kết quả: dùng ngôn ngữ định tính ("tăng trưởng rõ rệt", "cải thiện tương tác đáng kể"), KHÔNG dùng con số cụ thể
- FORMAT bắt buộc cho case study: Bối cảnh (Context) → Quyết định chiến lược (Strategic Move) → Mô hình triển khai (Execution Model) → Tại sao hiệu quả (Why It Worked) → Bài học chiến lược (Strategic Takeaway)
- Mục đích case study là PHÂN TÍCH CHIẾN LƯỢC, không phải báo cáo số liệu

TONE VARIETY:
- VARY paragraph lengths: mix 2-sentence and 4-sentence paragraphs
- VARY opening words: do NOT start 3+ paragraphs with the same word
- ALTERNATE between analysis and concrete examples
- Include at least one "Tuy nhiên" or contra-perspective per section

WRITING STYLE:
- Professional, authoritative tone
- Short paragraphs (2-3 sentences max)
- Active voice preferred
- Technical precision — use correct terminology
- Logical flow — each paragraph connects to the next
- No AI clichés ("in today's digital landscape", "it's important to note that", etc.)
- No "Hiện nay", "Trong thời đại số", "Không thể phủ nhận" patterns
"""

INTRO_WRITER_SYSTEM = """You are an expert content writer. Write a compelling article introduction.

RULES:
1. Hook the reader in SENTENCE ONE — use a specific data point, surprising fact, or counter-intuitive observation. NEVER a generic statement.
2. Establish expertise by showing you understand the NUANCE of this topic (not just the surface level)
3. Preview what the article covers WITHOUT listing headings
4. Include the target keyword naturally in the first 100 words
5. 100-150 words MAXIMUM — be extremely concise
6. NO generic statements like "In this article, we will discuss...", "Hiện nay...", "Trong thời đại số..."
7. Create urgency or curiosity — why should they read NOW?
8. Do NOT mention any product or brand in the intro — pure value first
9. Use Markdown formatting
10. Start with a NUMBER or SPECIFIC FACT, not a question or generic claim

BAD intro pattern (NEVER do this):
"Tăng follow TikTok là một trong những chiến lược quan trọng nhất để phát triển thương hiệu..."

GOOD intro pattern:
"Tài khoản TikTok vượt mốc 50.000 follow có tỷ lệ được đề xuất cao hơn 23% so với nhóm dưới 10.000 — nhưng con số follow thôi chưa đủ..."
"""

CONCLUSION_WRITER_SYSTEM = """You are an expert content writer and conversion copywriter. Write a powerful article conclusion that MOVES the reader to action.

STRUCTURE (3 parts, seamlessly connected):

1. SYNTHESIS (3-4 sentences):
   - Distill the 3-5 most important insights from the article using SPECIFIC data from the article
   - Connect them into a coherent narrative — do NOT just list headers
   - Reference specific numbers or findings mentioned in the article body
   - Emphasize the TRANSFORMATION: what the reader now understands that they didn't before

2. BRIDGE TO PRODUCT (2-3 sentences):
   - Identify the reader's remaining challenge: "Biết WHAT to do is one thing, having the right TOOL to execute is another"
   - Naturally transition from the article's insights to how the product solves the exact problem discussed
   - Be specific about WHICH feature/capability maps to WHICH problem from the article
   - Reference the article's authority findings to strengthen the recommendation
   - Do NOT be generic — tie the product directly to the article's content

3. CALL TO ACTION (1-2 sentences):
   - Create urgency or opportunity: what they gain by acting NOW
   - Use confident, inviting language — not pushy or salesy
   - End with a clear next step (try, explore, contact, sign up)
   - If a URL or trial is available, mention it naturally

TONE RULES:
- Sound like a trusted advisor — someone who has demonstrated expertise throughout the article
- The conclusion must REFLECT the authority built in the article body
- Never use words like "CTA", "Kêu gọi hành động", "Tóm tắt" explicitly
- Avoid generic hype: "tốt nhất", "tuyệt vời", "không thể bỏ qua" without backing
- Be specific and concrete — mention actual features, actual benefits
- Write naturally, like an expert sharing their honest recommendation
- 150-200 words maximum including the ## heading
- Use Markdown formatting (## for heading)
"""

FAQ_WRITER_SYSTEM = """You are an expert content writer. Write comprehensive FAQ answers.

RULES:
1. Each answer: 2-3 sentences, SPECIFIC and actionable
2. Include technical details, not vague generalities
3. Reference information from the article context when relevant
4. Use Markdown formatting
5. Format: ## FAQ heading, then ### for each question
6. Keep the entire FAQ section under 300 words total
"""


class SectionWriter:
    """Generates article content section-by-section for maximum depth."""

    def __init__(self, temperature: float = 0.65):
        """
        Args:
            temperature: LLM temperature (0.6-0.7 recommended for professional content)
        """
        self.temperature = temperature

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def write_intro(
        self,
        keyword: str,
        title: str,
        section_summaries: list[str],
        knowledge_context: str = "",
        language: str = "vi",
        search_intent: dict | None = None,
    ) -> str:
        """Generate the article introduction, aligned with search intent.

        Args:
            keyword: Target keyword
            title: Article title
            section_summaries: List of H2 headings (to hint at article scope)
            knowledge_context: Optional RAG knowledge context
            language: Content language
            search_intent: Dict with intent_type, confidence, signals

        Returns:
            Markdown intro text
        """
        lang_instruction = self._lang_instruction(language)

        # Intent-specific intro guidance
        intent_type = (search_intent or {}).get("intent_type", "informational")
        intent_guidance = {
            "informational": (
                "Hook the reader with a surprising fact or common misconception about this topic. "
                "Establish WHY understanding this topic matters. "
                "Promise clear, actionable insights the reader will gain."
            ),
            "commercial": (
                "Acknowledge the reader's decision-making challenge directly. "
                "Establish your authority to compare/evaluate options. "
                "Promise an honest, criteria-based analysis that helps them choose."
            ),
            "transactional": (
                "Address the reader's readiness to act/buy. "
                "Highlight the key benefit or value proposition immediately. "
                "Promise a clear path to making the right purchase/decision."
            ),
            "navigational": (
                "Quickly confirm the reader is in the right place. "
                "Provide a brief overview of what they'll find. "
                "Guide them to the most relevant section."
            ),
        }.get(intent_type, "")

        prompt = f"""ARTICLE TITLE: {title}
KEYWORD: {keyword}
LANGUAGE: {lang_instruction}
SEARCH INTENT: {intent_type}

Article covers these main topics:
{chr(10).join(f'- {s}' for s in section_summaries)}

INTRO STRATEGY (based on {intent_type} intent):
{intent_guidance}
"""
        if knowledge_context:
            prompt += f"\nBRAND/PRODUCT CONTEXT:\n{knowledge_context[:1500]}\n"

        prompt += "\nWrite the introduction. Start directly with the hook — no heading needed.\nKeep it under 180 words."

        try:
            system = _build_language_enforcement(language) + INTRO_WRITER_SYSTEM
            return llm.chat(
                system, prompt,
                temperature=self.temperature, max_tokens=600,
            )
        except Exception as e:
            logger.error(f"[SectionWriter] Intro generation failed: {e}")
            raise

    def write_section(
        self,
        keyword: str,
        title: str,
        section: dict,
        children: list[dict],
        section_index: int,
        total_sections: int,
        knowledge_context: str = "",
        product_context: str = "",
        internal_links: list[dict] | None = None,
        previous_section_summary: str = "",
        language: str = "vi",
        word_budget: int = 0,
        depth_strategy: str = "",
    ) -> str:
        """Generate content for a single H2 section and its children.

        Args:
            keyword: Target keyword
            title: Article title
            section: The H2 section dict (heading, purpose, content_type, etc.)
            children: List of H3/H4 children under this H2
            section_index: Index of this H2 (0-based) for context
            total_sections: Total number of H2 sections
            knowledge_context: RAG knowledge relevant to this section
            product_context: Product/brand info relevant to this section
            internal_links: Pre-calculated internal links (for reference only, NOT passed to LLM)
            previous_section_summary: 1-2 sentence summary of the previous section (for flow)
            language: Content language
            depth_strategy: Formatted depth strategy from DepthController

        Returns:
            Markdown content for this section (H2 + all children, NO links — links injected post-gen)
        """
        lang_instruction = self._lang_instruction(language)
        heading = section.get("heading", "")

        # Build the section structure for the prompt
        structure = self._format_section_structure(section, children)

        # Build context injection blocks (NO links — links injected post-generation)
        context_blocks = self._build_context_blocks(
            knowledge_context, product_context, internal_links=None
        )

        # Build section-specific EEAT instructions
        eeat_instruction = self._build_eeat_instruction(section)

        prompt = f"""ARTICLE: "{title}" (keyword: "{keyword}")
LANGUAGE: {lang_instruction}
SECTION {section_index + 1} OF {total_sections}

═══ SECTION STRUCTURE ═══
{structure}
═══ END STRUCTURE ═══

{eeat_instruction}
{context_blocks}
"""
        # Inject depth strategy from DepthController (4-layer, tone, mandatory elements)
        if depth_strategy:
            prompt += f"\n{depth_strategy}\n"
        if previous_section_summary:
            prompt += f"\nPREVIOUS SECTION ENDED WITH: {previous_section_summary}\n"
            prompt += "Ensure smooth transition — do NOT repeat what was already covered.\n"

        # Use explicit word budget if provided, otherwise fall back to section spec
        target_words = word_budget if word_budget > 0 else section.get('word_count_target', 250)

        prompt += f"""
WRITE the complete section now. Include the H2 heading as ## and all H3/H4 as ### / ####.
Every sub-section MUST have substantive content (not just 1-2 sentences).

CRITICAL — HEADING PRESERVATION:
- Copy EVERY heading EXACTLY as shown in the structure above — do NOT change even a single word.
- Do NOT add new headings that are not in the structure.
- Do NOT remove or skip any heading from the structure.
- EVERY heading (H2, H3, H4) MUST be followed by at least one full paragraph of content.
- NEVER place two headings back-to-back without content between them.

CRITICAL — NO LINKS:
- Do NOT insert any markdown links [text](url) in the content.
- Write pure content — links will be added separately in a post-processing step.
- If you need to mention a product or resource, just write the name as plain text.

WORD COUNT CONSTRAINT (STRICT):
- Target: approximately {target_words} words for this ENTIRE section (including all H3/H4).
- Do NOT exceed {int(target_words * 1.15)} words.
- Be concise but substantive — quality over quantity.
- Cut filler, keep only high-value content.
"""

        # Limit max_tokens proportionally to word budget (1 word ≈ 1.5 tokens for Vietnamese)
        max_tok = min(3000, max(800, int(target_words * 2.0)))

        try:
            system = _build_language_enforcement(language) + SECTION_WRITER_SYSTEM
            content = self._generate_with_truncation_guard(
                system_prompt=system,
                user_prompt=prompt,
                initial_max_tokens=max_tok,
                section_heading=heading,
                word_budget=target_words,
            )
            logger.info(
                f"[SectionWriter] Section '{heading}' — "
                f"{len(content.split())} words generated (budget: {target_words})"
            )
            return content
        except Exception as e:
            logger.error(f"[SectionWriter] Section '{heading}' failed: {e}")
            raise

    def write_conclusion(
        self,
        keyword: str,
        title: str,
        key_takeaways: list[str],
        product_context: str = "",
        language: str = "vi",
        brand_info: dict | None = None,
        search_intent: dict | None = None,
        conclusion_heading: str = "",
        authority_context: str = "",
    ) -> str:
        """Generate article conclusion with compelling product CTA.

        Args:
            keyword: Target keyword
            title: Article title
            key_takeaways: List of main points from the article
            product_context: Product info from RAG knowledge base
            language: Content language
            brand_info: Dict with product_name, product_description,
                        use_cases, competitive_advantages, target_audience
            search_intent: Dict with intent_type, confidence
            conclusion_heading: Exact heading text to use
            authority_context: Authority summary from global EEAT analysis

        Returns:
            Markdown conclusion text with integrated CTA
        """
        lang_instruction = self._lang_instruction(language)
        intent_type = (search_intent or {}).get("intent_type", "informational")

        prompt = f"""ARTICLE: "{title}" (keyword: "{keyword}")
LANGUAGE: {lang_instruction}
SEARCH INTENT: {intent_type}

KEY TAKEAWAYS FROM THE ARTICLE:
{chr(10).join(f'- {t}' for t in key_takeaways)}
"""
        # Inject authority context so conclusion reflects article's authority level
        if authority_context:
            prompt += f"""
═══ ARTICLE AUTHORITY ANALYSIS ═══
{authority_context}
═══ END AUTHORITY ═══
The conclusion MUST reflect this authority level. If the article demonstrated strong data-backed analysis,
the conclusion should reference specific findings. The conclusion is the CULMINATION of the article's authority.
"""

        # Build rich product/brand CTA context
        brand = brand_info or {}
        product_name = brand.get("product_name", "")
        product_desc = brand.get("product_description", "")
        advantages = brand.get("competitive_advantages", [])
        use_cases = brand.get("use_cases", [])
        target_audience = brand.get("target_audience", "")

        if product_name:
            prompt += f"\n═══ PRODUCT/BRAND FOR CTA ═══\n"
            prompt += f"Product Name: {product_name}\n"
            if product_desc:
                prompt += f"What it does: {product_desc[:300]}\n"
            if advantages:
                prompt += f"Key advantages: {', '.join(advantages[:4])}\n"
            if use_cases:
                prompt += f"Use cases: {', '.join(use_cases[:3])}\n"
            if target_audience:
                prompt += f"Target audience: {target_audience}\n"
            prompt += f"═══ END PRODUCT INFO ═══\n"

            # Intent-specific CTA direction
            cta_direction = {
                "informational": (
                    f"The reader just learned about {keyword}. Bridge from their new knowledge "
                    f"to how {product_name} helps them APPLY what they learned. "
                    f"Emphasize the practical advantage of using {product_name}."
                ),
                "commercial": (
                    f"The reader is comparing options for {keyword}. Position {product_name} "
                    f"as the smart choice by highlighting its unique advantages over alternatives. "
                    f"Be specific about what makes it different."
                ),
                "transactional": (
                    f"The reader is ready to act on {keyword}. Make {product_name} "
                    f"the obvious next step. Emphasize ease of getting started, "
                    f"immediate value, and low risk (free trial, easy setup, etc.)."
                ),
                "navigational": (
                    f"Guide the reader to the specific {product_name} resource they need. "
                    f"Be direct and helpful."
                ),
            }.get(intent_type, "")

            prompt += f"\nCTA DIRECTION: {cta_direction}\n"
        elif product_context:
            prompt += f"\nPRODUCT CONTEXT:\n{product_context[:800]}\n"
            prompt += (
                "\nCTA DIRECTION: Naturally recommend this product as the solution "
                "to the problems discussed in the article. Be specific about features.\n"
            )

        if conclusion_heading:
            prompt += (
                f"\nWrite the conclusion section. "
                f"Use EXACTLY this heading: ## {conclusion_heading}\n"
                f"Do NOT change, rephrase, or translate the heading above.\n"
                "Structure: Synthesis → Bridge to product → Clear call-to-action. "
                "Keep it 150-200 words. Make the CTA feel like a natural, honest recommendation."
            )
        else:
            prompt += (
                "\nWrite the conclusion section with ## heading. "
                "Structure: Synthesis → Bridge to product → Clear call-to-action. "
                "Keep it 150-200 words. Make the CTA feel like a natural, honest recommendation."
            )

        try:
            system = _build_language_enforcement(language) + CONCLUSION_WRITER_SYSTEM
            result = llm.chat(
                system, prompt,
                temperature=self.temperature, max_tokens=700,
            )
            # Enforce the original heading regardless of what the LLM produced
            if conclusion_heading:
                import re as _re
                result = _re.sub(
                    r'^(##\s+)(.+)$',
                    rf'\1{conclusion_heading}',
                    result,
                    count=1,
                    flags=_re.MULTILINE,
                )
            return result
        except Exception as e:
            logger.error(f"[SectionWriter] Conclusion generation failed: {e}")
            raise

    def write_faq(
        self,
        keyword: str,
        title: str,
        faq_items: list[dict],
        article_context: str = "",
        language: str = "vi",
    ) -> str:
        """Generate FAQ section with comprehensive answers.

        Args:
            keyword: Target keyword
            title: Article title
            faq_items: List of FAQ dicts (question, answer_outline)
            article_context: Brief context from the article for better answers
            language: Content language

        Returns:
            Markdown FAQ section
        """
        lang_instruction = self._lang_instruction(language)

        # Limit to 5 FAQ items max to control word count
        faq_items_limited = faq_items[:5]

        faq_text = "\n".join(
            f"Q: {f.get('question', '')}\n"
            f"Answer outline: {f.get('answer_outline', 'Provide a thorough answer')}"
            for f in faq_items_limited
        )

        prompt = f"""ARTICLE: "{title}" (keyword: "{keyword}")
LANGUAGE: {lang_instruction}

FAQ QUESTIONS:
{faq_text}

ARTICLE CONTEXT (for reference):
{article_context[:2000]}

Write the FAQ section. Start with ## FAQ (or ## Câu Hỏi Thường Gặp for Vietnamese).
Each question as ### heading, followed by a concise answer (2-3 sentences each).
Keep the ENTIRE FAQ section under 300 words total. Only include the {min(len(faq_items), 5)} most important questions.
"""

        try:
            system = _build_language_enforcement(language) + FAQ_WRITER_SYSTEM
            return llm.chat(
                system, prompt,
                temperature=self.temperature, max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"[SectionWriter] FAQ generation failed: {e}")
            raise

    # ──────────────────────────────────────────
    # Truncation guard (auto-retry on incomplete output)
    # ──────────────────────────────────────────

    def _generate_with_truncation_guard(
        self,
        system_prompt: str,
        user_prompt: str,
        initial_max_tokens: int,
        section_heading: str = "",
        word_budget: int = 0,
    ) -> str:
        """Generate content with automatic truncation detection and retry.

        Strategy:
            1. Call LLM with initial max_tokens
            2. Check finish_reason + text-level truncation signals
            3. If truncated:
               a. Try to continue from where it left off (cheaper)
               b. If still bad, retry from scratch with higher max_tokens
            4. Up to _MAX_TRUNCATION_RETRIES attempts

        Returns:
            The (hopefully complete) generated content
        """
        current_max_tokens = initial_max_tokens

        for attempt in range(1 + _MAX_TRUNCATION_RETRIES):
            # Use chat_with_metadata to get finish_reason
            result = llm.chat_with_metadata(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=current_max_tokens,
            )

            content = result["content"]
            finish_reason = result["finish_reason"]

            # Run truncation detection
            truncation = detect_truncation(content, finish_reason)

            if not truncation["is_truncated"]:
                if attempt > 0:
                    logger.info(
                        f"[SectionWriter] Section '{section_heading}' — "
                        f"truncation resolved on attempt {attempt + 1}"
                    )
                return content

            # Truncation detected
            logger.warning(
                f"[SectionWriter] Section '{section_heading}' — "
                f"truncation detected (attempt {attempt + 1}/{1 + _MAX_TRUNCATION_RETRIES}): "
                f"{'; '.join(truncation['reasons'])}"
            )

            if attempt >= _MAX_TRUNCATION_RETRIES:
                # Max retries exhausted — return what we have
                logger.warning(
                    f"[SectionWriter] Section '{section_heading}' — "
                    f"max retries exhausted, returning potentially truncated content"
                )
                return content

            # Strategy A: If finish_reason == "length", try to continue the content
            if finish_reason == "length":
                continued = self._continue_truncated_content(
                    content, system_prompt, section_heading,
                    continuation_max_tokens=min(1500, current_max_tokens),
                )
                if continued:
                    # Verify the continuation fixed the issue
                    merged = content.rstrip() + "\n" + continued.lstrip()
                    recheck = detect_truncation(merged, finish_reason="stop")
                    if not recheck["is_truncated"]:
                        logger.info(
                            f"[SectionWriter] Section '{section_heading}' — "
                            f"truncation fixed via continuation"
                        )
                        return merged
                    # Continuation didn't fully fix it — fall through to retry

            # Strategy B: Retry from scratch with higher max_tokens
            current_max_tokens = min(
                8192,
                int(current_max_tokens * _TOKEN_RETRY_MULTIPLIER),
            )
            logger.info(
                f"[SectionWriter] Section '{section_heading}' — "
                f"retrying with max_tokens={current_max_tokens}"
            )

        return content  # Fallback (should not reach here)

    def _continue_truncated_content(
        self,
        truncated_content: str,
        system_prompt: str,
        section_heading: str,
        continuation_max_tokens: int = 1500,
    ) -> str:
        """Ask the LLM to continue from where truncated content left off.

        Returns:
            The continuation text (empty string if failed)
        """
        # Take the last ~500 chars as context for the continuation
        tail_context = truncated_content[-500:] if len(truncated_content) > 500 else truncated_content

        continuation_prompt = f"""The following section content was cut off mid-generation. Continue writing EXACTLY from where it stopped.

DO NOT repeat any content that's already written. Start your output with the NEXT word after the cutoff.
DO NOT add any new headings that weren't planned. Just complete the interrupted content naturally.
If the content was cut mid-sentence, complete that sentence first, then continue.
Maintain the same writing style, tone, and quality.

SECTION HEADING: {section_heading}

LAST PORTION OF CONTENT (continue from here):
---
{tail_context}
---

Continue the content now:"""

        try:
            result = llm.chat_with_metadata(
                system_prompt=system_prompt,
                user_prompt=continuation_prompt,
                temperature=self.temperature,
                max_tokens=continuation_max_tokens,
            )
            continuation = result["content"]
            if continuation and len(continuation.split()) > 5:
                logger.info(
                    f"[SectionWriter] Continuation for '{section_heading}' — "
                    f"{len(continuation.split())} words generated"
                )
                return continuation
            return ""
        except Exception as e:
            logger.warning(
                f"[SectionWriter] Continuation for '{section_heading}' failed: {e}"
            )
            return ""

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _format_section_structure(self, section: dict, children: list[dict]) -> str:
        """Format the H2 section and its children for the prompt."""
        lines = []
        heading = section.get("heading", "")
        lines.append(f"## {heading}")
        if section.get("purpose"):
            lines.append(f"  Purpose: {section['purpose']}")
        if section.get("content_type"):
            lines.append(f"  Content type: {section['content_type']}")
        if section.get("eeat_signal"):
            lines.append(f"  EEAT signal: {section['eeat_signal']}")
        if section.get("key_points"):
            for p in section["key_points"]:
                lines.append(f"  - {p}")
        if section.get("word_count_target"):
            lines.append(f"  Target: ~{section['word_count_target']} words")
        lines.append("")

        for child in children:
            level = child.get("level", "h3")
            prefix = {"h3": "### ", "h4": "#### "}.get(level, "### ")
            lines.append(f"{prefix}{child.get('heading', '')}")
            if child.get("purpose"):
                lines.append(f"  Purpose: {child['purpose']}")
            if child.get("content_type"):
                lines.append(f"  Content type: {child['content_type']}")
            if child.get("key_points"):
                for p in child["key_points"]:
                    lines.append(f"  - {p}")
            if child.get("word_count_target"):
                lines.append(f"  Target: ~{child['word_count_target']} words")
            lines.append("")

        return "\n".join(lines)

    def _build_context_blocks(
        self,
        knowledge_context: str,
        product_context: str,
        internal_links: list[dict] | None,
    ) -> str:
        """Build context injection blocks for the section prompt."""
        blocks = []

        if knowledge_context:
            blocks.append(
                f"═══ INTERNAL KNOWLEDGE (use to make content unique & authoritative) ═══\n"
                f"{knowledge_context}\n"
                f"═══ END KNOWLEDGE ═══"
            )

        if product_context:
            blocks.append(
                f"═══ PRODUCT DATA (integrate naturally when contextually relevant) ═══\n"
                f"{product_context}\n"
                f"═══ END PRODUCT DATA ═══"
            )

        if internal_links:
            link_text = "\n".join(
                f"- {link['anchor_suggestion']} → {link['url']}"
                for link in internal_links
            )
            blocks.append(
                f"═══ INTERNAL LINKS (MANDATORY — you MUST use these links) ═══\n"
                f"{link_text}\n\n"
                f"LINK RULES:\n"
                f"- You MUST insert ALL links listed above into the section content.\n"
                f"- Use [anchor text](URL) Markdown format.\n"
                f"- Place each link naturally within a relevant sentence — do NOT just list them.\n"
                f"- The anchor text should be the product/page name (bold it: **[anchor](URL)**).\n"
                f"- If a heading mentions a product name that matches a link above, that link MUST appear in that sub-section.\n"
                f"═══ END LINKS ═══"
            )

        return "\n\n".join(blocks)

    def _build_eeat_instruction(self, section: dict) -> str:
        """Build section-specific EEAT enforcement instructions."""
        eeat_signal = section.get("eeat_signal", "expertise")
        content_type = section.get("content_type", "text")

        instructions = ["EEAT REQUIREMENTS FOR THIS SECTION:"]

        # Base requirements for every section
        instructions.append("✓ Explain WHY this topic matters to the reader")
        instructions.append("✓ Explain HOW it works (technical mechanism or process)")
        instructions.append("✓ Provide at least 1 PRACTICAL SCENARIO or real-world example")

        # EEAT signal-specific instructions
        if eeat_signal == "experience":
            instructions.append("✓ Include a first-hand experience perspective or strategic case study")
            instructions.append("✓ Describe real scenarios with QUALITATIVE outcomes (not fabricated numbers)")
            instructions.append("✓ Show practical knowledge that only comes from hands-on work")
            instructions.append("✓ CASE STUDY FORMAT: Bối cảnh → Quyết định chiến lược → Mô hình triển khai → Tại sao hiệu quả → Bài học. KHÔNG bịa số liệu cụ thể.")
        elif eeat_signal == "expertise":
            instructions.append("✓ Provide technical depth — explain the underlying mechanism")
            instructions.append("✓ Use precise terminology with clear explanations")
            instructions.append("✓ Include methodology or step-by-step technical reasoning")
        elif eeat_signal == "authority":
            instructions.append("✓ Include data, statistics, or structured comparisons")
            instructions.append("✓ Reference industry standards or best practices")
            instructions.append("✓ Compare alternatives with clear pros/cons")
        elif eeat_signal == "trust":
            instructions.append("✓ Be transparent about limitations and trade-offs")
            instructions.append("✓ Provide honest, balanced assessments")
            instructions.append("✓ Include clear, actionable recommendations")

        # Content type-specific instructions
        if content_type == "comparison_table":
            instructions.append("✓ Include a detailed comparison table in Markdown format")
            instructions.append("✓ Explain the criteria used for comparison")
        elif content_type == "case_study":
            instructions.append("✓ Structure as: Context → Strategic Move → Execution Model → Why It Worked → Strategic Takeaway")
            instructions.append("✓ Focus on strategic decisions, content positioning, collaboration model, posting logic")
            instructions.append("✓ KHÔNG bịa số liệu cụ thể (follower, view, %, doanh thu). Dùng ngôn ngữ định tính.")
            instructions.append("✓ KHÔNG dùng tên brand thật trừ khi có nguồn công khai")
            instructions.append("✓ Mục đích là phân tích chiến lược, KHÔNG phải báo cáo số liệu")
        elif content_type == "step_by_step":
            instructions.append("✓ Number each step clearly")
            instructions.append("✓ Include tips or warnings for each step")
        elif content_type == "technical_breakdown":
            instructions.append("✓ Explain the technical architecture or mechanism")
            instructions.append("✓ Use diagrams (described in text) if helpful")
        elif content_type == "product_solution":
            instructions.append("✓ Position the product as a SOLUTION to the problem discussed")
            instructions.append("✓ Use specific product data from the provided context")
            instructions.append("✓ Keep it natural — NOT promotional language")

        return "\n".join(instructions)

    @staticmethod
    def _lang_instruction(language: str) -> str:
        """Get language instruction text."""
        lang_map = {
            "vi": "Viết toàn bộ bằng tiếng Việt.",
            "en": "Write entirely in English.",
            "en-us": "Write entirely in American English.",
            "en-gb": "Write entirely in British English.",
            "en-au": "Write entirely in Australian English.",
            "fr": "Rédigez entièrement en français.",
            "de": "Schreiben Sie vollständig auf Deutsch.",
            "es": "Escriba completamente en español.",
            "ja": "すべて日本語で書いてください。",
            "ko": "전체를 한국어로 작성하세요。",
            "zh": "全部用简体中文写。",
            "zh-tw": "全部用繁體中文寫。",
            "th": "เขียนทั้งหมดเป็นภาษาไทย",
            "pt": "Escreva inteiramente em português brasileiro.",
            "it": "Scrivi interamente in italiano.",
        }
        return lang_map.get(language, f"Write in '{language}' language.")
