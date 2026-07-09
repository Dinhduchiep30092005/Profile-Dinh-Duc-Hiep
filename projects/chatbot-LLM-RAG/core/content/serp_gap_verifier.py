"""
SERP Gap Verifier — Compares generated article against SERP top results coverage.

After an article is generated, this module:
    1. Extracts H2-level topics from the article
    2. Compares against top SERP H2 headings
    3. Detects missing subtopics the article should cover
    4. Generates content patches for critical gaps
    5. Verifies search expectation coverage

This ensures articles don't just have good content — they cover what
searchers actually expect to find based on SERP analysis.
"""

import logging
import re
from typing import Optional

from core.llm.llm_client import llm
from core.content.section_writer import _build_language_enforcement

logger = logging.getLogger(__name__)


SERP_GAP_SYSTEM = """You are an SEO content strategist. Analyze the article against SERP competitor coverage.

Your task: identify CRITICAL missing subtopics that top SERP results cover but this article does not.

RULES:
1. Only flag genuinely MISSING topics — not reformulations of existing content
2. A topic is "covered" if the article discusses it under any heading, even with different wording
3. Prioritize gaps that affect search intent satisfaction
4. Ignore decorative differences (e.g., different section ordering is fine)
5. Rate each gap as "critical" (must add), "recommended" (should add), or "minor" (nice to have)

Return JSON:
{
    "coverage_score": float (0-100, percentage of SERP topics covered),
    "gaps": [
        {
            "topic": "The missing topic/subtopic",
            "serp_evidence": "Which SERP results cover this and how",
            "priority": "critical|recommended|minor",
            "suggested_heading": "Suggested H2 or H3 heading for this gap",
            "insertion_point": "After which existing heading this should be inserted",
            "word_budget": int (suggested word count for this gap section)
        }
    ],
    "redundant_sections": [
        {
            "heading": "Heading that overlaps significantly with another",
            "overlaps_with": "The other heading it overlaps with",
            "suggestion": "How to differentiate or merge"
        }
    ],
    "assessment": "Brief overall assessment of coverage quality"
}"""

GAP_FILLER_SYSTEM = """You are an expert content writer. Write a focused section to fill a content gap.

RULES:
1. Write ONLY the content for the specified gap — keep it focused and substantive
2. Match the tone and depth of the surrounding article
3. Include specific data, examples, or practical scenarios
4. Use Markdown heading as specified
5. Every paragraph must add concrete value — NO filler
6. Include at least one specific number, metric, or concrete example
7. Address both benefits AND limitations/risks of the topic
"""


class SERPGapVerifier:
    """Verifies article coverage against SERP expectations and fills critical gaps."""

    # Minimum coverage score to pass without gap filling
    COVERAGE_THRESHOLD = 75.0

    # Maximum gaps to fill in one pass
    MAX_GAPS_TO_FILL = 3

    def __init__(self):
        pass

    def verify_coverage(
        self,
        article_text: str,
        keyword: str,
        serp_analysis: dict | None = None,
        heading_structure: dict | None = None,
    ) -> dict:
        """Verify article covers SERP-expected topics.

        Args:
            article_text: The generated article in Markdown
            keyword: Target keyword
            serp_analysis: SERP analysis data (with structure_map, common_h2, etc.)
            heading_structure: The approved heading structure

        Returns:
            dict with coverage_score, gaps, redundancies, and assessment
        """
        if not serp_analysis:
            logger.info("[SERPGap] No SERP analysis provided — skipping verification")
            return {
                "coverage_score": 100,
                "gaps": [],
                "redundant_sections": [],
                "assessment": "No SERP data available for comparison",
                "skipped": True,
            }

        # Extract article headings
        article_headings = self._extract_article_headings(article_text)

        # Extract SERP common H2s
        structure_map = serp_analysis.get("structure_map", {})
        common_h2 = structure_map.get("common_h2", [])
        people_also_ask = serp_analysis.get("people_also_ask", [])
        content_features = structure_map.get("content_features", {})

        if not common_h2:
            logger.info("[SERPGap] No SERP H2 data available — skipping")
            return {
                "coverage_score": 100,
                "gaps": [],
                "redundant_sections": [],
                "assessment": "No SERP heading data for comparison",
                "skipped": True,
            }

        # Build comparison prompt
        article_headings_text = "\n".join(
            f"- [{h['level']}] {h['heading']}" for h in article_headings
        )
        serp_h2_text = "\n".join(
            f"- {h['text']} (found in {h['frequency']}/{len(serp_analysis.get('results', []))} results)"
            for h in common_h2[:20]
        )
        paa_text = "\n".join(
            f"- {q}" for q in (people_also_ask[:10] if isinstance(people_also_ask, list) else [])
        )

        # Brief article content summary (first 100 words of each section)
        article_sections = self._extract_section_summaries(article_text)
        sections_text = "\n".join(
            f"[{s['heading']}]: {s['summary']}" for s in article_sections[:15]
        )

        prompt = f"""KEYWORD: "{keyword}"

═══ ARTICLE HEADINGS ═══
{article_headings_text}

═══ ARTICLE SECTION SUMMARIES ═══
{sections_text}

═══ TOP SERP COMMON H2 HEADINGS ═══
{serp_h2_text}

═══ PEOPLE ALSO ASK ═══
{paa_text if paa_text else "None available"}

═══ SERP CONTENT FEATURES ═══
- {content_features.get('pct_with_tables', 0):.0f}% of top results have tables
- {content_features.get('pct_with_lists', 0):.0f}% have structured lists
- {content_features.get('pct_with_faq', 0):.0f}% have FAQ sections
- {content_features.get('pct_with_images', 0):.0f}% have multiple images

Analyze the article's coverage against SERP expectations.
Focus on CRITICAL gaps that would hurt rankings — not minor stylistic differences."""

        try:
            result = llm.chat_json(
                SERP_GAP_SYSTEM, prompt,
                temperature=0.3, max_tokens=2048,
            )

            coverage = result.get("coverage_score", 0)
            gaps = result.get("gaps", [])
            critical_gaps = [g for g in gaps if g.get("priority") == "critical"]
            recommended_gaps = [g for g in gaps if g.get("priority") == "recommended"]

            logger.info(
                f"[SERPGap] Coverage: {coverage:.0f}% | "
                f"Gaps: {len(critical_gaps)} critical, {len(recommended_gaps)} recommended, "
                f"{len(gaps) - len(critical_gaps) - len(recommended_gaps)} minor"
            )

            return result

        except Exception as e:
            logger.warning(f"[SERPGap] Verification failed: {e}")
            return {
                "coverage_score": 100,
                "gaps": [],
                "redundant_sections": [],
                "assessment": f"Verification error: {e}",
                "skipped": True,
            }

    def fill_gaps(
        self,
        article_text: str,
        gaps: list[dict],
        keyword: str,
        language: str = "vi",
        knowledge_context: str = "",
    ) -> tuple[str, int]:
        """Generate content for critical gaps and insert into article.

        Args:
            article_text: Current article text
            gaps: List of gap dicts from verify_coverage()
            keyword: Target keyword
            language: Content language
            knowledge_context: Optional RAG context

        Returns:
            (updated_article_text, number_of_gaps_filled)
        """
        # Only fill critical and recommended gaps
        fillable = [
            g for g in gaps
            if g.get("priority") in ("critical", "recommended")
        ][:self.MAX_GAPS_TO_FILL]

        if not fillable:
            return article_text, 0

        filled_count = 0
        for gap in fillable:
            topic = gap.get("topic", "")
            heading = gap.get("suggested_heading", topic)
            insertion_point = gap.get("insertion_point", "")
            word_budget = gap.get("word_budget", 200)

            logger.info(f"[SERPGap] Filling gap: '{heading}' ({gap.get('priority')}, ~{word_budget}w)")

            # Generate gap content
            gap_content = self._generate_gap_content(
                keyword=keyword,
                topic=topic,
                heading=heading,
                word_budget=word_budget,
                language=language,
                knowledge_context=knowledge_context,
            )

            if not gap_content:
                continue

            # Insert into article at the specified location
            article_text = self._insert_gap_content(
                article_text, gap_content, insertion_point
            )
            filled_count += 1

        return article_text, filled_count

    def _generate_gap_content(
        self,
        keyword: str,
        topic: str,
        heading: str,
        word_budget: int,
        language: str,
        knowledge_context: str,
    ) -> str:
        """Generate content for a single gap."""
        lang_map = {
            "vi": "Viết bằng tiếng Việt.",
            "en": "Write in English.",
        }
        lang_instruction = lang_map.get(language, f"Write in {language}.")

        prompt = f"""KEYWORD: "{keyword}"
GAP TOPIC: {topic}
HEADING: ## {heading}
LANGUAGE: {lang_instruction}
WORD BUDGET: ~{word_budget} words

Write content for this section. Include:
1. Context framing (why this matters)
2. Core explanation with specific details
3. A practical example with concrete numbers
4. A brief takeaway
"""
        if knowledge_context:
            prompt += f"\nAVAILABLE KNOWLEDGE:\n{knowledge_context[:1500]}\n"

        prompt += f"\nWrite the section starting with ## {heading}. Target {word_budget} words."

        try:
            system = _build_language_enforcement(language) + GAP_FILLER_SYSTEM
            content = llm.chat(
                system, prompt,
                temperature=0.6, max_tokens=min(2000, word_budget * 3),
            )
            return content.strip()
        except Exception as e:
            logger.warning(f"[SERPGap] Gap content generation failed for '{topic}': {e}")
            return ""

    def _insert_gap_content(
        self, article_text: str, gap_content: str, insertion_point: str,
    ) -> str:
        """Insert gap content after the specified heading in the article."""
        if not insertion_point:
            # Insert before conclusion/FAQ if no specific point
            # Try to find conclusion heading
            conclusion_patterns = [
                r'(^## .*(?:kết luận|tổng kết|lời kết|conclusion|summary))',
                r'(^## .*(?:faq|câu hỏi|thường gặp))',
            ]
            for pattern in conclusion_patterns:
                match = re.search(pattern, article_text, re.MULTILINE | re.IGNORECASE)
                if match:
                    pos = match.start()
                    return article_text[:pos] + gap_content + "\n\n" + article_text[pos:]

            # Fallback: append before the last section
            return article_text + "\n\n" + gap_content

        # Find the insertion point heading and its section end
        lines = article_text.split('\n')
        insertion_normalized = self._normalize(insertion_point)
        insert_after_line = -1

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('## ') or stripped.startswith('### '):
                heading_text = re.sub(r'^#{2,4}\s+', '', stripped)
                if self._normalize(heading_text) == insertion_normalized:
                    # Find the end of this section (next H2)
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith('## '):
                            insert_after_line = j - 1
                            break
                    else:
                        insert_after_line = len(lines) - 1
                    break

        if insert_after_line >= 0:
            lines.insert(insert_after_line + 1, "\n" + gap_content + "\n")
            return '\n'.join(lines)

        # Fallback: append before conclusion
        return article_text + "\n\n" + gap_content

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _extract_article_headings(article_text: str) -> list[dict]:
        """Extract all headings from article text."""
        headings = []
        for line in article_text.split('\n'):
            stripped = line.strip()
            match = re.match(r'^(#{2,4})\s+(.+)$', stripped)
            if match:
                level = f"h{len(match.group(1))}"
                headings.append({
                    "level": level,
                    "heading": match.group(2).strip(),
                })
        return headings

    @staticmethod
    def _extract_section_summaries(article_text: str) -> list[dict]:
        """Extract first ~80 words of each H2 section for comparison."""
        sections = []
        current_heading = ""
        current_content = []

        for line in article_text.split('\n'):
            stripped = line.strip()
            if stripped.startswith('## '):
                if current_heading:
                    text = ' '.join(current_content)
                    words = text.split()[:80]
                    sections.append({
                        "heading": current_heading,
                        "summary": ' '.join(words),
                    })
                current_heading = re.sub(r'^##\s+', '', stripped)
                current_content = []
            elif stripped and not stripped.startswith('#'):
                current_content.append(stripped)

        if current_heading:
            text = ' '.join(current_content)
            words = text.split()[:80]
            sections.append({
                "heading": current_heading,
                "summary": ' '.join(words),
            })

        return sections

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        t = text.lower().strip()
        t = re.sub(r'[^\w\s]', '', t)
        t = re.sub(r'\s+', ' ', t)
        return t
