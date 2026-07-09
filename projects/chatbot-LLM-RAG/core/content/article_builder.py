"""
Article Builder — Enhanced 4-Layer Professional Article Generation Orchestrator.

v2 Pipeline flow (upgraded):
    Heading Structure (approved)
    ↓
    Phase 1: Detect Search Intent
    ↓
    Phase 2: Extract Knowledge + Product Context (RAG)
    ↓
    Phase 3: Parse & partition heading structure
    ↓
    Phase 4: Pre-calc internal links per H2
    ↓
    Phase 5: For each H2 (with DepthController):
        → SectionWriter (4-layer structure per H2)
        → DepthController (intent-based depth + tone variety)
        → EEATEnforcer (section-level with data-backed checks)
        → SemanticLinkInjector (sentence-level post-gen injection)
    ↓
    Phase 6: Assemble body
    ↓
    Phase 7: Global EEAT Analysis (BEFORE conclusion)
    ↓
    Phase 8: Inject authority signals if needed
    ↓
    Phase 9: Write intro (intent-aligned, data-first)
    ↓
    Phase 10: Write conclusion (authority-based, using EEAT analysis)
    ↓
    Phase 11: Write contextual FAQ
    ↓
    Phase 12: Final assembly
    ↓
    Phase 13: SERP Gap Verification
    ↓
    Phase 14: Fill critical gaps if needed
    ↓
    Phase 15: Validate hierarchy + link cleanup + heading enforcement
    ↓
    Phase 16: Duplicate scan
    ↓
    Phase 17: Convert to clean WordPress HTML
"""

import logging
import re
from typing import Optional

from config import settings
from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.knowledge.retrieval import (
    build_rag_context,
    build_product_solution_context,
    has_knowledge_base,
)
from core.content.section_writer import SectionWriter
from core.content.eeat_enforcer import EEATEnforcer
from core.content.internal_link_injector import InternalLinkInjector
from core.content.depth_controller import DepthController
from core.content.serp_gap_verifier import SERPGapVerifier
from core.content.truncation_detector import validate_sections_bulk
logger = logging.getLogger(__name__)


class ArticleBuilder:
    """Enhanced 4-Layer professional article generation orchestrator."""

    # Target word count range
    TARGET_MIN_WORDS = 2200
    TARGET_MAX_WORDS = 2500

    # Reserved word budgets for non-section parts
    INTRO_BUDGET = 150        # ~100-150 words (shorter, data-first)
    CONCLUSION_BUDGET = 200   # ~150-250 words
    FAQ_BUDGET = 300          # ~250-350 words

    # Keywords to detect conclusion/FAQ sections in heading structure
    _CONCLUSION_KEYWORDS = {
        "kết luận", "tổng kết", "lời kết", "conclusion", "summary",
        "tóm tắt", "nhận định cuối", "final thoughts", "wrap up",
    }
    _FAQ_KEYWORDS = {
        "faq", "câu hỏi", "thường gặp", "frequently asked",
        "hỏi đáp", "q&a", "questions",
    }

    def __init__(
        self,
        temperature: float = 0.65,
        eeat_strict: bool = True,
    ):
        """
        Args:
            temperature: LLM temperature for content (0.6-0.7 for professional tone)
            eeat_strict: If True, sections below EEAT threshold get enhanced
        """
        self.section_writer = SectionWriter(temperature=temperature)
        self.eeat_enforcer = EEATEnforcer(strict_mode=eeat_strict)
        self.link_injector = InternalLinkInjector()
        self.depth_controller = DepthController()
        self.gap_verifier = SERPGapVerifier()
        self.temperature = temperature

    def build(
        self,
        keyword: str,
        title: str,
        heading_structure: dict,
        language: str = "vi",
        serp_analysis: dict | None = None,
    ) -> dict:
        """
        Build a complete professional article using the enhanced 4-layer pipeline.

        Returns:
            dict with article_text, article_html, internal_links,
                  word_count, eeat_scores, quality_report, search_intent,
                  serp_gap_report, global_eeat_analysis
        """
        logger.info(f"[ArticleBuilder] ══════════════════════════════════")
        logger.info(f"[ArticleBuilder] START: '{title}'")
        logger.info(f"[ArticleBuilder] ══════════════════════════════════")

        self.link_injector.reset()

        # ── Phase 1: Detect Search Intent ──
        search_intent = self._detect_intent(keyword, heading_structure)
        logger.info(f"[ArticleBuilder] Phase 1 — Intent: {search_intent['intent_type']} "
                     f"(confidence: {search_intent['confidence']:.0%})")

        # ── Phase 2: Extract Knowledge + Product Context (RAG) ──
        knowledge_context, product_context = self._inject_knowledge(
            keyword, title, intent_type=search_intent["intent_type"]
        )
        logger.info("[ArticleBuilder] Phase 2 — Knowledge + product context extracted")

        # ── Phase 3: Parse & partition heading structure ──
        sections = heading_structure.get("sections", [])
        faq_items = heading_structure.get("faq", [])

        all_h2_groups = self._group_sections_by_h2(sections)
        h2_groups = []
        conclusion_heading_original = ""  # preserve approved heading

        for h2_sec, children in all_h2_groups:
            heading_lower = h2_sec.get("heading", "").lower().strip()
            content_type = h2_sec.get("content_type", "").lower()

            if content_type in ("summary", "conclusion") or any(
                kw in heading_lower for kw in self._CONCLUSION_KEYWORDS
            ):
                conclusion_heading_original = h2_sec.get("heading", "")
                logger.info(f"[ArticleBuilder]   ↳ Conclusion H2 separated: '{conclusion_heading_original}'")
                continue

            if content_type == "faq" or any(
                kw in heading_lower for kw in self._FAQ_KEYWORDS
            ):
                logger.info(f"[ArticleBuilder]   ↳ FAQ H2 separated: '{h2_sec.get('heading')}'")
                continue

            h2_groups.append((h2_sec, children))

        total_h2 = len(h2_groups)

        # Word budget
        reserved = self.INTRO_BUDGET + self.CONCLUSION_BUDGET
        if faq_items:
            reserved += self.FAQ_BUDGET
        body_budget = self.TARGET_MAX_WORDS - reserved
        per_section_budget = max(150, body_budget // max(total_h2, 1))

        logger.info(
            f"[ArticleBuilder] Phase 3 — {total_h2} body sections | "
            f"Budget: ~{per_section_budget} w/section | "
            f"Target total: {self.TARGET_MIN_WORDS}-{self.TARGET_MAX_WORDS}"
        )

        # ── Phase 3.5: EARLY Validation — catch structural issues before
        #    wasting LLM tokens on content generation ──
        early_issues = self._early_validate_structure(heading_structure, h2_groups)
        if early_issues:
            logger.warning(
                f"[ArticleBuilder] Phase 3.5 — {len(early_issues)} structural issue(s) detected early:"
            )
            for issue in early_issues:
                logger.warning(f"[ArticleBuilder]   ⚠ {issue}")
        else:
            logger.info("[ArticleBuilder] Phase 3.5 — Structure pre-validation OK")

        # ── Phase 4: Pre-calc internal links per H2 (including children context) ──
        h2_links_map = {}
        for h2_section, children in h2_groups:
            heading = h2_section.get("heading", "")
            # Build richer context from H2 + all H3/H4 children
            children_headings = [c.get("heading", "") for c in children]
            children_purposes = [c.get("purpose", "") for c in children]
            children_key_points = []
            for c in children:
                children_key_points.extend(c.get("key_points", []))

            h2_links_map[heading] = self.link_injector.pre_calculate_links(
                section_heading=heading,
                section_purpose=h2_section.get("purpose", ""),
                section_key_points=h2_section.get("key_points", []),
                children_headings=children_headings,
                children_purposes=children_purposes,
                article_keyword=keyword,
            )
        logger.info("[ArticleBuilder] Phase 4 — Internal links pre-calculated")

        # ── Phase 5: For each H2 → generate (4-layer) + EEAT enforce + semantic link inject ──
        generated_sections = []
        eeat_scores = []
        previous_summary = ""

        for idx, (h2_section, children) in enumerate(h2_groups):
            heading = h2_section.get("heading", "")
            logger.info(f"[ArticleBuilder] ── Section {idx + 1}/{total_h2}: '{heading}' ──")

            section_knowledge = self._get_section_knowledge(
                heading, h2_section.get("key_points", []), knowledge_context
            )

            # Determine product context for this section
            section_product = ""
            if h2_section.get("is_product_section"):
                section_product = product_context
            elif product_context:
                section_text = f"{heading} {h2_section.get('purpose', '')}"
                if any(kw in section_text.lower() for kw in [
                    "sản phẩm", "giải pháp", "product", "solution", "tool", "công cụ"
                ]):
                    section_product = product_context

            # Get depth strategy from DepthController (4-layer + tone + anti-patterns)
            depth_strategy = self.depth_controller.get_section_strategy(
                intent_type=search_intent["intent_type"],
                section=h2_section,
                section_index=idx,
                total_sections=total_h2,
                children=children,
                language=language,
            )
            depth_prompt = self.depth_controller.format_for_prompt(depth_strategy)

            # Generate section (Layer 1) — NO links in prompt, links injected post-gen
            section_content = self.section_writer.write_section(
                keyword=keyword,
                title=title,
                section=h2_section,
                children=children,
                section_index=idx,
                total_sections=total_h2,
                knowledge_context=section_knowledge,
                product_context=section_product,
                internal_links=h2_links_map.get(heading, []),
                previous_section_summary=previous_summary,
                language=language,
                word_budget=per_section_budget,
                depth_strategy=depth_prompt,
            )

            # Section EEAT enforce (Layer 2)
            section_content, validation = self.eeat_enforcer.enforce(
                section_content=section_content,
                section_heading=heading,
                knowledge_context=section_knowledge,
                product_context=section_product,
                language=language,
            )

            # Semantic link injection (Layer 3) — sentence-level, post-generation
            pre_calc_links = h2_links_map.get(heading, [])
            if pre_calc_links:
                section_content, link_count = self.link_injector.inject_links_semantic(
                    section_content=section_content,
                    pre_calculated_links=pre_calc_links,
                    section_heading=heading,
                )
                logger.info(f"[ArticleBuilder]   Semantic links: {link_count}/{len(pre_calc_links)} injected")

            # Enforce headings RIGHT AFTER all enhancements
            section_headings = []
            section_headings.append({"level": "h2", "heading": heading})
            for c in children:
                section_headings.append({
                    "level": c.get("level", "h3"),
                    "heading": c.get("heading", ""),
                })
            section_content, section_fixes = self._enforce_original_headings(
                section_content, section_headings
            )
            if section_fixes:
                logger.info(
                    f"[ArticleBuilder]   Heading fix after EEAT: "
                    f"{len(section_fixes)} correction(s) in '{heading}'"
                )

            eeat_scores.append({
                "heading": heading,
                "score": validation.get("overall_score", 0),
                "pass": validation.get("pass", False),
            })
            generated_sections.append(section_content)
            previous_summary = self._extract_summary(section_content, heading)

        logger.info(f"[ArticleBuilder] Phase 5 — {total_h2} sections generated & enforced")

        # ── Phase 5.5: Post-generation validation — detect truncated/broken sections ──
        section_issues = validate_sections_bulk(generated_sections)
        sections_needing_regen = [
            s for s in section_issues if s["needs_regeneration"]
        ]

        if sections_needing_regen:
            logger.warning(
                f"[ArticleBuilder] Phase 5.5 — {len(sections_needing_regen)} section(s) "
                f"need regeneration:"
            )
            for si in sections_needing_regen:
                logger.warning(
                    f"[ArticleBuilder]   ⚠ Section {si['index'] + 1} "
                    f"'{si['heading']}': {'; '.join(si['issues'])}"
                )

            # Regenerate problematic sections (max 3 to avoid infinite loops)
            regen_count = 0
            for si in sections_needing_regen[:3]:
                idx = si["index"]
                if idx >= len(h2_groups):
                    continue

                h2_section, children = h2_groups[idx]
                heading = h2_section.get("heading", "")
                logger.info(
                    f"[ArticleBuilder]   Regenerating section {idx + 1}: '{heading}'"
                )

                # Regenerate with higher word budget (+30%)
                regen_budget = int(per_section_budget * 1.3)

                section_knowledge = self._get_section_knowledge(
                    heading, h2_section.get("key_points", []), knowledge_context
                )
                section_product = ""
                if h2_section.get("is_product_section"):
                    section_product = product_context
                elif product_context:
                    section_text = f"{heading} {h2_section.get('purpose', '')}"
                    if any(kw in section_text.lower() for kw in [
                        "sản phẩm", "giải pháp", "product", "solution", "tool", "công cụ"
                    ]):
                        section_product = product_context

                depth_strategy = self.depth_controller.get_section_strategy(
                    intent_type=search_intent["intent_type"],
                    section=h2_section,
                    section_index=idx,
                    total_sections=total_h2,
                    children=children,
                    language=language,
                )
                depth_prompt = self.depth_controller.format_for_prompt(depth_strategy)

                prev_summary = ""
                if idx > 0:
                    prev_heading = h2_groups[idx - 1][0].get("heading", "")
                    prev_summary = self._extract_summary(
                        generated_sections[idx - 1], prev_heading
                    )

                try:
                    new_content = self.section_writer.write_section(
                        keyword=keyword,
                        title=title,
                        section=h2_section,
                        children=children,
                        section_index=idx,
                        total_sections=total_h2,
                        knowledge_context=section_knowledge,
                        product_context=section_product,
                        internal_links=h2_links_map.get(heading, []),
                        previous_section_summary=prev_summary,
                        language=language,
                        word_budget=regen_budget,
                        depth_strategy=depth_prompt,
                    )

                    # Re-run EEAT enforce
                    new_content, _ = self.eeat_enforcer.enforce(
                        section_content=new_content,
                        section_heading=heading,
                        knowledge_context=section_knowledge,
                        product_context=section_product,
                        language=language,
                    )

                    # Re-inject links
                    pre_calc_links = h2_links_map.get(heading, [])
                    if pre_calc_links:
                        new_content, _ = self.link_injector.inject_links_semantic(
                            section_content=new_content,
                            pre_calculated_links=pre_calc_links,
                            section_heading=heading,
                        )

                    # Re-enforce headings
                    section_headings = [{"level": "h2", "heading": heading}]
                    for c in children:
                        section_headings.append({
                            "level": c.get("level", "h3"),
                            "heading": c.get("heading", ""),
                        })
                    new_content, _ = self._enforce_original_headings(
                        new_content, section_headings
                    )

                    generated_sections[idx] = new_content
                    regen_count += 1
                    logger.info(
                        f"[ArticleBuilder]   ✓ Section '{heading}' regenerated "
                        f"({len(new_content.split())} words)"
                    )
                except Exception as e:
                    logger.error(
                        f"[ArticleBuilder]   ✗ Regeneration failed for '{heading}': {e}"
                    )

            logger.info(
                f"[ArticleBuilder] Phase 5.5 — {regen_count}/{len(sections_needing_regen)} "
                f"section(s) regenerated successfully"
            )
        else:
            # Log any minor (non-regen) issues
            minor_issues = [s for s in section_issues if s["issues"]]
            if minor_issues:
                logger.info(
                    f"[ArticleBuilder] Phase 5.5 — {len(minor_issues)} section(s) "
                    f"with minor issues (no regeneration needed)"
                )
                for si in minor_issues:
                    logger.info(
                        f"[ArticleBuilder]   ℹ Section '{si['heading']}': "
                        f"{'; '.join(si['issues'])}"
                    )
            else:
                logger.info(
                    "[ArticleBuilder] Phase 5.5 — All sections validated OK"
                )

        # ── Phase 6: Assemble body sections ──
        body_text = "\n\n".join(s.strip() for s in generated_sections if s.strip())
        logger.info(f"[ArticleBuilder] Phase 6 — Body assembled ({len(body_text.split())}w)")

        # ── Phase 7: Global EEAT Analysis BEFORE conclusion ──
        global_eeat_analysis = self.eeat_enforcer.analyze_global_eeat(body_text, keyword)
        global_eeat_scores = self._global_eeat_check(eeat_scores)
        logger.info(
            f"[ArticleBuilder] Phase 7 — Global EEAT: {global_eeat_scores['avg_score']:.1f}/10 | "
            f"Data density: {global_eeat_analysis['data_density_pct']:.0f}% | "
            f"Counter-args: {global_eeat_analysis['counter_argument_count']}"
        )

        # ── Phase 8: Inject authority signals if needed ──
        if global_eeat_analysis.get("needs_improvement"):
            body_text = self.eeat_enforcer.inject_authority_signals(
                body_text, global_eeat_analysis,
                knowledge_context=knowledge_context,
                language=language,
            )
            # Re-analyze after injection
            global_eeat_analysis = self.eeat_enforcer.analyze_global_eeat(body_text, keyword)
            logger.info(
                f"[ArticleBuilder] Phase 8 — Authority signals injected | "
                f"New data density: {global_eeat_analysis['data_density_pct']:.0f}%"
            )
        else:
            logger.info("[ArticleBuilder] Phase 8 — Authority OK, no injection needed")

        # ── Phase 9: Write intent-aligned intro (data-first, no product mention) ──
        h2_headings = [h2.get("heading", "") for h2, _ in h2_groups]
        intro_text = self.section_writer.write_intro(
            keyword=keyword,
            title=title,
            section_summaries=h2_headings,
            knowledge_context=product_context or knowledge_context,
            language=language,
            search_intent=search_intent,
        )
        logger.info(f"[ArticleBuilder] Phase 9 — Intro ({len(intro_text.split())}w, "
                     f"intent: {search_intent['intent_type']})")

        # ── Phase 10: Write conclusion AFTER global EEAT (authority-based) ──
        key_takeaways = self._extract_key_takeaways(generated_sections, h2_headings)
        brand_info = {
            "product_name": settings.brand.product_name,
            "product_description": settings.brand.product_description,
            "use_cases": settings.brand.use_cases,
            "competitive_advantages": settings.brand.competitive_advantages,
            "target_audience": settings.brand.target_audience,
        }
        conclusion_text = self.section_writer.write_conclusion(
            keyword=keyword,
            title=title,
            key_takeaways=key_takeaways,
            product_context=product_context,
            language=language,
            brand_info=brand_info if brand_info.get("product_name") else None,
            search_intent=search_intent,
            conclusion_heading=conclusion_heading_original,
            authority_context=global_eeat_analysis.get("authority_summary", ""),
        )
        logger.info(f"[ArticleBuilder] Phase 10 — Conclusion + CTA ({len(conclusion_text.split())}w)")

        # ── Phase 11: Write contextual FAQ ──
        faq_text = ""
        if faq_items:
            article_context_for_faq = "\n".join(generated_sections[:3])
            faq_text = self.section_writer.write_faq(
                keyword=keyword,
                title=title,
                faq_items=faq_items,
                article_context=article_context_for_faq,
                language=language,
            )
            logger.info(f"[ArticleBuilder] Phase 11 — FAQ ({len(faq_text.split())}w)")

        # ── Phase 12: Final assembly ──
        article_text = self._assemble_article(
            intro_text, generated_sections, conclusion_text, faq_text
        )
        logger.info(f"[ArticleBuilder] Phase 12 — Assembled ({len(article_text.split())}w)")

        # ── Phase 13: SERP Gap Verification ──
        serp_gap_report = self.gap_verifier.verify_coverage(
            article_text=article_text,
            keyword=keyword,
            serp_analysis=serp_analysis,
            heading_structure=heading_structure,
        )
        logger.info(
            f"[ArticleBuilder] Phase 13 — SERP Coverage: "
            f"{serp_gap_report.get('coverage_score', 0):.0f}% | "
            f"Gaps: {len(serp_gap_report.get('gaps', []))}"
        )

        # ── Phase 14: Fill critical gaps if needed ──
        critical_gaps = [
            g for g in serp_gap_report.get("gaps", [])
            if g.get("priority") in ("critical", "recommended")
        ]
        if critical_gaps and serp_gap_report.get("coverage_score", 100) < 80:
            article_text, gaps_filled = self.gap_verifier.fill_gaps(
                article_text=article_text,
                gaps=critical_gaps,
                keyword=keyword,
                language=language,
                knowledge_context=knowledge_context,
            )
            logger.info(f"[ArticleBuilder] Phase 14 — {gaps_filled} gap(s) filled")
        else:
            logger.info("[ArticleBuilder] Phase 14 — SERP coverage adequate, no gaps to fill")

        # ── Phase 15: Validate hierarchy + link cleanup + heading enforcement ──
        quality_report = self.eeat_enforcer.validate_full_article(article_text)
        link_report = self.link_injector.validate_links_in_content(article_text)
        if link_report["invalid_count"] > 0:
            article_text = self.link_injector.remove_invalid_links(article_text)
        logger.info(
            f"[ArticleBuilder] Phase 15 — Hierarchy: "
            f"{'OK' if quality_report['pass'] else str(len(quality_report['issues'])) + ' issues'} | "
            f"Links: {link_report['valid_count']} valid, {link_report['invalid_count']} removed"
        )

        # Phase 15.2: Post-gen link injection (catch missed product mentions)
        article_text, injected_count = self.link_injector.inject_missing_links(article_text, article_keyword=keyword)
        if injected_count > 0:
            link_report = self.link_injector.validate_links_in_content(article_text)
            logger.info(
                f"[ArticleBuilder] Phase 15.2 — {injected_count} missing links injected | "
                f"Total links now: {link_report['valid_count']}"
            )

        # Phase 15.5: Enforce original headings from approved structure
        original_headings = self._extract_original_headings(heading_structure)
        article_text, heading_fixes = self._enforce_original_headings(article_text, original_headings)
        if heading_fixes:
            logger.warning(
                f"[ArticleBuilder] Phase 15.5 — {len(heading_fixes)} heading(s) corrected to match approved structure"
            )
            for fix in heading_fixes:
                logger.info(f"[ArticleBuilder]   '{fix['from']}' → '{fix['to']}'")

        # Phase 15.6: Remove empty headings
        article_text, empty_count = self._remove_empty_headings(article_text)
        if empty_count > 0:
            logger.warning(f"[ArticleBuilder] Phase 15.6 — {empty_count} empty heading(s) removed")

        # ── Phase 16: Duplicate scan ──
        dup_report = self._scan_duplicates(article_text)
        if dup_report["duplicates_found"]:
            article_text = dup_report["cleaned_text"]
            logger.warning(
                f"[ArticleBuilder] Phase 16 — {dup_report['count']} duplicate paragraph(s) removed"
            )
        else:
            logger.info("[ArticleBuilder] Phase 16 — No duplicates found")

        # ── Phase 17: Convert to clean WordPress HTML ──
        article_html = self._convert_to_html(
            article_text, title, link_report.get("valid_links", [])
        )
        logger.info("[ArticleBuilder] Phase 17 — HTML generated")

        # ── Final result ──
        word_count = len(article_text.split())
        result = {
            "article_text": article_text,
            "article_html": article_html,
            "internal_links": link_report.get("valid_links", []),
            "word_count": word_count,
            "search_intent": search_intent,
            "eeat_scores": eeat_scores,
            "global_eeat": global_eeat_scores,
            "global_eeat_analysis": global_eeat_analysis,
            "serp_gap_report": serp_gap_report,
            "quality_report": quality_report,
            "duplicate_report": dup_report,
            "generation_method": "4-layer-pipeline-v2",
        }

        logger.info(f"[ArticleBuilder] ══════════════════════════════════")
        logger.info(f"[ArticleBuilder] COMPLETE: {word_count} words")
        logger.info(f"[ArticleBuilder] EEAT: {global_eeat_scores['avg_score']:.1f}/10 | "
                     f"Data density: {global_eeat_analysis['data_density_pct']:.0f}% | "
                     f"SERP coverage: {serp_gap_report.get('coverage_score', 'N/A')}% | "
                     f"Intent: {search_intent['intent_type']} | "
                     f"Links: {link_report['valid_count']}")
        logger.info(f"[ArticleBuilder] ══════════════════════════════════")

        return result

    # ──────────────────────────────────────────
    # Knowledge Injection
    # ──────────────────────────────────────────

    def _inject_knowledge(
        self, keyword: str, title: str, intent_type: str = "informational"
    ) -> tuple[str, str]:
        """Retrieve relevant knowledge from KB, guided by search intent."""
        knowledge_context = ""
        product_context = ""

        if not has_knowledge_base():
            logger.info("[ArticleBuilder] No knowledge base — skipping injection")
            return knowledge_context, product_context

        try:
            knowledge_context, _ = build_rag_context(
                keyword=keyword,
                title=title,
                intent_type=intent_type,
                max_tokens_approx=4000,
            )
        except Exception as e:
            logger.warning(f"[ArticleBuilder] RAG context failed: {e}")

        # For commercial/transactional intents, always try product context
        try:
            product_context, _ = build_product_solution_context(
                keyword=keyword,
                title=title,
                relevance_threshold=0.25 if intent_type in ("commercial", "transactional") else 0.30,
                max_tokens_approx=2000,
            )
        except Exception as e:
            logger.warning(f"[ArticleBuilder] Product context failed: {e}")

        return knowledge_context, product_context

    # ──────────────────────────────────────────
    # Search Intent Detection
    # ──────────────────────────────────────────

    _INTENT_COMMERCIAL_SIGNALS = {
        "tốt nhất", "top", "so sánh", "review", "đánh giá", "giá",
        "mua", "chi phí", "best", "comparison", "vs", "price",
        "pricing", "cost", "buy", "recommend", "nên dùng", "nên chọn",
    }
    _INTENT_TRANSACTIONAL_SIGNALS = {
        "mua ngay", "đặt hàng", "order", "download", "tải", "đăng ký",
        "sign up", "free trial", "dùng thử", "coupon", "discount",
        "giảm giá", "khuyến mãi",
    }
    _INTENT_NAVIGATIONAL_SIGNALS = {
        "login", "đăng nhập", "trang chủ", "homepage", "contact",
        "liên hệ", "documentation", "hướng dẫn sử dụng",
    }

    def _detect_intent(
        self, keyword: str, heading_structure: dict
    ) -> dict:
        """Detect search intent from keyword + heading analysis.

        Returns:
            dict with intent_type (informational|commercial|transactional|navigational),
                  confidence (0-1), signals (list of matched signals)
        """
        kw_lower = keyword.lower()
        headings_text = " ".join(
            s.get("heading", "").lower()
            for s in heading_structure.get("sections", [])
        )
        combined = f"{kw_lower} {headings_text}"

        scores = {
            "informational": 0.3,  # default bias
            "commercial": 0.0,
            "transactional": 0.0,
            "navigational": 0.0,
        }
        matched_signals = []

        # Question patterns → informational
        if any(q in kw_lower for q in ["là gì", "what is", "how to", "cách", "tại sao", "why", "when"]):
            scores["informational"] += 0.5
            matched_signals.append("question_pattern")

        # Commercial signals
        for signal in self._INTENT_COMMERCIAL_SIGNALS:
            if signal in combined:
                scores["commercial"] += 0.25
                matched_signals.append(f"commercial:{signal}")

        # Transactional signals
        for signal in self._INTENT_TRANSACTIONAL_SIGNALS:
            if signal in combined:
                scores["transactional"] += 0.35
                matched_signals.append(f"transactional:{signal}")

        # Navigational signals
        for signal in self._INTENT_NAVIGATIONAL_SIGNALS:
            if signal in combined:
                scores["navigational"] += 0.4
                matched_signals.append(f"navigational:{signal}")

        # Heading content_type clues
        for sec in heading_structure.get("sections", []):
            ct = sec.get("content_type", "").lower()
            if ct in ("comparison_table", "product_solution"):
                scores["commercial"] += 0.15
            elif ct in ("step_by_step", "tutorial", "how_to"):
                scores["informational"] += 0.1

        # Determine winner
        intent_type = max(scores, key=scores.get)
        max_score = scores[intent_type]
        total_score = sum(scores.values()) or 1
        confidence = min(1.0, max_score / total_score + 0.1)

        return {
            "intent_type": intent_type,
            "confidence": round(confidence, 2),
            "signals": matched_signals[:10],
            "scores": {k: round(v, 2) for k, v in scores.items()},
        }

    # ──────────────────────────────────────────
    # Global EEAT Check
    # ──────────────────────────────────────────

    def _global_eeat_check(self, eeat_scores: list[dict]) -> dict:
        """Compute global EEAT score from per-section scores."""
        if not eeat_scores:
            return {"avg_score": 0, "min_score": 0, "pass": False, "sections_passed": 0, "total": 0}

        scores = [s["score"] for s in eeat_scores]
        avg = sum(scores) / len(scores)
        passed = sum(1 for s in eeat_scores if s.get("pass"))

        return {
            "avg_score": round(avg, 1),
            "min_score": round(min(scores), 1),
            "max_score": round(max(scores), 1),
            "pass": avg >= 7.0 and passed >= len(scores) * 0.7,
            "sections_passed": passed,
            "total": len(scores),
        }

    # ──────────────────────────────────────────
    # Early Structure Validation (Phase 3.5)
    # ──────────────────────────────────────────

    def _early_validate_structure(
        self,
        heading_structure: dict,
        h2_groups: list,
    ) -> list[str]:
        """Pre-validate heading structure BEFORE content generation.

        Catches issues that would waste LLM tokens:
            - No H2 sections found
            - Duplicate H2 headings
            - H3/H4 without parent H2
            - Too many or too few sections

        Returns list of issue strings (empty = all OK).
        """
        issues = []
        sections = heading_structure.get("sections", [])

        # 1. No body sections
        if not h2_groups:
            issues.append("No H2 body sections found — article will be empty")

        # 2. Too few sections
        if 0 < len(h2_groups) < 2:
            issues.append(
                f"Only {len(h2_groups)} H2 section(s) — article may be too thin"
            )

        # 3. Too many sections (>15 H2s is usually over-structured)
        if len(h2_groups) > 15:
            issues.append(
                f"{len(h2_groups)} H2 sections — consider consolidating for readability"
            )

        # 4. Duplicate H2 headings
        h2_headings = [
            h2.get("heading", "").strip().lower()
            for h2, _ in h2_groups
        ]
        seen_headings = set()
        for h in h2_headings:
            if h in seen_headings:
                issues.append(f"Duplicate H2 heading: '{h}'")
            seen_headings.add(h)

        # 5. Check hierarchy — H3/H4 orphans (sections with level > h2 before any h2)
        first_non_meta = None
        for s in sections:
            level = s.get("level", "h2")
            ct = (s.get("content_type") or "").lower()
            if ct in ("summary", "conclusion", "faq"):
                continue
            first_non_meta = s
            break

        if first_non_meta and first_non_meta.get("level", "h2") in ("h3", "h4"):
            issues.append(
                f"First body section is {first_non_meta['level']} "
                f"('{first_non_meta.get('heading', '')}') — expected H2"
            )

        return issues

    # ──────────────────────────────────────────
    # Duplicate Scan
    # ──────────────────────────────────────────

    def _scan_duplicates(self, article_text: str) -> dict:
        """Scan for duplicate paragraphs/sections and remove them.

        Returns:
            dict with duplicates_found (bool), count (int),
                  duplicated_snippets (list), cleaned_text (str)
        """
        paragraphs = [p.strip() for p in article_text.split("\n\n") if p.strip()]
        seen = {}
        duplicates = []
        unique_paragraphs = []

        for para in paragraphs:
            # Normalize for comparison: lowercase, collapse whitespace
            normalized = re.sub(r'\s+', ' ', para.lower().strip())

            # Skip very short paragraphs (headings, etc.) — don't dedupe those
            if len(normalized.split()) < 15:
                unique_paragraphs.append(para)
                continue

            # Check exact duplicate
            if normalized in seen:
                duplicates.append(para[:100] + "...")
                continue

            # Check near-duplicate (>85% word overlap)
            is_near_dup = False
            for seen_norm in seen:
                if len(seen_norm.split()) < 15:
                    continue
                words_a = set(normalized.split())
                words_b = set(seen_norm.split())
                if not words_a or not words_b:
                    continue
                overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                if overlap > 0.85:
                    is_near_dup = True
                    duplicates.append(para[:100] + "...")
                    break

            if not is_near_dup:
                seen[normalized] = True
                unique_paragraphs.append(para)

        cleaned = "\n\n".join(unique_paragraphs)

        return {
            "duplicates_found": len(duplicates) > 0,
            "count": len(duplicates),
            "duplicated_snippets": duplicates[:5],
            "cleaned_text": cleaned,
        }

    # ──────────────────────────────────────────
    # Heading Enforcement
    # ──────────────────────────────────────────

    @staticmethod
    def _extract_original_headings(heading_structure: dict) -> list[dict]:
        """Extract ordered list of original headings from the approved structure.

        Returns:
            List of dicts with 'level' (h2/h3/h4) and 'heading' (exact text)
        """
        headings = []
        for sec in heading_structure.get("sections", []):
            level = sec.get("level", "h2")
            heading = sec.get("heading", "").strip()
            if heading:
                headings.append({"level": level, "heading": heading})
        return headings

    @staticmethod
    def _normalize_heading(text: str) -> str:
        """Normalize heading text for fuzzy matching."""
        # Lowercase, strip punctuation/extra spaces
        t = text.lower().strip()
        t = re.sub(r'[^\w\s]', '', t)  # remove punctuation
        t = re.sub(r'\s+', ' ', t)      # collapse whitespace
        return t

    def _enforce_original_headings(
        self, article_text: str, original_headings: list[dict]
    ) -> tuple[str, list[dict]]:
        """Replace any modified headings in the article with the original approved headings.

        Uses fuzzy matching (normalized word overlap) to detect paraphrased headings
        and replaces them with the exact original text.

        Args:
            article_text: The full article in Markdown
            original_headings: List from _extract_original_headings()

        Returns:
            (corrected_text, list_of_fixes)
        """
        lines = article_text.split('\n')
        heading_prefix_map = {"h2": "## ", "h3": "### ", "h4": "#### "}
        fixes = []

        # Build lookup: for each original heading, store its normalized form and markdown prefix
        original_lookup = []
        for oh in original_headings:
            prefix = heading_prefix_map.get(oh["level"], "## ")
            original_lookup.append({
                "level": oh["level"],
                "heading": oh["heading"],
                "normalized": self._normalize_heading(oh["heading"]),
                "prefix": prefix,
                "matched": False,
            })

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Detect markdown heading
            heading_match = re.match(r'^(#{2,4})\s+(.+)$', stripped)
            if not heading_match:
                continue

            hashes = heading_match.group(1)
            heading_text = heading_match.group(2).strip()
            level_map = {"##": "h2", "###": "h3", "####": "h4"}
            line_level = level_map.get(hashes, "h2")

            # Check if this heading exactly matches any original
            exact_match = False
            for ol in original_lookup:
                if ol["level"] == line_level and ol["heading"] == heading_text and not ol["matched"]:
                    ol["matched"] = True
                    exact_match = True
                    break

            if exact_match:
                continue

            # No exact match — try fuzzy match
            normalized_line = self._normalize_heading(heading_text)
            if not normalized_line:
                continue

            best_match = None
            best_score = 0.0

            # First pass: try same level
            for ol in original_lookup:
                if ol["matched"]:
                    continue
                if ol["level"] != line_level:
                    continue

                words_a = set(normalized_line.split())
                words_b = set(ol["normalized"].split())
                if not words_a or not words_b:
                    continue
                overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

                if overlap > best_score:
                    best_score = overlap
                    best_match = ol

            # Second pass: if no good same-level match, try cross-level
            # (LLM sometimes writes ### instead of ## or vice versa)
            if best_score < 0.4:
                for ol in original_lookup:
                    if ol["matched"]:
                        continue
                    # Allow 1 level difference (h2↔h3 or h3↔h4)
                    level_map = {"h2": 2, "h3": 3, "h4": 4}
                    orig_num = level_map.get(ol["level"], 2)
                    line_num = level_map.get(line_level, 2)
                    if abs(orig_num - line_num) > 1:
                        continue

                    words_a = set(normalized_line.split())
                    words_b = set(ol["normalized"].split())
                    if not words_a or not words_b:
                        continue
                    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

                    if overlap > best_score:
                        best_score = overlap
                        best_match = ol

            # If >40% word overlap, it's likely a paraphrase → replace with original
            if best_match and best_score >= 0.4:
                original_line = f"{best_match['prefix']}{best_match['heading']}"
                if lines[i].strip() != original_line:
                    fixes.append({
                        "from": lines[i].strip(),
                        "to": original_line,
                    })
                    lines[i] = original_line
                best_match["matched"] = True

        return '\n'.join(lines), fixes

    # ──────────────────────────────────────────
    # Section Grouping
    # ──────────────────────────────────────────

    @staticmethod
    def _remove_empty_headings(text: str) -> tuple[str, int]:
        """Remove headings that have no content before the next heading.

        Only removes a heading if it is followed by another heading of the
        SAME OR HIGHER level (same or fewer #). Parent headings followed by
        their children (e.g. ## H2 followed by ### H3) are KEPT.

        Returns:
            (cleaned_text, count_of_removed_headings)
        """
        lines = text.split('\n')
        cleaned = []
        removed = 0
        i = 0

        def _heading_level(line: str) -> int:
            """Return heading level (2 for ##, 3 for ###, etc.) or 0 if not a heading."""
            m = re.match(r'^(#{1,6})\s', line.strip())
            return len(m.group(1)) if m else 0

        while i < len(lines):
            line = lines[i].strip()
            current_level = _heading_level(line)

            if current_level > 0:
                # Look ahead: find the next non-blank line
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1

                if j >= len(lines):
                    # Heading at end of document with no content → remove
                    removed += 1
                    i += 1
                    continue

                next_level = _heading_level(lines[j].strip())
                if next_level > 0:
                    # Next non-blank line is also a heading
                    if next_level > current_level:
                        # It's a child heading (e.g. ## followed by ###) → KEEP the parent
                        cleaned.append(lines[i])
                    else:
                        # Same or higher level heading follows → this heading is truly empty
                        removed += 1
                        i += 1
                        continue
                else:
                    # Next line is content → heading is fine
                    cleaned.append(lines[i])
            else:
                cleaned.append(lines[i])
            i += 1

        return '\n'.join(cleaned), removed

    def _group_sections_by_h2(self, sections: list) -> list[tuple[dict, list]]:
        """Group sections into (H2, [children]) tuples.

        Args:
            sections: Flat list of section dicts with 'level' field

        Returns:
            List of (h2_section_dict, [h3/h4 children]) tuples
        """
        groups = []
        current_h2 = None
        current_children = []

        for sec in sections:
            level = sec.get("level", "h2")
            if level == "h2":
                if current_h2 is not None:
                    groups.append((current_h2, current_children))
                current_h2 = sec
                current_children = []
            else:
                current_children.append(sec)

        if current_h2 is not None:
            groups.append((current_h2, current_children))

        return groups

    # ──────────────────────────────────────────
    # Context helpers
    # ──────────────────────────────────────────

    def _get_section_knowledge(
        self,
        heading: str,
        key_points: list[str],
        full_knowledge: str,
    ) -> str:
        """Extract knowledge relevant to a specific section.

        For now, returns the full knowledge context (since it's already filtered by RAG).
        In the future, this could do section-level retrieval for even more targeted context.
        """
        if not full_knowledge:
            return ""

        # Return full knowledge — the section writer prompt tells the LLM
        # to use only what's relevant to this section
        return full_knowledge

    def _extract_summary(self, section_content: str, heading: str) -> str:
        """Extract a brief summary from a section for transition context."""
        # Take the last meaningful paragraph
        paragraphs = [p.strip() for p in section_content.split("\n\n") if p.strip()]
        # Filter out headings and very short lines
        content_paragraphs = [
            p for p in paragraphs
            if not p.startswith("#") and len(p.split()) > 10
        ]
        if content_paragraphs:
            last = content_paragraphs[-1]
            # Truncate to ~50 words
            words = last.split()
            return " ".join(words[:50]) + ("..." if len(words) > 50 else "")
        return f"Section about: {heading}"

    def _extract_key_takeaways(
        self, sections: list[str], headings: list[str]
    ) -> list[str]:
        """Extract key takeaways from generated sections for the conclusion."""
        takeaways = []
        for heading, content in zip(headings, sections):
            # Use the heading + first substantive paragraph
            paragraphs = [
                p.strip() for p in content.split("\n\n")
                if p.strip() and not p.strip().startswith("#") and len(p.strip().split()) > 15
            ]
            if paragraphs:
                first = paragraphs[0]
                words = first.split()
                summary = " ".join(words[:30])
                takeaways.append(f"{heading}: {summary}")
            else:
                takeaways.append(heading)
        return takeaways[:8]  # Max 8 takeaways

    # ──────────────────────────────────────────
    # Article Assembly
    # ──────────────────────────────────────────

    def _assemble_article(
        self,
        intro: str,
        sections: list[str],
        conclusion: str,
        faq: str,
    ) -> str:
        """Assemble all parts into a complete article.

        Also strips any accidental 'Kết Luận' headings that the LLM may have
        inserted inside body sections (these belong only in the real conclusion).
        """
        # Regex to detect conclusion-like headings inside body sections
        _conclusion_heading_re = re.compile(
            r'^\s*#{2,4}\s+'
            r'(?:Kết\s*Luận|Tổng\s*Kết|Lời\s*Kết|Tóm\s*Tắt|Conclusion|Summary|Final\s*Thoughts|Wrap\s*Up)'
            r'\s*$',
            re.IGNORECASE | re.MULTILINE,
        )

        parts = [intro.strip()]

        for section in sections:
            section_stripped = section.strip()
            if not section_stripped:
                continue
            # Remove any leaked conclusion headings from body sections
            cleaned = _conclusion_heading_re.sub('', section_stripped)
            # Also remove stray bold conclusion labels like **Kết Luận**
            cleaned = re.sub(
                r'^\s*\*{2,3}\s*(?:Kết\s*Luận|Tổng\s*Kết|Lời\s*Kết)\s*\*{2,3}\s*$',
                '', cleaned, flags=re.IGNORECASE | re.MULTILINE,
            )
            # Collapse any triple+ blank lines left behind
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
            if cleaned:
                parts.append(cleaned)

        if conclusion.strip():
            parts.append(conclusion.strip())

        if faq.strip():
            parts.append(faq.strip())

        return "\n\n".join(parts)

    # Legacy methods preserved for backward compatibility

    def _pick_infographic_h2(self, sections: list) -> dict | None:
        """Pick the single best H2 for a supporting infographic.

        Priority: comparison_table > data_table > step_by_step > technical_breakdown > process_diagram
        Falls back to first H2 with authority/expertise EEAT signal.
        """
        priority_types = [
            "comparison_table", "data_table", "step_by_step",
            "technical_breakdown", "process_diagram",
        ]
        style_map = {
            "comparison_table": "comparison_chart",
            "data_table": "data_visualization",
            "step_by_step": "process_flow",
            "technical_breakdown": "diagram",
            "process_diagram": "process_flow",
        }

        # Pass 1: find by content_type priority
        for target_type in priority_types:
            for s in sections:
                if s.get("level") != "h2":
                    continue
                ct = (s.get("content_type") or "").lower()
                if ct == target_type:
                    return {
                        "heading": s["heading"],
                        "content_type": ct,
                        "style": style_map.get(ct, "infographic"),
                    }

        # Pass 2: fall back to authority/expertise H2
        for s in sections:
            if s.get("level") != "h2":
                continue
            eeat = (s.get("eeat_signal") or "").lower()
            if eeat in ("authority", "expertise"):
                return {
                    "heading": s["heading"],
                    "content_type": eeat,
                    "style": "infographic",
                }

        return None

    # ──────────────────────────────────────────
    # HTML Conversion (reused from ArticleGenerator)
    # ──────────────────────────────────────────

    @staticmethod
    def _format_ol_li_content(content: str) -> str:
        """Split '<strong>Title:</strong> body' onto separate lines for clean HTML."""
        m = re.match(r'^(<strong>[^<]+</strong>)\s+(.*)', content, re.DOTALL)
        if m and m.group(2).strip():
            return f"{m.group(1)}\n    {m.group(2).strip()}"
        return content

    def _convert_to_html(
        self,
        article_text: str,
        title: str,
        internal_links: list,
    ) -> str:
        """Convert markdown article to WordPress-ready HTML."""
        html = article_text

        # Convert headings
        html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Strip bold/italic markers around links BEFORE conversion
        # e.g. **[text](url)** → [text](url)
        html = re.sub(r'\*{1,3}(\[[^\]]+\]\([^\)]+\))\*{1,3}', r'\1', html)

        # Convert markdown links FIRST (before bold/italic to avoid conflicts)
        # Build set of known internal URLs from KB for dofollow/nofollow decision
        internal_url_set = set()
        if internal_links:
            for lnk in internal_links:
                u = lnk.get("url", "") if isinstance(lnk, dict) else ""
                if u:
                    internal_url_set.add(u.rstrip("/"))

        seen_urls = set()  # Track URLs to deduplicate

        def _link_replacer(match):
            anchor = match.group(1)
            url = match.group(2)
            url_normalized = url.rstrip("/")

            # Deduplicate: if this exact URL already appeared, skip (keep anchor text only)
            if url_normalized in seen_urls:
                return anchor
            seen_urls.add(url_normalized)

            # Internal links → no nofollow; External links → nofollow
            # All target="_blank" links get noopener noreferrer for security
            if url_normalized in internal_url_set:
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="font-weight: bold;">{anchor}</a>'
            else:
                return f'<a href="{url}" target="_blank" rel="nofollow noopener noreferrer" style="font-weight: bold;">{anchor}</a>'

        html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', _link_replacer, html)

        # Convert bold and italic (AFTER links, so ** won't wrap link syntax)
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Clean up any stray * markers around <a> tags
        html = re.sub(r'\*{1,3}(<a\s[^>]*>.*?</a>)\*{1,3}', r'\1', html)

        # Convert lists and tables
        lines = html.split('\n')
        result_lines = []
        in_ol = False            # Inside an <ol>
        ol_li_open = False       # Unclosed <li> inside <ol> (deferred close for nesting)
        in_nested_ul = False     # Inside <ul> nested within <ol><li>
        in_standalone_ul = False # Inside a standalone <ul>
        ol_counter = 0
        in_table = False
        table_lines = []

        for line in lines:
            stripped = line.strip()

            # ── Table handling ──
            if stripped.startswith('|') and stripped.endswith('|'):
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(stripped)
                continue
            elif in_table:
                in_table = False
                result_lines.append(self._convert_table(table_lines))
                table_lines = []

            is_ul_item = bool(re.match(r'^[-*+] ', stripped))
            is_ol_item = bool(re.match(r'^\d+\.\s', stripped))
            is_heading = stripped.startswith('<h')
            is_empty = not stripped

            # ── Closing logic (only on non-empty lines) ──
            if not is_empty:
                # Close nested <ul> if moving away from bullets
                if in_nested_ul and not is_ul_item:
                    result_lines.append('    </ul>')
                    in_nested_ul = False

                # Close open ol <li> if NOT a nested bullet
                if ol_li_open and not is_ul_item:
                    result_lines.append('  </li>')
                    ol_li_open = False

                # Close <ol> completely if meeting non-list content
                if in_ol and not is_ol_item and not is_ul_item:
                    result_lines.append('</ol>')
                    in_ol = False
                    ol_counter = 0

                # Close standalone <ul> if not a bullet
                if in_standalone_ul and not is_ul_item:
                    result_lines.append('</ul>')
                    in_standalone_ul = False

            # ── Ordered list item ──
            if is_ol_item:
                if not in_ol:
                    result_lines.append('<ol>')
                    in_ol = True
                    ol_counter = 0
                content = re.sub(r'^\d+\.\s', '', stripped)
                formatted = self._format_ol_li_content(content)
                result_lines.append(f'\n  <li>\n    {formatted}')
                ol_li_open = True
                ol_counter += 1
                continue

            # ── Unordered list item ──
            if is_ul_item:
                content = re.sub(r'^[-*+] ', '', stripped)
                if in_ol and ol_li_open:
                    # Nest inside the current <ol><li>
                    if not in_nested_ul:
                        result_lines.append('    <ul>')
                        in_nested_ul = True
                    result_lines.append(f'      <li>{content}</li>')
                else:
                    # Standalone <ul>
                    if not in_standalone_ul:
                        result_lines.append('<ul>')
                        in_standalone_ul = True
                    result_lines.append(f'<li>{content}</li>')
                continue

            # ── Blockquotes ──
            if stripped.startswith('&gt;') or stripped.startswith('>'):
                quote_content = re.sub(r'^(&gt;|>)\s*', '', stripped)
                result_lines.append(
                    f'<blockquote style="border-left: 4px solid #e0e0e0; padding: 10px 20px; '
                    f'margin: 15px 0; font-style: italic; color: #555;">{quote_content}</blockquote>'
                )
                continue

            # ── Paragraphs ──
            # Block-level tags that should NOT be wrapped in <p>
            _block_tags = ('<h', '<div', '<table', '<ol', '<ul', '<li', '<blockquote',
                           '<figure', '<img', '<hr', '<br', '<pre', '<code', '</') 
            is_block = stripped.startswith(_block_tags)
            if stripped and not is_heading and not is_block:
                result_lines.append(f'<p>{stripped}</p>')
            elif stripped:
                result_lines.append(stripped)
            else:
                result_lines.append('')

        # ── Cleanup: close any open tags ──
        if in_nested_ul:
            result_lines.append('    </ul>')
        if ol_li_open:
            result_lines.append('  </li>')
        if in_ol:
            result_lines.append('</ol>')
        if in_standalone_ul:
            result_lines.append('</ul>')
        if in_table:
            result_lines.append(self._convert_table(table_lines))

        html = '\n'.join(result_lines)

        html = re.sub(r'\n{3,}', '\n\n', html)

        # ── Final link attribute validation ──
        # Ensure every <a> tag has correct rel attribute
        def _validate_link_attrs(m):
            tag = m.group(0)
            href_match = re.search(r'href="([^"]+)"', tag)
            if not href_match:
                return tag
            url = href_match.group(1).rstrip('/')
            anchor_content = m.group(1)
            # Remove any existing rel attribute
            tag_clean = re.sub(r'\s*rel="[^"]*"', '', tag)
            if url in internal_url_set:
                # Internal link -> noopener noreferrer only
                tag_clean = tag_clean.replace('>', ' rel="noopener noreferrer">', 1)
            else:
                # External link -> nofollow noopener noreferrer
                tag_clean = tag_clean.replace('>', ' rel="nofollow noopener noreferrer">', 1)
            return tag_clean

        html = re.sub(r'<a\s[^>]*>(.*?)</a>', _validate_link_attrs, html)

        return html.strip()

    def _convert_table(self, table_lines: list) -> str:
        """Convert markdown table to styled HTML table."""
        if len(table_lines) < 2:
            return ''

        table_style = (
            'width: 100%; border-collapse: collapse; margin: 25px 0; '
            'font-family: sans-serif; min-width: 400px; border: 1px solid #dddddd;'
        )
        html = f'<table style="{table_style}">\n'

        headers = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
        html += '<thead>\n<tr style="background-color: #f8f8f8; text-align: left;">\n'
        for i, h in enumerate(headers):
            border_right = ' border-right: 2px solid #eeeeee;' if i < len(headers) - 1 else ''
            html += (
                f'<th style="padding: 12px 15px; border-bottom: 2px solid #eeeeee; '
                f'color: #333333; font-weight: bold;{border_right} text-align:center;">{h}</th>\n'
            )
        html += '</tr>\n</thead>\n'

        html += '<tbody>\n'
        data_rows = table_lines[2:]
        for row_line in data_rows:
            cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
            html += '<tr style="border-bottom: 1px solid #dddddd;">\n'
            for i, cell in enumerate(cells):
                border_right = ' border-right: 2px solid #eeeeee;' if i < len(cells) - 1 else ''
                if i == 0:
                    html += (
                        f'<td style="padding: 12px 15px; font-weight: bold; '
                        f'color: #555555;{border_right} text-align:center;">{cell}</td>\n'
                    )
                else:
                    html += f'<td style="padding: 12px 15px; color: #666666;{border_right}">{cell}</td>\n'
            html += '</tr>\n'
        html += '</tbody>\n</table>'

        return html
