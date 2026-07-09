"""
Heading Generator — Creates SEO + EEAT optimized heading structures.
Uses SERP data, intent mapping, brand knowledge RAG, and EEAT framework.

Full RAG-powered flow:
    1. Click title → receive title + keyword + SERP context
    2. Classify intent (informational / commercial / comparison / transactional)
    3. Intent-aware multi-query semantic search in uploaded documents
    4. Retrieve & categorise chunks (use_case / feature / technical / guideline / case_study)
    5. Inject structured knowledge context into prompt
    6. Generate comprehensive EEAT heading structure
"""

import logging
from typing import Optional

from config import settings
from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.knowledge.retrieval import (
    build_rag_context,
    build_product_solution_context,
    has_knowledge_base,
)

logger = logging.getLogger(__name__)


class HeadingGenerator:
    """Generates SEO + EEAT heading structures from a title and SERP context."""

    def generate(
        self,
        title: str,
        keyword: str,
        intent: Optional[dict] = None,
        gaps: Optional[dict] = None,
        serp_results: Optional[list] = None,
        people_also_ask: Optional[list] = None,
        language: str = "vi",
    ) -> dict:
        """
        Generate a complete heading structure for an article.

        Pipeline:
            Step 1 — Parse intent from SERP data
            Step 2 — RAG: intent-aware semantic search in documents
            Step 3 — Build structured prompt with SERP + RAG context
            Step 4 — LLM generates heading structure
            Step 5 — Return result + RAG metadata

        Returns:
            dict with heading structure, FAQ, EEAT blocks, rag_info
        """
        logger.info(f"[HeadingGen] ══ Pipeline start: '{title}' ══")

        intent = intent or {}
        gaps = gaps or {}
        serp_results = serp_results or []
        people_also_ask = people_also_ask or []

        # ── Step 1: Resolve intent type ──
        intent_type = self._resolve_intent(intent, keyword)
        logger.info(f"[HeadingGen] Step 1 — Intent: {intent_type}")

        # ── Step 2: RAG retrieval (intent-aware) ──
        knowledge_context = ""
        rag_metadata = {"has_knowledge": False}
        product_context = ""
        product_metadata = {"is_product_relevant": False}

        if has_knowledge_base():
            logger.info("[HeadingGen] Step 2a — RAG: multi-query semantic search")
            knowledge_context, rag_metadata = build_rag_context(
                keyword=keyword,
                title=title,
                intent_type=intent_type,
                max_tokens_approx=3000,
            )
            if knowledge_context:
                logger.info(
                    f"[HeadingGen] Step 2a — Retrieved {rag_metadata.get('chunks_found', 0)} chunks "
                    f"across {rag_metadata.get('categories_found', 0)} categories "
                    f"from {rag_metadata.get('documents_used', 0)} documents"
                )
            else:
                logger.info("[HeadingGen] Step 2a — No relevant chunks found")

            # Step 2b: Product solution detection
            logger.info("[HeadingGen] Step 2b — Product solution relevance check")
            product_context, product_metadata = build_product_solution_context(
                keyword=keyword,
                title=title,
                intent_type=intent_type,
                relevance_threshold=0.35,
                max_tokens_approx=1500,
            )
            if product_metadata.get("is_product_relevant"):
                logger.info(
                    f"[HeadingGen] Step 2b — Product RELEVANT (score: {product_metadata.get('relevance_score', 0):.3f}) "
                    f"| Products: {product_metadata.get('product_names', [])} "
                    f"| Section type: {product_metadata.get('recommended_section_type', 'N/A')}"
                )
            else:
                logger.info(
                    f"[HeadingGen] Step 2b — Product not relevant enough "
                    f"(score: {product_metadata.get('relevance_score', 0):.3f})"
                )
        else:
            logger.info("[HeadingGen] Step 2 — Skipped (no documents in knowledge base)")

        # ── Step 3: Build SERP context ──
        serp_context = self._build_serp_context(serp_results, people_also_ask)
        intent_context = self._build_intent_context(intent, gaps)
        logger.info("[HeadingGen] Step 3 — Prompt built")

        # ── Step 4: Build user prompt & call LLM ──
        user_prompt = self._build_user_prompt(
            title=title,
            keyword=keyword,
            intent_type=intent_type,
            serp_context=serp_context,
            intent_context=intent_context,
            knowledge_context=knowledge_context,
            product_context=product_context,
            product_metadata=product_metadata,
            language=language,
        )

        try:
            result = llm.chat_json(
                PromptTemplates.HEADING_GENERATOR_SYSTEM,
                user_prompt,
                temperature=0.5,
                max_tokens=4096,
            )
            logger.info(
                f"[HeadingGen] Step 4 — Generated {len(result.get('sections', []))} sections"
            )
        except Exception as e:
            logger.error(f"[HeadingGen] Step 4 — LLM failed: {e}")
            raise

        # ── Step 5: Attach RAG metadata for frontend ──
        result["rag_info"] = {
            "intent_type": intent_type,
            "has_knowledge": rag_metadata.get("has_knowledge", False),
            "chunks_found": rag_metadata.get("chunks_found", 0),
            "categories_found": rag_metadata.get("categories_found", 0),
            "documents_used": rag_metadata.get("documents_used", 0),
            "queries_executed": rag_metadata.get("queries_executed", 0),
            "categories": rag_metadata.get("categories", []),
            "queries_used": rag_metadata.get("queries_used", []),
        }

        # Attach product solution metadata
        result["product_solution"] = {
            "is_product_relevant": product_metadata.get("is_product_relevant", False),
            "relevance_score": product_metadata.get("relevance_score", 0),
            "product_names": product_metadata.get("product_names", []),
            "recommended_section_type": product_metadata.get("recommended_section_type", ""),
        }

        logger.info("[HeadingGen] ══ Pipeline complete ══")
        return result

    # ──────────────────────────────────────────
    # Step 1: Intent resolution
    # ──────────────────────────────────────────

    def _resolve_intent(self, intent: dict, keyword: str) -> str:
        """
        Extract a normalised intent type from the SERP analysis.
        Falls back to keyword heuristic if no intent data is provided.
        """
        if intent and intent.get("intent_type"):
            raw = intent["intent_type"].lower().strip()
            valid = {"informational", "commercial", "comparison", "transactional"}
            if raw in valid:
                return raw

        # Heuristic fallback based on keyword patterns
        kw = keyword.lower()
        if any(w in kw for w in ["vs", "so sánh", "alternative", "đối thủ"]):
            return "comparison"
        if any(w in kw for w in ["best", "top", "review", "đánh giá", "nên mua"]):
            return "commercial"
        if any(w in kw for w in ["buy", "mua", "price", "giá", "coupon", "download"]):
            return "transactional"
        return "informational"

    # ──────────────────────────────────────────
    # SERP & Intent context builders
    # ──────────────────────────────────────────

    def _build_serp_context(self, serp_results: list, paa: list) -> str:
        """Build SERP analysis context for the prompt."""
        parts = []

        if serp_results:
            parts.append("TOP SERP RESULTS:")
            for r in serp_results[:10]:
                parts.append(
                    f"  #{r.get('position', '?')}: {r.get('title', 'N/A')} "
                    f"[{r.get('domain', '')}]"
                )

        if paa:
            parts.append("\nPEOPLE ALSO ASK:")
            for q in paa[:8]:
                question = q.get("question", q) if isinstance(q, dict) else q
                parts.append(f"  - {question}")

        return "\n".join(parts) if parts else "Không có dữ liệu SERP"

    def _build_intent_context(self, intent: dict, gaps: dict) -> str:
        """Build intent and gap analysis context."""
        parts = []

        if intent:
            parts.append(f"SEARCH INTENT: {intent.get('intent_type', 'unknown')}")
            parts.append(f"Confidence: {intent.get('confidence', 0):.0%}")
            if intent.get("reasoning"):
                parts.append(f"Reasoning: {intent['reasoning']}")
            if intent.get("sub_intents"):
                parts.append(f"Sub-intents: {', '.join(intent['sub_intents'])}")
            if intent.get("content_expectation"):
                parts.append(f"Content expectation: {intent['content_expectation']}")

        if gaps:
            intent_gaps = gaps.get("intent_gaps", [])
            if intent_gaps:
                parts.append("\nCONTENT GAPS (untapped angles):")
                for g in intent_gaps[:5]:
                    parts.append(
                        f"  - {g.get('gap', '')} "
                        f"(opportunity: {g.get('opportunity_score', '?')}/10)"
                    )

            if gaps.get("technical_depth_missing"):
                parts.append("⚠️ Technical depth is MISSING in current SERP")

            missing_types = gaps.get("missing_content_types", [])
            if missing_types:
                parts.append(f"Missing content types: {', '.join(missing_types)}")

        return "\n".join(parts) if parts else ""

    # ──────────────────────────────────────────
    # Prompt assembly
    # ──────────────────────────────────────────

    def _build_user_prompt(
        self,
        title: str,
        keyword: str,
        intent_type: str,
        serp_context: str,
        intent_context: str,
        knowledge_context: str,
        product_context: str = "",
        product_metadata: dict = None,
        language: str = "vi",
    ) -> str:
        """Assemble the full user prompt with RAG + product solution context."""
        product_metadata = product_metadata or {}

        # Determine language instruction based on language code
        lang_instructions = {
            "vi": "Viết toàn bộ headings bằng tiếng Việt.",
            "en": "Write all headings in English (British English).",
            "en-us": "Write all headings in English (American English).",
            "en-au": "Write all headings in English (Australian English).",
            "fr": "Rédigez tous les headings en français.",
            "de": "Schreiben Sie alle Überschriften auf Deutsch.",
            "es": "Escribe todos los encabezados en español.",
            "ja": "すべての見出しを日本語で書いてください。",
            "ko": "모든 제목을 한국어로 작성하세요.",
            "zh": "用简体中文写所有标题。",
            "zh-tw": "用繁體中文寫所有標題。",
            "th": "เขียนหัวข้อทั้งหมดเป็นภาษาไทย",
            "pt": "Escreva todos os títulos em português.",
            "it": "Scrivi tutti i titoli in italiano.",
            "ru": "Напишите все заголовки на русском языке.",
            "id": "Tulis semua heading dalam Bahasa Indonesia.",
        }
        lang_instruction = lang_instructions.get(
            language,
            f"Write all headings in the language matching code '{language}'."
        )

        parts = [
            f"TITLE: {title}",
            f"KEYWORD: {keyword}",
            f"INTENT: {intent_type}",
            f"LANGUAGE: {lang_instruction}",
            "",
            serp_context,
        ]

        if intent_context:
            parts.append("")
            parts.append(intent_context)

        if knowledge_context:
            parts.append("")
            parts.append(knowledge_context)
            parts.append("")
            parts.append(
                "QUAN TRỌNG: Sử dụng brand knowledge ở trên để tạo heading chuyên sâu, "
                "cụ thể, khác biệt. Ưu tiên sử dụng thuật ngữ, tính năng, use case "
                "từ tài liệu nội bộ thay vì heading chung chung.\n"
                "Ví dụ thay vì 'What is Proxy Router?' → "
                "'How Router-Level IP Isolation Prevents Multi-Account Detection'"
            )

        # ── Product as Solution injection ──
        if product_context and product_metadata.get("is_product_relevant"):
            section_type = product_metadata.get("recommended_section_type", "product_as_solution")
            product_names = product_metadata.get("product_names", [])
            names_str = ", ".join(product_names) if product_names else "sản phẩm của chúng tôi"

            parts.append("")
            parts.append(product_context)
            parts.append("")

            if section_type == "product_recommendation":
                parts.append(
                    f"🎯 YÊU CẦU ĐẶC BIỆT — GIỚI THIỆU SẢN PHẨM NHƯ GIẢI PHÁP:\n"
                    f"Bài viết này CÓ LIÊN QUAN trực tiếp đến sản phẩm: {names_str}.\n"
                    f"Hãy tạo 1-2 section H2 hoặc H3 giới thiệu sản phẩm như giải pháp được khuyến nghị.\n"
                    f"- Đề cập cụ thể tính năng, thông số từ tài liệu sản phẩm ở trên\n"
                    f"- Section nên có tiêu đề tự nhiên, không quảng cáo lộ liễu\n"
                    f"- Ví dụ: 'Giải pháp [feature] cho [vấn đề trong keyword]'\n"
                    f"- Đánh dấu brand_knowledge_used: true cho các section này"
                )
            elif section_type == "product_comparison":
                parts.append(
                    f"🎯 YÊU CẦU ĐẶC BIỆT — SO SÁNH & GIỚI THIỆU SẢN PHẨM:\n"
                    f"Bài viết này liên quan đến lĩnh vực sản phẩm: {names_str}.\n"
                    f"Hãy tạo 1-2 section H2/H3 so sánh hoặc đánh giá, trong đó nhắc đến sản phẩm\n"
                    f"như một lựa chọn đáng cân nhắc với dữ liệu cụ thể từ tài liệu.\n"
                    f"- Sử dụng thông số kỹ thuật, tính năng thực tế từ tài liệu sản phẩm\n"
                    f"- Có thể dùng bảng so sánh nếu phù hợp\n"
                    f"- Section nên khách quan, có dữ liệu cụ thể\n"
                    f"- Đánh dấu brand_knowledge_used: true cho các section này"
                )
            elif section_type == "product_as_alternative":
                parts.append(
                    f"🎯 YÊU CẦU ĐẶC BIỆT — SẢN PHẨM NHƯ LỰA CHỌN THAY THẾ:\n"
                    f"Bài viết này liên quan đến lĩnh vực: {names_str}.\n"
                    f"Hãy tạo 1 section H2/H3 giới thiệu sản phẩm như một lựa chọn thay thế/bổ sung.\n"
                    f"- Nêu rõ ưu điểm vượt trội dựa trên tài liệu sản phẩm\n"
                    f"- Tự nhiên, không ép buộc — chỉ khi phù hợp với mạch bài\n"
                    f"- Đánh dấu brand_knowledge_used: true cho section này"
                )
            else:  # product_as_solution (informational)
                parts.append(
                    f"🎯 YÊU CẦU ĐẶC BIỆT — NHẮC ĐẾN SẢN PHẨM NHƯ GIẢI PHÁP:\n"
                    f"Bài viết này CÓ LIÊN QUAN đến sản phẩm: {names_str}.\n"
                    f"Hãy tạo 1 section H2 hoặc H3 (đặt tự nhiên trong mạch bài) nhắc đến sản phẩm\n"
                    f"như một giải pháp thực tế cho vấn đề mà bài viết đang thảo luận.\n"
                    f"- KHÔNG quảng cáo lộ liễu — giới thiệu tự nhiên như \"Giải pháp thực tế\" hoặc\n"
                    f"  \"Ứng dụng [tính năng] trong [vấn đề keyword]\"\n"
                    f"- Sử dụng dữ liệu cụ thể từ tài liệu sản phẩm (tính năng, thông số, kết quả thực tế)\n"
                    f"- Section nên nằm ở phần giữa-cuối bài (sau khi đã cung cấp giá trị thông tin)\n"
                    f"- Đánh dấu brand_knowledge_used: true cho section này"
                )
        elif not knowledge_context:
            parts.append("")
            if language == "vi":
                parts.append(
                    "Hãy tạo cấu trúc heading đầy đủ, chuẩn SEO và EEAT cho bài viết này."
                )
            else:
                parts.append(
                    "Generate a complete, SEO and EEAT optimized heading structure for this article."
                )

        return "\n".join(parts)
