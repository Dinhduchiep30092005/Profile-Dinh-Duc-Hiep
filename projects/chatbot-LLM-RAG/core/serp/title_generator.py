"""
SERP-based Title Generator
============================
Full pipeline: Keyword → SERP crawl → Extract → Intent → Gap → Generate titles → CTR score.

Flow:
    1. Keyword input
    2. SERP API (top 20)
    3. Extract structured data (titles, snippets, URLs, domains, features)
    4. Intent classification
    5. Angle gap detection
    6. AI title generation
    7. CTR prediction scoring
"""

import logging
import re
from typing import Optional

from serpapi import GoogleSearch

from config import settings

logger = logging.getLogger(__name__)


class SerpTitleGenerator:
    """Generates optimized titles based on SERP analysis of a keyword."""

    def __init__(self):
        self.api_key = settings.serpapi.api_key
        self.num_results = min(settings.serpapi.results_count, 20)

    # ──────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────

    def generate(self, keyword: str, num_titles: int = 5, language: str = "vi") -> dict:
        """
        Full pipeline: keyword → SERP → analysis → titles.

        Returns dict with:
        - keyword, serp_results, intent, gaps, titles, summary
        """
        logger.info(f"[TitleGen] Starting pipeline for: '{keyword}'")

        # Step 1 — Fetch SERP top 20
        serp_data = self._fetch_serp(keyword, language)
        organic = serp_data.get("organic_results", [])[:self.num_results]
        paa = serp_data.get("people_also_ask", [])
        related = serp_data.get("related_searches", [])

        if not organic:
            return {
                "keyword": keyword,
                "error": "Không tìm thấy kết quả SERP cho từ khoá này.",
                "serp_results": [],
                "intent": {},
                "gaps": {},
                "titles": [],
            }

        # Step 2 — Extract structured data
        structured = self._extract_structured(organic, paa, related)

        # Step 3 — Intent classification (LLM)
        intent = self._classify_intent(keyword, structured)

        # Step 4 — Angle gap detection (LLM)
        gaps = self._detect_gaps(keyword, structured, intent)

        # Step 5 — AI title generation + CTR scoring (LLM)
        titles = self._generate_titles(keyword, structured, intent, gaps, num_titles, language)

        result = {
            "keyword": keyword,
            "serp_results": structured["results"],
            "people_also_ask": structured["paa"],
            "related_searches": structured["related"],
            "intent": intent,
            "gaps": gaps,
            "titles": titles,
            "summary": {
                "total_serp": len(structured["results"]),
                "dominant_intent": intent.get("intent_type", "unknown"),
                "confidence": intent.get("confidence", 0),
                "gap_count": len(gaps.get("intent_gaps", [])),
                "titles_generated": len(titles),
            },
        }

        logger.info(f"[TitleGen] Pipeline complete: {len(titles)} titles generated")
        return result

    # ──────────────────────────────────────────
    # Step 1: Fetch SERP
    # ──────────────────────────────────────────

    def _fetch_serp(self, keyword: str, language: str = "vi") -> dict:
        """Fetch SERP data from SerpAPI. Auto-paginate if first page returns fewer than expected."""
        from config import LANGUAGE_LOCALE_MAP
        locale = LANGUAGE_LOCALE_MAP.get(language, LANGUAGE_LOCALE_MAP.get("vi"))
        gl = locale["gl"]
        hl = locale["hl"]
        logger.info(f"[TitleGen] Using locale: gl={gl}, hl={hl} for language '{language}'")

        params = {
            "q": keyword,
            "api_key": self.api_key,
            "engine": "google",
            "num": min(self.num_results, 10),
            "gl": gl,
            "hl": hl,
        }

        try:
            # Page 1
            search = GoogleSearch(params)
            results = search.get_dict()
            organic = results.get("organic_results", [])
            paa = results.get("people_also_ask", [])
            related = results.get("related_searches", [])
            logger.info(f"[TitleGen] Page 1: {len(organic)} organic results for '{keyword}'")

            # Page 2 if we need more and page 1 returned results
            if len(organic) < self.num_results and len(organic) > 0:
                params2 = {**params, "start": len(organic)}
                try:
                    search2 = GoogleSearch(params2)
                    results2 = search2.get_dict()
                    page2 = results2.get("organic_results", [])
                    logger.info(f"[TitleGen] Page 2: {len(page2)} more results")

                    # Deduplicate by URL
                    existing_urls = {r.get("link", "") for r in organic}
                    for r in page2:
                        if r.get("link", "") not in existing_urls:
                            organic.append(r)
                            existing_urls.add(r.get("link", ""))

                    # Merge PAA & related from page 2
                    existing_paa = {q.get("question", "") for q in paa}
                    for q in results2.get("people_also_ask", []):
                        if q.get("question", "") not in existing_paa:
                            paa.append(q)

                    existing_related = {r.get("query", r) if isinstance(r, dict) else str(r) for r in related}
                    for r in results2.get("related_searches", []):
                        rq = r.get("query", r) if isinstance(r, dict) else str(r)
                        if rq not in existing_related:
                            related.append(r)
                except Exception as e2:
                    logger.warning(f"[TitleGen] Page 2 fetch failed (non-critical): {e2}")

            # Cap at num_results
            organic = organic[:self.num_results]
            logger.info(f"[TitleGen] Total: {len(organic)} organic results for '{keyword}'")

            return {
                "organic_results": organic,
                "people_also_ask": paa,
                "related_searches": related,
            }
        except Exception as e:
            logger.error(f"[TitleGen] SerpAPI error: {e}")
            return {"organic_results": [], "people_also_ask": [], "related_searches": []}

    # ──────────────────────────────────────────
    # Step 2: Extract structured data
    # ──────────────────────────────────────────

    def _extract_structured(self, organic: list, paa: list, related: list) -> dict:
        """Extract clean, structured data from raw SERP results."""
        results = []
        for r in organic:
            domain = ""
            displayed = r.get("displayed_link", "")
            if displayed:
                domain = displayed.split("/")[0].replace("https://", "").replace("http://", "")

            results.append({
                "position": r.get("position", 0),
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", ""),
                "domain": domain,
                "title_length": len(r.get("title", "")),
                "has_number": bool(re.search(r"\d+", r.get("title", ""))),
                "has_year": bool(re.search(r"202[0-9]", r.get("title", ""))),
                "has_bracket": bool(re.search(r"[\[\(]", r.get("title", ""))),
                "has_power_word": self._has_power_word(r.get("title", "")),
            })

        paa_list = [
            {"question": q.get("question", ""), "snippet": q.get("snippet", "")}
            for q in paa
        ]

        related_list = []
        for r in related:
            if isinstance(r, str):
                related_list.append(r)
            elif isinstance(r, dict):
                # SerpAPI may use "query", "text", or other keys
                text = r.get("query") or r.get("text") or r.get("title") or ""
                if text and isinstance(text, str):
                    related_list.append(text)
                else:
                    # Last resort: stringify the dict values
                    vals = [str(v) for v in r.values() if v and isinstance(v, (str, int, float))]
                    if vals:
                        related_list.append(vals[0])
            else:
                related_list.append(str(r))

        return {
            "results": results,
            "paa": paa_list,
            "related": related_list,
        }

    def _has_power_word(self, title: str) -> bool:
        """Check if title contains power/emotional words."""
        power_words = [
            "best", "top", "ultimate", "complete", "proven", "essential",
            "free", "secret", "amazing", "powerful", "easy", "fast",
            "tốt nhất", "hướng dẫn", "chi tiết", "đầy đủ", "miễn phí",
            "bí quyết", "cách", "mẹo", "top", "review", "đánh giá",
        ]
        title_lower = title.lower()
        return any(w in title_lower for w in power_words)

    # ──────────────────────────────────────────
    # Step 3: Intent classification
    # ──────────────────────────────────────────

    def _classify_intent(self, keyword: str, structured: dict) -> dict:
        """Use LLM to classify search intent from SERP data."""
        from core.llm.llm_client import llm

        titles = [r["title"] for r in structured["results"]]
        snippets = [r["snippet"] for r in structured["results"]]
        paa_questions = [q["question"] for q in structured["paa"]]

        system_prompt = """Bạn là chuyên gia SEO phân tích search intent. Phân tích từ khoá và dữ liệu SERP.

Trả về JSON object:
{
  "intent_type": "informational" | "commercial" | "comparison" | "transactional" | "navigational",
  "confidence": 0.0-1.0,
  "sub_intents": ["intent phụ 1", "intent phụ 2"],
  "reasoning": "giải thích ngắn gọn",
  "user_persona": "mô tả ngắn người tìm kiếm",
  "content_expectation": "người tìm kiếm mong đợi loại nội dung gì"
}

Quy tắc:
- informational: muốn tìm hiểu (cách, hướng dẫn, là gì)
- commercial: nghiên cứu trước mua (tốt nhất, review, top)
- comparison: so sánh lựa chọn (A vs B, thay thế)
- transactional: sẵn sàng mua/hành động (mua, giá, download)
- navigational: tìm trang cụ thể (tên thương hiệu)"""

        user_prompt = f"""Từ khoá: {keyword}

Top Titles:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(titles[:15]))}

Top Snippets:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(snippets[:10]))}

People Also Ask:
{chr(10).join(f'- {q}' for q in paa_questions[:8])}
"""

        try:
            return llm.chat_json(system_prompt, user_prompt, temperature=0.2)
        except Exception as e:
            logger.error(f"[TitleGen] Intent classification failed: {e}")
            return {
                "intent_type": "informational",
                "confidence": 0.5,
                "sub_intents": [],
                "reasoning": "Fallback - LLM error",
            }

    # ──────────────────────────────────────────
    # Step 4: Angle gap detection
    # ──────────────────────────────────────────

    def _detect_gaps(self, keyword: str, structured: dict, intent: dict) -> dict:
        """Use LLM to detect content angle gaps in current SERP."""
        from core.llm.llm_client import llm

        titles = [r["title"] for r in structured["results"]]
        snippets = [r["snippet"] for r in structured["results"][:10]]

        system_prompt = """Bạn là chuyên gia SEO phân tích gap trong SERP. Phân tích các kết quả hiện tại và tìm ra góc nhìn/angle bị thiếu.

Trả về JSON object:
{
  "exploited_angles": ["góc đã khai thác nhiều 1", "góc 2"],
  "intent_gaps": [
    {
      "gap": "mô tả góc chưa được khai thác",
      "opportunity_score": 1-10,
      "reason": "tại sao đây là cơ hội",
      "suggested_approach": "cách tiếp cận nội dung"
    }
  ],
  "title_patterns": {
    "common_format": "format title phổ biến nhất",
    "avg_title_length": 50,
    "common_elements": ["số liệu", "năm", "brackets"],
    "overused_words": ["từ bị dùng quá nhiều"]
  },
  "differentiation_tips": ["tip tạo sự khác biệt 1", "tip 2"]
}"""

        user_prompt = f"""Từ khoá: {keyword}
Intent chính: {intent.get('intent_type', 'unknown')}

Titles hiện có trong SERP:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(titles))}

Snippets:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(snippets))}

Related searches: {', '.join(structured['related'][:10])}

Phân tích gap và cơ hội cho title mới."""

        try:
            return llm.chat_json(system_prompt, user_prompt, temperature=0.3)
        except Exception as e:
            logger.error(f"[TitleGen] Gap detection failed: {e}")
            return {
                "exploited_angles": [],
                "intent_gaps": [],
                "title_patterns": {},
                "differentiation_tips": [],
            }

    # ──────────────────────────────────────────
    # Step 5: Generate titles + CTR scoring
    # ──────────────────────────────────────────

    def _generate_titles(
        self,
        keyword: str,
        structured: dict,
        intent: dict,
        gaps: dict,
        num_titles: int,
        language: str,
    ) -> list:
        """Use LLM to generate optimized titles with CTR prediction."""
        from core.llm.llm_client import llm

        titles = [r["title"] for r in structured["results"]]
        gap_list = gaps.get("intent_gaps", [])
        patterns = gaps.get("title_patterns", {})
        diff_tips = gaps.get("differentiation_tips", [])

        lang_instructions = {
            "vi": "Viết title bằng tiếng Việt.",
            "en": "Write titles in British English.",
            "en-us": "Write titles in American English.",
            "en-au": "Write titles in Australian English.",
            "fr": "Rédigez les titres en français.",
            "de": "Schreiben Sie die Titel auf Deutsch.",
            "es": "Escriba los títulos en español.",
            "ja": "タイトルを日本語で書いてください。",
            "ko": "제목을 한국어로 작성하세요.",
            "zh": "用简体中文写标题。",
            "zh-tw": "用繁體中文寫標題。",
            "th": "เขียนหัวข้อเป็นภาษาไทย",
            "pt": "Escreva os títulos em português.",
            "it": "Scrivi i titoli in italiano.",
            "ru": "Напишите заголовки на русском языке.",
            "id": "Tulis judul dalam Bahasa Indonesia.",
        }
        lang_instruction = lang_instructions.get(
            language,
            f"Write titles in the language matching code '{language}'."
        )

        system_prompt = f"""Bạn là chuyên gia SEO title optimization với khả năng dự đoán CTR.

NHIỆM VỤ: Tạo {num_titles} title tối ưu cho từ khoá dựa trên phân tích SERP.

{lang_instruction}

QUY TẮC TỐI ƯU TITLE:
1. Độ dài 50-60 ký tự (tối ưu hiển thị Google)
2. Chứa từ khoá chính ở đầu title khi có thể
3. Sử dụng power words để tăng CTR (số liệu, năm hiện tại, brackets, emotional triggers)
4. Mỗi title phải khai thác một ANGLE KHÁC NHAU
5. Tránh lặp lại format/pattern giống các title đã có trong SERP
6. Phù hợp với search intent đã phân tích
7. Tạo sự tò mò nhưng không clickbait

PHƯƠNG PHÁP DỰ ĐOÁN CTR:
- Baseline CTR theo vị trí: Pos1=31%, Pos2=15%, Pos3=11%, Pos4=8%, Pos5=6%
- Bonus +2-5% cho: số liệu cụ thể, năm hiện tại, brackets, power words
- Bonus +1-3% cho: unique angle, emotional trigger, clarity
- Penalty -2-5% cho: quá dài, không rõ ràng, clickbait

Trả về JSON:
{{
  "titles": [
    {{
      "title": "title gợi ý",
      "meta_description": "meta description 150-160 ký tự",
      "angle": "góc tiếp cận của title này",
      "target_intent": "intent mà title này nhắm đến",
      "power_elements": ["yếu tố tăng CTR trong title"],
      "predicted_ctr": 0.08,
      "ctr_reasoning": "giải thích tại sao CTR dự đoán ở mức này",
      "differentiation": "điểm khác biệt so với SERP hiện tại",
      "title_length": 55
    }}
  ]
}}"""

        user_prompt = f"""Từ khoá: {keyword}
Search intent: {intent.get('intent_type', 'unknown')} (confidence: {intent.get('confidence', 0)})
User persona: {intent.get('user_persona', 'N/A')}

Titles hiện có trong SERP (cần KHÁC BIỆT):
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(titles))}

Gaps/cơ hội phát hiện:
{chr(10).join(f'- {g["gap"]} (score: {g.get("opportunity_score", "?")})' for g in gap_list)}

Title patterns phổ biến: {patterns.get('common_format', 'N/A')}
Overused words: {', '.join(patterns.get('overused_words', []))}

Tips tạo sự khác biệt:
{chr(10).join(f'- {t}' for t in diff_tips)}

Tạo {num_titles} title tối ưu, mỗi title khai thác một angle khác nhau."""

        try:
            result = llm.chat_json(system_prompt, user_prompt, temperature=0.7, max_tokens=4096)
            titles_list = result.get("titles", [])

            # Sort by predicted CTR
            titles_list.sort(key=lambda t: t.get("predicted_ctr", 0), reverse=True)

            # Add rank
            for i, t in enumerate(titles_list):
                t["rank"] = i + 1
                # Ensure title_length is calculated
                t["title_length"] = len(t.get("title", ""))

            return titles_list
        except Exception as e:
            logger.error(f"[TitleGen] Title generation failed: {e}")
            return []
