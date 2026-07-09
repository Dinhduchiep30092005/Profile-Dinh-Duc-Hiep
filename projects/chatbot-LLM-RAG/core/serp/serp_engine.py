"""
[1] SERP Intelligence Engine
=============================
Analyzes top Google results for a keyword using SerpAPI.

Extracts:
- Top 10-20 organic results
- People Also Ask (PAA)
- Related Searches
- Reddit/forum mentions

Produces:
- SERP structure map (common H2/H3 patterns)
- Average word count
- Dominant intent
- Content patterns
- Saturation score
"""

import logging
from statistics import mean
from typing import Optional

from serpapi import GoogleSearch

from config import settings
from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.serp.serp_parser import SerpParser
from core.serp.serp_snapshot import SerpSnapshot

logger = logging.getLogger(__name__)


class SerpEngine:
    """SERP Intelligence Engine - Analyzes search results for a keyword."""

    def __init__(self):
        self.api_key = settings.serpapi.api_key
        self.num_results = settings.serpapi.results_count
        self.parser = SerpParser()
        self.snapshot = SerpSnapshot()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def analyze(self, keyword: str, language: str = "vi") -> dict:
        """
        Full SERP analysis pipeline for a keyword.

        Args:
            keyword: The keyword to analyze.
            language: Language code for Google locale (vi, en, en-us, ja, ko, etc.).

        Returns dict with:
        - serp_results: raw top results
        - people_also_ask: PAA questions
        - related_searches: related queries
        - structure_map: common H2/H3 patterns
        - avg_word_count: average word count of top results
        - dominant_intent: inferred intent type
        - content_patterns: recurring content patterns
        - saturation_score: how saturated the SERP is (0-100)
        """
        logger.info(f"[SERP] Analyzing keyword: '{keyword}' (language={language})")

        # Step 1: Fetch SERP data from SerpAPI
        serp_data = self._fetch_serp(keyword, language)

        # Step 2: Extract structured info from top results
        organic = serp_data.get("organic_results", [])[:self.num_results]
        paa = serp_data.get("people_also_ask", [])
        related = serp_data.get("related_searches", [])

        # Step 3: Scrape and analyze top pages (delegated to parser)
        page_analyses = self.parser.analyze_top_pages(organic[:10])

        # Step 4: Build structure map from page analyses
        structure_map = self.parser.build_structure_map(page_analyses)

        # Step 5: Calculate average word count
        word_counts = [p["word_count"] for p in page_analyses if p.get("word_count")]
        avg_word_count = int(mean(word_counts)) if word_counts else 0

        # Step 6: Use LLM to analyze intent and patterns
        llm_analysis = self._llm_analyze_serp(keyword, organic, paa, related, structure_map)

        # Step 7: Calculate saturation score
        saturation_score = self._calculate_saturation(organic, llm_analysis)

        result = {
            "keyword": keyword,
            "serp_results": [
                {
                    "position": r.get("position"),
                    "title": r.get("title"),
                    "link": r.get("link"),
                    "snippet": r.get("snippet"),
                    "domain": r.get("displayed_link", "").split("/")[0] if r.get("displayed_link") else "",
                }
                for r in organic
            ],
            "people_also_ask": [
                {"question": q.get("question"), "snippet": q.get("snippet")}
                for q in paa
            ],
            "related_searches": [
                r.get("query") for r in related
            ],
            "structure_map": structure_map,
            "avg_word_count": avg_word_count,
            "dominant_intent": llm_analysis.get("dominant_intent", "informational"),
            "content_patterns": llm_analysis.get("content_patterns", []),
            "saturation_score": saturation_score,
            "gap_opportunities": llm_analysis.get("gap_opportunities", []),
        }

        # Step 8: Save to database
        self.snapshot.save(keyword, result)

        logger.info(f"[SERP] Analysis complete for '{keyword}' | "
                     f"Saturation: {saturation_score}/100 | "
                     f"Intent: {result['dominant_intent']}")

        return result

    # ──────────────────────────────────────────
    # SerpAPI interaction
    # ──────────────────────────────────────────

    def _fetch_serp(self, keyword: str, language: str = "vi") -> dict:
        """Fetch SERP data from SerpAPI with locale matching the selected language."""
        from config import LANGUAGE_LOCALE_MAP
        locale = LANGUAGE_LOCALE_MAP.get(language, LANGUAGE_LOCALE_MAP.get("vi"))
        gl = locale["gl"]
        hl = locale["hl"]
        logger.info(f"[SERP] Using locale: gl={gl}, hl={hl} for language '{language}'")

        params = {
            "q": keyword,
            "api_key": self.api_key,
            "engine": "google",
            "num": self.num_results,
            "gl": gl,
            "hl": hl,
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            logger.info(f"[SERP] Fetched {len(results.get('organic_results', []))} organic results")
            return results
        except Exception as e:
            logger.error(f"[SERP] SerpAPI error: {e}")
            return {"organic_results": [], "people_also_ask": [], "related_searches": []}

    # ──────────────────────────────────────────
    # LLM Analysis
    # ──────────────────────────────────────────

    def _llm_analyze_serp(
        self,
        keyword: str,
        organic: list,
        paa: list,
        related: list,
        structure_map: dict,
    ) -> dict:
        """Use GPT to analyze SERP data and extract insights."""

        titles = [r.get("title", "") for r in organic]
        snippets = [r.get("snippet", "") for r in organic]
        paa_questions = [q.get("question", "") for q in paa]
        related_queries = [r.get("query", r) if isinstance(r, dict) else r for r in related]

        user_prompt = f"""Keyword: {keyword}

Top Result Titles:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(titles))}

Top Result Snippets:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(snippets))}

People Also Ask:
{chr(10).join(f'- {q}' for q in paa_questions)}

Related Searches:
{chr(10).join(f'- {q}' for q in related_queries)}

Common H2 Headings:
{chr(10).join(f'- {h["text"]} (found in {h["frequency"]} results)' for h in structure_map.get("common_h2", []))}

Analyze this SERP data and identify the dominant intent, content patterns, and gap opportunities."""

        return llm.chat_json(PromptTemplates.SERP_ANALYSIS_SYSTEM, user_prompt, temperature=0.3)

    # ──────────────────────────────────────────
    # Saturation Score
    # ──────────────────────────────────────────

    def _calculate_saturation(self, organic: list, llm_analysis: dict) -> float:
        """
        Calculate SERP saturation score (0-100).
        Higher = more saturated = harder to rank.
        """
        score = 0.0

        # Factor 1: Number of strong domains (30 points)
        strong_domains = {"wikipedia", "forbes", "nytimes", "bbc", "cnn",
                          "healthline", "webmd", "gov", "edu"}
        domain_hits = sum(
            1 for r in organic
            if any(d in r.get("displayed_link", "").lower() for d in strong_domains)
        )
        score += min(domain_hits * 6, 30)

        # Factor 2: Title similarity (25 points)
        titles = [r.get("title", "").lower() for r in organic if r.get("title")]
        if titles:
            from collections import Counter
            all_words = []
            for t in titles:
                all_words.extend(t.split())
            common_words = [w for w, c in Counter(all_words).items() if c >= len(titles) * 0.5]
            title_similarity = min(len(common_words) * 3, 25)
            score += title_similarity

        # Factor 3: Content features coverage (20 points)
        score += 20

        # Factor 4: Gap opportunities (25 points, inverse)
        gaps = llm_analysis.get("gap_opportunities", [])
        gap_reduction = min(len(gaps) * 5, 25)
        score -= gap_reduction

        return max(0, min(100, round(score, 1)))
