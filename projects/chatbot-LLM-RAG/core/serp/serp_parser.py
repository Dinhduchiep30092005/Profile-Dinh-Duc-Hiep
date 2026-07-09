"""
SERP Parser - Handles scraping and structural analysis of SERP result pages.
Extracts headings, word counts, content features from organic result pages.
"""

import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class SerpParser:
    """Scrapes and analyzes individual SERP result pages for structure."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    def analyze_top_pages(self, organic_results: list) -> list:
        """Scrape and analyze top organic result pages."""
        analyses = []
        for result in organic_results:
            url = result.get("link", "")
            try:
                analysis = self.scrape_page(url)
                analyses.append(analysis)
            except Exception as e:
                logger.warning(f"[SERP] Failed to scrape {url}: {e}")
                analyses.append({
                    "url": url,
                    "headings": [],
                    "word_count": 0,
                    "has_tables": False,
                    "has_lists": False,
                    "has_images": False,
                    "has_faq": False,
                })
        return analyses

    def scrape_page(self, url: str) -> dict:
        """Scrape a single page for structure analysis."""
        resp = requests.get(url, headers=self.HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract headings
        headings = []
        for level in ["h1", "h2", "h3"]:
            for h in soup.find_all(level):
                text = h.get_text(strip=True)
                if text:
                    headings.append({"level": level, "text": text})

        # Get body text for word count
        body_text = soup.get_text(separator=" ", strip=True)
        word_count = len(body_text.split())

        # Detect content features
        has_tables = bool(soup.find_all("table"))
        has_lists = bool(soup.find_all(["ul", "ol"]))
        has_images = len(soup.find_all("img")) > 2
        has_faq = any(
            "faq" in str(tag.get("class", [])).lower() or
            "faq" in str(tag.get("id", "")).lower()
            for tag in soup.find_all(["div", "section"])
        )

        return {
            "url": url,
            "headings": headings,
            "word_count": word_count,
            "has_tables": has_tables,
            "has_lists": has_lists,
            "has_images": has_images,
            "has_faq": has_faq,
        }

    def build_structure_map(self, page_analyses: list) -> dict:
        """Build a map of common heading structures across top results."""
        h2_counts: dict[str, int] = {}
        h3_counts: dict[str, int] = {}

        for page in page_analyses:
            for h in page.get("headings", []):
                text = h["text"].lower().strip()
                if h["level"] == "h2":
                    h2_counts[text] = h2_counts.get(text, 0) + 1
                elif h["level"] == "h3":
                    h3_counts[text] = h3_counts.get(text, 0) + 1

        # Find common headings (appear in 2+ results)
        common_h2 = sorted(
            [{"text": k, "frequency": v} for k, v in h2_counts.items() if v >= 2],
            key=lambda x: x["frequency"],
            reverse=True,
        )[:15]
        common_h3 = sorted(
            [{"text": k, "frequency": v} for k, v in h3_counts.items() if v >= 2],
            key=lambda x: x["frequency"],
            reverse=True,
        )[:15]

        # Content feature stats
        total = len(page_analyses) or 1
        features = {
            "pct_with_tables": sum(1 for p in page_analyses if p.get("has_tables")) / total * 100,
            "pct_with_lists": sum(1 for p in page_analyses if p.get("has_lists")) / total * 100,
            "pct_with_images": sum(1 for p in page_analyses if p.get("has_images")) / total * 100,
            "pct_with_faq": sum(1 for p in page_analyses if p.get("has_faq")) / total * 100,
        }

        return {
            "common_h2": common_h2,
            "common_h3": common_h3,
            "content_features": features,
        }
