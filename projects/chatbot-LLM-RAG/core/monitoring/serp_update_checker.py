"""
SERP Update Checker - Detects changes in the SERP landscape.
Compares current vs previous SERP state for an article's keyword.
"""

import logging
from typing import Optional

from core.models import Article, RankSnapshot

logger = logging.getLogger(__name__)


class SerpUpdateChecker:
    """Detects changes in SERP landscape between monitoring intervals."""

    def detect_changes(
        self,
        article: Article,
        current_serp: dict,
        prev_snapshot: Optional[RankSnapshot],
    ) -> dict:
        """Detect changes in the SERP landscape since last check."""

        changes = {
            "word_count_shift": 0,
            "new_competitors": [],
            "new_serp_features": [],
            "structure_changes": [],
        }

        # Word count change
        prev_avg = article.keyword.serp_avg_word_count if article.keyword else 0
        current_avg = current_serp.get("avg_word_count", 0)
        if prev_avg and current_avg:
            changes["word_count_shift"] = current_avg - prev_avg

        # New competitors (domains that weren't there before)
        if prev_snapshot and prev_snapshot.competitor_changes:
            prev_domains = set(
                prev_snapshot.competitor_changes.get("domains", [])
            )
            current_domains = set(
                r.get("domain", "") for r in current_serp.get("serp_results", [])
            )
            changes["new_competitors"] = list(current_domains - prev_domains)

        # Store current domains for next comparison
        changes["domains"] = [
            r.get("domain", "") for r in current_serp.get("serp_results", [])
        ]

        # New headings/sections in top results
        new_h2s = current_serp.get("structure_map", {}).get("common_h2", [])
        changes["new_serp_features"] = [h["text"] for h in new_h2s[:5]]

        return changes
