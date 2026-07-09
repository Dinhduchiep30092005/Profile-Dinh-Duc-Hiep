"""
[9] Performance Analyzer — Google Search Console Feedback Loop
================================================================
Reads real GSC data to close the feedback loop with Google.

Capabilities:
- Fetch GSC query/page performance data (clicks, impressions, CTR, position)
- Detect pages with high impressions but low clicks (CTR problem)
- Identify keywords where we rank but underperform on CTR
- Suggest title/meta rewrites using LLM based on actual SERP performance
- Prioritize optimization opportunities by impressions × CTR gap

Core philosophy: SEO bền = feedback từ Google thật.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

from config import settings
from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.models import Article, Keyword, ArticleStatus
from core.database import get_session

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Google Search Console Performance Analyzer.

    Reads real click/impression data from GSC and generates
    actionable optimization recommendations.
    """

    # Thresholds
    LOW_CTR_THRESHOLD = 0.03        # CTR < 3% is considered low
    HIGH_IMPRESSION_MIN = 100       # At least 100 impressions to be significant
    OPPORTUNITY_CTR_GAP = 0.05      # Expected CTR minus actual CTR threshold

    # Expected CTR by position (industry benchmarks)
    EXPECTED_CTR_BY_POSITION = {
        1: 0.28, 2: 0.15, 3: 0.11, 4: 0.08, 5: 0.07,
        6: 0.05, 7: 0.04, 8: 0.03, 9: 0.03, 10: 0.02,
    }

    def __init__(self):
        self._service = None

    # ──────────────────────────────────────────
    # GSC API Connection
    # ──────────────────────────────────────────

    def _get_service(self):
        """Lazily build authorized GSC service client."""
        if self._service is None:
            creds = Credentials.from_service_account_file(
                settings.gsc.credentials_path,
                scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
            )
            self._service = build("searchconsole", "v1", credentials=creds)
        return self._service

    # ──────────────────────────────────────────
    # Data Fetching
    # ──────────────────────────────────────────

    def fetch_summary(self, days: int = 90) -> dict:
        """
        Fetch aggregate GSC totals (no dimensions) for accurate site-wide stats.

        This returns the TRUE totals (clicks, impressions, CTR, position)
        without being limited by row_limit constraints.

        Returns:
            Dict with total_clicks, total_impressions, avg_ctr, avg_position
        """
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)

        request_body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": [],  # No dimensions → single aggregated row
            "dataState": "final",
        }

        try:
            service = self._get_service()
            response = (
                service.searchanalytics()
                .query(siteUrl=settings.gsc.site_url, body=request_body)
                .execute()
            )

            rows = response.get("rows", [])
            if rows:
                row = rows[0]
                summary = {
                    "total_clicks": int(row["clicks"]),
                    "total_impressions": int(row["impressions"]),
                    "avg_ctr": row["ctr"],
                    "avg_position": row["position"],
                }
                logger.info(
                    f"[GSC] Summary ({days}d): "
                    f"{summary['total_clicks']:,} clicks, "
                    f"{summary['total_impressions']:,} impressions, "
                    f"CTR {summary['avg_ctr']:.2%}, "
                    f"Pos {summary['avg_position']:.1f}"
                )
                return summary
            else:
                logger.warning("[GSC] No summary data returned")
                return {
                    "total_clicks": 0,
                    "total_impressions": 0,
                    "avg_ctr": 0,
                    "avg_position": 0,
                }

        except Exception as e:
            logger.error(f"[GSC] Summary API error: {e}")
            return {
                "total_clicks": 0,
                "total_impressions": 0,
                "avg_ctr": 0,
                "avg_position": 0,
            }

    def fetch_performance(
        self,
        days: int = 90,
        row_limit: int = 25000,
        dimensions: Optional[list[str]] = None,
        fetch_all: bool = True,
    ) -> list[dict]:
        """
        Fetch GSC Search Analytics data with automatic pagination.

        Args:
            days: Number of days to look back (max 16 months)
            row_limit: Max rows per API request (max 25000)
            dimensions: Dimensions to group by (default: query + page)
            fetch_all: If True, paginate to fetch ALL rows (not just first page)

        Returns:
            List of row dicts with keys, clicks, impressions, ctr, position
        """
        if dimensions is None:
            dimensions = ["query", "page"]

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)

        all_rows = []
        start_row = 0

        try:
            service = self._get_service()

            while True:
                request_body = {
                    "startDate": start_date.isoformat(),
                    "endDate": end_date.isoformat(),
                    "dimensions": dimensions,
                    "rowLimit": row_limit,
                    "startRow": start_row,
                    "dataState": "final",
                }

                response = (
                    service.searchanalytics()
                    .query(siteUrl=settings.gsc.site_url, body=request_body)
                    .execute()
                )

                rows = response.get("rows", [])
                if not rows:
                    break

                for row in rows:
                    all_rows.append({
                        "keys": row["keys"],
                        "query": row["keys"][0] if len(row["keys"]) > 0 else "",
                        "page": row["keys"][1] if len(row["keys"]) > 1 else "",
                        "clicks": row["clicks"],
                        "impressions": row["impressions"],
                        "ctr": row["ctr"],
                        "position": row["position"],
                    })

                logger.info(
                    f"[GSC] Fetched {len(rows)} rows (startRow={start_row}, "
                    f"total so far: {len(all_rows)})"
                )

                # If we got fewer rows than the limit, we've reached the end
                if len(rows) < row_limit or not fetch_all:
                    break

                start_row += len(rows)

            logger.info(f"[GSC] Total fetched: {len(all_rows)} rows ({days}d window)")
            return all_rows

        except Exception as e:
            logger.error(f"[GSC] API error: {e}")
            return all_rows if all_rows else []

    def fetch_page_performance(
        self,
        page_url: str,
        days: int = 90,
    ) -> list[dict]:
        """Fetch GSC data filtered to a specific page."""
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)

        request_body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": ["query"],
            "dimensionFilterGroups": [
                {
                    "filters": [
                        {
                            "dimension": "page",
                            "operator": "equals",
                            "expression": page_url,
                        }
                    ]
                }
            ],
            "rowLimit": 500,
        }

        try:
            service = self._get_service()
            response = (
                service.searchanalytics()
                .query(siteUrl=settings.gsc.site_url, body=request_body)
                .execute()
            )

            rows = response.get("rows", [])
            logger.info(f"[GSC] Page '{page_url}': {len(rows)} queries")

            return [
                {
                    "query": row["keys"][0],
                    "clicks": row["clicks"],
                    "impressions": row["impressions"],
                    "ctr": row["ctr"],
                    "position": row["position"],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"[GSC] Page fetch error: {e}")
            return []

    # ──────────────────────────────────────────
    # Analysis: Low CTR Detection
    # ──────────────────────────────────────────

    def detect_low_ctr(
        self,
        data: Optional[list[dict]] = None,
        days: int = 90,
    ) -> list[dict]:
        """
        Detect queries/pages with abnormally low CTR.

        Finds pages where:
        - Impressions >= HIGH_IMPRESSION_MIN
        - CTR < LOW_CTR_THRESHOLD
        - Sorted by impressions (highest opportunity first)

        Returns:
            List of low-CTR entries with opportunity_score
        """
        if data is None:
            data = self.fetch_performance(days=days)

        low_ctr = []
        for row in data:
            if (
                row["impressions"] >= self.HIGH_IMPRESSION_MIN
                and row["ctr"] < self.LOW_CTR_THRESHOLD
            ):
                # Calculate opportunity: how many clicks we'd gain at expected CTR
                position_bucket = min(int(row["position"]), 10)
                expected_ctr = self.EXPECTED_CTR_BY_POSITION.get(position_bucket, 0.02)
                ctr_gap = expected_ctr - row["ctr"]
                potential_clicks = row["impressions"] * ctr_gap

                low_ctr.append({
                    **row,
                    "expected_ctr": round(expected_ctr, 4),
                    "ctr_gap": round(ctr_gap, 4),
                    "potential_clicks": round(potential_clicks, 1),
                    "opportunity_score": round(
                        potential_clicks * (1 + ctr_gap), 1
                    ),
                })

        # Sort by opportunity score descending
        low_ctr.sort(key=lambda x: x["opportunity_score"], reverse=True)

        logger.info(f"[GSC] Found {len(low_ctr)} low-CTR opportunities")
        return low_ctr

    # ──────────────────────────────────────────
    # Analysis: High Impressions / Low Clicks
    # ──────────────────────────────────────────

    def detect_impression_click_gap(
        self,
        data: Optional[list[dict]] = None,
        days: int = 90,
        min_impressions: int = 200,
        max_clicks: int = 10,
    ) -> list[dict]:
        """
        Detect queries with high impressions but very low clicks.

        These are the biggest quick-win opportunities:
        users SEE us but DON'T CLICK.

        Args:
            data: Pre-fetched data or None to fetch
            days: Lookback period
            min_impressions: Minimum impressions threshold
            max_clicks: Maximum clicks to qualify as 'low'

        Returns:
            Sorted list of gap entries
        """
        if data is None:
            data = self.fetch_performance(days=days)

        gaps = []
        for row in data:
            if row["impressions"] >= min_impressions and row["clicks"] <= max_clicks:
                waste_ratio = row["impressions"] / max(row["clicks"], 1)
                position_bucket = min(int(row["position"]), 10)
                expected_ctr = self.EXPECTED_CTR_BY_POSITION.get(position_bucket, 0.02)

                gaps.append({
                    **row,
                    "waste_ratio": round(waste_ratio, 1),
                    "expected_ctr": round(expected_ctr, 4),
                    "expected_clicks": round(row["impressions"] * expected_ctr, 1),
                    "missed_clicks": round(
                        row["impressions"] * expected_ctr - row["clicks"], 1
                    ),
                })

        gaps.sort(key=lambda x: x["missed_clicks"], reverse=True)

        logger.info(f"[GSC] Found {len(gaps)} high-impression/low-click gaps")
        return gaps

    # ──────────────────────────────────────────
    # Title Rewrite Suggestions
    # ──────────────────────────────────────────

    def suggest_title_rewrites(
        self,
        low_ctr_entries: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        """
        Use LLM to suggest title rewrites for low-CTR pages.

        Takes the top N low-CTR entries and generates
        optimized title + meta description alternatives.

        Args:
            low_ctr_entries: Output from detect_low_ctr() or detect_impression_click_gap()
            top_n: How many entries to process

        Returns:
            List of rewrite suggestions
        """
        entries = low_ctr_entries[:top_n]
        if not entries:
            return []

        logger.info(f"[GSC] Generating title rewrites for {len(entries)} pages")

        # Build context for LLM
        entries_text = "\n".join(
            f"{i+1}. Query: \"{e['query']}\"\n"
            f"   Page: {e.get('page', 'N/A')}\n"
            f"   Position: {e['position']:.1f} | Impressions: {e['impressions']} | "
            f"Clicks: {e['clicks']} | CTR: {e['ctr']:.2%}\n"
            f"   Expected CTR: {e.get('expected_ctr', 0):.2%} | "
            f"Missed clicks: {e.get('missed_clicks', e.get('potential_clicks', 0)):.0f}"
            for i, e in enumerate(entries)
        )

        user_prompt = f"""Here are pages with high impressions but poor CTR.
Users see us in Google but don't click.

{entries_text}

For each entry, suggest an optimized title and meta description
that would increase CTR while keeping SEO relevance."""

        result = llm.chat_json(
            PromptTemplates.TITLE_REWRITE_SYSTEM,
            user_prompt,
            temperature=0.6,
            max_tokens=4096,
        )

        suggestions = result.get("rewrites", [])

        # Merge suggestions back with original data
        merged = []
        for i, entry in enumerate(entries):
            suggestion = suggestions[i] if i < len(suggestions) else {}
            merged.append({
                "query": entry["query"],
                "page": entry.get("page", ""),
                "current_metrics": {
                    "position": entry["position"],
                    "impressions": entry["impressions"],
                    "clicks": entry["clicks"],
                    "ctr": entry["ctr"],
                },
                "new_title": suggestion.get("new_title", ""),
                "new_meta_description": suggestion.get("new_meta_description", ""),
                "reasoning": suggestion.get("reasoning", ""),
                "expected_ctr_lift": suggestion.get("expected_ctr_lift", ""),
                "priority": suggestion.get("priority", "medium"),
            })

        logger.info(f"[GSC] Generated {len(merged)} rewrite suggestions")
        return merged

    # ──────────────────────────────────────────
    # Full Analysis Pipeline
    # ──────────────────────────────────────────

    def full_analysis(self, days: int = 90, top_n: int = 10) -> dict:
        """
        Run the complete GSC performance analysis pipeline.

        1. Fetch GSC data
        2. Detect low-CTR pages
        3. Detect high-impression/low-click gaps
        4. Suggest title rewrites for top opportunities

        Returns:
            Complete analysis report
        """
        logger.info(f"[GSC] Starting full analysis ({days}d window)")

        # Step 1: Fetch accurate site-wide totals (no dimensions)
        summary = self.fetch_summary(days=days)

        # Step 1b: Fetch detailed row-level data for analysis
        data = self.fetch_performance(days=days)

        if not data:
            logger.warning("[GSC] No data returned from GSC")
            return {
                "status": "no_data",
                "message": "No GSC data available. Check credentials and site verification.",
            }

        # Step 2: Use accurate totals from summary API (not from limited rows)
        total_clicks = summary["total_clicks"]
        total_impressions = summary["total_impressions"]
        avg_ctr = summary["avg_ctr"]
        avg_position = summary["avg_position"]

        # Step 3: Detect problems
        low_ctr = self.detect_low_ctr(data=data)
        impression_gaps = self.detect_impression_click_gap(data=data)

        # Step 4: Generate rewrites for top opportunities
        # Merge and deduplicate top entries from both analyses
        seen_queries = set()
        top_opportunities = []
        for entry in low_ctr + impression_gaps:
            key = (entry["query"], entry.get("page", ""))
            if key not in seen_queries:
                seen_queries.add(key)
                top_opportunities.append(entry)

        top_opportunities.sort(
            key=lambda x: x.get("opportunity_score", x.get("missed_clicks", 0)),
            reverse=True,
        )

        # Only generate rewrites if OpenAI API key is configured
        rewrites = []
        if settings.openai.api_key:
            try:
                rewrites = self.suggest_title_rewrites(top_opportunities[:top_n], top_n=top_n)
            except Exception as e:
                logger.warning(f"[GSC] Title rewrite generation failed: {e}")
        else:
            logger.info("[GSC] Skipping title rewrites — no OpenAI API key configured")

        # Step 5: Match with DB articles
        self._match_with_articles(low_ctr)

        # Count unique queries (not query+page pairs)
        unique_queries = len(set(r["query"] for r in data))

        report = {
            "period_days": days,
            "summary": {
                "total_queries": unique_queries,
                "total_rows": len(data),
                "total_clicks": total_clicks,
                "total_impressions": total_impressions,
                "avg_ctr": round(avg_ctr, 4),
                "avg_position": round(avg_position, 1),
            },
            "low_ctr_pages": low_ctr[:20],
            "impression_click_gaps": impression_gaps[:20],
            "title_rewrite_suggestions": rewrites,
            "quick_wins": len([
                r for r in rewrites if r.get("priority") == "high"
            ]),
        }

        logger.info(
            f"[GSC] Analysis complete: "
            f"{len(low_ctr)} low-CTR | "
            f"{len(impression_gaps)} impression gaps | "
            f"{len(rewrites)} rewrite suggestions"
        )

        return report

    # ──────────────────────────────────────────
    # Database Matching
    # ──────────────────────────────────────────

    def _match_with_articles(self, entries: list[dict]):
        """
        Match low-CTR entries with articles in our database.
        Flag matched articles as needing optimization.
        """
        session = get_session()
        try:
            for entry in entries[:50]:
                page_url = entry.get("page", "")
                if not page_url:
                    continue

                article = session.query(Article).filter(
                    Article.published_url == page_url
                ).first()

                if article:
                    entry["article_id"] = article.id
                    entry["article_title"] = article.title
                    entry["article_slug"] = article.slug

                    # Flag for update if CTR gap is significant
                    if entry.get("ctr_gap", 0) > self.OPPORTUNITY_CTR_GAP:
                        if article.status != ArticleStatus.NEEDS_UPDATE:
                            article.status = ArticleStatus.NEEDS_UPDATE
                            logger.info(
                                f"[GSC] Flagged '{article.title}' for title optimization "
                                f"(CTR gap: {entry['ctr_gap']:.2%})"
                            )

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"[GSC] DB match error: {e}")
        finally:
            session.close()
