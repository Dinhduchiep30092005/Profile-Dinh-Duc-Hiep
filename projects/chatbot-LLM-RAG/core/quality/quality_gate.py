"""
[7] Quality Gate (Human-in-the-loop)
=======================================
MANDATORY review step before publishing.
SEO dies without human oversight.

Outputs for review:
- Outline
- Draft content
- Intent explanation
- Angle explanation
- EEAT checklist

Flow: Bot outputs → Human reviews → Adjusts → Approves
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

from core.models import Article, ArticleStatus
from core.database import get_session

logger = logging.getLogger(__name__)
console = Console()


class QualityGate:
    """Quality Gate - Human-in-the-loop review system."""

    REVIEW_DIR = Path("reviews")

    def __init__(self):
        self.REVIEW_DIR.mkdir(exist_ok=True)

    def submit_for_review(self, article: dict) -> dict:
        """
        Submit an article for human review.

        Creates:
        1. A review file with outline, draft, and explanations
        2. Updates article status to 'review'
        3. Displays review summary in terminal
        """
        title = article.get("title", "Untitled")
        logger.info(f"[QualityGate] Submitting for review: '{title}'")

        review_package = self._build_review_package(article)
        review_path = self._save_review_file(review_package)
        self._update_status(article, ArticleStatus.REVIEW)
        self._display_review_summary(review_package)

        review_package["review_file"] = str(review_path)
        return review_package

    def approve(self, article_slug: str, reviewer_notes: str = "") -> bool:
        """Approve an article for publishing."""
        session = get_session()
        try:
            article = session.query(Article).filter_by(slug=article_slug).first()
            if not article:
                logger.error(f"[QualityGate] Article not found: {article_slug}")
                return False

            article.status = ArticleStatus.APPROVED
            article.approved_at = datetime.now(timezone.utc)
            article.reviewer_notes = reviewer_notes
            session.commit()

            console.print(f"[green]✓ Approved:[/green] {article.title}")
            logger.info(f"[QualityGate] Approved: '{article.title}'")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"[QualityGate] Approval error: {e}")
            return False
        finally:
            session.close()

    def reject(self, article_slug: str, feedback: str) -> bool:
        """Reject an article back to draft with feedback."""
        session = get_session()
        try:
            article = session.query(Article).filter_by(slug=article_slug).first()
            if not article:
                logger.error(f"[QualityGate] Article not found: {article_slug}")
                return False

            article.status = ArticleStatus.DRAFT
            article.reviewer_notes = feedback
            session.commit()

            console.print(f"[red]✗ Rejected:[/red] {article.title}")
            console.print(f"  Feedback: {feedback}")
            logger.info(f"[QualityGate] Rejected: '{article.title}' — {feedback}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"[QualityGate] Rejection error: {e}")
            return False
        finally:
            session.close()

    def list_pending_reviews(self) -> list[dict]:
        """List all articles awaiting review."""
        session = get_session()
        try:
            articles = session.query(Article).filter_by(
                status=ArticleStatus.REVIEW
            ).all()

            results = []
            table = Table(title="Pending Reviews", box=box.ROUNDED)
            table.add_column("#", style="dim")
            table.add_column("Title", style="bold")
            table.add_column("Keyword")
            table.add_column("Words", justify="right")
            table.add_column("Created")

            for i, a in enumerate(articles, 1):
                table.add_row(
                    str(i),
                    a.title,
                    a.keyword.keyword if a.keyword else "?",
                    str(a.word_count or 0),
                    a.created_at.strftime("%Y-%m-%d") if a.created_at else "?",
                )
                results.append({
                    "id": a.id,
                    "title": a.title,
                    "slug": a.slug,
                    "keyword": a.keyword.keyword if a.keyword else "",
                    "word_count": a.word_count,
                })

            console.print(table)
            return results
        finally:
            session.close()

    # ──────────────────────────────────────────
    # Review Package Building
    # ──────────────────────────────────────────

    def _build_review_package(self, article: dict) -> dict:
        """Build a comprehensive review package."""
        return {
            "title": article.get("title", ""),
            "keyword": article.get("keyword", ""),
            "slug": article.get("slug", ""),
            "meta_description": article.get("meta_description", ""),
            "word_count": article.get("word_count", 0),
            "cluster_role": article.get("cluster_role", ""),
            "target_intent": article.get("target_intent", ""),
            "intent_explanation": article.get("intent_explanation", ""),
            "angle_explanation": article.get("angle_explanation", ""),
            "brand_angle": article.get("brand_angle", ""),
            "eeat": {
                "experience": article.get("eeat_experience", ""),
                "expertise": article.get("eeat_expertise", ""),
                "authority": article.get("eeat_authority", ""),
                "trust": article.get("eeat_trust", ""),
            },
            "outline": article.get("outline", {}),
            "content_preview": article.get("content_markdown", "")[:2000],
            "full_content": article.get("content_markdown", ""),
            "faq_schema": article.get("faq_schema", []),
            "data_tables": article.get("data_tables", []),
            "internal_link_count": article.get("internal_link_count", 0),
            "quality_checklist": self._generate_checklist(article),
        }

    def _generate_checklist(self, article: dict) -> list[dict]:
        """Generate a quality checklist for the reviewer.

        v2: Includes tone variation, duplicate heading, and hierarchy checks.
        """
        import re as _re

        word_count = article.get("word_count", 0)
        content = article.get("content_markdown", "") or ""

        # ── Tone variation analysis ──
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and not p.strip().startswith("#")]
        opening_words = []
        for p in paragraphs:
            first_line = p.split("\n")[0].strip()
            if first_line and not first_line.startswith(("-", "*", "|", "!", "[")):
                first_word = _re.split(r'\s+', first_line)[0].rstrip(",:;")
                if first_word:
                    opening_words.append(first_word.lower())

        word_freq = {}
        for w in opening_words:
            word_freq[w] = word_freq.get(w, 0) + 1

        max_repeat = max(word_freq.values()) if word_freq else 0
        repeated_opener = max(word_freq, key=word_freq.get) if word_freq else ""
        tone_ok = max_repeat < 4
        tone_detail = (
            f"OK — max opener repeat: {max_repeat}"
            if tone_ok
            else f"REPETITIVE — '{repeated_opener}' opens {max_repeat} paragraphs"
        )

        # ── Duplicate heading detection ──
        headings = _re.findall(r'^(#{1,6})\s+(.+)$', content, _re.MULTILINE)
        heading_texts = [h[1].strip().lower() for h in headings]
        seen = {}
        duplicates = []
        for h in heading_texts:
            seen[h] = seen.get(h, 0) + 1
            if seen[h] == 2:
                duplicates.append(h)
        dup_ok = len(duplicates) == 0
        dup_detail = (
            "No duplicates"
            if dup_ok
            else f"Duplicates: {', '.join(duplicates[:3])}"
        )

        # ── Heading hierarchy check (H2 > H3 > H4, no skips) ──
        hierarchy_issues = []
        prev_level = 1
        for match in headings:
            level = len(match[0])
            if level > prev_level + 1:
                hierarchy_issues.append(f"H{prev_level}→H{level} skip before '{match[1][:40]}'")
            prev_level = level
        hierarchy_ok = len(hierarchy_issues) == 0
        hierarchy_detail = (
            "Proper hierarchy"
            if hierarchy_ok
            else f"{len(hierarchy_issues)} issue(s): {hierarchy_issues[0]}"
        )

        checks = [
            {
                "item": "Word Count",
                "status": "pass" if 1500 <= word_count <= 3000 else "warn",
                "detail": f"{word_count} words",
            },
            {
                "item": "Title Unique Angle",
                "status": "pass" if article.get("brand_angle") else "fail",
                "detail": article.get("brand_angle", "No brand angle")[:100],
            },
            {
                "item": "Meta Description",
                "status": "pass" if 140 <= len(article.get("meta_description", "")) <= 165 else "warn",
                "detail": f"{len(article.get('meta_description', ''))} chars",
            },
            {
                "item": "EEAT Experience",
                "status": "pass" if article.get("eeat_experience") else "fail",
                "detail": "Present" if article.get("eeat_experience") else "Missing",
            },
            {
                "item": "EEAT Expertise",
                "status": "pass" if article.get("eeat_expertise") else "fail",
                "detail": "Present" if article.get("eeat_expertise") else "Missing",
            },
            {
                "item": "FAQ Schema",
                "status": "pass" if article.get("faq_schema") else "warn",
                "detail": f"{len(article.get('faq_schema', []))} FAQs",
            },
            {
                "item": "Internal Links",
                "status": "pass" if article.get("internal_link_count", 0) >= 2 else "warn",
                "detail": f"{article.get('internal_link_count', 0)} links",
            },
            {
                "item": "Data Tables",
                "status": "pass" if article.get("data_tables") else "info",
                "detail": f"{len(article.get('data_tables', []))} tables",
            },
            # ── v2 checks ──
            {
                "item": "Tone Variation",
                "status": "pass" if tone_ok else "warn",
                "detail": tone_detail,
            },
            {
                "item": "Duplicate Headings",
                "status": "pass" if dup_ok else "fail",
                "detail": dup_detail,
            },
            {
                "item": "Heading Hierarchy",
                "status": "pass" if hierarchy_ok else "warn",
                "detail": hierarchy_detail,
            },
        ]
        return checks

    # ──────────────────────────────────────────
    # File Output
    # ──────────────────────────────────────────

    def _save_review_file(self, review: dict) -> Path:
        """Save review package as a markdown file."""
        filename = f"{review['slug']}_review.md"
        path = self.REVIEW_DIR / filename

        lines = [
            f"# REVIEW: {review['title']}",
            "",
            "---",
            "",
            "## Article Info",
            f"- **Keyword:** {review['keyword']}",
            f"- **Intent:** {review['target_intent']}",
            f"- **Cluster Role:** {review['cluster_role']}",
            f"- **Word Count:** {review['word_count']}",
            f"- **Meta Description:** {review['meta_description']}",
            "",
            "## Intent Explanation",
            review['intent_explanation'] or "_Not provided_",
            "",
            "## Brand Angle Explanation",
            review['angle_explanation'] or "_Not provided_",
            "",
            "## EEAT Signals",
            f"- **Experience:** {review['eeat']['experience'] or '_Missing_'}",
            f"- **Expertise:** {review['eeat']['expertise'] or '_Missing_'}",
            f"- **Authority:** {review['eeat']['authority'] or '_Missing_'}",
            f"- **Trust:** {review['eeat']['trust'] or '_Missing_'}",
            "",
            "## Quality Checklist",
        ]

        for check in review.get("quality_checklist", []):
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}.get(
                check["status"], "❓"
            )
            lines.append(f"- {icon} **{check['item']}:** {check['detail']}")

        lines.extend([
            "",
            "## FAQ Schema",
            "",
        ])
        for faq in review.get("faq_schema", []):
            lines.append(f"**Q: {faq.get('question', '')}**")
            lines.append(f"A: {faq.get('answer', '')}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Full Content Draft",
            "",
            review['full_content'],
        ])

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[QualityGate] Review file saved: {path}")
        return path

    # ──────────────────────────────────────────
    # Console Display
    # ──────────────────────────────────────────

    def _display_review_summary(self, review: dict):
        """Display review summary in rich console."""
        console.print()
        console.print(Panel(
            f"[bold]{review['title']}[/bold]\n"
            f"Keyword: {review['keyword']} | Intent: {review['target_intent']} | "
            f"Words: {review['word_count']}",
            title="📝 Article Review",
            border_style="blue",
        ))

        table = Table(title="Quality Checklist", box=box.SIMPLE)
        table.add_column("Check", style="bold")
        table.add_column("Status")
        table.add_column("Detail")

        for check in review.get("quality_checklist", []):
            status_style = {
                "pass": "[green]PASS[/green]",
                "warn": "[yellow]WARN[/yellow]",
                "fail": "[red]FAIL[/red]",
                "info": "[blue]INFO[/blue]",
            }.get(check["status"], check["status"])
            table.add_row(check["item"], status_style, check["detail"])

        console.print(table)

        if review.get("brand_angle"):
            console.print(Panel(
                review["brand_angle"],
                title="Brand Angle",
                border_style="green",
            ))

        console.print(
            f"\n[bold]Review file:[/bold] {review.get('review_file', 'N/A')}\n"
        )

    def _update_status(self, article: dict, status: ArticleStatus):
        """Update article status in database."""
        session = get_session()
        try:
            slug = article.get("slug", "")
            db_article = session.query(Article).filter_by(slug=slug).first()
            if db_article:
                db_article.status = status
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"[QualityGate] Status update error: {e}")
        finally:
            session.close()

    # ──────────────────────────────────────────
    # Automated Quality Check (non-blocking)
    # ──────────────────────────────────────────

    def auto_check(self, result: dict) -> dict:
        """
        Automated quality check on a freshly generated article result dict.

        Non-blocking — the pipeline continues regardless of pass/fail.
        The caller may log warnings but will not abort generation.

        Checks:
          - Word count ≥ 800
          - Global EEAT avg score ≥ 6.0
          - SERP coverage ≥ 60 %
          - No duplicate paragraphs
          - At least 1 internal link

        Returns:
            dict:
                passed (bool)
                score  (float, 0–10)
                issues (list[str])
                word_count, eeat_avg, serp_coverage
        """
        issues: list[str] = []
        scores: list[float] = []

        # ── Word count ──
        word_count: int = result.get("word_count", 0)
        if word_count < 800:
            issues.append(f"Word count thấp: {word_count} (tối thiểu 800)")
            scores.append(3.0)
        elif word_count < 1200:
            scores.append(7.0)
        else:
            scores.append(10.0)

        # ── EEAT score ──
        global_eeat: dict = result.get("global_eeat", {})
        eeat_avg: float = float(global_eeat.get("avg_score", 0))
        if eeat_avg < 6.0:
            issues.append(f"EEAT score thấp: {eeat_avg:.1f}/10 (tối thiểu 6.0)")
            scores.append(max(eeat_avg, 0.0))
        else:
            scores.append(min(eeat_avg, 10.0))

        # ── SERP coverage ──
        serp_gap: dict = result.get("serp_gap_report", {})
        coverage: float = float(serp_gap.get("coverage_score", 100))
        if coverage < 60:
            issues.append(f"SERP coverage thấp: {coverage:.0f}% (tối thiểu 60%)")
            scores.append(coverage / 10)
        else:
            scores.append(10.0)

        # ── Duplicate paragraphs ──
        dup_report: dict = result.get("duplicate_report", {})
        if dup_report.get("duplicates_found"):
            dup_count = dup_report.get("count", 0)
            issues.append(f"{dup_count} đoạn trùng lặp còn sót")
            scores.append(6.0)
        else:
            scores.append(10.0)

        # ── Internal links ──
        internal_links = result.get("internal_links", [])
        if not internal_links:
            issues.append("Không có internal link nào được inject")
            scores.append(7.0)
        else:
            scores.append(10.0)

        score = round(sum(scores) / len(scores), 1) if scores else 0.0
        # Pass: average ≥ 6.5 and no hard-fail issues (word count / EEAT)
        hard_fail = any("thấp" in i for i in issues)
        passed = score >= 6.5 and not hard_fail

        logger.info(
            f"[QualityGate] auto_check — score={score}/10 | "
            f"passed={passed} | issues={len(issues)}"
        )

        return {
            "passed": passed,
            "score": score,
            "issues": issues,
            "word_count": word_count,
            "eeat_avg": eeat_avg,
            "serp_coverage": coverage,
        }
