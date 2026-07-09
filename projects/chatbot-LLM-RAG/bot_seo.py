"""
SEO Bot - Main CLI Entry Point
================================
Orchestrates all 8 engines in the SEO content pipeline.

Usage:
    python bot_seo.py run <keyword>          Full pipeline: SERP → Intent → Cluster → Brand → Content → Links → Review
    python bot_seo.py analyze <keyword>      SERP analysis only
    python bot_seo.py cluster <keyword>      Build topic cluster
    python bot_seo.py write <keyword>        Generate content for a keyword
    python bot_seo.py review                 List pending reviews
    python bot_seo.py approve <slug>         Approve an article
    python bot_seo.py reject <slug>          Reject an article with feedback
    python bot_seo.py monitor                Check rankings for all published articles
    python bot_seo.py update-queue           Show articles needing update
    python bot_seo.py init-db                Initialize database tables
    python bot_seo.py configure-brand        Interactive brand configuration
"""

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich import box

from config import settings, BrandConfig
from core.database import init_db
from core.serp.serp_engine import SerpEngine
from core.intent.intent_mapper import IntentMapper
from core.cluster.cluster_architect import ClusterArchitect
from core.content.brand_angle import BrandAngleGenerator
from core.content.eeat_builder import EEATContentBuilder
from core.linking.link_automation import LinkAutomation
from core.quality.quality_gate import QualityGate
from core.monitoring.rank_monitor import RankMonitor
from core.analytics.performance_analyzer import PerformanceAnalyzer

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("seo_bot")
console = Console()


# ──────────────────────────────────────────────
# CLI Group
# ──────────────────────────────────────────────

@click.group()
@click.version_option(version="1.0.0", prog_name="SEO Bot")
def cli():
    """🤖 SEO Bot - AI-Powered Content Engine"""
    pass


# ──────────────────────────────────────────────
# [FULL PIPELINE] run
# ──────────────────────────────────────────────

@cli.command()
@click.argument("keyword")
@click.option("--clusters", "-c", default=7, help="Number of cluster articles (5-10)")
@click.option("--brand-config", "-b", type=click.Path(exists=True), help="Brand config JSON file")
@click.option("--language", "-l", default="vi", help="Language for SERP crawl (vi, en, en-us, fr, de, ja, ko...)")
def run(keyword: str, clusters: int, brand_config: str, language: str):
    """Run the full SEO content pipeline for a keyword."""

    console.print(Panel(
        f"[bold]Starting Full Pipeline[/bold]\n"
        f"Keyword: {keyword}\n"
        f"Language: {language}\n"
        f"Cluster Size: 1 pillar + {clusters} articles",
        title="🚀 SEO Bot Pipeline",
        border_style="green",
    ))

    # Load brand config if provided
    brand_gen = BrandAngleGenerator()
    if brand_config:
        with open(brand_config, "r") as f:
            bc = json.load(f)
        brand_gen.configure_brand(**bc)
        console.print(f"[green]✓[/green] Brand configured: {bc.get('product_name', '?')}")

    # ── Stage 1: SERP Intelligence ──
    console.print("\n[bold blue]█ Stage 1/7: SERP Intelligence Engine[/bold blue]")
    serp = SerpEngine()
    serp_analysis = serp.analyze(keyword, language=language)
    _display_serp_summary(serp_analysis)

    # ── Stage 2: Intent Mapping ──
    console.print("\n[bold blue]█ Stage 2/7: Intent Mapping & Gap Detection[/bold blue]")
    mapper = IntentMapper()
    intent_analysis = mapper.analyze(keyword, serp_analysis)
    _display_intent_summary(intent_analysis)

    # ── Stage 3: Topic Cluster ──
    console.print("\n[bold blue]█ Stage 3/7: Topic Cluster Architect[/bold blue]")
    architect = ClusterArchitect()
    cluster_blueprint = architect.design_cluster(
        keyword, serp_analysis, intent_analysis, num_clusters=clusters
    )
    _display_cluster_summary(cluster_blueprint)

    # ── Stage 4: Brand Angles ──
    console.print("\n[bold blue]█ Stage 4/7: Brand Angle Generator[/bold blue]")
    branded_blueprint = brand_gen.generate_angles(keyword, cluster_blueprint, intent_analysis)
    _display_brand_summary(branded_blueprint)

    # ── Stage 5: EEAT Content Building ──
    console.print("\n[bold blue]█ Stage 5/7: EEAT Content Builder[/bold blue]")
    builder = EEATContentBuilder()

    articles = []

    # Build pillar article
    pillar_spec = branded_blueprint.get("pillar", {})
    console.print(f"  Writing pillar: {pillar_spec.get('branded_title', pillar_spec.get('title', keyword))}")
    pillar_article = builder.build_article(
        keyword=keyword,
        article_spec=pillar_spec,
        brand_data=branded_blueprint,
        serp_analysis=serp_analysis,
        cluster_role="pillar",
    )
    articles.append(pillar_article)

    # Build cluster articles
    for i, cluster_spec in enumerate(branded_blueprint.get("clusters", []), 1):
        cluster_kw = cluster_spec.get("keyword", keyword)
        console.print(f"  Writing cluster {i}: {cluster_spec.get('branded_title', cluster_spec.get('title', cluster_kw))}")
        cluster_article = builder.build_article(
            keyword=cluster_kw,
            article_spec=cluster_spec,
            brand_data=branded_blueprint,
            serp_analysis=serp_analysis,
            cluster_role="cluster",
        )
        articles.append(cluster_article)

    console.print(f"[green]✓[/green] Built {len(articles)} articles")

    # ── Stage 6: Internal Links ──
    console.print("\n[bold blue]█ Stage 6/7: Internal Link Automation[/bold blue]")
    linker = LinkAutomation()
    linked_articles = linker.process_cluster_links(
        articles, branded_blueprint.get("linking_map", {})
    )
    console.print(f"[green]✓[/green] Internal links processed")

    # ── Stage 7: Quality Gate ──
    console.print("\n[bold blue]█ Stage 7/7: Quality Gate (Review)[/bold blue]")
    gate = QualityGate()
    for article in linked_articles:
        gate.submit_for_review(article)

    console.print(Panel(
        f"[bold green]Pipeline Complete![/bold green]\n\n"
        f"• {len(linked_articles)} articles generated\n"
        f"• Review files saved to ./reviews/\n"
        f"• Run [bold]python bot_seo.py review[/bold] to see pending reviews\n"
        f"• Run [bold]python bot_seo.py approve <slug>[/bold] to approve",
        title="✅ Done",
        border_style="green",
    ))


# ──────────────────────────────────────────────
# Individual Commands
# ──────────────────────────────────────────────

@cli.command()
@click.argument("keyword")
@click.option("--language", "-l", default="vi", help="Language for SERP crawl")
def analyze(keyword: str, language: str):
    """Analyze SERP data for a keyword."""
    console.print(f"[bold]Analyzing SERP for:[/bold] {keyword} (language={language})")
    serp = SerpEngine()
    result = serp.analyze(keyword, language=language)
    _display_serp_summary(result)

    # Also run intent mapping
    mapper = IntentMapper()
    intent = mapper.analyze(keyword, result)
    _display_intent_summary(intent)


@cli.command()
@click.argument("keyword")
@click.option("--clusters", "-c", default=7, help="Number of cluster articles")
@click.option("--language", "-l", default="vi", help="Language for SERP crawl")
def cluster(keyword: str, clusters: int, language: str):
    """Build a topic cluster for a keyword."""
    console.print(f"[bold]Building topic cluster for:[/bold] {keyword} (language={language})")

    serp = SerpEngine()
    serp_analysis = serp.analyze(keyword, language=language)

    mapper = IntentMapper()
    intent = mapper.analyze(keyword, serp_analysis)

    architect = ClusterArchitect()
    blueprint = architect.design_cluster(keyword, serp_analysis, intent, num_clusters=clusters)
    _display_cluster_summary(blueprint)


@cli.command()
@click.argument("keyword")
@click.option("--brand-config", "-b", type=click.Path(exists=True), help="Brand config JSON")
@click.option("--language", "-l", default="vi", help="Language for SERP crawl")
def write(keyword: str, brand_config: str, language: str):
    """Generate a single article for a keyword."""
    console.print(f"[bold]Writing article for:[/bold] {keyword} (language={language})")

    brand_gen = BrandAngleGenerator()
    if brand_config:
        with open(brand_config, "r") as f:
            bc = json.load(f)
        brand_gen.configure_brand(**bc)

    serp = SerpEngine()
    serp_analysis = serp.analyze(keyword, language=language)

    mapper = IntentMapper()
    intent = mapper.analyze(keyword, serp_analysis)

    # Generate brand angle for single article
    article_spec = {
        "keyword": keyword,
        "title": keyword.title(),
        "target_intent": intent.get("intent_type", "informational"),
        "content_type": "article",
    }

    if intent.get("suggested_angles"):
        best_angle = intent["suggested_angles"][0]
        article_spec["title"] = best_angle.get("title_suggestion", keyword.title())
        article_spec["brand_angle"] = best_angle.get("angle_description", "")

    branded = brand_gen._brand_angle_for_article(
        keyword=keyword,
        original_title=article_spec["title"],
        content_type=article_spec["content_type"],
        target_intent=article_spec["target_intent"],
        is_pillar=False,
    )
    article_spec.update(branded)

    builder = EEATContentBuilder()
    article = builder.build_article(
        keyword=keyword,
        article_spec=article_spec,
        brand_data={"brand_strategy": {}},
        serp_analysis=serp_analysis,
    )

    gate = QualityGate()
    gate.submit_for_review(article)


@cli.command()
def review():
    """List all articles pending review."""
    gate = QualityGate()
    gate.list_pending_reviews()


@cli.command()
@click.argument("slug")
@click.option("--notes", "-n", default="", help="Reviewer notes")
def approve(slug: str, notes: str):
    """Approve an article for publishing."""
    gate = QualityGate()
    gate.approve(slug, notes)


@cli.command()
@click.argument("slug")
@click.option("--feedback", "-f", required=True, help="Rejection feedback")
def reject(slug: str, feedback: str):
    """Reject an article back to draft."""
    gate = QualityGate()
    gate.reject(slug, feedback)


@cli.command()
def monitor():
    """Check rankings for all published articles."""
    console.print("[bold]Checking rankings for published articles...[/bold]")
    rm = RankMonitor()
    results = rm.check_all_published()

    if not results:
        console.print("[dim]No articles to check or all recently checked.[/dim]")
        return

    table = Table(title="Rank Monitor Results", box=box.ROUNDED)
    table.add_column("Article", style="bold")
    table.add_column("Keyword")
    table.add_column("Position", justify="center")
    table.add_column("Change", justify="center")
    table.add_column("Update?", justify="center")

    for r in results:
        change = r.get("position_change")
        change_str = ""
        if change is not None:
            if change < 0:
                change_str = f"[green]↑{abs(change)}[/green]"
            elif change > 0:
                change_str = f"[red]↓{change}[/red]"
            else:
                change_str = "[dim]→[/dim]"

        table.add_row(
            r["title"][:40],
            r["keyword"],
            str(r.get("current_position", "?")),
            change_str,
            "[red]YES[/red]" if r.get("needs_update") else "[green]No[/green]",
        )

    console.print(table)


@cli.command("update-queue")
def update_queue():
    """Show articles that need updating."""
    rm = RankMonitor()
    queue = rm.get_update_queue()

    if not queue:
        console.print("[green]No articles need updating![/green]")
        return

    table = Table(title="Update Queue", box=box.ROUNDED)
    table.add_column("Article", style="bold")
    table.add_column("Keyword")
    table.add_column("Position", justify="center")
    table.add_column("Last Checked")

    for item in queue:
        table.add_row(
            item["title"][:40],
            item["keyword"],
            str(item.get("current_position", "?")),
            item.get("last_checked", "?")[:10],
        )

    console.print(table)


@cli.command("init-db")
def init_database():
    """Initialize the database tables."""
    console.print("[bold]Initializing database...[/bold]")
    init_db()
    console.print("[green]✓ Database initialized successfully![/green]")


@cli.command("configure-brand")
@click.option("--output", "-o", default="brand_config.json", help="Output file")
def configure_brand(output: str):
    """Interactive brand configuration."""
    console.print(Panel(
        "Configure your brand information for content differentiation.\n"
        "This is used by the Brand Angle Generator to create unique perspectives.",
        title="🏷️ Brand Configuration",
        border_style="blue",
    ))

    config = {
        "product_name": click.prompt("Product name"),
        "product_description": click.prompt("Product description"),
        "use_cases": click.prompt(
            "Use cases (comma-separated)"
        ).split(","),
        "competitive_advantages": click.prompt(
            "Competitive advantages (comma-separated)"
        ).split(","),
        "target_audience": click.prompt("Target audience"),
        "brand_voice": click.prompt(
            "Brand voice", default="professional, technical, authoritative"
        ),
    }

    # Clean up whitespace
    config["use_cases"] = [x.strip() for x in config["use_cases"]]
    config["competitive_advantages"] = [x.strip() for x in config["competitive_advantages"]]

    with open(output, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"\n[green]✓ Brand config saved to {output}[/green]")
    console.print(f"  Use with: [bold]python bot_seo.py run <keyword> -b {output}[/bold]")


@cli.command("gsc-analyze")
@click.option("--days", "-d", default=90, help="Lookback period in days")
@click.option("--top", "-n", default=10, help="Top N entries for title rewrites")
@click.option("--ctr-threshold", default=0.03, help="Low CTR threshold")
@click.option("--min-impressions", default=100, help="Minimum impressions")
def gsc_analyze(days: int, top: int, ctr_threshold: float, min_impressions: int):
    """Analyze Google Search Console data for CTR optimization opportunities."""
    console.print(Panel(
        f"Analyzing GSC data ({days}d window)\n"
        f"CTR threshold: {ctr_threshold:.1%} | Min impressions: {min_impressions}",
        title="📊 GSC Performance Analyzer",
        border_style="cyan",
    ))

    analyzer = PerformanceAnalyzer()
    analyzer.LOW_CTR_THRESHOLD = ctr_threshold
    analyzer.HIGH_IMPRESSION_MIN = min_impressions

    with console.status("[cyan]Fetching GSC data..."):
        report = analyzer.full_analysis(days=days, top_n=top)

    if report.get("status") == "no_data":
        console.print(f"[red]✗ {report['message']}[/red]")
        return

    # Summary table
    summary = report["summary"]
    table = Table(title="Overview", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Queries", f"{summary['total_queries']:,}")
    table.add_row("Total Clicks", f"{summary['total_clicks']:,}")
    table.add_row("Total Impressions", f"{summary['total_impressions']:,}")
    table.add_row("Avg CTR", f"{summary['avg_ctr']:.2%}")
    table.add_row("Avg Position", f"{summary['avg_position']:.1f}")
    console.print(table)

    # Low CTR table
    if report["low_ctr_pages"]:
        console.print(f"\n[bold yellow]⚠ Low CTR Pages ({len(report['low_ctr_pages'])})[/bold yellow]")
        lct = Table(box=box.SIMPLE)
        lct.add_column("Query", max_width=35)
        lct.add_column("Pos", justify="right")
        lct.add_column("Impr", justify="right")
        lct.add_column("Clicks", justify="right")
        lct.add_column("CTR", justify="right")
        lct.add_column("Expected", justify="right")
        lct.add_column("Potential+", justify="right", style="green")
        for e in report["low_ctr_pages"][:15]:
            lct.add_row(
                e["query"][:35], f"{e['position']:.1f}",
                str(e["impressions"]), str(e["clicks"]),
                f"{e['ctr']:.2%}", f"{e['expected_ctr']:.2%}",
                f"+{e['potential_clicks']:.0f}",
            )
        console.print(lct)

    # Impression/Click gap
    if report["impression_click_gaps"]:
        console.print(f"\n[bold red]🔥 High Impression / Low Click Gaps ({len(report['impression_click_gaps'])})[/bold red]")
        gt = Table(box=box.SIMPLE)
        gt.add_column("Query", max_width=35)
        gt.add_column("Pos", justify="right")
        gt.add_column("Impr", justify="right")
        gt.add_column("Clicks", justify="right")
        gt.add_column("Missed", justify="right", style="red")
        for e in report["impression_click_gaps"][:15]:
            gt.add_row(
                e["query"][:35], f"{e['position']:.1f}",
                str(e["impressions"]), str(e["clicks"]),
                f"{e['missed_clicks']:.0f}",
            )
        console.print(gt)

    # Title rewrite suggestions
    if report["title_rewrite_suggestions"]:
        console.print(f"\n[bold green]✏️ Title Rewrite Suggestions ({len(report['title_rewrite_suggestions'])})[/bold green]")
        for i, s in enumerate(report["title_rewrite_suggestions"], 1):
            pri_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(s["priority"], "white")
            console.print(Panel(
                f"[bold]Query:[/bold] {s['query']}\n"
                f"[bold]New Title:[/bold] {s['new_title']}\n"
                f"[bold]New Meta:[/bold] {s['new_meta_description']}\n"
                f"[bold]Reasoning:[/bold] {s['reasoning']}\n"
                f"[bold]Expected Lift:[/bold] {s['expected_ctr_lift']}",
                title=f"#{i} [{pri_color}]{s['priority'].upper()}[/{pri_color}]",
                border_style=pri_color,
            ))

    quick = report.get("quick_wins", 0)
    if quick:
        console.print(f"\n[bold green]🎯 {quick} quick-win opportunities identified![/bold green]")


# ──────────────────────────────────────────────
# Display Helpers
# ──────────────────────────────────────────────

def _display_serp_summary(analysis: dict):
    """Display SERP analysis summary."""
    table = Table(title="SERP Analysis", box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Keyword", analysis.get("keyword", "?"))
    table.add_row("Dominant Intent", analysis.get("dominant_intent", "?"))
    table.add_row("Avg Word Count", str(analysis.get("avg_word_count", "?")))
    table.add_row("Saturation Score", f"{analysis.get('saturation_score', '?')}/100")
    table.add_row("Organic Results", str(len(analysis.get("serp_results", []))))
    table.add_row("PAA Questions", str(len(analysis.get("people_also_ask", []))))
    table.add_row("Related Searches", str(len(analysis.get("related_searches", []))))

    console.print(table)

    if analysis.get("content_patterns"):
        console.print("\n[bold]Content Patterns:[/bold]")
        for p in analysis["content_patterns"][:5]:
            console.print(f"  • {p}")


def _display_intent_summary(analysis: dict):
    """Display intent analysis summary."""
    table = Table(title="Intent Analysis", box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Intent Type", analysis.get("intent_type", "?"))
    table.add_row("Confidence", f"{analysis.get('intent_confidence', '?')}")
    table.add_row("Gaps Found", str(len(analysis.get("intent_gaps", []))))
    table.add_row("Tech Depth Missing", str(analysis.get("technical_depth_missing", "?")))

    console.print(table)

    if analysis.get("intent_gaps"):
        console.print("\n[bold]Intent Gaps:[/bold]")
        for gap in analysis["intent_gaps"][:5]:
            score = gap.get("opportunity_score", "?")
            console.print(f"  [{score}/10] {gap.get('gap', '?')}")

    if analysis.get("suggested_angles"):
        console.print("\n[bold]Suggested Angles:[/bold]")
        for angle in analysis["suggested_angles"][:3]:
            console.print(f"  → {angle.get('title_suggestion', '?')}")
            console.print(f"    {angle.get('angle_description', '')[:100]}")


def _display_cluster_summary(blueprint: dict):
    """Display cluster blueprint summary."""
    pillar = blueprint.get("pillar", {})
    clusters = blueprint.get("clusters", [])

    console.print(Panel(
        f"[bold]{pillar.get('title', '?')}[/bold]\n"
        f"Words: ~{pillar.get('estimated_word_count', '?')} | "
        f"Intent: {pillar.get('target_intent', '?')}",
        title="📌 Pillar Page",
        border_style="yellow",
    ))

    table = Table(title="Cluster Articles", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("Title", style="bold")
    table.add_column("Type")
    table.add_column("Intent")
    table.add_column("Words", justify="right")

    for i, c in enumerate(clusters, 1):
        table.add_row(
            str(i),
            c.get("title", "?")[:50],
            c.get("content_type", "?"),
            c.get("target_intent", "?"),
            str(c.get("estimated_word_count", "?")),
        )

    console.print(table)


def _display_brand_summary(branded: dict):
    """Display branded angles summary."""
    pillar = branded.get("pillar", {})
    console.print(f"\n[bold]Pillar (branded):[/bold] {pillar.get('branded_title', pillar.get('title', '?'))}")

    if pillar.get("brand_angle"):
        console.print(f"  Angle: {pillar['brand_angle'][:100]}")

    clusters = branded.get("clusters", [])
    if clusters:
        console.print(f"\n[bold]Cluster Angles:[/bold]")
        for i, c in enumerate(clusters, 1):
            branded_title = c.get("branded_title", c.get("title", "?"))
            console.print(f"  {i}. {branded_title}")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cli()
