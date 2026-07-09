"""
GSC Performance Tool — Standalone Google Search Console Analyzer
==================================================================
Independent tool for analyzing GSC data, detecting CTR issues,
and generating title rewrite suggestions.

Usage:
    python gsc_tool.py                  Interactive menu
    python gsc_tool.py analyze          Full analysis (default 90 days)
    python gsc_tool.py analyze -d 30    Last 30 days
    python gsc_tool.py low-ctr          Show low-CTR pages only
    python gsc_tool.py gaps             Show impression/click gaps only
    python gsc_tool.py rewrite          Generate title rewrite suggestions
    python gsc_tool.py page <url>       Analyze a specific page
    python gsc_tool.py export           Full analysis + export to HTML/CSV/JSON
    python gsc_tool.py setup            Configure GSC credentials
"""

import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.logging import RichHandler
from rich import box

from config import settings
from core.analytics.performance_analyzer import PerformanceAnalyzer

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("gsc_tool")

EXPORT_DIR = Path("reports")


def _get_analyzer() -> PerformanceAnalyzer:
    """Create and validate analyzer instance."""
    if not settings.gsc.credentials_path:
        console.print("[red]✗ GSC_CREDENTIALS_PATH not set in .env[/red]")
        console.print("  Run: [bold]python gsc_tool.py setup[/bold]")
        sys.exit(1)
    if not settings.gsc.site_url:
        console.print("[red]✗ GSC_SITE_URL not set in .env[/red]")
        console.print("  Run: [bold]python gsc_tool.py setup[/bold]")
        sys.exit(1)
    return PerformanceAnalyzer()


# ──────────────────────────────────────────────
# Display Helpers
# ──────────────────────────────────────────────

def _show_summary(summary: dict):
    """Display overview summary table."""
    table = Table(title="📊 GSC Overview", box=box.ROUNDED, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Queries", f"{summary['total_queries']:,}")
    table.add_row("Total Clicks", f"{summary['total_clicks']:,}")
    table.add_row("Total Impressions", f"{summary['total_impressions']:,}")
    table.add_row("Avg CTR", f"{summary['avg_ctr']:.2%}")
    table.add_row("Avg Position", f"{summary['avg_position']:.1f}")
    console.print(table)


def _show_low_ctr(entries: list[dict], limit: int = 20):
    """Display low-CTR pages table."""
    if not entries:
        console.print("[dim]No low-CTR pages found.[/dim]")
        return

    console.print(f"\n[bold yellow]⚠ Low CTR Pages ({len(entries)} found)[/bold yellow]")
    table = Table(box=box.SIMPLE_HEAVY, border_style="yellow")
    table.add_column("#", style="dim", width=3)
    table.add_column("Query", max_width=40)
    table.add_column("Page", max_width=35, style="dim")
    table.add_column("Pos", justify="right")
    table.add_column("Impr", justify="right")
    table.add_column("Clicks", justify="right")
    table.add_column("CTR", justify="right")
    table.add_column("Expected", justify="right", style="cyan")
    table.add_column("Potential+", justify="right", style="green bold")

    for i, e in enumerate(entries[:limit], 1):
        page_short = e.get("page", "")
        if len(page_short) > 35:
            page_short = "..." + page_short[-32:]
        table.add_row(
            str(i),
            e["query"][:40],
            page_short,
            f"{e['position']:.1f}",
            f"{e['impressions']:,}",
            str(e["clicks"]),
            f"{e['ctr']:.2%}",
            f"{e['expected_ctr']:.2%}",
            f"+{e['potential_clicks']:.0f}",
        )

    console.print(table)


def _show_gaps(entries: list[dict], limit: int = 20):
    """Display impression/click gap table."""
    if not entries:
        console.print("[dim]No impression/click gaps found.[/dim]")
        return

    console.print(f"\n[bold red]🔥 High Impression / Low Click Gaps ({len(entries)} found)[/bold red]")
    table = Table(box=box.SIMPLE_HEAVY, border_style="red")
    table.add_column("#", style="dim", width=3)
    table.add_column("Query", max_width=40)
    table.add_column("Pos", justify="right")
    table.add_column("Impr", justify="right")
    table.add_column("Clicks", justify="right")
    table.add_column("Waste Ratio", justify="right", style="red")
    table.add_column("Expected Clicks", justify="right", style="cyan")
    table.add_column("Missed", justify="right", style="red bold")

    for i, e in enumerate(entries[:limit], 1):
        table.add_row(
            str(i),
            e["query"][:40],
            f"{e['position']:.1f}",
            f"{e['impressions']:,}",
            str(e["clicks"]),
            f"{e['waste_ratio']:.0f}x",
            f"{e['expected_clicks']:.0f}",
            f"{e['missed_clicks']:.0f}",
        )

    console.print(table)


def _show_rewrites(rewrites: list[dict]):
    """Display title rewrite suggestions."""
    if not rewrites:
        console.print("[dim]No rewrite suggestions.[/dim]")
        return

    console.print(f"\n[bold green]✏️ Title Rewrite Suggestions ({len(rewrites)})[/bold green]")
    for i, s in enumerate(rewrites, 1):
        pri = s.get("priority", "medium")
        color = {"high": "red", "medium": "yellow", "low": "dim"}.get(pri, "white")
        metrics = s.get("current_metrics", {})
        metrics_line = (
            f"Pos: {metrics.get('position', '?'):.1f} | "
            f"Impr: {metrics.get('impressions', '?'):,} | "
            f"Clicks: {metrics.get('clicks', '?')} | "
            f"CTR: {metrics.get('ctr', 0):.2%}"
            if metrics else "N/A"
        )

        console.print(Panel(
            f"[bold]Query:[/bold]  {s['query']}\n"
            f"[bold]Page:[/bold]   {s.get('page', 'N/A')}\n"
            f"[bold]Metrics:[/bold] {metrics_line}\n"
            f"{'─' * 50}\n"
            f"[bold green]New Title:[/bold green]  {s['new_title']}\n"
            f"[bold green]New Meta:[/bold green]   {s['new_meta_description']}\n"
            f"{'─' * 50}\n"
            f"[bold]Why:[/bold]    {s['reasoning']}\n"
            f"[bold]Lift:[/bold]   {s['expected_ctr_lift']}",
            title=f"#{i}  [{color}]● {pri.upper()}[/{color}]",
            border_style=color,
            padding=(1, 2),
        ))


# ──────────────────────────────────────────────
# Export Functions
# ──────────────────────────────────────────────

def _ensure_export_dir():
    """Create reports directory if not exists."""
    EXPORT_DIR.mkdir(exist_ok=True)


def _export_json(report: dict, filename: str) -> Path:
    """Export full report as JSON."""
    _ensure_export_dir()
    path = EXPORT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return path


def _export_csv(entries: list[dict], filename: str, fields: list[str]) -> Path:
    """Export list of dicts as CSV."""
    _ensure_export_dir()
    path = EXPORT_DIR / filename
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    return path


def _export_html(report: dict, filename: str) -> Path:
    """Export full analysis as a self-contained HTML report."""
    _ensure_export_dir()
    path = EXPORT_DIR / filename
    summary = report.get("summary", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build low-CTR rows
    low_ctr_rows = ""
    for e in report.get("low_ctr_pages", []):
        low_ctr_rows += f"""<tr>
            <td>{_h(e['query'])}</td>
            <td>{_h(e.get('page', ''))}</td>
            <td>{e['position']:.1f}</td>
            <td>{e['impressions']:,}</td>
            <td>{e['clicks']}</td>
            <td>{e['ctr']:.2%}</td>
            <td>{e.get('expected_ctr', 0):.2%}</td>
            <td class="positive">+{e.get('potential_clicks', 0):.0f}</td>
        </tr>"""

    # Build gaps rows
    gap_rows = ""
    for e in report.get("impression_click_gaps", []):
        gap_rows += f"""<tr>
            <td>{_h(e['query'])}</td>
            <td>{e['position']:.1f}</td>
            <td>{e['impressions']:,}</td>
            <td>{e['clicks']}</td>
            <td class="negative">{e.get('waste_ratio', 0):.0f}x</td>
            <td class="negative">{e.get('missed_clicks', 0):.0f}</td>
        </tr>"""

    # Build rewrite cards
    rewrite_cards = ""
    for i, s in enumerate(report.get("title_rewrite_suggestions", []), 1):
        pri = s.get("priority", "medium")
        pri_class = {"high": "priority-high", "medium": "priority-medium", "low": "priority-low"}.get(pri, "")
        metrics = s.get("current_metrics", {})
        rewrite_cards += f"""
        <div class="rewrite-card {pri_class}">
            <div class="card-header">#{i} — {pri.upper()}</div>
            <div class="card-body">
                <p><strong>Query:</strong> {_h(s['query'])}</p>
                <p><strong>Page:</strong> {_h(s.get('page', 'N/A'))}</p>
                <p class="metrics">Pos: {metrics.get('position', 0):.1f} | Impr: {metrics.get('impressions', 0):,} | Clicks: {metrics.get('clicks', 0)} | CTR: {metrics.get('ctr', 0):.2%}</p>
                <hr>
                <p class="suggestion"><strong>✏️ New Title:</strong> {_h(s.get('new_title', ''))}</p>
                <p class="suggestion"><strong>📝 New Meta:</strong> {_h(s.get('new_meta_description', ''))}</p>
                <hr>
                <p><strong>Why:</strong> {_h(s.get('reasoning', ''))}</p>
                <p><strong>Expected Lift:</strong> {_h(s.get('expected_ctr_lift', ''))}</p>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GSC Performance Report — {timestamp}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #38bdf8; font-size: 1.8rem; margin-bottom: 0.5rem; }}
    h2 {{ color: #f59e0b; font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
    .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
    .stat-card {{ background: #1e293b; border-radius: 12px; padding: 1.2rem; text-align: center; border: 1px solid #334155; }}
    .stat-card .value {{ font-size: 1.8rem; font-weight: 700; color: #38bdf8; }}
    .stat-card .label {{ font-size: 0.85rem; color: #94a3b8; margin-top: 0.3rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
    th {{ background: #1e293b; color: #38bdf8; padding: 0.7rem; text-align: left; position: sticky; top: 0; }}
    td {{ padding: 0.6rem 0.7rem; border-bottom: 1px solid #1e293b; }}
    tr:hover {{ background: #1e293b66; }}
    .positive {{ color: #4ade80; font-weight: 600; }}
    .negative {{ color: #f87171; font-weight: 600; }}
    .rewrite-card {{ background: #1e293b; border-radius: 12px; margin: 1rem 0; overflow: hidden; border: 1px solid #334155; }}
    .card-header {{ padding: 0.8rem 1.2rem; font-weight: 700; font-size: 0.95rem; }}
    .card-body {{ padding: 1rem 1.2rem; }}
    .card-body p {{ margin: 0.4rem 0; }}
    .card-body hr {{ border: none; border-top: 1px solid #334155; margin: 0.8rem 0; }}
    .metrics {{ color: #94a3b8; font-size: 0.85rem; }}
    .suggestion {{ background: #0f172a; padding: 0.5rem 0.8rem; border-radius: 6px; margin: 0.4rem 0; }}
    .priority-high .card-header {{ background: #dc2626; color: white; }}
    .priority-medium .card-header {{ background: #d97706; color: white; }}
    .priority-low .card-header {{ background: #475569; color: #cbd5e1; }}
    .quick-wins {{ background: linear-gradient(135deg, #065f46, #064e3b); border-radius: 12px; padding: 1.5rem; text-align: center; margin: 2rem 0; font-size: 1.2rem; font-weight: 700; color: #4ade80; }}
    .footer {{ text-align: center; color: #475569; margin-top: 3rem; font-size: 0.8rem; }}
    @media print {{ body {{ background: white; color: #111; }} .stat-card {{ border: 1px solid #ccc; }} th {{ background: #f1f5f9; color: #111; }} }}
</style>
</head>
<body>
<div class="container">

<h1>📊 GSC Performance Report</h1>
<p class="subtitle">Generated: {timestamp} | Period: {report.get('period_days', 90)} days | Site: {_h(settings.gsc.site_url)}</p>

<div class="summary-grid">
    <div class="stat-card"><div class="value">{summary.get('total_queries', 0):,}</div><div class="label">Total Queries</div></div>
    <div class="stat-card"><div class="value">{summary.get('total_clicks', 0):,}</div><div class="label">Total Clicks</div></div>
    <div class="stat-card"><div class="value">{summary.get('total_impressions', 0):,}</div><div class="label">Total Impressions</div></div>
    <div class="stat-card"><div class="value">{summary.get('avg_ctr', 0):.2%}</div><div class="label">Avg CTR</div></div>
    <div class="stat-card"><div class="value">{summary.get('avg_position', 0):.1f}</div><div class="label">Avg Position</div></div>
</div>

<h2>⚠️ Low CTR Pages ({len(report.get('low_ctr_pages', []))})</h2>
<table>
<tr><th>Query</th><th>Page</th><th>Pos</th><th>Impr</th><th>Clicks</th><th>CTR</th><th>Expected</th><th>Potential+</th></tr>
{low_ctr_rows if low_ctr_rows else '<tr><td colspan="8" style="text-align:center;color:#475569">No low-CTR pages found</td></tr>'}
</table>

<h2>🔥 High Impression / Low Click Gaps ({len(report.get('impression_click_gaps', []))})</h2>
<table>
<tr><th>Query</th><th>Pos</th><th>Impr</th><th>Clicks</th><th>Waste</th><th>Missed Clicks</th></tr>
{gap_rows if gap_rows else '<tr><td colspan="6" style="text-align:center;color:#475569">No gaps found</td></tr>'}
</table>

<h2>✏️ Title Rewrite Suggestions ({len(report.get('title_rewrite_suggestions', []))})</h2>
{rewrite_cards if rewrite_cards else '<p style="color:#475569">No rewrite suggestions</p>'}

{"<div class='quick-wins'>🎯 " + str(report.get('quick_wins', 0)) + " Quick-Win Opportunities Identified!</div>" if report.get('quick_wins', 0) else ""}

<div class="footer">Generated by GSC Performance Tool — SEO Bot Analytics Engine</div>
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _h(text: str) -> str:
    """HTML-escape a string."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ──────────────────────────────────────────────
# CLI Commands
# ──────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """📊 GSC Performance Tool — Google Search Console Analyzer"""
    if ctx.invoked_subcommand is None:
        _interactive_menu()


@cli.command("analyze")
@click.option("-d", "--days", default=90, help="Lookback period in days")
@click.option("-n", "--top", default=10, help="Top N for title rewrites")
@click.option("--ctr", default=0.03, help="Low CTR threshold (e.g., 0.03 = 3%)")
@click.option("--min-impr", default=100, help="Minimum impressions")
def cmd_analyze(days: int, top: int, ctr: float, min_impr: int):
    """Run full GSC analysis with all detections + title rewrites."""
    analyzer = _get_analyzer()
    analyzer.LOW_CTR_THRESHOLD = ctr
    analyzer.HIGH_IMPRESSION_MIN = min_impr

    console.print(Panel(
        f"Period: {days} days | CTR threshold: {ctr:.1%} | Min impressions: {min_impr}",
        title="📊 Full GSC Analysis",
        border_style="cyan",
    ))

    with console.status("[cyan]Fetching & analyzing GSC data..."):
        report = analyzer.full_analysis(days=days, top_n=top)

    if report.get("status") == "no_data":
        console.print(f"[red]✗ {report['message']}[/red]")
        return

    _show_summary(report["summary"])
    _show_low_ctr(report["low_ctr_pages"])
    _show_gaps(report["impression_click_gaps"])
    _show_rewrites(report["title_rewrite_suggestions"])

    quick = report.get("quick_wins", 0)
    if quick:
        console.print(f"\n[bold green]🎯 {quick} quick-win opportunities![/bold green]")


@cli.command("low-ctr")
@click.option("-d", "--days", default=90, help="Lookback period")
@click.option("--ctr", default=0.03, help="CTR threshold")
@click.option("--min-impr", default=100, help="Min impressions")
@click.option("--limit", default=30, help="Max results to show")
def cmd_low_ctr(days: int, ctr: float, min_impr: int, limit: int):
    """Detect pages with low CTR despite good impressions."""
    analyzer = _get_analyzer()
    analyzer.LOW_CTR_THRESHOLD = ctr
    analyzer.HIGH_IMPRESSION_MIN = min_impr

    with console.status("[cyan]Fetching GSC data..."):
        results = analyzer.detect_low_ctr(days=days)

    _show_low_ctr(results, limit=limit)
    console.print(f"\n[dim]Showing {min(len(results), limit)} of {len(results)} entries[/dim]")


@cli.command("gaps")
@click.option("-d", "--days", default=90, help="Lookback period")
@click.option("--min-impr", default=200, help="Min impressions")
@click.option("--max-clicks", default=10, help="Max clicks (to qualify as low)")
@click.option("--limit", default=30, help="Max results to show")
def cmd_gaps(days: int, min_impr: int, max_clicks: int, limit: int):
    """Detect high impression / low click gaps."""
    analyzer = _get_analyzer()

    with console.status("[cyan]Fetching GSC data..."):
        results = analyzer.detect_impression_click_gap(
            days=days, min_impressions=min_impr, max_clicks=max_clicks
        )

    _show_gaps(results, limit=limit)
    console.print(f"\n[dim]Showing {min(len(results), limit)} of {len(results)} entries[/dim]")


@cli.command("rewrite")
@click.option("-d", "--days", default=90, help="Lookback period")
@click.option("-n", "--top", default=10, help="Number of pages to rewrite")
def cmd_rewrite(days: int, top: int):
    """Generate LLM-powered title rewrite suggestions."""
    analyzer = _get_analyzer()

    with console.status("[cyan]Fetching GSC data..."):
        data = analyzer.fetch_performance(days=days)
        low_ctr = analyzer.detect_low_ctr(data=data)
        gaps = analyzer.detect_impression_click_gap(data=data)

    # Merge top opportunities
    seen = set()
    combined = []
    for e in low_ctr + gaps:
        key = (e["query"], e.get("page", ""))
        if key not in seen:
            seen.add(key)
            combined.append(e)

    combined.sort(
        key=lambda x: x.get("opportunity_score", x.get("missed_clicks", 0)),
        reverse=True,
    )

    console.print(f"[cyan]Generating rewrites for top {top} pages...[/cyan]")
    with console.status("[cyan]LLM is writing title suggestions..."):
        rewrites = analyzer.suggest_title_rewrites(combined[:top], top_n=top)

    _show_rewrites(rewrites)


@cli.command("page")
@click.argument("url")
@click.option("-d", "--days", default=90, help="Lookback period")
def cmd_page(url: str, days: int):
    """Analyze a specific page's GSC performance."""
    analyzer = _get_analyzer()

    console.print(Panel(f"URL: {url}\nPeriod: {days}d", title="🔍 Page Analysis", border_style="cyan"))

    with console.status("[cyan]Fetching page data..."):
        data = analyzer.fetch_page_performance(url, days=days)

    if not data:
        console.print("[red]✗ No data found for this URL[/red]")
        return

    total_clicks = sum(r["clicks"] for r in data)
    total_impr = sum(r["impressions"] for r in data)
    avg_ctr = total_clicks / total_impr if total_impr else 0

    # Summary
    console.print(f"\n[bold]Queries: {len(data)} | Clicks: {total_clicks:,} | Impressions: {total_impr:,} | CTR: {avg_ctr:.2%}[/bold]")

    # Query table
    table = Table(title="Queries driving this page", box=box.SIMPLE_HEAVY)
    table.add_column("#", style="dim", width=3)
    table.add_column("Query", max_width=50)
    table.add_column("Pos", justify="right")
    table.add_column("Impr", justify="right")
    table.add_column("Clicks", justify="right")
    table.add_column("CTR", justify="right")

    for i, r in enumerate(sorted(data, key=lambda x: x["impressions"], reverse=True)[:30], 1):
        ctr_style = "red" if r["ctr"] < 0.03 else ("yellow" if r["ctr"] < 0.05 else "green")
        table.add_row(
            str(i), r["query"][:50],
            f"{r['position']:.1f}", f"{r['impressions']:,}",
            str(r["clicks"]), f"[{ctr_style}]{r['ctr']:.2%}[/{ctr_style}]",
        )

    console.print(table)


@cli.command("export")
@click.option("-d", "--days", default=90, help="Lookback period")
@click.option("-n", "--top", default=10, help="Top N for rewrites")
@click.option("--format", "fmt", type=click.Choice(["all", "html", "csv", "json"]), default="all", help="Export format")
def cmd_export(days: int, top: int, fmt: str):
    """Export full analysis to HTML/CSV/JSON files."""
    analyzer = _get_analyzer()

    console.print(Panel(f"Period: {days}d | Rewrites: top {top} | Format: {fmt}", title="📁 Export", border_style="cyan"))

    with console.status("[cyan]Running full analysis..."):
        report = analyzer.full_analysis(days=days, top_n=top)

    if report.get("status") == "no_data":
        console.print(f"[red]✗ {report['message']}[/red]")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exported = []

    if fmt in ("all", "json"):
        p = _export_json(report, f"gsc_report_{timestamp}.json")
        exported.append(("JSON", p))

    if fmt in ("all", "html"):
        p = _export_html(report, f"gsc_report_{timestamp}.html")
        exported.append(("HTML", p))

    if fmt in ("all", "csv"):
        if report.get("low_ctr_pages"):
            p = _export_csv(
                report["low_ctr_pages"],
                f"gsc_low_ctr_{timestamp}.csv",
                ["query", "page", "position", "impressions", "clicks", "ctr", "expected_ctr", "ctr_gap", "potential_clicks", "opportunity_score"],
            )
            exported.append(("CSV (Low CTR)", p))
        if report.get("impression_click_gaps"):
            p = _export_csv(
                report["impression_click_gaps"],
                f"gsc_gaps_{timestamp}.csv",
                ["query", "page", "position", "impressions", "clicks", "waste_ratio", "expected_clicks", "missed_clicks"],
            )
            exported.append(("CSV (Gaps)", p))
        if report.get("title_rewrite_suggestions"):
            p = _export_csv(
                report["title_rewrite_suggestions"],
                f"gsc_rewrites_{timestamp}.csv",
                ["query", "page", "new_title", "new_meta_description", "reasoning", "expected_ctr_lift", "priority"],
            )
            exported.append(("CSV (Rewrites)", p))

    # Report results
    console.print()
    table = Table(title="Exported Files", box=box.ROUNDED, border_style="green")
    table.add_column("Format", style="bold")
    table.add_column("File")

    for label, path in exported:
        table.add_row(label, str(path))

    console.print(table)
    console.print(f"\n[green]✓ {len(exported)} files exported to [bold]{EXPORT_DIR}/[/bold][/green]")


@cli.command("setup")
def cmd_setup():
    """Interactive setup wizard for GSC credentials."""
    console.print(Panel(
        "This wizard will help you configure Google Search Console API access.\n\n"
        "[bold]Prerequisites:[/bold]\n"
        "1. Create a GCP project at https://console.cloud.google.com\n"
        "2. Enable the Search Console API\n"
        "3. Create a Service Account and download the JSON key\n"
        "4. Add the service account email as a user in GSC property",
        title="🔧 GSC Setup Wizard",
        border_style="cyan",
        padding=(1, 2),
    ))

    console.print()

    # Step 1: Credentials path
    creds_path = Prompt.ask(
        "[bold]Path to service account JSON file[/bold]",
        default=settings.gsc.credentials_path or "credentials/gsc-service-account.json",
    )

    # Validate file exists
    if not Path(creds_path).exists():
        console.print(f"[yellow]⚠ File '{creds_path}' does not exist yet.[/yellow]")
        console.print(f"  Place your service account JSON at this path before running analysis.")
    else:
        console.print(f"[green]✓ File found: {creds_path}[/green]")

    # Step 2: Site URL
    site_url = Prompt.ask(
        "[bold]GSC Property URL[/bold]  (e.g., https://example.com or sc-domain:example.com)",
        default=settings.gsc.site_url or "https://",
    )

    # Step 3: Write to .env
    env_path = Path(__file__).parent / ".env"

    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text(encoding="utf-8").splitlines()

    # Update or append GSC settings
    updated_creds = False
    updated_url = False
    for i, line in enumerate(env_lines):
        if line.startswith("GSC_CREDENTIALS_PATH="):
            env_lines[i] = f"GSC_CREDENTIALS_PATH={creds_path}"
            updated_creds = True
        elif line.startswith("GSC_SITE_URL="):
            env_lines[i] = f"GSC_SITE_URL={site_url}"
            updated_url = True

    if not updated_creds:
        env_lines.append(f"\n# Google Search Console")
        env_lines.append(f"GSC_CREDENTIALS_PATH={creds_path}")
    if not updated_url:
        env_lines.append(f"GSC_SITE_URL={site_url}")

    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    console.print(f"\n[green]✓ Settings saved to .env[/green]")
    console.print(Panel(
        f"GSC_CREDENTIALS_PATH={creds_path}\n"
        f"GSC_SITE_URL={site_url}",
        title="Saved Configuration",
        border_style="green",
    ))

    # Step 4: Test connection
    if Path(creds_path).exists():
        if Confirm.ask("\n[bold]Test GSC connection now?[/bold]", default=True):
            try:
                # Reload settings
                os.environ["GSC_CREDENTIALS_PATH"] = creds_path
                os.environ["GSC_SITE_URL"] = site_url

                analyzer = PerformanceAnalyzer()
                with console.status("[cyan]Testing connection..."):
                    data = analyzer.fetch_performance(days=7, row_limit=5)

                if data:
                    console.print(f"[green]✓ Connected! Got {len(data)} rows from last 7 days.[/green]")
                else:
                    console.print("[yellow]⚠ Connected but no data returned (site may be new or unverified).[/yellow]")
            except Exception as e:
                console.print(f"[red]✗ Connection failed: {e}[/red]")
    else:
        console.print(f"\n[dim]Place your JSON key at '{creds_path}' and run [bold]python gsc_tool.py setup[/bold] again to test.[/dim]")


# ──────────────────────────────────────────────
# Interactive Menu
# ──────────────────────────────────────────────

def _interactive_menu():
    """Rich interactive menu for the GSC tool."""
    console.print(Panel(
        "[bold cyan]GSC Performance Tool[/bold cyan]\n"
        "Google Search Console Analyzer — SEO Feedback Loop\n\n"
        "Select an action below:",
        border_style="cyan",
        padding=(1, 2),
    ))

    MENU = {
        "1": ("📊  Full Analysis", "Run complete GSC analysis with CTR detection + title rewrites"),
        "2": ("⚠️  Low CTR Detection", "Find pages with high impressions but low CTR"),
        "3": ("🔥  Impression/Click Gaps", "Find pages users see but don't click"),
        "4": ("✏️  Title Rewrites", "Generate AI-powered title optimization suggestions"),
        "5": ("🔍  Page Analysis", "Analyze a specific URL's performance"),
        "6": ("📁  Export Report", "Export full analysis to HTML/CSV/JSON"),
        "7": ("🔧  Setup", "Configure GSC credentials"),
        "0": ("👋  Exit", ""),
    }

    for key, (label, desc) in MENU.items():
        if desc:
            console.print(f"  [bold cyan]{key}[/bold cyan]  {label}  [dim]— {desc}[/dim]")
        else:
            console.print(f"  [bold cyan]{key}[/bold cyan]  {label}")

    console.print()
    choice = Prompt.ask("[bold]Choose[/bold]", choices=list(MENU.keys()), default="1")

    if choice == "0":
        console.print("[dim]Bye![/dim]")
        return

    if choice == "7":
        ctx = click.Context(cmd_setup)
        ctx.invoke(cmd_setup)
        return

    # Common parameters
    days = IntPrompt.ask("Lookback period (days)", default=90)

    if choice == "1":
        top = IntPrompt.ask("Top N for rewrites", default=10)
        ctx = click.Context(cmd_analyze)
        ctx.invoke(cmd_analyze, days=days, top=top, ctr=0.03, min_impr=100)

    elif choice == "2":
        ctx = click.Context(cmd_low_ctr)
        ctx.invoke(cmd_low_ctr, days=days, ctr=0.03, min_impr=100, limit=30)

    elif choice == "3":
        ctx = click.Context(cmd_gaps)
        ctx.invoke(cmd_gaps, days=days, min_impr=200, max_clicks=10, limit=30)

    elif choice == "4":
        top = IntPrompt.ask("How many pages to rewrite", default=10)
        ctx = click.Context(cmd_rewrite)
        ctx.invoke(cmd_rewrite, days=days, top=top)

    elif choice == "5":
        url = Prompt.ask("[bold]Page URL[/bold]")
        ctx = click.Context(cmd_page)
        ctx.invoke(cmd_page, url=url, days=days)

    elif choice == "6":
        fmt = Prompt.ask("Format", choices=["all", "html", "csv", "json"], default="all")
        top = IntPrompt.ask("Top N for rewrites", default=10)
        ctx = click.Context(cmd_export)
        ctx.invoke(cmd_export, days=days, top=top, fmt=fmt)


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cli()
