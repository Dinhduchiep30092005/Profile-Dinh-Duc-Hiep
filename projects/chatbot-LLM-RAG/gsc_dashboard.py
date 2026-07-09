"""
GSC Dashboard — Web-based GUI for Google Search Console Analytics
==================================================================
Beautiful dark-theme dashboard to analyze GSC performance data.

Usage:
    python gsc_dashboard.py             Start on port 5000
    python gsc_dashboard.py --port 8080 Start on custom port
"""

import json
import logging
import os
import sys
import webbrowser
import threading
import requests
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_from_directory

from config import settings
from core.analytics.performance_analyzer import PerformanceAnalyzer

# Knowledge base uploads
UPLOAD_MAX_SIZE = 30 * 1024 * 1024  # 30 MB

# ──────────────────────────────────────────────
# Batch Article Generation — Background Worker
# ──────────────────────────────────────────────
_batch_jobs = {}  # job_id -> { status, total, completed, failed, results, items }
_batch_lock = threading.Lock()


def _run_batch_article_generation(job_id: str, item_ids: list, language_map: dict):
    """Background worker to generate articles sequentially."""
    import json as _json

    with _batch_lock:
        _batch_jobs[job_id]["status"] = "running"

    for item_id in item_ids:
        # Check if job was cancelled
        with _batch_lock:
            if _batch_jobs[job_id].get("cancelled"):
                _batch_jobs[job_id]["status"] = "cancelled"
                return

        try:
            from core.projects import _get_conn, init_projects_db, _migrate_db, update_content_item

            init_projects_db()
            _migrate_db()
            conn = _get_conn()
            row = conn.execute("SELECT * FROM content_items WHERE id = ?", (item_id,)).fetchone()
            conn.close()

            if not row:
                with _batch_lock:
                    _batch_jobs[job_id]["failed"] += 1
                    _batch_jobs[job_id]["results"].append({
                        "item_id": item_id, "success": False, "error": "Item không tồn tại"
                    })
                continue

            item = dict(row)
            heading_structure = {}
            try:
                heading_structure = _json.loads(item.get("heading_structure", "{}"))
            except (ValueError, TypeError):
                pass

            if not heading_structure or not heading_structure.get("sections"):
                with _batch_lock:
                    _batch_jobs[job_id]["failed"] += 1
                    _batch_jobs[job_id]["results"].append({
                        "item_id": item_id, "success": False,
                        "error": "Chưa có heading structure", "title": item.get("title", "")
                    })
                continue

            # Check if already generated
            if item.get("article_generated") == 1 and item.get("article_text"):
                with _batch_lock:
                    _batch_jobs[job_id]["completed"] += 1
                    _batch_jobs[job_id]["results"].append({
                        "item_id": item_id, "success": True, "skipped": True,
                        "title": item.get("title", ""), "message": "Đã có bài viết — bỏ qua"
                    })
                continue

            language = language_map.get(str(item_id), item.get("language", "vi") or "vi")

            with _batch_lock:
                _batch_jobs[job_id]["current_item"] = item.get("title", f"Item #{item_id}")

            # ── ArticleController: PreValidation → Generator → QualityGate → Renderer ──
            from core.content.article_controller import ArticleController
            controller = ArticleController()
            result = controller.generate(
                keyword=item["keyword"],
                title=item["title"],
                heading_structure=heading_structure,
                language=language,
            )

            if not result.get("success"):
                with _batch_lock:
                    _batch_jobs[job_id]["failed"] += 1
                    _batch_jobs[job_id]["results"].append({
                        "item_id": item_id, "success": False,
                        "error": result.get("error", "Generation failed"),
                        "title": item.get("title", "")
                    })
                continue

            # ── Save DB (draft) ──
            update_content_item(
                item_id,
                article_text=result["article_text"],
                article_html=result["article_html"],
                internal_links=result.get("internal_links", []),
                status="article_generated",
                article_generated=1,
            )

            with _batch_lock:
                _batch_jobs[job_id]["completed"] += 1
                _batch_jobs[job_id]["results"].append({
                    "item_id": item_id, "success": True,
                    "title": item.get("title", ""),
                    "word_count": result.get("word_count", 0),
                })

        except Exception as e:
            logger.exception(f"Batch article generation failed for item {item_id}")
            with _batch_lock:
                _batch_jobs[job_id]["failed"] += 1
                _batch_jobs[job_id]["results"].append({
                    "item_id": item_id, "success": False, "error": str(e),
                    "title": ""
                })

    with _batch_lock:
        _batch_jobs[job_id]["status"] = "completed"
        _batch_jobs[job_id]["current_item"] = ""

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = Flask(__name__, template_folder="dashboard/templates", static_folder="dashboard/static")
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_SIZE

logger = logging.getLogger(__name__)

EXPORT_DIR = Path("reports")
EXPORT_DIR.mkdir(exist_ok=True)

# Cache for analysis results
_cache = {
    "report": None,
    "timestamp": None,
    "days": None,
}


def _get_analyzer(**overrides) -> PerformanceAnalyzer:
    analyzer = PerformanceAnalyzer()
    for k, v in overrides.items():
        if hasattr(analyzer, k):
            setattr(analyzer, k, v)
    return analyzer


def _is_configured() -> bool:
    return bool(settings.gsc.credentials_path and settings.gsc.site_url)


# ──────────────────────────────────────────────
# Page Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    # Mask the OpenAI key for display (show only last 4 chars)
    raw_key = settings.openai.api_key or ""
    masked = ("sk-..." + raw_key[-4:]) if len(raw_key) > 4 else ""
    serpapi_raw = settings.serpapi.api_key or ""
    serpapi_masked = ("***" + serpapi_raw[-4:]) if len(serpapi_raw) > 4 else ""
    return render_template(
        "index.html",
        configured=_is_configured(),
        site_url=settings.gsc.site_url,
        settings_creds=settings.gsc.credentials_path,
        openai_key_masked=masked,
        openai_model=settings.openai.model,
        serpapi_key_masked=serpapi_masked,
    )


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({
        "configured": _is_configured(),
        "site_url": settings.gsc.site_url,
        "credentials_path": settings.gsc.credentials_path,
        "has_cache": _cache["report"] is not None,
        "cache_time": _cache["timestamp"],
        "cache_days": _cache["days"],
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        data = request.get_json() or {}
        days = int(data.get("days", 90))
        top_n = int(data.get("top_n", 10))
        ctr_threshold = float(data.get("ctr_threshold", 0.03))
        min_impressions = int(data.get("min_impressions", 100))

        analyzer = _get_analyzer(
            LOW_CTR_THRESHOLD=ctr_threshold,
            HIGH_IMPRESSION_MIN=min_impressions,
        )

        report = analyzer.full_analysis(days=days, top_n=top_n)

        if report.get("status") == "no_data":
            return jsonify({"error": report["message"]}), 404

        # Cache results
        _cache["report"] = report
        _cache["timestamp"] = datetime.now().isoformat()
        _cache["days"] = days

        return jsonify(report)

    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/low-ctr", methods=["POST"])
def api_low_ctr():
    try:
        data = request.get_json() or {}
        days = int(data.get("days", 90))
        ctr = float(data.get("ctr_threshold", 0.03))
        min_impr = int(data.get("min_impressions", 100))

        analyzer = _get_analyzer(LOW_CTR_THRESHOLD=ctr, HIGH_IMPRESSION_MIN=min_impr)
        results = analyzer.detect_low_ctr(days=days)

        return jsonify({"entries": results, "total": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gaps", methods=["POST"])
def api_gaps():
    try:
        data = request.get_json() or {}
        days = int(data.get("days", 90))
        min_impr = int(data.get("min_impressions", 200))
        max_clicks = int(data.get("max_clicks", 10))

        analyzer = _get_analyzer()
        results = analyzer.detect_impression_click_gap(
            days=days, min_impressions=min_impr, max_clicks=max_clicks
        )

        return jsonify({"entries": results, "total": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rewrite", methods=["POST"])
def api_rewrite():
    """New SERP-based title generator: keyword → SERP crawl → AI titles."""
    try:
        if not settings.openai.api_key:
            return jsonify({"error": "Chưa cấu hình OpenAI API Key. Vào Cài đặt để nhập API key."}), 400
        if not settings.serpapi.api_key:
            return jsonify({"error": "Chưa cấu hình SerpAPI Key. Vào Cài đặt để nhập SerpAPI key."}), 400

        data = request.get_json() or {}
        keyword = data.get("keyword", "").strip()
        num_titles = int(data.get("num_titles", 5))
        language = data.get("language", "vi")

        if not keyword:
            return jsonify({"error": "Vui lòng nhập từ khoá."}), 400

        from core.serp.title_generator import SerpTitleGenerator
        generator = SerpTitleGenerator()
        result = generator.generate(keyword=keyword, num_titles=num_titles, language=language)

        if result.get("error"):
            return jsonify({"error": result["error"]}), 404

        return jsonify(result)
    except Exception as e:
        logger.exception("Title generation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/page", methods=["POST"])
def api_page():
    try:
        data = request.get_json() or {}
        url = data.get("url", "")
        days = int(data.get("days", 90))

        if not url:
            return jsonify({"error": "URL is required"}), 400

        analyzer = _get_analyzer()
        results = analyzer.fetch_page_performance(url, days=days)

        total_clicks = sum(r["clicks"] for r in results)
        total_impr = sum(r["impressions"] for r in results)
        avg_ctr = total_clicks / total_impr if total_impr else 0

        return jsonify({
            "url": url,
            "queries": results,
            "total_queries": len(results),
            "total_clicks": total_clicks,
            "total_impressions": total_impr,
            "avg_ctr": round(avg_ctr, 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/save-settings", methods=["POST"])
def api_save_settings():
    try:
        data = request.get_json() or {}
        creds_path = data.get("credentials_path", "")
        site_url = data.get("site_url", "")
        openai_key = data.get("openai_api_key", "")
        openai_model = data.get("openai_model", "")

        if not creds_path or not site_url:
            return jsonify({"error": "GSC credentials path and site URL are required"}), 400

        env_path = Path(".env")
        lines = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()

        serpapi_key = data.get("serpapi_key", "")

        env_keys = {
            "GSC_CREDENTIALS_PATH": creds_path,
            "GSC_SITE_URL": site_url,
        }
        # Only update OpenAI key if user provided a new one (not masked)
        if openai_key and not openai_key.startswith("sk-..."):
            env_keys["OPENAI_API_KEY"] = openai_key
        if openai_model:
            env_keys["OPENAI_MODEL"] = openai_model
        if serpapi_key and not serpapi_key.startswith("***"):
            env_keys["SERPAPI_KEY"] = serpapi_key

        updated = {k: False for k in env_keys}
        for i, line in enumerate(lines):
            for key, val in env_keys.items():
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={val}"
                    updated[key] = True

        # Add missing keys
        needs_gsc_header = not updated.get("GSC_CREDENTIALS_PATH", True)
        needs_openai_header = not updated.get("OPENAI_API_KEY", True) or not updated.get("OPENAI_MODEL", True)

        if needs_gsc_header and (not updated["GSC_CREDENTIALS_PATH"] or not updated["GSC_SITE_URL"]):
            lines.append("\n# Google Search Console")
        for key in ["GSC_CREDENTIALS_PATH", "GSC_SITE_URL"]:
            if key in env_keys and not updated[key]:
                lines.append(f"{key}={env_keys[key]}")

        if needs_openai_header:
            has_openai_section = any("OPENAI" in l for l in lines)
            if not has_openai_section:
                lines.append("\n# OpenAI")
        for key in ["OPENAI_API_KEY", "OPENAI_MODEL"]:
            if key in env_keys and not updated[key]:
                lines.append(f"{key}={env_keys[key]}")

        # SerpAPI section
        if "SERPAPI_KEY" in env_keys and not updated.get("SERPAPI_KEY", False):
            has_serpapi_section = any("SERPAPI" in l for l in lines)
            if not has_serpapi_section:
                lines.append("\n# SerpAPI")
            lines.append(f"SERPAPI_KEY={env_keys['SERPAPI_KEY']}")

        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Update runtime settings
        os.environ["GSC_CREDENTIALS_PATH"] = creds_path
        os.environ["GSC_SITE_URL"] = site_url
        settings.gsc.credentials_path = creds_path
        settings.gsc.site_url = site_url

        if openai_key and not openai_key.startswith("sk-..."):
            os.environ["OPENAI_API_KEY"] = openai_key
            settings.openai.api_key = openai_key
            # Reinitialize LLM client with new key
            from core.llm.llm_client import LLMClient, llm as _old_llm
            import core.llm.llm_client as llm_module
            llm_module.llm = LLMClient()

        if openai_model:
            os.environ["OPENAI_MODEL"] = openai_model
            settings.openai.model = openai_model

        if serpapi_key and not serpapi_key.startswith("***"):
            os.environ["SERPAPI_KEY"] = serpapi_key
            settings.serpapi.api_key = serpapi_key

        return jsonify({"success": True, "message": "Settings saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["POST"])
def api_export():
    try:
        if not _cache["report"]:
            return jsonify({"error": "No analysis data. Run analysis first."}), 400

        report = _cache["report"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gsc_report_{timestamp}.json"
        filepath = EXPORT_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return jsonify({"success": True, "filename": filename, "path": str(filepath)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(EXPORT_DIR, filename)


# ──────────────────────────────────────────────
# Knowledge Base API Routes
# ──────────────────────────────────────────────

@app.route("/api/documents", methods=["GET"])
def api_list_documents():
    """List all uploaded documents."""
    try:
        from core.knowledge.document_loader import list_documents
        docs = list_documents()
        return jsonify({"documents": docs, "total": len(docs)})
    except Exception as e:
        logger.exception("Failed to list documents")
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-document", methods=["POST"])
def api_upload_document():
    """Upload a document to the knowledge base. Embedding is LOCAL (free)."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file. Vui lòng chọn file để upload."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Tên file trống."}), 400

        from core.knowledge.document_loader import save_uploaded_file
        from core.knowledge.embedding_engine import embed_document_chunks

        # Save and extract text
        doc_info = save_uploaded_file(file, file.filename)

        # Create embeddings LOCALLY (free, no API key needed)
        try:
            embed_count = embed_document_chunks(doc_info["id"])
            doc_info["embeddings_created"] = embed_count
            return jsonify({
                "success": True,
                "message": f"Upload thành công! Đã tạo {embed_count} embeddings (local, miễn phí).",
                "document": doc_info,
            })
        except Exception as embed_err:
            logger.warning(f"Embedding failed (file saved OK): {embed_err}")
            doc_info["embeddings_created"] = 0
            return jsonify({
                "success": True,
                "warning": f"File đã lưu nhưng chưa tạo được embedding: {str(embed_err)}. Nhấn 'Re-embed' để thử lại.",
                "message": "Upload thành công! Nhưng chưa tạo được embedding. Nhấn Re-embed để thử lại.",
                "document": doc_info,
            })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Document upload failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete-document/<doc_id>", methods=["DELETE"])
def api_delete_document(doc_id):
    """Delete a document from the knowledge base."""
    try:
        from core.knowledge.document_loader import delete_document
        success = delete_document(doc_id)
        if success:
            return jsonify({"success": True, "message": "Đã xoá tài liệu."})
        return jsonify({"error": "Không tìm thấy tài liệu."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reembed-document/<doc_id>", methods=["POST"])
def api_reembed_document(doc_id):
    """Re-create embeddings for a document (LOCAL, free)."""
    try:
        from core.knowledge.embedding_engine import embed_document_chunks
        count = embed_document_chunks(doc_id)
        return jsonify({"success": True, "embeddings_created": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Heading Generator API Route
# ──────────────────────────────────────────────

@app.route("/api/generate-headings", methods=["POST"])
def api_generate_headings():
    """Generate SEO + EEAT heading structure from a title."""
    try:
        if not settings.openai.api_key:
            return jsonify({
                "error": "Chưa cấu hình OpenAI API Key. Vào Cài đặt để nhập API key."
            }), 400

        data = request.get_json() or {}
        title = data.get("title", "").strip()
        keyword = data.get("keyword", "").strip()
        language = data.get("language", "vi")

        if not title:
            return jsonify({"error": "Vui lòng cung cấp title."}), 400
        if not keyword:
            return jsonify({"error": "Vui lòng cung cấp keyword."}), 400

        # Optional SERP context passed from the title generator
        intent = data.get("intent", {})
        gaps = data.get("gaps", {})
        serp_results = data.get("serp_results", [])
        people_also_ask = data.get("people_also_ask", [])

        from core.content.heading_generator import HeadingGenerator
        generator = HeadingGenerator()
        result = generator.generate(
            title=title,
            keyword=keyword,
            intent=intent,
            gaps=gaps,
            serp_results=serp_results,
            people_also_ask=people_also_ask,
            language=language,
        )

        return jsonify(result)
    except Exception as e:
        logger.exception("Heading generation failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Project Management API Routes
# ──────────────────────────────────────────────

@app.route("/api/projects", methods=["GET"])
def api_list_projects():
    """List all projects with content count."""
    try:
        from core.projects import list_projects
        projects = list_projects()
        return jsonify({"projects": projects, "total": len(projects)})
    except Exception as e:
        logger.exception("Failed to list projects")
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects", methods=["POST"])
def api_create_project():
    """Create a new project."""
    try:
        from core.projects import create_project
        data = request.get_json() or {}
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        if not name:
            return jsonify({"error": "Tên project không được để trống."}), 400
        project = create_project(name, description)
        return jsonify({"success": True, "project": project})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Failed to create project")
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<int:project_id>", methods=["PUT"])
def api_update_project(project_id):
    """Update a project."""
    try:
        from core.projects import update_project
        data = request.get_json() or {}
        project = update_project(
            project_id,
            name=data.get("name"),
            description=data.get("description"),
        )
        return jsonify({"success": True, "project": project})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<int:project_id>", methods=["DELETE"])
def api_delete_project(project_id):
    """Delete a project and all its content items."""
    try:
        from core.projects import delete_project
        success = delete_project(project_id)
        if success:
            return jsonify({"success": True, "message": "Đã xoá project."})
        return jsonify({"error": "Project không tồn tại."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<int:project_id>/items", methods=["GET"])
def api_list_content_items(project_id):
    """List all content items for a project."""
    try:
        from core.projects import list_content_items
        items = list_content_items(project_id)
        return jsonify({"items": items, "total": len(items)})
    except Exception as e:
        logger.exception("Failed to list content items")
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<int:project_id>/items", methods=["POST"])
def api_create_content_item(project_id):
    """Save a content item (keyword + title + heading) to a project.
    Also supports manual blog entry with article_text / article_html."""
    try:
        from core.projects import create_content_item, update_content_item
        data = request.get_json() or {}
        keyword = data.get("keyword", "").strip()
        title = data.get("title", "").strip()
        heading_structure = data.get("heading_structure", {})
        status = data.get("status", "draft")
        priority = int(data.get("priority", 0))
        language = data.get("language", "vi").strip()

        # Optional: manual article content
        article_text = data.get("article_text", "").strip()
        article_html = data.get("article_html", "").strip()

        if not keyword:
            return jsonify({"error": "Keyword không được để trống."}), 400
        if not title:
            return jsonify({"error": "Title không được để trống."}), 400

        item = create_content_item(
            project_id=project_id,
            keyword=keyword,
            title=title,
            heading_structure=heading_structure,
            status=status,
            priority=priority,
            language=language,
        )

        # If article content was provided (manual entry), update the item
        if article_text or article_html:
            update_data = {}
            if article_text:
                update_data["article_text"] = article_text
            if article_html:
                update_data["article_html"] = article_html
            update_data["article_generated"] = 1
            update_data["status"] = "article_generated"
            item = update_content_item(item["id"], **update_data)

        return jsonify({"success": True, "item": item})
    except Exception as e:
        logger.exception("Failed to create content item")
        return jsonify({"error": str(e)}), 500


@app.route("/api/content-items/<int:item_id>", methods=["PUT"])
def api_update_content_item(item_id):
    """Update a content item (heading structure, status, etc.)."""
    try:
        from core.projects import update_content_item
        data = request.get_json() or {}
        item = update_content_item(item_id, **data)
        return jsonify({"success": True, "item": item})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/content-items/<int:item_id>/generate-article", methods=["POST"])
def api_generate_article(item_id):
    """Generate a full blog article from an approved heading structure.

    Pipeline:
        UI
        ↓ ArticleController
        ↓ PreValidationLayer
        ↓ ArticleGenerator → ArticleBuilder (17 phases)
        ↓ QualityGate (automated check)
        ↓ Renderer
        ↓ Save DB (draft)
    """
    try:
        from core.projects import _get_conn, init_projects_db, _migrate_db, update_content_item
        import json as _json

        init_projects_db()
        _migrate_db()
        conn = _get_conn()
        row = conn.execute("SELECT * FROM content_items WHERE id = ?", (item_id,)).fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Content item không tồn tại."}), 404

        item = dict(row)

        try:
            heading_structure = _json.loads(item.get("heading_structure", "{}"))
        except (ValueError, TypeError):
            heading_structure = {}

        data = request.get_json() or {}
        language = data.get("language", item.get("language", "vi") or "vi")

        # ── ArticleController orchestrates the full pipeline ──
        from core.content.article_controller import ArticleController
        controller = ArticleController()
        result = controller.generate(
            keyword=item["keyword"],
            title=item["title"],
            heading_structure=heading_structure,
            language=language,
        )

        if not result.get("success"):
            return jsonify({"error": result.get("error", "Generation failed")}), 400

        # ── Save DB (draft) ──
        update_content_item(
            item_id,
            article_text=result["article_text"],
            article_html=result["article_html"],
            internal_links=result.get("internal_links", []),
            status="article_generated",
            article_generated=1,
        )

        response = {
            "success": True,
            "article_text": result["article_text"],
            "article_html": result["article_html"],
            "internal_links": result.get("internal_links", []),
            "word_count": result.get("word_count", 0),
            "quality_check": result.get("quality_check", {}),
        }
        if result.get("validation_warnings"):
            response["warnings"] = result["validation_warnings"]
        return jsonify(response)

    except Exception as e:
        logger.exception("Article generation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<int:project_id>/batch-generate", methods=["POST"])
def api_batch_generate_articles(project_id):
    """Start batch article generation for multiple items in a project."""
    try:
        if not settings.openai.api_key:
            return jsonify({
                "error": "Chưa cấu hình OpenAI API Key. Vào Cài đặt để nhập API key."
            }), 400

        data = request.get_json() or {}
        item_ids = data.get("item_ids", [])
        skip_existing = data.get("skip_existing", True)

        if not item_ids:
            # If no specific IDs, get all items with headings but no article
            from core.projects import list_content_items
            items = list_content_items(project_id)
            for item in items:
                has_heading = item.get("heading_structure") and item["heading_structure"].get("sections")
                has_article = item.get("article_generated") == 1 and item.get("article_text")
                if has_heading and (not has_article or not skip_existing):
                    item_ids.append(item["id"])

        if not item_ids:
            return jsonify({"error": "Không có bài nào cần tạo. Tất cả đã có bài viết hoặc chưa có heading."}), 400

        # Build language map from items
        language_map = {}
        if data.get("language_map"):
            language_map = data["language_map"]

        # Create batch job
        import uuid
        job_id = str(uuid.uuid4())[:8]

        with _batch_lock:
            _batch_jobs[job_id] = {
                "status": "queued",
                "total": len(item_ids),
                "completed": 0,
                "failed": 0,
                "results": [],
                "current_item": "",
                "cancelled": False,
                "project_id": project_id,
            }

        # Start background thread
        thread = threading.Thread(
            target=_run_batch_article_generation,
            args=(job_id, item_ids, language_map),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "total": len(item_ids),
            "message": f"Đang tạo {len(item_ids)} bài viết trong nền...",
        })
    except Exception as e:
        logger.exception("Batch generation start failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch-jobs/<job_id>", methods=["GET"])
def api_batch_job_status(job_id):
    """Get status of a batch generation job."""
    with _batch_lock:
        job = _batch_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job không tồn tại."}), 404
    return jsonify(job)


@app.route("/api/batch-jobs/<job_id>/cancel", methods=["POST"])
def api_cancel_batch_job(job_id):
    """Cancel a running batch job."""
    with _batch_lock:
        job = _batch_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job không tồn tại."}), 404
        job["cancelled"] = True
    return jsonify({"success": True, "message": "Đã gửi lệnh huỷ. Job sẽ dừng sau bài hiện tại."})


@app.route("/api/content-items/<int:item_id>/article", methods=["GET"])
def api_get_article(item_id):
    """Get the generated article for a content item."""
    try:
        from core.projects import _get_conn, init_projects_db, _migrate_db
        import json as _json
        init_projects_db()
        _migrate_db()
        conn = _get_conn()
        row = conn.execute("SELECT * FROM content_items WHERE id = ?", (item_id,)).fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Content item không tồn tại."}), 404

        item = dict(row)

        internal_links = []
        try:
            internal_links = _json.loads(item.get("internal_links", "[]"))
        except (ValueError, TypeError):
            pass

        return jsonify({
            "article_text": item.get("article_text", ""),
            "article_html": item.get("article_html", ""),
            "internal_links": internal_links,
            "word_count": len(item.get("article_text", "").split()) if item.get("article_text") else 0,
            "has_article": bool(item.get("article_text")),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/content-items/<int:item_id>", methods=["DELETE"])
def api_delete_content_item(item_id):
    """Delete a content item."""
    try:
        from core.projects import delete_content_item
        success = delete_content_item(item_id)
        if success:
            return jsonify({"success": True, "message": "Đã xoá content item."})
        return jsonify({"error": "Content item không tồn tại."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Publish Accounts API Routes
# ──────────────────────────────────────────────

@app.route("/api/publish-accounts", methods=["GET"])
def api_list_publish_accounts():
    """List all publish accounts."""
    try:
        from core.projects import list_publish_accounts
        project_id = request.args.get("project_id", None, type=int)
        accounts = list_publish_accounts(project_id)
        return jsonify({"accounts": accounts, "total": len(accounts)})
    except Exception as e:
        logger.exception("Failed to list publish accounts")
        return jsonify({"error": str(e)}), 500


@app.route("/api/publish-accounts", methods=["POST"])
def api_create_publish_account():
    """Create a new publish account."""
    try:
        from core.projects import create_publish_account
        data = request.get_json() or {}
        platform = data.get("platform", "").strip()
        site_url = data.get("site_url", "").strip()

        if not platform or platform not in ("wordpress", "shopify"):
            return jsonify({"error": "Platform phải là 'wordpress' hoặc 'shopify'."}), 400
        if not site_url:
            return jsonify({"error": "Site URL không được để trống."}), 400

        account = create_publish_account(
            platform=platform,
            site_url=site_url,
            name=data.get("name", "").strip(),
            username=data.get("username", "").strip(),
            app_password=data.get("app_password", "").strip(),
            api_token=data.get("api_token", "").strip(),
            blog_id=data.get("blog_id", "").strip(),
            project_id=data.get("project_id"),
            multilingual_plugin=data.get("multilingual_plugin", "auto").strip(),
            lang_mapping=data.get("lang_mapping", {}),
        )
        return jsonify({"success": True, "account": account})
    except Exception as e:
        logger.exception("Failed to create publish account")
        return jsonify({"error": str(e)}), 500


@app.route("/api/publish-accounts/<int:account_id>", methods=["PUT"])
def api_update_publish_account(account_id):
    """Update a publish account."""
    try:
        from core.projects import update_publish_account
        data = request.get_json() or {}
        account = update_publish_account(account_id, **data)
        return jsonify({"success": True, "account": account})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/publish-accounts/<int:account_id>", methods=["DELETE"])
def api_delete_publish_account(account_id):
    """Delete a publish account."""
    try:
        from core.projects import delete_publish_account
        success = delete_publish_account(account_id)
        if success:
            return jsonify({"success": True, "message": "Đã xoá publish account."})
        return jsonify({"error": "Account không tồn tại."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/publish-accounts/<int:account_id>/test", methods=["POST"])
def api_test_publish_account(account_id):
    """Test connection to a publish account."""
    try:
        from core.publisher import test_connection
        result = test_connection(account_id)
        if result["success"]:
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/publish-accounts/<int:account_id>/categories", methods=["GET"])
def api_fetch_wp_categories(account_id):
    """Fetch WordPress categories for a publish account, optionally filtered by language."""
    try:
        from core.publisher import fetch_wp_categories
        language = request.args.get("lang", "").strip()
        result = fetch_wp_categories(account_id, language=language)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "categories": []}), 500


@app.route("/api/publish-accounts/<int:account_id>/tags", methods=["GET"])
def api_fetch_wp_tags(account_id):
    """Fetch tags for a publish account (WordPress or Shopify)."""
    try:
        from core.publisher import fetch_wp_tags, fetch_shopify_tags, get_publish_account
        account = get_publish_account(account_id)
        if not account:
            return jsonify({"error": "Account không tồn tại.", "tags": []}), 404
        platform = (account.get("platform") or "wordpress").lower()
        if platform == "shopify":
            result = fetch_shopify_tags(account_id)
        else:
            language = request.args.get("lang", "").strip()
            result = fetch_wp_tags(account_id, language=language)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "tags": []}), 500


@app.route("/api/content-items/<int:item_id>/publish", methods=["POST"])
def api_publish_article(item_id):
    """Publish an article via WordPressAdapter.

    Publish flow:
        Click Publish
        ↓
        Upload Featured Image  (already saved via /featured-image endpoint)
        ↓
        WordPressAdapter.publish()  ← uploads image to WP Media + creates post
        ↓
        Update DB status=published
    """
    try:
        from core.publisher import WordPressAdapter, publish_article
        data = request.get_json() or {}
        publish_account_id = data.get("publish_account_id")
        schedule_time = data.get("schedule_time")
        language = data.get("language", "").strip()
        extra_categories = data.get("extra_categories", []) or []
        extra_tags = data.get("extra_tags", []) or []

        if not publish_account_id:
            return jsonify({"error": "Vui lòng chọn publish account."}), 400

        # Use WordPressAdapter for WordPress accounts; fall back to publish_article for others
        adapter = WordPressAdapter(int(publish_account_id))
        categories = data.get("categories", []) or []
        tags = data.get("tags", []) or []
        result = adapter.publish(
            item_id=item_id,
            language=language,
            schedule_time=schedule_time,
            categories=categories,
            tags=tags,
            extra_category_names=extra_categories,
            extra_tag_names=extra_tags,
        )

        if result["success"]:
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.exception("Publish failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Featured Image Upload API
# ──────────────────────────────────────────────

FEATURED_IMAGES_DIR = Path("uploads/featured_images")
FEATURED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/api/content-items/<int:item_id>/featured-image", methods=["POST"])
def api_upload_featured_image(item_id):
    """Upload a featured image for a content item."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file ảnh."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Tên file trống."}), 400

        alt_text = request.form.get("alt_text", "").strip()

        # Validate extension
        from core.publisher import ALLOWED_IMAGE_EXTENSIONS
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({
                "error": f"Định dạng không hỗ trợ: {ext}. Chỉ chấp nhận: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            }), 400

        # Save file with item_id prefix for easy identification
        import hashlib
        import time
        safe_name = f"item_{item_id}_{int(time.time())}{ext}"
        save_path = FEATURED_IMAGES_DIR / safe_name
        file.save(str(save_path))

        # Validate size
        file_size = save_path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            save_path.unlink()
            return jsonify({"error": "Ảnh quá lớn. Tối đa 10MB."}), 400

        if file_size == 0:
            save_path.unlink()
            return jsonify({"error": "File ảnh trống."}), 400

        # Update content item in DB
        from core.projects import update_content_item
        abs_path = str(save_path.resolve())
        update_content_item(
            item_id,
            featured_image_path=abs_path,
            featured_image_alt=alt_text,
        )

        return jsonify({
            "success": True,
            "message": f"Đã upload ảnh đại diện ({file_size / 1024:.0f}KB).",
            "featured_image_path": abs_path,
            "featured_image_url": f"/uploads/featured_images/{safe_name}",
            "featured_image_alt": alt_text,
        })

    except Exception as e:
        logger.exception("Featured image upload failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/content-items/<int:item_id>/featured-image", methods=["DELETE"])
def api_delete_featured_image(item_id):
    """Remove the featured image from a content item."""
    try:
        from core.projects import update_content_item, _get_conn, init_projects_db, _migrate_db

        init_projects_db()
        _migrate_db()
        conn = _get_conn()
        row = conn.execute("SELECT featured_image_path FROM content_items WHERE id = ?", (item_id,)).fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Content item không tồn tại."}), 404

        # Delete file if exists
        old_path = row["featured_image_path"] if row["featured_image_path"] else ""
        if old_path and Path(old_path).exists():
            try:
                Path(old_path).unlink()
            except Exception:
                pass

        # Clear DB fields
        update_content_item(item_id, featured_image_path="", featured_image_alt="")

        return jsonify({"success": True, "message": "Đã xoá ảnh đại diện."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/content-items/<int:item_id>/featured-image", methods=["GET"])
def api_get_featured_image(item_id):
    """Get featured image info for a content item."""
    try:
        from core.projects import _get_conn, init_projects_db, _migrate_db

        init_projects_db()
        _migrate_db()
        conn = _get_conn()
        row = conn.execute(
            "SELECT featured_image_path, featured_image_alt FROM content_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Content item không tồn tại."}), 404

        img_path = row["featured_image_path"] or ""
        img_alt = row["featured_image_alt"] or ""
        has_image = bool(img_path and Path(img_path).exists())

        # Build relative URL for serving
        img_url = ""
        if has_image:
            img_url = f"/uploads/featured_images/{Path(img_path).name}"

        return jsonify({
            "has_image": has_image,
            "featured_image_path": img_path,
            "featured_image_url": img_url,
            "featured_image_alt": img_alt,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/uploads/featured_images/<path:filename>")
def serve_featured_image(filename):
    """Serve uploaded featured images."""
    return send_from_directory(str(FEATURED_IMAGES_DIR), filename)


# ──────────────────────────────────────────────
# WordPress Media — Upload & Browse
# ──────────────────────────────────────────────

TEMP_UPLOAD_DIR = Path("uploads/temp_media")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _shopify_admin_graphql_url(base_url: str, api_version: str = "2025-10") -> str:
    """Build Shopify Admin GraphQL endpoint URL."""
    return f"{base_url.rstrip('/')}/admin/api/{api_version}/graphql.json"


def _resolve_shopify_admin_base_url(base_url: str, api_token: str, api_version: str = "2025-10") -> str:
    """Resolve Shopify Admin base URL; prefer myshopify domain when available."""
    shop_resp = requests.get(
        f"{base_url.rstrip('/')}/admin/api/{api_version}/shop.json",
        headers={"X-Shopify-Access-Token": api_token},
        timeout=20,
    )
    if shop_resp.status_code != 200:
        return base_url.rstrip("/")

    shop_data = shop_resp.json().get("shop", {}) if shop_resp.content else {}
    myshopify_domain = (shop_data.get("myshopify_domain") or "").strip()
    if myshopify_domain:
        return f"https://{myshopify_domain}".rstrip("/")

    return base_url.rstrip("/")


def _shopify_graphql_request(base_url: str, api_token: str, query: str, variables: dict | None = None) -> dict:
    """Execute a Shopify GraphQL request and raise on API/user errors."""
    graphql_url = _shopify_admin_graphql_url(base_url)
    resp = requests.post(
        graphql_url,
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": api_token,
        },
        json={"query": query, "variables": variables or {}},
        timeout=45,
    )

    # Some stores work for REST on custom domain but GraphQL only on myshopify domain.
    if resp.status_code == 404:
        admin_base = _resolve_shopify_admin_base_url(base_url, api_token)
        fallback_url = _shopify_admin_graphql_url(admin_base)
        if fallback_url != graphql_url:
            resp = requests.post(
                fallback_url,
                headers={
                    "Content-Type": "application/json",
                    "X-Shopify-Access-Token": api_token,
                },
                json={"query": query, "variables": variables or {}},
                timeout=45,
            )

    if resp.status_code != 200:
        raise RuntimeError(f"Shopify GraphQL lỗi HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json() if resp.content else {}
    if data.get("errors"):
        raise RuntimeError(f"Shopify GraphQL lỗi: {data.get('errors')}")

    return data.get("data", {})


def _upload_media_to_shopify(
    base_url: str,
    api_token: str,
    upload_path: str,
    upload_filename: str,
    upload_content_type: str,
    alt_text: str = "",
) -> dict:
    """Upload image bytes to Shopify Files using staged upload + fileCreate."""
    file_size = Path(upload_path).stat().st_size

    staged_query = """
    mutation stagedUploadsCreate($input: [StagedUploadInput!]!) {
      stagedUploadsCreate(input: $input) {
        stagedTargets {
          url
          resourceUrl
          parameters {
            name
            value
          }
        }
        userErrors {
          field
          message
        }
      }
    }
    """
    staged_vars = {
        "input": [{
            "filename": upload_filename,
            "mimeType": upload_content_type,
            "httpMethod": "POST",
            "resource": "FILE",
            "fileSize": str(file_size),
        }]
    }

    staged_data = _shopify_graphql_request(base_url, api_token, staged_query, staged_vars)
    staged = staged_data.get("stagedUploadsCreate", {})
    staged_errors = staged.get("userErrors", [])
    if staged_errors:
        raise RuntimeError(f"Shopify staged upload lỗi: {staged_errors[0].get('message', staged_errors)}")

    targets = staged.get("stagedTargets", [])
    if not targets:
        raise RuntimeError("Shopify không trả về stagedTargets.")

    target = targets[0]
    upload_url = target.get("url", "")
    resource_url = target.get("resourceUrl", "")
    params = target.get("parameters", [])

    if not upload_url or not resource_url:
        raise RuntimeError("Shopify staged target không hợp lệ (thiếu url/resourceUrl).")

    form_data = {p.get("name", ""): p.get("value", "") for p in params if p.get("name")}
    with open(upload_path, "rb") as f:
        files = {"file": (upload_filename, f, upload_content_type)}
        upload_resp = requests.post(upload_url, data=form_data, files=files, timeout=120)

    if upload_resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Shopify staged POST lỗi {upload_resp.status_code}: {upload_resp.text[:300]}")

    file_create_query = """
    mutation fileCreate($files: [FileCreateInput!]!) {
      fileCreate(files: $files) {
        files {
          __typename
          ... on MediaImage {
            id
            alt
            image {
              url
              width
              height
            }
          }
          ... on GenericFile {
            id
            url
          }
        }
        userErrors {
          field
          message
        }
      }
    }
    """

    file_create_vars = {
        "files": [{
            "originalSource": resource_url,
            "contentType": "IMAGE",
            "alt": alt_text or upload_filename,
        }]
    }

    create_data = _shopify_graphql_request(base_url, api_token, file_create_query, file_create_vars)
    created = create_data.get("fileCreate", {})
    create_errors = created.get("userErrors", [])
    if create_errors:
        raise RuntimeError(f"Shopify fileCreate lỗi: {create_errors[0].get('message', create_errors)}")

    files_out = created.get("files", [])
    if not files_out:
        raise RuntimeError("Shopify fileCreate không trả về file nào.")

    file_obj = files_out[0]
    file_type = file_obj.get("__typename")
    file_id = file_obj.get("id", "")

    if file_type == "MediaImage":
        image_obj = file_obj.get("image") or {}
        source_url = image_obj.get("url", "")
        width = image_obj.get("width", 0) or 0
        height = image_obj.get("height", 0) or 0
    else:
        source_url = file_obj.get("url", "")
        width = 0
        height = 0

    # Shopify may process MediaImage asynchronously; poll shortly for final CDN URL.
    if not source_url and file_id:
        import time as _time
        poll_query = """
        query fileNode($id: ID!) {
            node(id: $id) {
                __typename
                ... on MediaImage {
                    image {
                        url
                        width
                        height
                    }
                }
            }
        }
        """
        for _ in range(6):
            _time.sleep(1)
            poll_data = _shopify_graphql_request(base_url, api_token, poll_query, {"id": file_id})
            node = poll_data.get("node") or {}
            if node.get("__typename") != "MediaImage":
                continue
            img = node.get("image") or {}
            source_url = img.get("url", "") or source_url
            width = img.get("width", 0) or width
            height = img.get("height", 0) or height
            if source_url:
                break

    # Additional fallback: scan recent media and match by ID/alt to resolve URL.
    if not source_url:
        list_query = """
        query recentFiles($first: Int!) {
            files(first: $first, query: "media_type:IMAGE", sortKey: CREATED_AT, reverse: true) {
                edges {
                    node {
                        __typename
                        ... on MediaImage {
                            id
                            alt
                            image {
                                url
                                width
                                height
                            }
                        }
                    }
                }
            }
        }
        """
        list_data = _shopify_graphql_request(base_url, api_token, list_query, {"first": 40})
        for edge in (list_data.get("files", {}) or {}).get("edges", []):
            node = edge.get("node") or {}
            if node.get("__typename") != "MediaImage":
                continue
            img = node.get("image") or {}
            node_id = node.get("id", "")
            node_alt = node.get("alt", "") or ""
            is_match = (file_id and node_id == file_id) or (alt_text and node_alt == alt_text)
            if img.get("url") and is_match:
                source_url = img.get("url", "")
                width = img.get("width", 0) or width
                height = img.get("height", 0) or height
                break

        # Final best-effort fallback: if still no match, use the latest image URL.
        if not source_url:
            for edge in (list_data.get("files", {}) or {}).get("edges", []):
                node = edge.get("node") or {}
                if node.get("__typename") != "MediaImage":
                    continue
                img = node.get("image") or {}
                if img.get("url"):
                    source_url = img.get("url", "")
                    width = img.get("width", 0) or width
                    height = img.get("height", 0) or height
                    break

    return {
        "success": True,
        "media_id": file_id,
        "source_url": source_url,
        "thumbnail_url": source_url,
        "alt_text": alt_text or upload_filename,
        "filename": upload_filename,
        "size": file_size,
        "width": width,
        "height": height,
    }


def _optimize_image_for_seo(file_path: str, keyword: str = "", max_width: int = 1600, quality: int = 82) -> dict:
    """
    Optimize an image for SEO: rename by keyword, convert to WebP, compress.

    Returns dict with 'path', 'filename', 'content_type', 'size'.
    """
    from PIL import Image
    import unicodedata
    import re as _re

    path = Path(file_path)

    # Open and optimize
    img = Image.open(path)

    # Convert RGBA to RGB if needed (WebP supports RGBA but for photos RGB is smaller)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize if too large
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)

    # Generate SEO-friendly filename from keyword
    if keyword:
        # Normalize unicode, create slug
        slug = unicodedata.normalize("NFKD", keyword)
        ascii_slug = slug.encode("ascii", "ignore").decode("ascii").strip()
        if ascii_slug:
            slug = ascii_slug
        else:
            # For non-latin keywords (Vietnamese, Chinese, etc.), keep unicode letters
            slug = _re.sub(r'[^\w\s-]', '', keyword, flags=_re.UNICODE)
        slug = _re.sub(r'[-\s]+', '-', slug).strip('-').lower()
        if not slug:
            slug = path.stem
        import time
        seo_filename = f"{slug}-{int(time.time())}.webp"
    else:
        import time
        seo_filename = f"{path.stem}-{int(time.time())}.webp"

    # Save as WebP
    out_path = path.parent / seo_filename
    img.save(str(out_path), "WEBP", quality=quality, method=4)

    return {
        "path": str(out_path),
        "filename": seo_filename,
        "content_type": "image/webp",
        "size": out_path.stat().st_size,
    }


@app.route("/api/publish-accounts/<int:account_id>/upload-media", methods=["POST"])
def api_upload_media_to_wp(account_id):
    """
    Upload an image to WordPress Media Library.

    Optimizes image (rename by keyword, convert to WebP, compress),
    then uploads to WordPress REST API.

    Form fields:
        file: image file
        alt_text: alt text for SEO
        keyword: keyword for SEO filename
    """
    try:
        from core.projects import get_publish_account

        account = get_publish_account(account_id)
        if not account:
            return jsonify({"error": "Publish account không tồn tại."}), 404

        platform = (account.get("platform") or "").lower()
        if platform not in ("wordpress", "shopify"):
            return jsonify({"error": "Platform không được hỗ trợ cho upload ảnh."}), 400

        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file ảnh."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Tên file trống."}), 400

        alt_text = request.form.get("alt_text", "").strip()
        keyword = request.form.get("keyword", "").strip()

        # Validate extension
        from core.publisher import ALLOWED_IMAGE_EXTENSIONS
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({
                "error": f"Định dạng không hỗ trợ: {ext}. Chỉ chấp nhận: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            }), 400

        # Save temp file
        import time as _time
        temp_name = f"temp_{int(_time.time())}_{file.filename}"
        temp_path = TEMP_UPLOAD_DIR / temp_name
        file.save(str(temp_path))

        # Validate size
        file_size = temp_path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            temp_path.unlink(missing_ok=True)
            return jsonify({"error": "Ảnh quá lớn. Tối đa 10MB."}), 400
        if file_size == 0:
            temp_path.unlink(missing_ok=True)
            return jsonify({"error": "File ảnh trống."}), 400

        # Optimize image (rename, convert to webp, compress)
        try:
            optimized = _optimize_image_for_seo(str(temp_path), keyword=keyword)
            upload_path = optimized["path"]
            upload_filename = optimized["filename"]
            upload_content_type = optimized["content_type"]
        except Exception as e:
            logger.warning(f"Image optimization failed, using original: {e}")
            upload_path = str(temp_path)
            upload_filename = file.filename
            ext_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
            upload_content_type = ext_map.get(ext, "image/png")

        base_url = account["site_url"].rstrip("/")
        username = account.get("username", "")
        app_password = account.get("app_password_decrypted", "")
        api_token = account.get("api_token_decrypted", "")

        try:
            with open(upload_path, "rb") as f:
                image_data = f.read()

            # Auto-fill alt text from keyword if user didn't provide one
            effective_alt = alt_text or keyword

            if platform == "wordpress":
                media_url = f"{base_url}/wp-json/wp/v2/media"
                resp = requests.post(
                    media_url,
                    data=image_data,
                    auth=(username, app_password),
                    headers={
                        "Content-Type": upload_content_type,
                        "Content-Disposition": f'attachment; filename="{upload_filename}"',
                    },
                    timeout=60,
                )

                if resp.status_code not in (200, 201):
                    return jsonify({"error": f"WordPress upload thất bại: {resp.status_code} — {resp.text[:300]}"}), 400

                media = resp.json()
                media_id = media.get("id")
                # Prefer source_url, fall back to guid.rendered
                source_url = media.get("source_url", "") or media.get("guid", {}).get("rendered", "")

                # Set alt text + title on WP media
                if effective_alt and media_id:
                    try:
                        requests.post(
                            f"{media_url}/{media_id}",
                            json={"alt_text": effective_alt, "title": effective_alt},
                            auth=(username, app_password),
                            timeout=10,
                        )
                    except Exception:
                        pass

                return jsonify({
                    "success": True,
                    "media_id": media_id,
                    "source_url": source_url,
                    "alt_text": effective_alt,
                    "filename": upload_filename,
                    "size": len(image_data),
                })

            # Shopify upload via GraphQL Files API
            result = _upload_media_to_shopify(
                base_url=base_url,
                api_token=api_token,
                upload_path=upload_path,
                upload_filename=upload_filename,
                upload_content_type=upload_content_type,
                alt_text=effective_alt,
            )
            return jsonify(result)

        finally:
            # Cleanup temp files
            temp_path.unlink(missing_ok=True)
            opt_path = Path(upload_path)
            if opt_path.exists() and opt_path != temp_path:
                opt_path.unlink(missing_ok=True)

    except Exception as e:
        logger.exception("Media upload failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/publish-accounts/<int:account_id>/media", methods=["GET"])
def api_browse_wp_media(account_id):
    """
    Browse WordPress Media Library.

    Query params:
        page: page number (default 1)
        per_page: items per page (default 20, max 100)
        search: search term
        mime_type: filter by mime type (default: image)
    """
    try:
        from core.projects import get_publish_account

        account = get_publish_account(account_id)
        if not account:
            return jsonify({"error": "Publish account không tồn tại."}), 404

        platform = (account.get("platform") or "").lower()
        if platform not in ("wordpress", "shopify"):
            return jsonify({"error": "Platform không được hỗ trợ cho Media Library."}), 400

        base_url = account["site_url"].rstrip("/")
        username = account.get("username", "")
        app_password = account.get("app_password_decrypted", "")
        api_token = account.get("api_token_decrypted", "")

        page = request.args.get("page", 1, type=int)
        per_page = min(request.args.get("per_page", 20, type=int), 100)
        search = request.args.get("search", "").strip()

        if platform == "wordpress":
            params = {
                "page": page,
                "per_page": per_page,
                "mime_type": "image",
                "orderby": "date",
                "order": "desc",
            }
            if search:
                params["search"] = search

            resp = requests.get(
                f"{base_url}/wp-json/wp/v2/media",
                params=params,
                auth=(username, app_password),
                timeout=30,
            )

            if resp.status_code != 200:
                return jsonify({"error": f"WordPress API lỗi: {resp.status_code}"}), 400

            media_items = resp.json()
            total = int(resp.headers.get("X-WP-Total", 0))
            total_pages = int(resp.headers.get("X-WP-TotalPages", 0))

            results = []
            for item in media_items:
                sizes = item.get("media_details", {}).get("sizes", {})
                thumbnail_url = ""
                if "thumbnail" in sizes:
                    thumbnail_url = sizes["thumbnail"].get("source_url", "")
                elif "medium" in sizes:
                    thumbnail_url = sizes["medium"].get("source_url", "")
                else:
                    thumbnail_url = item.get("source_url", "")

                results.append({
                    "id": item.get("id"),
                    "source_url": item.get("source_url", ""),
                    "thumbnail_url": thumbnail_url,
                    "title": item.get("title", {}).get("rendered", ""),
                    "alt_text": item.get("alt_text", ""),
                    "caption": item.get("caption", {}).get("rendered", ""),
                    "width": item.get("media_details", {}).get("width", 0),
                    "height": item.get("media_details", {}).get("height", 0),
                    "date": item.get("date", ""),
                    "mime_type": item.get("mime_type", ""),
                })
        else:
            # Shopify Files query (MediaImage)
            gql_query_filter = "media_type:IMAGE"
            if search:
                gql_query_filter += f" AND filename:*{search}*"

            gql = """
            query listFiles($first: Int!, $after: String, $query: String!) {
              files(first: $first, after: $after, query: $query, sortKey: CREATED_AT, reverse: true) {
                edges {
                  cursor
                  node {
                    __typename
                    ... on MediaImage {
                      id
                      alt
                      createdAt
                      image {
                        url
                        width
                        height
                      }
                    }
                  }
                }
                pageInfo {
                  hasNextPage
                  endCursor
                }
              }
            }
            """

            # Manual pagination to emulate page/per_page for existing frontend contract.
            target_count = page * per_page
            after_cursor = None
            collected = []
            has_next = False

            while len(collected) < target_count:
                data = _shopify_graphql_request(
                    base_url,
                    api_token,
                    gql,
                    {"first": min(100, target_count - len(collected) + 20), "after": after_cursor, "query": gql_query_filter},
                )
                files = (data.get("files") or {})
                edges = files.get("edges", [])
                page_info = files.get("pageInfo") or {}
                has_next = bool(page_info.get("hasNextPage"))

                if not edges:
                    break

                collected.extend(edges)
                after_cursor = page_info.get("endCursor")
                if not has_next:
                    break

            start_idx = (page - 1) * per_page
            page_edges = collected[start_idx:start_idx + per_page]

            results = []
            for edge in page_edges:
                node = edge.get("node") or {}
                if node.get("__typename") != "MediaImage":
                    continue
                image_obj = node.get("image") or {}
                source_url = image_obj.get("url", "")
                results.append({
                    "id": node.get("id", ""),
                    "source_url": source_url,
                    "thumbnail_url": source_url,
                    "title": node.get("alt", "") or "Shopify Image",
                    "alt_text": node.get("alt", "") or "",
                    "caption": "",
                    "width": image_obj.get("width", 0) or 0,
                    "height": image_obj.get("height", 0) or 0,
                    "date": node.get("createdAt", ""),
                    "mime_type": "image",
                })

            # We don't get exact total cheaply without extra queries; provide best effort.
            total = len(collected) + (1 if has_next else 0)
            total_pages = max(page, (total + per_page - 1) // per_page) if total else page

        return jsonify({
            "media": results,
            "total": total,
            "total_pages": total_pages,
            "page": page,
            "per_page": per_page,
        })

    except Exception as e:
        logger.exception("Browse media failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

def open_browser(port):
    """Open browser after a short delay."""
    import time
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GSC Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print(f"\n  📊 GSC Dashboard starting at http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    app.run(host="0.0.0.0", port=args.port, debug=False)
