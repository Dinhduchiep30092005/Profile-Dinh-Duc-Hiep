"""
Publisher Module — Publish articles to WordPress / Shopify
==========================================================
Handles direct API calls to WordPress REST API and Shopify Admin API.
Supports manual featured image upload from local files.

Flow:
  1. Load article from DB
  2. Check if featured_image_path exists
  3. Upload featured image to WordPress Media API
  4. Set alt text on media
  5. Create or update post with featured_media = media_id
  6. Save wp_post_id + published_url
"""

import base64
import json
import logging
import os
import re
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from core.projects import (
    get_publish_account,
    update_content_item,
)

logger = logging.getLogger(__name__)

# Allowed image formats and max size (10MB)
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
SHOPIFY_API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2025-10")


def _shopify_api_url(base_url: str, path: str) -> str:
    """Build a Shopify Admin REST API URL with configurable API version."""
    return f"{base_url}/admin/api/{SHOPIFY_API_VERSION}/{path.lstrip('/')}"


def _resolve_shopify_admin_base_url(base_url: str, api_token: str) -> str:
    """Resolve Shopify Admin base URL; prefer myshopify domain when available."""
    clean_base = base_url.rstrip("/")
    try:
        resp = requests.get(
            _shopify_api_url(clean_base, "shop.json"),
            headers={"X-Shopify-Access-Token": api_token},
            timeout=15,
        )
        if resp.status_code != 200:
            return clean_base
        shop = (resp.json() or {}).get("shop", {})
        myshopify_domain = (shop.get("myshopify_domain") or "").strip()
        if myshopify_domain:
            return f"https://{myshopify_domain}".rstrip("/")
    except Exception:
        pass
    return clean_base


class PublishResult:
    """Result of a publish operation."""

    def __init__(self, success: bool, url: str = "", error: str = "", post_id: str = "", slug: str = ""):
        self.success = success
        self.url = url
        self.error = error
        self.post_id = post_id
        self.slug = slug


# ──────────────────────────────────────────────
# Image Validation
# ──────────────────────────────────────────────


def validate_image(image_path: str) -> dict:
    """
    Validate an image file before upload.

    Returns dict with 'valid' (bool), 'error' (str), 'size' (int), 'ext' (str).
    """
    if not image_path:
        return {"valid": False, "error": "Đường dẫn ảnh trống."}

    path = Path(image_path)
    if not path.exists():
        return {"valid": False, "error": f"File không tồn tại: {image_path}"}

    ext = path.suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return {
            "valid": False,
            "error": f"Định dạng ảnh không hỗ trợ: {ext}. Chỉ chấp nhận: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        }

    size = path.stat().st_size
    if size > MAX_IMAGE_SIZE:
        return {
            "valid": False,
            "error": f"Ảnh quá lớn: {size / 1024 / 1024:.1f}MB. Tối đa {MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB.",
        }

    if size == 0:
        return {"valid": False, "error": "File ảnh trống (0 bytes)."}

    return {"valid": True, "size": size, "ext": ext}


# ──────────────────────────────────────────────
# WordPress Media Upload
# ──────────────────────────────────────────────


def _upload_featured_image_to_wp(
    base_url: str,
    username: str,
    app_password: str,
    image_path: str,
    alt_text: str = "",
    max_retries: int = 2,
) -> dict:
    """
    Upload a featured image to WordPress Media Library.

    Args:
        base_url: WordPress site URL
        username: WordPress username
        app_password: Application password
        image_path: Local path to the image file
        alt_text: Alt text for the image
        max_retries: Number of retries on failure

    Returns dict with 'media_id', 'url', 'success', 'error'.
    """
    # Validate image first
    validation = validate_image(image_path)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"]}

    path = Path(image_path)
    filename = path.name

    # Determine content type
    ext = path.suffix.lower()
    content_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    content_type = content_type_map.get(ext, "image/png")

    # Read image data
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
    except Exception as e:
        return {"success": False, "error": f"Không thể đọc file ảnh: {e}"}

    logger.info(f"[WP Upload] Uploading featured image: {filename} ({len(image_data)} bytes)")

    # Upload with retries
    media_url = f"{base_url}/wp-json/wp/v2/media"
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                media_url,
                data=image_data,
                auth=(username, app_password),
                headers={
                    "Content-Type": content_type,
                    "Content-Disposition": f'attachment; filename="{filename}"',
                },
                timeout=60,
            )

            if resp.status_code in (200, 201):
                media = resp.json()
                media_id = media.get("id")
                source_url = media.get("source_url", "")

                # Set alt text
                if alt_text and media_id:
                    try:
                        requests.post(
                            f"{media_url}/{media_id}",
                            json={"alt_text": alt_text},
                            auth=(username, app_password),
                            timeout=10,
                        )
                        logger.info(f"[WP Upload] Alt text set: '{alt_text}'")
                    except Exception:
                        logger.warning("[WP Upload] Failed to set alt text (non-critical)")

                logger.info(f"[WP Upload] ✓ Featured image uploaded: media_id={media_id}, URL={source_url}")
                return {"success": True, "media_id": media_id, "url": source_url}
            else:
                last_error = f"WP Media API {resp.status_code}: {resp.text[:200]}"
                logger.warning(f"[WP Upload] Attempt {attempt + 1} failed: {last_error}")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"[WP Upload] Attempt {attempt + 1} exception: {e}")

        # Wait before retry
        if attempt < max_retries:
            import time
            time.sleep(2 * (attempt + 1))

    return {"success": False, "error": f"Upload ảnh thất bại sau {max_retries + 1} lần thử: {last_error}"}


# ──────────────────────────────────────────────
# Main Publish Entry Point
# ──────────────────────────────────────────────


def publish_article(
    item_id: int,
    publish_account_id: int,
    schedule_time: str = None,
    language: str = "",
    categories: list = None,
    tags: list = None,
    extra_category_names: list = None,
    extra_tag_names: list = None,
) -> dict:
    """
    Publish an article to WordPress with featured image.
    Supports multilingual publishing via Polylang / WPML.

    Pipeline:
      1. Load article from DB
      2. Check featured_image_path
      3. Set publish_status = publishing
      4. Validate image (size / format)
      5. Upload image to WordPress Media API → get media_id
      6. Set alt text (if provided)
      7. Create or update post with featured_media = media_id
         (with language param for Polylang/WPML)
      8. Save wp_post_id + published_url
      9. Set publish_status = published

    If error at any step:
      - Set publish_status = failed
      - Save error_message

    Args:
        item_id: Content item ID
        publish_account_id: Publish account ID
        schedule_time: Optional ISO datetime for scheduled publishing
        language: Language code (e.g. 'vi', 'en') for multilingual sites

    Returns:
        dict with success status, published_url or error_message
    """
    from core.projects import _get_conn, init_projects_db, _migrate_db

    init_projects_db()
    _migrate_db()

    # 1. Load article
    conn = _get_conn()
    row = conn.execute("SELECT * FROM content_items WHERE id = ?", (item_id,)).fetchone()
    conn.close()

    if not row:
        return {"success": False, "error": "Article không tồn tại."}

    item = dict(row)
    article_html = item.get("article_html", "")
    title = item.get("title", "")
    featured_image_path = item.get("featured_image_path", "")
    featured_image_alt = item.get("featured_image_alt", "")
    existing_wp_post_id = item.get("wp_post_id", "")
    url_slug = item.get("url_slug", "") or ""
    categories = categories or []  # Use categories passed from request
    tags = tags or []  # Use tags passed from request
    extra_category_names = extra_category_names or []
    extra_tag_names = extra_tag_names or []

    # Use provided language or fall back to item's language
    if not language:
        language = item.get("language", "vi") or "vi"

    if not article_html:
        return {"success": False, "error": "Bài viết chưa được tạo. Hãy viết bài trước khi đăng."}

    # 2. Check featured image
    if not featured_image_path:
        return {"success": False, "error": "Chưa upload ảnh đại diện. Hãy upload Featured Image trước khi đăng."}

    # 3. Set publish_status = publishing
    update_content_item(item_id, publish_status="publishing", publish_account_id=publish_account_id)

    # 4. Load publish account
    account = get_publish_account(publish_account_id)
    if not account:
        update_content_item(item_id, publish_status="failed", error_message="Publish account không tồn tại.")
        return {"success": False, "error": "Publish account không tồn tại."}

    if account.get("status") != "active":
        update_content_item(item_id, publish_status="failed", error_message="Publish account đang bị tắt.")
        return {"success": False, "error": "Publish account đang bị tắt (inactive)."}

    platform = account.get("platform", "").lower()
    multilingual_plugin = account.get("multilingual_plugin", "auto") or "auto"
    lang_mapping = account.get("lang_mapping", {}) or {}

    # Resolve language: use lang_mapping if available
    resolved_lang = language
    if lang_mapping and language in lang_mapping:
        resolved_lang = lang_mapping[language]
        if resolved_lang != language:
            logger.info(f"[Publish] Language mapped: {language} → {resolved_lang} (via lang_mapping)")

    # 5. Publish based on platform
    try:
        if platform == "wordpress":
            categories = _resolve_wp_term_ids(
                base_url=account["site_url"].rstrip("/"),
                username=account.get("username", ""),
                app_password=account.get("app_password_decrypted", ""),
                endpoint="categories",
                existing_ids=categories,
                manual_names=extra_category_names,
                language=resolved_lang,
                multilingual_plugin=multilingual_plugin,
            )
            tags = _resolve_wp_term_ids(
                base_url=account["site_url"].rstrip("/"),
                username=account.get("username", ""),
                app_password=account.get("app_password_decrypted", ""),
                endpoint="tags",
                existing_ids=tags,
                manual_names=extra_tag_names,
                language=resolved_lang,
                multilingual_plugin=multilingual_plugin,
            )
            result = _publish_to_wordpress(
                site_url=account["site_url"],
                username=account.get("username", ""),
                app_password=account.get("app_password_decrypted", ""),
                title=title,
                content=article_html,
                featured_image_path=featured_image_path,
                featured_image_alt=featured_image_alt or title,
                existing_post_id=existing_wp_post_id,
                slug=url_slug,
                schedule_time=schedule_time,
                language=resolved_lang,
                multilingual_plugin=multilingual_plugin,
                categories=categories or [],
                tags=tags or [],
            )
        elif platform == "shopify":
            result = _publish_to_shopify(
                site_url=account["site_url"],
                api_token=account.get("api_token_decrypted", ""),
                blog_id=account.get("blog_id", ""),
                title=title,
                content=article_html,
                featured_image_path=featured_image_path,
                featured_image_alt=featured_image_alt or title,
                tags=tags or [],
            )
        else:
            result = PublishResult(success=False, error=f"Platform '{platform}' không được hỗ trợ.")

        # 6. Update status
        if result.success:
            update_content_item(
                item_id,
                publish_status="published",
                published_url=result.url,
                url_slug=result.slug,
                wp_post_id=result.post_id,
                error_message="",
            )
            return {
                "success": True,
                "published_url": result.url,
                "post_id": result.post_id,
                "message": f"Đã đăng bài thành công lên {platform}!",
            }
        else:
            update_content_item(
                item_id,
                publish_status="failed",
                error_message=result.error,
            )
            return {"success": False, "error": result.error}

    except Exception as e:
        error_msg = f"Lỗi khi đăng bài: {str(e)}"
        logger.exception(error_msg)
        update_content_item(
            item_id,
            publish_status="failed",
            error_message=error_msg,
        )
        return {"success": False, "error": error_msg}


# ──────────────────────────────────────────────
# WordPress REST API (with featured image)
# ──────────────────────────────────────────────


def _publish_to_wordpress(
    site_url: str,
    username: str,
    app_password: str,
    title: str,
    content: str,
    featured_image_path: str = "",
    featured_image_alt: str = "",
    existing_post_id: str = "",
    slug: str = "",
    schedule_time: str = None,
    language: str = "",
    multilingual_plugin: str = "auto",
    categories: list = None,
    tags: list = None,
) -> PublishResult:
    """
    Publish to WordPress via REST API with featured image upload.
    Supports multilingual plugins (Polylang / WPML).

    Steps:
      1. Validate & upload featured image to WP Media Library
      2. Set alt text on the media
      3. Create or update post with featured_media = media_id
      4. Set language via Polylang (lang) / WPML (wpml_language)
    """
    base_url = site_url.rstrip("/")

    # ── Step 1: Upload featured image ──
    featured_media_id = None

    if featured_image_path:
        upload_result = _upload_featured_image_to_wp(
            base_url=base_url,
            username=username,
            app_password=app_password,
            image_path=featured_image_path,
            alt_text=featured_image_alt,
        )

        if upload_result.get("success"):
            featured_media_id = upload_result["media_id"]
            logger.info(f"[WP] Featured image uploaded: media_id={featured_media_id}")
        else:
            return PublishResult(
                success=False,
                error=f"Upload ảnh đại diện thất bại: {upload_result.get('error', 'unknown')}",
            )

    # ── Step 2: Create or update post ──
    post_data = {
        "title": title,
        "content": content,
        "status": "publish",
    }

    if slug:
        post_data["slug"] = slug

    if categories:
        post_data["categories"] = [int(c) for c in categories]

    if tags:
        post_data["tags"] = [int(t) for t in tags]

    if schedule_time:
        post_data["status"] = "future"
        post_data["date"] = schedule_time

    if featured_media_id:
        post_data["featured_media"] = featured_media_id

    # ── Multilingual support (Polylang / WPML) ──
    if language:
        plugin = multilingual_plugin.lower() if multilingual_plugin else "auto"

        # Normalize language code for Polylang: "en-us", "en-au", "en-gb" → "en"
        # Polylang uses short 2-letter slugs; WPML may accept full codes.
        # Preserve special cases like "zh-tw" (Traditional Chinese) vs "zh" (Simplified).
        _PRESERVE_FULL = {"zh-tw", "zh-hk", "pt-br"}
        if plugin in ("polylang", "auto") and "-" in language and language.lower() not in _PRESERVE_FULL:
            normalized_lang = language.split("-")[0]
            if normalized_lang != language:
                logger.info(f"[WP] Language normalized for Polylang: {language} → {normalized_lang}")
            language = normalized_lang

        if plugin == "polylang":
            # Polylang: set lang in body only (NOT in URL query)
            post_data["lang"] = language
            logger.info(f"[WP] Setting language: {language} (Polylang)")
        elif plugin == "wpml":
            # WPML only
            post_data["wpml_language"] = language
            logger.info(f"[WP] Setting language: {language} (WPML)")
        elif plugin == "none":
            # No multilingual plugin — skip language params
            logger.info(f"[WP] Skipping language param (plugin=none)")
        else:
            # Auto: send both for maximum compatibility
            post_data["lang"] = language
            post_data["wpml_language"] = language
            logger.info(f"[WP] Setting language: {language} (auto — Polylang + WPML compat)")

    # Decide: create new or update existing
    if existing_post_id:
        api_url = f"{base_url}/wp-json/wp/v2/posts/{existing_post_id}"
        logger.info(f"[WP] Updating existing post: {existing_post_id}")
    else:
        api_url = f"{base_url}/wp-json/wp/v2/posts"
        logger.info("[WP] Creating new post")

    try:
        response = requests.post(
            api_url,
            json=post_data,
            auth=(username, app_password),
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code in (200, 201):
            data = response.json()
            post_url = data.get("link", "")
            post_id = str(data.get("id", ""))
            post_slug = data.get("slug", "")
            # Clean any Polylang/WPML ?lang= query param from the canonical URL
            if post_url and "?" in post_url:
                from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
                _p = urlparse(post_url)
                _qs = parse_qs(_p.query)
                _qs.pop("lang", None)
                post_url = urlunparse(_p._replace(query=urlencode(_qs, doseq=True)))
            action = "Updated" if existing_post_id else "Published"
            logger.info(f"[WP] \u2713 {action}: {post_url} (post_id={post_id}, slug={post_slug}, featured_media={featured_media_id})")
            return PublishResult(success=True, url=post_url, post_id=post_id, slug=post_slug)
        else:
            error_detail = ""
            try:
                err_data = response.json()
                error_detail = err_data.get("message", response.text[:200])
            except Exception:
                error_detail = response.text[:200]
            return PublishResult(
                success=False,
                error=f"WordPress API lỗi ({response.status_code}): {error_detail}",
            )
    except requests.exceptions.ConnectionError:
        return PublishResult(success=False, error=f"Không thể kết nối tới {base_url}. Kiểm tra URL.")
    except requests.exceptions.Timeout:
        return PublishResult(success=False, error="Timeout khi kết nối WordPress.")
    except Exception as e:
        return PublishResult(success=False, error=f"Lỗi WordPress: {str(e)}")


# ──────────────────────────────────────────────
# Shopify Admin API (with featured image)
# ──────────────────────────────────────────────


def _publish_to_shopify(
    site_url: str,
    api_token: str,
    blog_id: str,
    title: str,
    content: str,
    featured_image_path: str = "",
    featured_image_alt: str = "",
    tags: list = None,
) -> PublishResult:
    """
    Publish to Shopify via Admin API with featured image.

    Steps:
      1. Read & encode featured image as base64
      2. Create article with body_html + image attachment
    """
    base_url = site_url.rstrip("/")
    admin_base_url = _resolve_shopify_admin_base_url(base_url, api_token)

    # Default blog_id if not specified
    if not blog_id:
        blog_id = _get_shopify_default_blog(base_url, api_token)
        if not blog_id:
            return PublishResult(success=False, error="Không tìm thấy Blog ID. Vui lòng nhập Blog ID trong cài đặt.")

    # ── Step 1: Prepare featured image ──
    article_data = {
        "title": title,
        "body_html": content,
        "published": True,
    }

    if tags:
        article_data["tags"] = ", ".join(str(t) for t in tags)

    # Set published_at to ensure article is immediately live
    article_data["published_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if featured_image_path:
        validation = validate_image(featured_image_path)
        if not validation["valid"]:
            return PublishResult(success=False, error=f"Featured image lỗi: {validation['error']}")

        try:
            with open(featured_image_path, "rb") as f:
                img_data = f.read()

            b64 = base64.b64encode(img_data).decode("ascii")
            article_data["image"] = {
                "attachment": b64,
                "alt": featured_image_alt or title,
            }
            logger.info(f"[Shopify] Featured image prepared: {Path(featured_image_path).name} ({len(img_data)} bytes)")
        except Exception as e:
            return PublishResult(success=False, error=f"Không thể đọc ảnh đại diện: {e}")

    # ── Step 2: Create article ──
    api_url = _shopify_api_url(admin_base_url, f"blogs/{blog_id}/articles.json")
    post_data = {"article": article_data}

    logger.info(f"[Shopify] POST {api_url}")
    logger.info(f"[Shopify] Payload keys: {list(article_data.keys())}, title={title!r}, blog_id={blog_id}")

    try:
        response = requests.post(
            api_url,
            json=post_data,
            headers={
                "Content-Type": "application/json",
                "X-Shopify-Access-Token": api_token,
            },
            timeout=30,
            allow_redirects=False,
        )

        if response.status_code in (301, 302, 307, 308):
            location = response.headers.get("Location", "")
            return PublishResult(
                success=False,
                error=(
                    "Shopify Admin API bị redirect khi publish. "
                    f"Status={response.status_code}, Location={location or 'N/A'}. "
                    "Đã chặn redirect để tránh POST bị đổi thành GET."
                ),
            )

        logger.info(f"[Shopify] Response status: {response.status_code}")
        try:
            resp_json = response.json()
            logger.info(f"[Shopify] Response JSON: {str(resp_json)[:500]}")
        except Exception:
            logger.info(f"[Shopify] Response text: {response.text[:500]}")

        if response.status_code in (200, 201):
            try:
                data = response.json()
            except Exception:
                return PublishResult(
                    success=False,
                    error=f"Shopify trả về dữ liệu không phải JSON: {response.text[:240]}",
                )

            if not isinstance(data, dict):
                return PublishResult(success=False, error="Shopify response không hợp lệ (không phải object JSON).")

            # Some edge cases return HTTP 200 with top-level errors and no article payload.
            if data.get("errors") and not data.get("article"):
                return PublishResult(success=False, error=f"Shopify API lỗi: {data.get('errors')}")

            # Some misrouted responses may return list payload {'articles': [...]} instead of {'article': {...}}.
            if data.get("articles") and not data.get("article"):
                return PublishResult(
                    success=False,
                    error=(
                        "Shopify trả về danh sách 'articles' thay vì 'article' khi publish. "
                        "Thường do request bị redirect sai admin domain."
                    ),
                )

            article = data.get("article")
            if not isinstance(article, dict):
                article = {}

            handle = article.get("handle", "")
            article_id = str(article.get("id", "") or "")
            published_at = article.get("published_at")
            status = article.get("published")
            logger.info(f"[Shopify] Article created: id={article_id}, handle={handle!r}, published={status}, published_at={published_at}")

            # Fallback: find the most recent article by exact title if Shopify omitted id/handle.
            if not article_id or not handle:
                try:
                    lookup_resp = requests.get(
                        _shopify_api_url(admin_base_url, f"blogs/{blog_id}/articles.json"),
                        params={"limit": 20, "published_status": "any"},
                        headers={"X-Shopify-Access-Token": api_token},
                        timeout=12,
                    )
                    if lookup_resp.status_code == 200:
                        recent_articles = lookup_resp.json().get("articles", [])
                        exact_matches = [a for a in recent_articles if (a.get("title") or "").strip() == title.strip()]
                        if exact_matches:
                            candidate = sorted(exact_matches, key=lambda a: a.get("created_at", ""), reverse=True)[0]
                            article_id = article_id or str(candidate.get("id", "") or "")
                            handle = handle or (candidate.get("handle", "") or "")
                            logger.info(f"[Shopify] Recovered article identity from recent list: id={article_id}, handle={handle!r}")
                except Exception as lookup_err:
                    logger.warning(f"[Shopify] Could not recover article identity: {lookup_err}")

            if not article_id:
                return PublishResult(
                    success=False,
                    error=f"Shopify trả về thành công nhưng thiếu article.id. Response: {str(data)[:280]}",
                )

            # Fetch blog handle for correct URL (Shopify URLs use handle, not numeric ID)
            blog_handle = blog_id  # fallback
            try:
                blog_resp = requests.get(
                    _shopify_api_url(admin_base_url, f"blogs/{blog_id}.json"),
                    headers={"X-Shopify-Access-Token": api_token},
                    timeout=10,
                )
                if blog_resp.status_code == 200:
                    blog_data = blog_resp.json().get("blog", {})
                    blog_handle = blog_data.get("handle", blog_id)
                    logger.info(f"[Shopify] Blog handle resolved: {blog_id} → {blog_handle!r}")
            except Exception:
                pass

            if not handle:
                # Keep success if post exists, but provide a usable admin URL fallback.
                post_url = f"{base_url}/admin/articles/{article_id}"
            else:
                post_url = f"{base_url}/blogs/{blog_handle}/{handle}"

            logger.info(f"[Shopify] ✓ Published: {post_url}")
            return PublishResult(success=True, url=post_url, post_id=article_id)
        else:
            error_detail = ""
            try:
                err_data = response.json()
                errors = err_data.get("errors", response.text[:200])
                error_detail = str(errors)
            except Exception:
                error_detail = response.text[:200]
            return PublishResult(
                success=False,
                error=f"Shopify API lỗi ({response.status_code}): {error_detail}",
            )
    except requests.exceptions.ConnectionError:
        return PublishResult(success=False, error=f"Không thể kết nối tới {base_url}. Kiểm tra URL.")
    except requests.exceptions.Timeout:
        return PublishResult(success=False, error="Timeout khi kết nối Shopify.")
    except Exception as e:
        return PublishResult(success=False, error=f"Lỗi Shopify: {str(e)}")


def _get_shopify_default_blog(base_url: str, api_token: str) -> Optional[str]:
    """Get the default blog ID from Shopify."""
    try:
        response = requests.get(
            _shopify_api_url(base_url, "blogs.json"),
            headers={
                "Content-Type": "application/json",
                "X-Shopify-Access-Token": api_token,
            },
            timeout=15,
        )
        if response.status_code == 200:
            blogs = response.json().get("blogs", [])
            if blogs:
                return str(blogs[0]["id"])
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────
# WordPressAdapter — OOP wrapper for the publish flow
# ──────────────────────────────────────────────


class WordPressAdapter:
    """
    Adapter that wraps the WordPress publish pipeline as an object.

    Publish flow:
        Click Publish
        ↓
        Upload Featured Image   (via WP Media API)
        ↓
        WordPressAdapter.publish()
        ↓
        Update DB status=published

    Usage:
        adapter = WordPressAdapter(account_id)
        result  = adapter.publish(item_id, language="en")
    """

    def __init__(self, publish_account_id: int):
        self.publish_account_id = publish_account_id
        self._account = None  # lazy-loaded

    def _load_account(self) -> dict:
        if self._account is None:
            self._account = get_publish_account(self.publish_account_id)
            if not self._account:
                raise ValueError(f"Publish account #{self.publish_account_id} không tồn tại.")
        return self._account

    def upload_featured_image(self, image_path: str, alt_text: str = "") -> dict:
        """
        Upload a local featured image to WordPress Media Library.

        Returns:
            dict with success (bool), media_id (int), url (str), error (str)
        """
        account = self._load_account()
        if account.get("platform", "").lower() != "wordpress":
            return {"success": False, "error": "upload_featured_image chỉ hỗ trợ WordPress."}

        return _upload_featured_image_to_wp(
            base_url=account["site_url"].rstrip("/"),
            username=account.get("username", ""),
            app_password=account.get("app_password_decrypted", ""),
            image_path=image_path,
            alt_text=alt_text,
        )

    def publish(
        self,
        item_id: int,
        language: str = "",
        schedule_time: str = None,
        categories: list = None,
        tags: list = None,
        extra_category_names: list = None,
        extra_tag_names: list = None,
    ) -> dict:
        """
        Full publish pipeline for a content item.

        Steps:
          1. Upload featured image → get media_id
          2. Create / update WordPress post with featured_media
          3. Update DB: status=published, published_url, wp_post_id

        Returns:
            dict with success (bool), published_url (str), post_id (str), error (str)
        """
        return publish_article(
            item_id=item_id,
            publish_account_id=self.publish_account_id,
            schedule_time=schedule_time,
            language=language,
            categories=categories,
            tags=tags,
            extra_category_names=extra_category_names,
            extra_tag_names=extra_tag_names,
        )


# ──────────────────────────────────────────────
# Connection Test
# ──────────────────────────────────────────────


def test_connection(account_id: int) -> dict:
    """Test connection to a publish account. Auto-detects multilingual plugins."""
    from core.projects import update_publish_account

    account = get_publish_account(account_id)
    if not account:
        return {"success": False, "error": "Account không tồn tại."}

    platform = account.get("platform", "").lower()

    if platform == "wordpress":
        result = _test_wordpress_connection(
            site_url=account["site_url"],
            username=account.get("username", ""),
            app_password=account.get("app_password_decrypted", ""),
        )
        # Auto-detect multilingual plugin if connection succeeded
        if result.get("success"):
            detected = _detect_multilingual_plugin(
                site_url=account["site_url"],
                username=account.get("username", ""),
                app_password=account.get("app_password_decrypted", ""),
            )
            if detected["plugin"] != "unknown":
                update_data = {"multilingual_plugin": detected["plugin"]}
                if detected.get("languages"):
                    # Build lang_mapping from detected languages
                    lang_mapping = {lang: lang for lang in detected["languages"]}
                    update_data["lang_mapping"] = lang_mapping
                try:
                    update_publish_account(account_id, **update_data)
                except Exception:
                    pass  # Non-critical — don't fail the test

                plugin_name = detected["plugin"].capitalize()
                langs = ", ".join(detected.get("languages", []))
                result["message"] += f" | 🌐 Plugin: {plugin_name}"
                if langs:
                    result["message"] += f" ({langs})"
                result["multilingual_plugin"] = detected["plugin"]
                result["detected_languages"] = detected.get("languages", [])
        return result
    elif platform == "shopify":
        return _test_shopify_connection(
            site_url=account["site_url"],
            api_token=account.get("api_token_decrypted", ""),
        )
    else:
        return {"success": False, "error": f"Platform '{platform}' không được hỗ trợ."}


def _detect_multilingual_plugin(site_url: str, username: str, app_password: str) -> dict:
    """
    Auto-detect which multilingual plugin is active on a WordPress site.
    Checks Polylang REST API and WPML REST API endpoints.

    Returns dict with 'plugin' ('polylang', 'wpml', 'none', 'unknown') and 'languages' list.
    """
    base_url = site_url.rstrip("/")
    result = {"plugin": "unknown", "languages": []}

    # Check Polylang: GET /wp-json/pll/v1/languages (Polylang REST API plugin)
    # Also try: GET /wp-json/wp/v2/languages (Polylang Pro exposes this)
    try:
        for endpoint in [
            f"{base_url}/wp-json/pll/v1/languages",
            f"{base_url}/wp-json/wp/v2/languages",
        ]:
            resp = requests.get(endpoint, auth=(username, app_password), timeout=8)
            if resp.status_code == 200:
                langs_data = resp.json()
                if isinstance(langs_data, list) and len(langs_data) > 0:
                    # Polylang returns list of language objects
                    slugs = []
                    for lang in langs_data:
                        slug = lang.get("slug") or lang.get("locale", "")[:2]
                        if slug:
                            slugs.append(slug)
                    if slugs:
                        result["plugin"] = "polylang"
                        result["languages"] = slugs
                        logger.info(f"[Detect] Polylang detected: {slugs}")
                        return result
    except Exception as e:
        logger.debug(f"[Detect] Polylang check failed: {e}")

    # Check WPML: GET /wp-json/wpml/v1/languages (WPML REST API)
    try:
        resp = requests.get(
            f"{base_url}/wp-json/wpml/v1/languages",
            auth=(username, app_password),
            timeout=8,
        )
        if resp.status_code == 200:
            langs_data = resp.json()
            if isinstance(langs_data, (list, dict)):
                slugs = []
                # WPML may return a dict of code -> lang_info or a list
                if isinstance(langs_data, dict):
                    slugs = list(langs_data.keys())
                elif isinstance(langs_data, list):
                    for lang in langs_data:
                        code = lang.get("code") or lang.get("language_code", "")
                        if code:
                            slugs.append(code)
                if slugs:
                    result["plugin"] = "wpml"
                    result["languages"] = slugs
                    logger.info(f"[Detect] WPML detected: {slugs}")
                    return result
    except Exception as e:
        logger.debug(f"[Detect] WPML check failed: {e}")

    # No multilingual plugin detected
    # Check if the site has any custom taxonomy for languages
    try:
        resp = requests.get(
            f"{base_url}/wp-json/wp/v2/taxonomies",
            auth=(username, app_password),
            timeout=8,
        )
        if resp.status_code == 200:
            taxonomies = resp.json()
            if isinstance(taxonomies, dict):
                if "language" in taxonomies or "post_translations" in taxonomies:
                    result["plugin"] = "polylang"
                    logger.info("[Detect] Polylang detected via taxonomy (no REST endpoint)")
                    return result
    except Exception:
        pass

    result["plugin"] = "none"
    logger.info("[Detect] No multilingual plugin detected")
    return result


def _fetch_wp_taxonomy(
    base_url: str,
    username: str,
    app_password: str,
    endpoint: str,
    language: str = "",
    multilingual_plugin: str = "auto",
    max_pages: int = 20,
) -> list:
    """
    Paginate through a WordPress REST API taxonomy endpoint (categories or tags).
    Returns a flat list of raw items.
    """
    _PRESERVE_FULL = {"zh-tw", "zh-hk", "pt-br"}
    lang_param = ""
    if language and multilingual_plugin in ("polylang", "auto"):
        if "-" in language and language.lower() not in _PRESERVE_FULL:
            language = language.split("-")[0]
        lang_param = language

    all_items = []
    page = 1
    while page <= max_pages:
        params = {"per_page": 100, "page": page, "hide_empty": False}
        if lang_param:
            params["lang"] = lang_param
        try:
            resp = requests.get(
                f"{base_url}/wp-json/wp/v2/{endpoint}",
                params=params,
                auth=(username, app_password),
                timeout=15,
            )
        except Exception:
            break

        if resp.status_code != 200:
            break

        batch = resp.json()
        if not batch:
            break
        all_items.extend(batch)

        # Check X-WP-TotalPages header to know if more pages exist
        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        if page >= total_pages:
            break
        page += 1

    return all_items


def _normalize_manual_term_names(term_names: list) -> list:
    """Normalize a list of manual term names and remove duplicates."""
    normalized = []
    seen = set()
    for raw in term_names or []:
        name = str(raw or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)
    return normalized


def _slugify_term_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-\s]", "", s)
    s = re.sub(r"[\s\-]+", "-", s).strip("-")
    return s


def _normalize_polylang_lang(language: str) -> str:
    lang = (language or "").strip().lower()
    preserve_full = {"zh-tw", "zh-hk", "pt-br"}
    if "-" in lang and lang not in preserve_full:
        return lang.split("-")[0]
    return lang


def _find_or_create_wp_term_id(
    base_url: str,
    username: str,
    app_password: str,
    endpoint: str,
    term_name: str,
    language: str = "",
    multilingual_plugin: str = "auto",
) -> int:
    """Find or create a WP category/tag by name and return its numeric ID."""
    api_url = f"{base_url.rstrip('/')}/wp-json/wp/v2/{endpoint}"
    plugin = (multilingual_plugin or "auto").lower()
    search_params = {"search": term_name, "per_page": 100}
    if language and plugin in ("polylang", "auto"):
        search_params["lang"] = _normalize_polylang_lang(language)

    resp = requests.get(api_url, params=search_params, auth=(username, app_password), timeout=20)
    if resp.status_code == 200:
        target_name = term_name.strip().lower()
        target_slug = _slugify_term_name(term_name)
        for term in resp.json() or []:
            term_name_l = str(term.get("name", "")).strip().lower()
            term_slug = str(term.get("slug", "")).strip().lower()
            if term_name_l == target_name or (target_slug and term_slug == target_slug):
                term_id = term.get("id")
                if term_id:
                    return int(term_id)

    payload = {"name": term_name}
    if language:
        if plugin == "polylang":
            payload["lang"] = _normalize_polylang_lang(language)
        elif plugin == "wpml":
            payload["wpml_language"] = language
        elif plugin == "auto":
            payload["lang"] = _normalize_polylang_lang(language)
            payload["wpml_language"] = language

    create_resp = requests.post(
        api_url,
        json=payload,
        auth=(username, app_password),
        headers={"Content-Type": "application/json"},
        timeout=20,
    )
    if create_resp.status_code in (200, 201):
        term_id = (create_resp.json() or {}).get("id")
        if term_id:
            return int(term_id)

    err_data = {}
    try:
        err_data = create_resp.json() or {}
    except Exception:
        err_data = {}

    if err_data.get("code") == "term_exists":
        data = err_data.get("data") or {}
        term_id = data.get("term_id") or data.get("resource_id")
        if term_id:
            return int(term_id)

    raise RuntimeError(
        f"Không thể tạo {endpoint[:-1]} '{term_name}' (HTTP {create_resp.status_code}): {str(err_data)[:240]}"
    )


def _resolve_wp_term_ids(
    base_url: str,
    username: str,
    app_password: str,
    endpoint: str,
    existing_ids: list,
    manual_names: list,
    language: str = "",
    multilingual_plugin: str = "auto",
) -> list:
    """Merge selected term IDs with IDs resolved from manual names."""
    resolved_ids = []
    seen = set()

    for value in existing_ids or []:
        try:
            term_id = int(value)
        except Exception:
            continue
        if term_id in seen:
            continue
        seen.add(term_id)
        resolved_ids.append(term_id)

    for name in _normalize_manual_term_names(manual_names):
        term_id = _find_or_create_wp_term_id(
            base_url=base_url,
            username=username,
            app_password=app_password,
            endpoint=endpoint,
            term_name=name,
            language=language,
            multilingual_plugin=multilingual_plugin,
        )
        if term_id in seen:
            continue
        seen.add(term_id)
        resolved_ids.append(term_id)

    return resolved_ids


def fetch_wp_categories(account_id: int, language: str = "") -> dict:
    """
    Fetch ALL WordPress categories via REST API (paginated, up to 2000).
    If language is specified and Polylang is active, filters by language.

    Returns dict with 'success', 'categories' (list of {id, name, slug, count}).
    """
    account = get_publish_account(account_id)
    if not account:
        return {"success": False, "error": "Account không tồn tại.", "categories": []}

    base_url = account["site_url"].rstrip("/")
    multilingual_plugin = (account.get("multilingual_plugin") or "auto").lower()

    try:
        raw = _fetch_wp_taxonomy(
            base_url=base_url,
            username=account.get("username", ""),
            app_password=account.get("app_password_decrypted", ""),
            endpoint="categories",
            language=language,
            multilingual_plugin=multilingual_plugin,
        )
        categories = [
            {"id": c.get("id"), "name": c.get("name", ""), "slug": c.get("slug", ""), "count": c.get("count", 0)}
            for c in raw
        ]
        return {"success": True, "categories": categories}
    except Exception as e:
        return {"success": False, "error": str(e), "categories": []}


def fetch_wp_tags(account_id: int, language: str = "") -> dict:
    """
    Fetch ALL WordPress tags via REST API (paginated, up to 2000).
    If language is specified and Polylang is active, filters by language.

    Returns dict with 'success', 'tags' (list of {id, name, slug, count}).
    """
    account = get_publish_account(account_id)
    if not account:
        return {"success": False, "error": "Account không tồn tại.", "tags": []}

    base_url = account["site_url"].rstrip("/")
    multilingual_plugin = (account.get("multilingual_plugin") or "auto").lower()

    try:
        raw = _fetch_wp_taxonomy(
            base_url=base_url,
            username=account.get("username", ""),
            app_password=account.get("app_password_decrypted", ""),
            endpoint="tags",
            language=language,
            multilingual_plugin=multilingual_plugin,
        )
        tags = [
            {"id": t.get("id"), "name": t.get("name", ""), "slug": t.get("slug", ""), "count": t.get("count", 0)}
            for t in raw
        ]
        return {"success": True, "tags": tags}
    except Exception as e:
        return {"success": False, "error": str(e), "tags": []}


def _test_wordpress_connection(site_url: str, username: str, app_password: str) -> dict:
    """Test WordPress REST API connection."""
    base_url = site_url.rstrip("/")
    try:
        response = requests.get(
            f"{base_url}/wp-json/wp/v2/users/me",
            auth=(username, app_password),
            timeout=10,
        )
        if response.status_code == 200:
            user = response.json()
            return {
                "success": True,
                "message": f"Kết nối thành công! User: {user.get('name', 'Unknown')}",
                "user": user.get("name", ""),
            }
        elif response.status_code == 401:
            return {"success": False, "error": "Sai username hoặc Application Password."}
        elif response.status_code == 403:
            return {"success": False, "error": "Không có quyền. Kiểm tra role của user."}
        else:
            return {"success": False, "error": f"WordPress trả về mã {response.status_code}."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"Không thể kết nối tới {base_url}. Kiểm tra URL."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout khi kết nối."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def fetch_shopify_tags(account_id: int) -> dict:
    """
    Fetch existing tags from a Shopify blog by sampling recent articles.
    Shopify tags are free-form strings; we collect unique tags from published articles.

    Returns dict with 'success', 'tags' (list of {id, name}).
    Note: 'id' equals the tag name string (Shopify has no numeric tag IDs).
    """
    account = get_publish_account(account_id)
    if not account:
        return {"success": False, "error": "Account không tồn tại.", "tags": []}

    base_url = account["site_url"].rstrip("/")
    api_token = account.get("api_token_decrypted", "")
    blog_id = account.get("blog_id", "")

    try:
        if not blog_id:
            blog_id = _get_shopify_default_blog(base_url, api_token)
            if not blog_id:
                return {"success": False, "error": "Không tìm thấy Blog ID.", "tags": []}

        all_tag_names: set = set()
        page_info = None

        for _ in range(20):  # max 20 pages × 250 = 5000 articles
            if page_info:
                params = {"limit": 250, "fields": "tags", "page_info": page_info}
            else:
                params = {"limit": 250, "fields": "tags", "status": "published"}

            response = requests.get(
                _shopify_api_url(base_url, f"blogs/{blog_id}/articles.json"),
                params=params,
                headers={"X-Shopify-Access-Token": api_token},
                timeout=20,
            )

            if response.status_code != 200:
                break

            articles = response.json().get("articles", [])
            for article in articles:
                raw_tags = article.get("tags", "")
                for t in raw_tags.split(","):
                    t = t.strip()
                    if t:
                        all_tag_names.add(t)

            if not articles or len(articles) < 250:
                break

            # Cursor-based pagination via Link header
            link_header = response.headers.get("Link", "")
            if 'rel="next"' not in link_header:
                break
            match = re.search(r'page_info=([^&>]+)[^>]*>;\s*rel="next"', link_header)
            if not match:
                break
            page_info = match.group(1)

        tags = [{"id": name, "name": name} for name in sorted(all_tag_names, key=str.lower)]
        return {"success": True, "tags": tags}
    except Exception as e:
        return {"success": False, "error": str(e), "tags": []}


def _test_shopify_connection(site_url: str, api_token: str) -> dict:
    """Test Shopify Admin API connection."""
    base_url = site_url.rstrip("/")
    try:
        response = requests.get(
            _shopify_api_url(base_url, "shop.json"),
            headers={
                "Content-Type": "application/json",
                "X-Shopify-Access-Token": api_token,
            },
            timeout=10,
        )
        if response.status_code == 200:
            shop = response.json().get("shop", {})
            return {
                "success": True,
                "message": f"Kết nối thành công! Shop: {shop.get('name', 'Unknown')}",
                "shop_name": shop.get("name", ""),
            }
        elif response.status_code == 401:
            return {"success": False, "error": "Token không hợp lệ."}
        elif response.status_code == 403:
            return {"success": False, "error": "Token không có quyền. Kiểm tra scopes."}
        else:
            return {"success": False, "error": f"Shopify trả về mã {response.status_code}."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"Không thể kết nối tới {base_url}. Kiểm tra URL."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout khi kết nối."}
    except Exception as e:
        return {"success": False, "error": str(e)}
