"""
Project & Content Item Management — SQLite-based storage.
Manages projects, content items, and publish accounts.

Tables:
    projects         — id, name, description, created_at
    content_items    — id, project_id (FK), keyword, title, heading_structure (JSON),
                       status, article_generated, published, priority, created_at,
                       publish_status, publish_account_id, published_url, error_message
    publish_accounts — id, project_id (FK), platform, site_url, username,
                       app_password, api_token, blog_id, status, created_at
"""

import json
import sqlite3
import base64
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "projects.db"

# ──────────────────────────────────────────────
# AES-like encryption for credentials (using Fernet via base64 + XOR)
# Simple but effective obfuscation without heavy dependencies
# ──────────────────────────────────────────────
_ENCRYPT_KEY = os.getenv("PUBLISH_ENCRYPT_KEY", "seo-bot-publish-secret-2024")


def _derive_key(secret: str) -> bytes:
    """Derive a 32-byte key from a secret string."""
    return hashlib.sha256(secret.encode()).digest()


def encrypt_credential(plaintext: str) -> str:
    """Encrypt a credential string for safe DB storage."""
    if not plaintext:
        return ""
    key = _derive_key(_ENCRYPT_KEY)
    data = plaintext.encode("utf-8")
    # XOR with repeating key
    encrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])
    return base64.urlsafe_b64encode(encrypted).decode("ascii")


def decrypt_credential(ciphertext: str) -> str:
    """Decrypt a stored credential string."""
    if not ciphertext:
        return ""
    try:
        key = _derive_key(_ENCRYPT_KEY)
        data = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
        decrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])
        return decrypted.decode("utf-8")
    except Exception:
        return ciphertext  # fallback: return as-is


def _migrate_db():
    """Add new columns if they don't exist (safe to call multiple times)."""
    conn = _get_conn()
    # Get existing columns for content_items
    cursor = conn.execute("PRAGMA table_info(content_items)")
    existing = {row["name"] for row in cursor.fetchall()}
    migrations = {
        "article_text": "TEXT DEFAULT ''",
        "article_html": "TEXT DEFAULT ''",
        "images": "TEXT DEFAULT '[]'",
        "internal_links": "TEXT DEFAULT '[]'",
        "updated_at": "TEXT DEFAULT ''",
        "publish_status": "TEXT DEFAULT 'draft'",
        "publish_account_id": "INTEGER DEFAULT NULL",
        "published_url": "TEXT DEFAULT ''",
        "error_message": "TEXT DEFAULT ''",
        "featured_image_path": "TEXT DEFAULT ''",
        "featured_image_alt": "TEXT DEFAULT ''",
        "wp_post_id": "TEXT DEFAULT ''",
        "language": "TEXT DEFAULT 'vi'",
        "url_slug": "TEXT DEFAULT ''",
    }
    for col, col_type in migrations.items():
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE content_items ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass

    # Create publish_accounts table if not exists
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS publish_accounts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  INTEGER DEFAULT NULL REFERENCES projects(id) ON DELETE SET NULL,
            platform    TEXT NOT NULL DEFAULT 'wordpress',
            site_url    TEXT NOT NULL DEFAULT '',
            username    TEXT DEFAULT '',
            app_password TEXT DEFAULT '',
            api_token   TEXT DEFAULT '',
            blog_id     TEXT DEFAULT '',
            status      TEXT NOT NULL DEFAULT 'active',
            name        TEXT DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_publish_accounts_project ON publish_accounts(project_id);
    """)

    # Migrate publish_accounts columns
    cursor_pa = conn.execute("PRAGMA table_info(publish_accounts)")
    existing_pa = {row["name"] for row in cursor_pa.fetchall()}
    pa_migrations = {
        "multilingual_plugin": "TEXT DEFAULT 'auto'",  # 'auto', 'none', 'polylang', 'wpml'
        "lang_mapping": "TEXT DEFAULT '{}'",  # JSON: {"vi": "vi", "en": "en", ...}
    }
    for col, col_type in pa_migrations.items():
        if col not in existing_pa:
            try:
                conn.execute(f"ALTER TABLE publish_accounts ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass

    conn.commit()
    conn.close()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_projects_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS content_items (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            keyword             TEXT NOT NULL,
            title               TEXT NOT NULL,
            heading_structure   TEXT DEFAULT '{}',
            article_text        TEXT DEFAULT '',
            article_html        TEXT DEFAULT '',
            images              TEXT DEFAULT '[]',
            internal_links      TEXT DEFAULT '[]',
            status              TEXT NOT NULL DEFAULT 'draft',
            article_generated   INTEGER NOT NULL DEFAULT 0,
            published           INTEGER NOT NULL DEFAULT 0,
            priority            INTEGER NOT NULL DEFAULT 0,
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_content_project ON content_items(project_id);
    """)
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
# Projects CRUD
# ──────────────────────────────────────────────

def list_projects() -> list[dict]:
    """List all projects with content count."""
    init_projects_db()
    conn = _get_conn()
    rows = conn.execute("""
        SELECT p.*, COUNT(c.id) as content_count
        FROM projects p
        LEFT JOIN content_items c ON c.project_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_project(name: str, description: str = "") -> dict:
    """Create a new project."""
    init_projects_db()
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    try:
        cursor = conn.execute(
            "INSERT INTO projects (name, description, created_at) VALUES (?, ?, ?)",
            (name.strip(), description.strip(), now),
        )
        conn.commit()
        project_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        conn.close()
        return dict(row)
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"Project '{name}' đã tồn tại.")


def update_project(project_id: int, name: str = None, description: str = None) -> dict:
    """Update a project's name and/or description."""
    init_projects_db()
    conn = _get_conn()
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name.strip())
    if description is not None:
        fields.append("description = ?")
        values.append(description.strip())
    if not fields:
        conn.close()
        raise ValueError("Nothing to update.")
    values.append(project_id)
    try:
        conn.execute(f"UPDATE projects SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        conn.close()
        if not row:
            raise ValueError("Project not found.")
        return dict(row)
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"Project '{name}' đã tồn tại.")


def delete_project(project_id: int) -> bool:
    """Delete a project and all its content items."""
    init_projects_db()
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ──────────────────────────────────────────────
# Content Items CRUD
# ──────────────────────────────────────────────

def list_content_items(project_id: int) -> list[dict]:
    """List all content items for a project."""
    init_projects_db()
    _migrate_db()  # Ensure new columns exist
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM content_items WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        # Parse JSON fields
        for json_field in ["heading_structure", "images", "internal_links"]:
            try:
                val = d.get(json_field, "{}" if json_field == "heading_structure" else "[]")
                d[json_field] = json.loads(val) if isinstance(val, str) else val
            except (json.JSONDecodeError, TypeError):
                d[json_field] = {} if json_field == "heading_structure" else []
        result.append(d)
    return result


def create_content_item(
    project_id: int,
    keyword: str,
    title: str,
    heading_structure: Optional[dict] = None,
    status: str = "draft",
    priority: int = 0,
    language: str = "vi",
) -> dict:
    """Save a content item (keyword + title + heading) to a project."""
    init_projects_db()
    _migrate_db()
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    hs_json = json.dumps(heading_structure or {}, ensure_ascii=False)
    cursor = conn.execute(
        """INSERT INTO content_items
           (project_id, keyword, title, heading_structure, status, priority, language, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (project_id, keyword.strip(), title.strip(), hs_json, status, priority, language, now),
    )
    conn.commit()
    item_id = cursor.lastrowid
    row = conn.execute("SELECT * FROM content_items WHERE id = ?", (item_id,)).fetchone()
    conn.close()
    d = dict(row)
    for json_field in ["heading_structure", "images", "internal_links"]:
        try:
            val = d.get(json_field, "{}" if json_field == "heading_structure" else "[]")
            d[json_field] = json.loads(val) if isinstance(val, str) else val
        except (json.JSONDecodeError, TypeError):
            d[json_field] = {} if json_field == "heading_structure" else []
    return d


def update_content_item(item_id: int, **kwargs) -> dict:
    """Update a content item's fields."""
    init_projects_db()
    _migrate_db()
    allowed = {"keyword", "title", "heading_structure", "article_text", "article_html",
               "images", "internal_links", "status", "article_generated", "published", "priority",
               "publish_status", "publish_account_id", "published_url", "error_message",
               "featured_image_path", "featured_image_alt", "wp_post_id", "language", "url_slug"}
    fields = []
    values = []
    for k, v in kwargs.items():
        if k not in allowed:
            continue
        if k in ("heading_structure", "images", "internal_links"):
            v = json.dumps(v, ensure_ascii=False)
        fields.append(f"{k} = ?")
        values.append(v)
    # Always update updated_at
    fields.append("updated_at = ?")
    values.append(datetime.now(timezone.utc).isoformat())
    if not fields:
        raise ValueError("Nothing to update.")
    values.append(item_id)
    conn = _get_conn()
    conn.execute(f"UPDATE content_items SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()
    row = conn.execute("SELECT * FROM content_items WHERE id = ?", (item_id,)).fetchone()
    conn.close()
    if not row:
        raise ValueError("Content item not found.")
    d = dict(row)
    for json_field in ["heading_structure", "images", "internal_links"]:
        try:
            val = d.get(json_field, "{}" if json_field == "heading_structure" else "[]")
            d[json_field] = json.loads(val) if isinstance(val, str) else val
        except (json.JSONDecodeError, TypeError):
            d[json_field] = {} if json_field == "heading_structure" else []
    return d


def delete_content_item(item_id: int) -> bool:
    """Delete a content item."""
    init_projects_db()
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM content_items WHERE id = ?", (item_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ──────────────────────────────────────────────
# Publish Accounts CRUD
# ──────────────────────────────────────────────

def list_publish_accounts(project_id: int = None) -> list[dict]:
    """List publish accounts, optionally filtered by project."""
    init_projects_db()
    _migrate_db()
    conn = _get_conn()
    if project_id:
        rows = conn.execute(
            "SELECT * FROM publish_accounts WHERE project_id = ? OR project_id IS NULL ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM publish_accounts ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        # Mask sensitive fields for list display
        d["app_password_masked"] = "***" + decrypt_credential(d.get("app_password", ""))[-4:] if d.get("app_password") else ""
        d["api_token_masked"] = "***" + decrypt_credential(d.get("api_token", ""))[-4:] if d.get("api_token") else ""
        # Parse lang_mapping JSON
        try:
            d["lang_mapping"] = json.loads(d.get("lang_mapping", "{}")) if isinstance(d.get("lang_mapping"), str) else d.get("lang_mapping", {})
        except (json.JSONDecodeError, TypeError):
            d["lang_mapping"] = {}
        result.append(d)
    return result


def create_publish_account(
    platform: str,
    site_url: str,
    name: str = "",
    username: str = "",
    app_password: str = "",
    api_token: str = "",
    blog_id: str = "",
    project_id: int = None,
    multilingual_plugin: str = "auto",
    lang_mapping: dict = None,
) -> dict:
    """Create a new publish account with encrypted credentials."""
    init_projects_db()
    _migrate_db()
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()

    # Encrypt sensitive fields
    enc_password = encrypt_credential(app_password) if app_password else ""
    enc_token = encrypt_credential(api_token) if api_token else ""
    lang_mapping_json = json.dumps(lang_mapping or {}, ensure_ascii=False)

    cursor = conn.execute(
        """INSERT INTO publish_accounts
           (project_id, platform, site_url, name, username, app_password, api_token, blog_id, status, multilingual_plugin, lang_mapping, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
        (project_id, platform, site_url.strip(), name.strip(), username.strip(),
         enc_password, enc_token, blog_id.strip(), multilingual_plugin, lang_mapping_json, now),
    )
    conn.commit()
    account_id = cursor.lastrowid
    row = conn.execute("SELECT * FROM publish_accounts WHERE id = ?", (account_id,)).fetchone()
    conn.close()
    d = dict(row)
    d["app_password_masked"] = "***" + app_password[-4:] if app_password else ""
    d["api_token_masked"] = "***" + api_token[-4:] if api_token else ""
    # Parse lang_mapping JSON
    try:
        d["lang_mapping"] = json.loads(d.get("lang_mapping", "{}")) if isinstance(d.get("lang_mapping"), str) else d.get("lang_mapping", {})
    except (json.JSONDecodeError, TypeError):
        d["lang_mapping"] = {}
    return d


def update_publish_account(account_id: int, **kwargs) -> dict:
    """Update a publish account."""
    init_projects_db()
    _migrate_db()
    allowed = {"platform", "site_url", "name", "username", "app_password",
               "api_token", "blog_id", "status", "project_id",
               "multilingual_plugin", "lang_mapping"}
    fields = []
    values = []
    for k, v in kwargs.items():
        if k not in allowed:
            continue
        if k in ("app_password", "api_token") and v:
            # Don't re-encrypt masked values
            if not v.startswith("***"):
                v = encrypt_credential(v)
            else:
                continue  # Skip masked values
        if k == "lang_mapping" and isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False)
        fields.append(f"{k} = ?")
        values.append(v)
    if not fields:
        raise ValueError("Nothing to update.")
    values.append(account_id)
    conn = _get_conn()
    conn.execute(f"UPDATE publish_accounts SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()
    row = conn.execute("SELECT * FROM publish_accounts WHERE id = ?", (account_id,)).fetchone()
    conn.close()
    if not row:
        raise ValueError("Publish account not found.")
    d = dict(row)
    d["app_password_masked"] = "***" + decrypt_credential(d.get("app_password", ""))[-4:] if d.get("app_password") else ""
    d["api_token_masked"] = "***" + decrypt_credential(d.get("api_token", ""))[-4:] if d.get("api_token") else ""
    try:
        d["lang_mapping"] = json.loads(d.get("lang_mapping", "{}")) if isinstance(d.get("lang_mapping"), str) else d.get("lang_mapping", {})
    except (json.JSONDecodeError, TypeError):
        d["lang_mapping"] = {}
    return d


def delete_publish_account(account_id: int) -> bool:
    """Delete a publish account."""
    init_projects_db()
    _migrate_db()
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM publish_accounts WHERE id = ?", (account_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def get_publish_account(account_id: int) -> Optional[dict]:
    """Get a single publish account with decrypted credentials."""
    init_projects_db()
    _migrate_db()
    conn = _get_conn()
    row = conn.execute("SELECT * FROM publish_accounts WHERE id = ?", (account_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    # Decrypt credentials for actual use
    d["app_password_decrypted"] = decrypt_credential(d.get("app_password", ""))
    d["api_token_decrypted"] = decrypt_credential(d.get("api_token", ""))
    # Parse lang_mapping JSON
    try:
        d["lang_mapping"] = json.loads(d.get("lang_mapping", "{}")) if isinstance(d.get("lang_mapping"), str) else d.get("lang_mapping", {})
    except (json.JSONDecodeError, TypeError):
        d["lang_mapping"] = {}
    return d
