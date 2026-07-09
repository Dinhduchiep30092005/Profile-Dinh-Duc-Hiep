"""
Configuration module for SEO Bot.
Loads settings from .env file and provides typed config access.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


class OpenAIConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    temperature: float = 0.7
    max_tokens: int = 4096


class SerpAPIConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("SERPAPI_KEY", ""))
    results_count: int = Field(
        default_factory=lambda: int(os.getenv("SERP_RESULTS_COUNT", "20"))
    )


class DatabaseConfig(BaseModel):
    url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/seo_bot"
        )
    )


class ContentConfig(BaseModel):
    language: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_LANGUAGE", "en")
    )
    min_words: int = Field(
        default_factory=lambda: int(os.getenv("CONTENT_MIN_WORDS", "1500"))
    )
    max_words: int = Field(
        default_factory=lambda: int(os.getenv("CONTENT_MAX_WORDS", "3000"))
    )


class GSCConfig(BaseModel):
    """Google Search Console API configuration."""

    credentials_path: str = Field(
        default_factory=lambda: os.getenv("GSC_CREDENTIALS_PATH", "")
    )
    site_url: str = Field(
        default_factory=lambda: os.getenv("GSC_SITE_URL", "")
    )


class BrandConfig(BaseModel):
    """Brand-specific configuration injected by user."""

    product_name: str = ""
    product_description: str = ""
    use_cases: list[str] = Field(default_factory=list)
    competitive_advantages: list[str] = Field(default_factory=list)
    target_audience: str = ""
    brand_voice: str = "professional, technical, authoritative"


class Settings(BaseModel):
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    serpapi: SerpAPIConfig = Field(default_factory=SerpAPIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    gsc: GSCConfig = Field(default_factory=GSCConfig)
    brand: BrandConfig = Field(default_factory=BrandConfig)


# Global settings instance
settings = Settings()


# ──────────────────────────────────────────────
# Language → Google Locale mapping
# ──────────────────────────────────────────────
# gl = Google country domain, hl = interface language
LANGUAGE_LOCALE_MAP = {
    "vi":    {"gl": "vn", "hl": "vi",    "label": "Tiếng Việt"},
    "en":    {"gl": "uk", "hl": "en",    "label": "English (UK)"},
    "en-us": {"gl": "us", "hl": "en",    "label": "English (US)"},
    "en-au": {"gl": "au", "hl": "en",    "label": "English (AU)"},
    "fr":    {"gl": "fr", "hl": "fr",    "label": "Français"},
    "de":    {"gl": "de", "hl": "de",    "label": "Deutsch"},
    "es":    {"gl": "es", "hl": "es",    "label": "Español"},
    "ja":    {"gl": "jp", "hl": "ja",    "label": "日本語"},
    "ko":    {"gl": "kr", "hl": "ko",    "label": "한국어"},
    "zh":    {"gl": "cn", "hl": "zh-cn", "label": "中文 (简体)"},
    "zh-tw": {"gl": "tw", "hl": "zh-tw", "label": "中文 (繁體)"},
    "th":    {"gl": "th", "hl": "th",    "label": "ภาษาไทย"},
    "pt":    {"gl": "br", "hl": "pt",    "label": "Português (BR)"},
    "it":    {"gl": "it", "hl": "it",    "label": "Italiano"},
    "ru":    {"gl": "ru", "hl": "ru",    "label": "Русский"},
    "id":    {"gl": "id", "hl": "id",    "label": "Bahasa Indonesia"},
}
