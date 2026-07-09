"""
LLM Client - OpenAI GPT wrapper for SEO Bot.
Handles all AI-powered text generation with structured prompts.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around OpenAI API for structured SEO content generation."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.model = settings.openai.model
        self.temperature = settings.openai.temperature
        self.max_tokens = settings.openai.max_tokens

    @staticmethod
    def _date_context() -> str:
        """Return a short date-awareness prefix for system prompts."""
        now = datetime.now()
        return (
            f"CURRENT DATE: {now.strftime('%Y-%m-%d')} (year {now.year}).\n"
            f"When referencing years, statistics, trends, or freshness signals, "
            f"always use the current year {now.year}. Never use outdated years.\n\n"
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        # Inject current date context so LLM always knows the current time
        system_prompt_with_date = self._date_context() + system_prompt

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt_with_date},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise

    def chat_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> dict:
        """Send a chat completion request and return both text and metadata.

        Returns:
            dict with 'content' (str), 'finish_reason' (str), 'usage' (dict)
            finish_reason is 'stop' (normal), 'length' (truncated), or other.
        """
        system_prompt_with_date = self._date_context() + system_prompt

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt_with_date},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            usage = response.usage
            return {
                "content": (choice.message.content or "").strip(),
                "finish_reason": choice.finish_reason or "unknown",
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                } if usage else {},
            }
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Send a chat request and parse JSON response."""
        raw = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)

    def chat_long(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Chat with extended token limit for long-form content."""
        return self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=8192,
        )


# Global instance  — initialised lazily if no key yet
try:
    llm = LLMClient()
except Exception:
    llm = None  # will be reinitialised once API key is set via dashboard
