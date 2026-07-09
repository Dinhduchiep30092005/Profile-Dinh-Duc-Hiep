"""
Truncation Detector — Detects incomplete/truncated LLM output.

Checks for:
    1. finish_reason == "length" from the API (token limit hit)
    2. Incomplete sentences (no terminal punctuation)
    3. Unclosed HTML/Markdown tags
    4. Structural issues: heading with no body, very short trailing paragraph
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# Terminal punctuation patterns
# ──────────────────────────────────────────
_SENTENCE_TERMINATORS = re.compile(r'[.!?…。！？]$')
_MARKDOWN_HEADING = re.compile(r'^#{1,6}\s+')

# HTML tags that should be paired
_PAIRED_TAGS = {"p", "ul", "ol", "li", "table", "tr", "td", "th", "thead",
                "tbody", "blockquote", "div", "span", "strong", "em", "a",
                "h1", "h2", "h3", "h4", "h5", "h6"}

_OPEN_TAG_RE = re.compile(r'<(\w+)(?:\s[^>]*)?>(?!</\1>)')
_CLOSE_TAG_RE = re.compile(r'</(\w+)>')


def detect_truncation(
    text: str,
    finish_reason: str = "stop",
) -> dict:
    """Run all truncation checks and return a report.

    Args:
        text: The generated content
        finish_reason: finish_reason from the API response

    Returns:
        dict with:
            is_truncated (bool): True if any truncation detected
            reasons (list[str]): Human-readable reasons
            severity (str): 'none' | 'minor' | 'major'
    """
    reasons = []

    # 1. API-level truncation (most reliable signal)
    if finish_reason == "length":
        reasons.append("API finish_reason='length' — output hit token limit")

    # 2. Incomplete last sentence
    if _has_incomplete_sentence(text):
        reasons.append("Last sentence lacks terminal punctuation (truncated mid-sentence)")

    # 3. Unclosed HTML tags
    unclosed = _find_unclosed_tags(text)
    if unclosed:
        reasons.append(f"Unclosed HTML tags: {', '.join(unclosed)}")

    # 4. Trailing heading with no body
    if _has_trailing_empty_heading(text):
        reasons.append("Content ends with a heading but no body text follows")

    # 5. Very short trailing paragraph (< 20 words, not a list item)
    trailing_issue = _has_short_trailing_paragraph(text)
    if trailing_issue:
        reasons.append(trailing_issue)

    # Determine severity
    if not reasons:
        severity = "none"
    elif finish_reason == "length" or len(reasons) >= 2:
        severity = "major"
    else:
        severity = "minor"

    return {
        "is_truncated": len(reasons) > 0,
        "reasons": reasons,
        "severity": severity,
        "finish_reason": finish_reason,
    }


def validate_sections_bulk(sections: list[str]) -> list[dict]:
    """Validate a list of generated sections for completeness issues.

    Used between Phase 5 and Phase 6 (post-generation validation).

    Args:
        sections: List of generated section texts

    Returns:
        List of dicts, one per section, with:
            index (int): Section index
            issues (list[str]): List of issues found
            needs_regeneration (bool): True if section should be regenerated
            heading (str): First heading found in the section
    """
    results = []
    for idx, section_text in enumerate(sections):
        issues = []
        heading = _extract_first_heading(section_text)

        # 1. Empty or near-empty section
        word_count = len(section_text.split())
        if word_count < 30:
            issues.append(f"Section extremely short ({word_count} words)")

        # 2. Truncation detection (without finish_reason, use text-only checks)
        truncation = detect_truncation(section_text, finish_reason="stop")
        if truncation["is_truncated"]:
            issues.extend(truncation["reasons"])

        # 3. Headings with empty body
        empty_headings = _find_headings_without_body(section_text)
        if empty_headings:
            issues.append(
                f"Heading(s) with no body content: {', '.join(empty_headings)}"
            )

        # 4. Orphan short paragraphs (< 20 words standing alone between headings)
        orphan_paras = _find_orphan_short_paragraphs(section_text)
        if orphan_paras:
            issues.append(
                f"{len(orphan_paras)} very short paragraph(s) "
                f"(< 20 words) that may be truncated"
            )

        # 5. Abruptly different language / garbled text at the end
        if _has_garbled_ending(section_text):
            issues.append("Section ending appears garbled or incoherent")

        # Decide if regeneration needed
        needs_regen = (
            word_count < 30
            or len(empty_headings) > 0
            or any("finish_reason" in i for i in issues)
            or any("mid-sentence" in i for i in issues)
            or any("garbled" in i for i in issues)
        )

        results.append({
            "index": idx,
            "heading": heading,
            "word_count": word_count,
            "issues": issues,
            "needs_regeneration": needs_regen,
        })

    return results


# ──────────────────────────────────────────
# Internal detection functions
# ──────────────────────────────────────────

def _has_incomplete_sentence(text: str) -> bool:
    """Check if the last meaningful line ends without terminal punctuation."""
    if not text.strip():
        return False

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return False

    last_line = lines[-1]

    # Skip if last line is a heading (headings don't need terminal punctuation)
    if _MARKDOWN_HEADING.match(last_line):
        return True  # Actually, ending on a heading IS a problem (no body follows)

    # Skip if it's a list item marker only
    if re.match(r'^[-*]\s*$', last_line):
        return True

    # Strip trailing markdown formatting
    clean = re.sub(r'[*_`]+$', '', last_line).strip()

    # Strip trailing HTML tags for check
    clean = re.sub(r'<[^>]+>\s*$', '', clean).strip()

    if not clean:
        return False

    # Check for terminal punctuation
    return not _SENTENCE_TERMINATORS.search(clean)


def _find_unclosed_tags(text: str) -> list[str]:
    """Find HTML tags that are opened but not closed."""
    # Simple stack-based tag matching
    tag_stack = []
    unclosed = []

    for match in re.finditer(r'<(/?)(\w+)(?:\s[^>]*)?\s*/?>', text):
        is_closing = match.group(1) == '/'
        tag_name = match.group(2).lower()

        if tag_name not in _PAIRED_TAGS:
            continue

        # Self-closing check
        if match.group(0).endswith('/>'):
            continue

        if is_closing:
            if tag_stack and tag_stack[-1] == tag_name:
                tag_stack.pop()
            # else: mismatched close tag, but not truncation per se
        else:
            tag_stack.append(tag_name)

    if tag_stack:
        unclosed = list(set(tag_stack))

    return unclosed


def _has_trailing_empty_heading(text: str) -> bool:
    """Check if content ends with a heading that has no body after it."""
    lines = [l for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return False

    # Check last 1-3 non-empty lines
    for i in range(len(lines) - 1, max(len(lines) - 4, -1), -1):
        line = lines[i].strip()
        if _MARKDOWN_HEADING.match(line):
            return True
        if line:  # Found a non-heading non-empty line
            return False

    return False


def _has_short_trailing_paragraph(text: str) -> Optional[str]:
    """Check if the last paragraph is suspiciously short (likely truncated)."""
    paragraphs = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        return None

    last_para = paragraphs[-1]

    # Skip if it's a heading or list
    if _MARKDOWN_HEADING.match(last_para.split('\n')[0]):
        return None
    if last_para.startswith(('-', '*', '1.')):
        return None

    word_count = len(last_para.split())
    if word_count < 15:
        # Check if the paragraph before it is much longer
        prev_para = paragraphs[-2]
        prev_words = len(prev_para.split())
        if prev_words > 40 and word_count < prev_words * 0.3:
            return (
                f"Trailing paragraph only {word_count} words "
                f"(previous was {prev_words}) — likely truncated"
            )

    return None


def _find_headings_without_body(text: str) -> list[str]:
    """Find headings that are immediately followed by another heading or end of text."""
    lines = text.strip().split('\n')
    empty_headings = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not _MARKDOWN_HEADING.match(stripped):
            continue

        # Look ahead for body content
        has_body = False
        for j in range(i + 1, min(i + 5, len(lines))):
            next_line = lines[j].strip()
            if not next_line:
                continue
            if _MARKDOWN_HEADING.match(next_line):
                break  # Another heading before any body → empty
            has_body = True
            break

        # If this is the last heading and no body follows
        if not has_body:
            heading_text = _MARKDOWN_HEADING.sub('', stripped).strip()
            # Don't flag if it's the very first line and the only heading
            if heading_text:
                empty_headings.append(heading_text)

    return empty_headings


def _find_orphan_short_paragraphs(text: str) -> list[str]:
    """Find standalone paragraphs < 20 words that seem orphaned/truncated."""
    paragraphs = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
    orphans = []

    for i, para in enumerate(paragraphs):
        # Skip headings, lists, tables
        first_line = para.split('\n')[0].strip()
        if _MARKDOWN_HEADING.match(first_line):
            continue
        if first_line.startswith(('-', '*', '|', '1.', '>')):
            continue

        word_count = len(para.split())
        if word_count < 20:
            # Check if it's between two headings (orphaned)
            is_before_heading = False
            is_after_heading = False

            if i + 1 < len(paragraphs):
                next_first = paragraphs[i + 1].split('\n')[0].strip()
                is_before_heading = bool(_MARKDOWN_HEADING.match(next_first))

            if i > 0:
                prev_lines = paragraphs[i - 1].strip().split('\n')
                prev_last = prev_lines[-1].strip() if prev_lines else ""
                is_after_heading = bool(_MARKDOWN_HEADING.match(
                    paragraphs[i - 1].split('\n')[0].strip()
                ))

            if is_before_heading or (word_count < 10 and i == len(paragraphs) - 1):
                orphans.append(para[:80])

    return orphans


def _has_garbled_ending(text: str) -> bool:
    """Detect if the end of text looks garbled (encoding issues, random chars)."""
    if not text.strip():
        return False

    last_100 = text.strip()[-100:]

    # High ratio of non-word characters (excluding markdown/HTML)
    clean = re.sub(r'[#*_`<>/\[\]()|-]', '', last_100)
    if not clean.strip():
        return False

    alpha_ratio = sum(1 for c in clean if c.isalnum() or c.isspace()) / max(len(clean), 1)
    if alpha_ratio < 0.5:
        return True

    return False


def _extract_first_heading(text: str) -> str:
    """Extract the first markdown heading from text."""
    for line in text.strip().split('\n'):
        stripped = line.strip()
        match = _MARKDOWN_HEADING.match(stripped)
        if match:
            return stripped[match.end():].strip()
    return "(no heading)"
