"""
Article Generator — Generates full blog articles from approved heading structures.

NEW: Uses 3-layer pipeline via ArticleBuilder for professional-quality content:
    Layer 1 — Section-by-section generation (deep, focused content per H2)
    Layer 2 — EEAT enforcement (validate & enhance each section)
    Layer 3 — Internal linking (pre-calculated links injected naturally)

Legacy pipeline (fallback):
    A. Inject Knowledge (RAG)
    B. Auto Internal Linking (semantic match from uploaded docs)
    C. Output in Read Mode (plain text) and HTML Mode (WordPress-ready)
"""

import logging
import re
import json
from typing import Optional

from config import settings
from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.content.section_writer import _build_language_enforcement
from core.knowledge.retrieval import (
    build_rag_context,
    build_product_solution_context,
    has_knowledge_base,
    retrieve_relevant_chunks,
)
from core.knowledge.embedding_engine import create_single_embedding, load_vectors, cosine_similarity_batch
from core.content.internal_link_injector import InternalLinkInjector

logger = logging.getLogger(__name__)


class ArticleGenerator:
    """Generates full SEO + EEAT blog articles from heading structures.

    Default: uses the new 3-layer ArticleBuilder pipeline.
    Fallback: legacy single-shot generation if use_legacy=True.
    """

    MAX_INTERNAL_LINKS = 50
    MIN_INTERNAL_LINKS = 1

    def __init__(self, use_legacy: bool = False):
        """
        Args:
            use_legacy: If True, use the old single-shot generation instead of 3-layer pipeline.
        """
        self.use_legacy = use_legacy

    def generate(
        self,
        keyword: str,
        title: str,
        heading_structure: dict,
        language: str = "vi",
    ) -> dict:
        """
        Generate a complete blog article from an approved heading structure.

        By default, uses the new 3-layer pipeline (ArticleBuilder):
            Layer 1 — Section-by-section generation
            Layer 2 — EEAT enforcement per section
            Layer 3 — Pre-calculated internal linking

        Set use_legacy=True to fall back to the old single-shot generation.

        Returns:
            dict with article_text, article_html, internal_links, word_count
        """
        if not self.use_legacy:
            return self._generate_with_builder(keyword, title, heading_structure, language)

        return self._generate_legacy(keyword, title, heading_structure, language)

    def _generate_with_builder(
        self,
        keyword: str,
        title: str,
        heading_structure: dict,
        language: str,
    ) -> dict:
        """Generate article using the new 3-layer ArticleBuilder pipeline."""
        from core.content.article_builder import ArticleBuilder

        builder = ArticleBuilder(temperature=0.65, eeat_strict=True)
        return builder.build(
            keyword=keyword,
            title=title,
            heading_structure=heading_structure,
            language=language,
        )

    def _generate_legacy(
        self,
        keyword: str,
        title: str,
        heading_structure: dict,
        language: str,
    ) -> dict:
        """Legacy single-shot article generation (original pipeline)."""
        logger.info(f"[ArticleGen] ══ Start (legacy): '{title}' ══")

        # ── Layer A: Knowledge Injection ──
        knowledge_context, product_context = self._inject_knowledge(keyword, title)
        logger.info("[ArticleGen] Layer A — Knowledge injected")

        # ── Generate article content section by section ──
        article_text = self._generate_article_text(
            keyword=keyword,
            title=title,
            heading_structure=heading_structure,
            knowledge_context=knowledge_context,
            product_context=product_context,
            language=language,
        )
        logger.info(f"[ArticleGen] Article text generated — {len(article_text.split())} words")

        # ── Layer B: Auto Internal Linking ──
        article_text, internal_links = self._auto_internal_linking(
            article_text, keyword, title
        )
        logger.info(f"[ArticleGen] Layer B — {len(internal_links)} internal links inserted")

        # ── Layer C: Output Modes ──
        article_html = self._convert_to_html(article_text, title, internal_links)
        logger.info("[ArticleGen] Layer C — HTML generated")

        word_count = len(article_text.split())

        result = {
            "article_text": article_text,
            "article_html": article_html,
            "internal_links": internal_links,
            "word_count": word_count,
        }

        logger.info(f"[ArticleGen] ══ Complete: {word_count} words, {len(internal_links)} links ══")
        return result

    # ──────────────────────────────────────────
    # Layer A: Knowledge Injection
    # ──────────────────────────────────────────

    def _inject_knowledge(self, keyword: str, title: str) -> tuple[str, str]:
        """Retrieve relevant knowledge from KB."""
        knowledge_context = ""
        product_context = ""

        if not has_knowledge_base():
            logger.info("[ArticleGen] No knowledge base — skipping injection")
            return knowledge_context, product_context

        try:
            knowledge_context, _ = build_rag_context(
                keyword=keyword,
                title=title,
                intent_type="informational",
                max_tokens_approx=4000,
            )
        except Exception as e:
            logger.warning(f"[ArticleGen] RAG context failed: {e}")

        try:
            product_context, _ = build_product_solution_context(
                keyword=keyword,
                title=title,
                relevance_threshold=0.30,
                max_tokens_approx=2000,
            )
        except Exception as e:
            logger.warning(f"[ArticleGen] Product context failed: {e}")

        return knowledge_context, product_context

    # ──────────────────────────────────────────
    # Article Text Generation
    # ──────────────────────────────────────────

    def _generate_article_text(
        self,
        keyword: str,
        title: str,
        heading_structure: dict,
        knowledge_context: str,
        product_context: str,
        language: str,
    ) -> str:
        """Generate the full article text from heading structure."""

        # Build the heading outline with details
        sections = heading_structure.get("sections", [])
        intro = heading_structure.get("intro", {})
        faq = heading_structure.get("faq", [])

        outline_text = self._format_heading_for_prompt(sections, intro, faq)

        system_prompt = _build_language_enforcement(language) + PromptTemplates.ARTICLE_WRITER_SYSTEM

        lang_map = {
            "vi": "Viết toàn bộ bài viết bằng tiếng Việt.",
            "en": "Write the entire article in English.",
            "en-us": "Write the entire article in American English.",
            "fr": "Rédigez l'article en français.",
            "de": "Schreiben Sie den Artikel auf Deutsch.",
            "es": "Escribe el artículo en español.",
            "ja": "記事全体を日本語で書いてください。",
            "ko": "기사 전체를 한국어로 작성하세요。",
            "zh": "用简体中文写整篇文章。",
            "th": "เขียนบทความทั้งหมดเป็นภาษาไทย",
        }
        lang_instruction = lang_map.get(language, f"Write in the language matching code '{language}'.")

        user_prompt = f"""TITLE: {title}
KEYWORD: {keyword}
LANGUAGE: {lang_instruction}

═══ HEADING STRUCTURE (ĐÃ ĐƯỢC DUYỆT — VIẾT THEO ĐÚNG CẤU TRÚC NÀY) ═══
{outline_text}
═══ END HEADING STRUCTURE ═══
"""

        if knowledge_context:
            user_prompt += f"\n{knowledge_context}\n"

        if product_context:
            user_prompt += f"\n{product_context}\n"

        user_prompt += """
QUAN TRỌNG:
- Viết ĐÚNG theo cấu trúc heading đã duyệt ở trên — KHÔNG thêm, bớt, hoặc thay đổi heading
- Mỗi section phải có nội dung thực sự giá trị, chi tiết, chuyên sâu
- Sử dụng kiến thức từ tài liệu nội bộ (nếu có) để tạo content độc đáo
- Viết theo chuẩn SEO + EEAT
- Kết thúc bài viết với phần kết luận tóm tắt
"""

        try:
            article = llm.chat_long(system_prompt, user_prompt, temperature=0.7)
            return article
        except Exception as e:
            logger.error(f"[ArticleGen] LLM generation failed: {e}")
            raise

    def _format_heading_for_prompt(self, sections: list, intro: dict, faq: list) -> str:
        """Format the heading structure for the prompt."""
        lines = []

        if intro:
            lines.append(f"[INTRO]")
            if intro.get("purpose"):
                lines.append(f"  Purpose: {intro['purpose']}")
            if intro.get("key_points"):
                for p in intro["key_points"]:
                    lines.append(f"  - {p}")
            lines.append("")

        for sec in sections:
            level = sec.get("level", "h2")
            heading = sec.get("heading", "")
            prefix = {"h2": "## ", "h3": "### ", "h4": "#### "}.get(level, "## ")
            lines.append(f"{prefix}{heading}")

            if sec.get("purpose"):
                lines.append(f"  Purpose: {sec['purpose']}")
            if sec.get("content_type"):
                lines.append(f"  Content type: {sec['content_type']}")
            if sec.get("eeat_signal"):
                lines.append(f"  EEAT signal: {sec['eeat_signal']}")
            if sec.get("key_points"):
                for p in sec["key_points"]:
                    lines.append(f"  - {p}")
            if sec.get("word_count_target"):
                lines.append(f"  Target: ~{sec['word_count_target']} words")
            lines.append("")

        if faq:
            lines.append("## FAQ")
            for f in faq:
                lines.append(f"  Q: {f.get('question', '')}")
                if f.get("answer_outline"):
                    lines.append(f"  A outline: {f['answer_outline']}")
            lines.append("")

        return "\n".join(lines)

    # ──────────────────────────────────────────
    # Layer B: Auto Internal Linking
    # ──────────────────────────────────────────

    def _auto_internal_linking(
        self,
        article_text: str,
        keyword: str,
        title: str,
    ) -> tuple[str, list]:
        """
        Find relevant URLs from uploaded knowledge base documents and insert internal links.
        Scans ALL uploaded docs directly to extract every URL with its context,
        then uses LLM to place them naturally in the article.
        Only URLs that actually exist in the uploaded files are used.
        """
        internal_links = []

        if not has_knowledge_base():
            return article_text, internal_links

        try:
            # ── Step 1: Extract ALL URLs directly from uploaded documents ──
            from core.knowledge.document_loader import _load_index, UPLOAD_DIR

            index = _load_index()
            documents = index.get("documents", {})

            url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')
            all_urls = []  # list of {url, context, doc_name}
            seen_urls = set()

            for doc_id, doc_info in documents.items():
                doc_name = doc_info.get("original_name", doc_id)

                # ── Strategy 1: Read text directly from file ──
                # Try "saved_name" first (correct key), fallback to "stored_name"
                file_name = doc_info.get("saved_name") or doc_info.get("stored_name", "")
                stored_path = UPLOAD_DIR / file_name if file_name else None
                doc_text = ""

                if stored_path and stored_path.exists() and stored_path.is_file():
                    ext = stored_path.suffix.lower()
                    try:
                        if ext == ".txt" or ext in (".md", ".markdown"):
                            from core.knowledge.document_loader import extract_text_from_txt
                            doc_text = extract_text_from_txt(str(stored_path))
                        elif ext == ".pdf":
                            from core.knowledge.document_loader import extract_text_from_pdf
                            doc_text = extract_text_from_pdf(str(stored_path))
                        elif ext == ".docx":
                            from core.knowledge.document_loader import extract_text_from_docx
                            doc_text = extract_text_from_docx(str(stored_path))
                    except Exception as e:
                        logger.debug(f"[ArticleGen] Could not read file {file_name}: {e}")

                # ── Strategy 2: Fallback to chunked text from index ──
                if not doc_text:
                    chunks = doc_info.get("chunks", [])
                    if chunks:
                        doc_text = "\n".join(c.get("text", "") for c in chunks)

                if not doc_text:
                    continue

                # Find all URLs in document with surrounding context
                for m in url_pattern.finditer(doc_text):
                    url = m.group(0).rstrip(".,;:)")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # Get the line containing the URL for context
                    line_start = doc_text.rfind("\n", 0, m.start()) + 1
                    line_end = doc_text.find("\n", m.end())
                    if line_end == -1:
                        line_end = len(doc_text)
                    context_line = doc_text[line_start:line_end].strip()

                    all_urls.append({
                        "url": url,
                        "context": context_line,
                        "doc_name": doc_name,
                    })

            if not all_urls:
                logger.info("[ArticleGen] No URLs found in uploaded documents")
                return article_text, internal_links

            logger.info(f"[ArticleGen] Found {len(all_urls)} unique URLs in uploaded docs")

            # ── Step 1.5: Filter out URLs from mismatched platforms ──
            article_platforms = InternalLinkInjector._detect_platforms(keyword)
            if article_platforms:
                filtered = []
                for u in all_urls:
                    url_text = f"{u.get('context', '')} {u['url']}"
                    url_platforms = InternalLinkInjector._detect_platforms(url_text)
                    if url_platforms and not url_platforms.intersection(article_platforms):
                        logger.debug(
                            f"[ArticleGen] Platform mismatch EXCLUDED: "
                            f"article={article_platforms}, url={url_platforms} → {u['url']}"
                        )
                        continue
                    filtered.append(u)
                logger.info(
                    f"[ArticleGen] Platform filter: {len(all_urls)} → {len(filtered)} URLs"
                )
                all_urls = filtered

            if not all_urls:
                logger.info("[ArticleGen] No relevant URLs after platform filter")
                return article_text, internal_links

            # ── Step 2: Send ALL URLs + article to LLM for placement ──
            urls_info = "\n".join(
                f"- URL: {u['url']}\n  Mô tả: {u['context'][:200]}"
                for u in all_urls
            )

            user_prompt = f"""BÀI VIẾT:
{article_text}

═══ DANH SÁCH URL TỪ TÀI LIỆU ĐÃ TẢI LÊN ═══
{urls_info}
═══ END URLs ═══

Hãy chèn internal link vào bài viết. CHỈ sử dụng các URL ở trên — KHÔNG được tự tạo URL.

RULES:
1. CHỈ dùng URL có trong danh sách trên — TUYỆT ĐỐI không tự bịa URL
2. Mỗi URL chỉ dùng ĐÚNG 1 lần duy nhất trong toàn bài
3. Chèn link khi URL liên quan đến nội dung đoạn văn đó
4. Anchor text phải TỰ NHIÊN — là cụm từ có sẵn trong bài viết
5. KHÔNG chèn link vào heading hoặc câu đầu tiên của đoạn
6. KHÔNG gắn link vào từ chung chung ('tại đây', 'xem thêm', 'chi tiết')
7. Nếu URL không phù hợp với bất kỳ đoạn nào thì BỎ QUA — không cần ép

Trả về JSON:
{{"link_insertions": [
    {{"find_text": "đoạn text gốc", "replace_text": "đoạn text có [anchor](url)", "anchor_text": "anchor", "url": "url", "position": "section name"}}
]}}"""

            result = llm.chat_json(
                PromptTemplates.ARTICLE_LINK_SYSTEM,
                user_prompt,
                temperature=0.3,
            )

            insertions = result.get("link_insertions", [])

            for ins in insertions:
                find_text = ins.get("find_text", "")
                replace_text = ins.get("replace_text", "")
                url = ins.get("url", "")

                # Validate: URL must be from the uploaded docs
                if url not in seen_urls:
                    logger.warning(f"[ArticleGen] Skipping unknown URL: {url}")
                    continue

                if find_text and replace_text and find_text in article_text:
                    article_text = article_text.replace(find_text, replace_text, 1)
                    internal_links.append({
                        "url": url,
                        "anchor_text": ins.get("anchor_text", ""),
                        "position": ins.get("position", ""),
                    })
                    logger.info(f"[ArticleGen] Link inserted: {url}")

        except Exception as e:
            logger.warning(f"[ArticleGen] Internal linking failed: {e}")

        return article_text, internal_links

    # ──────────────────────────────────────────
    # Layer C: HTML Conversion (WordPress-ready)
    # ──────────────────────────────────────────

    @staticmethod
    def _format_ol_li_content(content: str) -> str:
        """Split '<strong>Title:</strong> body' onto separate lines for clean HTML."""
        m = re.match(r'^(<strong>[^<]+</strong>)\s+(.*)', content, re.DOTALL)
        if m and m.group(2).strip():
            return f"{m.group(1)}\n    {m.group(2).strip()}"
        return content

    def _convert_to_html(
        self,
        article_text: str,
        title: str,
        internal_links: list,
    ) -> str:
        """Convert markdown article to WordPress-ready HTML."""

        html = article_text

        # Convert headings
        html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Strip bold/italic markers around links BEFORE conversion
        # e.g. **[text](url)** → [text](url)
        html = re.sub(r'\*{1,3}(\[[^\]]+\]\([^\)]+\))\*{1,3}', r'\1', html)

        # Convert markdown links FIRST (before bold/italic to avoid conflicts)
        # Build set of known internal URLs from KB for dofollow/nofollow decision
        internal_url_set = set()
        if internal_links:
            for lnk in internal_links:
                u = lnk.get("url", "") if isinstance(lnk, dict) else ""
                if u:
                    internal_url_set.add(u.rstrip("/"))

        seen_urls = set()  # Track URLs to deduplicate

        def _link_replacer(match):
            anchor = match.group(1)
            url = match.group(2)
            url_normalized = url.rstrip("/")

            # Deduplicate: if this exact URL already appeared, skip (keep anchor text only)
            if url_normalized in seen_urls:
                return anchor
            seen_urls.add(url_normalized)

            # Internal links → no nofollow; External links → nofollow
            # All target="_blank" links get noopener noreferrer for security
            if url_normalized in internal_url_set:
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="font-weight: bold;">{anchor}</a>'
            else:
                return f'<a href="{url}" target="_blank" rel="nofollow noopener noreferrer" style="font-weight: bold;">{anchor}</a>'

        html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', _link_replacer, html)

        # Convert bold and italic (AFTER links, so ** won't wrap link syntax)
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Clean up any stray * markers around <a> tags
        html = re.sub(r'\*{1,3}(<a\s[^>]*>.*?</a>)\*{1,3}', r'\1', html)

        # Convert unordered lists
        lines = html.split('\n')
        result_lines = []
        in_ol = False            # Inside an <ol>
        ol_li_open = False       # Unclosed <li> inside <ol> (deferred close for nesting)
        in_nested_ul = False     # Inside <ul> nested within <ol><li>
        in_standalone_ul = False # Inside a standalone <ul>
        ol_counter = 0
        in_table = False
        table_lines = []

        for line in lines:
            stripped = line.strip()

            # ── Table handling ──
            if stripped.startswith('|') and stripped.endswith('|'):
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(stripped)
                continue
            elif in_table:
                in_table = False
                result_lines.append(self._convert_table(table_lines))
                table_lines = []

            is_ul_item = bool(re.match(r'^[-*+] ', stripped))
            is_ol_item = bool(re.match(r'^\d+\.\s', stripped))
            is_heading = stripped.startswith('<h')
            is_empty = not stripped

            # ── Closing logic (only on non-empty lines) ──
            if not is_empty:
                # Close nested <ul> if moving away from bullets
                if in_nested_ul and not is_ul_item:
                    result_lines.append('    </ul>')
                    in_nested_ul = False

                # Close open ol <li> if NOT a nested bullet
                if ol_li_open and not is_ul_item:
                    result_lines.append('  </li>')
                    ol_li_open = False

                # Close <ol> completely if meeting non-list content
                if in_ol and not is_ol_item and not is_ul_item:
                    result_lines.append('</ol>')
                    in_ol = False
                    ol_counter = 0

                # Close standalone <ul> if not a bullet
                if in_standalone_ul and not is_ul_item:
                    result_lines.append('</ul>')
                    in_standalone_ul = False

            # ── Ordered list item ──
            if is_ol_item:
                if not in_ol:
                    result_lines.append('<ol>')
                    in_ol = True
                    ol_counter = 0
                content = re.sub(r'^\d+\.\s', '', stripped)
                formatted = self._format_ol_li_content(content)
                result_lines.append(f'\n  <li>\n    {formatted}')
                ol_li_open = True
                ol_counter += 1
                continue

            # ── Unordered list item ──
            if is_ul_item:
                content = re.sub(r'^[-*+] ', '', stripped)
                if in_ol and ol_li_open:
                    # Nest inside the current <ol><li>
                    if not in_nested_ul:
                        result_lines.append('    <ul>')
                        in_nested_ul = True
                    result_lines.append(f'      <li>{content}</li>')
                else:
                    # Standalone <ul>
                    if not in_standalone_ul:
                        result_lines.append('<ul>')
                        in_standalone_ul = True
                    result_lines.append(f'<li>{content}</li>')
                continue

            # ── Blockquotes ──
            if stripped.startswith('&gt;') or stripped.startswith('>'):
                quote_content = re.sub(r'^(&gt;|>)\s*', '', stripped)
                result_lines.append(
                    f'<blockquote style="border-left: 4px solid #e0e0e0; padding: 10px 20px; '
                    f'margin: 15px 0; font-style: italic; color: #555;">{quote_content}</blockquote>'
                )
                continue

            # ── Paragraphs ──
            # Block-level tags that should NOT be wrapped in <p>
            _block_tags = ('<h', '<div', '<table', '<ol', '<ul', '<li', '<blockquote',
                           '<figure', '<img', '<hr', '<br', '<pre', '<code', '</') 
            is_block = stripped.startswith(_block_tags)
            if stripped and not is_heading and not is_block:
                result_lines.append(f'<p>{stripped}</p>')
            elif stripped:
                result_lines.append(stripped)
            else:
                result_lines.append('')

        # ── Cleanup: close any open tags ──
        if in_nested_ul:
            result_lines.append('    </ul>')
        if ol_li_open:
            result_lines.append('  </li>')
        if in_ol:
            result_lines.append('</ol>')
        if in_standalone_ul:
            result_lines.append('</ul>')
        if in_table:
            result_lines.append(self._convert_table(table_lines))

        html = '\n'.join(result_lines)

        # Clean up multiple blank lines
        html = re.sub(r'\n{3,}', '\n\n', html)

        # ── Final link attribute validation ──
        # Ensure every <a> tag has correct rel attribute
        def _validate_link_attrs(m):
            tag = m.group(0)
            href_match = re.search(r'href="([^"]+)"', tag)
            if not href_match:
                return tag
            url = href_match.group(1).rstrip('/')
            anchor_content = m.group(1)
            # Remove any existing rel attribute
            tag_clean = re.sub(r'\s*rel="[^"]*"', '', tag)
            if url in internal_url_set:
                # Internal link -> noopener noreferrer only
                tag_clean = tag_clean.replace('>', ' rel="noopener noreferrer">', 1)
            else:
                # External link -> nofollow noopener noreferrer
                tag_clean = tag_clean.replace('>', ' rel="nofollow noopener noreferrer">', 1)
            return tag_clean

        html = re.sub(r'<a\s[^>]*>(.*?)</a>', _validate_link_attrs, html)

        return html.strip()

    def _convert_table(self, table_lines: list) -> str:
        """Convert markdown table to inline-styled HTML table (WordPress-ready)."""
        if len(table_lines) < 2:
            return ''

        table_style = 'width: 100%; border-collapse: collapse; margin: 25px 0; font-family: sans-serif; min-width: 400px; border: 1px solid #dddddd;'
        html = f'<table style="{table_style}">\n'

        # Header row
        headers = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
        html += '<thead>\n<tr style="background-color: #f8f8f8; text-align: left;">\n'
        for i, h in enumerate(headers):
            border_right = ' border-right: 2px solid #eeeeee;' if i < len(headers) - 1 else ''
            html += f'<th style="padding: 12px 15px; border-bottom: 2px solid #eeeeee; color: #333333; font-weight: bold;{border_right} text-align:center;">{h}</th>\n'
        html += '</tr>\n</thead>\n'

        # Body rows (skip separator row)
        html += '<tbody>\n'
        data_rows = table_lines[2:]  # Skip header and separator
        for row_line in data_rows:
            cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
            html += '<tr style="border-bottom: 1px solid #dddddd;">\n'
            for i, cell in enumerate(cells):
                border_right = ' border-right: 2px solid #eeeeee;' if i < len(cells) - 1 else ''
                if i == 0:
                    html += f'<td style="padding: 12px 15px; font-weight: bold; color: #555555;{border_right} text-align:center;">{cell}</td>\n'
                else:
                    html += f'<td style="padding: 12px 15px; color: #666666;{border_right}">{cell}</td>\n'
            html += '</tr>\n'
        html += '</tbody>\n</table>'

        return html
