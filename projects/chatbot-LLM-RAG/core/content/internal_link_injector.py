"""
Internal Link Injector — Pre-calculates and injects internal links per section.

Layer 3 of the 3-layer article generation architecture:
    1. Before generation: extract all URLs from knowledge base documents
    2. Pre-calculate 2-4 relevant links per H2 section (semantic matching)
    3. Pass pre-calculated links into section prompts so AI writes them naturally
    4. Post-generation: sentence-level semantic injection (replaces LLM-chosen placement)
    5. Validate only KB-sourced URLs were used

v2 improvements:
    - Sentence-level semantic matching: after content is generated, links are injected
      into sentences with highest topical similarity (not LLM-chosen placement)
    - This prevents the "sales-like" feel of AI placing links too prominently
    - Links land in contextually relevant sentences, not promotional paragraphs
"""

import logging
import re
from pathlib import Path
from typing import Optional

from core.knowledge.retrieval import has_knowledge_base
from core.knowledge.embedding_engine import (
    create_single_embedding,
    load_vectors,
    cosine_similarity_batch,
)

logger = logging.getLogger(__name__)


class InternalLinkInjector:
    """Pre-calculates and manages internal links for article sections."""

    # Max links to suggest per H2 section
    MAX_LINKS_PER_SECTION = 10
    MIN_LINKS_PER_SECTION = 1

    # Similarity threshold for link-to-section relevance
    RELEVANCE_THRESHOLD = 0.25

    # Penalty applied to relevance score for each time a URL has already been used
    # Keep low — URLs CAN and SHOULD be reused across sections
    REUSE_PENALTY = 0.05

    def __init__(self):
        self._url_cache: list[dict] | None = None
        self._used_urls: dict[str, int] = {}  # url -> usage count

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def extract_all_urls(self) -> list[dict]:
        """Extract all URLs from uploaded knowledge base documents.

        Returns:
            List of dicts: [{url, context, description, doc_name}, ...]
        """
        if self._url_cache is not None:
            return self._url_cache

        if not has_knowledge_base():
            self._url_cache = []
            return self._url_cache

        try:
            from core.knowledge.document_loader import _load_index, UPLOAD_DIR

            index = _load_index()
            documents = index.get("documents", {})

            url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')
            all_urls = []
            seen_urls = set()

            for doc_id, doc_info in documents.items():
                doc_name = doc_info.get("original_name", doc_id)

                # Read document text
                file_name = doc_info.get("saved_name") or doc_info.get("stored_name", "")
                stored_path = UPLOAD_DIR / file_name if file_name else None
                doc_text = ""

                if stored_path and stored_path.exists() and stored_path.is_file():
                    ext = stored_path.suffix.lower()
                    try:
                        if ext in (".txt", ".md", ".markdown"):
                            from core.knowledge.document_loader import extract_text_from_txt
                            doc_text = extract_text_from_txt(str(stored_path))
                        elif ext == ".pdf":
                            from core.knowledge.document_loader import extract_text_from_pdf
                            doc_text = extract_text_from_pdf(str(stored_path))
                        elif ext == ".docx":
                            from core.knowledge.document_loader import extract_text_from_docx
                            doc_text = extract_text_from_docx(str(stored_path))
                    except Exception as e:
                        logger.debug(f"[LinkInjector] Could not read {file_name}: {e}")

                # Fallback to chunks
                if not doc_text:
                    chunks = doc_info.get("chunks", [])
                    if chunks:
                        doc_text = "\n".join(c.get("text", "") for c in chunks)

                if not doc_text:
                    continue

                # Extract URLs with context
                for m in url_pattern.finditer(doc_text):
                    url = m.group(0).rstrip(".,;:)")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # Get surrounding context (the line containing the URL)
                    line_start = doc_text.rfind("\n", 0, m.start()) + 1
                    line_end = doc_text.find("\n", m.end())
                    if line_end == -1:
                        line_end = len(doc_text)
                    context_line = doc_text[line_start:line_end].strip()

                    # Try to extract a description/title near the URL
                    description = self._extract_url_description(doc_text, m.start(), url)

                    all_urls.append({
                        "url": url,
                        "context": context_line,
                        "description": description,
                        "doc_name": doc_name,
                    })

            self._url_cache = all_urls
            logger.info(f"[LinkInjector] Extracted {len(all_urls)} unique URLs from KB")
            return all_urls

        except Exception as e:
            logger.warning(f"[LinkInjector] URL extraction failed: {e}")
            self._url_cache = []
            return self._url_cache

    # Platform/topic keywords for cross-platform mismatch detection
    _PLATFORM_KEYWORDS = {
        "facebook": {"facebook", "fb", "fanpage", "facebook ads", "facebook boost"},
        "tiktok": {"tiktok", "tik tok", "tiktok boost", "tiktok trust"},
        "youtube": {"youtube", "yt", "youtube ads"},
        "instagram": {"instagram", "ig", "insta"},
        "twitter": {"twitter", "x.com", "tweet"},
        "zalo": {"zalo"},
        "shopee": {"shopee"},
        "lazada": {"lazada"},
        "google": {"google", "google ads", "adwords"},
        "threads": {"threads"},
        "pinterest": {"pinterest"},
        "linkedin": {"linkedin"},
    }

    @classmethod
    def _detect_platforms(cls, text: str) -> set[str]:
        """Detect which platforms are mentioned in text."""
        text_lower = text.lower()
        found = set()
        for platform, keywords in cls._PLATFORM_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                found.add(platform)
        return found

    def pre_calculate_links(
        self,
        section_heading: str,
        section_purpose: str,
        section_key_points: list[str],
        children_headings: list[str] | None = None,
        children_purposes: list[str] | None = None,
        article_keyword: str = "",
    ) -> list[dict]:
        """Pre-calculate relevant internal links for a specific H2 section.

        Uses semantic similarity between section context (including H3/H4 children)
        and URL descriptions to find the most relevant links.

        Args:
            section_heading: The H2 heading text
            section_purpose: Purpose/description of the section
            section_key_points: Key points the section will cover
            children_headings: Headings of H3/H4 children under this H2
            children_purposes: Purposes of H3/H4 children
            article_keyword: The main keyword of the article (for platform filtering)

        Returns:
            List of relevant link dicts: [{url, anchor_suggestion, context, relevance_score}, ...]
        """
        all_urls = self.extract_all_urls()
        if not all_urls:
            return []

        # Filter out URLs from mismatched platforms BEFORE any scoring
        article_context_text = f"{article_keyword} {section_heading} {section_purpose}"
        article_platforms = self._detect_platforms(article_context_text)
        if article_platforms:
            filtered = []
            for u in all_urls:
                url_text = f"{u.get('description', '')} {u.get('context', '')} {u['url']}"
                url_platforms = self._detect_platforms(url_text)
                if url_platforms and not url_platforms.intersection(article_platforms):
                    logger.debug(
                        f"[LinkInjector] Platform mismatch EXCLUDED: "
                        f"article={article_platforms}, url={url_platforms} → {u['url']}"
                    )
                    continue
                filtered.append(u)
            logger.info(
                f"[LinkInjector] Platform filter: {len(all_urls)} → {len(filtered)} URLs "
                f"(excluded {len(all_urls) - len(filtered)} mismatched)"
            )
            all_urls = filtered

        if not all_urls:
            return []

        # Build rich section context including children H3/H4
        context_parts = [section_heading, section_purpose]
        context_parts.extend(section_key_points)
        if children_headings:
            context_parts.extend(children_headings)
        if children_purposes:
            context_parts.extend(children_purposes)
        section_text = ". ".join(p for p in context_parts if p)

        # Also build keyword set from all headings for direct matching
        all_heading_text = f"{section_heading} {' '.join(children_headings or [])}".lower()

        try:
            # Create embedding for section context
            section_embedding = create_single_embedding(section_text)
            if section_embedding is None:
                return self._fallback_keyword_match(
                    section_heading, all_urls,
                    children_headings=children_headings,
                )

            # Create embeddings for URL contexts
            url_texts = [
                f"{u['description']} {u['context'][:200]}" for u in all_urls
            ]

            # Calculate similarity
            scored_urls = []
            for i, url_info in enumerate(all_urls):
                url_embedding = create_single_embedding(url_texts[i])
                if url_embedding is None:
                    continue

                similarity = float(cosine_similarity_batch(
                    section_embedding.reshape(1, -1),
                    url_embedding.reshape(1, -1),
                )[0][0])

                # Boost: if the URL description/context keywords appear directly in headings
                keyword_boost = self._calc_keyword_boost(url_info, all_heading_text)
                similarity = min(1.0, similarity + keyword_boost)

                # Apply reuse penalty: reduce score for URLs already used in other sections
                use_count = self._used_urls.get(url_info["url"], 0)
                adjusted_similarity = similarity - (use_count * self.REUSE_PENALTY)

                if adjusted_similarity >= self.RELEVANCE_THRESHOLD:
                    scored_urls.append({
                        "url": url_info["url"],
                        "anchor_suggestion": url_info["description"] or self._suggest_anchor(url_info),
                        "context": url_info["context"][:200],
                        "relevance_score": round(adjusted_similarity, 3),
                        "doc_name": url_info["doc_name"],
                        "is_reuse": use_count > 0,
                    })

            # Sort by relevance, take top N
            scored_urls.sort(key=lambda x: x["relevance_score"], reverse=True)
            selected = scored_urls[:self.MAX_LINKS_PER_SECTION]

            # Track usage count
            for link in selected:
                self._used_urls[link["url"]] = self._used_urls.get(link["url"], 0) + 1

            logger.info(
                f"[LinkInjector] Section '{section_heading}' — "
                f"{len(selected)} links pre-calculated "
                f"(from {len(scored_urls)} candidates)"
            )
            return selected

        except Exception as e:
            logger.warning(f"[LinkInjector] Semantic matching failed: {e}")
            return self._fallback_keyword_match(
                section_heading, all_urls,
                children_headings=children_headings,
            )

    def validate_links_in_content(self, content: str) -> dict:
        """Validate that all links in the content are from the knowledge base.

        Args:
            content: The generated article content (Markdown)

        Returns:
            dict with valid_links, invalid_links, and total counts
        """
        all_urls = self.extract_all_urls()
        kb_urls = {u["url"] for u in all_urls}

        # Find all Markdown links in content
        link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
        found_links = link_pattern.findall(content)

        valid = []
        invalid = []

        for anchor, url in found_links:
            url_clean = url.rstrip(".,;:)")
            if url_clean in kb_urls:
                valid.append({"anchor": anchor, "url": url_clean})
            else:
                invalid.append({"anchor": anchor, "url": url_clean})

        if invalid:
            logger.warning(
                f"[LinkInjector] {len(invalid)} invalid links found "
                f"(not in KB): {[l['url'] for l in invalid]}"
            )

        return {
            "valid_links": valid,
            "invalid_links": invalid,
            "total_links": len(found_links),
            "valid_count": len(valid),
            "invalid_count": len(invalid),
        }

    def remove_invalid_links(self, content: str) -> str:
        """Remove links that are not from the knowledge base, keeping the anchor text.

        Args:
            content: Article content with Markdown links

        Returns:
            Content with invalid links converted to plain text
        """
        all_urls = self.extract_all_urls()
        kb_urls = {u["url"] for u in all_urls}

        def _check_link(match):
            anchor = match.group(1)
            url = match.group(2).rstrip(".,;:)")
            if url in kb_urls:
                return match.group(0)  # Keep valid link
            else:
                return anchor  # Remove link, keep text

        return re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', _check_link, content)

    def inject_links_semantic(
        self,
        section_content: str,
        pre_calculated_links: list[dict],
        section_heading: str = "",
    ) -> tuple[str, int]:
        """Post-generation sentence-level semantic link injection.

        Instead of letting the LLM decide where to place links (which often feels
        promotional/sales-like), this method:
            1. Splits section into individual sentences
            2. Creates embeddings for each sentence
            3. Matches each link to the most semantically similar sentence
            4. Injects the link into that sentence naturally

        This ensures links appear in contextually relevant sentences, not in
        promotional paragraphs the LLM might create.

        Args:
            section_content: Generated section content (Markdown, no links yet)
            pre_calculated_links: Links from pre_calculate_links()
            section_heading: H2 heading for logging

        Returns:
            (content_with_links, count_of_injected_links)
        """
        if not pre_calculated_links:
            return section_content, 0

        # Remove any existing links inserted by LLM (we'll re-inject semantically)
        clean_content = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', section_content)

        # Split into lines, identify content lines (not headings, not empty)
        lines = clean_content.split('\n')
        content_lines_info = []  # (line_index, sentence_text, is_heading)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            is_heading = stripped.startswith('#')
            if is_heading:
                continue

            # Split line into sentences for finer granularity
            sentences = re.split(r'(?<=[.!?])\s+', stripped)
            for sent in sentences:
                if len(sent.split()) >= 5:  # Only consider sentences with 5+ words
                    content_lines_info.append({
                        "line_index": i,
                        "sentence": sent,
                        "original_line": line,
                    })

        if not content_lines_info:
            return section_content, 0

        injected_count = 0
        used_line_indices = set()  # Don't inject 2 links into same line

        try:
            for link in pre_calculated_links:
                anchor = link.get("anchor_suggestion", "")
                url = link.get("url", "")
                if not anchor or not url:
                    continue

                # Find the best sentence for this link using semantic similarity
                link_text = f"{anchor} {link.get('context', '')[:100]}"
                link_embedding = create_single_embedding(link_text)
                if link_embedding is None:
                    continue

                best_score = -1
                best_info = None

                for info in content_lines_info:
                    if info["line_index"] in used_line_indices:
                        continue

                    sent_embedding = create_single_embedding(info["sentence"])
                    if sent_embedding is None:
                        continue

                    similarity = float(cosine_similarity_batch(
                        link_embedding.reshape(1, -1),
                        sent_embedding.reshape(1, -1),
                    )[0][0])

                    # Boost if anchor text keywords appear in sentence
                    anchor_words = set(anchor.lower().split())
                    sent_words = set(info["sentence"].lower().split())
                    keyword_overlap = len(anchor_words & sent_words) / max(len(anchor_words), 1)
                    similarity += keyword_overlap * 0.15

                    if similarity > best_score:
                        best_score = similarity
                        best_info = info

                # Only inject if similarity is reasonable (> 0.25)
                if best_info and best_score > 0.25:
                    line_idx = best_info["line_index"]
                    original_line = lines[line_idx]

                    # Find the best word/phrase in this line to use as anchor
                    actual_anchor = self._find_best_anchor_in_sentence(
                        original_line, anchor, url
                    )

                    if actual_anchor:
                        # Replace first occurrence of anchor with linked version
                        escaped = re.escape(actual_anchor)
                        new_line = re.sub(
                            escaped,
                            f'[{actual_anchor}]({url})',
                            original_line,
                            count=1,
                        )
                        if new_line != original_line:
                            lines[line_idx] = new_line
                            used_line_indices.add(line_idx)
                            injected_count += 1
                            logger.debug(
                                f"[LinkInjector] Semantic inject: '{actual_anchor}' → {url} "
                                f"(score: {best_score:.3f})"
                            )

        except Exception as e:
            logger.warning(f"[LinkInjector] Semantic injection failed for '{section_heading}': {e}")
            # Fall back to returning clean content without links
            return clean_content, 0

        result = '\n'.join(lines)
        logger.info(
            f"[LinkInjector] Semantic injection for '{section_heading}': "
            f"{injected_count}/{len(pre_calculated_links)} links placed"
        )
        return result, injected_count

    @staticmethod
    def _find_best_anchor_in_sentence(sentence: str, suggested_anchor: str, url: str) -> str:
        """Find the best phrase in a sentence to use as anchor text for a link.

        Looks for:
        1. Exact match of suggested anchor
        2. Partial match (product name from URL slug)
        3. Longest relevant phrase overlap

        Returns:
            The best anchor text found in the sentence, or empty string
        """
        # 1. Try exact anchor match (case-insensitive)
        pattern = re.compile(re.escape(suggested_anchor), re.IGNORECASE)
        match = pattern.search(sentence)
        if match:
            return match.group(0)

        # 2. Try product name from URL
        slug = url.rstrip('/').split('/')[-1].replace('-', ' ')
        if slug and len(slug) >= 4:
            slug_pattern = re.compile(re.escape(slug), re.IGNORECASE)
            match = slug_pattern.search(sentence)
            if match:
                return match.group(0)

        # 3. Try partial anchor words (at least 2 consecutive words from anchor)
        anchor_words = suggested_anchor.split()
        if len(anchor_words) >= 2:
            for length in range(len(anchor_words), 1, -1):
                for start in range(len(anchor_words) - length + 1):
                    phrase = ' '.join(anchor_words[start:start + length])
                    if len(phrase) >= 4:
                        phrase_pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                        match = phrase_pattern.search(sentence)
                        if match:
                            return match.group(0)

        return ""

    def inject_missing_links(self, article_text: str, article_keyword: str = "") -> tuple[str, int]:
        """Post-generation: scan article text and inject links where products are mentioned but not linked.

        Builds a mapping of product names → URLs from the KB, then scans the article
        for unlinked mentions and wraps them with markdown links.

        Args:
            article_text: Full article in Markdown format
            article_keyword: Main keyword of the article (for platform filtering)

        Returns:
            (updated_text, count_of_injected_links)
        """
        all_urls = self.extract_all_urls()
        if not all_urls:
            return article_text, 0

        # Filter out URLs from mismatched platforms
        article_platforms = self._detect_platforms(article_keyword) if article_keyword else set()
        if article_platforms:
            filtered_urls = []
            for u in all_urls:
                url_text = f"{u.get('description', '')} {u.get('context', '')} {u['url']}"
                url_platforms = self._detect_platforms(url_text)
                if not url_platforms or url_platforms.intersection(article_platforms):
                    filtered_urls.append(u)
                else:
                    logger.debug(
                        f"[LinkInjector] inject_missing: skipping {u['url']} "
                        f"(platform mismatch: article={article_platforms}, url={url_platforms})"
                    )
            all_urls = filtered_urls

        # Build product name → URL mapping from KB
        product_links = self._build_product_link_map(all_urls)
        if not product_links:
            return article_text, 0

        # Sort product names by length (longest first) to avoid partial matches
        sorted_products = sorted(product_links.keys(), key=len, reverse=True)

        injected_count = 0
        lines = article_text.split('\n')
        result_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip heading lines — don't inject links into headings
            if stripped.startswith('#'):
                result_lines.append(line)
                continue

            # Skip empty lines
            if not stripped:
                result_lines.append(line)
                continue

            # Process this content line — inject links for unlinked product mentions
            for product_name in sorted_products:
                url = product_links[product_name]
                escaped_name = re.escape(product_name)

                # Find all mentions NOT already inside a markdown link [...](...)
                # Strategy: split line into linked vs unlinked segments, only modify unlinked
                new_line = self._inject_link_in_line(line, escaped_name, product_name, url)
                if new_line != line:
                    injected_count += new_line.count(f']({url})') - line.count(f']({url})')
                    line = new_line

            result_lines.append(line)

        result = '\n'.join(result_lines)
        logger.info(f"[LinkInjector] Post-gen injection: {injected_count} links added")
        return result, injected_count

    @staticmethod
    def _inject_link_in_line(line: str, escaped_name: str, product_name: str, url: str) -> str:
        """Inject markdown link for product_name in a single line, skipping already-linked segments."""
        # Split the line into segments: already-linked parts vs plain text
        # Regex to find existing markdown links
        link_pattern = re.compile(r'\[[^\]]*\]\([^\)]*\)')

        segments = []
        last_end = 0
        for m in link_pattern.finditer(line):
            if m.start() > last_end:
                segments.append(('text', line[last_end:m.start()]))
            segments.append(('link', m.group(0)))
            last_end = m.end()
        if last_end < len(line):
            segments.append(('text', line[last_end:]))

        # Only replace in 'text' segments
        modified = False
        result_parts = []
        name_pattern = re.compile(escaped_name, re.IGNORECASE)

        for seg_type, seg_text in segments:
            if seg_type == 'link':
                result_parts.append(seg_text)
            else:
                new_text = name_pattern.sub(
                    lambda m: f'[{m.group(0)}]({url})',
                    seg_text
                )
                if new_text != seg_text:
                    modified = True
                result_parts.append(new_text)

        return ''.join(result_parts) if modified else line

    def _build_product_link_map(self, all_urls: list[dict]) -> dict[str, str]:
        """Build a mapping of identifiable product/page names to their URLs.

        Returns:
            dict: {product_name_lower: url}
        """
        product_map = {}

        for url_info in all_urls:
            url = url_info["url"]
            description = (url_info.get("description") or "").strip()

            # Extract product name from description
            # e.g., "Acme Cloud Suite" from "Acme Cloud Suite: Công cụ giúp..."
            if description:
                # Take the part before ':' or the first sentence
                name = description.split(':')[0].strip()
                name = re.sub(r'^(công cụ|phần mềm|phần cứng|nền tảng|gói|giải pháp|sản phẩm|thuê)\s*', '', name, flags=re.IGNORECASE).strip()
                if name and len(name) >= 4:
                    product_map[name] = url

            # Also extract from URL slug as fallback
            slug = url.rstrip('/').split('/')[-1].replace('-', ' ').strip()
            if slug and len(slug) >= 5:
                # Convert slug to title case for matching
                slug_title = slug.title()
                if slug_title not in product_map:
                    product_map[slug_title] = url

        logger.info(f"[LinkInjector] Product link map: {len(product_map)} entries: {list(product_map.keys())}")
        return product_map

    def reset(self):
        """Reset used URLs tracking (call before generating a new article)."""
        self._used_urls = {}
        self._url_cache = None

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _extract_url_description(self, doc_text: str, url_pos: int, url: str) -> str:
        """Try to extract a human-readable description near the URL."""
        # Look for text before the URL on the same line
        line_start = doc_text.rfind("\n", 0, url_pos) + 1
        before_url = doc_text[line_start:url_pos].strip()

        # Common patterns: "Product Name: URL" or "- Product Name (URL)"
        if before_url:
            # Remove common prefixes
            cleaned = re.sub(r'^[-*•]\s*', '', before_url)
            cleaned = re.sub(r'[:\(\[]\s*$', '', cleaned).strip()
            if cleaned and len(cleaned) > 3:
                return cleaned

        # Try to extract product/page name from URL
        path = url.rstrip("/").split("/")[-1]
        name = path.replace("-", " ").replace("_", " ").strip()
        if name and len(name) > 2:
            return name.title()

        return url

    def _suggest_anchor(self, url_info: dict) -> str:
        """Suggest anchor text for a URL."""
        if url_info.get("description"):
            return url_info["description"]

        # Extract from URL path
        url = url_info["url"]
        path = url.rstrip("/").split("/")[-1]
        anchor = path.replace("-", " ").replace("_", " ").strip()
        return anchor.title() if anchor else url

    def _fallback_keyword_match(
        self, section_heading: str, all_urls: list[dict],
        children_headings: list[str] | None = None,
    ) -> list[dict]:
        """Fallback: match URLs to section by keyword overlap (including children)."""
        # Combine H2 heading + all children headings
        all_text = section_heading
        if children_headings:
            all_text += " " + " ".join(children_headings)

        heading_words = set(all_text.lower().split())
        # Remove common stop words
        stop_words = {"của", "và", "cho", "với", "là", "các", "một", "trong", "để",
                       "the", "a", "an", "is", "are", "for", "with", "and", "of", "to", "in",
                       "-", "–", ":", "về", "khi", "hay", "hoặc", "như", "theo", "từ", "đến"}
        heading_words -= stop_words

        if not heading_words:
            return []

        scored = []
        for url_info in all_urls:
            context_words = set(url_info["context"].lower().split())
            desc_words = set(url_info["description"].lower().split()) if url_info.get("description") else set()
            all_url_words = context_words | desc_words

            overlap = heading_words & all_url_words
            if overlap:
                use_count = self._used_urls.get(url_info["url"], 0)
                base_score = len(overlap) / len(heading_words)
                adjusted_score = base_score - (use_count * self.REUSE_PENALTY)
                if adjusted_score > 0:
                    scored.append({
                        "url": url_info["url"],
                        "anchor_suggestion": url_info.get("description", "") or self._suggest_anchor(url_info),
                        "context": url_info["context"][:200],
                        "relevance_score": adjusted_score,
                        "doc_name": url_info["doc_name"],
                        "is_reuse": use_count > 0,
                    })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        selected = scored[:self.MAX_LINKS_PER_SECTION]

        for link in selected:
            self._used_urls[link["url"]] = self._used_urls.get(link["url"], 0) + 1

        return selected

    @staticmethod
    def _calc_keyword_boost(url_info: dict, headings_text_lower: str) -> float:
        """Calculate a relevance boost if URL description keywords appear in section headings.

        This catches direct product name mentions like 'Acme Cloud Suite' in headings
        matched against URL descriptions like 'Acme Cloud Suite: ...'

        Returns:
            Boost value (0.0 to 0.35)
        """
        description = (url_info.get("description") or "").lower().strip()
        if not description or len(description) < 4:
            return 0.0

        # Extract meaningful multi-word product names from description
        # e.g. "Acme Cloud Suite" → check if this appears in headings
        # Remove common prefixes like "Công cụ giúp..."
        clean_desc = re.sub(
            r'^(công cụ|phần mềm|phần cứng|nền tảng|gói|giải pháp|sản phẩm)\s+(giúp\s+)?',
            '', description
        ).strip()

        # Check if the product name (from URL path) appears in headings
        url = url_info.get("url", "")
        url_slug = url.rstrip("/").split("/")[-1].replace("-", " ").lower()

        # Direct product name match in headings → strong boost
        if url_slug and len(url_slug) > 5 and url_slug in headings_text_lower:
            return 0.35

        # Check if description keywords overlap significantly with headings
        desc_words = set(clean_desc.split())
        heading_words = set(headings_text_lower.split())
        stop_words = {"của", "và", "cho", "với", "là", "các", "một", "trong", "để",
                       "the", "a", "an", "is", "are", "for", "with", "and", "of", "to",
                       "giúp", "bạn", "như", "có", "được", "này", "đó", "rất", "để"}
        desc_words -= stop_words
        heading_words -= stop_words

        if not desc_words:
            return 0.0

        overlap = desc_words & heading_words
        ratio = len(overlap) / len(desc_words)

        if ratio >= 0.5:
            return 0.25
        elif ratio >= 0.3:
            return 0.15

        return 0.0
