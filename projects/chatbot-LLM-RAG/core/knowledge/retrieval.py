"""
Knowledge Retrieval — Intent-aware RAG pipeline with LOCAL embeddings.

Upload (FREE):
    1. Extract text from PDF/DOCX/TXT/MD
    2. Chunk text (500-800 chars)
    3. Embed locally (sentence-transformers all-MiniLM-L6-v2)
    4. Save vectors as .npy files

When click title (minimal OpenAI cost):
    1. Embed title/keyword locally (free)
    2. Semantic search in stored vectors (free)
    3. Return top 3 relevant chunks
    4. Only these 3 chunks + title sent to OpenAI for heading generation
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Core vector search (LOCAL — free)
# ──────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.25,
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Semantic search across all documents using LOCAL embeddings.
    100% free — no API calls.

    Args:
        query:          Search text
        top_k:          Max chunks to return
        min_similarity: Minimum cosine similarity threshold
        doc_ids:        Optional filter by specific document IDs

    Returns:
        List of dicts: doc_id, doc_name, text, similarity, chunk_index
    """
    from core.knowledge.document_loader import _load_index
    from core.knowledge.embedding_engine import (
        create_single_embedding,
        load_vectors,
        cosine_similarity_batch,
    )

    index = _load_index()
    documents = index.get("documents", {})
    if not documents:
        return []

    if doc_ids:
        documents = {k: v for k, v in documents.items() if k in doc_ids}

    # Create query embedding locally (free)
    try:
        query_vec = create_single_embedding(query)
    except Exception as e:
        logger.error(f"[Retrieval] Failed to embed query: {e}")
        return []

    results = []
    for doc_id, doc in documents.items():
        # Load vectors from .npy file
        vectors = load_vectors(doc_id)
        if vectors is None or len(vectors) == 0:
            continue

        chunks = doc.get("chunks", [])
        if len(chunks) != len(vectors):
            logger.warning(
                f"[Retrieval] Chunk/vector mismatch for '{doc['original_name']}': "
                f"{len(chunks)} chunks vs {len(vectors)} vectors"
            )
            continue

        # Batch cosine similarity (fast numpy dot product)
        similarities = cosine_similarity_batch(query_vec, vectors)

        for i, sim in enumerate(similarities):
            if sim >= min_similarity:
                results.append({
                    "doc_id": doc_id,
                    "doc_name": doc["original_name"],
                    "chunk_index": chunks[i].get("chunk_index", i),
                    "text": chunks[i]["text"],
                    "similarity": round(float(sim), 4),
                })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# ──────────────────────────────────────────────
# Simple context builder
# ──────────────────────────────────────────────

def build_knowledge_context(
    query: str,
    top_k: int = 3,
    max_tokens_approx: int = 2000,
) -> str:
    """
    Simple context builder — retrieve top_k chunks and format for prompt.
    Embedding is FREE (local). Only the final LLM call costs tokens.
    """
    chunks = retrieve_relevant_chunks(query, top_k=top_k)
    if not chunks:
        return ""
    return _format_flat_chunks(chunks, max_tokens_approx)


# ──────────────────────────────────────────────
# RAG context builder (for heading generator)
# ──────────────────────────────────────────────

def build_rag_context(
    keyword: str,
    title: str,
    intent_type: str = "informational",
    max_tokens_approx: int = 2000,
) -> tuple[str, dict]:
    """
    RAG context builder — search with multiple queries, return top 3 chunks.

    Flow:
        1. Build 2-3 search queries from keyword + title
        2. Local semantic search (free)
        3. De-duplicate & re-rank
        4. Return formatted context + metadata

    Returns:
        (context_string, rag_metadata)
    """
    if not has_knowledge_base():
        return "", {"has_knowledge": False}

    from core.knowledge.embedding_engine import create_single_embedding, load_vectors, cosine_similarity_batch
    from core.knowledge.document_loader import _load_index

    index = _load_index()
    documents = index.get("documents", {})

    # Build search queries (2-3 queries, all embedded locally for free)
    queries = [
        {"label": "title", "text": title},
        {"label": "keyword", "text": keyword},
    ]
    # Add a combined query for better coverage
    if keyword.lower() not in title.lower():
        queries.append({"label": "combined", "text": f"{keyword} {title}"})

    logger.info(f"[RAG] Running {len(queries)} search queries (LOCAL embedding)")

    # Collect all candidate chunks with scores
    seen_keys: set[str] = set()
    all_candidates: list[dict] = []

    for q_info in queries:
        try:
            query_vec = create_single_embedding(q_info["text"])
        except Exception as e:
            logger.error(f"[RAG] Embed query failed: {e}")
            continue

        for doc_id, doc in documents.items():
            vectors = load_vectors(doc_id)
            if vectors is None or len(vectors) == 0:
                continue

            chunks = doc.get("chunks", [])
            if len(chunks) != len(vectors):
                continue

            sims = cosine_similarity_batch(query_vec, vectors)

            for i, sim in enumerate(sims):
                if sim < 0.25:
                    continue
                key = f"{doc_id}:{i}"
                if key in seen_keys:
                    # Boost score if matched by multiple queries
                    for c in all_candidates:
                        if c["_key"] == key:
                            c["similarity"] = max(c["similarity"], float(sim))
                            c["matched_queries"] += 1
                            break
                    continue

                seen_keys.add(key)
                all_candidates.append({
                    "_key": key,
                    "doc_id": doc_id,
                    "doc_name": doc["original_name"],
                    "chunk_index": chunks[i].get("chunk_index", i),
                    "text": chunks[i]["text"],
                    "similarity": float(sim),
                    "matched_queries": 1,
                })

    # Re-rank: boost chunks matched by multiple queries
    for c in all_candidates:
        c["final_score"] = c["similarity"] * (1 + 0.15 * (c["matched_queries"] - 1))

    all_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Take top 3 only → minimize OpenAI tokens
    top_chunks = all_candidates[:3]

    if not top_chunks:
        return "", {
            "has_knowledge": True,
            "chunks_found": 0,
            "queries_executed": len(queries),
        }

    # Format context
    context = _format_rag_chunks(top_chunks, max_tokens_approx)

    docs_used = set(c["doc_id"] for c in top_chunks)
    metadata = {
        "has_knowledge": True,
        "chunks_found": len(top_chunks),
        "documents_used": len(docs_used),
        "queries_executed": len(queries),
        "categories_found": len(set(c["doc_name"] for c in top_chunks)),
        "categories": [c["doc_name"] for c in top_chunks],
        "queries_used": [q["text"] for q in queries],
    }

    logger.info(
        f"[RAG] Found {len(top_chunks)} relevant chunks "
        f"from {len(docs_used)} documents (LOCAL, FREE)"
    )
    return context, metadata


def _format_rag_chunks(chunks: list[dict], max_tokens_approx: int) -> str:
    """Format top chunks for prompt injection."""
    max_chars = max_tokens_approx * 4
    total_chars = 0

    parts = ["═══ BRAND KNOWLEDGE (từ tài liệu nội bộ) ═══"]

    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"].strip()
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                text = text[:remaining] + "..."
            else:
                break

        parts.append(
            f"[Đoạn {i} | {chunk['doc_name']} | Relevance: {chunk['similarity']:.2f}]\n{text}"
        )
        total_chars += len(text)

    parts.append("═══ END KNOWLEDGE ═══")
    return "\n\n".join(parts)


def _format_flat_chunks(chunks: list[dict], max_tokens_approx: int) -> str:
    """Simple flat formatting."""
    max_chars = max_tokens_approx * 4
    total_chars = 0
    context_parts = []

    for chunk in chunks:
        text = chunk["text"].strip()
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 200:
                text = text[:remaining] + "..."
                context_parts.append(
                    f"[Từ: {chunk['doc_name']} | Relevance: {chunk['similarity']:.2f}]\n{text}"
                )
            break
        context_parts.append(
            f"[Từ: {chunk['doc_name']} | Relevance: {chunk['similarity']:.2f}]\n{text}"
        )
        total_chars += len(text)

    if not context_parts:
        return ""
    return (
        "═══ BRAND KNOWLEDGE (từ tài liệu nội bộ) ═══\n\n"
        + "\n\n---\n\n".join(context_parts)
        + "\n\n═══ END KNOWLEDGE ═══"
    )


# ──────────────────────────────────────────────
# Product Solution Context (for heading generator)
# ──────────────────────────────────────────────

def build_product_solution_context(
    keyword: str,
    title: str,
    intent_type: str = "informational",
    relevance_threshold: float = 0.35,
    max_tokens_approx: int = 1500,
) -> tuple[str, dict]:
    """
    Retrieve product-specific context to determine if the article topic
    is relevant enough to mention products as a solution.

    Uses intent-aware queries focused on product features, use cases,
    and benefits. Returns formatted product context + relevance metadata.

    Args:
        keyword:              Target keyword
        title:                Article title
        intent_type:          Search intent type
        relevance_threshold:  Minimum similarity to consider product relevant
        max_tokens_approx:    Max tokens for context

    Returns:
        (product_context_string, product_metadata)
        product_metadata includes:
            - is_product_relevant: bool — should we mention product?
            - relevance_score: float — how relevant the product is
            - product_names: list — product names found
            - product_features: list — key features extracted
            - recommended_section_type: str — how to mention product
    """
    if not has_knowledge_base():
        return "", {"is_product_relevant": False}

    from core.knowledge.embedding_engine import (
        create_single_embedding,
        load_vectors,
        cosine_similarity_batch,
    )
    from core.knowledge.document_loader import _load_index

    index = _load_index()
    documents = index.get("documents", {})

    # Build product-focused search queries
    product_queries = [
        {"label": "product_match", "text": f"{keyword} giải pháp sản phẩm"},
        {"label": "use_case", "text": f"{keyword} ứng dụng thực tế"},
        {"label": "feature_match", "text": f"{title} tính năng lợi ích"},
    ]

    # For commercial/comparison intents, add direct product query
    if intent_type in ("commercial", "comparison", "transactional"):
        product_queries.append(
            {"label": "direct", "text": f"{keyword} mua sản phẩm so sánh"}
        )

    logger.info(f"[ProductRAG] Running {len(product_queries)} product-focused queries")

    seen_keys: set[str] = set()
    all_candidates: list[dict] = []

    for q_info in product_queries:
        try:
            query_vec = create_single_embedding(q_info["text"])
        except Exception as e:
            logger.error(f"[ProductRAG] Embed failed: {e}")
            continue

        for doc_id, doc in documents.items():
            vectors = load_vectors(doc_id)
            if vectors is None or len(vectors) == 0:
                continue

            chunks = doc.get("chunks", [])
            if len(chunks) != len(vectors):
                continue

            sims = cosine_similarity_batch(query_vec, vectors)

            for i, sim in enumerate(sims):
                if sim < 0.20:  # Lower initial threshold, filter later
                    continue
                key = f"{doc_id}:{i}"
                if key in seen_keys:
                    for c in all_candidates:
                        if c["_key"] == key:
                            c["similarity"] = max(c["similarity"], float(sim))
                            c["matched_queries"] += 1
                            break
                    continue

                seen_keys.add(key)
                all_candidates.append({
                    "_key": key,
                    "doc_id": doc_id,
                    "doc_name": doc["original_name"],
                    "chunk_index": chunks[i].get("chunk_index", i),
                    "text": chunks[i]["text"],
                    "similarity": float(sim),
                    "matched_queries": 1,
                })

    # Re-rank with multi-query boost
    for c in all_candidates:
        c["final_score"] = c["similarity"] * (1 + 0.2 * (c["matched_queries"] - 1))

    all_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Calculate relevance score (average of top chunks)
    top_chunks = all_candidates[:5]
    if not top_chunks:
        return "", {"is_product_relevant": False, "relevance_score": 0.0}

    avg_score = sum(c["final_score"] for c in top_chunks) / len(top_chunks)
    max_score = top_chunks[0]["final_score"]

    is_relevant = max_score >= relevance_threshold

    # Extract product names from chunks
    product_names = set()
    for c in top_chunks:
        text_lower = c["text"].lower()
        # Extract product names from document names and text
        doc_name = c["doc_name"].lower()
        if "giới thiệu" in doc_name:
            # Extract product name from doc title pattern
            name_part = c["doc_name"].replace("GIỚI THIỆU VỀ", "").replace(".pdf", "").strip()
            if name_part:
                product_names.add(name_part)

    # Determine recommended section type based on intent
    if intent_type == "transactional":
        section_type = "product_recommendation"
    elif intent_type == "commercial":
        section_type = "product_comparison"
    elif intent_type == "comparison":
        section_type = "product_as_alternative"
    else:
        section_type = "product_as_solution"

    # Build product context (take top 3 most relevant)
    context_chunks = top_chunks[:3]
    product_context = _format_product_chunks(context_chunks, max_tokens_approx)

    metadata = {
        "is_product_relevant": is_relevant,
        "relevance_score": round(max_score, 4),
        "avg_relevance": round(avg_score, 4),
        "product_names": list(product_names),
        "chunks_found": len(context_chunks),
        "recommended_section_type": section_type,
        "documents_used": len(set(c["doc_id"] for c in context_chunks)),
    }

    logger.info(
        f"[ProductRAG] Relevance: {max_score:.3f} (threshold: {relevance_threshold}) "
        f"→ {'RELEVANT' if is_relevant else 'NOT relevant'} | "
        f"Products: {list(product_names)}"
    )

    return product_context, metadata


def _format_product_chunks(chunks: list[dict], max_tokens_approx: int) -> str:
    """Format product-specific chunks for prompt injection."""
    max_chars = max_tokens_approx * 4
    total_chars = 0

    parts = ["═══ THÔNG TIN SẢN PHẨM (từ tài liệu nội bộ) ═══"]

    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"].strip()
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                text = text[:remaining] + "..."
            else:
                break

        parts.append(
            f"[Sản phẩm {i} | {chunk['doc_name']} | Relevance: {chunk['similarity']:.2f}]\n{text}"
        )
        total_chars += len(text)

    parts.append("═══ END PRODUCT INFO ═══")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def has_knowledge_base() -> bool:
    """Check if there are any documents with embeddings."""
    from core.knowledge.document_loader import _load_index
    from core.knowledge.embedding_engine import has_vectors

    index = _load_index()
    for doc_id in index.get("documents", {}):
        if has_vectors(doc_id):
            return True
    return False
