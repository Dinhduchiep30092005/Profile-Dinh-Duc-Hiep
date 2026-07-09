"""
Embedding Engine — LOCAL embeddings using sentence-transformers.

Model: all-MiniLM-L6-v2 (384 dimensions)
   → Runs 100% on local machine
   → No API calls, no cost
   → Fast (~50ms per embedding on CPU)

Vectors stored as .npy files alongside the knowledge_index.json.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Model config
# ──────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Store vectors on disk
VECTORS_DIR = Path("uploads/vectors")
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded model singleton
_model = None


def _get_model():
    """Lazy-load the sentence-transformers model (first call downloads ~80MB)."""
    global _model
    if _model is None:
        logger.info(f"[Embedding] Loading model '{MODEL_NAME}' (first time may download ~80MB)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        logger.info(f"[Embedding] Model loaded: {MODEL_NAME} ({EMBEDDING_DIMENSIONS}d)")
    return _model


# ──────────────────────────────────────────────
# Embedding creation (LOCAL — free)
# ──────────────────────────────────────────────

def create_embeddings(texts: list[str]) -> np.ndarray:
    """
    Create embeddings for a list of texts using local model.

    Args:
        texts: List of text strings to embed

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIMENSIONS)
    """
    if not texts:
        return np.array([])

    model = _get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=len(texts) > 20,
        normalize_embeddings=True,  # L2 normalized → cosine sim = dot product
        batch_size=64,
    )
    logger.info(f"[Embedding] Created {len(embeddings)} embeddings locally")
    return np.array(embeddings, dtype=np.float32)


def create_single_embedding(text: str) -> np.ndarray:
    """Create embedding for a single text. Returns 1D array."""
    model = _get_model()
    embedding = model.encode(
        [text],
        normalize_embeddings=True,
    )
    return np.array(embedding[0], dtype=np.float32)


# ──────────────────────────────────────────────
# Vector storage — NumPy .npy files
# ──────────────────────────────────────────────

def _vectors_path(doc_id: str) -> Path:
    """Path to a document's stored vectors."""
    return VECTORS_DIR / f"{doc_id}.npy"


def save_vectors(doc_id: str, vectors: np.ndarray):
    """Save embedding vectors for a document to disk."""
    path = _vectors_path(doc_id)
    np.save(str(path), vectors)
    logger.info(f"[Embedding] Saved {len(vectors)} vectors for doc '{doc_id}'")


def load_vectors(doc_id: str) -> Optional[np.ndarray]:
    """Load embedding vectors for a document from disk."""
    path = _vectors_path(doc_id)
    if not path.exists():
        return None
    return np.load(str(path))


def delete_vectors(doc_id: str):
    """Delete stored vectors for a document."""
    path = _vectors_path(doc_id)
    path.unlink(missing_ok=True)


def has_vectors(doc_id: str) -> bool:
    """Check if a document has stored vectors."""
    return _vectors_path(doc_id).exists()


# ──────────────────────────────────────────────
# Similarity search
# ──────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Fast with numpy."""
    return float(np.dot(vec_a, vec_b))


def cosine_similarity_batch(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between a query vector and a matrix of vectors.
    All vectors assumed L2-normalized (dot product = cosine sim).

    Args:
        query_vec: shape (384,)
        matrix: shape (N, 384)

    Returns:
        shape (N,) similarity scores
    """
    return matrix @ query_vec


# ──────────────────────────────────────────────
# Document embedding pipeline
# ──────────────────────────────────────────────

def embed_document_chunks(doc_id: str) -> int:
    """
    Create LOCAL embeddings for all chunks of a document and save to disk.
    FREE — no API calls.

    Args:
        doc_id: Document ID

    Returns:
        Number of embeddings created
    """
    from core.knowledge.document_loader import _load_index, _save_index

    index = _load_index()
    doc = index["documents"].get(doc_id)
    if not doc:
        raise ValueError(f"Document not found: {doc_id}")

    chunks = doc.get("chunks", [])
    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]

    # Create embeddings LOCALLY (free!)
    vectors = create_embeddings(texts)

    # Save vectors to disk as .npy
    save_vectors(doc_id, vectors)

    # Update index to mark that embeddings exist
    doc["has_embeddings"] = True
    doc["embedding_count"] = len(vectors)
    doc["embedding_model"] = MODEL_NAME
    doc["embedding_dimensions"] = EMBEDDING_DIMENSIONS
    # Remove old OpenAI embeddings if they existed
    doc.pop("embeddings", None)
    _save_index(index)

    logger.info(
        f"[Embedding] Embedded {len(vectors)} chunks for '{doc['original_name']}' "
        f"using {MODEL_NAME} (LOCAL, FREE)"
    )
    return len(vectors)
