"""
Document Loader — Extracts text from uploaded files (PDF / DOCX / TXT / Markdown).
Supports chunking for embedding-based retrieval.
"""

import logging
import os
import re
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Directory for storing uploaded files and knowledge base index
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

KNOWLEDGE_INDEX_PATH = UPLOAD_DIR / "knowledge_index.json"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def _load_index() -> dict:
    """Load the knowledge base index from disk."""
    if KNOWLEDGE_INDEX_PATH.exists():
        try:
            return json.loads(KNOWLEDGE_INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return {"documents": {}}
    return {"documents": {}}


def _save_index(index: dict):
    """Persist the knowledge base index to disk."""
    KNOWLEDGE_INDEX_PATH.write_text(
        json.dumps(index, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────
# Text Extraction
# ──────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("pdfplumber not installed. Trying PyPDF2...")
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            raise ImportError(
                "Cần cài đặt pdfplumber hoặc PyPDF2 để đọc file PDF. "
                "Chạy: pip install pdfplumber"
            )


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise ImportError(
            "Cần cài đặt python-docx để đọc file DOCX. "
            "Chạy: pip install python-docx"
        )


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text or Markdown file."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return Path(file_path).read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode file: {file_path}")


def extract_text(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in (".txt", ".md", ".markdown"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")


# ──────────────────────────────────────────────
# Text Chunking
# ──────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    Split text into overlapping chunks for embedding.

    Returns list of dicts with: text, start_char, end_char, chunk_index.
    """
    if not text or not text.strip():
        return []

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Split by paragraphs first, then combine into chunks
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current_chunk = ""
    current_start = 0
    char_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            char_pos += 2
            continue

        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "start_char": current_start,
                "end_char": current_start + len(current_chunk),
                "chunk_index": len(chunks),
            })
            # Overlap: keep last portion
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap else ""
            current_chunk = overlap_text + "\n\n" + para if overlap_text else para
            current_start = char_pos - len(overlap_text) if overlap_text else char_pos
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                current_start = char_pos

        char_pos += len(para) + 2

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "start_char": current_start,
            "end_char": current_start + len(current_chunk),
            "chunk_index": len(chunks),
        })

    return chunks


# ──────────────────────────────────────────────
# Document Management
# ──────────────────────────────────────────────

def save_uploaded_file(file_storage, original_filename: str) -> dict:
    """
    Save an uploaded file and extract its text.

    Args:
        file_storage: Flask FileStorage object or file-like object
        original_filename: Original filename

    Returns:
        dict with document metadata
    """
    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Định dạng không hỗ trợ: {ext}. "
            f"Hỗ trợ: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    doc_id = str(uuid.uuid4())[:8]
    safe_name = f"{doc_id}_{Path(original_filename).stem}{ext}"
    save_path = UPLOAD_DIR / safe_name

    # Save file to disk
    file_storage.save(str(save_path))

    # Extract text
    text = extract_text(str(save_path))
    if not text.strip():
        save_path.unlink(missing_ok=True)
        raise ValueError("Không thể trích xuất nội dung từ file này.")

    # Chunk text
    chunks = chunk_text(text)

    # Build document record
    doc_record = {
        "id": doc_id,
        "original_name": original_filename,
        "saved_name": safe_name,
        "extension": ext,
        "upload_time": datetime.now(timezone.utc).isoformat(),
        "text_length": len(text),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "embeddings": [],  # Will be filled by embedding engine
        "tags": [],
        "description": "",
    }

    # Save to index
    index = _load_index()
    index["documents"][doc_id] = doc_record
    _save_index(index)

    logger.info(
        f"[KnowledgeBase] Uploaded '{original_filename}' → "
        f"{len(chunks)} chunks, {len(text)} chars"
    )

    return {
        "id": doc_id,
        "name": original_filename,
        "text_length": len(text),
        "chunk_count": len(chunks),
        "upload_time": doc_record["upload_time"],
    }


def list_documents() -> list[dict]:
    """List all uploaded documents (without full content)."""
    index = _load_index()
    docs = []
    for doc_id, doc in index["documents"].items():
        # Check for numpy vector files instead of old inline embeddings
        try:
            from core.knowledge.embedding_engine import has_vectors
            has_emb = has_vectors(doc_id)
        except Exception:
            has_emb = doc.get("has_embeddings", False)

        docs.append({
            "id": doc["id"],
            "name": doc["original_name"],
            "extension": doc["extension"],
            "text_length": doc["text_length"],
            "chunk_count": doc["chunk_count"],
            "upload_time": doc["upload_time"],
            "has_embeddings": has_emb,
            "tags": doc.get("tags", []),
            "description": doc.get("description", ""),
        })
    return sorted(docs, key=lambda d: d["upload_time"], reverse=True)


def get_document(doc_id: str) -> Optional[dict]:
    """Get a specific document with its full content."""
    index = _load_index()
    return index["documents"].get(doc_id)


def delete_document(doc_id: str) -> bool:
    """Delete a document, its file, and its vector embeddings."""
    index = _load_index()
    doc = index["documents"].get(doc_id)
    if not doc:
        return False

    # Delete file
    file_path = UPLOAD_DIR / doc["saved_name"]
    file_path.unlink(missing_ok=True)

    # Delete vector embeddings (.npy file)
    try:
        from core.knowledge.embedding_engine import delete_vectors
        delete_vectors(doc_id)
    except Exception:
        pass

    # Remove from index
    del index["documents"][doc_id]
    _save_index(index)

    logger.info(f"[KnowledgeBase] Deleted document: {doc['original_name']}")
    return True


def get_all_chunks() -> list[dict]:
    """Get all chunks from all documents with their metadata."""
    index = _load_index()
    all_chunks = []
    for doc_id, doc in index["documents"].items():
        for chunk in doc.get("chunks", []):
            all_chunks.append({
                "doc_id": doc_id,
                "doc_name": doc["original_name"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            })
    return all_chunks
