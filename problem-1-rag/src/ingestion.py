import os
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from pypdf import PdfReader
from bs4 import BeautifulSoup
from qdrant_client.models import PointStruct

from src.config import settings
from src.embeddings import embedding_provider
from src.store import vector_store

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF pages using pypdf."""
    reader = PdfReader(str(file_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_html(file_path: Path) -> str:
    """Extract clean text from HTML using BeautifulSoup."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
        return soup.get_text(separator="\n", strip=True)


def extract_text_from_md(file_path: Path) -> str:
    """Extract raw text from Markdown."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_document(file_path: Path) -> Dict[str, Any]:
    """Route document extraction based on extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        source_type = "pdf"
        text = extract_text_from_pdf(file_path)
    elif ext in [".html", ".htm"]:
        source_type = "html"
        text = extract_text_from_html(file_path)
    elif ext in [".md", ".markdown"]:
        source_type = "md"
        text = extract_text_from_md(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return {
        "document_id": file_path.name,
        "source_type": source_type,
        "raw_text": text
    }


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap
) -> List[str]:
    """Split text into overlapping chunks using sliding window with word boundaries."""
    if not text.strip():
        return []
    
    # Normalize whitespace
    normalized = " ".join(text.split())
    chunks = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        end = start + chunk_size
        chunk = normalized[start:end]
        
        # If not at the end of text, avoid cutting in the middle of a word
        if end < text_len:
            last_space = chunk.rfind(" ")
            if last_space > chunk_size // 2:
                chunk = chunk[:last_space]
                end = start + last_space + 1
        
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
            
        start += (chunk_size - chunk_overlap)
        if start >= text_len:
            break

    return chunks


def generate_deterministic_chunk_id(document_id: str, chunk_index: int, chunk_text: str) -> str:
    """Generate a reproducible UUIDv5 from content hash for idempotent upserts."""
    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    unique_key = f"{document_id}:{chunk_index}:{content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_key))


class IngestionPipeline:
    """Extracts, chunks, embeds, and stores documents in Qdrant idempotently."""

    def __init__(self, store=vector_store, embedder=embedding_provider):
        self.store = store
        self.embedder = embedder

    def ingest_directory(
        self,
        directory_path: str = settings.raw_data_path,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap
    ) -> Dict[str, Any]:
        """Ingest all supported files (.pdf, .html, .md) in a directory."""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            os.makedirs(dir_path, exist_ok=True)
            return {"status": "empty", "message": f"Created empty directory {dir_path}", "points_indexed": 0}

        supported_extensions = {".pdf", ".html", ".htm", ".md", ".markdown"}
        files = [p for p in dir_path.glob("*") if p.suffix.lower() in supported_extensions]

        if not files:
            logger.warning(f"No supported files found in {directory_path}")
            return {"status": "no_files", "files_found": 0, "points_indexed": 0}

        total_chunks = 0
        points_to_upsert: List[PointStruct] = []

        for file_path in files:
            doc_data = extract_document(file_path)
            chunks = chunk_text(doc_data["raw_text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                continue

            logger.info(f"Processing {file_path.name}: extracted {len(chunks)} chunks")
            vectors = self.embedder.embed_documents(chunks)

            now_iso = datetime.now(timezone.utc).isoformat()
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                point_id = generate_deterministic_chunk_id(doc_data["document_id"], idx, chunk)
                payload = {
                    "text": chunk,
                    "document_id": doc_data["document_id"],
                    "source_type": doc_data["source_type"],
                    "chunk_index": idx,
                    "created_at": now_iso
                }
                points_to_upsert.append(PointStruct(id=point_id, vector=vector, payload=payload))

            total_chunks += len(chunks)

        # Upsert into Qdrant idempotently
        self.store.upsert_points(points_to_upsert)
        current_count = self.store.count()

        return {
            "status": "success",
            "files_processed": len(files),
            "chunks_processed": total_chunks,
            "total_points_in_store": current_count
        }


ingestion_pipeline = IngestionPipeline()


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("      SYNLABS RAG: IDEMPOTENT CORPUS INGESTION PIPELINE")
    print("=" * 70)

    try:
        initial_count = vector_store.count()
        print(f">> Initial points in Qdrant store: {initial_count}")
        print(f">> Scanning corpus directory     : {settings.raw_data_path}")

        result = ingestion_pipeline.ingest_directory()

        final_count = result["total_points_in_store"]
        new_points = final_count - initial_count

        print(f">> Files successfully processed  : {result['files_processed']}")
        print(f">> Total chunks created          : {result['chunks_processed']}")
        print(f">> New vectors added             : {new_points}")
        status_msg = "(UNCHANGED - IDEMPOTENT)" if new_points == 0 else "(NEW EMBEDDINGS STORED)"
        print(f">> Total points in vector store  : {final_count} {status_msg}")
        print("=" * 70 + "\n")
    except RuntimeError as re:
        if "already accessed" in str(re):
            print("\n>> [NOTICE]: The Qdrant database file is currently LOCKED by your running")
            print(">>           'uvicorn src.api:app' dev server in your other terminal.")
            print(">> ")
            print(">> TO FIX:")
            print(">> Option A: Press Ctrl+C in your uvicorn terminal to stop the server,")
            print(">>           then re-run: uv run python -m src.ingestion")
            print(">> Option B: Trigger ingestion via the running server using:")
            print(">>           curl.exe -X POST http://127.0.0.1:8000/ingest\n")
            print("=" * 70 + "\n")
        else:
            raise
