import time
import logging
from typing import List
from fastembed import TextEmbedding
from src.config import settings

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider:
    """Fast, local CPU embedding using ONNX-optimized BGE-small-en-v1.5 via fastembed."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading local embedding model: {self.model_name}")
        self._model = TextEmbedding(model_name=self.model_name)
        self.dim = settings.embedding_dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text chunks in batch."""
        if not texts:
            return []
        start_time = time.perf_counter()
        embeddings_gen = self._model.embed(texts)
        embeddings = [vec.tolist() for vec in embeddings_gen]
        duration = time.perf_counter() - start_time
        logger.debug(f"Embedded {len(texts)} chunks in {duration*1000:.2f}ms")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        start_time = time.perf_counter()
        embeddings_gen = self._model.embed([query])
        vec = next(embeddings_gen).tolist()
        duration = time.perf_counter() - start_time
        logger.debug(f"Embedded query in {duration*1000:.2f}ms")
        return vec

    def get_info(self) -> dict:
        return {
            "provider": "fastembed-local",
            "model": self.model_name,
            "dimension": self.dim
        }


# Singleton instance
embedding_provider = LocalEmbeddingProvider()
