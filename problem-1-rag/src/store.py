import os
import logging
import atexit
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint
)
from src.config import settings

logger = logging.getLogger(__name__)


class QdrantStore:
    """Manages the embedded local Qdrant vector database."""

    def __init__(self, storage_path: Optional[str] = None, collection_name: Optional[str] = None):
        self.storage_path = storage_path or settings.qdrant_storage_path
        self.collection_name = collection_name or settings.collection_name
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"Connecting to embedded Qdrant at: {self.storage_path}")
        self.client = QdrantClient(path=self.storage_path)
        self._ensure_collection()
        atexit.register(self.close)

    def close(self) -> None:
        """Safely close client connection on exit."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def _ensure_collection(self) -> None:
        """Create collection if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={settings.embedding_dim}, metric=Cosine)")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    def upsert_points(self, points: List[PointStruct]) -> int:
        """Idempotently insert or update points in Qdrant."""
        if not points:
            return 0
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        return len(points)

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search top-k nearest neighbors with optional metadata payload filtering."""
        query_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items() if v is not None
            ]
            if conditions:
                query_filter = Filter(must=conditions)

        # In qdrant-client >= 1.10, query_points is the preferred search method
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True
        ).points

        formatted = []
        for point in results:
            formatted.append({
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload or {}
            })
        return formatted

    def count(self) -> int:
        """Return total number of points in collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count


# Singleton instance
vector_store = QdrantStore()
