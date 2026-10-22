import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data
    assert "embedding_model" in data
    assert data["embedding_model"]["model"] == "BAAI/bge-small-en-v1.5"


def test_api_query_endpoint():
    mock_rag_result = {
        "question": "What is vector search?",
        "answer": "Vector search finds nearest neighbors [Chunk 1].",
        "citations": [
            {
                "citation": "[Chunk 1]",
                "point_id": "test-id-1",
                "document_id": "doc.md",
                "chunk_index": 0,
                "score": 0.89,
                "snippet": "Vector search finds nearest neighbors..."
            }
        ],
        "chunk_count": 1,
        "is_grounded": True,
        "retrieval_latency_ms": 12.5,
        "generation_latency_ms": 250.0,
        "total_latency_ms": 262.5,
        "token_usage": {
            "prompt_tokens": 40,
            "completion_tokens": 15,
            "total_tokens": 55
        }
    }

    with patch("src.api.rag_service.query", return_value=mock_rag_result):
        response = client.post(
            "/query",
            json={"question": "What is vector search?", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is vector search?"
        assert data["answer"] == "Vector search finds nearest neighbors [Chunk 1]."
        assert data["chunk_count"] == 1
        assert len(data["citations"]) == 1
        assert data["retrieval_latency_ms"] == 12.5


def test_api_ingest_endpoint():
    mock_ingest_result = {
        "status": "success",
        "files_processed": 3,
        "chunks_processed": 42,
        "total_points_in_store": 42
    }

    with patch("src.api.ingestion_pipeline.ingest_directory", return_value=mock_ingest_result):
        response = client.post("/ingest", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["files_processed"] == 3
        assert data["chunks_processed"] == 42
        assert data["total_points_in_store"] == 42
