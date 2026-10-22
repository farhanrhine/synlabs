import pytest
from pathlib import Path
from qdrant_client.models import PointStruct
from src.store import QdrantStore


def test_qdrant_store_init_and_idempotent_upsert(tmp_path: Path):
    test_db_path = str(tmp_path / "test_qdrant")
    store = QdrantStore(storage_path=test_db_path, collection_name="test_collection")

    assert store.count() == 0

    dummy_vector = [0.1] * 384
    point = PointStruct(
        id="c56a4180-65aa-42ec-a945-5fd21dec0538",
        vector=dummy_vector,
        payload={"text": "hello test", "source_type": "md", "document_id": "doc1.md"}
    )

    # First upsert
    upserted_count = store.upsert_points([point])
    assert upserted_count == 1
    assert store.count() == 1

    # Second upsert with identical ID (idempotency test)
    upserted_count_2 = store.upsert_points([point])
    assert upserted_count_2 == 1
    # Total count must remain unchanged!
    assert store.count() == 1


def test_qdrant_store_search_and_filter(tmp_path: Path):
    test_db_path = str(tmp_path / "test_qdrant_search")
    store = QdrantStore(storage_path=test_db_path, collection_name="search_collection")

    p1 = PointStruct(
        id="a1111111-1111-1111-1111-111111111111",
        vector=[1.0] + [0.0] * 383,
        payload={"text": "PDF chunk", "source_type": "pdf"}
    )
    p2 = PointStruct(
        id="a2222222-2222-2222-2222-222222222222",
        vector=[0.9] + [0.1] + [0.0] * 382,
        payload={"text": "HTML chunk", "source_type": "html"}
    )
    store.upsert_points([p1, p2])
    assert store.count() == 2

    # Search with filter
    results = store.search(
        query_vector=[1.0] + [0.0] * 383,
        limit=5,
        filters={"source_type": "pdf"}
    )
    assert len(results) == 1
    assert results[0]["payload"]["source_type"] == "pdf"
    assert results[0]["id"] == "a1111111-1111-1111-1111-111111111111"
