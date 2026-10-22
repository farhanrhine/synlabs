from qdrant_client import QdrantClient

def main():
    client = QdrantClient(path="data/qdrant")
    total_count = client.count("corpus_docs").count
    print("=" * 60)
    print(f"  QDRANT VECTOR DATABASE INSPECTOR")
    print(f"  Total Vectors Stored: {total_count}")
    print("=" * 60)

    # Fetch sample points
    points, _ = client.scroll("corpus_docs", limit=5)
    for i, p in enumerate(points, 1):
        doc_id = p.payload.get("document_id", "Unknown")
        text = p.payload.get("text", "")[:150].replace("\n", " ")
        print(f"\n[Sample {i}]")
        print(f"  Point UUID : {p.id}")
        print(f"  Document   : {doc_id}")
        print(f"  Text Chunk : {text}...")
        print("-" * 60)

if __name__ == "__main__":
    main()
