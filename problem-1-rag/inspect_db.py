from qdrant_client import QdrantClient, models
from collections import Counter

def main():
    client = QdrantClient(path="data/qdrant")
    total_count = client.count("corpus_docs").count
    
    # Calculate chunk counts per document
    counts = Counter()
    offset = None
    while True:
        res, offset = client.scroll("corpus_docs", limit=200, offset=offset, with_payload=["document_id"])
        for p in res:
            counts[p.payload.get("document_id", "Unknown")] += 1
        if offset is None:
            break

    print("=" * 65)
    print("        QDRANT EMBEDDED VECTOR DATABASE INSPECTOR")
    print(f"        Total Vectors Stored: {total_count}")
    print("=" * 65)
    print("\n[Document Breakdown]:")
    for doc, count in counts.items():
        print(f"  • {doc:<15} : {count:>4} chunks")
    print("-" * 65)

    # Show 1 sample from each document
    print("\n[Sample Chunks from Each Real Document]:")
    for doc in ["resume.md", "index.html", "NVIDIA.pdf"]:
        filt = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc))])
        pts, _ = client.scroll("corpus_docs", scroll_filter=filt, limit=1)
        if pts:
            p = pts[0]
            txt = p.payload.get("text", "").replace("\n", " ")[:130]
            print(f"\n  Document : {doc}")
            print(f"  Point ID : {p.id}")
            print(f"  Text     : {txt}...")
            print("  " + "-" * 60)

if __name__ == "__main__":
    main()
