import sys
from qdrant_client import QdrantClient, models
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

    print("=" * 70)
    print("           QDRANT EMBEDDED VECTOR DATABASE INSPECTOR")
    print(f"           Total Vectors Stored: {total_count}")
    print("=" * 70)
    print("\n[Document Breakdown]:")
    for doc, count in counts.items():
        print(f"  • {doc:<15} : {count:>4} chunks")
    print("-" * 70)

    # 1. Print Chunks from resume.md
    print("\n" + "=" * 70)
    print("  CHUNKS STORED FOR: resume.md (All Chunks)")
    print("=" * 70)
    filt_resume = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="resume.md"))])
    pts_resume, _ = client.scroll("corpus_docs", scroll_filter=filt_resume, limit=20)
    for i, p in enumerate(pts_resume, 1):
        txt = p.payload.get("text", "").replace("\n", " ").strip()
        print(f"\n[resume.md - Chunk {i}] (UUID: {p.id})")
        print(f"  \"{txt}\"")
        print("-" * 70)

    # 2. Print Chunks from index.html
    print("\n" + "=" * 70)
    print("  CHUNKS STORED FOR: index.html (Portfolio Website)")
    print("=" * 70)
    filt_html = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="index.html"))])
    pts_html, _ = client.scroll("corpus_docs", scroll_filter=filt_html, limit=5)
    for i, p in enumerate(pts_html, 1):
        txt = p.payload.get("text", "").replace("\n", " ").strip()[:200]
        print(f"\n[index.html - Chunk {i}] (UUID: {p.id})")
        print(f"  \"{txt}...\"")
        print("-" * 70)

if __name__ == "__main__":
    main()
