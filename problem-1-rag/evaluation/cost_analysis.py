"""
Cost Analysis Model across Scale: 100K, 1M, and 10M vectors
Compares Self-Hosted/Embedded Qdrant vs Managed Vector Database (Pinecone / Weaviate Cloud).
Generates the exact cost comparison table required for Section 1.3.3 of the Submission Template.
"""

import json
from pathlib import Path


def generate_cost_analysis() -> dict:
    assumptions = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "vector_dimension": 384,
        "precision": "float32 (1,536 bytes/vector) with scalar quantization option to int8",
        "hnsw_index_overhead": "1.5x vector size in memory/disk",
        "metadata_storage_per_doc": "500 bytes (chunk text snippet + source metadata)",
        "query_volume_assumed": "500,000 queries/month",
        "region": "us-east-1 / EU-central",
        "managed_db_reference": "Pinecone Standard Pod (s1/p1 instances) and Serverless compute"
    }

    scale_tiers = [
        {
            "scale": "100K vectors",
            "vectors_count": 100_000,
            "raw_vector_size_mb": 150,
            "total_ram_disk_mb": 350,
            "qdrant_hardware": "Embedded on application server or AWS t4g.small (2GB RAM, 2 vCPU)",
            "qdrant_monthly_cost": 8.00,
            "managed_db_tier": "Pinecone Standard (minimum 1 pod tier)",
            "managed_db_monthly_cost": 70.00,
            "monthly_savings_dollars": 62.00,
            "savings_percentage": "88.6%",
            "notes": "At 100K vectors, a managed DB charges a heavy fixed pod tax for idle time. Embedded Qdrant runs inside app process for $0 incremental cost."
        },
        {
            "scale": "1M vectors",
            "vectors_count": 1_000_000,
            "raw_vector_size_mb": 1536,
            "total_ram_disk_mb": 3500,
            "qdrant_hardware": "Dedicated VPS / AWS t4g.medium (4GB RAM, 20GB NVMe SSD)",
            "qdrant_monthly_cost": 24.00,
            "managed_db_tier": "Pinecone Standard (1-2 pods with high-availability replication)",
            "managed_db_monthly_cost": 140.00,
            "monthly_savings_dollars": 116.00,
            "savings_percentage": "82.9%",
            "notes": "Qdrant uses memory-mapped (mmap) storage on NVMe SSD, keeping active vectors hot in cache without requiring all vectors in expensive RAM."
        },
        {
            "scale": "10M vectors",
            "vectors_count": 10_000_000,
            "raw_vector_size_mb": 15360,
            "total_ram_disk_mb": 35000,
            "qdrant_hardware": "Cloud instance e.g. AWS c6g.xlarge / Hetzner AX41 (16GB RAM, 64GB NVMe)",
            "qdrant_monthly_cost": 75.00,
            "managed_db_tier": "Pinecone 4x-8x pod cluster (s1/p1 tiers for 10M 384-dim vectors)",
            "managed_db_monthly_cost": 560.00,
            "monthly_savings_dollars": 485.00,
            "savings_percentage": "86.6%",
            "notes": "Enabling Qdrant's Scalar Quantization (SQ) reduces RAM footprint by 75% with <1% recall degradation, avoiding massive cloud pod bills."
        }
    ]

    report = {
        "assumptions": assumptions,
        "comparison_table": scale_tiers
    }

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/cost_comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Table
    print("\n" + "="*85)
    print("      PROBLEM 1: MONTHLY COST COMPARISON ACROSS SCALE (QDRANT VS MANAGED DB)")
    print("="*85)
    print(f"{'Vectors':<12} | {'Qdrant ($/mo)':<15} | {'Managed DB ($/mo)':<20} | {'Savings / Notes'}")
    print("-" * 85)
    for tier in scale_tiers:
        savings = f"${tier['monthly_savings_dollars']:.0f}/mo ({tier['savings_percentage']})"
        print(f"{tier['scale']:<12} | ${tier['qdrant_monthly_cost']:<14.2f} | ${tier['managed_db_monthly_cost']:<19.2f} | {savings}")
    print("="*85)
    print("\nDetailed assumptions and breakdown saved to artifacts/cost_comparison.json\n")
    return report


if __name__ == "__main__":
    generate_cost_analysis()
