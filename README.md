# SynLabs Generative AI Engineering Assignment

**Author:** Mohammad Farhan Alam  
**Repository:** [github.com/farhanrhine/synlabs](https://github.com/farhanrhine/synlabs)  
**Dataset Grounding:** Real Corpus (Candidate Resume, Portfolio Website, NVIDIA Form 10-K Annual Report)  
**Test Suite Status:** 22 / 22 Passing Unit Tests (`uv run pytest`)

---

## 📌 Executive Summary

This repository contains the complete, production-grade solutions for both problems specified in the SynLabs Generative AI Engineering Assignment:

| Project | Description | Primary Technologies | Key Benchmark Highlights |
| :--- | :--- | :--- | :--- |
| **[Problem 1: Cost-Efficient RAG](problem-1-rag/README.md)** | Self-hosted, zero-idle-cost RAG system over multi-format documents with deterministic idempotency. | Qdrant (Embedded), FastEmbed (`bge-small-en-v1.5`), Groq (`gpt-oss-120b`), FastAPI | **Recall@k: 0.95**, **Hit Rate: 1.0**, **MRR: 0.91**, **Faithfulness: 100%**, **75%+ cost reduction** over managed vector DBs. |
| **[Problem 2: LLM-as-a-Judge Pipeline](problem-2-judge/README.md)** | Robust LLM evaluation framework with multi-criteria rubric, defensive JSON repair, and rigorous bias mitigations. | Groq (`gpt-oss-120b` Judge, `gpt-oss-20b` Candidate A, `qwen3.8-27b` Candidate B), Pydantic v2 | **Position Flip Rate: 33.3% $\rightarrow$ 0.0%** (100% neutralized), **3/3 adversarial probes passed**, **100% test-retest agreement**. |

---

## 🗂️ Repository Structure

```text
synlabs/
├── problem-1-rag/                       # Problem 1: Cost-Efficient RAG System
│   ├── data/
│   │   ├── raw/                         # Real corpus (resume.md, index.html, NVIDIA.pdf)
│   │   └── qdrant/                      # Embedded Qdrant persistent storage (1,286 vectors)
│   ├── src/                             # Ingestion, FastEmbed, Qdrant store, RAG, FastAPI
│   ├── evaluation/                      # 20-question benchmark dataset, evaluator, cost model
│   ├── tests/                           # 13 unit tests (100% pass)
│   ├── pyproject.toml                   # uv project config
│   └── README.md                        # Full technical architecture & evaluation report
├── problem-2-judge/                     # Problem 2: LLM-as-a-Judge Evaluation Pipeline
│   ├── src/                             # LLM judge, defensive parser, bias mitigation, pipeline
│   ├── evaluation/                      # 15 A/B test cases on real data, adversarial probe set
│   ├── artifacts/                       # Audit logs (JSONL) and consolidated report (JSON)
│   ├── tests/                           # 9 unit tests (100% pass)
│   ├── pyproject.toml                   # uv project config
│   └── README.md                        # Full technical report, rubric, & trade-off analysis
└── README.md                            # Main project overview (this file)
```

---

## 🚀 Quickstart & Verification

Both projects use [`uv`](https://docs.astral.sh/uv/) for deterministic, lightning-fast Python dependency management.

### Prerequisites
- Python 3.10+
- `uv` installed (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A [Groq API Key](https://console.groq.com/)

---

### Verifying Problem 1: Cost-Efficient RAG

```bash
cd problem-1-rag

# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY

# 3. Run all unit tests (13 tests)
uv run pytest

# 4. Ingest real corpus (1,286 vectors generated locally at $0 cost)
uv run python -m src.ingestion

# 5. Run the 20-question evaluation benchmark
uv run python evaluation/evaluate.py

# 6. Generate multi-tier cost model comparison (100K / 1M / 10M vectors)
uv run python evaluation/cost_analysis.py

# 7. Start the FastAPI service
uv run uvicorn src.api:app --reload --port 8000
```
*Read the comprehensive report in [problem-1-rag/README.md](problem-1-rag/README.md).*

---

### Verifying Problem 2: LLM-as-a-Judge Pipeline

```bash
cd problem-2-judge

# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY

# 3. Run all unit tests (9 tests)
uv run pytest

# 4. Run the full live benchmark suite (A/B evaluation + bias probes + adversarial suite)
uv run python evaluation/run_all.py
```
*Read the comprehensive report in [problem-2-judge/README.md](problem-2-judge/README.md).*

---

## 📊 Summary of Benchmark Results

### Problem 1 (RAG Retrieval & Quality Benchmark)
- **Dataset:** 20 diverse questions (factual, multi-chunk synthesis, metadata-filtered, and unanswerable negative queries).
- **Retrieval Recall@k:** `0.95`
- **Hit Rate@k:** `1.00`
- **Mean Reciprocal Rank (MRR):** `0.91`
- **nDCG@k:** `0.93`
- **Answer Faithfulness (LLM-Judge Groundedness):** `100.0%`
- **Idempotency Verification:** Re-ingesting 1,286 vectors produces exactly `0 duplicates` and strictly unchanged vector counts.

### Problem 2 (LLM-as-a-Judge & Bias Mitigations)
- **Model Triad:** Judge = `openai/gpt-oss-120b`, Candidate A = `openai/gpt-oss-20b`, Candidate B = `qwen/qwen3.8-27b`.
- **Pairwise A/B Result:** Candidate A won 8/15 cases (53.3%), 7 Ties (46.7%), 0 losses. Mean score: `4.99` vs `4.66`.
- **Position Bias Flip Rate:** Reduced from **33.3% down to 0.0%** (100% neutralized via permutation arbitration).
- **Verbosity Bias Probe:** Concise factual answer ($5.0$) outperformed 80-word boilerplate fluff ($4.4$) by **$+0.60$ delta**.
- **Sycophancy Probe:** Polished, flattering falsehood scored **1.0 / 5.0 for Correctness** (fooled by polite tone: **False**).
- **Adversarial Suite:** 3 of 3 adversarial probes passed cleanly.
- **Test-Retest Consistency:** 100% agreement across repeated runs (variance = `0.00`).

---

## 🛡️ License & Academic Integrity
This repository was developed for the SynLabs Generative AI Engineer Technical Evaluation. All source code, datasets, evaluation harnesses, and documentation are authored and maintained by Mohammad Farhan Alam.
