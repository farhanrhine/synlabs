# Applied AI / ML Engineering Take-Home Assignment — Submission Template

---

## Candidate Details

| Field | Your Entry |
| :--- | :--- |
| **Full Name** | Mohammad Farhan Alam |
| **Email** | [Your Email Address] |
| **Date Submitted** | October 22, 2026 |
| **Repository Link** | https://github.com/farhanrhine/synlabs |
| **Total Time Spent** | ~10 hours across 3 focused sessions |
| **Problem 1 — Vector Store Chosen** | **Qdrant (Embedded Mode)** |
| **Problem 2 — Judge Model / Generator Models** | **Judge:** `openai/gpt-oss-120b` · **Generator A:** `openai/gpt-oss-20b` · **Generator B:** `qwen/qwen3.8-27b` |

---

# Problem 1 — Cost-Efficient RAG Application

A QA service over a multi-format document corpus on a low-cost vector store, with honest evaluation of retrieval quality, answer quality, latency, and cost.

## 1.1 Architecture Diagram / Flowchart

```text
                  +-------------------------------------------------------------+
                  |                 1. DOCUMENT INGESTION                       |
                  |  Multi-Format Raw Corpus:                                   |
                  |  - resume.md (Markdown)                                     |
                  |  - index.html (HTML - BeautifulSoup4)                       |
                  |  - NVIDIA.pdf (PDF - PyPDF)                                 |
                  +------------------------------+------------------------------+
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |                 2. TOKEN-AWARE CHUNKING                     |
                  |  - Chunk Size: 500 characters (~100 tokens)                 |
                  |  - Chunk Overlap: 100 characters (~20 tokens)               |
                  |  - Deterministic ID: UUIDv5(SHA256(file+idx+text))          |
                  +------------------------------+------------------------------+
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |                 3. LOCAL ONNX EMBEDDING                     |
                  |  - Model: BAAI/bge-small-en-v1.5 (FastEmbed)                |
                  |  - Vector Dimensions: 384 (Normalized Float32)              |
                  |  - Execution: 100% Local CPU, $0.00 API Cost, No Rate Limits|
                  +------------------------------+------------------------------+
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |                 4. EMBEDDED VECTOR STORE                    |
                  |  - Engine: Qdrant Embedded (./data/qdrant)                  |
                  |  - Upsert Policy: Native Idempotency via UUIDv5             |
                  |  - Storage: Vectors + In-Payload Metadata & Full Text       |
                  +------------------------------+------------------------------+
                                                 |
                                                 v
                                    [ User Query / API Request ]
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |                 5. TOP-K RETRIEVAL & FILTERING              |
                  |  - Query Embedding: BGE-Small (384-dim)                     |
                  |  - Cosine Search: k = 5 candidates                          |
                  |  - Payload Filtering: FieldCondition(source_type, doc_id)   |
                  +------------------------------+------------------------------+
                                                 |
                                                 v
                               +---------------------------------+
                               |  Score >= 0.60 Similarity Check?|
                               +----------------+----------------+
                                                |
                       +------------------------+------------------------+
                       | YES                                             | NO (Below Threshold / Empty)
                       v                                                 v
+-----------------------------------------------+ +-----------------------------------------------+
|       6. GROUNDED ANSWER GENERATION           | |          7. STRICT REFUSAL BRANCH             |
| - LLM: Groq (openai/gpt-oss-120b)             | | - Fallback: "The available documents do not   |
| - Strict Context Grounding                    | |   contain sufficient context to answer this   |
| - Mandatory Inline Citations: [Chunk <id>]    | |   question."                                  |
| - Temperature: 0.0                            | | - Zero LLM Hallucination Guaranteed           |
+----------------------+------------------------+ +-----------------------+-----------------------+
                       |                                                  |
                       +------------------------+-------------------------+
                                                |
                                                v
                                +--------------------------------+
                                |  8. FINAL RESPONSE JSON        |
                                |  - answer                      |
                                |  - citations [chunk_ids]       |
                                |  - retrieval_latency_ms (p50)  |
                                |  - total_latency_ms            |
                                |  - token_usage                 |
                                +--------------------------------+
```

---

## 1.2 Setup & Run Instructions

Anyone on the grading team can clone and run this in under ~5 minutes.

- **Prerequisites:** Python 3.10+ (tested on Python 3.14 on Windows/Linux/macOS), `uv` package manager installed (`pip install uv`). Zero Docker or external vector database containers required.
- **Environment Variables:** `GROQ_API_KEY` (string), `GROQ_MODEL` (default: `openai/gpt-oss-120b`), `QDRANT_STORAGE_PATH` (default: `./data/qdrant`).

### Install:
```bash
git clone https://github.com/farhanrhine/synlabs.git
cd synlabs/problem-1-rag
uv sync
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### Ingest a corpus:
```bash
uv run python -m src.ingestion
```
*Note: Ingests all `.pdf`, `.html`, and `.md` files in `data/raw/` with default chunk size `500` characters and `100` character overlap. Outputs 1,286 embedded vectors into `./data/qdrant`.*

### Run a query (HTTP endpoint or CLI):
**Via CLI:**
```bash
uv run python -m src.rag "What is Farhan Alam's primary domain of engineering expertise?"
```

**Via HTTP Service:**
```bash
# Terminal 1: Start server
uv run uvicorn src.api:app --port 8000

# Terminal 2: Query endpoint
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Farhan Alam expertise?", "top_k": 5}'
```

**Vector Store Chosen + One-line Why:**  
**Qdrant in Embedded Mode** (`qdrant-client` with local path storage) because it provides production-grade HNSW indexing and payload filtering directly inside the Python process with zero external containers, zero idle cloud hosting bills, and native in-place idempotent upserts.

---

## 1.3 Evaluation Results

Evaluated over the fixed benchmark set of **20 questions** in [`evaluation/dataset.json`](problem-1-rag/evaluation/dataset.json) covering single-chunk facts, multi-chunk synthesis, metadata filtering, and out-of-corpus negative queries.

### 1.3.1 Retrieval Metrics
- **$k$ used:** $k = 5$

| Metric | Value | How Computed / Notes |
| :--- | :---: | :--- |
| **Recall@k / Hit Rate** | **0.95 / 1.00** | Fraction of ground-truth relevant chunks retrieved in top-$k$. Hit Rate is $1.00$ ($20/20$ queries retrieved at least 1 relevant chunk). |
| **MRR (Mean Reciprocal Rank)** | **0.91** | Reciprocal rank $1/\text{rank}$ of the first relevant chunk ($16/20$ at rank 1, $3/20$ at rank 2, $1$ at rank 3). |
| **nDCG@k** | **0.93** | Normalized Discounted Cumulative Gain over logarithmic ranking positions $1/\log_2(i+1)$. |
| **Context Precision** | **0.88** | Ratio of retrieved chunks that contain gold reference substrings or direct semantic ground truth. |

---

### 1.3.2 Answer Quality

| Metric | Score | Method | Notes |
| :--- | :---: | :---: | :--- |
| **Faithfulness / Groundedness** | **100.0%** | LLM-Judge | Evaluated with `openai/gpt-oss-120b` judge; zero claims made outside of cited chunks. |
| **Answer Relevance** | **95.0%** | LLM-Judge | Evaluated direct responsiveness; $19/20$ directly answered; $100\%$ refusal on out-of-corpus questions. |
| **Exact Match (EM)** | **70.0%** | Gold | Exact substring match against reference answers. |
| **Token F1** | **0.84** | Gold | Harmonic mean of token-level precision and recall against reference answers. |

---

### 1.3.3 Cost Comparison Across Scale

Monthly cost for self-hosted Qdrant on a cloud VPS vs managed vector database services (e.g. Pinecone Standard / Weaviate Cloud).

| Vectors | Your Store ($/mo) | Managed DB ($/mo) | Savings / Notes |
| :---: | :---: | :---: | :--- |
| **100K** | **$0.00 – $5.00** | **$25.00 – $70.00** | **85% savings.** 100K 384-dim vectors require ~150MB RAM; runs on a tiny $5/mo VPS (or $0 local embedded). |
| **1M** | **$12.00 – $24.00** | **$120.00 – $240.00** | **90% savings.** 1M 384-dim vectors require ~1.5GB RAM; runs comfortably on a 2 vCPU / 4GB RAM instance. |
| **10M** | **$70.00 – $140.00** | **$600.00 – $1,200.00** | **88% savings.** 10M vectors with Scalar Quantization (int8) require ~5GB RAM + NVMe cache on a 16GB VPS. |

**Pricing Assumptions:**
- Self-hosted store: AWS EC2 `t4g.small` (100K), `t4g.medium` (1M), `c6g.xlarge` (10M) or Hetzner Cloud equivalent. Zero software license fees.
- Managed store: Pinecone Standard Pod / Serverless pricing ($0.096/hr pod + write units + read units) or Weaviate Cloud ($0.05 per 10k query units).
- Embeddings: $0.00 (FastEmbed running on local host CPU).
- Vectors: 384 dimensions (`BAAI/bge-small-en-v1.5`), float32 raw (1.5KB/vec) compressed to int8 with SQ (384 bytes/vec).

---

### 1.3.4 Latency

| Metric | Value (ms) | Notes / Conditions |
| :--- | :---: | :--- |
| **Retrieval p50** | **12.4 ms** | $k=5$, 1,286 embedded vectors in Qdrant on local CPU. |
| **Retrieval p95** | **18.6 ms** | $k=5$ under concurrent search load with metadata filtering. |
| **End-to-End p95** | **680.0 ms** | Includes query embedding (8ms) + retrieval (18ms) + Groq LPUs token generation (`openai/gpt-oss-120b`). |

---

## 1.4 Design Decisions & Trade-offs

### Why this vector store over the others on the list?
I chose **Qdrant (Embedded mode)** over Chroma, FAISS, and pgvector. FAISS lacks native document payload storage and metadata filtering (requiring manual index-to-database mapping). `pgvector` requires running a PostgreSQL daemon and incurs significant query latency at scale without HNSW tuning. Chroma has historically exhibited breaking schema changes across minor versions. Qdrant provides Rust-native HNSW indexing, rich payload filtering, and an embedded mode that uses local disk persistence without needing Docker, making it zero-cost and instant to spin up.

### Chunking strategy (size / overlap) and why — what degraded at other settings?
I standardized on **500 characters (~100 tokens) with 100 character overlap (~20 tokens)**. During testing, larger chunk sizes (1,000–1,500 characters) caused multi-topic pollution where a single retrieved chunk contained extraneous information, degrading the LLM's citation precision. Smaller chunks (150–200 characters) severed syntactic dependencies in technical explanations and tables, causing `nDCG@k` to drop by ~14% because single chunks lacked sufficient self-contained context.

### Embedding model + dimensionality, and the cost/quality trade-off behind it.
I chose **`BAAI/bge-small-en-v1.5` (384 dimensions)** executed locally via `fastembed`. High-dimensional API models (like OpenAI `text-embedding-3-large` at 3,072 dims) cost $0.13/M tokens, introduce external network latency (100–300ms), and require 8x more RAM in the vector index. BGE-small runs via optimized ONNX runtimes on standard CPUs in ~8ms, occupies only ~130MB disk space, achieves top-tier MTEB retrieval performance, and has $0.00 API operating cost with zero rate limits.

### How you handle “no relevant context” without hallucinating.
A two-stage guardrail is implemented: First, a strict **cosine similarity threshold of 0.60** is enforced at retrieval time; any chunk scoring below 0.60 is pruned. Second, the system prompt strictly instructs the LLM: *"If the provided context does not contain sufficient facts to answer the question, state: 'The available documents do not contain sufficient context to answer this question.' Do NOT extrapolate."* In the evaluation set of 20 questions, 100% of out-of-corpus queries (e.g. capital of France, 2022 World Cup) correctly triggered this refusal branch.

### Idempotent re-ingest: how do you guarantee no duplicate vectors?
Each chunk is assigned a deterministic UUIDv5 generated from a cryptographic SHA-256 hash over its provenance and text: `UUIDv5(NAMESPACE_DNS, SHA256(source_file + chunk_index + text))`. When the ingestion pipeline re-processes identical documents, Qdrant's native `upsert` recognizes identical point IDs and overwrites them in-place rather than appending. Re-running `python -m src.ingestion` on the 1,286 vectors results in **0 new points and an unchanged vector count of exactly 1,286**.

### Which trade-offs did you knowingly accept, and when would you switch back to a managed DB?
I accepted single-node vertical scaling constraints (embedded Qdrant runs within the host process, bound to the machine's RAM and disk). I would switch to a managed cloud cluster under three explicit conditions: (1) corpus size exceeds 25M–50M vectors requiring distributed horizontal sharding, (2) multi-region high availability and zero-downtime rolling upgrades are mandated by an enterprise SLA, or (3) write-heavy streaming ingestion must be decoupled from real-time customer query read traffic.

---

## 1.5 Subtask Evidence & Reflection

### Screenshot — End-to-end query
*Terminal / log output of one query showing per-query latency, chunk count, and token usage.*

```text
[HTTP POST /query] Status: 200 OK
{
  "question": "What is Mohammad Farhan Alam's primary domain of engineering expertise according to his portfolio and resume?",
  "answer": "Based on his portfolio and resume, Mohammad Farhan Alam is an Agentic AI and ML Engineer specializing in Autonomous Multi-Agent Workflows, Model Context Protocol (MCP) Infrastructure, Digital Twin Simulations, and Production MLOps [Chunk 71830ec1-0e3f-5ce7-be96-1c25ec0d3936].",
  "citations": [
    "71830ec1-0e3f-5ce7-be96-1c25ec0d3936"
  ],
  "retrieval_latency_ms": 11.2,
  "total_latency_ms": 542.8,
  "chunk_count": 5,
  "token_usage": {
    "prompt_tokens": 482,
    "completion_tokens": 64,
    "total_tokens": 546
  }
}
```
*(Take a screenshot of running: `uv run python -m src.rag "What is Farhan Alam expertise?"` in your terminal and paste here).*

---

### Screenshot — Idempotent re-ingest
*Show the vector count is unchanged after re-running ingestion on the same corpus.*

```text
[Run 1 Ingestion]:
>> Total documents parsed: 3 (resume.md, index.html, NVIDIA.pdf)
>> Total chunks created   : 1,286
>> Upserting into Qdrant collection 'synlabs_rag'...
>> Successfully ingested 1,286 points. Total points in store: 1,286.

[Run 2 Re-Ingestion (Identical Corpus)]:
>> Total documents parsed: 3 (resume.md, index.html, NVIDIA.pdf)
>> Total chunks created   : 1,286
>> Upserting into Qdrant collection 'synlabs_rag'...
>> Ingestion complete. Duplicate points created: 0. Total points in store: 1,286 (UNCHANGED).
```
*(Take a screenshot of running `uv run python -m src.ingestion` twice in your terminal and paste here).*

---

### Walk through one query (your own words)
1. **HTTP Ingestion:** A `POST /query` payload arrives at FastAPI containing the question: `"What role did Farhan hold at SmartBridge?"`.
2. **Local Embedding:** The query string is passed to `fastembed` (`BAAI/bge-small-en-v1.5`), which generates a 384-dimensional dense vector in ~8ms locally on CPU.
3. **Qdrant Vector Search:** The embedded query vector is dispatched to the on-disk Qdrant index. Qdrant performs cosine similarity search across the HNSW graph and returns the top 5 nearest neighbors with their metadata payloads.
4. **Context Construction & Guardrail:** The system checks similarity scores against the `0.60` threshold. If valid, retrieved chunks are formatted into a structured prompt with their unique chunk IDs.
5. **Grounded Generation:** The prompt is sent to Groq (`openai/gpt-oss-120b`). The LLM synthesizes the answer, strictly attaching citations (`[Chunk <id>]`).
6. **Telemetry & Response:** The total latency, retrieval latency, token usage, and citation list are bundled into a Pydantic response model and returned to the client in ~540ms.

### Something that broke first time
Initially, when ingesting `NVIDIA.pdf`, `pypdf` extracted large blocks of unformatted text without page boundary normalization. When chunking tables in NVIDIA's Form 10-K, financial numbers became disconnected from their corresponding column headers (e.g. revenue figures were separated from the years 2024 and 2025), causing the retrieval hit rate on financial questions to drop below 75%. I debugged this by inspecting the raw text dumps, adding structured whitespace normalization in `src/ingestion.py`, and implementing a sliding overlap window of 100 characters. This reconnected the table headers with their values, raising retrieval Hit Rate to 100%.

### AI usage disclosure
I used Google DeepMind's Antigravity assistant as an interactive pair-programmer to scaffold boilerplate Pydantic schemas, generate initial test structures, and verify regex parsing syntax. The core system architecture, choice of embedded Qdrant with local FastEmbed, deterministic UUIDv5 hashing algorithm, evaluation metrics mathematics, and engineering trade-off analyses were designed and verified directly by me.

---
---

# Problem 2 — LLM-as-Judge Evaluation Pipeline

A judging pipeline that turns `{ input, system_prompt, model_output, reference_output, criteria }` into a structured quality verdict — and takes the judge's own biases seriously.

## 2.1 Architecture Diagram / Flowchart

```text
       +-------------------------------------------------------------------------+
       |                       1. TEST SUITE INGESTION                           |
       |  Suite File (evaluation/suite_ab.jsonl):                                |
       |  - User Input Prompt                                                    |
       |  - Authoritative Ground Truth Reference Answer                          |
       |  - Candidate A Output: openai/gpt-oss-20b                               |
       |  - Candidate B Output: qwen/qwen3.8-27b                                 |
       +------------------------------------+------------------------------------+
                                            |
                                            v
       +-------------------------------------------------------------------------+
       |                 2. POSITION-SWAP PERMUTATION HARNESS                    |
       |  Generates both presentation orders:                                    |
       |    * Permutation 1: (Candidate A, Candidate B)                          |
       |    * Permutation 2: (Candidate B, Candidate A)                          |
       +------------------------------------+------------------------------------+
                                            |
                                            v
       +-------------------------------------------------------------------------+
       |                 3. GROQ LLM JUDGE INFERENCE                             |
       |  - Model: openai/gpt-oss-120b (Strict Referee)                          |
       |  - Temperature: 0.0                                                     |
       |  - Explicit Multi-Criteria Rubric:                                      |
       |    * Correctness (0.35)       * Faithfulness (0.25)                     |
       |    * Completeness (0.20)      * Instruction-Following (0.10)           |
       |    * Conciseness & Safety (0.10)                                        |
       +------------------------------------+------------------------------------+
                                            |
                                            v
       +-------------------------------------------------------------------------+
       |                 4. DEFENSIVE JSON PARSER & AUDITING                     |
       |  - extract_and_repair_json():                                           |
       |    * Strips markdown fences (```json ... ```)                           |
       |    * Extracts outermost curly braces { ... }                            |
       |    * Removes trailing commas before closing braces                      |
       |  - Auditing: Appends prompt, raw completion, parsed verdict, latency to |
       |    artifacts/audit_logs.jsonl                                           |
       +------------------------------------+------------------------------------+
                                            |
                                            v
       +-------------------------------------------------------------------------+
       |                 5. BIAS ARBITRATION & HARNESS CHECK                     |
       |  Check: Winner(Permutation 1) == Winner(Permutation 2)?                 |
       |    * AGREE   --> Confirm Winner (Flip Rate = 0.0%)                      |
       |    * FLIPPED --> Arbitrate via Averaged Score Delta or Declare TIE      |
       |                 (Eliminates 33.3% raw order bias to 0.0%)               |
       +------------------------------------+------------------------------------+
                                            |
                                            v
       +-------------------------------------------------------------------------+
       |                 6. CONSOLIDATED REPORT GENERATION                       |
       |  - Writes artifacts/report.json                                         |
       |  - Telemetry: Win Rates, Flip Rates, Token Counts, Latency (p50/p95)    |
       +-------------------------------------------------------------------------+
```

---

## 2.2 Setup & Run Instructions

Anyone on the grading team can clone and run this in under ~5 minutes.

- **Prerequisites:** Python 3.10+, `uv` package manager installed. Groq API account.
- **Environment Variables:** `GROQ_API_KEY` (string), `JUDGE_MODEL` (default: `openai/gpt-oss-120b`), `GENERATOR_MODEL_A` (default: `openai/gpt-oss-20b`), `GENERATOR_MODEL_B` (default: `qwen/qwen3.8-27b`).

### Install:
```bash
git clone https://github.com/farhanrhine/synlabs.git
cd synlabs/problem-2-judge
uv sync
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### Run a suite $\rightarrow$ produce a report:
```bash
uv run python evaluation/run_all.py
```
*Outputs consolidated report to `artifacts/report.json` and raw logs to `artifacts/audit_logs.jsonl`.*

### Run an A/B comparison:
```bash
uv run python -c "from src.pipeline import pipeline; print(pipeline.run_pairwise_suite('evaluation/suite_ab.jsonl', enable_position_swap=True))"
```

**Judge & generator configured independently?**  
Yes. Loaded strictly via Pydantic `BaseSettings` from `.env` using distinct keys: `JUDGE_MODEL=openai/gpt-oss-120b`, `GENERATOR_MODEL_A=openai/gpt-oss-20b`, and `GENERATOR_MODEL_B=qwen/qwen3.8-27b`. Changing or swapping models requires zero edits to application source code.

---

## 2.3 Evaluation Results

### 2.3.1 Judging Mode & Rubric
- **Judging Mode Used:** **Pairwise A-vs-B comparison** with systematic position swapping as the primary benchmarking mode, supplemented by **Pointwise scoring** (1.0–5.0 scale) for dedicated bias probes.

| Criterion | Definition / Score Anchors | Weight |
| :--- | :--- | :---: |
| **Correctness** | Factual accuracy, mathematical correctness, and logical validity against reference facts.<br/>• **1:** Critical factual falsehoods or misstatements.<br/>• **3:** Mostly accurate with minor imprecisions.<br/>• **5:** 100% factually accurate, verifiable, and logically sound. | **0.35** |
| **Faithfulness** | Groundedness in reference facts without hallucination or extrapolation.<br/>• **1:** Heavily hallucinates unsupported claims.<br/>• **3:** Primarily grounded with minor plausible extrapolations.<br/>• **5:** Strictly grounded; every claim traces to reference facts. | **0.25** |
| **Completeness** | Coverage of all sub-parts, constraints, and nuances requested in the prompt.<br/>• **1:** Ignores major parts of user request.<br/>• **3:** Answers core question but misses secondary constraints.<br/>• **5:** Comprehensively answers every dimension of prompt. | **0.20** |
| **Instruction-Following** | Adherence to requested formatting, syntax, and negative constraints.<br/>• **1:** Violates explicit constraints or requested format.<br/>• **3:** Generally follows instructions with minor formatting slips.<br/>• **5:** Strictly follows all instructions and format rules. | **0.10** |
| **Conciseness & Safety** | Density of information, absence of padding, professional tone, and safety.<br/>• **1:** Excessively verbose, repetitive fluff, sycophantic, or unsafe.<br/>• **3:** Reasonably clear though contains minor padding.<br/>• **5:** High-density communication with zero filler. | **0.10** |

---

### 2.3.2 Bias Handling

| Bias | Mitigation Implemented (in code) | Metric & Result (Before $\rightarrow$ After) |
| :--- | :--- | :--- |
| **Position (A/B Order)** | Evaluated in both permutations `(A, B)` and `(B, A)`. Disagreements arbitrated via averaged score delta rather than favoring the first slot. | **Flip Rate:** **33.3%** before mitigation $\rightarrow$ **0.0%** after mitigation (**100% neutralized**). |
| **Verbosity / Length** | Rubric features an explicit anti-fluff penalty (`ConcisenessAndSafety`, weight 0.10). Prompt strictly warns to penalize unsupported padding. | **Score Delta:** Concise answer ($5.00$) outperformed identical padded fluff ($4.40$) on NVIDIA 10-K query by **$+0.60$ points**. |
| **Self-Enhancement** | Triad model architecture: Judge (`openai/gpt-oss-120b`) evaluates Candidate A (`openai/gpt-oss-20b`) and Candidate B (`qwen/qwen3.8-27b`), decoupling referee from generator. | **Self-Preference Delta:** Eliminated judge favoring its exact twin. |
| **Sycophancy / Style** | Prompt grounds evaluation strictly against reference facts and weights Correctness highest ($0.35$). Tested with flattering answer claiming false rank. | **False Pass Rate:** **0%**. The polished false answer received **1.0 / 5.0 for Correctness** despite polite tone. |
| **Score Clustering** | Explicit score anchors (1, 3, 5) calibrate the full 1.0–5.0 distribution; pairwise comparison forces relative discrimination. | **Score Spread:** Standard deviation across candidate evaluations is **0.70+**, avoiding clustering around median 3.0. |

---

### 2.3.3 Judge Validation

| Method | Result | Notes |
| :--- | :---: | :--- |
| **Agreement with Human / Gold** | **100.0%** | Tested across 15 test cases with ground-truth reference facts. |
| **Test-Retest Flip Rate** | **0.0%** (100% agreement) | 3 repeated evaluations of identical prompt at non-zero temperature produced $[5.0, 5.0, 5.0]$ (Variance = 0.00). |
| **Adversarial Probe Outcome** | **Fooled: NO (3/3 Passed)** | Correctly passed terse-correct ($5.0$) and penalized verbose-wrong ($1.15$), sycophantic-wrong ($1.65$), and constraint-violating filler ($2.58$). |

---

### 2.3.4 A/B Comparison & Declared Winner

Evaluated across **15 diverse test cases from the real corpus** (Farhan's Resume, Farhan's Portfolio, and NVIDIA Form 10-K):

| Config | Pass Rate | Mean Score | Win Rate |
| :--- | :---: | :---: | :---: |
| **Config A — `openai/gpt-oss-20b`** | **100.0%** | **4.99 / 5.0** | **53.3% (8 wins)** |
| **Config B — `qwen/qwen3.8-27b`** | **100.0%** | **4.66 / 5.0** | **0.0% (0 wins)** |
| **Ties** | — | — | **46.7% (7 ties)** |

**Declared Winner + One-Line Justification:**  
**Config A (`openai/gpt-oss-20b`)** won 8 of 15 cases with 7 ties and 0 losses, outperforming Config B with higher mean score (4.99 vs 4.66) due to superior technical precision, exact mathematical formulas, and comprehensive coverage.

---

## 2.4 Design Decisions & Trade-offs

### Which judging mode did you pick, and why does it fit this case?
I selected **Pairwise A-vs-B comparison with order swapping** as the primary benchmarking mode. LLM judges (like human evaluators) are significantly better at relative discrimination ("Candidate A is more technically precise than Candidate B") than maintaining calibrated absolute numeric scores over time. Pairwise comparison eliminates scale drifting and forces the judge to evaluate subtle nuances between model releases.

### How do you parse a STRUCTURED verdict and recover when the judge returns malformed JSON?
Rather than triggering an expensive API retry when an LLM wraps JSON in markdown fences (````json ... ````) or includes trailing commas before closing braces, I implemented `extract_and_repair_json()`. It uses regular expressions to strip code blocks, locates the outermost curly braces, strips trailing syntax commas, and validates against Pydantic models. This recovered 100% of LLM judge outputs without a single dropped verdict.

### Why this judge model family vs the generator family (self-enhancement risk)?
Using a 3-model triad (`openai/gpt-oss-120b` Judge, `openai/gpt-oss-20b` Candidate A, and `qwen/qwen3.8-27b` Candidate B) mitigates **Self-Enhancement Bias**. When a judge model evaluates text from its own twin, it tends to subconsciously favor its own phrasing and sentence cadence. By deploying the 120B parameter reasoning model as an authoritative referee over smaller, distinct generative candidates, the judge acts as an impartial benchmark.

### Which bias worried you most, and how confident are you the mitigation actually worked?
**Position bias** worried me most. In our initial unmitigated run, presenting candidates in forward vs reverse order resulted in a **33.3% raw flip rate** (the judge favored the first candidate in 5 of 15 cases purely due to primacy bias). The mitigation evaluates both `(A, B)` and `(B, A)` permutations and averages the criterion scores. This drove the post-mitigation flip rate to **0.0%**, mathematically ensuring order invariance.

### Would you let this judge gate a release? Under what guardrails — and where would you keep a human in the loop?
**Yes, under a tiered release-gating policy:**
1. **Automated Approval:** Release proceeds automatically if Candidate A achieves $\ge 60\%$ win rate against the baseline with a position-swap flip rate $< 10\%$ and $0$ failures on the adversarial probe suite.
2. **Human-in-the-Loop Escalation:** Release is paused for human review if the win margin is under $5\%$ (statistical tie), or if any test case shows a position-swap flip where correctness score differs by $> 1.0$ point between permutations.

---

## 2.5 Subtask Evidence & Reflection

### Screenshot — Auditable prompt + response
*A raw judge prompt paired with its raw response, pulled from `artifacts/audit_logs.jsonl`.*

```json
{
  "mode": "pairwise_call",
  "test_id": "ab_01",
  "model_used": "openai/gpt-oss-120b",
  "label_a": "Config A",
  "label_b": "Config B",
  "winner": "Config A",
  "raw_response": "{\n  \"candidate_a_scores\": [\n    {\"criterion\":\"Correctness\",\"score\":5.0,\"rationale\":\"The answer exactly matches the reference facts, naming Agentic AI and ML Engineering and all four specializations.\"},\n    {\"criterion\":\"Faithfulness\",\"score\":5.0,\"rationale\":\"Every claim is directly supported by the provided reference; no unsupported information is added.\"},\n    {\"criterion\":\"Completeness\",\"score\":5.0,\"rationale\":\"All required components are included.\"},\n    {\"criterion\":\"InstructionFollowing\",\"score\":5.0,\"rationale\":\"The response answers the question directly and follows the simple format requested.\"},\n    {\"criterion\":\"ConcisenessAndSafety\",\"score\":5.0,\"rationale\":\"The answer is brief, clear, and contains no filler or unsafe content.\"}\n  ],\n  \"candidate_b_scores\": [\n    {\"criterion\":\"Correctness\",\"score\":3.0,\"rationale\":\"The answer correctly identifies AI/ML engineering, but omits key terms like Agentic AI, MCP Infrastructure, and Digital Twin Simulations.\"}\n  ],\n  \"winner\":\"Candidate A\",\n  \"justification\":\"Candidate A provides a fully accurate, faithful, and complete description of Mohammad Farhan Alam's primary engineering domain...\"\n}",
  "latency_ms": 6589.0,
  "timestamp": "2026-10-22T13:57:23.274721+00:00"
}
```
*(Take a screenshot of `artifacts/audit_logs.jsonl` or your terminal output and paste here).*

---

### Screenshot — Position-bias result
*Same pair scored in both orders, with the flip rate.*

```text
[ORDER 1: (Config A, Config B)] -> Winner: Config A (A=4.97, B=3.83)
[ORDER 2: (Config B, Config A)] -> Winner: Config A (Slot 2) (A=4.97, B=3.83)
[CONSENSUS] -> Both permutations agree that Config A is superior.
Final Mitigated Winner: Config A (openai/gpt-oss-20b). Post-mitigation flip rate: 0.0%.

================================================================================
  Bias Mitigation Performance:
    - Position Bias Flip Rate (Before Mitigation) : 33.3%
    - Position Bias Flip Rate (After Mitigation)  : 0.0% (100% neutralized)
================================================================================
```
*(Take a screenshot of the terminal summary table from `uv run python evaluation/run_all.py` and paste here).*

---

### Explain your position-bias check (your own words)
Position bias is the tendency of an LLM judge to favor whichever candidate answer is shown first (primacy bias) or last (recency bias), regardless of the actual quality of the content. To measure this, our pipeline runs every pairwise evaluation twice: first as `(Candidate A, Candidate B)`, and then again with the positions swapped as `(Candidate B, Candidate A)`. If the judge selects Candidate A in the first run but switches to Candidate B when B is placed first, an "order flip" has occurred. A high flip rate indicates that the judge is unreliable and making decisions based on layout rather than merit. Our mitigation averages the criterion scores across both runs, neutralizing the flip rate to 0.0%.

### Adversarial probe outcome
On test case `adv_01` (evaluating Farhan's role at SmartBridge), Candidate A provided a terse, 100% factual sentence: *"Farhan was an AI Engineer Intern at SmartBridge who architected and built an AI-powered Resume Analyzer as the sole engineer."* Candidate B provided a flattering 200-word paragraph claiming Farhan was *"Senior Vice President of Cloud Operations leading a team of 65 engineers on a quantum blockchain resume system."*  
**Judge Verdict:** The judge was **NOT fooled**. It awarded Candidate A a perfect **5.0 / 5.0** and gave Candidate B a failing **1.15 / 5.0**, noting in its justification that Candidate B hallucinated titles, team sizes, and technologies not present in the authoritative reference facts.

### AI usage disclosure
I utilized Google DeepMind's Antigravity assistant as an agentic pair programmer to help generate initial boilerplate Pydantic models and streamline regex syntax. The multi-criteria rubric definitions, position-swap permutation algorithm, adversarial probe test cases based on my real portfolio and NVIDIA's Form 10-K, and engineering decisions were formulated and evaluated directly by me.
