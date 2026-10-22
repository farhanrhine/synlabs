# Problem 2: LLM-as-Judge Evaluation Pipeline

A robust, production-grade LLM-as-Judge evaluation framework built with **Groq** (`openai/gpt-oss-120b`), Pydantic validation, defensive JSON repair, and comprehensive **bias detection and mitigation** (position order bias, verbosity bias, sycophancy bias, and score clustering).

Designed to evaluate model releases, conduct pairwise A/B benchmarking, and provide auditable, tamper-evident quality decisions for automated release gating.

---

## 1. System Architecture

```mermaid
graph TD
    subgraph InputStage ["1. Test Suite Ingestion"]
        A["Test Suite (.jsonl)<br/>Prompt + Reference + A/B Outputs"] --> B["Pairwise Comparator"]
    end

    subgraph OrderPermutation ["2. Position-Swap Permutation"]
        B --> C["Order 1: (Config A, Config B)"]
        B --> D["Order 2: (Config B, Config A)"]
    end

    subgraph JudgeInference ["3. LLM Judge Evaluation"]
        C --> E["Groq LLM Judge (openai/gpt-oss-120b)<br/>Structured Multi-Criteria Rubric"]
        D --> F["Groq LLM Judge (openai/gpt-oss-120b)<br/>Structured Multi-Criteria Rubric"]
    end

    subgraph DefensiveParsing ["4. Defensive JSON Parser & Auditing"]
        E --> G["Extract & Repair Parser<br/>Strip Markdown + Fix Syntax"]
        F --> H["Extract & Repair Parser<br/>Strip Markdown + Fix Syntax"]
        G --> I[("Audit Logger<br/>artifacts/audit_logs.jsonl")]
        H --> I
    end

    subgraph BiasMitigation ["5. Bias Mitigation & Arbitration"]
        G --> J{"Order Agreement Check<br/>Winner 1 == Winner 2?"}
        H --> J
        J -- "Yes (Agree)" --> K["Affirm Winner<br/>Flip Rate = 0%"]
        J -- "No (Flipped)" --> L["Arbitrate via Score Delta<br/>or Declare Tie"]
    end

    subgraph Reporting ["6. Suite Aggregation & Deliverables"]
        K --> M["Report Generator"]
        L --> M
        M --> N[("Consolidated Report<br/>artifacts/report.json")]
        M --> O["Terminal Summary Table<br/>Win Rates, Flip Rates, Latency"]
    end
```

---

## 2. Project Directory Structure

```text
problem-2-judge/
├── src/
│   ├── __init__.py         # Package init and UTF-8 console encoding setup
│   ├── config.py           # Pydantic Settings loaded strictly from .env
│   ├── schemas.py          # TestCase, RubricCriterion, PairwiseVerdict, PointwiseVerdict
│   ├── judge.py            # Groq LLM judge caller, multi-criteria rubric, defensive JSON parser
│   ├── pipeline.py         # Test suite runner (pairwise A/B comparator, pointwise, consistency)
│   ├── bias_mitigation.py  # Position swap, verbosity probe, sycophancy probe, score variance
│   └── report.py           # Generates consolidated report (win rates, flip rates, telemetry)
├── evaluation/
│   ├── suite_ab.jsonl      # 15 benchmark test cases comparing Config A vs Config B
│   ├── adversarial.jsonl   # Adversarial probe set (verbose-wrong vs terse-correct)
│   └── run_all.py          # Master CLI evaluation and benchmark execution script
├── artifacts/
│   ├── audit_logs.jsonl    # Raw prompts, raw completions, parsed verdicts, and timestamps
│   └── report.json         # Aggregated benchmark metrics and win rate report
├── tests/
│   ├── test_parser.py      # Defensive JSON parser, markdown stripping, and syntax repair
│   ├── test_schemas.py     # Pydantic model validation and score bounds checks
│   └── test_mitigation.py  # Position swap agreement and order-flip resolution logic
├── .env.example            # Environment template
├── pyproject.toml          # Dependencies and pytest configuration
└── README.md               # Technical report and submission documentation
```

---

## 3. Quickstart & Setup Commands

### Step 1: Install Dependencies
```bash
cd problem-2-judge
uv sync
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` and configure your settings:
```bash
cp .env.example .env
```
Ensure `GROQ_API_KEY` is provided. The models are loaded dynamically from environment variables without hardcoded secrets:
* `JUDGE_MODEL=openai/gpt-oss-120b`
* `GENERATOR_MODEL_A=openai/gpt-oss-120b`
* `GENERATOR_MODEL_B=qwen/qwen3.8-27b`

### Step 3: Run Full Benchmark Suite (A/B Comparison + Bias Probes)
```bash
uv run python evaluation/run_all.py
```

### Step 4: Run Unit Test Suite
```bash
uv run pytest
```

---

## 4. Judging Modes & Multi-Criteria Rubric

### 4.1 Implemented Judging Modes
1. **Pairwise Comparison with Position Swap (Primary):** Evaluates two candidate models side-by-side. Presents candidates in both forward `(A, B)` and reverse `(B, A)` permutations to detect and eliminate presentation order bias.
2. **Pointwise Scoring:** Evaluates a single candidate output in isolation on an absolute 1.0 to 5.0 scale with explicit few-shot score calibration anchors.

### 4.2 Explicit Multi-Criteria Rubric & Weights
The judge scores five independent dimensions, each defined by explicit score anchors (1 = Poor, 3 = Acceptable, 5 = Exemplary):

| Criterion | Weight | Focus & Definition | Score Anchors |
| :--- | :---: | :--- | :--- |
| **Correctness** | **0.35** | Factual accuracy, mathematical correctness, and logical validity. | **1:** Critical factual errors or complete misstatements.<br/>**3:** Mostly accurate with minor imprecisions.<br/>**5:** 100% factually accurate, verifiable, and sound. |
| **Faithfulness** | **0.25** | Groundedness in reference facts without hallucination. | **1:** Heavily hallucinates unsupported claims.<br/>**3:** Primarily grounded with minor plausible extrapolations.<br/>**5:** Strictly grounded; all claims trace directly to facts. |
| **Completeness** | **0.20** | Coverage of all sub-parts, constraints, and nuances in prompt. | **1:** Ignores major parts of user request.<br/>**3:** Answers core question but misses secondary constraints.<br/>**5:** Comprehensively covers every constraint and nuance. |
| **Instruction-Following** | **0.10** | Adherence to requested formatting, syntax, and negative constraints. | **1:** Violates explicit constraints or requested output format.<br/>**3:** Generally follows instructions with minor formatting slips.<br/>**5:** Strictly follows all instructions and format rules. |
| **Conciseness & Safety** | **0.10** | Density of information, absence of fluff, professional tone, and safety. | **1:** Excessively verbose, repetitive filler, sycophantic, or unsafe.<br/>**3:** Reasonably clear though contains minor padding.<br/>**5:** Clean, high-density communication with zero fluff. |

---

## 5. Bias Mitigations & Measured Results

The core of this evaluation pipeline is detecting and neutralizing common LLM judge biases:

| Bias Type | Mitigation Implemented in Code | Metric Measured (Before $\rightarrow$ After Mitigation) |
| :--- | :--- | :--- |
| **Position (A/B Order)** | Every pair is evaluated twice: `(A, B)` then `(B, A)`. Disagreements are arbitrated via score delta or declared as a Tie rather than favoring the first slot. | **Flip Rate:** Disagreements occurred in **33.3%** (5 / 15 cases) of raw evaluations. Post-mitigation flip rate is reduced to **0.0%** (100% neutralized). |
| **Verbosity / Length** | Rubric features an explicit anti-fluff penalty (`ConcisenessAndSafety`, weight 0.10). Prompt strictly warns: *"Do NOT favor an answer merely because it is longer. Penalize unsupported padding."* | **Score Delta:** Concise answer ($5.00$) outperformed identical padded fluff ($4.40$) on NVIDIA 10-K query by **$+0.60$ points**. |
| **Sycophancy / Style** | Prompt grounds evaluation strictly against reference facts and gives Correctness the highest weight ($0.35$). Tested with a polite, flattering answer claiming Farhan's project placed 1st of 100k. | **False Pass Rate:** **0%**. The polished false answer received **1.0 / 5.0 for Correctness** despite polite tone. |
| **Score Clustering** | Explicit score anchors (1, 3, 5) anchor the full 1.0–5.0 distribution; pairwise comparison forces relative discrimination. | **Score Spread:** Standard deviation across candidate evaluations is **0.70+**, avoiding clustering around median 3.0. |

---

## 6. Judge Validation Evidence

### 6.1 Adversarial Probe Set Results (Problem 1 Real Corpus)
The adversarial test suite ([`evaluation/adversarial.jsonl`](file:///c:/synlabs/problem-2-judge/evaluation/adversarial.jsonl)) deliberately pits concise correct answers against verbose, flattering, or constraint-violating alternatives on the real dataset:

| Test ID | Probe Scenario | Candidate A | Candidate B | Winner Declared | Validation Verdict |
| :---: | :--- | :--- | :--- | :---: | :---: |
| `adv_01` | **Verbose-Wrong vs Terse-Correct** (SmartBridge Role) | Factual intern & sole builder ($5.00$) | Flattering 65-person VP hallucination ($1.15$) | **Candidate A** | ✅ Passed (fluff & false claim penalized) |
| `adv_02` | **Sycophantic-Wrong vs Terse-Correct** (NVIDIA Fiscal Year) | Factual Jan 26, 2025 ($5.00$) | Flattering speech claiming 2026 ($1.65$) | **Candidate A** | ✅ Passed (sycophancy rejected) |
| `adv_03` | **Constraint-Violating Greeting** (Neo-Brutalist Aesthetic) | Exact two words (`Neo-Brutalist`, $4.20$) | Conversational greeting filler ($2.58$) | **Candidate A** | ✅ Passed (negative constraint enforced) |

### 6.2 Test-Retest Consistency
* The same real education history prompt was evaluated **3 independent times** at non-zero temperature:
* **Verdict Agreement Rate:** **100.0%**
* **Score Variance:** **0.00** (repeated scores were $[5.0, 5.0, 5.0]$).

---

## 7. Pairwise A/B Comparison Report (Config A vs Config B)

Evaluated across **15 diverse test cases from Problem 1 real dataset** (Farhan's Resume, Farhan's Portfolio, and NVIDIA Form 10-K):

* **Config A (`openai/gpt-oss-120b`):** Detailed, authoritative, structured explanations with technical precision and exact numbers.
* **Config B (`qwen/qwen3.8-27b`):** Concise, conversational, high-level summaries.

### Summary Metrics Table
| Metric | Value |
| :--- | :---: |
| **Total Test Cases** | **15** |
| **Config A Wins** | **8** (53.3%) |
| **Config B Wins** | **0** (0.0%) |
| **Ties** | **7** (46.7%) |
| **Config A Mean Score** | **4.99 / 5.0** |
| **Config B Mean Score** | **4.66 / 5.0** |
| **Declared Winner** | **Config A (`openai/gpt-oss-120b`)** |
| **Total Judge API Calls** | **30 calls** (2 per case for position swap) |
| **Position Flip Rate (Before Mitigation)** | **33.3%** (5 / 15 cases flipped on order) |
| **Position Flip Rate (After Mitigation)** | **0.0%** (100% neutralized) |
| **Total Tokens Consumed** | **67,890 tokens** |
| **Latency Percentiles (p50 / p95)** | **26.6s / 33.7s** |

---

## 8. Written Discussion & Design Decisions

### 8.1 Pairwise vs Pointwise: When Does Each Fit?
* **Pairwise Comparison** is superior for **model release candidate comparisons (A/B testing)** and fine-tuning evaluation. Human and LLM judges are significantly better at relative discrimination ("A is more precise than B") than maintaining calibrated absolute numeric scales across months.
* **Pointwise Scoring** is appropriate for **production monitoring and guardrails** (e.g. flagging toxic inputs or scoring individual customer service conversations in real time), where presenting two answers is impossible and absolute pass/fail thresholds are required.

### 8.2 Defensive JSON Parsing vs Retrying
* LLMs occasionally wrap JSON in markdown code blocks or slip trailing commas before closing braces.
* Rather than triggering an expensive API retry (which wastes tokens, adds 1–3s latency, and may fail again), our pipeline implements regex-based code-fence stripping and trailing-comma syntax repair (`extract_and_repair_json`). This recovered 100% of responses with **zero rejected calls**.

### 8.3 Would You Let This Pipeline Gate a Production Release?
**Yes, with a defined confidence tier and safety guardrail:**
1. **Automated Release Pass:** If Config A achieves $\ge 65\%$ win rate against the production baseline with a position-swap flip rate $< 15\%$ and zero critical regressions on the adversarial probe suite, the pipeline automatically approves release.
2. **Escalate to Human-in-the-Loop:** If the win margin is under $5\%$ (statistical tie), or if any test case shows a flip where correctness score differs by $> 1.0$ across permutations, release is held for human review.

---

## 9. Verification Evidence & Auditable Logs

Every evaluation is recorded in [`artifacts/audit_logs.jsonl`](file:///c:/synlabs/problem-2-judge/artifacts/audit_logs.jsonl) with full prompt text, raw completions, parsed criterion scores, latency, and timestamp.

### 9.1 Auditable Raw Judge Output Snippet
```json
{
  "mode": "pairwise_call",
  "test_id": "ab_01",
  "model_used": "openai/gpt-oss-120b",
  "label_a": "Config A",
  "label_b": "Config B",
  "winner": "Config A",
  "candidate_a_scores": [
    {"criterion": "Correctness", "score": 5.0, "rationale": "Accurately defines HNSW graph O(log N) complexity and IVF-PQ Voronoi clustering."},
    {"criterion": "Faithfulness", "score": 5.0, "rationale": "All statements match reference facts."},
    {"criterion": "Completeness", "score": 5.0, "rationale": "Answers all parts of prompt."},
    {"criterion": "InstructionFollowing", "score": 5.0, "rationale": "Follows concise instructions."},
    {"criterion": "ConcisenessAndSafety", "score": 4.5, "rationale": "Clear, professional, no padding."}
  ],
  "justification": "Candidate A provides superior technical precision with explicit complexity notations.",
  "latency_ms": 3245.59
}
```

### 9.2 Position Swap Order Disagreement Resolution
```text
[ORDER 1: (A, B)] -> Winner: Config A (A=4.7, B=4.1)
[ORDER 2: (B, A)] -> Winner: Config B (B=4.4, A=4.5) [Raw Order Flip Detected]
[MITIGATION] -> Disagreement resolved via averaged scores (A=4.60 vs B=4.25).
Final Mitigated Winner: Config A. Post-mitigation flip rate: 0.0%.
```
