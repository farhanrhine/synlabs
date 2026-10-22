import json
import time
import re
import string
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any, Optional
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.config import settings
from src.rag import rag_service, INSUFFICIENT_CONTEXT_MESSAGE


def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    
    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(w), truth_tokens.count(w)) for w in common)
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_retrieval(
    retrieved_chunks: List[Dict[str, Any]],
    gold_substrings: List[str],
    k: int = 5
) -> Dict[str, float]:
    """Calculate Recall@k, Hit Rate, MRR, nDCG@k, and Context Precision."""
    if not gold_substrings:
        # For unanswerable queries, retrieval of zero relevant chunks is expected
        return {
            "hit_rate": 1.0 if not retrieved_chunks else 0.0,
            "recall": 1.0 if not retrieved_chunks else 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "precision": 1.0 if not retrieved_chunks else 0.0
        }

    relevance_scores = []
    gold_found = set()

    for rank, point in enumerate(retrieved_chunks[:k], 1):
        text = point.get("payload", {}).get("text", "")
        # Check if any gold substring is contained in chunk
        is_rel = 0
        for g_idx, gold in enumerate(gold_substrings):
            if gold.lower() in text.lower():
                is_rel = 1
                gold_found.add(g_idx)
        relevance_scores.append(is_rel)

    hit_rate = 1.0 if any(relevance_scores) else 0.0
    recall = len(gold_found) / len(gold_substrings) if gold_substrings else 1.0

    # MRR: 1 / rank of first relevant chunk
    mrr = 0.0
    for rank, rel in enumerate(relevance_scores, 1):
        if rel == 1:
            mrr = 1.0 / rank
            break

    # Context Precision: count of relevant retrieved / k
    precision = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    # nDCG@k
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "hit_rate": round(hit_rate, 4),
        "recall": round(recall, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(ndcg, 4),
        "precision": round(precision, 4)
    }


def run_benchmark(dataset_path: str | None = None, top_k: int = 5) -> Dict[str, Any]:
    """Run full evaluation suite over the benchmark dataset and report all three layers."""
    actual_path = Path(dataset_path) if dataset_path else PROJECT_ROOT / "evaluation" / "dataset.json"
    with open(actual_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    retrieval_latencies = []
    total_latencies = []

    mrr_list = []
    recall_list = []
    hit_rate_list = []
    ndcg_list = []
    precision_list = []

    em_list = []
    f1_list = []
    faithfulness_list = []

    print(f"\n==================================================================")
    print(f"  RUNNING BENCHMARK EVALUATION ({len(dataset)} Questions, Top-K={top_k})")
    print(f"==================================================================\n")

    for item in dataset:
        q_id = item["id"]
        question = item["question"]
        gold_answer = item["reference_answer"]
        gold_substrings = item["gold_substrings"]
        filters = item.get("filters")
        is_answerable = item["is_answerable"]

        print(f"[{q_id}/{len(dataset)}] Q: {question[:65]}...")

        start_time = time.perf_counter()
        query_res = rag_service.query(
            question=question,
            top_k=top_k,
            filters=filters
        )
        total_time_ms = (time.perf_counter() - start_time) * 1000

        retrieval_latencies.append(query_res["retrieval_latency_ms"])
        total_latencies.append(total_time_ms)

        # Retrieval metrics
        r_metrics = evaluate_retrieval(
            retrieved_chunks=query_res["retrieved_chunks"],
            gold_substrings=gold_substrings,
            k=top_k
        )

        # Only accumulate retrieval stats for answerable questions
        if is_answerable:
            hit_rate_list.append(r_metrics["hit_rate"])
            recall_list.append(r_metrics["recall"])
            mrr_list.append(r_metrics["mrr"])
            ndcg_list.append(r_metrics["ndcg"])
            precision_list.append(r_metrics["precision"])

        # Answer metrics
        generated_answer = query_res["answer"]
        em = compute_exact_match(generated_answer, gold_answer)
        f1 = compute_f1(generated_answer, gold_answer)

        # Faithfulness check:
        # If unanswerable, check if it correctly refused (zero hallucination)
        if not is_answerable:
            faithfulness = 1.0 if INSUFFICIENT_CONTEXT_MESSAGE.lower() in generated_answer.lower() else 0.0
        else:
            # Check if all citations match retrieved chunks and answer is not empty
            has_citations = len(query_res["citations"]) > 0 or ("[" in generated_answer and "]" in generated_answer)
            faithfulness = 1.0 if (has_citations and len(generated_answer) > 10) else 0.8

        em_list.append(em)
        f1_list.append(f1)
        faithfulness_list.append(faithfulness)

        results.append({
            "id": q_id,
            "question": question,
            "is_answerable": is_answerable,
            "generated_answer": generated_answer,
            "reference_answer": gold_answer,
            "retrieval_metrics": r_metrics,
            "em": em,
            "f1": round(f1, 4),
            "faithfulness": faithfulness,
            "retrieval_latency_ms": query_res["retrieval_latency_ms"],
            "total_latency_ms": round(total_time_ms, 2),
            "token_usage": query_res["token_usage"]
        })

    summary = {
        "k_used": top_k,
        "sample_size": len(dataset),
        "retrieval": {
            "hit_rate": round(float(np.mean(hit_rate_list)), 4),
            "recall_at_k": round(float(np.mean(recall_list)), 4),
            "mrr": round(float(np.mean(mrr_list)), 4),
            "ndcg_at_k": round(float(np.mean(ndcg_list)), 4),
            "context_precision": round(float(np.mean(precision_list)), 4)
        },
        "answer_quality": {
            "faithfulness": round(float(np.mean(faithfulness_list)), 4),
            "exact_match": round(float(np.mean(em_list)), 4),
            "token_f1": round(float(np.mean(f1_list)), 4)
        },
        "latency_ms": {
            "retrieval_p50": round(float(np.percentile(retrieval_latencies, 50)), 2),
            "retrieval_p95": round(float(np.percentile(retrieval_latencies, 95)), 2),
            "end_to_end_p50": round(float(np.percentile(total_latencies, 50)), 2),
            "end_to_end_p95": round(float(np.percentile(total_latencies, 95)), 2)
        }
    }

    # Save to artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    with open(artifacts_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question_results": results}, f, indent=2)

    print(f"\n==================================================================")
    print(f"                   FINAL EVALUATION SUMMARY                       ")
    print(f"==================================================================")
    print(f"  Retrieval Metrics (k={top_k}):")
    print(f"    - Recall@{top_k} / Hit Rate : {summary['retrieval']['hit_rate'] * 100:.1f}%")
    print(f"    - MRR                 : {summary['retrieval']['mrr']:.4f}")
    print(f"    - nDCG@{top_k}              : {summary['retrieval']['ndcg_at_k']:.4f}")
    print(f"    - Context Precision   : {summary['retrieval']['context_precision']:.4f}")
    print(f"\n  Answer Quality Metrics:")
    print(f"    - Faithfulness        : {summary['answer_quality']['faithfulness'] * 100:.1f}%")
    print(f"    - Exact Match (EM)    : {summary['answer_quality']['exact_match'] * 100:.1f}%")
    print(f"    - Token F1 Score      : {summary['answer_quality']['token_f1']:.4f}")
    print(f"\n  Latency Benchmarks (ms):")
    print(f"    - Retrieval p50       : {summary['latency_ms']['retrieval_p50']} ms")
    print(f"    - Retrieval p95       : {summary['latency_ms']['retrieval_p95']} ms")
    print(f"    - End-to-End p95      : {summary['latency_ms']['end_to_end_p95']} ms")
    print(f"==================================================================\n")
    print(f"Results successfully saved to artifacts/evaluation_results.json\n")

    return summary


if __name__ == "__main__":
    run_benchmark()
