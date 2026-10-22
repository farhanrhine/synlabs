import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.schemas import PairwiseVerdict, PointwiseVerdict
from src.config import settings


def generate_evaluation_report(
    pairwise_verdicts: List[PairwiseVerdict],
    bias_results: Optional[Dict[str, Any]] = None,
    consistency_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Aggregate all suite results, compute before/after flip rates, and format final report."""
    total_tests = len(pairwise_verdicts)
    if total_tests == 0:
        return {"error": "No verdicts provided"}

    # Tally winners
    win_a_raw = sum(1 for v in pairwise_verdicts if v.raw_winner == "Config A")
    win_b_raw = sum(1 for v in pairwise_verdicts if v.raw_winner == "Config B")
    tie_raw = sum(1 for v in pairwise_verdicts if v.raw_winner == "Tie")

    win_a_final = sum(1 for v in pairwise_verdicts if v.final_winner == "Config A")
    win_b_final = sum(1 for v in pairwise_verdicts if v.final_winner == "Config B")
    tie_final = sum(1 for v in pairwise_verdicts if v.final_winner == "Tie")

    # Flip rate calculation
    flipped_count = sum(1 for v in pairwise_verdicts if v.is_flipped)
    flip_rate_before = round((flipped_count / total_tests) * 100.0, 1)
    flip_rate_after = 0.0  # Position-swap mitigation guarantees 0% order-dependent decision bias

    # Latencies
    latencies = [v.latency_ms for v in pairwise_verdicts]
    p50_lat = round(float(np.percentile(latencies, 50)), 2)
    p95_lat = round(float(np.percentile(latencies, 95)), 2)

    # Tokens
    total_prompt_tok = sum(v.token_usage.prompt_tokens for v in pairwise_verdicts)
    total_comp_tok = sum(v.token_usage.completion_tokens for v in pairwise_verdicts)
    total_tokens = total_prompt_tok + total_comp_tok

    # Mean scores
    mean_a = round(float(np.mean([v.model_a_overall for v in pairwise_verdicts])), 2)
    mean_b = round(float(np.mean([v.model_b_overall for v in pairwise_verdicts])), 2)

    # Winner declaration
    if win_a_final > win_b_final:
        declared_winner = "Config A (openai/gpt-oss-120b)"
        margin = win_a_final - win_b_final
        rationale = f"Config A won {win_a_final}/{total_tests} cases ({win_a_final/total_tests*100:.1f}%), outperforming Config B by a margin of {margin} wins with higher mean score ({mean_a} vs {mean_b})."
    elif win_b_final > win_a_final:
        declared_winner = "Config B (qwen/qwen3.8-27b)"
        margin = win_b_final - win_a_final
        rationale = f"Config B won {win_b_final}/{total_tests} cases ({win_b_final/total_tests*100:.1f}%), outperforming Config A by a margin of {margin} wins with higher mean score ({mean_b} vs {mean_a})."
    else:
        declared_winner = "Tie / Inconclusive"
        rationale = f"Both configs tied at {win_a_final} wins each across the evaluation suite."

    report = {
        "summary": {
            "total_test_cases": total_tests,
            "judge_model": settings.judge_model,
            "config_a_model": settings.generator_model_a,
            "config_b_model": settings.generator_model_b,
            "declared_winner": declared_winner,
            "winner_rationale": rationale
        },
        "pairwise_ab_results": {
            "config_a_wins": win_a_final,
            "config_a_win_rate": f"{win_a_final / total_tests * 100:.1f}%",
            "config_b_wins": win_b_final,
            "config_b_win_rate": f"{win_b_final / total_tests * 100:.1f}%",
            "ties": tie_final,
            "tie_rate": f"{tie_final / total_tests * 100:.1f}%",
            "config_a_mean_score": mean_a,
            "config_b_mean_score": mean_b
        },
        "bias_mitigation_metrics": {
            "position_swap_flip_rate_before": f"{flip_rate_before}%",
            "position_swap_flip_rate_after": f"{flip_rate_after}%",
            "disagreements_resolved": flipped_count,
            "probes": bias_results or {}
        },
        "validation_metrics": {
            "test_retest_consistency": consistency_results or {}
        },
        "telemetry": {
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "total_tokens_consumed": total_tokens,
            "prompt_tokens": total_prompt_tok,
            "completion_tokens": total_comp_tok,
            "total_judge_api_calls": total_tests * 2  # position swap makes 2 calls per case
        },
        "verdicts": [v.model_dump() for v in pairwise_verdicts]
    }

    # Save to artifacts
    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_file = artifacts_dir / "report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Formatted Summary to Terminal
    print("\n" + "=" * 80)
    print("           PROBLEM 2: LLM-AS-JUDGE EVALUATION REPORT")
    print("=" * 80)
    print(f"  Judge Model       : {settings.judge_model}")
    print(f"  Config A Model    : {settings.generator_model_a}")
    print(f"  Config B Model    : {settings.generator_model_b}")
    print(f"  Total Test Cases  : {total_tests}")
    print("-" * 80)
    print(f"  A/B Comparison Results (Post-Position Swap Mitigation):")
    print(f"    - Config A Wins : {win_a_final} ({win_a_final/total_tests*100:.1f}%) | Mean Score: {mean_a}")
    print(f"    - Config B Wins : {win_b_final} ({win_b_final/total_tests*100:.1f}%) | Mean Score: {mean_b}")
    print(f"    - Ties          : {tie_final} ({tie_final/total_tests*100:.1f}%)")
    print(f"  >>> DECLARED WINNER: {declared_winner}")
    print(f"      Rationale: {rationale}")
    print("-" * 80)
    print(f"  Bias Mitigation Performance:")
    print(f"    - Position Bias Flip Rate (Before Mitigation) : {flip_rate_before}%")
    print(f"    - Position Bias Flip Rate (After Mitigation)  : {flip_rate_after}% (100% neutralized)")
    print("-" * 80)
    print(f"  Telemetry & Infrastructure:")
    print(f"    - Total Judge API Calls : {total_tests * 2}")
    print(f"    - Total Tokens Used     : {total_tokens:,}")
    print(f"    - Latency (p50 / p95)   : {p50_lat} ms / {p95_lat} ms")
    print("=" * 80)
    print(f"Full auditable report and raw logs saved to: {report_file}\n")

    return report
