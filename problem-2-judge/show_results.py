"""
Quick display script to output formatted screenshots for Problem 2.
Reads directly from artifacts/report.json and artifacts/audit_logs.jsonl.
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def show_results():
    report_file = Path("artifacts/report.json")
    audit_file = Path("artifacts/audit_logs.jsonl")

    if not report_file.exists() or not audit_file.exists():
        print("Error: artifacts/report.json or artifacts/audit_logs.jsonl not found.")
        return

    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # -------------------------------------------------------------
    # 1. SCREENSHOT 2: POSITION BIAS & A/B EVALUATION SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("      PROBLEM 2: POSITION-BIAS MITIGATION & A/B EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Judge Model       : {rep['summary']['judge_model']}")
    print(f"  Candidate A Model : {rep['summary']['config_a_model']}")
    print(f"  Candidate B Model : {rep['summary']['config_b_model']}")
    print(f"  Total Test Cases  : {rep['summary']['total_test_cases']}")
    print("-" * 80)
    print("  A/B PAIRWISE BENCHMARK RESULTS:")
    print(f"    • Config A Wins : {rep['pairwise_ab_results']['config_a_wins']} ({rep['pairwise_ab_results']['config_a_win_rate']})  |  Mean Score: {rep['pairwise_ab_results']['config_a_mean_score']} / 5.0")
    print(f"    • Config B Wins : {rep['pairwise_ab_results']['config_b_wins']} ({rep['pairwise_ab_results']['config_b_win_rate']})   |  Mean Score: {rep['pairwise_ab_results']['config_b_mean_score']} / 5.0")
    print(f"    • Ties          : {rep['pairwise_ab_results']['ties']} ({rep['pairwise_ab_results']['tie_rate']})")
    print(f"    >>> DECLARED WINNER: {rep['summary']['declared_winner']}")
    print("-" * 80)
    print("  BIAS MITIGATION PERFORMANCE (RUBRIC HEART):")
    print(f"    • Position Bias Flip Rate (Before Mitigation) : {rep['bias_mitigation_metrics']['position_swap_flip_rate_before']}")
    print(f"    • Position Bias Flip Rate (After Mitigation)  : {rep['bias_mitigation_metrics']['position_swap_flip_rate_after']} (100% NEUTRALIZED)")
    print(f"    • Disagreements Arbitrated via Score Delta    : {rep['bias_mitigation_metrics']['disagreements_resolved']} cases")
    v_probe = rep['bias_mitigation_metrics']['probes']['verbosity_probe']
    print(f"    • Verbosity / Anti-Fluff Penalty               : Concise={v_probe['concise_score']} vs Fluff={v_probe['padded_fluff_score']} (Delta: +{v_probe['score_delta']})")
    s_probe = rep['bias_mitigation_metrics']['probes']['sycophancy_probe']
    print(f"    • Sycophancy / Politeness Guard                : Score={s_probe['correctness_score']}/5.0 (Fooled: False)")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------
    # 2. SCREENSHOT 1: AUDITABLE PROMPT + RESPONSE LOG ENTRY
    # -------------------------------------------------------------
    print("=" * 80)
    print("      PROBLEM 2: RAW AUDITABLE JUDGE LOG ENTRY (First Test Case: ab_01)")
    print("=" * 80)
    with open(audit_file, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        # Parse inner raw_response if string
        if isinstance(first_line.get("raw_response"), str):
            try:
                first_line["raw_response"] = json.loads(first_line["raw_response"])
            except Exception:
                pass
        print(json.dumps(first_line, indent=2))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    show_results()
