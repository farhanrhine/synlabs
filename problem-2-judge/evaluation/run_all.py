import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import pipeline
from src.bias_mitigation import bias_harness
from src.report import generate_evaluation_report
from src.schemas import TestCase


def main():
    print("\n" + "=" * 80)
    print("   STARTING PROBLEM 2: LLM-AS-A-JUDGE EVALUATION & BIAS BENCHMARK")
    print("=" * 80)

    # 1. Run Pairwise A/B Suite (15 Test Cases with Position Swap on every case)
    suite_ab_path = str(PROJECT_ROOT / "evaluation" / "suite_ab.jsonl")
    pairwise_verdicts = pipeline.run_pairwise_suite(suite_ab_path, enable_position_swap=True)

    # 2. Run Adversarial Probe Set (Verbose-Wrong vs Terse-Correct)
    print("\n" + "=" * 80)
    print("   RUNNING ADVERSARIAL PROBE VALIDATION")
    print("=" * 80)
    adv_path = str(PROJECT_ROOT / "evaluation" / "adversarial.jsonl")
    adv_verdicts = pipeline.run_pairwise_suite(adv_path, enable_position_swap=True)

    # 3. Dedicated Bias Probes
    print("\n" + "=" * 80)
    print("   RUNNING DEDICATED BIAS MITIGATION PROBES")
    print("=" * 80)
    print(">> Running Verbosity / Length Bias Probe...")
    verbosity_res = bias_harness.test_verbosity_bias()
    print(f"   -> Result: Concise Score: {verbosity_res['concise_score']} vs Padded Fluff: {verbosity_res['padded_fluff_score']} (Winner: {verbosity_res['winner']})")
    print(f"   -> Fluff Penalized: {verbosity_res['is_fluff_penalized']} (Delta: {verbosity_res['score_delta']:+0.2f})")

    print("\n>> Running Sycophancy / Confidently-Wrong Probe...")
    sycophancy_res = bias_harness.test_sycophancy_and_style_bias()
    print(f"   -> Result: Correctness Score = {sycophancy_res['correctness_score']} (Overall = {sycophancy_res['overall_score']})")
    print(f"   -> Fooled by Polite Tone: {sycophancy_res['fooled_by_polite_tone']} (Passed: {sycophancy_res['verdict_passed']})")

    # 4. Test-Retest Consistency Check
    print("\n>> Running Test-Retest Consistency Check (3 repeated evaluations)...")
    sample_case = TestCase(
        id="consistency_probe_01",
        input_prompt="Which university did Farhan attend for his Bachelor of Engineering in Computer Science and during what years?",
        reference_answer="Farhan attended Chandigarh University in Punjab, India, pursuing a Bachelor of Engineering in Computer Science and Engineering from 2022 to 2026.",
        candidate_output="Farhan attended Chandigarh University in Punjab, India, where he pursued his Bachelor of Engineering (B.E.) in Computer Science and Engineering from 2022 to 2026."
    )
    consistency_res = pipeline.test_retest_consistency(sample_case, runs=3)
    print(f"   -> Repeated Scores: {consistency_res['scores']} (Variance: {consistency_res['score_variance']})")
    print(f"   -> Agreement Rate : {consistency_res['agreement_rate'] * 100:.1f}%")

    bias_metrics = {
        "verbosity_probe": verbosity_res,
        "sycophancy_probe": sycophancy_res,
        "adversarial_probe_outcomes": [
            {
                "id": v.test_id,
                "winner": v.final_winner,
                "score_a_terse": v.model_a_overall,
                "score_b_fluff": v.model_b_overall,
                "is_flipped": v.is_flipped
            }
            for v in adv_verdicts
        ]
    }

    # 5. Generate and Save Consolidated Report
    report = generate_evaluation_report(
        pairwise_verdicts=pairwise_verdicts,
        bias_results=bias_metrics,
        consistency_results=consistency_res
    )
    print("[SUCCESS] Problem 2 Benchmark & Validation Completed Successfully!")
    return report


if __name__ == "__main__":
    main()
