import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.schemas import TestCase, PairwiseVerdict, PointwiseVerdict
from src.judge import LLMJudge, judge_client
from src.bias_mitigation import BiasMitigationHarness, bias_harness

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Executes evaluation test suites, position-swap A/B benchmarks, and consistency checks."""

    def __init__(self, judge: Optional[LLMJudge] = None, harness: Optional[BiasMitigationHarness] = None):
        self.judge = judge or judge_client
        self.harness = harness or bias_harness

    def load_suite(self, suite_path: str) -> List[TestCase]:
        """Load test cases from JSONL or JSON file."""
        path = Path(suite_path)
        if not path.exists():
            raise FileNotFoundError(f"Test suite file not found at: {suite_path}")

        test_cases = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        test_cases.append(TestCase(**json.loads(line)))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                test_cases = [TestCase(**item) for item in data]

        return test_cases

    def run_pairwise_suite(
        self,
        suite_path: str,
        enable_position_swap: bool = True
    ) -> List[PairwiseVerdict]:
        """Run full Pairwise A-vs-B test suite with position swap bias mitigation on every case."""
        cases = self.load_suite(suite_path)
        verdicts: List[PairwiseVerdict] = []

        print(f"\n==================================================================")
        print(f"   RUNNING PAIRWISE A/B EVALUATION SUITE ({len(cases)} test cases)")
        print(f"==================================================================\n")

        for idx, tc in enumerate(cases, 1):
            print(f"[{idx}/{len(cases)}] Evaluating: {tc.input_prompt[:60]}...")
            if enable_position_swap:
                verdict = self.harness.evaluate_pairwise_with_position_swap(tc)
            else:
                w, sc_a, sc_b, cr_a, cr_b, just, tok, lat = self.judge.evaluate_pairwise(
                    test_case=tc,
                    candidate_a=tc.config_a_output or "",
                    candidate_b=tc.config_b_output or "",
                    label_a="Config A",
                    label_b="Config B"
                )
                verdict = PairwiseVerdict(
                    test_id=tc.id,
                    model_a_name=tc.config_a_model or "Config A",
                    model_b_name=tc.config_b_model or "Config B",
                    model_a_scores=cr_a,
                    model_b_scores=cr_b,
                    model_a_overall=sc_a,
                    model_b_overall=sc_b,
                    order_presented="A_first",
                    raw_winner=w,
                    final_winner=w,
                    is_flipped=False,
                    justification=just,
                    latency_ms=round(lat, 2),
                    token_usage=tok
                )

            flip_str = " [FLIPPED in raw swap]" if verdict.is_flipped else ""
            print(f"   -> Result: Winner = {verdict.final_winner}{flip_str} (Scores: A={verdict.model_a_overall}, B={verdict.model_b_overall})")
            verdicts.append(verdict)

        return verdicts

    def run_pointwise_suite(self, suite_path: str) -> List[PointwiseVerdict]:
        """Run pointwise evaluation suite across all test cases."""
        cases = self.load_suite(suite_path)
        verdicts: List[PointwiseVerdict] = []

        print(f"\n==================================================================")
        print(f"   RUNNING POINTWISE EVALUATION SUITE ({len(cases)} test cases)")
        print(f"==================================================================\n")

        for idx, tc in enumerate(cases, 1):
            print(f"[{idx}/{len(cases)}] Pointwise Scoring: {tc.input_prompt[:60]}...")
            verdict = self.judge.evaluate_pointwise(tc)
            status_str = "PASS" if verdict.passed else "FAIL"
            print(f"   -> Result: [{status_str}] Overall Score = {verdict.overall_score:.2f}")
            verdicts.append(verdict)

        return verdicts

    def test_retest_consistency(self, test_case: TestCase, runs: int = 3) -> Dict[str, Any]:
        """
        Judge Validation: Test-Retest Consistency
        Evaluates the exact same test case repeatedly to measure decision stability.
        """
        verdicts = []
        for i in range(runs):
            v = self.judge.evaluate_pointwise(test_case)
            verdicts.append(v)

        passed_list = [v.passed for v in verdicts]
        scores = [v.overall_score for v in verdicts]
        is_consistent = len(set(passed_list)) == 1

        agreement_rate = 1.0 if is_consistent else (passed_list.count(passed_list[0]) / runs)

        return {
            "runs": runs,
            "scores": scores,
            "score_variance": round(float(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)), 4),
            "verdicts_passed": passed_list,
            "is_100_percent_consistent": is_consistent,
            "agreement_rate": round(agreement_rate, 2)
        }


pipeline = EvaluationPipeline()
