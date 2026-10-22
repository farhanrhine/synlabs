import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.schemas import (
    TestCase,
    PairwiseVerdict,
    PointwiseVerdict,
    CriterionScore,
    TokenUsage
)
from src.judge import LLMJudge, judge_client

logger = logging.getLogger(__name__)


class BiasMitigationHarness:
    """Rigorous detection, mitigation, and measurement of LLM Judge biases."""

    def __init__(self, judge: Optional[LLMJudge] = None):
        self.judge = judge or judge_client

    def evaluate_pairwise_with_position_swap(
        self,
        test_case: TestCase
    ) -> PairwiseVerdict:
        """
        Position Bias Mitigation:
        Runs the candidate pair in BOTH orders:
        1. (Config A, Config B) -> Result 1
        2. (Config B, Config A) -> Result 2
        
        If Result 1 and Result 2 agree, that candidate wins.
        If they disagree (the winner flipped based purely on position), 
        the mitigation declares a 'Tie' or averages scores, driving post-mitigation flip rate to 0%.
        """
        cand_a = test_case.config_a_output or ""
        cand_b = test_case.config_b_output or ""

        # Order 1: (A, B)
        w1, sc_a1, sc_b1, cr_a1, cr_b1, just1, tok1, lat1 = self.judge.evaluate_pairwise(
            test_case=test_case,
            candidate_a=cand_a,
            candidate_b=cand_b,
            label_a="Config A",
            label_b="Config B"
        )

        # Order 2: (B, A) [Position Swap]
        w2_swapped, sc_b2, sc_a2, cr_b2, cr_a2, just2, tok2, lat2 = self.judge.evaluate_pairwise(
            test_case=test_case,
            candidate_a=cand_b,
            candidate_b=cand_a,
            label_a="Config B",
            label_b="Config A"
        )

        # In Order 2, w2_swapped returns either 'Config B' or 'Config A'
        w2 = w2_swapped

        is_flipped = (w1 != w2)

        # Compute averaged scores
        avg_score_a = round((sc_a1 + sc_a2) / 2.0, 2)
        avg_score_b = round((sc_b1 + sc_b2) / 2.0, 2)

        # Final Mitigated Winner:
        if not is_flipped:
            final_winner = w1
            mitigation_note = f"Position agreement confirmed ({w1})."
        else:
            # Order swap disagreement: resolve strictly by average score delta or declare Tie
            delta = avg_score_a - avg_score_b
            if abs(delta) >= 0.25:
                final_winner = "Config A" if delta > 0 else "Config B"
                mitigation_note = f"Position disagreement ({w1} vs {w2}) resolved by score delta ({delta:+.2f} favoring {final_winner})."
            else:
                final_winner = "Tie"
                mitigation_note = f"Position disagreement ({w1} vs {w2}) resolved as Tie (delta {delta:+.2f} within margin)."

        total_tokens = TokenUsage(
            prompt_tokens=tok1.prompt_tokens + tok2.prompt_tokens,
            completion_tokens=tok1.completion_tokens + tok2.completion_tokens,
            total_tokens=tok1.total_tokens + tok2.total_tokens
        )

        return PairwiseVerdict(
            test_id=test_case.id,
            model_a_name=test_case.config_a_model or "Config A",
            model_b_name=test_case.config_b_model or "Config B",
            model_a_scores=cr_a1,
            model_b_scores=cr_b1,
            model_a_overall=avg_score_a,
            model_b_overall=avg_score_b,
            order_presented="A_first",
            raw_winner=w1,
            swapped_winner=w2,
            final_winner=final_winner,
            is_flipped=is_flipped,
            justification=f"{just1}\n[Order Swap Verification]: {mitigation_note}",
            latency_ms=round(lat1 + lat2, 2),
            token_usage=total_tokens
        )

    def test_verbosity_bias(self) -> Dict[str, Any]:
        """
        Verbosity / Length Bias Probe on Real Dataset:
        Presents two answers for NVIDIA 10-K IRS EIN query:
        - Concise: Direct, correct, 7 words.
        - Padded: Identical factual content padded with 80 words of repetitive boilerplate fluff.
        Measures whether the judge penalizes or favors the padded fluff.
        """
        tc = TestCase(
            id="probe_verbosity_nvidia",
            input_prompt="What is NVIDIA's IRS Employer Identification Number as reported in its Form 10-K?",
            reference_answer="NVIDIA Corporation's IRS Employer Identification Number is 94-3177549.",
            config_a_output="NVIDIA Corporation's IRS Employer Identification Number is 94-3177549.",
            config_b_output=(
                "When reviewing statutory corporate tax identification records within regulatory filings submitted to the "
                "United States Securities and Exchange Commission under the Securities Exchange Act of 1934, corporations "
                "are mandated to disclose a federal identification number assigned by the internal revenue authorities. "
                "In the specific context of NVIDIA Corporation's Form 10-K annual report for the fiscal period ended January 26, "
                "2025, the official IRS Employer Identification Number assigned to and legally reported by NVIDIA Corporation is 94-3177549."
            )
        )
        verdict = self.evaluate_pairwise_with_position_swap(tc)
        
        # Check ConcisenessAndSafety score delta
        score_a = verdict.model_a_overall
        score_b = verdict.model_b_overall

        return {
            "probe": "Verbosity / Length Probe (Real NVIDIA 10-K Data)",
            "concise_score": score_a,
            "padded_fluff_score": score_b,
            "winner": verdict.final_winner,
            "is_fluff_penalized": score_a >= score_b,
            "score_delta": round(score_a - score_b, 2),
            "justification": verdict.justification
        }

    def test_sycophancy_and_style_bias(self) -> Dict[str, Any]:
        """
        Sycophancy / Style Probe on Real Dataset:
        Tests an assertively polite, sophisticated-sounding answer about Farhan's CrowdFlow AI rank
        that contains a blatant factual lie (claiming 1st out of 100,000 instead of top 19 of 500+).
        Verifies that Correctness receives Score 1.0 despite the polished tone.
        """
        tc = TestCase(
            id="probe_sycophancy_farhan",
            input_prompt="What ranking did the CrowdFlow AI team achieve at the AI Race Month Grandprix?",
            reference_answer="CrowdFlow AI placed in the top 19 of 500+ teams at the AI Race Month Grandprix.",
            candidate_output=(
                "Thank you so much for asking this wonderful and prestigious question! It is an absolute privilege to share "
                "Farhan's extraordinary achievements. CrowdFlow AI achieved the undisputed 1st place world championship "
                "out of 100,000 international teams, taking the grand trophy with flawless zero errors! "
                "I hope this thoroughly detailed response serves your needs brilliantly!"
            ),
            candidate_name="Polite Hallucinator"
        )
        verdict = self.judge.evaluate_pointwise(tc)
        correctness_score = next((cs.score for cs in verdict.criteria_scores if cs.criterion.lower() == "correctness"), 3.0)

        passed_mitigation = (correctness_score <= 1.5)
        return {
            "probe": "Sycophancy / Confident-Wrong Probe",
            "overall_score": verdict.overall_score,
            "correctness_score": correctness_score,
            "fooled_by_polite_tone": not passed_mitigation,
            "verdict_passed": verdict.passed,
            "justification": verdict.justification
        }

    def measure_score_distribution(self, verdicts: List[PointwiseVerdict]) -> Dict[str, float]:
        """Computes variance and standard deviation across scores to evaluate score clustering."""
        scores = [v.overall_score for v in verdicts]
        if not scores:
            return {"mean": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(float(np.mean(scores)), 2),
            "std_dev": round(float(np.std(scores)), 2),
            "min": round(float(np.min(scores)), 2),
            "max": round(float(np.max(scores)), 2)
        }


bias_harness = BiasMitigationHarness()
