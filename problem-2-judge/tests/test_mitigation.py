import pytest
from unittest.mock import MagicMock
from src.schemas import TestCase, TokenUsage, CriterionScore
from src.bias_mitigation import BiasMitigationHarness


def test_position_swap_agreement():
    mock_judge = MagicMock()
    # Order 1: Config A wins
    mock_judge.evaluate_pairwise.side_effect = [
        ("Config A", 4.5, 3.0, [], [], "A better", TokenUsage(), 50.0),
        # Order 2: Config A wins (swapped call returns Config A)
        ("Config A", 3.0, 4.5, [], [], "A better", TokenUsage(), 50.0)
    ]

    harness = BiasMitigationHarness(judge=mock_judge)
    tc = TestCase(id="tc_agree", input_prompt="Prompt", config_a_output="Out A", config_b_output="Out B")
    verdict = harness.evaluate_pairwise_with_position_swap(tc)

    assert verdict.is_flipped is False
    assert verdict.final_winner == "Config A"


def test_position_swap_flip_resolution():
    mock_judge = MagicMock()
    # Order 1: Config A wins
    # Order 2: Config B wins (flip detected!)
    mock_judge.evaluate_pairwise.side_effect = [
        ("Config A", 3.8, 3.2, [], [], "Favored first", TokenUsage(), 50.0),
        ("Config B", 3.8, 3.2, [], [], "Favored first again", TokenUsage(), 50.0)
    ]

    harness = BiasMitigationHarness(judge=mock_judge)
    tc = TestCase(id="tc_flip", input_prompt="Prompt", config_a_output="Out A", config_b_output="Out B")
    verdict = harness.evaluate_pairwise_with_position_swap(tc)

    # Flip detected
    assert verdict.is_flipped is True
    # Disagreement resolved to Tie because scores are identical
    assert verdict.final_winner == "Tie"
    assert "Position disagreement" in verdict.justification
