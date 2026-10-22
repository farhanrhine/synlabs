import pytest
from src.schemas import (
    RubricCriterion,
    CriterionScore,
    PointwiseVerdict,
    PairwiseVerdict,
    TestCase,
    TokenUsage
)


def test_rubric_criterion_creation():
    rc = RubricCriterion(
        name="Correctness",
        weight=0.35,
        description="Factual accuracy",
        anchors={1: "Poor", 3: "Average", 5: "Excellent"}
    )
    assert rc.name == "Correctness"
    assert rc.weight == 0.35
    assert len(rc.anchors) == 3


def test_criterion_score_validation():
    cs = CriterionScore(criterion="Correctness", score=4.5, weight=0.35, rationale="Great answer")
    assert cs.score == 4.5
    assert cs.weight == 0.35

    with pytest.raises(Exception):
        # Score out of bounds (must be <= 5.0)
        CriterionScore(criterion="Correctness", score=6.0, weight=0.35, rationale="Too high")


def test_pairwise_verdict_schema():
    pv = PairwiseVerdict(
        test_id="tc_01",
        model_a_name="Model A",
        model_b_name="Model B",
        model_a_scores=[CriterionScore(criterion="Correctness", score=4.0, weight=0.5, rationale="")],
        model_b_scores=[CriterionScore(criterion="Correctness", score=3.0, weight=0.5, rationale="")],
        model_a_overall=4.0,
        model_b_overall=3.0,
        order_presented="A_first",
        raw_winner="Config A",
        swapped_winner="Config A",
        final_winner="Config A",
        is_flipped=False,
        justification="Model A was superior"
    )
    assert pv.final_winner == "Config A"
    assert pv.is_flipped is False
