from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ScoreAnchor(BaseModel):
    score: int = Field(..., ge=1, le=5)
    description: str


class RubricCriterion(BaseModel):
    name: str
    weight: float = Field(..., ge=0.0, le=1.0)
    description: str
    anchors: Dict[int, str] = Field(default_factory=dict)


class CriterionScore(BaseModel):
    criterion: str
    score: float = Field(..., ge=1.0, le=5.0)
    weight: float = Field(default=0.2, ge=0.0, le=1.0)
    rationale: str


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class PointwiseVerdict(BaseModel):
    test_id: str
    candidate_name: str = "Candidate"
    criteria_scores: List[CriterionScore]
    overall_score: float = Field(..., ge=1.0, le=5.0)
    passed: bool
    justification: str
    latency_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class PairwiseVerdict(BaseModel):
    test_id: str
    model_a_name: str = "Config A"
    model_b_name: str = "Config B"
    model_a_scores: List[CriterionScore]
    model_b_scores: List[CriterionScore]
    model_a_overall: float
    model_b_overall: float
    
    # Position bias check
    order_presented: Literal["A_first", "B_first"] = "A_first"
    raw_winner: Literal["Config A", "Config B", "Tie"]
    swapped_winner: Optional[Literal["Config A", "Config B", "Tie"]] = None
    final_winner: Literal["Config A", "Config B", "Tie"]
    is_flipped: bool = False
    
    justification: str
    latency_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class TestCase(BaseModel):
    __test__ = False
    id: str
    input_prompt: str
    system_prompt: Optional[str] = None
    reference_answer: Optional[str] = None
    criteria: Optional[List[str]] = None
    
    # For Pointwise
    candidate_output: Optional[str] = None
    candidate_name: Optional[str] = None
    
    # For Pairwise
    config_a_output: Optional[str] = None
    config_b_output: Optional[str] = None
    config_a_model: Optional[str] = None
    config_b_model: Optional[str] = None
    
    # Metadata & Tags
    category: Optional[str] = "general"
    is_adversarial: bool = False
    expected_winner: Optional[str] = None
