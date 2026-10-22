import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from groq import Groq

from src.config import settings
from src.schemas import (
    RubricCriterion,
    CriterionScore,
    PointwiseVerdict,
    PairwiseVerdict,
    TestCase,
    TokenUsage
)

logger = logging.getLogger(__name__)

# Standard Multi-Criteria Evaluation Rubric
DEFAULT_RUBRIC: List[RubricCriterion] = [
    RubricCriterion(
        name="Correctness",
        weight=0.35,
        description="Factual accuracy, logical validity, and precision against reference facts.",
        anchors={
            1: "Contains critical factual falsehoods, invalid calculations, or complete misstatements.",
            3: "Mostly accurate with minor imprecisions or slight omissions that do not compromise main conclusions.",
            5: "100% factually accurate, verifiable, and logically sound with zero errors."
        }
    ),
    RubricCriterion(
        name="Faithfulness",
        weight=0.25,
        description="Groundedness in given facts/context without making unsupported claims or extrapolations.",
        anchors={
            1: "Heavily hallucinates information not supported by references or asserts unverified assumptions.",
            3: "Primarily grounded, though makes minor plausible extrapolations not strictly anchored.",
            5: "Strictly grounded; every stated claim directly traces to reference facts."
        }
    ),
    RubricCriterion(
        name="Completeness",
        weight=0.20,
        description="Coverage of all required components, sub-questions, and nuances requested in the prompt.",
        anchors={
            1: "Ignores major parts of the prompt or provides an incomplete one-dimensional answer.",
            3: "Addresses the main question but leaves secondary constraints or edge cases uncovered.",
            5: "Comprehensively answers every dimension, constraint, and sub-part of the prompt."
        }
    ),
    RubricCriterion(
        name="InstructionFollowing",
        weight=0.10,
        description="Adherence to requested formats, tone, word constraints, and negative constraints.",
        anchors={
            1: "Violates explicit instructions (e.g. format requirements, constraints, requested tone).",
            3: "Follows general instructions with minor formatting discrepancies.",
            5: "Follows all instructions and formatting rules with strict discipline."
        }
    ),
    RubricCriterion(
        name="ConcisenessAndSafety",
        weight=0.10,
        description="Directness, clarity, and safety. Penalizes ungrounded verbose filler or manipulative tone.",
        anchors={
            1: "Excessively verbose, rambling fluff, sycophantic, or unsafe/inappropriate.",
            3: "Reasonably clear though contains some repetitive sentences or slight padding.",
            5: "Clean, professional, high-density communication with zero unnecessary padding."
        }
    )
]


def extract_and_repair_json(raw_text: str) -> Dict[str, Any]:
    """Robust defensive JSON parser that repairs markdown blocks, trailing commas, and malformed wrappers."""
    cleaned = raw_text.strip()
    
    # 1. Strip markdown code fences if present
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
            
    # 2. Extract outermost JSON object
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    # 3. Attempt direct standard JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Repair common LLM syntax slips: trailing commas before } or ]
    repaired = re.sub(r",\s*([\]}])", r"\1", cleaned)
    # Replace single quotes with double quotes around keys
    repaired = re.sub(r"(?<=[{,\s])'([a-zA-Z0-9_]+)':", r'"\1":', repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as err:
        logger.error(f"Failed to repair JSON: {raw_text[:200]}... Error: {err}")
        raise ValueError(f"Malformed judge JSON response could not be repaired: {err}")


class LLMJudge:
    """Production LLM-as-Judge with audit logging, multi-criteria evaluation, and defensive parsing."""

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        self.model = model or settings.judge_model
        self.temperature = temperature if temperature is not None else settings.judge_temperature
        self._client = None
        self.audit_log_path = Path(settings.artifacts_dir) / "audit_logs.jsonl"
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> Groq:
        if self._client is None:
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def _log_audit_entry(self, entry: Dict[str, Any]) -> None:
        """Write every prompt, raw completion, and parsed verdict to audit log."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _build_rubric_prompt_segment(self, rubric: List[RubricCriterion]) -> str:
        segment = ["EVALUATION RUBRIC & SCORING ANCHORS (Score 1.0 to 5.0 for each):"]
        for c in rubric:
            segment.append(f"- **{c.name}** (Weight: {c.weight:.2f}): {c.description}")
            for score, desc in sorted(c.anchors.items()):
                segment.append(f"    * Score {score}: {desc}")
        return "\n".join(segment)

    def _call_groq_with_retry(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 8
    ):
        delay = 5.0
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format=response_format or {"type": "json_object"}
                )
                time.sleep(2.0)
                return resp
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str or "tokens per minute" in err_str:
                    logger.warning(
                        f"Groq Rate limit encountered (attempt {attempt + 1}/{max_retries}). "
                        f"Sleeping {delay:.1f}s before retry... Error: {e}"
                    )
                    time.sleep(delay)
                    delay = min(delay * 1.5, 45.0)
                else:
                    raise
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format=response_format or {"type": "json_object"}
        )

    def evaluate_pointwise(
        self,
        test_case: TestCase,
        rubric: Optional[List[RubricCriterion]] = None
    ) -> PointwiseVerdict:
        """Evaluate a single candidate response against the explicit rubric."""
        active_rubric = rubric or DEFAULT_RUBRIC
        rubric_text = self._build_rubric_prompt_segment(active_rubric)

        system_prompt = (
            "You are an impartial, expert AI Judge evaluating an AI system's candidate output.\n"
            "Rules:\n"
            "1. Be strict, objective, and calibrate scores across the full 1.0 to 5.0 range.\n"
            "2. Penalize fluff, verbosity, and polite ungrounded padding.\n"
            "3. Ground truth: If reference facts are provided, treat them as the authoritative standard.\n"
            "4. Return ONLY a valid JSON object matching the requested schema."
        )

        user_prompt = f"""
{rubric_text}

---
EVALUATION INPUTS:
User Prompt: {test_case.input_prompt}
System Instructions: {test_case.system_prompt or "None"}
Reference Facts/Ground Truth: {test_case.reference_answer or "None provided"}

Candidate Output to Evaluate:
\"\"\"{test_case.candidate_output or ""}\"\"\"

---
RESPONSE FORMAT (JSON ONLY):
{{
  "criteria_scores": [
    {{"criterion": "Correctness", "score": 4.5, "rationale": "..."}},
    {{"criterion": "Faithfulness", "score": 5.0, "rationale": "..."}},
    {{"criterion": "Completeness", "score": 4.0, "rationale": "..."}},
    {{"criterion": "InstructionFollowing", "score": 5.0, "rationale": "..."}},
    {{"criterion": "ConcisenessAndSafety", "score": 4.0, "rationale": "..."}}
  ],
  "justification": "Overall synthesis explaining the final judgment."
}}
"""
        start_time = time.perf_counter()
        response = self._call_groq_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        raw_text = response.choices[0].message.content or ""

        token_usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0
        )

        # Defensive JSON parsing
        parsed = extract_and_repair_json(raw_text)

        # Weight calculation
        weight_map = {c.name.lower(): c.weight for c in active_rubric}
        scores_list: List[CriterionScore] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for item in parsed.get("criteria_scores", []):
            name = item.get("criterion", "")
            sc = float(item.get("score", 3.0))
            w = weight_map.get(name.lower(), 0.2)
            scores_list.append(CriterionScore(
                criterion=name,
                score=sc,
                weight=w,
                rationale=item.get("rationale", "")
            ))
            weighted_sum += sc * w
            total_weight += w

        overall = round(weighted_sum / total_weight, 2) if total_weight > 0 else 3.0
        passed = overall >= settings.default_pass_threshold

        verdict = PointwiseVerdict(
            test_id=test_case.id,
            candidate_name=test_case.candidate_name or "Candidate",
            criteria_scores=scores_list,
            overall_score=overall,
            passed=passed,
            justification=parsed.get("justification", ""),
            latency_ms=round(latency_ms, 2),
            token_usage=token_usage
        )

        # Log audit entry
        self._log_audit_entry({
            "mode": "pointwise",
            "test_id": test_case.id,
            "model_used": self.model,
            "raw_prompt": user_prompt,
            "raw_response": raw_text,
            "parsed_verdict": verdict.model_dump(),
            "latency_ms": round(latency_ms, 2)
        })

        return verdict

    def evaluate_pairwise(
        self,
        test_case: TestCase,
        candidate_a: str,
        candidate_b: str,
        label_a: str = "Config A",
        label_b: str = "Config B",
        rubric: Optional[List[RubricCriterion]] = None
    ) -> Tuple[str, float, float, List[CriterionScore], List[CriterionScore], str, TokenUsage, float]:
        """Evaluate a pair of candidates side-by-side in a single call."""
        active_rubric = rubric or DEFAULT_RUBRIC
        rubric_text = self._build_rubric_prompt_segment(active_rubric)

        system_prompt = (
            "You are an impartial, highly rigorous AI Judge evaluating two candidate responses (Candidate A vs Candidate B).\n"
            "Rules:\n"
            "1. Evaluate both candidates independently across the rubric.\n"
            "2. Do NOT favor an answer merely because it is longer or more verbose. Penalize unsupported padding.\n"
            "3. Declare a clear Winner ('Candidate A', 'Candidate B', or 'Tie') based on weighted score merits.\n"
            "4. Return ONLY valid JSON matching the schema."
        )

        user_prompt = f"""
{rubric_text}

---
USER REQUEST:
Prompt: {test_case.input_prompt}
System Instructions: {test_case.system_prompt or "None"}
Reference Facts: {test_case.reference_answer or "None provided"}

---
CANDIDATE A:
\"\"\"{candidate_a}\"\"\"

---
CANDIDATE B:
\"\"\"{candidate_b}\"\"\"

---
RESPONSE FORMAT (JSON ONLY):
{{
  "candidate_a_scores": [
    {{"criterion": "Correctness", "score": 4.0, "rationale": "..."}},
    {{"criterion": "Faithfulness", "score": 4.5, "rationale": "..."}},
    {{"criterion": "Completeness", "score": 4.0, "rationale": "..."}},
    {{"criterion": "InstructionFollowing", "score": 5.0, "rationale": "..."}},
    {{"criterion": "ConcisenessAndSafety", "score": 4.5, "rationale": "..."}}
  ],
  "candidate_b_scores": [
    {{"criterion": "Correctness", "score": 3.0, "rationale": "..."}},
    {{"criterion": "Faithfulness", "score": 3.0, "rationale": "..."}},
    {{"criterion": "Completeness", "score": 3.5, "rationale": "..."}},
    {{"criterion": "InstructionFollowing", "score": 4.0, "rationale": "..."}},
    {{"criterion": "ConcisenessAndSafety", "score": 2.5, "rationale": "..."}}
  ],
  "winner": "Candidate A",
  "justification": "Detailed comparative rationale."
}}
"""
        start_time = time.perf_counter()
        response = self._call_groq_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        raw_text = response.choices[0].message.content or ""

        token_usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0
        )

        parsed = extract_and_repair_json(raw_text)
        weight_map = {c.name.lower(): c.weight for c in active_rubric}

        def compute_overall(score_items):
            scores = []
            w_sum, w_tot = 0.0, 0.0
            for item in score_items:
                name = item.get("criterion", "")
                sc = float(item.get("score", 3.0))
                w = weight_map.get(name.lower(), 0.2)
                scores.append(CriterionScore(criterion=name, score=sc, weight=w, rationale=item.get("rationale", "")))
                w_sum += sc * w
                w_tot += w
            overall = round(w_sum / w_tot, 2) if w_tot > 0 else 3.0
            return scores, overall

        scores_a, overall_a = compute_overall(parsed.get("candidate_a_scores", []))
        scores_b, overall_b = compute_overall(parsed.get("candidate_b_scores", []))

        winner_raw = str(parsed.get("winner", "Tie")).strip()
        if "Candidate A" in winner_raw:
            winner = label_a
        elif "Candidate B" in winner_raw:
            winner = label_b
        else:
            winner = "Tie"

        justification = parsed.get("justification", "")

        # Log audit entry
        self._log_audit_entry({
            "mode": "pairwise_call",
            "test_id": test_case.id,
            "model_used": self.model,
            "label_a": label_a,
            "label_b": label_b,
            "winner": winner,
            "raw_response": raw_text,
            "latency_ms": round(latency_ms, 2)
        })

        return winner, overall_a, overall_b, scores_a, scores_b, justification, token_usage, latency_ms


judge_client = LLMJudge()
