import pytest
from src.judge import extract_and_repair_json


def test_clean_json_parse():
    raw = '{"winner": "Candidate A", "score": 4.5}'
    res = extract_and_repair_json(raw)
    assert res["winner"] == "Candidate A"
    assert res["score"] == 4.5


def test_markdown_fence_removal():
    raw = """
Here is my evaluation verdict:
```json
{
  "winner": "Candidate B",
  "justification": "Candidate B had higher accuracy."
}
```
Thank you for using the evaluation service.
"""
    res = extract_and_repair_json(raw)
    assert res["winner"] == "Candidate B"
    assert "accuracy" in res["justification"]


def test_trailing_comma_repair():
    raw = """
{
  "criteria_scores": [
    {"criterion": "Correctness", "score": 4.0,},
    {"criterion": "Faithfulness", "score": 5.0,},
  ],
  "justification": "Well grounded.",
}
"""
    res = extract_and_repair_json(raw)
    assert len(res["criteria_scores"]) == 2
    assert res["criteria_scores"][0]["score"] == 4.0
    assert res["justification"] == "Well grounded."


def test_unrepairable_json_raises():
    raw = "Not a json at all, completely malformed"
    with pytest.raises(ValueError, match="Malformed judge JSON response"):
        extract_and_repair_json(raw)
