"""
Unit tests for the LLM-as-a-Judge evaluation module.
"""

from knowledge.evaluation.judge import LLMJudgeEvaluator


def test_json_extractor_clean_json() -> None:
    """Verify parser extracts clean JSON objects."""
    judge = LLMJudgeEvaluator()
    sample = '{"score": 0.95, "reason": "Accurate facts"}'
    parsed = judge._extract_json(sample)
    assert parsed.get("score") == 0.95
    assert parsed.get("reason") == "Accurate facts"


def test_json_extractor_surrounded_by_markdown() -> None:
    """Verify parser handles code blocks and extra text around JSON."""
    judge = LLMJudgeEvaluator()
    sample = """Here is the evaluation result:
```json
{
  "score": 0.85,
  "reason": "Mostly grounded in context"
}
```
Thank you."""
    parsed = judge._extract_json(sample)
    assert parsed.get("score") == 0.85
    assert parsed.get("reason") == "Mostly grounded in context"


def test_score_faithfulness_fallback() -> None:
    """Verify score_faithfulness returns structured format even if LLM fails or is offline."""
    judge = LLMJudgeEvaluator()
    res = judge.score_faithfulness("Context text", "Answer text")
    assert "score" in res
    assert 0.0 <= res["score"] <= 1.0
    assert "reason" in res


def test_score_relevance_fallback() -> None:
    """Verify score_relevance returns structured format even if LLM fails or is offline."""
    judge = LLMJudgeEvaluator()
    res = judge.score_relevance("Query text", "Answer text")
    assert "score" in res
    assert 0.0 <= res["score"] <= 1.0
    assert "reason" in res


def test_evaluate_response_structure() -> None:
    """Verify evaluate_response aggregates faithfulness and relevance."""
    judge = LLMJudgeEvaluator()
    res = judge.evaluate_response("Query", "Context", "Answer")
    assert "faithfulness" in res
    assert "relevance" in res
    assert "faithfulness_reason" in res
    assert "relevance_reason" in res
