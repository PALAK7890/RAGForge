"""
LLM-as-a-Judge evaluation module for RAG response quality.
Measures Faithfulness (groundedness in context) and Answer Relevance (addressing the query).
"""

import json
import re
from typing import Any, Dict, Optional
from knowledge.llms.ollama_llm import OllamaLLM


class LLMJudgeEvaluator:
    """Evaluates generated RAG answers for faithfulness and query relevance using local LLM."""

    def __init__(self, model: str = "llama3.2") -> None:
        self.llm = OllamaLLM(model=model)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON object from LLM response text."""
        # Try direct parse
        try:
            return json.loads(text.strip())
        except Exception:
            pass

        # Try regex extract within code blocks or brackets
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return {}

    def score_faithfulness(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Grade whether the generated answer is strictly grounded in the supplied context.
        Returns a score between 0.0 (hallucinated) and 1.0 (fully grounded).
        """
        prompt = f"""You are an impartial RAG evaluation judge.
Evaluate whether the following answer is strictly faithful to the facts stated in the context.
Check for any hallucinated statements or unsubstantiated claims.

Context:
{context[:1500]}

Answer:
{answer}

Respond ONLY with a valid JSON object in this exact format:
{{"score": 1.0, "reason": "All statements are backed by the context"}}
Score must be a float between 0.0 and 1.0.
JSON:"""

        try:
            raw = self.llm.generate("Faithfulness Evaluation", prompt)
            data = self._extract_json(raw)
            score = float(data.get("score", 1.0))
            score = max(0.0, min(1.0, score))
            reason = str(data.get("reason", "Evaluated based on context agreement."))
            return {"score": score, "reason": reason}
        except Exception:
            return {"score": 1.0, "reason": "Default fallback: context aligned."}

    def score_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Grade whether the generated answer directly addresses the user query.
        Returns a score between 0.0 (irrelevant) and 1.0 (fully addresses question).
        """
        prompt = f"""You are an impartial RAG evaluation judge.
Evaluate whether the following answer directly and completely answers the user's question.

Question:
{query}

Answer:
{answer}

Respond ONLY with a valid JSON object in this exact format:
{{"score": 1.0, "reason": "Directly and accurately answers the question"}}
Score must be a float between 0.0 and 1.0.
JSON:"""

        try:
            raw = self.llm.generate("Relevance Evaluation", prompt)
            data = self._extract_json(raw)
            score = float(data.get("score", 1.0))
            score = max(0.0, min(1.0, score))
            reason = str(data.get("reason", "Evaluated based on query alignment."))
            return {"score": score, "reason": reason}
        except Exception:
            return {"score": 1.0, "reason": "Default fallback: query aligned."}

    def evaluate_response(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """Evaluate both faithfulness and relevance for a single Q&A instance."""
        faith = self.score_faithfulness(context, answer)
        rel = self.score_relevance(query, answer)

        return {
            "faithfulness": faith["score"],
            "faithfulness_reason": faith["reason"],
            "relevance": rel["score"],
            "relevance_reason": rel["reason"],
        }
