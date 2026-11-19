"""
Ollama LLM wrapper with batch and streaming generation.
"""

from typing import Iterator

import ollama


class OllamaLLM:
    """Simple wrapper around a local Ollama model."""

    def __init__(self, model: str = "llama3.2") -> None:
        self.model = model

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a Retrieval-Augmented Generation assistant.

Answer ONLY from the supplied context.

When the context supplies bracketed source labels, cite each factual claim using
the supporting source label exactly as supplied. Do not invent filenames, page
numbers, or character offsets. Place citations next to the claims they support.

If the answer is not present, reply:
"I could not find the answer in the indexed documents."

Context:
{context}

Question:
{question}
"""

    def generate(self, question: str, context: str) -> str:
        """Generate a complete text response."""
        prompt = self._build_prompt(question, context)
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return str(response["message"]["content"])

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        """Stream response tokens as they arrive."""
        prompt = self._build_prompt(question, context)
        response_stream = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
        )
        for chunk in response_stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
