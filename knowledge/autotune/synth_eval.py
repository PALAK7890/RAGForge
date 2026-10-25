"""
Synthetic evaluation set generation for knowledge autotune.

``SynthEvalGenerator`` builds a question-answer evaluation set from indexed
chunks without requiring any human labels.  It uses an injected ``LLMInterface``
so tests can pass a canned stub instead of a real Ollama instance.

Gold labels are *source spans* — ``(source_path, char_start, char_end)`` —
rather than chunk IDs, so they remain stable across chunking strategy changes.

Output schema (list of ``EvalSample``)
---------------------------------------
::

    {
      "question":    str,            # synthetic question
      "source_path": str,            # absolute or relative path to origin file
      "char_start":  int | None,     # character offset in the source page text
      "char_end":    int | None,     # exclusive end offset
      "gold_text":   str,            # the exact passage the question answers
    }
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

# ---------------------------------------------------------------------------
# LLM interface (injectable for testing)
# ---------------------------------------------------------------------------


class LLMInterface(Protocol):
    """Structural protocol satisfied by ``OllamaLLM`` (and test stubs).

    Only the ``generate`` method is required; the signature matches
    ``OllamaLLM.generate(question, context) -> str``.
    """

    def generate(self, question: str, context: str) -> str:
        """Return a generated string.  ``question`` is used as a system prompt
        or label; ``context`` is the main content."""
        ...


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSample:
    """A single evaluation question with a source-span gold label.

    ``char_start`` / ``char_end`` are offsets into the *stripped* source page
    text (``LoadedDocument.text.strip()``), exactly as stored in
    ``DocumentChunk.char_start`` / ``char_end``.  Both are ``None`` when the
    originating chunk did not carry offset information.
    """

    question: str
    source_path: str
    char_start: Optional[int]
    char_end: Optional[int]
    gold_text: str

    def has_span(self) -> bool:
        """Return True when source-span offsets are available."""
        return self.char_start is not None and self.char_end is not None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "source_path": self.source_path,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "gold_text": self.gold_text,
        }


@dataclass
class EvalSet:
    """Paired train / held-out split of ``EvalSample`` instances."""

    train: List[EvalSample]
    holdout: List[EvalSample]

    @property
    def all_samples(self) -> List[EvalSample]:
        return self.train + self.holdout

    def __len__(self) -> int:
        return len(self.train) + len(self.holdout)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SynthEvalGenerator:
    """Generates a synthetic evaluation set from indexed document chunks.

    Args:
        llm:           An ``LLMInterface`` instance (real or stub).
        n:             Total number of eval samples to attempt to generate.
        holdout_ratio: Fraction of samples reserved for holdout (default 0.2).
        seed:          RNG seed for deterministic sampling and splitting.
        min_chunk_len: Skip chunks shorter than this many characters.
    """

    _PROMPT_TEMPLATE = (
        "You are generating evaluation questions for a semantic search engine.\n"
        "Read this passage and write exactly ONE natural question that a user would\n"
        "search to find this passage. Output ONLY the question itself, with no intro,\n"
        "quotation marks, or explanations.\n\n"
        "Passage:\n{passage}\n\nQuestion:"
    )

    def __init__(
        self,
        llm: LLMInterface,
        n: int = 30,
        holdout_ratio: float = 0.2,
        seed: int = 42,
        min_chunk_len: int = 80,
    ) -> None:
        if n < 2:
            raise ValueError(f"n must be >= 2 to allow a train/holdout split, got {n}")
        if not 0.0 < holdout_ratio < 1.0:
            raise ValueError(f"holdout_ratio must be in (0, 1), got {holdout_ratio}")
        self.llm = llm
        self.n = n
        self.holdout_ratio = holdout_ratio
        self.seed = seed
        self.min_chunk_len = min_chunk_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, chunks: Sequence[dict]) -> EvalSet:
        """Generate an ``EvalSet`` from a sequence of chunk dicts.

        Each element of *chunks* must be a dict with at least:
        - ``"text"`` (str)
        - ``"source_path"`` (str)
        - optionally ``"char_start"`` (int | None) and ``"char_end"`` (int | None)

        This is the schema stored in ``.knowledge/index/chunks.json``.
        Old-format dicts without ``char_start``/``char_end`` are supported;
        the resulting ``EvalSample`` will have ``has_span() == False``.

        Args:
            chunks: Sequence of chunk dicts (from chunks.json).

        Returns:
            An ``EvalSet`` with ``train`` and ``holdout`` splits.

        Raises:
            ValueError: If there are no chunks long enough to generate from.
        """
        eligible = self._filter_chunks(chunks)
        if not eligible:
            raise ValueError(
                f"No chunks with at least {self.min_chunk_len} characters found. "
                "Index more documents before running autotune."
            )

        selected = self._sample_chunks(eligible)
        samples = self._generate_samples(selected)
        return self._split(samples)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_chunks(self, chunks: Sequence[dict]) -> List[dict]:
        """Return chunks long enough for meaningful question generation."""
        return [c for c in chunks if len(c.get("text", "")) >= self.min_chunk_len]

    def _sample_chunks(self, eligible: List[dict]) -> List[dict]:
        """Select up to ``self.n`` chunks, deterministically shuffled."""
        rng = random.Random(self.seed)
        pool = list(eligible)
        rng.shuffle(pool)
        return pool[: self.n]

    def _generate_samples(self, chunks: List[dict]) -> List[EvalSample]:
        """Generate one ``EvalSample`` per chunk; fall back to heuristic on error."""
        samples: List[EvalSample] = []
        for chunk in chunks:
            text = chunk["text"].strip()
            question = self._ask_llm(text)
            samples.append(
                EvalSample(
                    question=question,
                    source_path=str(chunk.get("source_path", "")),
                    char_start=chunk.get("char_start"),
                    char_end=chunk.get("char_end"),
                    gold_text=text,
                )
            )
        return samples

    def _ask_llm(self, passage: str) -> str:
        """Call the LLM to generate a question; return a heuristic on failure."""
        prompt = self._PROMPT_TEMPLATE.format(passage=passage[:600])
        try:
            raw = self.llm.generate("Synthetic eval question generator", prompt)
            question = raw.strip().strip('"').strip("'")
            if len(question) > 10:
                return question
        except Exception:
            pass
        # Heuristic fallback: "What is <first 40 chars of first sentence>?"
        first = passage.split(".")[0].strip()
        return f"What is {first[:40]}?"

    def _split(self, samples: List[EvalSample]) -> EvalSet:
        """Deterministically split samples into train / holdout."""
        rng = random.Random(self.seed + 1)  # different seed from sampling
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n_holdout = max(1, round(len(shuffled) * self.holdout_ratio))
        return EvalSet(
            holdout=shuffled[:n_holdout],
            train=shuffled[n_holdout:],
        )
