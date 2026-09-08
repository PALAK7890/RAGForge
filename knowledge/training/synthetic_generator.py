"""
Synthetic training data generator for retrieval fine-tuning.
Uses local LLM to generate realistic query-document pairs from indexed corpus chunks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from knowledge.llms.ollama_llm import OllamaLLM


class SyntheticDataGenerator:
    """Generates synthetic (query, document) training pairs from indexed chunks."""

    def __init__(self, chunks_path: str = ".knowledge/index/chunks.json") -> None:
        self.chunks_path = Path(chunks_path)
        self.llm = OllamaLLM()

    def generate_pairs(
        self,
        num_pairs: int = 20,
        output_path: str = ".knowledge/training_pairs.json",
    ) -> List[Dict[str, str]]:
        """Generate query-chunk pairs and persist them to disk."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {self.chunks_path}. Index documents first.")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        if not chunks:
            raise ValueError("No indexed chunks found to generate training pairs from.")

        pairs: List[Dict[str, str]] = []
        selected_chunks = chunks[:num_pairs]

        for i, chunk in enumerate(selected_chunks):
            chunk_text = chunk["text"].strip()
            if len(chunk_text) < 50:
                continue

            prompt = f"""You are generating training queries for a semantic search engine.
Read this passage and write exactly ONE natural question that someone would search to find this passage.
Output ONLY the question itself, with no intro, quotation marks, or explanations.

Passage:
{chunk_text[:600]}

Question:"""

            try:
                query = self.llm.generate("Synthetic query generator", prompt).strip().strip('"').strip("'")
                if query and len(query) > 10:
                    pairs.append({
                        "query": query,
                        "positive": chunk_text,
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                        "source_path": chunk.get("source_path", ""),
                    })
            except Exception:
                # Fallback heuristic question from first sentence if LLM offline
                first_sentence = chunk_text.split(".")[0].strip()
                pairs.append({
                    "query": f"What is {first_sentence[:40]}?",
                    "positive": chunk_text,
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "source_path": chunk.get("source_path", ""),
                })

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        return pairs
