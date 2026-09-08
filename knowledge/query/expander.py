"""
Simple query expansion using domain-specific synonyms.
"""



class QueryExpander:

    def __init__(self) -> None:

        self.synonyms = {
            "rag": [
                "retrieval augmented generation",
                "semantic search",
                "document retrieval",
            ],
            "vector": [
                "embedding",
                "faiss",
                "vector database",
            ],
            "llm": [
                "language model",
                "large language model",
                "ollama",
            ],
            "embedding": [
                "sentence transformer",
                "semantic embedding",
            ],
            "pdf": [
                "document",
                "file",
            ],
        }

    def expand(self, query: str) -> str:

        expanded = [query]

        words = query.lower().split()

        for word in words:

            if word in self.synonyms:
                expanded.extend(self.synonyms[word])

        return " ".join(expanded)