"""
Configuration manager module for KnowledgeOS.
Handles reading and writing configurations from YAML.
"""

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Configuration model for the KnowledgeOS application."""
    
    # Embedding settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="The sentence-transformers model to use for generating embeddings"
    )
    
    # Chunking settings
    chunk_size: int = Field(
        default=500,
        description="The maximum size of each text chunk (in characters)"
    )
    chunk_overlap: int = Field(
        default=50,
        description="The overlap size between consecutive chunks"
    )
    chunking_strategy: str = Field(
        default="recursive",
        description="The chunking strategy: recursive, sentence, or fixed"
    )
    
    # Vector DB settings
    vector_db_path: str = Field(
        default="data/vectorstore",
        description="Directory path to save the FAISS vector database index"
    )
    
    # LLM Settings
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama' or 'openai'"
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        description="The Ollama model to query"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="The OpenAI model to query"
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature for the LLM responses"
    )
    
    # Retrieval Settings
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve"
    )
    reranking_enabled: bool = Field(
        default=False,
        description="Whether to enable cross-encoder reranking"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="SentenceTransformer CrossEncoder model for reranking"
    )
    grounding_threshold: float = Field(
        default=0.3,
        description="Cosine similarity threshold for grounding check (rejection)"
    )
    
    # Evaluation settings
    eval_judge_enabled: bool = Field(
        default=False,
        description="Whether to use LLM-as-judge during evaluation"
    )

    # Agent settings
    max_retrieval_attempts: int = Field(
        default=3,
        description="Maximum number of retrieve-grade-rewrite cycles before the agent forces generation"
    )


class ConfigManager:
    """Manages reading, writing, and accessing YAML-based configurations."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = Path(config_path)
        self.current_config = self.load()

    def load(self) -> AppConfig:
        """Loads configuration from YAML file or returns defaults if file doesn't exist."""
        if not self.config_path.exists():
            return AppConfig()
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return AppConfig(**data)
        except Exception:
            # Fallback to default config on corrupted yaml/read errors
            return AppConfig()

    def save(self, config: AppConfig) -> None:
        """Saves current configuration to the config YAML file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # Using model_dump (Pydantic v2 compatible)
        data = config.model_dump()
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        self.current_config = config

    def get(self) -> AppConfig:
        """Returns the current loaded configuration."""
        return self.current_config

    def update(self, updates: Dict[str, Any]) -> AppConfig:
        """Updates specific configuration values and saves them."""
        # Merge changes with current config dict
        current_data = self.current_config.model_dump()
        current_data.update(updates)
        
        # Re-validate with Pydantic
        new_config = AppConfig(**current_data)
        self.save(new_config)
        return new_config
