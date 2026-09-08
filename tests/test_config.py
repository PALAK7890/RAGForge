"""
Unit tests for the configuration manager.
"""

import os
import tempfile

from knowledge.config.manager import ConfigManager


def test_default_config() -> None:
    """Test that default configuration is loaded when no file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        manager = ConfigManager(config_path=config_path)
        
        config = manager.get()
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.llm_provider == "ollama"
        assert config.vector_db_path == "data/vectorstore"


def test_save_and_load_config() -> None:
    """Test saving config to a file and reloading it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        manager = ConfigManager(config_path=config_path)
        
        # Modify some values
        config = manager.get()
        config.embedding_model = "large-model"
        config.chunk_size = 1000
        config.top_k = 10
        
        manager.save(config)
        
        # Load in another manager instance to verify persistence
        new_manager = ConfigManager(config_path=config_path)
        loaded_config = new_manager.get()
        
        assert loaded_config.embedding_model == "large-model"
        assert loaded_config.chunk_size == 1000
        assert loaded_config.top_k == 10


def test_update_config() -> None:
    """Test the update method of the config manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        manager = ConfigManager(config_path=config_path)
        
        manager.update({"chunk_size": 800, "llm_provider": "openai"})
        
        loaded_config = manager.get()
        assert loaded_config.chunk_size == 800
        assert loaded_config.llm_provider == "openai"
        # Make sure other defaults are preserved
        assert loaded_config.embedding_model == "all-MiniLM-L6-v2"
