"""Incremental indexing must preserve unrelated documents and vector alignment."""

from pathlib import Path

from knowledge.api.service import RAGService


def test_incremental_index_preserves_old_docs_and_custom_workspace(tmp_path):
    class Embedder:
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    service = RAGService(str(tmp_path / "workspace"))
    service._embedder = Embedder()
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("First document retention is seven years.")
    second.write_text("Second document revenue is four million.")
    service.index_documents(str(first))
    service.index_documents(str(second))
    first.write_text("Updated first document retention is eight years.")
    service.index_documents(str(first))
    chunks, _, vector_store = service._load_search_context()
    assert {Path(c["source_path"]).name for c in chunks} == {"first.txt", "second.txt"}
    assert not any("seven" in c["text"] for c in chunks)
    assert any("eight" in c["text"] for c in chunks)
    assert vector_store.index.ntotal == len(chunks)
    assert (service.index_dir / "manifest.json").exists()
