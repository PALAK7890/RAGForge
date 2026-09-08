"""
Core service layer exposing RAGForge pipeline operations.
Used by both the CLI commands and FastAPI endpoints.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
import yaml

from knowledge.chunkers.recursive import RecursiveChunker
from knowledge.embeddings.sentence_transformer import SentenceTransformerEmbedding
from knowledge.evaluation.evaluator import RetrievalEvaluator
from knowledge.indexing.document_registry import DocumentRegistry
from knowledge.llms.ollama_llm import OllamaLLM
from knowledge.loaders.router import LoaderRouter
from knowledge.query.expander import QueryExpander
from knowledge.rerankers.cross_encoder import CrossEncoderReranker
from knowledge.retrievers.bm25 import BM25Retriever
from knowledge.retrievers.hybrid import HybridRetriever
from knowledge.vectorstores.faiss_store import FAISSStore


class RAGService:
    """Unified service interface for RAGForge operations."""

    def __init__(self, workspace_path: str = ".knowledge") -> None:
        self.workspace = Path(workspace_path)
        self.index_dir = self.workspace / "index"
        self.faiss_path = str(self.index_dir / "faiss.index")
        self.chunks_path = self.index_dir / "chunks.json"
        self.config_path = self.workspace / "config.yaml"

        self.router = LoaderRouter()
        self.chunker = RecursiveChunker()
        self.expander = QueryExpander()

        self._embedder: Optional[SentenceTransformerEmbedding] = None
        self._llm: Optional[OllamaLLM] = None
        self._reranker: Optional[CrossEncoderReranker] = None

    @property
    def embedder(self) -> SentenceTransformerEmbedding:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedding()
        return self._embedder

    @property
    def llm(self) -> OllamaLLM:
        if self._llm is None:
            self._llm = OllamaLLM()
        return self._llm

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    # -------------------------------------------------------------------------
    # Document Indexing & Management
    # -------------------------------------------------------------------------

    def index_documents(self, path_str: str) -> Dict[str, Any]:
        """Load, chunk, embed and index documents from a file or folder."""
        source = Path(path_str)
        if not source.exists():
            raise FileNotFoundError(f"Path not found: {path_str}")

        files = [source] if source.is_file() else [f for f in source.rglob("*") if f.is_file()]
        if not files:
            return {"indexed": 0, "message": "No files found to index."}

        registry = DocumentRegistry()
        vector_store = FAISSStore()
        bm25 = BM25Retriever()

        all_chunks: List[Any] = []
        all_embeddings: List[List[float]] = []
        indexed_count = 0

        for file in files:
            status = registry.get_status(file)
            if status == "UNCHANGED":
                continue

            try:
                loader = self.router.get_loader(file)
                loaded_docs = loader.load(file)
                chunks = self.chunker.chunk(loaded_docs, chunk_size=500, chunk_overlap=100)
                texts = [chunk.text for chunk in chunks]

                embeddings = self.embedder.embed_documents(texts)
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)
                registry.add_document(file, len(chunks))
                indexed_count += 1
            except Exception as e:
                continue

        if not all_embeddings:
            return {"indexed": 0, "message": "All documents are already indexed and unchanged."}

        # Save to vector store
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vector_store.add(all_embeddings)
        vector_store.save(self.faiss_path)

        document_records = [{"name": file.name, "path": str(file.resolve())} for file in files]
        chunk_records = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_id": chunk.document_id,
                "source_path": chunk.source_path,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in all_chunks
        ]

        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump({"documents": document_records, "chunks": chunk_records}, f, indent=2, ensure_ascii=False)

        registry.save()

        return {
            "indexed_files": indexed_count,
            "total_chunks": len(all_chunks),
            "embedding_dim": len(all_embeddings[0]) if all_embeddings else 0,
        }

    def list_documents(self) -> Dict[str, Any]:
        """List all currently indexed documents."""
        if not self.chunks_path.exists():
            return {"documents": [], "total_documents": 0, "total_chunks": 0}

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = data.get("documents", [])
        chunks = data.get("chunks", [])

        doc_list = []
        for doc in documents:
            chunk_count = sum(1 for c in chunks if Path(c.get("source_path", "")).name == doc["name"])
            doc_list.append({"name": doc["name"], "path": doc["path"], "chunks": chunk_count})

        return {
            "documents": doc_list,
            "total_documents": len(documents),
            "total_chunks": len(chunks),
        }

    # -------------------------------------------------------------------------
    # Search & Retrieval
    # -------------------------------------------------------------------------

    def _load_search_context(self) -> Tuple[List[Dict[str, Any]], BM25Retriever, FAISSStore]:
        """Helper to load chunks, BM25, and FAISS index."""
        if not self.chunks_path.exists() or not Path(self.faiss_path).exists():
            raise FileNotFoundError("Index not found. Please initialize and index documents first.")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks", [])
        texts = [chunk["text"] for chunk in chunks]

        bm25 = BM25Retriever()
        bm25.build(texts)

        vector_store = FAISSStore()
        vector_store.load(self.faiss_path)

        return chunks, bm25, vector_store

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run hybrid retrieval (FAISS + BM25) and Cross-Encoder reranking."""
        chunks, bm25, vector_store = self._load_search_context()
        expanded_query = self.expander.expand(query)

        query_embedding = self.embedder.embed_query(query)
        scores, indices = vector_store.search(query_embedding, top_k=20)

        faiss_results: List[Tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                faiss_results.append((int(idx), float(score)))

        bm25_results = bm25.search(expanded_query, top_k=20)

        hybrid = HybridRetriever()
        fused = hybrid.fuse(faiss_results, bm25_results)

        candidate_chunks = []
        for idx, hybrid_score in fused:
            if idx < len(chunks):
                chunk_copy = chunks[idx].copy()
                chunk_copy["hybrid_score"] = float(hybrid_score)
                candidate_chunks.append(chunk_copy)

        ranked = self.reranker.rerank(query=query, chunks=candidate_chunks, top_k=top_k)
        return ranked

    # -------------------------------------------------------------------------
    # Question Answering
    # -------------------------------------------------------------------------

    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform end-to-end RAG question answering."""
        ranked = self.search(question, top_k=top_k)
        context = "\n\n".join(chunk["text"] for chunk in ranked)

        answer = self.llm.generate(question, context)
        sources = list(dict.fromkeys(Path(c["source_path"]).name for c in ranked))

        formatted_chunks = [
            {
                "chunk_id": c.get("chunk_id"),
                "text": c["text"],
                "source_path": c["source_path"],
                "page_number": c.get("page_number"),
                "cross_score": c.get("rerank_score"),
                "hybrid_score": c.get("hybrid_score"),
            }
            for c in ranked
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks": formatted_chunks,
        }

    def ask_stream(self, question: str, top_k: int = 5) -> Tuple[List[str], List[Dict[str, Any]], Iterator[str]]:
        """Perform RAG search and stream answer tokens."""
        ranked = self.search(question, top_k=top_k)
        context = "\n\n".join(chunk["text"] for chunk in ranked)
        sources = list(dict.fromkeys(Path(c["source_path"]).name for c in ranked))

        formatted_chunks = [
            {
                "chunk_id": c.get("chunk_id"),
                "text": c["text"],
                "source_path": c["source_path"],
                "page_number": c.get("page_number"),
                "cross_score": c.get("rerank_score"),
                "hybrid_score": c.get("hybrid_score"),
            }
            for c in ranked
        ]

        token_generator = self.llm.generate_stream(question, context)
        return sources, formatted_chunks, token_generator

    # -------------------------------------------------------------------------
    # Stats, Benchmark, Evaluate
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get index and workspace statistics."""
        chunks_info = self.list_documents()
        embedding_model = "all-MiniLM-L6-v2"
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                embedding_model = cfg.get("embedding_model", embedding_model)

        dim = "Unknown"
        index_size = "0 KB"
        if Path(self.faiss_path).exists():
            index_size = f"{os.path.getsize(self.faiss_path) / 1024:.2f} KB"
            try:
                vs = FAISSStore()
                vs.load(self.faiss_path)
                if vs.index is not None:
                    dim = str(vs.index.d)
            except Exception:
                pass

        chunk_size = f"{os.path.getsize(self.chunks_path) / 1024:.2f} KB" if self.chunks_path.exists() else "0 KB"

        return {
            "workspace": str(self.workspace.resolve()),
            "documents": chunks_info["total_documents"],
            "chunks": chunks_info["total_chunks"],
            "embedding_model": embedding_model,
            "embedding_dim": dim,
            "vector_store": "FAISS",
            "index_size": index_size,
            "chunk_file_size": chunk_size,
        }

    def benchmark(self, query: str) -> Dict[str, Any]:
        """Benchmark latency across every pipeline stage."""
        total_start = time.perf_counter()
        chunks, bm25, vector_store = self._load_search_context()
        expanded_query = self.expander.expand(query)

        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(expanded_query)
        embedding_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        scores, indices = vector_store.search(query_embedding, top_k=20)
        faiss_ms = (time.perf_counter() - t0) * 1000

        faiss_results = [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]

        t0 = time.perf_counter()
        bm25_results = bm25.search(expanded_query, top_k=20)
        bm25_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hybrid = HybridRetriever()
        fused = hybrid.fuse(faiss_results, bm25_results)
        hybrid_ms = (time.perf_counter() - t0) * 1000

        candidate_chunks = []
        for idx, score in fused:
            if idx < len(chunks):
                chunk = chunks[idx].copy()
                chunk["hybrid_score"] = float(score)
                candidate_chunks.append(chunk)

        t0 = time.perf_counter()
        ranked = self.reranker.rerank(query, candidate_chunks, top_k=5)
        rerank_ms = (time.perf_counter() - t0) * 1000

        context = "\n\n".join(c["text"] for c in ranked)
        t0 = time.perf_counter()
        answer = self.llm.generate(query, context)
        llm_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "query": query,
            "embedding_ms": embedding_ms,
            "faiss_ms": faiss_ms,
            "bm25_ms": bm25_ms,
            "hybrid_ms": hybrid_ms,
            "rerank_ms": rerank_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
            "answer": answer,
        }

    def evaluate(self, dataset_path: str) -> Dict[str, Any]:
        """Run evaluation metrics on benchmark dataset."""
        evaluator = RetrievalEvaluator()
        return evaluator.evaluate(dataset_path)
