"""
Measure cold vs warm single-query retrieval latency in one process.

Cold = SentenceTransformer + CrossEncoder loaded from disk + first query.
Warm = second query with models already in memory.

Usage (from repo root):
    HF_HUB_OFFLINE=1 python scripts/measure_latency.py

Requires a built index (.knowledge/index/) and config (.knowledge/config.yaml).
"""

import json
import sys
import time

sys.path.insert(0, ".")

from knowledge.config.manager import ConfigManager
from knowledge.embeddings.sentence_transformer import SentenceTransformerEmbedding
from knowledge.rerankers.cross_encoder import CrossEncoderReranker
from knowledge.retrievers.bm25 import BM25Retriever
from knowledge.retrievers.hybrid import reciprocal_rank_fusion
from knowledge.vectorstores.faiss_store import FAISSStore

cfg = ConfigManager(".knowledge/config.yaml").get()

with open(".knowledge/index/chunks.json") as f:
    chunks = json.load(f).get("chunks", [])
texts_corpus = [c["text"] for c in chunks]

# --- COLD: load all models + build BM25 + first query ---
t0 = time.perf_counter()

embedder = SentenceTransformerEmbedding(cfg.embedding_model)
store = FAISSStore()
store.load(".knowledge/index/faiss.index")
bm25 = BM25Retriever()
bm25.build(texts_corpus)
reranker = CrossEncoderReranker(cfg.reranker_model)

query = "What is KnowledgeOS?"
qvec = embedder.embed_query(query)
sc, idx = store.search(qvec, top_k=cfg.top_k)
faiss_res = list(zip(idx[0].tolist(), sc[0].tolist()))
bm25_res = bm25.search(query, top_k=cfg.top_k)
fused = reciprocal_rank_fusion(faiss_res, bm25_res)
pool = [chunks[i] for i, _ in fused[: cfg.top_k] if 0 <= i < len(chunks)]
reranker.rerank(query, pool)

cold_ms = (time.perf_counter() - t0) * 1000
print(f"Cold (includes model load): {cold_ms:.1f} ms")

# --- WARM: second query, models already in memory ---
t1 = time.perf_counter()

query2 = "What does this project demonstrate?"
qvec2 = embedder.embed_query(query2)
sc2, idx2 = store.search(qvec2, top_k=cfg.top_k)
faiss_res2 = list(zip(idx2[0].tolist(), sc2[0].tolist()))
bm25_res2 = bm25.search(query2, top_k=cfg.top_k)
fused2 = reciprocal_rank_fusion(faiss_res2, bm25_res2)
pool2 = [chunks[i] for i, _ in fused2[: cfg.top_k] if 0 <= i < len(chunks)]
reranker.rerank(query2, pool2)

warm_ms = (time.perf_counter() - t1) * 1000
print(f"Warm (models already loaded): {warm_ms:.1f} ms")
