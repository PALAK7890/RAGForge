"""Frozen fictional benchmark v2: source-grounded retrieval and optional generation."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from knowledge.agent.graph import run_agent_workflow
from knowledge.agentic.citations import CitationValidator
from knowledge.agentic.controller import AgenticController
from knowledge.agentic.schemas import EvidenceRecord
from knowledge.agentic.tools import FindTool, OpenTool, SearchTool
from knowledge.api.service import RAGService
from knowledge.llms.ollama_llm import OllamaLLM

ROOT = Path(__file__).resolve().parents[2] / "examples" / "benchmark_v2"
COUNTS = dict(
    factual=20,
    exact_keyword=15,
    multihop=20,
    comparative=15,
    numerical=15,
    document_navigation=15,
    ambiguous=10,
    unanswerable=10,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_benchmark(root=ROOT):
    root = Path(root)
    data = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    if data["version"] != "2.0.0" or not data["synthetic"]:
        raise ValueError("Unsupported benchmark version")
    questions, passages, documents = data["questions"], data["passages"], data["documents"]
    if Counter(q["category"] for q in questions) != COUNTS:
        raise ValueError("Category counts do not match v2")
    if Counter(q["split"] for q in questions) != {"dev": 62, "held_out": 58}:
        raise ValueError("Split counts do not match v2")
    if len({q["id"] for q in questions}) != 120 or len({q["question"] for q in questions}) != 120:
        raise ValueError("Duplicate question ID or wording")
    texts = {}
    for name, spec in documents.items():
        if Path(name).name != name or spec["split"] not in {"dev", "held_out"}:
            raise ValueError("Unsafe document name or split")
        path = root / "corpus" / name
        if sha(path) != spec["sha256"]:
            raise ValueError(f"Corpus fingerprint mismatch: {name}")
        texts[name] = path.read_text(encoding="utf-8")
    if set(p.name for p in (root / "corpus").glob("*.md")) != set(documents):
        raise ValueError("Unexpected corpus documents")
    for pid, passage in passages.items():
        name = passage["document"]
        if passage["split"] != documents[name]["split"]:
            raise ValueError("Passage split mismatch")
        start, end = passage["char_start"], passage["char_end"]
        if not 0 <= start < end <= len(texts[name]) or texts[name][start:end] != passage["quote"]:
            raise ValueError(f"Stale gold passage: {pid}")
    for q in questions:
        groups = q["evidence_groups"]
        if (
            q["expected_behavior"] not in {"answer", "clarify", "abstain"}
            or not q["expected_answer"]
        ):
            raise ValueError("Invalid reference behavior or answer")
        if q["is_unanswerable"] != (q["expected_behavior"] == "abstain"):
            raise ValueError("Abstention label mismatch")
        if q["is_unanswerable"] != (not groups):
            raise ValueError("Answerability/evidence mismatch")
        refs = [pid for group in groups for pid in group]
        if any(not group for group in groups) or len(set(refs)) != len(refs):
            raise ValueError("Empty or duplicate evidence group")
        for pid in refs:
            if passages[pid]["split"] != q["split"]:
                raise ValueError("Question crosses document-family split")
        if q["category"] == "multihop" and len({passages[p]["document"] for p in refs}) < 2:
            raise ValueError("Multi-hop questions must use multiple documents")
    return data


def unique_chunks(chunks):
    seen, output = set(), []
    for chunk in chunks:
        key = (chunk["source_path"], chunk.get("chunk_id"), chunk["text"])
        if key not in seen:
            seen.add(key)
            output.append(chunk)
    return output


def covered_passages(chunks, passages):
    return {
        pid
        for pid, p in passages.items()
        if any(
            Path(c["source_path"]).name == p["document"] and p["quote"] in c["text"] for c in chunks
        )
    }


def group_recall(question, hits):
    groups = question["evidence_groups"]
    return sum(bool(set(group) & hits) for group in groups) / len(groups) if groups else None


def retrieval_scores(question, ranked, indexed, passages, k):
    """Recall uses gold passage requirements; nDCG/precision use binary chunk relevance."""
    if not question["evidence_groups"]:
        return {
            name: None
            for name in (
                "passage_recall_at_k",
                "all_requirements_at_k",
                "mrr_at_k",
                "precision_at_k",
                "ndcg_at_k",
                "all_evidence_group_recall",
            )
        }
    ranked = unique_chunks(ranked)
    refs = {pid: passages[pid] for group in question["evidence_groups"] for pid in group}
    top = ranked[:k]
    relevance = [bool(covered_passages([c], refs)) for c in top]
    recall = group_recall(question, covered_passages(top, refs))
    relevant_count = sum(bool(covered_passages([c], refs)) for c in unique_chunks(indexed))
    dcg = sum(int(hit) / math.log2(i + 2) for i, hit in enumerate(relevance))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, relevant_count)))
    return dict(
        passage_recall_at_k=recall,
        all_requirements_at_k=float(recall == 1),
        mrr_at_k=next((1 / (i + 1) for i, hit in enumerate(relevance) if hit), 0.0),
        precision_at_k=sum(relevance) / k,
        ndcg_at_k=dcg / ideal if ideal else 0.0,
        all_evidence_group_recall=group_recall(question, covered_passages(ranked, refs)),
    )


def abstains(answer):
    return any(
        s in answer.lower()
        for s in (
            "cannot answer",
            "cannot determine",
            "not specified",
            "not provided",
            "not disclosed",
            "could not find the answer",
            "insufficient information",
            "not enough information",
            "no information",
            "not available",
        )
    )


def generation_scores(question, output, chunks):
    answer = output.get("answer", "")
    failed = bool(output.get("error")) or not answer.strip()
    records = [
        EvidenceRecord(
            document_id=Path(c["source_path"]).name,
            source_path=c["source_path"],
            text=c["text"],
            page_number=c.get("page_number"),
            char_start=(c.get("metadata") or {}).get("char_start", c.get("char_start")),
            char_end=(c.get("metadata") or {}).get("char_end", c.get("char_end")),
        )
        for c in chunks
    ]
    report = CitationValidator(records).validate(answer)
    normal = lambda s: " ".join(re.findall(r"\w+", s.lower()))
    return dict(
        exact_answer_match_proxy=float(
            not failed and normal(answer) == normal(question["expected_answer"])
        ),
        abstention_accuracy_heuristic=float(
            not failed
            and bool(output.get("abstained") or abstains(answer)) == question["is_unanswerable"]
        ),
        citation_reference_validity=report["citation_integrity"]
        if report["total_citations"] and not failed
        else None,
        citation_count=report["total_citations"],
        answer_correctness=None,
        faithfulness=None,
        citation_support=None,
    )


GENERATION_OPTIONS = {"temperature": 0, "seed": 17, "num_ctx": 8192, "num_predict": 1024}


class BenchmarkLLM(OllamaLLM):
    """One bounded, explicit generation configuration shared by all three systems."""

    def generate(self, question, context):
        import ollama

        response = ollama.Client(timeout=120).chat(
            model=self.model,
            messages=[{"role": "user", "content": self._build_prompt(question, context)}],
            options=GENERATION_OPTIONS,
        )
        answer = str(response["message"]["content"]).strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty answer")
        return answer


class BenchmarkService(RAGService):
    """Cache identical retrieval context for all benchmark systems in one process."""

    def _load_search_context(self):
        if not hasattr(self, "_benchmark_context"):
            self._benchmark_context = super()._load_search_context()
        return self._benchmark_context


def validate_index(service, data, root=ROOT):
    indexed = json.loads(service.chunks_path.read_text(encoding="utf-8"))["chunks"]
    by_name = {name: [] for name in data["documents"]}
    for c in indexed:
        name = Path(c["source_path"]).name
        if (
            name not in by_name
            or Path(c["source_path"]).resolve() != (root / "corpus" / name).resolve()
        ):
            raise ValueError("Index contains unrelated sources; choose a fresh v2 workspace")
        source = (root / "corpus" / name).read_text(encoding="utf-8")
        if not c["text"] or c["text"] not in source:
            raise ValueError("Stale or noncontiguous index chunk; rebuild in a fresh workspace")
        by_name[name].append(c)
    if not all(by_name.values()):
        raise ValueError("Index is missing benchmark documents")
    needed = {
        pid: data["passages"][pid]
        for q in data["questions"]
        for group in q["evidence_groups"]
        for pid in group
    }
    missing = set(needed) - covered_passages(indexed, needed)
    if missing:
        raise ValueError(
            "Gold passages are not fully represented in chunks: " + ", ".join(sorted(missing))
        )
    return indexed


def ollama_ready(model):
    import ollama

    try:
        models = ollama.Client(timeout=5).list().models
        names = {m.model for m in models}
        if model not in names and model + ":latest" not in names:
            return f"Ollama model {model} is not installed"
    except Exception as exc:
        return f"Ollama unavailable: {type(exc).__name__}: {exc}"
    return None


def summarize(records):
    result = {}
    for system in sorted({r["system"] for r in records}):
        rows = [r for r in records if r["system"] == system]
        names = {key for row in rows for key in row["metrics"]}
        aggregates = {}
        for key in sorted(names):
            values = [r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]
            aggregates[key] = {
                "mean": sum(values) / len(values) if values else None,
                "n": len(values),
            }
        result[system] = dict(
            queries=len(rows), errors=sum(bool(r.get("error")) for r in rows), metrics=aggregates
        )
    return result


def run(
    split="dev",
    systems=("retrieval",),
    workspace=".knowledge/benchmark-v2",
    index=False,
    k=5,
    max_samples=None,
    model="llama3.2",
    max_steps=5,
    max_retries=2,
):
    if split not in {"dev", "held_out", "all"} or not 1 <= k <= 50:
        raise ValueError("Invalid split or retrieval depth")
    if not systems or set(systems) - {"retrieval", "standard", "self_correcting", "agentic"}:
        raise ValueError("Unknown system")
    if len(set(systems)) != len(systems):
        raise ValueError("Systems must not be repeated")
    if not 1 <= max_steps <= 15 or not 0 <= max_retries <= 10:
        raise ValueError("Invalid iteration budgets")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    data = load_benchmark()
    selected = [q for q in data["questions"] if split == "all" or q["split"] == split]
    if max_samples is not None:
        selected = selected[:max_samples]
    metadata = dict(
        version=data["version"],
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_sha256=sha(ROOT / "benchmark.json"),
        corpus=data["documents"],
        split=split,
        question_ids=[q["id"] for q in selected],
        systems=list(systems),
        k=k,
        model=model,
        generation_options=GENERATION_OPTIONS,
        max_steps=max_steps,
        max_retries=max_retries,
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name)
            for name in ["sentence-transformers", "faiss-cpu", "ollama", "langgraph"]
        },
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    )
    metadata["run_id"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    result = dict(metadata=metadata, status="complete", records=[], summary={}, by_category={})
    if set(systems) - {"retrieval"}:
        blocker = ollama_ready(model)
        if blocker:
            result.update(status="blocked", blocker=blocker)
            return result
    service = BenchmarkService(workspace)
    service._llm = BenchmarkLLM(model=model)
    if index:
        # Validate an existing workspace before mutating it; never merge private sources into a benchmark.
        if service.chunks_path.exists():
            validate_index(service, data)
        indexed_result = service.index_documents(str(ROOT / "corpus"))
        if indexed_result.get("errors"):
            raise RuntimeError(str(indexed_result["errors"]))
    indexed = validate_index(service, data)
    metadata.update(
        index_chunks_sha256=sha(service.chunks_path),
        index_faiss_sha256=sha(service.faiss_path),
        chunks=len(indexed),
        embedding_model="all-MiniLM-L6-v2",
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ordering="Fixed system order, shared warm models. Agentic ranks are evidence collection order.",
    )
    _, _, vector_store = service._load_search_context()
    if vector_store.index.ntotal != len(indexed):
        raise ValueError("FAISS and chunk counts disagree; use a fresh workspace")
    # Warm the same embedding and reranker once outside measured query latency.
    service.search("benchmark warmup", top_k=k)
    controller = AgenticController(
        search_tool=SearchTool(service),
        find_tool=FindTool(str(service.chunks_path)),
        open_tool=OpenTool(str(service.chunks_path)),
        llm=service.llm,
    )
    controller.top_k = k
    for q in selected:
        for system in systems:
            row = dict(
                id=q["id"],
                system=system,
                category=q["category"],
                question=q["question"],
                expected_answer=q["expected_answer"],
                expected_behavior=q["expected_behavior"],
                evidence_groups=q["evidence_groups"],
            )
            start = time.perf_counter()
            chunks, output = [], {}
            try:
                if system in {"retrieval", "standard"}:
                    chunks = service.search(q["question"], top_k=k)
                    if system == "standard":
                        context = "Cite factual claims using [filename].\n\n" + "\n\n".join(
                            f"[{Path(c['source_path']).name}] {c['text']}" for c in chunks
                        )
                        output = dict(answer=service.llm.generate(q["question"], context), steps=1)
                elif system == "self_correcting":
                    output = run_agent_workflow(
                        q["question"],
                        max_retries=max_retries,
                        service=service,
                        llm=service.llm,
                        top_k=k,
                    )
                    chunks = output["retrieved_chunks"]
                else:
                    output = controller.run(q["question"], max_steps=max_steps)
                    chunks = output["evidence_records"]
                row["error"] = output.get("error")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["latency_ms"] = (time.perf_counter() - start) * 1000
            row["output"] = output
            row["retrieved_chunks"] = chunks
            metrics = retrieval_scores(q, chunks, indexed, data["passages"], k)
            if row.get("error") and not chunks:
                metrics = {
                    name: 0.0 if value is not None else None for name, value in metrics.items()
                }
            if system != "retrieval":
                metrics.update(generation_scores(q, {**output, "error": row.get("error")}, chunks))
            metrics["latency_ms"] = row["latency_ms"]
            row["metrics"] = metrics
            result["records"].append(row)
    result["summary"] = summarize(result["records"])
    result["by_category"] = {
        category: summarize([r for r in result["records"] if r["category"] == category])
        for category in sorted({r["category"] for r in result["records"]})
    }
    result["status"] = (
        "partial_failure" if any(r.get("error") for r in result["records"]) else "complete"
    )
    return result


def review_template(result):
    return {
        "run_id": result["metadata"]["run_id"],
        "reviews": [
            {
                "id": r["id"],
                "system": r["system"],
                "answer_correctness": None,
                "faithfulness": None,
                "citation_support": None,
                "notes": "",
            }
            for r in result["records"]
            if r["system"] != "retrieval" and not r.get("error")
        ],
    }


def apply_reviews(result, reviews):
    if reviews["run_id"] != result["metadata"]["run_id"]:
        raise ValueError("Review belongs to another run")
    valid = {
        (r["id"], r["system"]): r
        for r in result["records"]
        if r["system"] != "retrieval" and not r.get("error")
    }
    seen = set()
    for review in reviews["reviews"]:
        key = (review["id"], review["system"])
        if key not in valid or key in seen:
            raise ValueError("Unknown, failed, or duplicate reviewed result")
        seen.add(key)
        for name in ("answer_correctness", "faithfulness", "citation_support"):
            value = review[name]
            if value is not None and (
                type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1
            ):
                raise ValueError(
                    "Review scores must be null or finite numbers between zero and one"
                )
            valid[key]["metrics"][name] = value
        valid[key]["review_notes"] = review.get("notes", "")
    result["summary"] = summarize(result["records"])
    result["by_category"] = {
        cat: summarize([r for r in result["records"] if r["category"] == cat])
        for cat in result["by_category"]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "held_out", "all"], default="dev")
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["retrieval", "standard", "self_correcting", "agentic"],
        default=["retrieval"],
    )
    parser.add_argument("--workspace", default=".knowledge/benchmark-v2")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--review-input", help="Existing result JSON to annotate instead of executing models"
    )
    parser.add_argument("--reviews", help="Completed human review JSON, tied to the input run")
    parser.add_argument("--review-template", help="Write a blank review file alongside the result")
    args = parser.parse_args()
    if args.review_input:
        if not args.reviews:
            parser.error("--review-input requires --reviews")
        result = apply_reviews(
            json.loads(Path(args.review_input).read_text()),
            json.loads(Path(args.reviews).read_text()),
        )
    else:
        if args.reviews:
            parser.error("--reviews requires --review-input")
        result = run(
            args.split,
            tuple(args.systems),
            args.workspace,
            args.index,
            args.k,
            args.max_samples,
            args.model,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.review_template:
        target = Path(args.review_template)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(review_template(result), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "blocker": result.get("blocker"),
            },
            indent=2,
        )
    )
    if result["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
