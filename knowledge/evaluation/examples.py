"""Run the authored fictional practice corpus in an isolated workspace."""

import argparse
import hashlib
import json
from pathlib import Path

from knowledge.agentic.controller import AgenticController
from knowledge.agentic.tools import FindTool, OpenTool, SearchTool
from knowledge.api.service import RAGService

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "evaluation"


def load_examples(root=EXAMPLES):
    root = Path(root)
    questions = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    ids = set()
    for item in questions:
        if item["id"] in ids:
            raise ValueError("Duplicate question ID")
        ids.add(item["id"])
        if item["is_unanswerable"] and (item["expected_documents"] or item["gold_evidence"]):
            raise ValueError("Unanswerable items cannot claim positive evidence")
        if not item["is_unanswerable"] and not item["gold_evidence"]:
            raise ValueError("Answerable items need evidence")
        for evidence in item["gold_evidence"]:
            name = evidence["document"]
            if Path(name).name != name:
                raise ValueError("Evidence document must be a filename")
            text = (root / "corpus" / name).read_text(encoding="utf-8")
            if text[evidence["char_start"] : evidence["char_end"]] != evidence["quote"]:
                raise ValueError(f"Stale evidence in {item['id']}")
        if set(item["expected_documents"]) != {e["document"] for e in item["gold_evidence"]}:
            raise ValueError("Expected sources and evidence disagree")
    return questions


def run_examples(
    workspace=".knowledge/examples",
    mode="retrieval",
    split=None,
    max_samples=None,
    top_k=5,
    index=False,
):
    questions = load_examples()
    service = RAGService(workspace)
    if index:
        result = service.index_documents(str(EXAMPLES / "corpus"))
        if result.get("errors"):
            raise RuntimeError(f"Example indexing failed: {result['errors']}")
    if not service.chunks_path.exists():
        raise FileNotFoundError("Build the example index with --index first")
    indexed = json.loads(service.chunks_path.read_text(encoding="utf-8"))["chunks"]
    actual_sources = {Path(c["source_path"]).resolve() for c in indexed}
    expected_sources = {p.resolve() for p in (EXAMPLES / "corpus").glob("*.md")}
    if actual_sources != expected_sources:
        raise ValueError(
            "Workspace must contain only the eight example documents; use a fresh workspace"
        )
    selected = [q for q in questions if split is None or q["split"] == split]
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        selected = selected[:max_samples]
    if mode not in {"retrieval", "standard", "agentic"}:
        raise ValueError("Unsupported mode")
    controller = AgenticController(
        search_tool=SearchTool(service),
        find_tool=FindTool(str(service.chunks_path)),
        open_tool=OpenTool(str(service.chunks_path)),
        llm=service.llm,
    )
    records = []
    for question in selected:
        row = {
            "id": question["id"],
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "gold_evidence": question["gold_evidence"],
            "is_unanswerable": question["is_unanswerable"],
        }
        try:
            if mode == "agentic":
                output = controller.run(question["question"])
                retrieved = output["discovered_docs"]
            else:
                chunks = service.search(question["question"], top_k=top_k)
                retrieved = list(dict.fromkeys(Path(c["source_path"]).name for c in chunks))
                output = {"chunks": chunks}
                if mode == "standard":
                    output["answer"] = service.llm.generate(
                        question["question"], "\n\n".join(c["text"] for c in chunks)
                    )
            expected = set(question["expected_documents"])
            row.update(
                output=output,
                retrieved_documents=retrieved,
                document_recall=(
                    len(expected.intersection(retrieved)) / len(expected) if expected else None
                ),
            )
        except Exception as exc:
            row.update(error=str(exc), document_recall=None)
        records.append(row)
    return {
        "mode": mode,
        "split": split,
        "top_k": top_k,
        "records": records,
        "corpus_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((EXAMPLES / "corpus").glob("*.md"))
        },
        "questions_sha256": hashlib.sha256((EXAMPLES / "questions.json").read_bytes()).hexdigest(),
        "limitations": "Authored fictional practice set. Review answers manually; document recall is not answer correctness. No retrieval credit is assigned to unanswerable questions. Errors are retained.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".knowledge/examples")
    parser.add_argument("--mode", choices=["retrieval", "standard", "agentic"], default="retrieval")
    parser.add_argument("--split", choices=["dev", "held_out"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--top-k", type=int, choices=range(1, 51), default=5)
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_examples(
        args.workspace, args.mode, args.split, args.max_samples, args.top_k, args.index
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {len(result['records'])} example results to {output}")


if __name__ == "__main__":
    main()
