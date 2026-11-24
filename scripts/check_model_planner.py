"""Real retrieval and Ollama smoke check for an adaptive, two-document investigation."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge.agentic.controller import AgenticController
from knowledge.agentic.tools import FindTool, OpenTool, SearchTool
from knowledge.evaluation.benchmark_v2 import GENERATION_OPTIONS, BenchmarkLLM, BenchmarkService

SOURCES = {
    "routing.md": "# Audit routing\nThe Project KESTREL audit packet is approved by the Meridian Steward. "
    "The named person holding this role is recorded in the ownership register.\n",
    "owners.md": "# Ownership register\nThe Meridian Steward is Nisha Rao. "
    "The Solstice Steward is Arun Mehta.\n",
    "unrelated.md": "# Workshop supplies\nThe workshop supply coordinator is Leena Shah. "
    "The coordinator orders printer paper and office supplies.\n",
}
QUESTION = "Who is the named person who approves the Project KESTREL audit packet?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="llama3.2")
    args = parser.parse_args()
    import ollama

    metadata = {
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "model": args.model, "options": GENERATION_OPTIONS, "top_k": 1, "max_steps": 5,
        "model_inventory": ollama.Client(timeout=120).list().model_dump(mode="json"),
        "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in SOURCES.items()},
        "implementation_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("knowledge/agentic/controller.py", "knowledge/llms/ollama_llm.py", "knowledge/agentic/schemas.py")
        },
    }
    with tempfile.TemporaryDirectory(prefix="ragforge-planner-") as directory:
        corpus = Path(directory) / "corpus"
        corpus.mkdir()
        for name, text in SOURCES.items():
            (corpus / name).write_text(text, encoding="utf-8")
        service = BenchmarkService(str(Path(directory) / "workspace"))
        indexed = service.index_documents(str(corpus))
        if indexed.get("errors"):
            raise RuntimeError(str(indexed["errors"]))
        service.search("audit routing", top_k=1)  # Warm retrieval before timing the investigation.
        controller = AgenticController(
            search_tool=SearchTool(service), find_tool=FindTool(str(service.chunks_path)),
            open_tool=OpenTool(str(service.chunks_path)), llm=BenchmarkLLM(args.model), planner_mode="model",
        )
        controller.top_k = 1
        started = time.perf_counter()
        output = controller.run(QUESTION, max_steps=5)
        elapsed = time.perf_counter() - started
    searches = [step for step in output["tool_history"] if step["tool"] == "SEARCH"]
    adaptive = bool(
        len(searches) >= 2 and searches[0].get("documents") == ["routing.md"]
        and any("owners.md" in step.get("documents", []) for step in searches[1:])
        and "Nisha Rao" in output["answer"] and not output["error"]
    )
    result = dict(metadata=metadata, sources=SOURCES, question=QUESTION, output=output,
                  latency_seconds=elapsed, adaptive_hop_observed=adaptive,
                  limitations="Authored three-document smoke check with top_k=1, not a benchmark or general quality estimate.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"adaptive_hop_observed": adaptive, "error": output["error"],
                      "steps": output["steps"], "answer": output["answer"], "output": str(args.output)}, indent=2))
    return 0 if adaptive else 1


if __name__ == "__main__":
    raise SystemExit(main())
