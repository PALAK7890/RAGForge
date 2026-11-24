import copy
import json
import shutil
from collections import Counter
from unittest.mock import MagicMock

import pytest

from knowledge.evaluation.benchmark_v2 import (
    ROOT,
    apply_reviews,
    generation_scores,
    load_benchmark,
    retrieval_scores,
    review_template,
    run,
    summarize,
    validate_index,
)


def test_v2_has_exact_counts_separate_document_families_and_real_multihop():
    data = load_benchmark()
    assert len(data["documents"]) == 24
    assert Counter(q["split"] for q in data["questions"]) == {"dev": 62, "held_out": 58}
    assert sum(q["is_unanswerable"] for q in data["questions"]) == 10
    assert sum(q["expected_behavior"] == "clarify" for q in data["questions"]) == 10
    families = {
        split: {p["document"] for p in data["passages"].values() if p["split"] == split}
        for split in ("dev", "held_out")
    }
    assert not families["dev"] & families["held_out"]
    for q in data["questions"]:
        if q["category"] == "multihop":
            assert (
                len(
                    {
                        data["passages"][pid]["document"]
                        for group in q["evidence_groups"]
                        for pid in group
                    }
                )
                >= 2
            )


def test_corpus_drift_and_cross_split_labels_fail(tmp_path):
    root = tmp_path / "benchmark"
    shutil.copytree(ROOT, root)
    data = load_benchmark(root)
    file = root / "corpus" / next(iter(data["documents"]))
    file.write_text(file.read_text() + "\nchanged\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_benchmark(root)
    shutil.copy2(ROOT / "corpus" / file.name, file)
    data["questions"][0]["evidence_groups"] = [["held_out.map_instrument"]]
    (root / "benchmark.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="split"):
        load_benchmark(root)


def test_passage_metrics_require_text_not_just_correct_filename():
    passages = {
        "a": {"document": "a.md", "quote": "Gold fact."},
        "b": {"document": "b.md", "quote": "Other fact."},
        "alt": {"document": "c.md", "quote": "Alternate fact."},
    }
    q = {"evidence_groups": [["a"], ["b", "alt"]]}
    wrong = {"source_path": "/a.md", "text": "Unrelated section.", "chunk_id": "wrong"}
    right = {"source_path": "/a.md", "text": "Gold fact.", "chunk_id": "right"}
    alternate = {"source_path": "/c.md", "text": "Alternate fact.", "chunk_id": "alt"}
    indexed = [wrong, right, alternate]
    scores = retrieval_scores(q, [wrong, right, alternate], indexed, passages, 2)
    assert scores["passage_recall_at_k"] == 0.5
    assert scores["all_evidence_group_recall"] == 1
    assert scores["precision_at_k"] == 0.5
    assert scores["mrr_at_k"] == 0.5
    assert 0 < scores["ndcg_at_k"] < 1
    assert retrieval_scores(q, [right, alternate], indexed, passages, 2)["passage_recall_at_k"] == 1
    assert retrieval_scores(q, [right, right], indexed, passages, 2)["precision_at_k"] == 0.5


def test_unanswerable_has_no_retrieval_denominator():
    scores = retrieval_scores({"evidence_groups": []}, [], [], {}, 5)
    assert all(value is None for value in scores.values())


def test_unsupported_answers_and_failures_do_not_get_generation_credit():
    q = {"expected_answer": "42", "is_unanswerable": True}
    scores = generation_scores(q, {"answer": "42", "error": "model failed"}, [])
    assert scores["exact_answer_match_proxy"] == 0
    assert scores["abstention_accuracy_heuristic"] == 0
    assert scores["citation_reference_validity"] is None
    assert scores["answer_correctness"] is None


def test_human_review_is_run_bound_and_only_scores_actual_annotations():
    result = {
        "metadata": {"run_id": "run1"},
        "records": [
            {
                "id": "q",
                "system": "standard",
                "category": "factual",
                "metrics": {"answer_correctness": None},
                "error": None,
            }
        ],
        "by_category": {"factual": {}},
    }
    template = review_template(result)
    assert template["reviews"][0]["answer_correctness"] is None
    template["reviews"][0].update(answer_correctness=1, faithfulness=0.5, citation_support=0)
    scored = apply_reviews(copy.deepcopy(result), template)
    assert scored["summary"]["standard"]["metrics"]["answer_correctness"] == {"mean": 1, "n": 1}
    template["run_id"] = "other"
    with pytest.raises(ValueError, match="another run"):
        apply_reviews(result, template)
    template["run_id"] = "run1"
    template["reviews"][0]["faithfulness"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        apply_reviews(result, template)


def test_generation_preflight_records_blocker_without_running_models(monkeypatch):
    monkeypatch.setattr(
        "knowledge.evaluation.benchmark_v2.ollama_ready", lambda model: "model not available"
    )
    service = MagicMock()
    monkeypatch.setattr("knowledge.evaluation.benchmark_v2.BenchmarkService", service)
    result = run(systems=("standard", "self_correcting", "agentic"), max_samples=2)
    assert result["status"] == "blocked"
    assert result["records"] == []
    assert result["summary"] == {}
    service.assert_not_called()


def test_index_rejects_wrong_workspace_before_using_corpus(tmp_path):
    service = MagicMock(chunks_path=tmp_path / "chunks.json")
    service.chunks_path.write_text(
        json.dumps({"chunks": [{"source_path": "/private/user.md", "text": "private"}]})
    )
    with pytest.raises(ValueError, match="unrelated"):
        validate_index(service, load_benchmark())


def test_summary_discloses_metric_denominators_and_errors():
    rows = [
        {"system": "standard", "error": "failed", "metrics": {"answer_correctness": None}},
        {"system": "standard", "error": None, "metrics": {"answer_correctness": 1}},
    ]
    summary = summarize(rows)["standard"]
    assert summary["queries"] == 2 and summary["errors"] == 1
    assert summary["metrics"]["answer_correctness"] == {"mean": 1, "n": 1}


def test_legacy_graph_injected_service_and_model_are_used(monkeypatch):
    import knowledge.agent.nodes as nodes
    from knowledge.agent.graph import run_agent_workflow

    service, llm = MagicMock(), MagicMock()
    service.search.return_value = [
        {"source_path": "/benchmark/a.md", "text": "answer from the selected corpus"}
    ]
    llm.generate.side_effect = ["SUFFICIENT", "answer from the selected corpus"]
    # Make a leaked use of global dependencies fail loudly.
    monkeypatch.setattr(nodes, "service", None)
    monkeypatch.setattr(nodes, "llm", None)
    result = run_agent_workflow("q", max_retries=0, service=service, llm=llm, top_k=3)
    service.search.assert_called_once_with(query="q", top_k=3)
    assert result["retrieved_chunks"][0]["source_path"] == "/benchmark/a.md"
    assert result["answer"] == "answer from the selected corpus"


def test_authored_snapshot_regenerates_byte_for_byte(tmp_path):
    import subprocess
    import sys

    shutil.copy2(ROOT / "build.py", tmp_path / "build.py")
    subprocess.run([sys.executable, str(tmp_path / "build.py")], check=True, capture_output=True)
    assert (tmp_path / "benchmark.json").read_bytes() == (ROOT / "benchmark.json").read_bytes()
    for source in (ROOT / "corpus").glob("*.md"):
        assert (tmp_path / "corpus" / source.name).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("use_model_planner", [False, True])
def test_adapters_share_corpus_model_and_depth(tmp_path, monkeypatch, use_model_planner):
    data = load_benchmark()
    q = data["questions"][0]
    p = data["passages"][q["evidence_groups"][0][0]]
    chunk = {
        "source_path": str(ROOT / "corpus" / p["document"]),
        "text": p["quote"],
        "chunk_id": "c",
    }
    service = MagicMock()
    service.chunks_path = tmp_path / "chunks.json"
    service.faiss_path = str(tmp_path / "faiss.index")
    service.chunks_path.write_text("{}")
    (tmp_path / "faiss.index").write_bytes(b"fake index for unit test")
    service.search.return_value = [chunk]

    # Match the service's real property behavior when the runner selects its model.
    class Service:
        chunks_path = service.chunks_path
        faiss_path = service.faiss_path
        search = service.search

        def _load_search_context(self):
            vector = MagicMock()
            vector.index.ntotal = 1
            return [], None, vector

        @property
        def llm(self):
            return self._llm

    shared = Service()
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Reference answer"
    monkeypatch.setattr("knowledge.evaluation.benchmark_v2.BenchmarkLLM", lambda model: mock_llm)
    monkeypatch.setattr(
        "knowledge.evaluation.benchmark_v2.BenchmarkService", lambda workspace: shared
    )
    monkeypatch.setattr("knowledge.evaluation.benchmark_v2.ollama_ready", lambda model: None)
    monkeypatch.setattr(
        "knowledge.evaluation.benchmark_v2.validate_index", lambda service, data: [chunk]
    )
    legacy = MagicMock(return_value={"answer": "Reference answer", "retrieved_chunks": [chunk]})
    monkeypatch.setattr("knowledge.evaluation.benchmark_v2.run_agent_workflow", legacy)
    controllers = {}

    def create_controller(**kwargs):
        controller = MagicMock()
        mode = kwargs["planner_mode"]
        controllers[mode] = controller
        controller.run.return_value = {
            "answer": "Reference answer", "evidence_records": [chunk], "planner_mode": mode,
        }
        return controller

    factory = MagicMock(side_effect=create_controller)
    monkeypatch.setattr("knowledge.evaluation.benchmark_v2.AgenticController", factory)
    systems = ("standard", "self_correcting", "agentic")
    if use_model_planner:
        systems += ("agentic_model",)
    result = run(systems=systems, k=3, max_samples=1)
    assert len(result["records"]) == len(systems)
    assert result["status"] == "complete"
    assert legacy.call_args.kwargs["service"] is shared
    assert legacy.call_args.kwargs["llm"] is mock_llm
    assert legacy.call_args.kwargs["top_k"] == 3
    for call in factory.call_args_list:
        assert call.kwargs["search_tool"].service is shared
        assert call.kwargs["llm"] is mock_llm
        assert call.kwargs["find_tool"].chunks_path == shared.chunks_path
        assert call.kwargs["open_tool"].chunks_path == shared.chunks_path
    assert set(controllers) == ({"heuristic", "model"} if use_model_planner else {"heuristic"})
    for controller in controllers.values():
        assert controller.top_k == 3
        controller.run.assert_called_once_with(q["question"], max_steps=5)
    for row in result["records"]:
        if row["system"] in {"agentic", "agentic_model"}:
            assert row["output"]["planner_mode"] == (
                "model" if row["system"] == "agentic_model" else "heuristic"
            )
    assert all(row["metrics"]["passage_recall_at_k"] == 1 for row in result["records"])


def test_benchmark_planner_and_generation_share_recorded_options(monkeypatch):
    from knowledge.evaluation.benchmark_v2 import GENERATION_OPTIONS, BenchmarkLLM

    client = MagicMock()
    decision = {"action": "search", "arguments": {"queries": ["target fact"]}, "reason": "Locate the missing fact"}
    client.chat.return_value = {"message": {"content": json.dumps(decision)}}
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("ollama.Client", factory)
    llm = BenchmarkLLM("selected-model")
    assert llm.plan_action("question", "planner state") == decision
    request = client.chat.call_args.kwargs
    assert request["model"] == "selected-model"
    assert request["options"] == GENERATION_OPTIONS
    assert request["format"]["oneOf"][0]["type"] == "object"
    assert "action" in request["format"]["oneOf"][0]["properties"]
    assert "timeout" not in request
    factory.assert_called_with(timeout=120)

    client.chat.return_value = {"message": {"content": "answer"}}
    assert llm.generate("question", "context") == "answer"
    request = client.chat.call_args.kwargs
    assert request["messages"] == [{"role": "user", "content": llm._build_prompt("question", "context")}]
    assert request["options"] == GENERATION_OPTIONS
    assert "format" not in request
