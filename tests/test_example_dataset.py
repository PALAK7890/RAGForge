import hashlib
import json
import shutil
from collections import Counter
from unittest.mock import MagicMock

import pytest

from knowledge.evaluation.examples import EXAMPLES, load_examples, run_examples


def test_examples_have_verifiable_sources_and_balanced_practice_splits():
    questions = load_examples()
    assert len(questions) == 40
    assert len({q["question"] for q in questions}) == 40
    assert Counter(q["split"] for q in questions) == {"dev": 20, "held_out": 20}
    assert sum(q["is_unanswerable"] for q in questions) == 8
    assert len(list((EXAMPLES / "corpus").glob("*.md"))) == 8
    assert all(q["expected_answer"] for q in questions)


def test_stale_evidence_is_rejected(tmp_path):
    shutil.copytree(EXAMPLES, tmp_path / "examples")
    root = tmp_path / "examples"
    questions = json.loads((root / "questions.json").read_text())
    questions[0]["gold_evidence"][0]["quote"] = "invented passage"
    (root / "questions.json").write_text(json.dumps(questions))
    with pytest.raises(ValueError, match="Stale evidence"):
        load_examples(root)


def test_project_documentation_examples_have_verifiable_frozen_sources():
    root = EXAMPLES.parent / "project_docs"
    questions = load_examples(root)
    assert len(questions) == 12
    assert len({q["category"] for q in questions}) == 8
    assert sum(q["is_unanswerable"] for q in questions) == 2
    manifest = json.loads((root / "sources.json").read_text())
    for name, source in manifest["documents"].items():
        assert hashlib.sha256((root / "corpus" / name).read_bytes()).hexdigest() == source["sha256"]


def test_example_index_rejects_user_workspace_before_mutating_it(tmp_path, monkeypatch):
    service = MagicMock(chunks_path=tmp_path / "chunks.json")
    service.chunks_path.write_text(json.dumps({"chunks": [{"source_path": "/private/user.md"}]}))
    monkeypatch.setattr("knowledge.evaluation.examples.RAGService", lambda _: service)
    with pytest.raises(ValueError, match="fresh workspace"):
        run_examples(index=True, root=EXAMPLES.parent / "project_docs")
    service.index_documents.assert_not_called()
