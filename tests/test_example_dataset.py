import json
import shutil
from collections import Counter

import pytest

from knowledge.evaluation.examples import EXAMPLES, load_examples


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
