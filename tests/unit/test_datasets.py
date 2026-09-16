"""Unit tests for the golden dataset manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.datasets import (
    EvalSample,
    GoldenDatasetManager,
)


def _sample(question: str = "What is RRF?") -> EvalSample:
    return EvalSample(
        question=question,
        answer="RRF is Reciprocal Rank Fusion.",
        contexts=["Reciprocal Rank Fusion (RRF) merges ranked lists."],
        ground_truth="Reciprocal Rank Fusion",
    )


def test_load_starter_dataset_returns_twelve_samples() -> None:
    manager = GoldenDatasetManager()
    samples = manager.load("starter_dataset.json")
    assert len(samples) == 12
    assert all(isinstance(s, EvalSample) for s in samples)
    assert all(s.question.strip() and s.answer.strip() for s in samples)
    assert all(isinstance(s.contexts, list) for s in samples)


def test_load_starter_dataset_covers_factual_multihop_and_edge() -> None:
    manager = GoldenDatasetManager()
    samples = manager.load("starter_dataset.json")
    assert any(s.ground_truth for s in samples)  # reference-based rows
    assert any(s.ground_truth is None for s in samples)  # edge rows
    assert any(len(s.contexts) >= 2 for s in samples)  # multi-hop rows
    assert any(len(s.contexts) == 0 for s in samples)  # edge rows


def test_save_and_load_roundtrip_preserves_samples(tmp_path: Path) -> None:
    manager = GoldenDatasetManager(tmp_path)
    original = [_sample("What is RRF?"), _sample("What is BM25?")]
    path = manager.save(original, "roundtrip.json")
    assert path.is_file()
    loaded = manager.load("roundtrip.json")
    assert loaded == original


def test_add_sample_with_missing_file_creates_dataset(tmp_path: Path) -> None:
    manager = GoldenDatasetManager(tmp_path)
    manager.add_sample(_sample(), "new.json")
    assert manager.load("new.json") == [_sample()]


def test_add_sample_with_existing_file_appends(tmp_path: Path) -> None:
    manager = GoldenDatasetManager(tmp_path)
    manager.save([_sample("First?")], "curated.json")
    manager.add_sample(_sample("Second?"), "curated.json")
    loaded = manager.load("curated.json")
    assert [s.question for s in loaded] == ["First?", "Second?"]


def test_load_with_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    manager = GoldenDatasetManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.load("nope.json")


def test_load_with_malformed_json_raises_valueerror(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        GoldenDatasetManager(tmp_path).load("bad.json")


def test_load_with_non_array_json_raises_valueerror(tmp_path: Path) -> None:
    (tmp_path / "obj.json").write_text('{"question": "hi"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        GoldenDatasetManager(tmp_path).load("obj.json")


@pytest.mark.parametrize(
    "record",
    [
        {"answer": "a", "contexts": []},  # missing question
        {"question": "q", "contexts": []},  # missing answer
        {"question": "q", "answer": "a"},  # missing contexts
        {"question": "q", "answer": "a", "contexts": "nope"},  # bad contexts
        {"question": "  ", "answer": "a", "contexts": []},  # blank question
    ],
)
def test_load_with_invalid_rows_raises_valueerror(
    tmp_path: Path, record: dict[str, object]
) -> None:
    (tmp_path / "invalid.json").write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError):
        GoldenDatasetManager(tmp_path).load("invalid.json")


def test_list_datasets_returns_sorted_json_names(tmp_path: Path) -> None:
    manager = GoldenDatasetManager(tmp_path)
    assert manager.list_datasets() == []
    manager.save([_sample()], "b.json")
    manager.save([_sample()], "a.json")
    assert manager.list_datasets() == ["a.json", "b.json"]


def test_create_starter_dataset_writes_twelve_samples(tmp_path: Path) -> None:
    path = GoldenDatasetManager.create_starter_dataset(tmp_path)
    assert path.name == "starter_dataset.json"
    assert len(GoldenDatasetManager(tmp_path).load(path.name)) == 12


def test_sample_ids_are_stable_and_unique() -> None:
    from src.evaluation.datasets import sample_id_for

    assert sample_id_for("What is RRF?", 0) == sample_id_for("What is RRF?", 0)
    assert sample_id_for("What is RRF?", 0) != sample_id_for("What is RRF?", 1)
    assert sample_id_for("What is RRF?", 0) != sample_id_for("What is BM25?", 0)
