import json

import pytest

from semantic_paper_router import load_references


def test_load_references_returns_labeled_records(tmp_path) -> None:
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps([
            {"id": "one", "category": "a", "text": "A paper"},
        ]),
        encoding="utf-8",
    )

    references = load_references(path)

    assert references[0].id == "one"
    assert references[0].category == "a"


def test_duplicate_ids_are_rejected(tmp_path) -> None:
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps([
            {"id": "same", "category": "a", "text": "one"},
            {"id": "same", "category": "b", "text": "two"},
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_references(path)


def test_empty_corpus_is_rejected(tmp_path) -> None:
    path = tmp_path / "references.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        load_references(path)
