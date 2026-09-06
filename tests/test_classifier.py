import pytest

from semantic_paper_router import Reference, SemanticClassifier


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, text: str) -> list[float]:
        return self.vectors[text]


def test_classifier_returns_winner_and_margin() -> None:
    embedder = FakeEmbedder(
        {
            "cs reference": [1, 0],
            "biology reference": [0, 1],
            "new paper": [0.9, 0.1],
        }
    )
    classifier = SemanticClassifier(
        [
            Reference("cs-001", "computer_science", "cs reference"),
            Reference("bio-001", "biology", "biology reference"),
        ],
        embedder,
    )

    result = classifier.classify("new paper")

    assert result.category == "computer_science"
    assert result.reference_id == "cs-001"
    assert result.score == pytest.approx(0.9938837)
    assert result.margin > 0


def test_classifier_keeps_first_reference_on_exact_tie() -> None:
    embedder = FakeEmbedder(
        {"first": [1, 0], "second": [0, 1], "query": [1, 1]}
    )
    classifier = SemanticClassifier(
        [Reference("one", "a", "first"), Reference("two", "b", "second")],
        embedder,
    )

    result = classifier.classify("query")

    assert result.reference_id == "one"
    assert result.margin == pytest.approx(0.0)


@pytest.mark.parametrize("text", ["", "   "])
def test_classifier_rejects_empty_text(text: str) -> None:
    embedder = FakeEmbedder({"reference": [1, 0]})
    classifier = SemanticClassifier(
        [Reference("ref", "category", "reference")], embedder
    )

    with pytest.raises(ValueError, match="must not be empty"):
        classifier.classify(text)


def test_classifier_rejects_empty_references() -> None:
    with pytest.raises(ValueError, match="at least one reference"):
        SemanticClassifier([], FakeEmbedder({}))
