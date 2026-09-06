import pytest

from semantic_paper_router import Reference, ScopePolicy, SemanticClassifier


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, text: str) -> list[float]:
        return self.vectors[text]


def make_classifier() -> SemanticClassifier:
    return SemanticClassifier(
        [
            Reference("cs", "computer_science", "cs reference"),
            Reference("bio", "biology", "bio reference"),
        ],
        FakeEmbedder(
            {
                "cs reference": [1, 0],
                "bio reference": [0, 1],
                "clear cs": [1, 0],
                "ambiguous": [0.7, 0.7],
            }
        ),
    )


def test_scope_policy_accepts_clear_classification() -> None:
    decision = make_classifier().classify_with_scope(
        "clear cs", ScopePolicy(min_score=0.9, min_margin=0.5)
    )

    assert decision.in_scope is True
    assert decision.category == "computer_science"
    assert decision.reason == "accepted"


def test_scope_policy_rejects_low_margin_as_unknown() -> None:
    decision = make_classifier().classify_with_scope(
        "ambiguous", ScopePolicy(min_score=0.7, min_margin=0.1)
    )

    assert decision.in_scope is False
    assert decision.category == "unknown"
    assert decision.reason == "margin_below_threshold"


@pytest.mark.parametrize("kwargs", [{"min_score": -0.1}, {"min_margin": -0.1}])
def test_scope_policy_rejects_negative_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ScopePolicy(**kwargs)
