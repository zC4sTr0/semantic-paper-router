"""Domain contracts for semantic paper classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .vectors import cosine_similarity


class Embedder(Protocol):
    """Provider capable of converting text into one numeric vector."""

    def embed(self, text: str) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class Reference:
    """Labeled text used as evidence for a classification."""

    id: str
    category: str
    text: str


@dataclass(frozen=True)
class ClassificationResult:
    """Decision plus the evidence needed to inspect how it was made."""

    category: str
    score: float
    reference_id: str
    margin: float


class SemanticClassifier:
    """Classify text by the most similar labeled reference."""

    def __init__(self, references: Sequence[Reference], embedder: Embedder) -> None:
        if not references:
            raise ValueError("at least one reference is required")
        self._references = tuple(references)
        self._embedder = embedder
        self._reference_vectors = tuple(
            embedder.embed(reference.text) for reference in self._references
        )

    def classify(self, text: str) -> ClassificationResult:
        if not text.strip():
            raise ValueError("text must not be empty")

        query_vector = self._embedder.embed(text)
        scored = [
            (cosine_similarity(query_vector, vector), reference)
            for reference, vector in zip(self._references, self._reference_vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_reference = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else best_score
        return ClassificationResult(
            category=best_reference.category,
            score=best_score,
            reference_id=best_reference.id,
            margin=best_score - second_score,
        )
