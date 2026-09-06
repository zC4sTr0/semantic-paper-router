"""Core building blocks for semantic paper routing."""

from .vectors import cosine_similarity
from .classifier import (
    ClassificationResult,
    Embedder,
    Reference,
    SemanticClassifier,
)

__all__ = [
    "ClassificationResult",
    "Embedder",
    "Reference",
    "SemanticClassifier",
    "cosine_similarity",
]
