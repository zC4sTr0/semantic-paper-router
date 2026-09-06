"""Core building blocks for semantic paper routing."""

from .vectors import cosine_similarity
from .classifier import (
    ClassificationResult,
    Embedder,
    Reference,
    SemanticClassifier,
)
from .embeddings import BagOfWordsEmbedder

__all__ = [
    "ClassificationResult",
    "BagOfWordsEmbedder",
    "Embedder",
    "Reference",
    "SemanticClassifier",
    "cosine_similarity",
]
