"""Core building blocks for semantic paper routing."""

from .vectors import cosine_similarity
from .classifier import (
    ClassificationResult,
    Embedder,
    Reference,
    SemanticClassifier,
)
from .embeddings import BagOfWordsEmbedder
from .corpus import load_references

__all__ = [
    "ClassificationResult",
    "BagOfWordsEmbedder",
    "load_references",
    "Embedder",
    "Reference",
    "SemanticClassifier",
    "cosine_similarity",
]
