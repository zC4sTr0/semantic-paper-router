"""Core building blocks for semantic paper routing."""

from .vectors import cosine_similarity
from .classifier import (
    ClassificationResult,
    Embedder,
    Reference,
    ScopeDecision,
    ScopePolicy,
    SemanticClassifier,
)
from .embeddings import BagOfWordsEmbedder, BedrockTitanEmbedder
from .corpus import load_references

__all__ = [
    "ClassificationResult",
    "BagOfWordsEmbedder",
    "BedrockTitanEmbedder",
    "load_references",
    "Embedder",
    "Reference",
    "ScopeDecision",
    "ScopePolicy",
    "SemanticClassifier",
    "cosine_similarity",
]
