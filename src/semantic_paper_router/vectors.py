"""Small, dependency-free vector operations."""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity between two non-zero vectors.

    The function deliberately validates its input here instead of relying on
    a numerical library. This keeps the first implementation inspectable and
    makes malformed embeddings fail close to their source.
    """
    if not left or not right:
        raise ValueError("vectors must not be empty")
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")

    try:
        values = [(*left, *right)]
    except TypeError as exc:
        raise TypeError("vectors must be finite numeric sequences") from exc

    flattened = values[0]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) for value in flattened):
        raise TypeError("vectors must be finite numeric sequences")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("zero vectors have no cosine direction")

    return dot / (left_norm * right_norm)
