"""A small deterministic baseline embedder for local experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class BagOfWordsEmbedder:
    """Represent text by normalized counts over a fitted vocabulary.

    This is intentionally a lexical baseline. It is useful for exercising the
    router locally, but it does not capture meaning the way a trained model
    does.
    """

    def __init__(self, vocabulary: Sequence[str]) -> None:
        if not vocabulary:
            raise ValueError("vocabulary must not be empty")
        self._vocabulary = tuple(vocabulary)
        self._positions = {token: index for index, token in enumerate(self._vocabulary)}

    @classmethod
    def fit(cls, texts: Iterable[str]) -> "BagOfWordsEmbedder":
        vocabulary = sorted({token for text in texts for token in _TOKEN.findall(text.lower())})
        return cls(vocabulary)

    def embed(self, text: str) -> Sequence[float]:
        vector = [0.0] * len(self._vocabulary)
        for token in _TOKEN.findall(text.lower()):
            position = self._positions.get(token)
            if position is not None:
                vector[position] += 1.0
        return vector
