"""A small deterministic baseline embedder for local experiments."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from math import isfinite
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")


class BagOfWordsEmbedder:
    """Represent text by counts over a fitted vocabulary.

    This is intentionally a lexical baseline. It is useful for exercising the
    router locally, but it does not capture meaning the way a trained model
    does.
    """

    def __init__(self, vocabulary: Sequence[str]) -> None:
        if not vocabulary:
            raise ValueError("vocabulary must not be empty")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary must not contain duplicates")
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


class BedrockTitanEmbedder:
    """Create embeddings with Amazon Bedrock Titan Text Embeddings.

    ``boto3`` is optional and imported only when a client is not injected. This
    keeps local development and deterministic tests dependency-free.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region_name: str | None = None,
    ) -> None:
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a string")
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if model_id != model_id.strip():
            raise ValueError("model_id must not have leading or trailing whitespace")
        self._model_id = model_id
        self._client = client
        self._region_name = region_name

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "boto3 is required when no Bedrock client is injected; "
                "install the 'bedrock' extra"
            ) from exc
        kwargs = {}
        if self._region_name is not None:
            kwargs["region_name"] = self._region_name
        self._client = boto3.client("bedrock-runtime", **kwargs)
        return self._client

    def embed(self, text: str) -> Sequence[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

        response = self._client_or_create().invoke_model(
            modelId=self._model_id,
            body=json.dumps({"inputText": text}),
            contentType="application/json",
            accept="application/json",
        )
        body = response["body"]
        payload = json.loads(body.read() if hasattr(body, "read") else body)
        if not isinstance(payload, dict):
            raise ValueError("Bedrock response must be a JSON object containing embedding")
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Bedrock response must contain a non-empty embedding")
        normalized: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Bedrock response embedding must contain finite numbers")
            try:
                normalized_value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    "Bedrock response embedding must contain finite numbers"
                ) from exc
            if not isfinite(normalized_value):
                raise ValueError("Bedrock response embedding must contain finite numbers")
            normalized.append(normalized_value)
        return tuple(normalized)
