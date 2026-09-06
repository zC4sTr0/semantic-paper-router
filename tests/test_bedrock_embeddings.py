import json

import pytest

from semantic_paper_router import BedrockTitanEmbedder
from semantic_paper_router import Reference, SemanticClassifier


class FakeBody:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class RawBody:
    def __init__(self, content: str) -> None:
        self.content = content

    def read(self) -> str:
        return self.content


class FakeBedrockClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": FakeBody(self.response)}


def test_titan_embedder_sends_expected_request_and_parses_embedding() -> None:
    client = FakeBedrockClient({"embedding": [0.1, -0.2, 0.3]})
    embedder = BedrockTitanEmbedder(client=client, model_id="test-model")

    result = embedder.embed("a scientific abstract")

    assert result == (0.1, -0.2, 0.3)
    assert client.calls == [
        {
            "modelId": "test-model",
            "body": json.dumps({"inputText": "a scientific abstract"}),
            "contentType": "application/json",
            "accept": "application/json",
        }
    ]


def test_titan_embedder_rejects_empty_text() -> None:
    client = FakeBedrockClient({"embedding": [1.0]})
    embedder = BedrockTitanEmbedder(client=client)

    with pytest.raises(ValueError, match="must not be empty"):
        embedder.embed("  ")
    assert client.calls == []


@pytest.mark.parametrize("model_id", ["", "  model  "])
def test_titan_embedder_rejects_invalid_model_id(model_id: str) -> None:
    with pytest.raises(ValueError, match="model_id"):
        BedrockTitanEmbedder(client=FakeBedrockClient({}), model_id=model_id)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"embedding": []},
        {"embedding": [1.0, "not-a-number"]},
        {"embedding": [10**1000]},
    ],
)
def test_titan_embedder_rejects_invalid_response(payload: object) -> None:
    embedder = BedrockTitanEmbedder(client=FakeBedrockClient(payload))

    with pytest.raises(ValueError, match="embedding"):
        embedder.embed("valid text")


def test_titan_embedder_can_create_client_lazily(monkeypatch) -> None:
    captured: dict = {}

    class FakeBoto3:
        @staticmethod
        def client(service_name: str, **kwargs):
            captured.update(service_name=service_name, **kwargs)
            return FakeBedrockClient({"embedding": [1.0]})

    monkeypatch.setitem(__import__("sys").modules, "boto3", FakeBoto3)

    embedder = BedrockTitanEmbedder(region_name="us-east-1")

    assert embedder.embed("text") == (1.0,)
    assert captured == {"service_name": "bedrock-runtime", "region_name": "us-east-1"}


def test_titan_embedder_accepts_text_response_body() -> None:
    class TextBodyClient:
        def invoke_model(self, **kwargs):
            return {"body": RawBody('{"embedding": [0.25]}')}

    assert BedrockTitanEmbedder(client=TextBodyClient()).embed("text") == (0.25,)


def test_titan_embedder_rejects_invalid_json() -> None:
    class InvalidJsonClient:
        def invoke_model(self, **kwargs):
            return {"body": RawBody("not-json")}

    with pytest.raises(ValueError):
        BedrockTitanEmbedder(client=InvalidJsonClient()).embed("text")


def test_titan_embedder_propagates_client_errors() -> None:
    class FailingClient:
        def invoke_model(self, **kwargs):
            raise RuntimeError("service unavailable")

    with pytest.raises(RuntimeError, match="service unavailable"):
        BedrockTitanEmbedder(client=FailingClient()).embed("text")


def test_titan_embedder_works_with_semantic_classifier() -> None:
    client = FakeBedrockClient({"embedding": [1.0, 0.0]})
    classifier = SemanticClassifier(
        [Reference("ref-1", "computer_science", "reference")],
        BedrockTitanEmbedder(client=client),
    )

    result = classifier.classify("query")

    assert result.category == "computer_science"
    assert result.reference_id == "ref-1"
    assert len(client.calls) == 2
