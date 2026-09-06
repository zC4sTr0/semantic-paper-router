# Semantic Paper Router

Semantic classification of scientific paper abstracts into broad research areas, built from first principles as a public learning project.

## Problem

Given the abstract of a scientific paper, decide which broad area it belongs to — initially **Computer Science**, **Biology**, or **Economics**.

Keyword matching is brittle: the same idea can be expressed with completely different words. We want to classify by *meaning*, not by vocabulary.

## Intended architecture

The system will work conceptually as follows:

1. A small set of reference texts, each already labeled with a category.
2. An embedding model converts each reference text into a numeric vector.
3. A new abstract is converted into a vector using the same model.
4. The new vector is compared against the reference vectors using cosine similarity.
5. The category of the closest reference vector is returned.

The first local version now covers the vector, corpus, and classification
pieces. It still uses a lexical baseline rather than a trained embedding
model.

## Current status

**Phase 1–3 — local prototype**

The package is importable and has a dependency-free cosine implementation, a
JSON corpus with nine labeled references, a deterministic bag-of-words
baseline, and a nearest-reference classifier. The baseline is useful for
testing the flow, but it does not understand synonyms or broader meaning.

## Learning goals

- Understand what vectors and embeddings are, and why semantically similar texts end up close together in vector space.
- Implement cosine similarity from scratch.
- Call a real embedding model (Amazon Bedrock Titan Embeddings via boto3).
- Combine the pieces into a small semantic classifier.
- Evolve toward vector databases, retrieval interfaces (MCP), and agent/evaluation layers — one step at a time, understanding each layer before adding the next.

## Roadmap

```text
Phase 0 — Repository bootstrap                         [done]
Phase 1 — Local vectors and cosine similarity           [done]
Phase 2 — Amazon Bedrock embeddings                     [next]
Phase 3 — Semantic classification                      [baseline done]
Phase 4 — Automated tests and CI
Phase 5 — Vector database
Phase 6 — MCP retrieval interface
Phase 7 — Agent/evaluation layer
```

Status is tracked honestly: a phase is only "done" when it actually works.

## Project layout

```text
semantic-paper-router/
├── README.md
├── LEARNING.md      # personal study notes (pt-BR), not for recruiters
├── data/            # labeled reference corpus
├── src/             # package source
├── tests/           # test suite (21 tests)
├── .gitignore
└── pyproject.toml
```

## License

Not yet defined.
