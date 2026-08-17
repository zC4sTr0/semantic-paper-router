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

This is the target shape of the system. Only Phase 0 exists today.

## Current status

**Phase 0 — repository bootstrap**

Nothing is implemented yet: no embeddings, no similarity, no classification, no external services. This repository currently contains only the project skeleton (package layout, docs, tooling config).

## Learning goals

- Understand what vectors and embeddings are, and why semantically similar texts end up close together in vector space.
- Implement cosine similarity from scratch.
- Call a real embedding model (Amazon Bedrock Titan Embeddings via boto3).
- Combine the pieces into a small semantic classifier.
- Evolve toward vector databases, retrieval interfaces (MCP), and agent/evaluation layers — one step at a time, understanding each layer before adding the next.

## Roadmap

```text
Phase 0 — Repository bootstrap
Phase 1 — Local vectors and cosine similarity
Phase 2 — Amazon Bedrock embeddings
Phase 3 — Semantic classification
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
├── data/            # reference corpus (empty in Phase 0)
├── src/             # package source
├── tests/           # test suite
├── .gitignore
└── pyproject.toml
```

## License

Not yet defined.
