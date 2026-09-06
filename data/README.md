# data/

Reference corpus directory.

This directory holds **nine short scientific reference texts** used as the
labeled examples for the first local classifier:

- 3 Computer Science
- 3 Biology
- 3 Economics

Each text is stored with an id and category label in `references.json`. The
current baseline converts the texts into bag-of-words vectors for comparison.
Those vectors are a local test baseline, not the final semantic model.
