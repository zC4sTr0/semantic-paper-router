"""Offline evaluation harness for the paper router."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from semantic_paper_router import (
    BagOfWordsEmbedder,
    ScopePolicy,
    SemanticClassifier,
    load_references,
)


def evaluate_cases(
    classifier: Any,
    cases: list[dict[str, str]],
    *,
    min_score: float,
    min_margin: float,
) -> dict[str, Any]:
    policy = ScopePolicy(min_score=min_score, min_margin=min_margin)
    predictions: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0

    for case in cases:
        decision = classifier.classify_with_scope(case["text"], policy)
        predicted = decision.category
        expected = case["expected_label"]
        correct += predicted == expected or (
            expected == "out_of_scope" and predicted == "unknown"
        )
        confusion[expected][predicted] += 1
        predictions.append(
            {
                "id": case["id"],
                "expected": expected,
                "predicted": predicted,
                "in_scope": decision.in_scope,
                "reason": decision.reason,
                "score": decision.classification.score,
                "margin": decision.classification.margin,
            }
        )

    evaluated = len(cases)
    labels = sorted({case["expected_label"] for case in cases} | {"unknown"})
    matrix = {
        label: {predicted: confusion[label][predicted] for predicted in labels}
        for label in labels
    }
    per_class = {}
    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[other][label] for other in labels if other != label)
        false_negative = sum(matrix[label][other] for other in labels if other != label)
        per_class[label] = {
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            ),
            "recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            ),
        }
    return {
        "counts": {
            "evaluated": evaluated,
            "correct": correct,
            "incorrect": evaluated - correct,
        },
        "accuracy": correct / evaluated if evaluated else 0.0,
        "unknown": {
            "expected": sum(case["expected_label"] == "out_of_scope" for case in cases),
            "predicted": sum(row["predicted"] == "unknown" for row in predictions),
        },
        "confusion_matrix": matrix,
        "per_class": per_class,
        "predictions": predictions,
    }


def build_classifier(reference_path: Path) -> SemanticClassifier:
    references = load_references(reference_path)
    embedder = BagOfWordsEmbedder.fit(reference.text for reference in references)
    return SemanticClassifier(references, embedder)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = json.loads(args.examples.read_text(encoding="utf-8"))
    report = evaluate_cases(
        build_classifier(args.references),
        cases,
        min_score=args.min_score,
        min_margin=args.min_margin,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.format == "text":
        rendered = (
            f"accuracy: {report['accuracy']:.3f}\n"
            f"evaluated: {report['counts']['evaluated']}\n"
            f"correct: {report['counts']['correct']}\n"
            f"incorrect: {report['counts']['incorrect']}\n"
            f"unknown expected/predicted: {report['unknown']['expected']}/"
            f"{report['unknown']['predicted']}\n"
            f"confusion_matrix: {json.dumps(report['confusion_matrix'], ensure_ascii=False)}\n"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
