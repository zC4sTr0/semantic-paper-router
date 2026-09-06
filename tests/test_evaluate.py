import pytest

from scripts.evaluate import evaluate_cases


def test_evaluate_cases_reports_accuracy_and_unknown_counts() -> None:
    rows = [
        {"id": "one", "text": "one", "expected_label": "a"},
        {"id": "two", "text": "two", "expected_label": "out_of_scope"},
    ]

    class FakeClassifier:
        def classify_with_scope(self, text, policy):
            class Decision:
                category = "a" if text == "one" else "unknown"
                in_scope = text == "one"
                reason = "accepted" if text == "one" else "score_below_threshold"
                classification = type("Result", (), {"score": 0.8, "margin": 0.4})()
            return Decision()

    report = evaluate_cases(FakeClassifier(), rows, min_score=0.5, min_margin=0.1)

    assert report["counts"] == {"evaluated": 2, "correct": 2, "incorrect": 0}
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["unknown"]["predicted"] == 1
