import math

import pytest

from semantic_paper_router import cosine_similarity


def test_identical_vectors_have_similarity_one() -> None:
    assert cosine_similarity([3, 4], [3, 4]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one() -> None:
    assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "left,right,message",
    [
        ([], [], "empty"),
        ([0, 0], [1, 2], "zero"),
        ([1], [1, 2], "same dimension"),
    ],
)
def test_invalid_vector_shapes_raise_value_error(
    left: list[int], right: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_similarity(left, right)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "1"])
def test_non_finite_or_non_numeric_values_raise_type_error(value: object) -> None:
    with pytest.raises(TypeError, match="finite numeric"):
        cosine_similarity([1, value], [1, 2])  # type: ignore[list-item]
