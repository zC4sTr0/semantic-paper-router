import pytest

from semantic_paper_router import BagOfWordsEmbedder


def test_bag_of_words_has_stable_fitted_vocabulary() -> None:
    embedder = BagOfWordsEmbedder.fit(["Neural network", "protein folding"])

    assert embedder.embed("neural neural") == [0.0, 0.0, 2.0, 0.0]


def test_unknown_text_becomes_zero_vector() -> None:
    embedder = BagOfWordsEmbedder(["known"])

    assert embedder.embed("unseen") == [0.0]


def test_empty_vocabulary_is_invalid() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        BagOfWordsEmbedder([])


def test_duplicate_vocabulary_is_invalid() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        BagOfWordsEmbedder(["known", "known"])
