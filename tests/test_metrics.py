import pytest

from meet_in_multi_language.metrics import character_error_rate, edit_distance, word_error_rate


def test_edit_distance_counts_insertions_deletions_and_substitutions() -> None:
    assert edit_distance(list("kitten"), list("sitting")) == 3


def test_character_error_rate_normalizes_spacing_and_punctuation() -> None:
    assert character_error_rate("逐字稿，完成！", "逐字稿完成") == 0
    assert character_error_rate("逐字稿", "逐字搞") == pytest.approx(1 / 3)


def test_word_error_rate_is_case_insensitive() -> None:
    assert word_error_rate("Deploy the API", "deploy API") == pytest.approx(1 / 3)


def test_empty_reference_has_defined_result() -> None:
    assert character_error_rate("", "") == 0
    assert character_error_rate("", "額外內容") == 1

