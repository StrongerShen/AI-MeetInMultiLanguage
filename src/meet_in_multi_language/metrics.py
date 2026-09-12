from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence


_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, 1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_PUNCTUATION.sub(" ", normalized).split())


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_units = list(_normalized_text(reference).replace(" ", ""))
    hypothesis_units = list(_normalized_text(hypothesis).replace(" ", ""))
    if not reference_units:
        return 0.0 if not hypothesis_units else 1.0
    return edit_distance(reference_units, hypothesis_units) / len(reference_units)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_units = _normalized_text(reference).split()
    hypothesis_units = _normalized_text(hypothesis).split()
    if not reference_units:
        return 0.0 if not hypothesis_units else 1.0
    return edit_distance(reference_units, hypothesis_units) / len(reference_units)

