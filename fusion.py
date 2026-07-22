from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from ocr import OCRResult


@dataclass(frozen=True)
class FusionDecision:
    prediction: str
    score: float
    votes: int
    supporting_variants: tuple[str, ...]


def select_highest_confidence(results: Iterable[OCRResult]) -> FusionDecision:
    """Reproduce the best measured deterministic strategy (467/513)."""
    grouped: dict[str, list[OCRResult]] = defaultdict(list)
    for result in results:
        if result.valid:
            grouped[result.prediction].append(result)

    if not grouped:
        return FusionDecision("", 0.0, 0, ())

    def rank(item: tuple[str, list[OCRResult]]) -> tuple[float, int, float, str]:
        prediction, matches = item
        confidences = [match.confidence for match in matches]
        # Same ordering used by the historical fusion benchmark:
        # highest confidence, then votes, then summed confidence,
        # then deterministic lexical preference.
        return (
            max(confidences),
            len(matches),
            sum(confidences),
            "".join(chr(255 - ord(char)) for char in prediction),
        )

    prediction, matches = max(grouped.items(), key=rank)
    return FusionDecision(
        prediction=prediction,
        score=max(match.confidence for match in matches),
        votes=len(matches),
        supporting_variants=tuple(sorted(match.variant for match in matches)),
    )
