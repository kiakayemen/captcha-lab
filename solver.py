from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from fusion import select_highest_confidence
from ocr import OCRResult, recognize
from preprocess import preprocessing_variants


def solve_tile(
    tile: np.ndarray,
    reader: Any,
    target: str | None = None,
) -> dict[str, object]:
    if target is not None and (len(target) != 3 or not target.isdigit()):
        raise ValueError("target must be exactly three digits")

    attempts: list[OCRResult] = []
    for variant_name, processed in preprocessing_variants(tile).items():
        attempts.append(recognize(reader, processed, variant_name))

    decision = select_highest_confidence(attempts)

    return {
        "prediction": decision.prediction,
        "score": decision.score,
        "votes": decision.votes,
        "supporting_variants": list(decision.supporting_variants),
        "matches_target": (
            decision.prediction == target if target is not None else None
        ),
        "uncertain": decision.prediction == "",
        "attempts": [asdict(attempt) for attempt in attempts],
    }
