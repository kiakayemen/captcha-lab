from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class FusionDecision:
    prediction: str
    score: float
    votes: int
    supporting_variants: tuple[str, ...]

@dataclass(frozen=True)
class OCRResult:
    """
    One OCR prediction produced by a preprocessing variant.

    This is the runtime input format expected by the learned fusion selector.
    """

    variant: str
    prediction: str
    confidence: float


def learned_candidate_features(
    results: Iterable[OCRResult],
    candidate: str,
    variants: list[str],
) -> dict[str, float]:
    """
    Build the exact feature row expected by the trained fusion model.

    Each possible OCR candidate receives aggregate vote and confidence
    features, plus presence and confidence features for every preprocessing
    variant used during model training.
    """

    matches = [
        result
        for result in results
        if result.prediction == candidate
    ]
    confidences = [result.confidence for result in matches]

    features: dict[str, float] = {
        "vote_count": float(len(matches)),
        "confidence_sum": float(sum(confidences)),
        "confidence_max": float(max(confidences, default=0.0)),
        "confidence_mean": float(
            np.mean(confidences) if confidences else 0.0
        ),
        "valid_three_digits": float(
            len(candidate) == 3 and candidate.isdigit()
        ),
    }

    by_variant = {
        result.variant: result
        for result in matches
    }

    for variant in variants:
        result = by_variant.get(variant)
        features[f"{variant}__present"] = float(result is not None)
        features[f"{variant}__confidence"] = (
            result.confidence if result is not None else 0.0
        )

    return features


class FusionSelector:
    """
    Runtime wrapper around the frozen learned fusion model.

    The underlying scikit-learn pipeline scores every distinct OCR candidate.
    The candidate with the highest predicted probability is returned as the
    final tile prediction.
    """

    def __init__(
        self,
        model: Pipeline,
        variants: list[str],
        feature_names: list[str],
    ) -> None:
        self.model = model
        self.variants = variants
        self.feature_names = feature_names

    def predict(
        self,
        results: Iterable[OCRResult],
    ) -> tuple[str, float]:
        """
        Select the most likely number from a tile's OCR results.

        Returns:
            A tuple containing the selected prediction and its learned
            probability score. Empty input produces ("", 0.0).
        """

        usable_results = [
            result
            for result in results
            if result.prediction
        ]

        candidates = sorted(
            {
                result.prediction
                for result in usable_results
            }
        )

        if not candidates:
            return "", 0.0

        feature_rows = [
            learned_candidate_features(
                usable_results,
                candidate,
                self.variants,
            )
            for candidate in candidates
        ]

        frame = pd.DataFrame(feature_rows).reindex(
            columns=self.feature_names,
            fill_value=0.0,
        )

        probabilities = self.model.predict_proba(frame)[:, 1]
        best_index = int(np.argmax(probabilities))

        return (
            candidates[best_index],
            float(probabilities[best_index]),
        )

    def save(self, path: Path) -> None:
        """
        Save the model and its feature metadata in the established format.
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "variants": self.variants,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "FusionSelector":
        """
        Load a frozen fusion model from disk.

        Raises a clear error when the artifact is missing or does not contain
        the metadata expected by the production runtime.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"Fusion model was not found: {path}"
            )

        payload = joblib.load(path)

        required_keys = {
            "model",
            "variants",
            "feature_names",
        }
        missing_keys = required_keys - set(payload)

        if missing_keys:
            raise ValueError(
                "Fusion model artifact is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        return cls(
            model=payload["model"],
            variants=list(payload["variants"]),
            feature_names=list(payload["feature_names"]),
        )


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
