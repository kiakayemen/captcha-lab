from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class OCRResult:
    variant: str
    prediction: str
    confidence: float


def _is_valid_prediction(value: str) -> bool:
    return len(value) == 3 and value.isdigit()


def candidate_features(
    results: Iterable[OCRResult],
    candidate: str,
    variants: list[str],
) -> dict[str, float]:
    matches = [r for r in results if r.prediction == candidate]
    confidences = [r.confidence for r in matches]

    features: dict[str, float] = {
        "vote_count": float(len(matches)),
        "confidence_sum": float(sum(confidences)),
        "confidence_max": float(max(confidences, default=0.0)),
        "confidence_mean": float(np.mean(confidences) if confidences else 0.0),
        "valid_three_digits": float(_is_valid_prediction(candidate)),
    }

    by_variant = {r.variant: r for r in matches}
    for variant in variants:
        result = by_variant.get(variant)
        features[f"{variant}__present"] = float(result is not None)
        features[f"{variant}__confidence"] = result.confidence if result else 0.0

    return features


class FusionSelector:
    def __init__(self, model: Pipeline, variants: list[str], feature_names: list[str]):
        self.model = model
        self.variants = variants
        self.feature_names = feature_names

    def predict(self, results: Iterable[OCRResult]) -> tuple[str, float]:
        results = [r for r in results if r.prediction]
        candidates = sorted({r.prediction for r in results})
        if not candidates:
            return "", 0.0

        frame = pd.DataFrame(
            [candidate_features(results, candidate, self.variants) for candidate in candidates]
        ).reindex(columns=self.feature_names, fill_value=0.0)

        probabilities = self.model.predict_proba(frame)[:, 1]
        best_index = int(np.argmax(probabilities))
        return candidates[best_index], float(probabilities[best_index])

    def save(self, path: Path) -> None:
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
        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            variants=payload["variants"],
            feature_names=payload["feature_names"],
        )


def load_benchmark(path: Path):
    grouped: dict[tuple[str, str, str, str], list[OCRResult]] = defaultdict(list)
    variants: set[str] = set()

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            variants.add(variant)
            grouped[(row["image"], row["tile"], row["ground_truth"], row["tile_path"])].append(
                OCRResult(
                    variant=variant,
                    prediction=row["prediction"].strip(),
                    confidence=float(row["confidence"] or 0.0),
                )
            )

    return grouped, sorted(variants)


def build_training_table(grouped, variants: list[str]):
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    metadata: list[tuple[tuple[str, str, str, str], str]] = []

    for key, results in grouped.items():
        ground_truth = key[2]
        for candidate in sorted({r.prediction for r in results if r.prediction}):
            rows.append(candidate_features(results, candidate, variants))
            labels.append(int(candidate == ground_truth))
            metadata.append((key, candidate))

    frame = pd.DataFrame(rows).fillna(0.0)
    return frame, np.asarray(labels), metadata


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def cross_validated_predictions(frame, labels, metadata, folds: int = 5):
    image_groups = np.asarray([key[0] for key, _ in metadata])
    probabilities = np.zeros(len(frame), dtype=float)

    splitter = GroupKFold(n_splits=folds)
    for train_index, test_index in splitter.split(frame, labels, image_groups):
        model = make_model()
        model.fit(frame.iloc[train_index], labels[train_index])
        probabilities[test_index] = model.predict_proba(frame.iloc[test_index])[:, 1]

    selected: dict[tuple[str, str, str, str], tuple[float, str]] = {}
    for probability, (key, candidate) in zip(probabilities, metadata):
        if key not in selected or probability > selected[key][0]:
            selected[key] = (float(probability), candidate)

    return selected


def confidence_sum_prediction(results: list[OCRResult]) -> str:
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        if result.prediction:
            scores[result.prediction] += result.confidence
    return max(scores, key=scores.get) if scores else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate OCR prediction fusion.")
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--model-out", type=Path, default=Path("fusion_model.joblib"))
    parser.add_argument("--predictions-out", type=Path, default=Path("fusion_predictions.csv"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    grouped, variants = load_benchmark(args.results_csv)
    frame, labels, metadata = build_training_table(grouped, variants)

    selected = cross_validated_predictions(frame, labels, metadata, folds=args.folds)
    learned_correct = sum(
        key in selected and selected[key][1] == key[2] for key in grouped
    )
    baseline_correct = sum(
        confidence_sum_prediction(results) == key[2] for key, results in grouped.items()
    )
    total = len(grouped)

    final_model = make_model()
    final_model.fit(frame, labels)
    selector = FusionSelector(final_model, variants, list(frame.columns))
    selector.save(args.model_out)

    with args.predictions_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "tile",
                "tile_path",
                "ground_truth",
                "fusion_prediction",
                "fusion_score",
                "correct",
            ],
        )
        writer.writeheader()
        for key in sorted(grouped):
            probability, prediction = selected.get(key, (0.0, ""))
            writer.writerow(
                {
                    "image": key[0],
                    "tile": key[1],
                    "tile_path": key[3],
                    "ground_truth": key[2],
                    "fusion_prediction": prediction,
                    "fusion_score": f"{probability:.6f}",
                    "correct": prediction == key[2],
                }
            )

    print(f"Tiles: {total}")
    print(f"Confidence-sum baseline: {baseline_correct}/{total} ({baseline_correct / total:.2%})")
    print(f"Cross-validated learned fusion: {learned_correct}/{total} ({learned_correct / total:.2%})")
    print(f"Saved model: {args.model_out}")
    print(f"Saved predictions: {args.predictions_out}")


if __name__ == "__main__":
    main()
