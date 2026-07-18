from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


RESULTS_PATH = Path("preprocessing_results.csv")

SELECTED_VARIANTS = [
    "gray_clahe",
    "raw",
    "upscale_2x",
    "upscale_3x",
    "sharpened",
]


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RESULTS_PATH.resolve()}"
        )

    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    grouped: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            row["image"],
            row["tile"],
        )
        grouped[key].append(row)

    total = len(grouped)

    raw_or_clahe_correct = 0
    oracle_selected_correct = 0
    oracle_all_correct = 0
    majority_correct = 0
    unanimous_correct = 0
    unanimous_wrong = 0
    disagreements = 0

    for tile_rows in grouped.values():
        by_variant = {
            row["variant"]: row
            for row in tile_rows
        }

        ground_truth = tile_rows[0]["ground_truth"]

        raw = by_variant.get("raw")
        clahe = by_variant.get("gray_clahe")

        if (
            raw
            and clahe
            and (
                raw["prediction"] == ground_truth
                or clahe["prediction"] == ground_truth
            )
        ):
            raw_or_clahe_correct += 1

        selected_rows = [
            by_variant[name]
            for name in SELECTED_VARIANTS
            if name in by_variant
        ]

        if any(
            row["prediction"] == ground_truth
            for row in selected_rows
        ):
            oracle_selected_correct += 1

        if any(
            row["prediction"] == ground_truth
            for row in tile_rows
        ):
            oracle_all_correct += 1

        predictions = [
            row["prediction"]
            for row in selected_rows
            if len(row["prediction"]) == 3
            and row["prediction"].isdigit()
        ]

        if not predictions:
            continue

        counts = Counter(predictions)
        majority_prediction, majority_count = counts.most_common(1)[0]

        if majority_prediction == ground_truth:
            majority_correct += 1

        unique_predictions = set(predictions)

        if len(unique_predictions) > 1:
            disagreements += 1
        else:
            if majority_prediction == ground_truth:
                unanimous_correct += 1
            else:
                unanimous_wrong += 1

    def percentage(value: int) -> float:
        return value / total * 100 if total else 0.0

    print(f"Tiles analyzed: {total}")
    print()

    print(
        "Raw OR CLAHE oracle: "
        f"{raw_or_clahe_correct}/{total} "
        f"({percentage(raw_or_clahe_correct):.2f}%)"
    )

    print(
        "Selected-variant oracle: "
        f"{oracle_selected_correct}/{total} "
        f"({percentage(oracle_selected_correct):.2f}%)"
    )

    print(
        "All-variant oracle: "
        f"{oracle_all_correct}/{total} "
        f"({percentage(oracle_all_correct):.2f}%)"
    )

    print(
        "Majority vote: "
        f"{majority_correct}/{total} "
        f"({percentage(majority_correct):.2f}%)"
    )

    print()
    print(f"Tiles with disagreement: {disagreements}")
    print(f"Unanimous and correct: {unanimous_correct}")
    print(f"Unanimous but wrong: {unanimous_wrong}")


if __name__ == "__main__":
    main()
