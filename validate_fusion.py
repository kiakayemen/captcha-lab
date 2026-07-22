from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from fusion import select_highest_confidence
from ocr import OCRResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=Path("preprocessing_results.csv"),
    )
    args = parser.parse_args()

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with args.results.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["image"], int(row["tile"]))].append(row)

    correct = 0
    no_prediction = 0

    for rows in grouped.values():
        ground_truth = rows[0]["ground_truth"].strip()
        attempts = [
            OCRResult(
                variant=row["variant"],
                prediction=row["prediction"].strip(),
                confidence=float(row["confidence"] or 0.0),
            )
            for row in rows
        ]
        decision = select_highest_confidence(attempts)
        correct += int(decision.prediction == ground_truth)
        no_prediction += int(not decision.prediction)

    total = len(grouped)
    print(f"Tiles: {total}")
    print(f"Correct: {correct}/{total} ({correct / total:.2%})")
    print(f"No prediction: {no_prediction}")


if __name__ == "__main__":
    main()
