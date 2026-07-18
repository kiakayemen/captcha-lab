from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FUSION_RESULTS = Path("fusion_results.csv")
DEFAULT_PREPROCESSING_RESULTS = Path("preprocessing_results.csv")
DEFAULT_OUTPUT = Path("fusion_error_analysis.csv")
DEFAULT_ERROR_DIR = Path("fusion_errors")
DEFAULT_STRATEGY = "highest_confidence"


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def safe_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_valid_prediction(value: str) -> bool:
    return len(value) == 3 and value.isdigit()


def load_raw_predictions(
    path: Path,
) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image",
            "tile",
            "variant",
            "prediction",
            "confidence",
            "ground_truth",
            "tile_path",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            grouped[(row["image"], int(row["tile"]))].append(row)

    return grouped


def classify_error(
    fusion_prediction: str,
    ground_truth: str,
    raw_rows: list[dict[str, str]],
) -> tuple[str, int, int, float, float]:
    valid_rows = [
        row for row in raw_rows
        if is_valid_prediction((row.get("prediction") or "").strip())
    ]
    correct_rows = [
        row for row in valid_rows
        if row["prediction"].strip() == ground_truth
    ]

    selected_confidence = max(
        (
            safe_float(row.get("confidence"))
            for row in valid_rows
            if row["prediction"].strip() == fusion_prediction
        ),
        default=0.0,
    )
    best_correct_confidence = max(
        (safe_float(row.get("confidence")) for row in correct_rows),
        default=0.0,
    )

    if not valid_rows:
        category = "no_valid_prediction"
    elif not correct_rows:
        category = "oracle_failure"
    elif selected_confidence >= 0.90:
        category = "confidently_wrong_despite_correct_candidate"
    else:
        category = "fusion_selection_error"

    return (
        category,
        len(valid_rows),
        len(correct_rows),
        selected_confidence,
        best_correct_confidence,
    )


def copy_error_tile(
    raw_rows: list[dict[str, str]],
    output_dir: Path,
    image: str,
    tile: int,
) -> str:
    tile_path_text = next(
        (
            (row.get("tile_path") or "").strip()
            for row in raw_rows
            if (row.get("tile_path") or "").strip()
        ),
        "",
    )
    if not tile_path_text:
        return ""

    source = Path(tile_path_text)
    if not source.exists():
        return ""

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix or ".png"
    destination = output_dir / f"{image}_tile-{tile}{suffix}"
    shutil.copy2(source, destination)
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Categorize and export the remaining OCR fusion errors."
    )
    parser.add_argument(
        "--fusion-results",
        type=Path,
        default=DEFAULT_FUSION_RESULTS,
    )
    parser.add_argument(
        "--preprocessing-results",
        type=Path,
        default=DEFAULT_PREPROCESSING_RESULTS,
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--error-dir",
        type=Path,
        default=DEFAULT_ERROR_DIR,
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy wrong tile images into the error directory.",
    )
    args = parser.parse_args()

    raw_by_tile = load_raw_predictions(args.preprocessing_results)
    output_rows: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()

    with args.fusion_results.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["strategy"] != args.strategy:
                continue
            if as_bool(row["correct"]):
                continue

            key = (row["image"], int(row["tile"]))
            raw_rows = raw_by_tile.get(key, [])
            (
                category,
                valid_count,
                correct_candidate_count,
                selected_confidence,
                best_correct_confidence,
            ) = classify_error(
                row["fusion_prediction"],
                row["ground_truth"],
                raw_rows,
            )
            category_counts[category] += 1

            candidate_summary = "; ".join(
                f"{raw['variant']}={raw['prediction']}"
                f"({safe_float(raw.get('confidence')):.3f})"
                for raw in sorted(
                    raw_rows,
                    key=lambda item: safe_float(item.get("confidence")),
                    reverse=True,
                )
            )

            copied_path = ""
            if not args.no_copy:
                copied_path = copy_error_tile(
                    raw_rows,
                    args.error_dir,
                    row["image"],
                    int(row["tile"]),
                )

            output_rows.append(
                {
                    "image": row["image"],
                    "tile": row["tile"],
                    "tile_path": row["tile_path"],
                    "copied_tile_path": copied_path,
                    "ground_truth": row["ground_truth"],
                    "fusion_prediction": row["fusion_prediction"],
                    "category": category,
                    "selected_confidence": f"{selected_confidence:.6f}",
                    "best_correct_confidence": (
                        f"{best_correct_confidence:.6f}"
                    ),
                    "valid_prediction_count": valid_count,
                    "correct_candidate_count": correct_candidate_count,
                    "all_variant_predictions": candidate_summary,
                }
            )

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image",
            "tile",
            "tile_path",
            "copied_tile_path",
            "ground_truth",
            "fusion_prediction",
            "category",
            "selected_confidence",
            "best_correct_confidence",
            "valid_prediction_count",
            "correct_candidate_count",
            "all_variant_predictions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Strategy: {args.strategy}")
    print(f"Wrong/no-prediction tiles: {len(output_rows)}")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count}")
    print()
    print(f"Analysis CSV: {args.output}")
    if not args.no_copy:
        print(f"Copied error tiles: {args.error_dir}")


if __name__ == "__main__":
    main()
