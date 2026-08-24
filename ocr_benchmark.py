from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
from ocr import build_reader, recognize


EXTRACTED_DIR = Path("extracted")
RESULTS_PATH = Path("ocr_results.csv")

TILE_NAME_PATTERN = re.compile(
    r"tile_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def clean_prediction(text: str) -> str:
    return re.sub(r"\D", "", text)


def load_existing_labels() -> dict[tuple[str, int], str]:
    """
    Preserve manually entered ground-truth labels when the script is rerun.
    """

    labels: dict[tuple[str, int], str] = {}

    if not RESULTS_PATH.exists():
        return labels

    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            image = row.get("image", "")
            tile_text = row.get("tile", "")
            ground_truth = row.get("ground_truth", "").strip()

            if not image or not tile_text or not ground_truth:
                continue

            try:
                tile_number = int(tile_text)
            except ValueError:
                continue

            labels[(image, tile_number)] = ground_truth

    return labels


def recognize_tile(
    reader,
    tile_path: Path,
) -> tuple[str, float]:
    tile = cv2.imread(str(tile_path))

    if tile is None:
        raise ValueError(f"Could not read tile: {tile_path}")

    result = recognize(reader, tile, "benchmark")
    return result.prediction, result.confidence


def find_tiles() -> list[tuple[str, int, Path]]:
    tiles: list[tuple[str, int, Path]] = []

    for image_directory in sorted(EXTRACTED_DIR.iterdir()):
        if not image_directory.is_dir():
            continue

        image_name = image_directory.name

        for tile_path in sorted(image_directory.iterdir()):
            match = TILE_NAME_PATTERN.fullmatch(tile_path.name)

            if not match:
                continue

            tile_number = int(match.group(1))

            tiles.append(
                (
                    image_name,
                    tile_number,
                    tile_path,
                )
            )

    return tiles


def main() -> None:
    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(
            f"Missing extracted directory: {EXTRACTED_DIR.resolve()}"
        )

    tiles = find_tiles()

    if not tiles:
        raise FileNotFoundError(
            f"No tile images found inside {EXTRACTED_DIR.resolve()}"
        )

    existing_labels = load_existing_labels()

    print("Loading fine-tuned PARSeq...")
    reader = build_reader(gpu=False)

    rows: list[dict[str, object]] = []

    for index, (image_name, tile_number, tile_path) in enumerate(
        tiles,
        start=1,
    ):
        try:
            prediction, confidence = recognize_tile(
                reader,
                tile_path,
            )

            error = ""

        except (ValueError, cv2.error) as exception:
            prediction = ""
            confidence = 0.0
            error = str(exception)

        ground_truth = existing_labels.get(
            (image_name, tile_number),
            "",
        )

        correct = ""

        if ground_truth:
            correct = str(prediction == ground_truth)

        rows.append(
            {
                "image": image_name,
                "tile": tile_number,
                "prediction": prediction,
                "confidence": round(confidence, 6),
                "ground_truth": ground_truth,
                "correct": correct,
                "tile_path": str(tile_path),
                "error": error,
            }
        )

        print(
            f"[{index}/{len(tiles)}] "
            f"{image_name} tile {tile_number}: "
            f"{prediction!r} "
            f"confidence={confidence:.3f}"
        )

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "tile",
                "prediction",
                "confidence",
                "ground_truth",
                "correct",
                "tile_path",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    labeled_rows = [
        row
        for row in rows
        if row["ground_truth"]
    ]

    correct_rows = [
        row
        for row in labeled_rows
        if row["correct"] == "True"
    ]

    print()
    print(f"Processed tiles: {len(rows)}")
    print(f"Results written to: {RESULTS_PATH.resolve()}")

    if labeled_rows:
        accuracy = len(correct_rows) / len(labeled_rows) * 100

        print(f"Labeled tiles: {len(labeled_rows)}")
        print(f"Correct: {len(correct_rows)}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Accuracy unavailable: no ground-truth labels yet.")


if __name__ == "__main__":
    main()
