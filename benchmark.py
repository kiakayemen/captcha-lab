from __future__ import annotations
from main import clean_prediction, preprocessing_variants

import csv
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np


def intersection_over_union(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    intersection_width = max(0, right - left)
    intersection_height = max(0, bottom - top)
    intersection_area = intersection_width * intersection_height

    area_a = aw * ah
    area_b = bw * bh

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def remove_duplicate_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    Contour detection often finds the inner and outer edge
    of the same rectangle. This removes overlapping duplicates.
    """
    boxes = sorted(
        boxes,
        key=lambda box: box[2] * box[3],
        reverse=True,
    )

    kept: list[tuple[int, int, int, int]] = []

    for box in boxes:
        duplicate = any(
            intersection_over_union(box, existing) > 0.7
            for existing in kept
        )

        if not duplicate:
            kept.append(box)

    return kept


def detect_tile_boxes(
    image: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        40,
        120,
    )

    # Join small gaps in the square borders.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width

    candidates: list[tuple[int, int, int, int]] = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        approximation = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True,
        )

        # Tile borders should be approximately rectangular.
        if len(approximation) != 4:
            continue

        x, y, width, height = cv2.boundingRect(approximation)

        box_area = width * height

        # Ignore tiny objects and huge screenshot-sized contours.
        if box_area < image_area * 0.005:
            continue

        if box_area > image_area * 0.15:
            continue

        aspect_ratio = width / height

        # Tiles should be roughly square.
        if not 0.75 <= aspect_ratio <= 1.25:
            continue

        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / box_area

        if rectangularity < 0.65:
            continue

        candidates.append((x, y, width, height))

    candidates = remove_duplicate_boxes(candidates)

    if len(candidates) < 9:
        raise RuntimeError(
            f"Only found {len(candidates)} possible tiles. "
            "The edge thresholds may need adjusting."
        )

    # The nine tiles should have similar dimensions.
    widths = np.array([box[2] for box in candidates])
    heights = np.array([box[3] for box in candidates])

    median_width = float(np.median(widths))
    median_height = float(np.median(heights))

    candidates.sort(
        key=lambda box: (
            abs(box[2] - median_width)
            + abs(box[3] - median_height)
        )
    )

    boxes = candidates[:9]

    # First sort approximately from top to bottom.
    boxes.sort(
        key=lambda box: box[1] + box[3] / 2,
    )

    ordered: list[tuple[int, int, int, int]] = []

    # Split into three rows, then sort each row left to right.
    for row_start in range(0, 9, 3):
        row = boxes[row_start: row_start + 3]

        row.sort(
            key=lambda box: box[0] + box[2] / 2,
        )

        ordered.extend(row)

    return ordered


def crop_detected_tiles(
    image: np.ndarray,
) -> list[np.ndarray]:
    boxes = detect_tile_boxes(image)

    tiles: list[np.ndarray] = []

    for x, y, width, height in boxes:
        # Small inward margin removes most of the tile border.
        margin_x = max(3, int(width * 0.04))
        margin_y = max(3, int(height * 0.04))

        tile = image[
            y + margin_y: y + height - margin_y,
            x + margin_x: x + width - margin_x,
        ]

        tiles.append(tile)

    return tiles


DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("output")
RESULTS_PATH = Path("preprocessing_results.csv")
LEADERBOARD_PATH = Path("preprocessing_leaderboard.csv")
LABELS_PATH = Path("labels.csv")
GPU = False

# Keep preprocessing centralized: benchmark and live runtime must use the same
# variant names and image transformations.


def recognize_tile(
    reader: easyocr.Reader,
    tile: np.ndarray,
) -> tuple[str, float]:
    results = reader.readtext(
        tile,
        detail=1,
        paragraph=False,
        allowlist="0123456789",
    )

    if not results:
        return "", 0.0

    cleaned = [
        (clean_prediction(str(item[1])), float(item[2]))
        for item in results
    ]
    valid = [item for item in cleaned if len(item[0]) == 3]
    return max(valid or cleaned, key=lambda item: item[1])


def load_labels() -> dict[tuple[str, int], str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {LABELS_PATH}. Expected columns: image,tile,ground_truth"
        )

    labels: dict[tuple[str, int], str] = {}
    with LABELS_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "tile", "ground_truth"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{LABELS_PATH} must contain columns: {sorted(required)}"
            )

        for row in reader:
            labels[(row["image"], int(row["tile"]))
                   ] = row["ground_truth"].strip()

    return labels


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    image_paths = sorted(
        path
        for path in DATASET_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No images found inside: {DATASET_DIR.resolve()}")

    print("Loading EasyOCR model...")
    reader = easyocr.Reader(["en"], gpu=GPU)

    rows: list[dict[str, object]] = []
    variant_totals: dict[str, int] = {}
    variant_correct: dict[str, int] = {}

    for image_index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        try:
            tiles = crop_detected_tiles(image)
        except (ValueError, RuntimeError) as error:
            print(f"Skipping {image_path.name}: {error}")
            continue

        print(f"[{image_index}/{len(image_paths)}] {image_path.name}")

        for tile_index, tile in enumerate(tiles, start=1):
            ground_truth = labels.get((image_path.name, tile_index), "")
            if not ground_truth:
                print(f"  Tile {tile_index}: missing label; skipped")
                continue

            tile_path = OUTPUT_DIR / f"{image_path.stem}_tile_{tile_index}.png"
            cv2.imwrite(str(tile_path), tile)

            for variant, processed in preprocessing_variants(tile).items():
                prediction, confidence = recognize_tile(reader, processed)
                correct = prediction == ground_truth

                variant_totals[variant] = variant_totals.get(variant, 0) + 1
                variant_correct[variant] = variant_correct.get(
                    variant, 0) + int(correct)

                rows.append(
                    {
                        "image": image_path.name,
                        "tile": tile_index,
                        "tile_path": str(tile_path),
                        "variant": variant,
                        "prediction": prediction,
                        "confidence": round(confidence, 6),
                        "ground_truth": ground_truth,
                        "correct": correct,
                    }
                )

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "tile",
                "tile_path",
                "variant",
                "prediction",
                "confidence",
                "ground_truth",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    leaderboard = sorted(
        (
            {
                "variant": variant,
                "correct": variant_correct[variant],
                "total": total,
                "accuracy": variant_correct[variant] / total if total else 0.0,
            }
            for variant, total in variant_totals.items()
        ),
        key=lambda row: row["accuracy"],
        reverse=True,
    )

    with LEADERBOARD_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "correct", "total", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(leaderboard)

    print(f"\nLabeled tiles: {len(rows) // max(len(variant_totals), 1)}")
    for row in leaderboard:
        print(
            f"{row['variant']:<22} "
            f"{row['correct']}/{row['total']} ({row['accuracy']:.2%})"
        )
    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Leaderboard: {LEADERBOARD_PATH}")


if __name__ == "__main__":
    main()
