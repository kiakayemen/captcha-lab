from __future__ import annotations

import csv
import re
from collections.abc import Callable
from pathlib import Path

import cv2
import easyocr
import numpy as np


LABELS_PATH = Path("ocr_results.csv")
RESULTS_PATH = Path("preprocessing_results.csv")
LEADERBOARD_PATH = Path("preprocessing_leaderboard.csv")

DIGITS_ONLY = re.compile(r"\D")


ImageVariant = Callable[[np.ndarray], np.ndarray]


def clean_prediction(text: str) -> str:
    return DIGITS_ONLY.sub("", text)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    EasyOCR accepts grayscale or color images, but returning BGR
    everywhere keeps the variants consistent.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def add_white_padding(
    image: np.ndarray,
    padding: int = 20,
) -> np.ndarray:
    return cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


# -------------------------------------------------------------------
# Preprocessing variants
# -------------------------------------------------------------------

def variant_raw(image: np.ndarray) -> np.ndarray:
    return image.copy()


def variant_upscale_2x(image: np.ndarray) -> np.ndarray:
    return cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )


def variant_upscale_3x(image: np.ndarray) -> np.ndarray:
    return cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )


def variant_gray_clahe(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.resize(
        enhanced,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    return ensure_bgr(enhanced)


def variant_sharpened(image: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    blurred = cv2.GaussianBlur(
        enlarged,
        (0, 0),
        sigmaX=1.2,
    )

    sharpened = cv2.addWeighted(
        enlarged,
        1.8,
        blurred,
        -0.8,
        0,
    )

    return sharpened


def variant_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    return ensure_bgr(thresholded)


def variant_otsu(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    _, thresholded = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return ensure_bgr(thresholded)


def variant_saturation_mask(image: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    hsv = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1]

    saturation = cv2.GaussianBlur(
        saturation,
        (3, 3),
        0,
    )

    _, mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Digits black, background white.
    mask = cv2.bitwise_not(mask)

    return ensure_bgr(mask)


def variant_lab_color_distance(image: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    lab = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2LAB,
    )

    _, a_channel, b_channel = cv2.split(lab)

    neutral = np.full_like(
        a_channel,
        128,
    )

    a_distance = cv2.absdiff(
        a_channel,
        neutral,
    )

    b_distance = cv2.absdiff(
        b_channel,
        neutral,
    )

    color_distance = cv2.addWeighted(
        a_distance,
        0.5,
        b_distance,
        0.5,
        0,
    )

    color_distance = cv2.GaussianBlur(
        color_distance,
        (3, 3),
        0,
    )

    _, mask = cv2.threshold(
        color_distance,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    mask = cv2.bitwise_not(mask)

    return ensure_bgr(mask)


VARIANTS: dict[str, ImageVariant] = {
    "raw": variant_raw,
    "upscale_2x": variant_upscale_2x,
    "upscale_3x": variant_upscale_3x,
    "gray_clahe": variant_gray_clahe,
    "sharpened": variant_sharpened,
    "adaptive_threshold": variant_adaptive_threshold,
    "otsu": variant_otsu,
    "saturation_mask": variant_saturation_mask,
    "lab_color_distance": variant_lab_color_distance,
}


# -------------------------------------------------------------------
# CSV handling
# -------------------------------------------------------------------

def load_labeled_tiles() -> list[dict[str, str]]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {LABELS_PATH.resolve()}"
        )

    with LABELS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    labeled_rows: list[dict[str, str]] = []

    for row in rows:
        ground_truth = row.get(
            "ground_truth",
            "",
        ).strip()

        tile_path = row.get(
            "tile_path",
            "",
        ).strip()

        if (
            len(ground_truth) != 3
            or not ground_truth.isdigit()
            or not tile_path
        ):
            continue

        labeled_rows.append(row)

    if not labeled_rows:
        raise ValueError(
            "No labeled rows with valid tile paths were found."
        )

    return labeled_rows


def load_existing_results() -> dict[
    tuple[str, str, str],
    dict[str, str],
]:
    """
    Makes the benchmark resumable.

    Key:
        image, tile, variant
    """
    existing: dict[
        tuple[str, str, str],
        dict[str, str],
    ] = {}

    if not RESULTS_PATH.exists():
        return existing

    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        for row in csv.DictReader(csv_file):
            key = (
                row["image"],
                row["tile"],
                row["variant"],
            )

            existing[key] = row

    return existing


def save_results(
    rows: list[dict[str, object]],
) -> None:
    fieldnames = [
        "image",
        "tile",
        "variant",
        "ground_truth",
        "prediction",
        "confidence",
        "correct",
        "valid_length",
        "tile_path",
        "error",
    ]

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# -------------------------------------------------------------------
# OCR
# -------------------------------------------------------------------

def recognize(
    reader: easyocr.Reader,
    image: np.ndarray,
) -> tuple[str, float]:
    image = add_white_padding(
        ensure_bgr(image),
        padding=15,
    )

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist="0123456789",
    )

    if not results:
        return "", 0.0

    parsed_results: list[tuple[str, float]] = []

    for result in results:
        text = clean_prediction(
            str(result[1])
        )

        confidence = float(result[2])

        parsed_results.append(
            (text, confidence)
        )

    # Prefer a three-digit result when EasyOCR returns several boxes.
    three_digit_results = [
        result
        for result in parsed_results
        if len(result[0]) == 3
    ]

    candidates = (
        three_digit_results
        if three_digit_results
        else parsed_results
    )

    return max(
        candidates,
        key=lambda result: result[1],
    )


def build_leaderboard(
    results: list[dict[str, object]],
) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}

    for row in results:
        variant = str(row["variant"])

        grouped.setdefault(
            variant,
            [],
        ).append(row)

    leaderboard: list[dict[str, object]] = []

    for variant, rows in grouped.items():
        total = len(rows)

        correct = sum(
            str(row["correct"]) == "True"
            for row in rows
        )

        valid_length = sum(
            str(row["valid_length"]) == "True"
            for row in rows
        )

        blank = sum(
            str(row["prediction"]) == ""
            for row in rows
        )

        accuracy = (
            correct / total * 100
            if total
            else 0.0
        )

        valid_length_rate = (
            valid_length / total * 100
            if total
            else 0.0
        )

        leaderboard.append(
            {
                "variant": variant,
                "tiles": total,
                "correct": correct,
                "incorrect": total - correct,
                "accuracy": round(
                    accuracy,
                    2,
                ),
                "valid_3_digit_predictions": valid_length,
                "valid_length_rate": round(
                    valid_length_rate,
                    2,
                ),
                "blank_predictions": blank,
            }
        )

    leaderboard.sort(
        key=lambda row: (
            float(row["accuracy"]),
            float(row["valid_length_rate"]),
        ),
        reverse=True,
    )

    fieldnames = [
        "variant",
        "tiles",
        "correct",
        "incorrect",
        "accuracy",
        "valid_3_digit_predictions",
        "valid_length_rate",
        "blank_predictions",
    ]

    with LEADERBOARD_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(leaderboard)

    print()
    print("=" * 76)
    print(
        f"{'VARIANT':<24}"
        f"{'ACCURACY':>12}"
        f"{'VALID 3-DIGIT':>18}"
        f"{'BLANKS':>10}"
    )
    print("-" * 76)

    for row in leaderboard:
        print(
            f"{str(row['variant']):<24}"
            f"{float(row['accuracy']):>11.2f}%"
            f"{float(row['valid_length_rate']):>17.2f}%"
            f"{int(row['blank_predictions']):>10}"
        )

    print("=" * 76)
    print(
        f"Detailed results: {RESULTS_PATH.resolve()}"
    )
    print(
        f"Leaderboard: {LEADERBOARD_PATH.resolve()}"
    )


def main() -> None:
    labeled_tiles = load_labeled_tiles()
    existing_results = load_existing_results()

    print(
        f"Loaded {len(labeled_tiles)} labeled tiles."
    )
    print(
        f"Testing {len(VARIANTS)} variants."
    )
    print(
        f"Total OCR attempts: "
        f"{len(labeled_tiles) * len(VARIANTS)}"
    )
    print()
    print("Loading EasyOCR...")

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
    )

    result_map = dict(existing_results)

    completed = len(existing_results)
    total_attempts = (
        len(labeled_tiles)
        * len(VARIANTS)
    )

    for source_row in labeled_tiles:
        image_name = source_row["image"]
        tile_number = source_row["tile"]
        ground_truth = source_row[
            "ground_truth"
        ].strip()

        tile_path = Path(
            source_row["tile_path"]
        )

        original = cv2.imread(
            str(tile_path)
        )

        if original is None:
            print(
                f"Could not read {tile_path}"
            )
            continue

        for variant_name, variant_function in VARIANTS.items():
            key = (
                image_name,
                tile_number,
                variant_name,
            )

            if key in result_map:
                continue

            completed += 1

            try:
                processed = variant_function(
                    original,
                )

                prediction, confidence = recognize(
                    reader,
                    processed,
                )

                error = ""

            except (
                cv2.error,
                ValueError,
            ) as exception:
                prediction = ""
                confidence = 0.0
                error = str(exception)

            correct = (
                prediction == ground_truth
            )

            valid_length = (
                len(prediction) == 3
                and prediction.isdigit()
            )

            result_map[key] = {
                "image": image_name,
                "tile": tile_number,
                "variant": variant_name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence": round(
                    confidence,
                    6,
                ),
                "correct": str(correct),
                "valid_length": str(
                    valid_length
                ),
                "tile_path": str(tile_path),
                "error": error,
            }

            print(
                f"[{completed}/{total_attempts}] "
                f"{image_name} "
                f"tile {tile_number} "
                f"{variant_name}: "
                f"{prediction!r} "
                f"{'✓' if correct else '✗'}"
            )

            # Save regularly so Ctrl+C does not destroy progress.
            if completed % 25 == 0:
                save_results(
                    list(result_map.values())
                )

    all_results = list(
        result_map.values()
    )

    save_results(all_results)
    build_leaderboard(all_results)


if __name__ == "__main__":
    main()
