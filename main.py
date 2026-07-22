from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import easyocr
import numpy as np

from fusion import FusionSelector, OCRResult

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIR / "images" / "captcha.png"
OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_PATH = PROJECT_ROOT / "models" / "fusion_model.joblib"
RESULTS_PATH = OUTPUT_DIR / "fused_predictions.csv"

# Coordinates of the complete 3x3 grid in images/captcha.png.
Y1, Y2 = 265, 905
X1, X2 = 78, 724

EXPECTED_DIGITS = 3
GPU = False


def clean_prediction(text: str) -> str:
    return re.sub(r"\D", "", text)


def crop_tile_content(tile: np.ndarray, margin: int = 12) -> np.ndarray:
    if tile.shape[0] <= margin * 2 or tile.shape[1] <= margin * 2:
        raise ValueError(
            f"Tile is too small for margin={margin}: {tile.shape}")

    return tile[
        margin: tile.shape[0] - margin,
        margin: tile.shape[1] - margin,
    ]


def preprocessing_variants(tile: np.ndarray) -> dict[str, np.ndarray]:
    """Create the exact preprocessing family used by the fusion model."""
    cropped = crop_tile_content(tile)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    upscale_2x = cv2.resize(
        cropped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )
    upscale_3x = cv2.resize(
        cropped, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
    )

    sharpen_kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
    )
    sharpened = cv2.filter2D(cropped, -1, sharpen_kernel)

    _, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive_threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred_saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, saturation_mask = cv2.threshold(
        blurred_saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    saturation_mask = cv2.bitwise_not(saturation_mask)

    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate(
        [lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]], axis=0
    )
    background_color = np.median(border, axis=0)
    color_distance = np.linalg.norm(lab - background_color, axis=2)
    lab_color_distance = cv2.normalize(
        color_distance, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    lab_color_distance = cv2.bitwise_not(lab_color_distance)

    return {
        "raw": cropped,
        "gray_clahe": gray_clahe,
        "upscale_2x": upscale_2x,
        "upscale_3x": upscale_3x,
        "sharpened": sharpened,
        "otsu": otsu,
        "adaptive_threshold": adaptive_threshold,
        "saturation_mask": saturation_mask,
        "lab_color_distance": lab_color_distance,
    }


def recognize_variant(
    reader: easyocr.Reader,
    image: np.ndarray,
) -> tuple[str, float]:
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist="0123456789",
    )

    if not results:
        return "", 0.0

    # Prefer a valid three-digit candidate. If none exists, use the
    # highest-confidence OCR result and let fusion decide how useful it is.
    cleaned = [
        (clean_prediction(str(item[1])), float(item[2]))
        for item in results
    ]
    valid = [item for item in cleaned if len(item[0]) == EXPECTED_DIGITS]
    candidates = valid or cleaned
    return max(candidates, key=lambda item: item[1])


    scores: dict[str, float] = defaultdict(float)
    for result in results:
        if result.prediction:
            scores[result.prediction] += result.confidence

    if not scores:
        return "", 0.0

    prediction = max(scores, key=scores.get)
    return prediction, scores[prediction]


def load_selector() -> FusionSelector:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Required fusion model is missing: "
            f"{MODEL_PATH.resolve()}\n"
            "Restore the frozen model before running the solver."
        )

    selector = FusionSelector.load(MODEL_PATH)

    print(f"Loaded fusion model: {MODEL_PATH.resolve()}")
    print(f"Model variants: {', '.join(selector.variants)}")

    return selector


def validate_variant_names(
    selector: FusionSelector | None,
    generated_variants: set[str],
) -> None:
    if selector is None:
        return

    expected = set(selector.variants)
    missing = expected - generated_variants
    extra = generated_variants - expected

    if missing:
        raise RuntimeError(
            "The fusion model expects preprocessing variants that main.py "
            f"does not generate: {sorted(missing)}. Retrain the model or "
            "restore those variant names."
        )

    if extra:
        print(
            "Note: these generated variants are not used by the current "
            f"fusion model: {sorted(extra)}"
        )


def split_grid(image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    grid = image[Y1:Y2, X1:X2]
    if grid.size == 0:
        raise ValueError(
            f"Grid crop is empty. Image shape={image.shape}, "
            f"crop=y[{Y1}:{Y2}], x[{X1}:{X2}]"
        )

    grid_height, grid_width = grid.shape[:2]
    tiles: list[np.ndarray] = []

    for row in range(3):
        for col in range(3):
            y1 = row * grid_height // 3
            y2 = (row + 1) * grid_height // 3
            x1 = col * grid_width // 3
            x2 = (col + 1) * grid_width // 3
            tiles.append(grid[y1:y2, x1:x2])

    return grid, tiles


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(
            f"Could not load {IMAGE_PATH}. Put the screenshot at "
            "images/captcha.png or change IMAGE_PATH in main.py."
        )

    grid, tiles = split_grid(image)
    cv2.imwrite(str(OUTPUT_DIR / "grid.png"), grid)

    print("Loading EasyOCR model...")
    reader = easyocr.Reader(["en"], gpu=GPU)
    selector: FusionSelector = load_selector()

    rows: list[dict[str, object]] = []
    final_predictions: list[str] = []
    names_checked = False

    for tile_number, tile in enumerate(tiles, start=1):
        tile_dir = OUTPUT_DIR / f"tile_{tile_number}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tile_dir / "original.png"), tile)

        variants = preprocessing_variants(tile)
        if not names_checked:
            validate_variant_names(selector, set(variants))
            names_checked = True

        ocr_results: list[OCRResult] = []
        print(f"\nTile {tile_number}")

        for variant_name, variant_image in variants.items():
            cv2.imwrite(str(tile_dir / f"{variant_name}.png"), variant_image)
            prediction, confidence = recognize_variant(reader, variant_image)
            result = OCRResult(variant_name, prediction, confidence)
            ocr_results.append(result)

            print(
                f"  {variant_name:<20} -> {prediction!r:<7} "
                f"confidence={confidence:.3f}"
            )

        final_prediction, fusion_score = selector.predict(ocr_results)
        method = "learned_fusion"

        final_predictions.append(final_prediction)
        print(
            f"  FINAL ({method}) -> {final_prediction!r} "
            f"score={fusion_score:.3f}"
        )

        rows.append(
            {
                "tile": tile_number,
                "prediction": final_prediction,
                "fusion_score": round(fusion_score, 6),
                "method": method,
                "variant_predictions": " | ".join(
                    f"{r.variant}:{r.prediction}:{r.confidence:.6f}"
                    for r in ocr_results
                ),
            }
        )

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile",
                "prediction",
                "fusion_score",
                "method",
                "variant_predictions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nFinal 3x3 predictions:")
    for row_start in range(0, 9, 3):
        print("  " + "  ".join(final_predictions[row_start: row_start + 3]))

    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Debug images: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
