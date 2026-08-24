from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

THREE_DIGITS = re.compile(r"(?<!\d)(\d{3})(?!\d)")


@dataclass(frozen=True)
class TargetResult:
    target: str
    confidence: float
    variant: str
    prompt_crop: np.ndarray


def crop_instruction_region(
    image: np.ndarray,
    grid_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop only the instruction strip immediately above the detected grid."""
    grid_x, grid_y, grid_width, grid_height = grid_box
    image_height, image_width = image.shape[:2]

    side_padding = int(round(grid_width * 0.05))
    instruction_height = max(50, int(round(grid_height * 0.28)))

    left = max(0, grid_x - side_padding)
    right = min(image_width, grid_x + grid_width + side_padding)
    top = max(0, grid_y - instruction_height)
    bottom = min(image_height, grid_y)

    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Instruction crop is empty")
    return crop


def instruction_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    enlarged = cv2.resize(
        image,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return {
        "raw": image,
        "upscale": enlarged,
        "gray_clahe": clahe,
        "otsu": otsu,
    }


def _box_left(result: list[Any] | tuple[Any, ...]) -> float:
    box = result[0]
    try:
        return float(min(point[0] for point in box))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _candidate_from_results(results: list[Any]) -> tuple[str, float] | None:
    candidates: list[tuple[str, float]] = []

    for result in results:
        text = str(result[1])
        confidence = float(result[2])
        for match in THREE_DIGITS.finditer(text):
            candidates.append((match.group(1), confidence))

    # Legacy helper retained for reviewing old OCR result files.
    ordered = sorted(results, key=_box_left)
    joined_digits = "".join(re.sub(r"\D", "", str(item[1])) for item in ordered)
    for match in THREE_DIGITS.finditer(joined_digits):
        confidences = [float(item[2]) for item in ordered if re.search(r"\d", str(item[1]))]
        confidence = min(confidences) if confidences else 0.0
        candidates.append((match.group(1), confidence))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def extract_target(
    reader: Any,
    image: np.ndarray,
    grid_box: tuple[int, int, int, int],
) -> TargetResult:
    raise RuntimeError(
        "Target extraction from screenshot text is not supported by PARSeq. "
        "Read the target from the CAPTCHA DOM instead."
    )
