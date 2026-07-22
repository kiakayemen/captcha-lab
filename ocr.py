from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from preprocess import ensure_bgr

DIGITS_ONLY = re.compile(r"\D")
EXPECTED_DIGITS = 3


@dataclass(frozen=True)
class OCRResult:
    variant: str
    prediction: str
    confidence: float

    @property
    def valid(self) -> bool:
        return len(self.prediction) == EXPECTED_DIGITS and self.prediction.isdigit()


def clean_prediction(text: str) -> str:
    return DIGITS_ONLY.sub("", text)


def add_white_padding(image: np.ndarray, padding: int = 15) -> np.ndarray:
    if padding < 0:
        raise ValueError("padding must be non-negative")
    return cv2.copyMakeBorder(
        ensure_bgr(image),
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def build_reader(gpu: bool = False) -> Any:
    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError(
            "EasyOCR is not installed. Run: pip install easyocr"
        ) from error
    return easyocr.Reader(["en"], gpu=gpu)


def recognize(reader: Any, image: np.ndarray, variant: str) -> OCRResult:
    prepared = add_white_padding(image, padding=15)
    results = reader.readtext(
        prepared,
        detail=1,
        paragraph=False,
        allowlist="0123456789",
    )

    if not results:
        return OCRResult(variant=variant, prediction="", confidence=0.0)

    parsed: list[tuple[str, float]] = []
    for result in results:
        text = clean_prediction(str(result[1]))
        confidence = float(result[2])
        parsed.append((text, confidence))

    valid = [item for item in parsed if len(item[0]) == EXPECTED_DIGITS]
    prediction, confidence = max(valid or parsed, key=lambda item: item[1])
    return OCRResult(variant=variant, prediction=prediction, confidence=confidence)
