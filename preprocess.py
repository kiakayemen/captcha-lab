from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

ImageVariant = Callable[[np.ndarray], np.ndarray]


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected grayscale or BGR image, received shape={image.shape}")
    return image


def variant_raw(image: np.ndarray) -> np.ndarray:
    return image.copy()


def variant_upscale_2x(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def variant_upscale_3x(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def variant_gray_clahe(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return ensure_bgr(enhanced)


def variant_sharpened(image: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(enlarged, 1.8, blurred, -0.8, 0)


def variant_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
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
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresholded = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return ensure_bgr(thresholded)


def variant_saturation_mask(image: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(enlarged, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (3, 3), 0)
    _, mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return ensure_bgr(cv2.bitwise_not(mask))


def variant_lab_color_distance(image: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    _, a_channel, b_channel = cv2.split(lab)
    neutral = np.full_like(a_channel, 128)
    a_distance = cv2.absdiff(a_channel, neutral)
    b_distance = cv2.absdiff(b_channel, neutral)
    color_distance = cv2.addWeighted(a_distance, 0.5, b_distance, 0.5, 0)
    color_distance = cv2.GaussianBlur(color_distance, (3, 3), 0)
    _, mask = cv2.threshold(
        color_distance,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return ensure_bgr(cv2.bitwise_not(mask))


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


def preprocessing_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess an empty image")
    image = ensure_bgr(image)
    return {name: transform(image) for name, transform in VARIANTS.items()}
