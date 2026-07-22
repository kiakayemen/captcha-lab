#!/usr/bin/env python3
"""
Build clean digit prototype families from the authentic glyph bank.

Outputs thin/medium/bold prototype variants for digits 0-9 by:
- loading real RGBA glyphs
- removing obvious long underlines
- normalizing and aligning masks
- rejecting damaged outliers
- grouping by stroke width
- averaging each group into a clean alpha prototype

Run:
python build_digit_prototypes.py \
  --assets data/real_assets \
  --output data/digit_prototypes
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SIZE = 96
INNER_HEIGHT = 78


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_alpha(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Invalid glyph: {path}")
    return image[:, :, 3]


def remove_long_horizontal_lines(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 16).astype(np.uint8) * 255
    h, w = binary.shape

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(8, int(round(w * 0.42))), 1),
    )
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    lower = np.zeros_like(lines)
    lower[int(h * 0.52):, :] = 255
    lines = cv2.bitwise_and(lines, lower)

    cleaned = cv2.subtract(binary, lines)
    return cleaned


def tight_crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    return mask[
        int(ys.min()):int(ys.max() + 1),
        int(xs.min()):int(xs.max() + 1),
    ]


def normalize(mask: np.ndarray) -> np.ndarray:
    mask = remove_long_horizontal_lines(mask)
    mask = tight_crop(mask)

    h, w = mask.shape
    if h < 8 or w < 2:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)

    scale = INNER_HEIGHT / h
    target_w = max(2, min(SIZE - 10, int(round(w * scale))))
    resized = cv2.resize(
        mask,
        (target_w, INNER_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((SIZE, SIZE), dtype=np.uint8)
    x = (SIZE - target_w) // 2
    y = (SIZE - INNER_HEIGHT) // 2
    canvas[y:y + INNER_HEIGHT, x:x + target_w] = resized
    return canvas


def stroke_width(mask: np.ndarray) -> float:
    binary = (mask > 32).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    values = distance[binary > 0] * 2.0
    return float(values.mean()) if values.size else 0.0


def valid(mask: np.ndarray) -> bool:
    binary = (mask > 32).astype(np.uint8)
    fraction = float(binary.mean())
    if not 0.015 <= fraction <= 0.45:
        return False

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    substantial = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area >= 12 and height >= 8:
            substantial += 1

    return 1 <= substantial <= 4


def align_to_reference(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    best = mask
    best_score = -1.0

    ref = reference.astype(np.float32) / 255.0
    src = mask.astype(np.float32) / 255.0

    for dy in range(-3, 4):
        for dx in range(-3, 4):
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                src,
                matrix,
                (SIZE, SIZE),
                flags=cv2.INTER_LINEAR,
                borderValue=0,
            )
            score = float((shifted * ref).sum())
            if score > best_score:
                best_score = score
                best = np.clip(shifted * 255, 0, 255).astype(np.uint8)

    return best


def robust_average(masks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(masks).astype(np.float32) / 255.0
    median = np.median(stack, axis=0)

    aligned = [
        align_to_reference(mask, (median * 255).astype(np.uint8))
        for mask in masks
    ]
    stack = np.stack(aligned).astype(np.float32) / 255.0

    # Trim extreme per-pixel outliers.
    low = np.percentile(stack, 15, axis=0)
    high = np.percentile(stack, 85, axis=0)
    clipped = np.clip(stack, low[None, ...], high[None, ...])
    mean = clipped.mean(axis=0)

    # Keep a soft antialiased alpha edge.
    mean = cv2.GaussianBlur(mean, (0, 0), 0.45)
    return np.clip(mean * 255, 0, 255).astype(np.uint8)


def write_rgba(alpha: np.ndarray, path: Path) -> None:
    rgb = np.full((*alpha.shape, 3), 255, dtype=np.uint8)
    rgba = np.dstack([rgb, alpha])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def build(args: argparse.Namespace) -> None:
    manifest = args.assets / "manifests" / "glyphs.csv"
    rows = read_csv(manifest)

    by_digit: dict[str, list[np.ndarray]] = defaultdict(list)
    rejected = 0

    for row in rows:
        quality = float(row.get("mask_quality", "1") or 1)
        if quality < args.min_quality:
            rejected += 1
            continue

        path = resolve_path(args.assets, row["glyph_path"])
        try:
            mask = normalize(load_alpha(path))
        except Exception:
            rejected += 1
            continue

        if not valid(mask):
            rejected += 1
            continue

        by_digit[row["digit"]].append(mask)

    summary: dict[str, object] = {
        "input_rows": len(rows),
        "rejected": rejected,
        "digits": {},
    }

    for digit in "0123456789":
        masks = by_digit[digit]
        if len(masks) < 6:
            raise RuntimeError(f"Too few clean glyphs for digit {digit}")

        widths = np.array([stroke_width(mask) for mask in masks])
        q1, q2 = np.quantile(widths, [0.33, 0.66])

        groups = {
            "thin": [m for m, s in zip(masks, widths) if s <= q1],
            "medium": [m for m, s in zip(masks, widths) if q1 < s <= q2],
            "bold": [m for m, s in zip(masks, widths) if s > q2],
        }

        digit_summary = {}
        for name, group in groups.items():
            prototype = robust_average(group)
            path = args.output / digit / f"{name}.png"
            write_rgba(prototype, path)
            digit_summary[name] = {
                "count": len(group),
                "stroke_mean": round(
                    float(np.mean([stroke_width(mask) for mask in group])),
                    4,
                ),
                "path": path.relative_to(args.output).as_posix(),
            }

        summary["digits"][digit] = digit_summary
        print(f"Digit {digit}: {len(masks)} clean glyphs")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Output: {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("data/real_assets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/digit_prototypes"),
    )
    parser.add_argument("--min-quality", type=float, default=0.35)
    args = parser.parse_args()

    if not args.assets.exists():
        parser.error(f"Missing assets: {args.assets}")
    return args


if __name__ == "__main__":
    build(parse_args())
