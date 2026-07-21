#!/usr/bin/env python3
"""
Build a real-glyph bank and authentic background bank from labeled CAPTCHA tiles.

Inputs
------
preprocessing_results.csv with columns:
    image,tile,ground_truth,tile_path

Outputs
-------
glyph_bank/
    0/
    1/
    ...
    9/
background_bank/
manifests/
    glyphs.csv
    backgrounds.csv
    rejected.csv
diagnostics/              # optional debug overlays

Example
-------
python build_glyph_bank.py \
    --csv preprocessing_results.csv \
    --project-root . \
    --output data/real_assets \
    --save-diagnostics

Notes
-----
- Keeps one unique sample per (image, tile), matching cnn/dataset.py.
- Expects exactly three digits per label.
- Segments the foreground using color distance from border-estimated background.
- Splits the foreground into three digit regions using vertical projection.
- Saves RGBA glyph crops with authentic RGB pixels and alpha masks.
- Inpaints the complete digit region to produce reusable 200x200 backgrounds.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Sample:
    screenshot: str
    tile: int
    tile_path: Path
    label: str


@dataclass(frozen=True)
class GlyphRecord:
    glyph_path: str
    source_tile: str
    screenshot: str
    tile: int
    label: str
    digit: str
    position: int
    x: int
    y: int
    width: int
    height: int
    foreground_pixels: int
    mask_quality: float


@dataclass(frozen=True)
class BackgroundRecord:
    background_path: str
    source_tile: str
    screenshot: str
    tile: int
    label: str
    mask_pixels: int


@dataclass(frozen=True)
class RejectRecord:
    screenshot: str
    tile: int
    tile_path: str
    label: str
    reason: str


def load_samples(csv_path: Path, project_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[tuple[str, int]] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "tile", "ground_truth", "tile_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            screenshot = row["image"].strip()
            tile = int(row["tile"])
            key = (screenshot, tile)

            if key in seen:
                continue
            seen.add(key)

            label = row["ground_truth"].strip()
            if len(label) != 3 or not label.isdigit():
                raise ValueError(
                    f"Invalid label {label!r} on CSV row {row_number}"
                )

            raw_path = Path(row["tile_path"].strip())
            tile_path = raw_path if raw_path.is_absolute() else project_root / raw_path

            samples.append(
                Sample(
                    screenshot=screenshot,
                    tile=tile,
                    tile_path=tile_path,
                    label=label,
                )
            )

    samples.sort(key=lambda item: (item.screenshot, item.tile))
    if not samples:
        raise RuntimeError(f"No labeled samples found in {csv_path}")
    return samples


def border_pixels(image: np.ndarray, border: int = 12) -> np.ndarray:
    h, w = image.shape[:2]
    border = max(2, min(border, h // 4, w // 4))
    parts = [
        image[:border, :, :].reshape(-1, 3),
        image[-border:, :, :].reshape(-1, 3),
        image[:, :border, :].reshape(-1, 3),
        image[:, -border:, :].reshape(-1, 3),
    ]
    return np.concatenate(parts, axis=0)


def estimate_background_color(image: np.ndarray) -> np.ndarray:
    pixels = border_pixels(image).astype(np.float32)

    # Median is robust to stray text/underline touching an edge.
    median = np.median(pixels, axis=0)

    # Keep pixels near the median and average them for a less noisy estimate.
    distances = np.linalg.norm(pixels - median[None, :], axis=1)
    keep = pixels[distances <= np.percentile(distances, 65)]
    if keep.size == 0:
        return median
    return keep.mean(axis=0)


def foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Segment colored/dark digits from the textured background.

    Combines:
    - Lab color distance from border-estimated background
    - local chroma/saturation difference
    - luminance difference
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_rgb = estimate_background_color(image_rgb).astype(np.uint8)
    bg_bgr = bg_rgb[::-1].reshape(1, 1, 3)
    bg_lab = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    lab_distance = np.linalg.norm(image_lab - bg_lab[None, None, :], axis=2)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = float(
        0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    )
    luminance_distance = np.abs(gray - bg_gray)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)

    # Adaptive threshold from border texture noise.
    border_lab = border_pixels(image_lab.astype(np.uint8)).astype(np.float32)
    border_dist = np.linalg.norm(border_lab - bg_lab[None, :], axis=1)
    noise_level = float(np.percentile(border_dist, 92))

    lab_threshold = max(15.0, noise_level + 8.0)
    mask = (
        (lab_distance >= lab_threshold)
        | (luminance_distance >= max(22.0, noise_level * 1.25))
        | ((saturation >= 55.0) & (lab_distance >= max(11.0, noise_level + 3.0)))
    ).astype(np.uint8) * 255

    h, w = mask.shape

    # Ignore a narrow outer boundary; digits almost never belong there.
    margin = max(2, round(min(h, w) * 0.02))
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0

    # Remove tiny woven-texture responses while preserving thin glyph strokes.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Keep plausible connected components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    min_area = max(12, round(h * w * 0.0003))

    for component in range(1, count):
        x, y, cw, ch, area = stats[component]
        if area < min_area:
            continue
        if ch < max(5, round(h * 0.04)):
            continue
        if cw > w * 0.95 and ch < h * 0.05:
            continue
        cleaned[labels == component] = 255

    return cleaned


def occupied_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def smooth_projection(values: np.ndarray, window: int = 7) -> np.ndarray:
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def split_three_digits(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Split the global foreground into exactly three x-ranges.

    Uses projection valleys around 1/3 and 2/3 of the occupied width, while
    allowing touching glyphs and underline connections.
    """
    bounds = occupied_bounds(mask)
    if bounds is None:
        raise ValueError("empty foreground mask")

    x0, _, x1, _ = bounds
    if x1 - x0 < 18:
        raise ValueError("foreground region is too narrow")

    cropped = mask[:, x0:x1]
    projection = (cropped > 0).sum(axis=0).astype(np.float32)
    smoothed = smooth_projection(projection, window=7)

    width = len(smoothed)
    expected_1 = width / 3
    expected_2 = 2 * width / 3
    radius = max(5, round(width * 0.16))

    def valley(expected: float, left_limit: int, right_limit: int) -> int:
        lo = max(left_limit, int(round(expected - radius)))
        hi = min(right_limit, int(round(expected + radius)))
        if hi <= lo:
            return int(round(expected))

        candidates = np.arange(lo, hi)
        # Prefer low projection, with a mild penalty for moving far from expected.
        normalized_projection = smoothed[candidates] / max(1.0, smoothed.max())
        position_penalty = np.abs(candidates - expected) / max(1.0, radius)
        score = normalized_projection + 0.18 * position_penalty
        return int(candidates[np.argmin(score)])

    cut1 = valley(expected_1, 4, width - 10)
    cut2 = valley(expected_2, cut1 + 5, width - 4)

    if cut2 - cut1 < 5:
        cut1 = round(width / 3)
        cut2 = round(2 * width / 3)

    ranges = [
        (x0, x0 + cut1),
        (x0 + cut1, x0 + cut2),
        (x0 + cut2, x1),
    ]

    for left, right in ranges:
        if right - left < 4:
            raise ValueError("digit split produced an implausibly narrow region")

    return ranges


def crop_digit_mask(
    full_mask: np.ndarray,
    x_range: tuple[int, int],
    padding: int = 4,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    left, right = x_range
    local = np.zeros_like(full_mask)
    local[:, left:right] = full_mask[:, left:right]

    bounds = occupied_bounds(local)
    if bounds is None:
        raise ValueError("digit region contains no foreground")

    x0, y0, x1, y1 = bounds
    h, w = full_mask.shape
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)

    crop = local[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1)


def soften_alpha(mask: np.ndarray) -> np.ndarray:
    """
    Preserve authentic edge shape but give compositing a small antialiased edge.
    """
    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.55, sigmaY=0.55)
    alpha = np.clip(alpha * 1.15, 0.0, 1.0)
    return np.round(alpha * 255).astype(np.uint8)


def mask_quality(mask: np.ndarray) -> float:
    """
    Heuristic quality score in [0, 1].
    Rewards a useful foreground fraction and coherent vertical extent.
    """
    h, w = mask.shape
    foreground = int((mask > 0).sum())
    fraction = foreground / max(1, h * w)

    bounds = occupied_bounds(mask)
    if bounds is None:
        return 0.0

    _, y0, _, y1 = bounds
    height_fraction = (y1 - y0) / max(1, h)

    density_score = 1.0 - min(1.0, abs(fraction - 0.22) / 0.22)
    height_score = min(1.0, height_fraction / 0.45)
    return float(np.clip(0.55 * density_score + 0.45 * height_score, 0, 1))


def remove_long_horizontal_lines(mask: np.ndarray) -> np.ndarray:
    """
    Produce a background inpainting mask that includes underlines/strike lines.
    The glyph alpha itself is left unchanged.
    """
    h, w = mask.shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(12, round(w * 0.10)), 1),
    )
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    return cv2.bitwise_or(mask, lines)


def make_inpainted_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = cv2.dilate(
        remove_long_horizontal_lines(mask),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    # Telea works well for textured local backgrounds at this scale.
    background = cv2.inpaint(image, expanded, 5, cv2.INPAINT_TELEA)

    # A light texture-preserving blend reduces obvious inpaint boundaries.
    blurred = cv2.GaussianBlur(background, (0, 0), 0.7)
    edge = cv2.GaussianBlur(expanded.astype(np.float32) / 255.0, (0, 0), 2.0)
    edge = edge[..., None]
    result = (
        background.astype(np.float32) * (1.0 - 0.20 * edge)
        + blurred.astype(np.float32) * (0.20 * edge)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def save_rgba_glyph(
    image_bgr: np.ndarray,
    digit_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    x0, y0, x1, y1 = bounds
    rgb = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    alpha = soften_alpha(digit_mask)
    rgba = np.dstack([rgb, alpha])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def diagnostic_overlay(
    image: np.ndarray,
    full_mask: np.ndarray,
    ranges: list[tuple[int, int]],
    label: str,
) -> np.ndarray:
    overlay = image.copy()
    color_mask = np.zeros_like(image)
    color_mask[:, :, 2] = full_mask
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.35, 0)

    colors = [(255, 80, 20), (20, 180, 20), (20, 80, 255)]
    for index, (left, right) in enumerate(ranges):
        cv2.rectangle(
            overlay,
            (left, 2),
            (right, image.shape[0] - 3),
            colors[index],
            2,
        )
        cv2.putText(
            overlay,
            label[index],
            (left + 3, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colors[index],
            2,
            cv2.LINE_AA,
        )
    return overlay


def write_csv(records: Iterable[object], path: Path) -> None:
    records = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build(args: argparse.Namespace) -> None:
    samples = load_samples(args.csv, args.project_root)

    output = args.output.resolve()
    glyph_root = output / "glyph_bank"
    background_root = output / "background_bank"
    diagnostics_root = output / "diagnostics"
    manifest_root = output / "manifests"

    glyph_records: list[GlyphRecord] = []
    background_records: list[BackgroundRecord] = []
    rejected: list[RejectRecord] = []

    accepted_samples = 0

    for index, sample in enumerate(samples, start=1):
        try:
            image = cv2.imread(str(sample.tile_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("could not read tile image")

            h, w = image.shape[:2]
            if min(h, w) < 32:
                raise ValueError(f"tile is too small: {w}x{h}")

            mask = foreground_mask(image)
            bounds = occupied_bounds(mask)
            if bounds is None:
                raise ValueError("no foreground detected")

            foreground_fraction = float((mask > 0).mean())
            if not 0.006 <= foreground_fraction <= 0.55:
                raise ValueError(
                    f"implausible foreground fraction: {foreground_fraction:.4f}"
                )

            ranges = split_three_digits(mask)

            sample_stem = (
                f"{Path(sample.screenshot).stem}_tile-{sample.tile}"
            )

            staged_glyphs: list[tuple[Path, GlyphRecord]] = []

            for position, (digit, x_range) in enumerate(
                zip(sample.label, ranges),
                start=1,
            ):
                digit_mask, digit_bounds = crop_digit_mask(
                    mask,
                    x_range,
                    padding=args.padding,
                )
                x0, y0, x1, y1 = digit_bounds
                width = x1 - x0
                height = y1 - y0
                foreground_pixels = int((digit_mask > 0).sum())
                quality = mask_quality(digit_mask)

                if width < 4 or height < 10 or foreground_pixels < 10:
                    raise ValueError(
                        f"bad glyph at position {position}: "
                        f"{width}x{height}, {foreground_pixels} pixels"
                    )

                glyph_filename = (
                    f"{sample_stem}_pos-{position}_{digit}.png"
                )
                glyph_path = glyph_root / digit / glyph_filename

                record = GlyphRecord(
                    glyph_path=glyph_path.relative_to(output).as_posix(),
                    source_tile=sample.tile_path.as_posix(),
                    screenshot=sample.screenshot,
                    tile=sample.tile,
                    label=sample.label,
                    digit=digit,
                    position=position,
                    x=x0,
                    y=y0,
                    width=width,
                    height=height,
                    foreground_pixels=foreground_pixels,
                    mask_quality=round(quality, 4),
                )
                staged_glyphs.append((glyph_path, record))

            # Save only after all three glyphs pass validation.
            for glyph_path, record in staged_glyphs:
                position = record.position - 1
                digit_mask, digit_bounds = crop_digit_mask(
                    mask,
                    ranges[position],
                    padding=args.padding,
                )
                save_rgba_glyph(
                    image,
                    digit_mask,
                    digit_bounds,
                    glyph_path,
                )
                glyph_records.append(record)

            background = make_inpainted_background(image, mask)
            background_path = background_root / f"{sample_stem}.png"
            background_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(background_path), background)

            background_records.append(
                BackgroundRecord(
                    background_path=background_path.relative_to(output).as_posix(),
                    source_tile=sample.tile_path.as_posix(),
                    screenshot=sample.screenshot,
                    tile=sample.tile,
                    label=sample.label,
                    mask_pixels=int((mask > 0).sum()),
                )
            )

            if args.save_diagnostics:
                diagnostic = diagnostic_overlay(
                    image,
                    mask,
                    ranges,
                    sample.label,
                )
                diagnostics_root.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(diagnostics_root / f"{sample_stem}.png"),
                    diagnostic,
                )

            accepted_samples += 1

        except Exception as exc:
            rejected.append(
                RejectRecord(
                    screenshot=sample.screenshot,
                    tile=sample.tile,
                    tile_path=sample.tile_path.as_posix(),
                    label=sample.label,
                    reason=str(exc),
                )
            )

        if index % args.log_every == 0 or index == len(samples):
            print(
                f"Processed {index:,}/{len(samples):,} | "
                f"accepted {accepted_samples:,} | "
                f"rejected {len(rejected):,}"
            )

    write_csv(glyph_records, manifest_root / "glyphs.csv")
    write_csv(background_records, manifest_root / "backgrounds.csv")
    write_csv(rejected, manifest_root / "rejected.csv")

    digit_counts = {
        str(digit): sum(record.digit == str(digit) for record in glyph_records)
        for digit in range(10)
    }

    summary = {
        "input_samples": len(samples),
        "accepted_samples": accepted_samples,
        "rejected_samples": len(rejected),
        "glyph_count": len(glyph_records),
        "background_count": len(background_records),
        "digit_counts": digit_counts,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Done.")
    print(json.dumps(summary, indent=2))
    print(f"Output: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build authentic digit glyph and background banks."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("preprocessing_results.csv"),
        help="Labeled CSV path.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Root used to resolve relative tile_path values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_assets"),
        help="Output directory.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=4,
        help="Transparent padding around each glyph crop.",
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Save segmentation overlays for visual review.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
    )
    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"CSV does not exist: {args.csv}")
    if args.padding < 0:
        parser.error("--padding cannot be negative")
    return args


if __name__ == "__main__":
    build(parse_args())
