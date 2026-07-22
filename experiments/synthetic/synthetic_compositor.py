#!/usr/bin/env python3
"""
Compose synthetic CAPTCHA tiles from authentic extracted glyphs and backgrounds.

Expected asset layout
---------------------
data/real_assets/
├── glyph_bank/
├── background_bank/
└── manifests/
    ├── glyphs.csv
    └── backgrounds.csv

Example
-------
python synthetic_compositor.py \
    --assets data/real_assets \
    --output data/composed_preview \
    --count 500 \
    --seed 42 \
    --preview-count 100

Large generation:
python synthetic_compositor.py \
    --assets data/real_assets \
    --output data/synthetic_real_glyphs \
    --count 20000 \
    --seed 42 \
    --preview-count 100

Outputs
-------
<output>/
├── images/
├── labels.csv
├── generation_summary.json
└── preview.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Glyph:
    path: Path
    digit: str
    position: int
    x: int
    y: int
    width: int
    height: int
    source_tile: str
    quality: float


@dataclass(frozen=True)
class Background:
    path: Path
    source_tile: str


@dataclass(frozen=True)
class PlacementStats:
    center_x_mean: float
    center_x_std: float
    y_mean: float
    y_std: float
    width_mean: float
    height_mean: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_asset_path(assets_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return assets_root / path


def load_assets(
    assets_root: Path,
    min_quality: float,
) -> tuple[list[Glyph], list[Background]]:
    glyph_manifest = assets_root / "manifests" / "glyphs.csv"
    background_manifest = assets_root / "manifests" / "backgrounds.csv"

    if not glyph_manifest.exists():
        raise FileNotFoundError(f"Missing glyph manifest: {glyph_manifest}")
    if not background_manifest.exists():
        raise FileNotFoundError(
            f"Missing background manifest: {background_manifest}"
        )

    glyphs: list[Glyph] = []
    for row in read_csv(glyph_manifest):
        quality = float(row.get("mask_quality", "1") or 1)
        if quality < min_quality:
            continue

        path = resolve_asset_path(assets_root, row["glyph_path"])
        if not path.exists():
            raise FileNotFoundError(f"Glyph image does not exist: {path}")

        glyphs.append(
            Glyph(
                path=path,
                digit=row["digit"],
                position=int(row["position"]),
                x=int(row["x"]),
                y=int(row["y"]),
                width=int(row["width"]),
                height=int(row["height"]),
                source_tile=row["source_tile"],
                quality=quality,
            )
        )

    backgrounds: list[Background] = []
    for row in read_csv(background_manifest):
        path = resolve_asset_path(assets_root, row["background_path"])
        if not path.exists():
            raise FileNotFoundError(f"Background image does not exist: {path}")

        backgrounds.append(
            Background(
                path=path,
                source_tile=row["source_tile"],
            )
        )

    if not glyphs:
        raise RuntimeError("No glyphs passed the quality threshold")
    if not backgrounds:
        raise RuntimeError("No backgrounds were loaded")

    return glyphs, backgrounds


def build_glyph_pools(
    glyphs: list[Glyph],
) -> tuple[
    dict[tuple[str, int], list[Glyph]],
    dict[str, list[Glyph]],
]:
    by_digit_position: dict[tuple[str, int], list[Glyph]] = defaultdict(list)
    by_digit: dict[str, list[Glyph]] = defaultdict(list)

    for glyph in glyphs:
        by_digit_position[(glyph.digit, glyph.position)].append(glyph)
        by_digit[glyph.digit].append(glyph)

    missing = [digit for digit in "0123456789" if not by_digit[digit]]
    if missing:
        raise RuntimeError(f"No glyphs found for digits: {missing}")

    return dict(by_digit_position), dict(by_digit)


def compute_position_stats(glyphs: list[Glyph]) -> dict[int, PlacementStats]:
    stats: dict[int, PlacementStats] = {}

    for position in (1, 2, 3):
        values = [glyph for glyph in glyphs if glyph.position == position]
        if not values:
            raise RuntimeError(f"No glyphs found for position {position}")

        centers = np.array(
            [glyph.x + glyph.width / 2 for glyph in values],
            dtype=np.float32,
        )
        ys = np.array([glyph.y for glyph in values], dtype=np.float32)
        widths = np.array([glyph.width for glyph in values], dtype=np.float32)
        heights = np.array([glyph.height for glyph in values], dtype=np.float32)

        stats[position] = PlacementStats(
            center_x_mean=float(np.mean(centers)),
            center_x_std=max(1.5, float(np.std(centers))),
            y_mean=float(np.mean(ys)),
            y_std=max(1.5, float(np.std(ys))),
            width_mean=float(np.mean(widths)),
            height_mean=float(np.mean(heights)),
        )

    return stats


def load_rgba(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read glyph: {path}")

    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Glyph is not RGBA/BGRA: {path}")

    return image


def resize_rgba(
    glyph: np.ndarray,
    scale: float,
) -> np.ndarray:
    h, w = glyph.shape[:2]
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        glyph,
        (new_w, new_h),
        interpolation=interpolation,
    )

    # Keep alpha crisp enough after cubic scaling.
    resized[:, :, 3] = np.clip(resized[:, :, 3], 0, 255)
    return resized


def alpha_composite(
    background_bgr: np.ndarray,
    foreground_bgra: np.ndarray,
    x: int,
    y: int,
) -> bool:
    bg_h, bg_w = background_bgr.shape[:2]
    fg_h, fg_w = foreground_bgra.shape[:2]

    left = max(0, x)
    top = max(0, y)
    right = min(bg_w, x + fg_w)
    bottom = min(bg_h, y + fg_h)

    if right <= left or bottom <= top:
        return False

    fg_x0 = left - x
    fg_y0 = top - y
    fg_x1 = fg_x0 + (right - left)
    fg_y1 = fg_y0 + (bottom - top)

    foreground = foreground_bgra[fg_y0:fg_y1, fg_x0:fg_x1]
    alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    rgb = foreground[:, :, :3].astype(np.float32)
    background = background_bgr[top:bottom, left:right].astype(np.float32)

    blended = rgb * alpha + background * (1.0 - alpha)
    background_bgr[top:bottom, left:right] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)

    return True


def sample_label(rng: random.Random) -> str:
    # Uniform per-position digit sampling prevents the original dataset's
    # digit imbalance from being copied into synthetic training data.
    return "".join(rng.choice("0123456789") for _ in range(3))


def choose_glyph(
    digit: str,
    position: int,
    by_digit_position: dict[tuple[str, int], list[Glyph]],
    by_digit: dict[str, list[Glyph]],
    rng: random.Random,
) -> Glyph:
    preferred = by_digit_position.get((digit, position), [])
    pool = preferred if preferred else by_digit[digit]
    return rng.choice(pool)


def sample_positions(
    glyph_images: list[np.ndarray],
    chosen_glyphs: list[Glyph],
    stats: dict[int, PlacementStats],
    canvas_width: int,
    canvas_height: int,
    rng: random.Random,
    x_jitter: float,
    y_jitter: float,
) -> list[tuple[int, int]]:
    """
    Use empirical position distributions while preventing severe overlap.

    Small negative gaps are allowed because real glyphs and underlines can
    visually touch.
    """
    positions: list[list[int]] = []

    for index, (image, source) in enumerate(
        zip(glyph_images, chosen_glyphs),
        start=1,
    ):
        h, w = image.shape[:2]
        position_stats = stats[index]

        # Blend the sampled glyph's original location with dataset-wide stats.
        source_center = source.x + source.width / 2
        center_mean = 0.65 * source_center + 0.35 * position_stats.center_x_mean
        center = rng.gauss(
            center_mean,
            min(position_stats.center_x_std, x_jitter),
        )

        source_y = source.y
        y_mean = 0.70 * source_y + 0.30 * position_stats.y_mean
        y = int(round(rng.gauss(
            y_mean,
            min(position_stats.y_std, y_jitter),
        )))
        x = int(round(center - w / 2))

        x = max(1, min(canvas_width - w - 1, x))
        y = max(1, min(canvas_height - h - 1, y))
        positions.append([x, y])

    # Enforce order with a small allowed overlap.
    min_gap = -2
    for index in range(1, 3):
        previous_x = positions[index - 1][0]
        previous_w = glyph_images[index - 1].shape[1]
        minimum_x = previous_x + previous_w + min_gap

        if positions[index][0] < minimum_x:
            positions[index][0] = minimum_x

    # Shift the whole group left if the third glyph exceeds the canvas.
    last_right = positions[2][0] + glyph_images[2].shape[1]
    if last_right >= canvas_width:
        shift = last_right - canvas_width + 2
        for item in positions:
            item[0] -= shift

    # Shift right if the first glyph now leaves the canvas.
    if positions[0][0] < 1:
        shift = 1 - positions[0][0]
        for item in positions:
            item[0] += shift

    # Final clamp.
    final: list[tuple[int, int]] = []
    for (x, y), image in zip(positions, glyph_images):
        h, w = image.shape[:2]
        x = max(0, min(canvas_width - w, x))
        y = max(0, min(canvas_height - h, y))
        final.append((x, y))

    return final


def generate_one(
    label: str,
    background: Background,
    by_digit_position: dict[tuple[str, int], list[Glyph]],
    by_digit: dict[str, list[Glyph]],
    stats: dict[int, PlacementStats],
    rng: random.Random,
    scale_min: float,
    scale_max: float,
    x_jitter: float,
    y_jitter: float,
    avoid_same_source: bool,
) -> tuple[np.ndarray, list[Glyph], list[tuple[int, int]], list[float]]:
    canvas = cv2.imread(str(background.path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise ValueError(f"Could not read background: {background.path}")

    chosen: list[Glyph] = []
    glyph_images: list[np.ndarray] = []
    scales: list[float] = []

    for position, digit in enumerate(label, start=1):
        candidates = by_digit_position.get((digit, position), [])
        pool = candidates if candidates else by_digit[digit]

        if avoid_same_source:
            filtered = [
                glyph
                for glyph in pool
                if glyph.source_tile != background.source_tile
            ]
            if filtered:
                pool = filtered

        glyph = rng.choice(pool)
        rgba = load_rgba(glyph.path)

        scale = rng.uniform(scale_min, scale_max)
        rgba = resize_rgba(rgba, scale)

        chosen.append(glyph)
        glyph_images.append(rgba)
        scales.append(scale)

    positions = sample_positions(
        glyph_images=glyph_images,
        chosen_glyphs=chosen,
        stats=stats,
        canvas_width=canvas.shape[1],
        canvas_height=canvas.shape[0],
        rng=rng,
        x_jitter=x_jitter,
        y_jitter=y_jitter,
    )

    for glyph_image, (x, y) in zip(glyph_images, positions):
        if not alpha_composite(canvas, glyph_image, x, y):
            raise RuntimeError("Glyph did not overlap the output canvas")

    return canvas, chosen, positions, scales


def make_preview(
    image_paths: list[Path],
    labels: list[str],
    output_path: Path,
    max_images: int,
) -> None:
    count = min(max_images, len(image_paths))
    if count <= 0:
        return

    selected_paths = image_paths[:count]
    selected_labels = labels[:count]

    thumbnails: list[np.ndarray] = []
    target_w, target_h = 200, 200
    caption_h = 28

    for path, label in zip(selected_paths, selected_labels):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        image = cv2.resize(image, (target_w, target_h))
        tile = np.full(
            (target_h + caption_h, target_w, 3),
            255,
            dtype=np.uint8,
        )
        tile[:target_h] = image
        cv2.putText(
            tile,
            label,
            (8, target_h + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(tile)

    if not thumbnails:
        return

    columns = min(10, max(1, math.ceil(math.sqrt(len(thumbnails)))))
    rows = math.ceil(len(thumbnails) / columns)
    cell_h, cell_w = thumbnails[0].shape[:2]

    sheet = np.full(
        (rows * cell_h, columns * cell_w, 3),
        240,
        dtype=np.uint8,
    )

    for index, tile in enumerate(thumbnails):
        row = index // columns
        column = index % columns
        y = row * cell_h
        x = column * cell_w
        sheet[y:y + cell_h, x:x + cell_w] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    assets_root = args.assets.resolve()
    output_root = args.output.resolve()
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    glyphs, backgrounds = load_assets(
        assets_root=assets_root,
        min_quality=args.min_quality,
    )
    by_digit_position, by_digit = build_glyph_pools(glyphs)
    stats = compute_position_stats(glyphs)

    rows: list[dict[str, object]] = []
    output_paths: list[Path] = []
    labels: list[str] = []
    digit_counts: Counter[str] = Counter()
    glyph_use_counts: Counter[str] = Counter()
    background_use_counts: Counter[str] = Counter()

    for index in range(args.count):
        label = sample_label(rng)
        background = rng.choice(backgrounds)

        image, chosen, positions, scales = generate_one(
            label=label,
            background=background,
            by_digit_position=by_digit_position,
            by_digit=by_digit,
            stats=stats,
            rng=rng,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            x_jitter=args.x_jitter,
            y_jitter=args.y_jitter,
            avoid_same_source=not args.allow_same_source,
        )

        filename = f"{index:07d}_{label}.png"
        output_path = images_root / filename
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not write image: {output_path}")

        row: dict[str, object] = {
            "image_path": output_path.relative_to(output_root).as_posix(),
            "label": label,
            "background_path": background.path.as_posix(),
        }

        for position, (glyph, (x, y), scale) in enumerate(
            zip(chosen, positions, scales),
            start=1,
        ):
            row[f"digit_{position}"] = label[position - 1]
            row[f"glyph_{position}"] = glyph.path.as_posix()
            row[f"x_{position}"] = x
            row[f"y_{position}"] = y
            row[f"scale_{position}"] = round(scale, 4)

            glyph_use_counts[glyph.path.as_posix()] += 1

        background_use_counts[background.path.as_posix()] += 1
        digit_counts.update(label)
        rows.append(row)
        output_paths.append(output_path)
        labels.append(label)

        completed = index + 1
        if completed % args.log_every == 0 or completed == args.count:
            print(f"Generated {completed:,}/{args.count:,}")

    labels_path = output_root / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_preview(
        image_paths=output_paths,
        labels=labels,
        output_path=output_root / "preview.png",
        max_images=args.preview_count,
    )

    summary = {
        "count": args.count,
        "seed": args.seed,
        "assets_root": assets_root.as_posix(),
        "glyph_pool_size": len(glyphs),
        "background_pool_size": len(backgrounds),
        "min_quality": args.min_quality,
        "scale_range": [args.scale_min, args.scale_max],
        "digit_counts": {
            digit: digit_counts[digit]
            for digit in "0123456789"
        },
        "unique_glyphs_used": len(glyph_use_counts),
        "unique_backgrounds_used": len(background_use_counts),
        "max_glyph_reuse": max(glyph_use_counts.values(), default=0),
        "max_background_reuse": max(
            background_use_counts.values(),
            default=0,
        ),
    }

    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Done.")
    print(json.dumps(summary, indent=2))
    print(f"Images:  {images_root}")
    print(f"Labels:  {labels_path}")
    print(f"Preview: {output_root / 'preview.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose synthetic CAPTCHA tiles from real glyph assets."
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("data/real_assets"),
        help="Directory created by build_glyph_bank.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/composed_preview"),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of synthetic images to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preview-count",
        type=int,
        default=100,
        help="Number of generated images shown in preview.png.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.25,
        help="Ignore extracted glyphs below this mask quality.",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.96,
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.04,
    )
    parser.add_argument(
        "--x-jitter",
        type=float,
        default=4.0,
        help="Maximum approximate horizontal placement stddev.",
    )
    parser.add_argument(
        "--y-jitter",
        type=float,
        default=3.0,
        help="Maximum approximate vertical placement stddev.",
    )
    parser.add_argument(
        "--allow-same-source",
        action="store_true",
        help=(
            "Allow glyphs to be pasted onto the background extracted from "
            "the same original tile."
        ),
    )
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if not args.assets.exists():
        parser.error(f"Assets directory does not exist: {args.assets}")
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.preview_count < 0:
        parser.error("--preview-count cannot be negative")
    if not 0 <= args.min_quality <= 1:
        parser.error("--min-quality must be between 0 and 1")
    if args.scale_min <= 0 or args.scale_max <= 0:
        parser.error("Scale values must be positive")
    if args.scale_min > args.scale_max:
        parser.error("--scale-min cannot exceed --scale-max")

    return args


if __name__ == "__main__":
    build(parse_args())
