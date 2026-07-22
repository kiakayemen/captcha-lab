#!/usr/bin/env python3
"""
Synthetic CAPTCHA compositor v2.

Uses one complete real CAPTCHA as a layout/style template:
- authentic inpainted background
- authentic three slot positions, spacing, baselines, and sizes
- authentic three-color palette
- digit shapes sampled from other real glyphs

Run:
python synthetic_compositor_v2.py \
  --assets data/real_assets \
  --output data/composed_preview_v2 \
  --count 500 \
  --seed 42 \
  --preview-count 100
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
class Template:
    source_tile: str
    background: Background
    slots: tuple[Glyph, Glyph, Glyph]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_asset_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_assets(
    assets_root: Path,
    min_quality: float,
) -> tuple[list[Glyph], list[Background]]:
    glyph_manifest = assets_root / "manifests" / "glyphs.csv"
    background_manifest = assets_root / "manifests" / "backgrounds.csv"

    if not glyph_manifest.exists():
        raise FileNotFoundError(glyph_manifest)
    if not background_manifest.exists():
        raise FileNotFoundError(background_manifest)

    glyphs: list[Glyph] = []
    for row in read_csv(glyph_manifest):
        quality = float(row.get("mask_quality", "1") or 1)
        if quality < min_quality:
            continue

        path = resolve_asset_path(assets_root, row["glyph_path"])
        if not path.exists():
            raise FileNotFoundError(path)

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
            raise FileNotFoundError(path)
        backgrounds.append(
            Background(path=path, source_tile=row["source_tile"])
        )

    if not glyphs or not backgrounds:
        raise RuntimeError("No usable glyphs or backgrounds found")

    return glyphs, backgrounds


def build_pools(
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

    missing = [d for d in "0123456789" if not by_digit[d]]
    if missing:
        raise RuntimeError(f"Missing glyphs for digits: {missing}")

    return dict(by_digit_position), dict(by_digit)


def build_templates(
    glyphs: list[Glyph],
    backgrounds: list[Background],
) -> list[Template]:
    grouped: dict[str, dict[int, Glyph]] = defaultdict(dict)
    for glyph in glyphs:
        grouped[glyph.source_tile][glyph.position] = glyph

    background_by_source = {
        background.source_tile: background for background in backgrounds
    }

    templates: list[Template] = []
    for source_tile, positions in grouped.items():
        if set(positions) != {1, 2, 3}:
            continue
        background = background_by_source.get(source_tile)
        if background is None:
            continue
        templates.append(
            Template(
                source_tile=source_tile,
                background=background,
                slots=(positions[1], positions[2], positions[3]),
            )
        )

    if not templates:
        raise RuntimeError("No complete templates found")

    return templates


def load_bgra(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected BGRA glyph: {path}")
    return image


def alpha_bounds(image_bgra: np.ndarray) -> tuple[int, int, int, int]:
    alpha = image_bgra[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return 0, 0, image_bgra.shape[1], image_bgra.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def trim_transparent(image_bgra: np.ndarray, padding: int = 2) -> np.ndarray:
    x0, y0, x1, y1 = alpha_bounds(image_bgra)
    h, w = image_bgra.shape[:2]
    return image_bgra[
        max(0, y0 - padding):min(h, y1 + padding),
        max(0, x0 - padding):min(w, x1 + padding),
    ]


def weighted_mean_lab(image_bgra: np.ndarray) -> np.ndarray:
    bgr = image_bgra[:, :, :3]
    alpha = image_bgra[:, :, 3].astype(np.float32) / 255.0
    valid = alpha > 0.08
    if not np.any(valid):
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.average(
        lab[valid],
        axis=0,
        weights=alpha[valid],
    ).astype(np.float32)


def recolor_to_template(
    source_bgra: np.ndarray,
    target_bgra: np.ndarray,
    strength: float,
) -> np.ndarray:
    source_lab = cv2.cvtColor(
        source_bgra[:, :, :3],
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    source_mean = weighted_mean_lab(source_bgra)
    target_mean = weighted_mean_lab(target_bgra)
    shift = (target_mean - source_mean) * strength

    source_lab[:, :, 0] += shift[0] * 0.65
    source_lab[:, :, 1] += shift[1]
    source_lab[:, :, 2] += shift[2]
    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)

    recolored = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
    return np.dstack([recolored, source_bgra[:, :, 3]])


def fit_to_slot(
    glyph_bgra: np.ndarray,
    slot: Glyph,
    scale_jitter: float,
    rng: random.Random,
) -> np.ndarray:
    glyph_bgra = trim_transparent(glyph_bgra)
    h, w = glyph_bgra.shape[:2]

    target_height = max(
        8,
        int(round(
            slot.height
            * rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        )),
    )
    scale = target_height / max(1, h)
    target_width = max(2, int(round(w * scale)))

    max_width = max(slot.width * 2, 12)
    if target_width > max_width:
        scale *= max_width / target_width
        target_width = max_width
        target_height = max(2, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(
        glyph_bgra,
        (target_width, target_height),
        interpolation=interpolation,
    )


def choose_glyph(
    digit: str,
    position: int,
    template_source: str,
    by_digit_position: dict[tuple[str, int], list[Glyph]],
    by_digit: dict[str, list[Glyph]],
    rng: random.Random,
) -> Glyph:
    pool = by_digit_position.get((digit, position), []) or by_digit[digit]
    filtered = [
        glyph for glyph in pool
        if glyph.source_tile != template_source
    ]
    return rng.choice(filtered or pool)


def place_from_template(
    rendered: list[np.ndarray],
    slots: tuple[Glyph, Glyph, Glyph],
    canvas_shape: tuple[int, int],
    rng: random.Random,
    x_jitter: float,
    y_jitter: float,
) -> list[tuple[int, int]]:
    canvas_h, canvas_w = canvas_shape
    positions: list[list[int]] = []

    for image, slot in zip(rendered, slots):
        h, w = image.shape[:2]
        center_x = slot.x + slot.width / 2
        bottom = slot.y + slot.height

        x = int(round(center_x - w / 2 + rng.uniform(-x_jitter, x_jitter)))
        y = int(round(bottom - h + rng.uniform(-y_jitter, y_jitter)))
        positions.append([x, y])

    # Preserve template ordering and approximate gaps.
    template_gaps = [
        slots[1].x - (slots[0].x + slots[0].width),
        slots[2].x - (slots[1].x + slots[1].width),
    ]

    for index in (1, 2):
        previous_x = positions[index - 1][0]
        previous_w = rendered[index - 1].shape[1]
        desired_gap = max(-3, min(template_gaps[index - 1], 8))
        minimum_x = previous_x + previous_w + desired_gap
        if positions[index][0] < minimum_x:
            positions[index][0] = minimum_x

    group_left = min(position[0] for position in positions)
    group_right = max(
        position[0] + image.shape[1]
        for position, image in zip(positions, rendered)
    )

    if group_left < 1:
        shift = 1 - group_left
        for position in positions:
            position[0] += shift

    if group_right >= canvas_w:
        shift = group_right - canvas_w + 2
        for position in positions:
            position[0] -= shift

    final: list[tuple[int, int]] = []
    for (x, y), image in zip(positions, rendered):
        h, w = image.shape[:2]
        final.append((
            max(0, min(canvas_w - w, x)),
            max(0, min(canvas_h - h, y)),
        ))
    return final


def alpha_composite(
    canvas: np.ndarray,
    foreground: np.ndarray,
    x: int,
    y: int,
) -> None:
    h, w = foreground.shape[:2]
    roi = canvas[y:y + h, x:x + w]

    alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    rgb = foreground[:, :, :3].astype(np.float32)
    canvas[y:y + h, x:x + w] = np.clip(
        rgb * alpha + roi.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def sample_label(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789") for _ in range(3))


def generate_one(
    label: str,
    template: Template,
    by_digit_position: dict[tuple[str, int], list[Glyph]],
    by_digit: dict[str, list[Glyph]],
    rng: random.Random,
    scale_jitter: float,
    x_jitter: float,
    y_jitter: float,
    recolor_strength: float,
) -> tuple[np.ndarray, list[Glyph], list[tuple[int, int]]]:
    canvas = cv2.imread(str(template.background.path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise ValueError(f"Could not read {template.background.path}")

    chosen: list[Glyph] = []
    rendered: list[np.ndarray] = []

    for position, (digit, slot) in enumerate(
        zip(label, template.slots),
        start=1,
    ):
        glyph = choose_glyph(
            digit,
            position,
            template.source_tile,
            by_digit_position,
            by_digit,
            rng,
        )
        source_image = load_bgra(glyph.path)
        template_style = load_bgra(slot.path)
        recolored = recolor_to_template(
            source_image,
            template_style,
            recolor_strength,
        )
        fitted = fit_to_slot(
            recolored,
            slot,
            scale_jitter,
            rng,
        )
        chosen.append(glyph)
        rendered.append(fitted)

    positions = place_from_template(
        rendered,
        template.slots,
        canvas.shape[:2],
        rng,
        x_jitter,
        y_jitter,
    )

    for image, (x, y) in zip(rendered, positions):
        alpha_composite(canvas, image, x, y)

    return canvas, chosen, positions


def make_preview(
    paths: list[Path],
    labels: list[str],
    output_path: Path,
    limit: int,
) -> None:
    count = min(limit, len(paths))
    if count <= 0:
        return

    tile_w, tile_h, caption_h = 200, 200, 26
    columns = min(10, max(1, math.ceil(math.sqrt(count))))
    rows = math.ceil(count / columns)

    sheet = np.full(
        (rows * (tile_h + caption_h), columns * tile_w, 3),
        245,
        dtype=np.uint8,
    )

    for index, (path, label) in enumerate(zip(paths[:count], labels[:count])):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (tile_w, tile_h))

        row = index // columns
        column = index % columns
        x = column * tile_w
        y = row * (tile_h + caption_h)

        sheet[y:y + tile_h, x:x + tile_w] = image
        cv2.putText(
            sheet,
            label,
            (x + 7, y + tile_h + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), sheet)


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    assets = args.assets.resolve()
    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    glyphs, backgrounds = load_assets(assets, args.min_quality)
    by_digit_position, by_digit = build_pools(glyphs)
    templates = build_templates(glyphs, backgrounds)

    records: list[dict[str, object]] = []
    paths: list[Path] = []
    labels: list[str] = []

    digit_counts: Counter[str] = Counter()
    glyph_use: Counter[str] = Counter()
    template_use: Counter[str] = Counter()

    for index in range(args.count):
        label = sample_label(rng)
        template = rng.choice(templates)

        image, chosen, positions = generate_one(
            label,
            template,
            by_digit_position,
            by_digit,
            rng,
            args.scale_jitter,
            args.x_jitter,
            args.y_jitter,
            args.recolor_strength,
        )

        filename = f"{index:07d}_{label}.png"
        path = images_dir / filename
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")

        record: dict[str, object] = {
            "image_path": path.relative_to(output).as_posix(),
            "label": label,
            "template_source": template.source_tile,
            "background_path": template.background.path.as_posix(),
        }

        for position, (glyph, (x, y)) in enumerate(
            zip(chosen, positions),
            start=1,
        ):
            record[f"digit_{position}"] = label[position - 1]
            record[f"glyph_{position}"] = glyph.path.as_posix()
            record[f"x_{position}"] = x
            record[f"y_{position}"] = y
            glyph_use[glyph.path.as_posix()] += 1

        records.append(record)
        paths.append(path)
        labels.append(label)
        digit_counts.update(label)
        template_use[template.source_tile] += 1

        done = index + 1
        if done % args.log_every == 0 or done == args.count:
            print(f"Generated {done:,}/{args.count:,}")

    with (output / "labels.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    make_preview(
        paths,
        labels,
        output / "preview.png",
        args.preview_count,
    )

    summary = {
        "count": args.count,
        "seed": args.seed,
        "glyph_pool_size": len(glyphs),
        "template_count": len(templates),
        "min_quality": args.min_quality,
        "scale_jitter": args.scale_jitter,
        "x_jitter": args.x_jitter,
        "y_jitter": args.y_jitter,
        "recolor_strength": args.recolor_strength,
        "digit_counts": {
            digit: digit_counts[digit]
            for digit in "0123456789"
        },
        "unique_glyphs_used": len(glyph_use),
        "unique_templates_used": len(template_use),
        "max_glyph_reuse": max(glyph_use.values(), default=0),
        "max_template_reuse": max(template_use.values(), default=0),
    }

    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Done.")
    print(json.dumps(summary, indent=2))
    print(f"Preview: {output / 'preview.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Template-based real-glyph CAPTCHA compositor."
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("data/real_assets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/composed_preview_v2"),
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=100)
    parser.add_argument("--min-quality", type=float, default=0.25)
    parser.add_argument("--scale-jitter", type=float, default=0.025)
    parser.add_argument("--x-jitter", type=float, default=1.5)
    parser.add_argument("--y-jitter", type=float, default=1.0)
    parser.add_argument("--recolor-strength", type=float, default=0.90)
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
    if not 0 <= args.scale_jitter <= 0.25:
        parser.error("--scale-jitter must be between 0 and 0.25")
    if not 0 <= args.recolor_strength <= 1:
        parser.error("--recolor-strength must be between 0 and 1")

    return args


if __name__ == "__main__":
    build(parse_args())
