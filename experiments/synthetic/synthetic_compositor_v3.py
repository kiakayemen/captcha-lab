#!/usr/bin/env python3
"""
Synthetic CAPTCHA compositor v3.

Fixed direction:
- real inpainted backgrounds
- real extracted glyph shapes
- one real CAPTCHA supplies layout, palette, and underline style
- candidate glyphs are matched to the template by stroke width, aspect,
  edge softness, density, and position
- underlines are removed from digit masks and pasted separately once

Run:
python synthetic_compositor_v3.py \
  --assets data/real_assets \
  --output data/composed_preview_v3 \
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
class GlyphStyle:
    aspect: float
    density: float
    stroke: float
    edge_softness: float
    opacity: float


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


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_assets(
    root: Path,
    min_quality: float,
) -> tuple[list[Glyph], list[Background]]:
    glyph_manifest = root / "manifests" / "glyphs.csv"
    background_manifest = root / "manifests" / "backgrounds.csv"

    glyphs: list[Glyph] = []
    for row in read_csv(glyph_manifest):
        quality = float(row.get("mask_quality", "1") or 1)
        if quality < min_quality:
            continue

        path = resolve_path(root, row["glyph_path"])
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
        path = resolve_path(root, row["background_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        backgrounds.append(
            Background(path=path, source_tile=row["source_tile"])
        )

    if not glyphs or not backgrounds:
        raise RuntimeError("No usable assets found")

    return glyphs, backgrounds


def load_bgra(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected BGRA image: {path}")
    return image


def build_templates(
    glyphs: list[Glyph],
    backgrounds: list[Background],
) -> list[Template]:
    grouped: dict[str, dict[int, Glyph]] = defaultdict(dict)
    for glyph in glyphs:
        grouped[glyph.source_tile][glyph.position] = glyph

    bg_by_source = {bg.source_tile: bg for bg in backgrounds}
    templates: list[Template] = []

    for source, slots in grouped.items():
        if set(slots) != {1, 2, 3} or source not in bg_by_source:
            continue
        templates.append(
            Template(
                source_tile=source,
                background=bg_by_source[source],
                slots=(slots[1], slots[2], slots[3]),
            )
        )

    if not templates:
        raise RuntimeError("No complete templates found")
    return templates


def split_digit_and_underline(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (digit_bgra, underline_bgra).

    Horizontal structures near the lower half are treated as underlines.
    Digit strokes remain in the digit image.
    """
    alpha = image[:, :, 3]
    binary = (alpha > 16).astype(np.uint8) * 255
    h, w = binary.shape

    kernel_w = max(7, int(round(w * 0.34)))
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_w, 1),
    )
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # Only consider line-like structures in the lower 60% of the crop.
    allowed = np.zeros_like(lines)
    allowed[int(h * 0.40):, :] = 255
    lines = cv2.bitwise_and(lines, allowed)

    # Expand slightly to capture antialiased underline edges.
    lines = cv2.dilate(
        lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    underline_alpha = cv2.bitwise_and(alpha, lines)
    digit_alpha = cv2.subtract(alpha, underline_alpha)

    # Remove tiny leftovers from the digit mask.
    digit_binary = (digit_alpha > 16).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        digit_binary,
        8,
    )
    cleaned = np.zeros_like(digit_binary)
    for idx in range(1, count):
        _, _, cw, ch, area = stats[idx]
        if area >= 5 and ch >= max(3, int(h * 0.06)):
            cleaned[labels == idx] = 255
    digit_alpha = cv2.bitwise_and(digit_alpha, cleaned)

    digit = image.copy()
    digit[:, :, 3] = digit_alpha

    underline = image.copy()
    underline[:, :, 3] = underline_alpha
    return digit, underline


def alpha_bounds(image: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(image[:, :, 3] > 16)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def trim(image: np.ndarray, padding: int = 2) -> np.ndarray:
    bounds = alpha_bounds(image)
    if bounds is None:
        return np.zeros((2, 2, 4), dtype=np.uint8)

    x0, y0, x1, y1 = bounds
    h, w = image.shape[:2]
    return image[
        max(0, y0 - padding):min(h, y1 + padding),
        max(0, x0 - padding):min(w, x1 + padding),
    ]


def style_features(image: np.ndarray) -> GlyphStyle:
    alpha = image[:, :, 3].astype(np.float32)
    binary = (alpha > 32).astype(np.uint8)
    bounds = alpha_bounds(image)

    if bounds is None:
        return GlyphStyle(0, 0, 0, 0, 0)

    x0, y0, x1, y1 = bounds
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    crop = binary[y0:y1, x0:x1]

    density = float(crop.mean())
    distance = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    stroke_values = distance[crop > 0] * 2.0
    stroke = float(stroke_values.mean()) if stroke_values.size else 0.0

    fractional = alpha[(alpha > 0) & (alpha < 255)]
    edge_softness = (
        float(len(fractional) / max(1, np.count_nonzero(alpha)))
        if np.count_nonzero(alpha)
        else 0.0
    )
    opacity = float(alpha[alpha > 0].mean() / 255.0) if np.any(alpha > 0) else 0.0

    return GlyphStyle(
        aspect=float(width / height),
        density=density,
        stroke=stroke,
        edge_softness=edge_softness,
        opacity=opacity,
    )


def style_distance(
    candidate: GlyphStyle,
    target: GlyphStyle,
) -> float:
    return (
        2.0 * abs(candidate.aspect - target.aspect)
        + 1.6 * abs(candidate.density - target.density)
        + 0.35 * abs(candidate.stroke - target.stroke)
        + 1.2 * abs(candidate.edge_softness - target.edge_softness)
        + 0.8 * abs(candidate.opacity - target.opacity)
    )


def weighted_mean_lab(image: np.ndarray) -> np.ndarray:
    alpha = image[:, :, 3].astype(np.float32) / 255.0
    valid = alpha > 0.08
    if not np.any(valid):
        return np.array([128, 128, 128], dtype=np.float32)

    lab = cv2.cvtColor(
        image[:, :, :3],
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    return np.average(lab[valid], axis=0, weights=alpha[valid]).astype(np.float32)


def recolor(
    source: np.ndarray,
    target: np.ndarray,
    strength: float,
) -> np.ndarray:
    source_lab = cv2.cvtColor(
        source[:, :, :3],
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    shift = (
        weighted_mean_lab(target)
        - weighted_mean_lab(source)
    ) * strength

    source_lab[:, :, 0] += shift[0] * 0.55
    source_lab[:, :, 1] += shift[1]
    source_lab[:, :, 2] += shift[2]
    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)

    result = source.copy()
    result[:, :, :3] = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
    return result


def resize_to_height(
    image: np.ndarray,
    target_height: int,
) -> np.ndarray:
    image = trim(image)
    h, w = image.shape[:2]
    scale = target_height / max(1, h)
    target_width = max(2, int(round(w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )


def alpha_composite(
    canvas: np.ndarray,
    foreground: np.ndarray,
    x: int,
    y: int,
) -> None:
    h, w = foreground.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]

    left = max(0, x)
    top = max(0, y)
    right = min(canvas_w, x + w)
    bottom = min(canvas_h, y + h)

    if right <= left or bottom <= top:
        return

    fx0 = left - x
    fy0 = top - y
    fx1 = fx0 + right - left
    fy1 = fy0 + bottom - top

    fg = foreground[fy0:fy1, fx0:fx1]
    roi = canvas[top:bottom, left:right]
    alpha = fg[:, :, 3:4].astype(np.float32) / 255.0

    canvas[top:bottom, left:right] = np.clip(
        fg[:, :, :3].astype(np.float32) * alpha
        + roi.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def build_pools(
    glyphs: list[Glyph],
) -> tuple[
    dict[tuple[str, int], list[Glyph]],
    dict[str, list[Glyph]],
    dict[Path, tuple[np.ndarray, np.ndarray, GlyphStyle]],
]:
    by_digit_position: dict[tuple[str, int], list[Glyph]] = defaultdict(list)
    by_digit: dict[str, list[Glyph]] = defaultdict(list)
    cache: dict[Path, tuple[np.ndarray, np.ndarray, GlyphStyle]] = {}

    for glyph in glyphs:
        original = load_bgra(glyph.path)
        digit_image, underline = split_digit_and_underline(original)
        style = style_features(digit_image)
        cache[glyph.path] = (digit_image, underline, style)
        by_digit_position[(glyph.digit, glyph.position)].append(glyph)
        by_digit[glyph.digit].append(glyph)

    return dict(by_digit_position), dict(by_digit), cache


def choose_compatible_glyph(
    digit: str,
    position: int,
    template_source: str,
    target_style: GlyphStyle,
    by_digit_position: dict[tuple[str, int], list[Glyph]],
    by_digit: dict[str, list[Glyph]],
    cache: dict[Path, tuple[np.ndarray, np.ndarray, GlyphStyle]],
    rng: random.Random,
    top_k: int,
) -> Glyph:
    pool = by_digit_position.get((digit, position), []) or by_digit[digit]
    pool = [
        glyph for glyph in pool
        if glyph.source_tile != template_source
    ] or pool

    ranked = sorted(
        pool,
        key=lambda glyph: style_distance(
            cache[glyph.path][2],
            target_style,
        ),
    )
    return rng.choice(ranked[:max(1, min(top_k, len(ranked)))])


def extract_template_underline_layer(
    template: Template,
    cache: dict[Path, tuple[np.ndarray, np.ndarray, GlyphStyle]],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    h, w = canvas_shape
    layer = np.zeros((h, w, 4), dtype=np.uint8)

    for slot in template.slots:
        underline = cache[slot.path][1]
        if alpha_bounds(underline) is None:
            continue

        uh, uw = underline.shape[:2]
        x0 = max(0, slot.x)
        y0 = max(0, slot.y)
        x1 = min(w, x0 + uw)
        y1 = min(h, y0 + uh)
        if x1 <= x0 or y1 <= y0:
            continue

        source = underline[:y1 - y0, :x1 - x0]
        existing = layer[y0:y1, x0:x1]

        # Keep whichever underline pixel has stronger alpha.
        take = source[:, :, 3] > existing[:, :, 3]
        existing[take] = source[take]

    return layer


def place_digits(
    rendered: list[np.ndarray],
    template: Template,
    rng: random.Random,
    x_jitter: float,
    y_jitter: float,
    canvas_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    canvas_h, canvas_w = canvas_shape
    positions: list[list[int]] = []

    for image, slot in zip(rendered, template.slots):
        ih, iw = image.shape[:2]
        center_x = slot.x + slot.width / 2
        baseline = slot.y + slot.height

        x = int(round(center_x - iw / 2 + rng.uniform(-x_jitter, x_jitter)))
        y = int(round(baseline - ih + rng.uniform(-y_jitter, y_jitter)))
        positions.append([x, y])

    # Preserve template inter-slot centers rather than source glyph positions.
    for idx in (1, 2):
        prev_right = positions[idx - 1][0] + rendered[idx - 1].shape[1]
        template_gap = (
            template.slots[idx].x
            - (template.slots[idx - 1].x + template.slots[idx - 1].width)
        )
        min_gap = max(-2, min(6, template_gap))
        positions[idx][0] = max(
            positions[idx][0],
            prev_right + min_gap,
        )

    left = min(p[0] for p in positions)
    right = max(
        p[0] + image.shape[1]
        for p, image in zip(positions, rendered)
    )

    if left < 1:
        shift = 1 - left
        for p in positions:
            p[0] += shift
    if right >= canvas_w:
        shift = right - canvas_w + 2
        for p in positions:
            p[0] -= shift

    result = []
    for (x, y), image in zip(positions, rendered):
        ih, iw = image.shape[:2]
        result.append((
            max(0, min(canvas_w - iw, x)),
            max(0, min(canvas_h - ih, y)),
        ))
    return result


def generate_one(
    label: str,
    template: Template,
    by_digit_position: dict[tuple[str, int], list[Glyph]],
    by_digit: dict[str, list[Glyph]],
    cache: dict[Path, tuple[np.ndarray, np.ndarray, GlyphStyle]],
    rng: random.Random,
    top_k: int,
    scale_jitter: float,
    x_jitter: float,
    y_jitter: float,
    recolor_strength: float,
    underline_probability: float,
) -> tuple[np.ndarray, list[Glyph], list[tuple[int, int]]]:
    canvas = cv2.imread(str(template.background.path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise ValueError(template.background.path)

    selected: list[Glyph] = []
    rendered: list[np.ndarray] = []

    for position, (digit, slot) in enumerate(
        zip(label, template.slots),
        start=1,
    ):
        template_digit, _, template_style = cache[slot.path]
        chosen = choose_compatible_glyph(
            digit=digit,
            position=position,
            template_source=template.source_tile,
            target_style=template_style,
            by_digit_position=by_digit_position,
            by_digit=by_digit,
            cache=cache,
            rng=rng,
            top_k=top_k,
        )

        source_digit = cache[chosen.path][0]
        source_digit = recolor(
            source_digit,
            template_digit,
            recolor_strength,
        )

        target_height = max(
            8,
            int(round(
                slot.height
                * rng.uniform(1 - scale_jitter, 1 + scale_jitter)
            )),
        )
        fitted = resize_to_height(source_digit, target_height)

        selected.append(chosen)
        rendered.append(fitted)

    positions = place_digits(
        rendered,
        template,
        rng,
        x_jitter,
        y_jitter,
        canvas.shape[:2],
    )

    for image, (x, y) in zip(rendered, positions):
        alpha_composite(canvas, image, x, y)

    underline_layer = extract_template_underline_layer(
        template,
        cache,
        canvas.shape[:2],
    )
    if (
        alpha_bounds(underline_layer) is not None
        and rng.random() <= underline_probability
    ):
        alpha_composite(canvas, underline_layer, 0, 0)

    return canvas, selected, positions


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

    for idx, (path, label) in enumerate(zip(paths[:count], labels[:count])):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (tile_w, tile_h))

        row = idx // columns
        col = idx % columns
        x = col * tile_w
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
    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    glyphs, backgrounds = load_assets(
        args.assets.resolve(),
        args.min_quality,
    )
    templates = build_templates(glyphs, backgrounds)
    by_digit_position, by_digit, cache = build_pools(glyphs)

    records: list[dict[str, object]] = []
    paths: list[Path] = []
    labels: list[str] = []
    digit_counts: Counter[str] = Counter()
    glyph_use: Counter[str] = Counter()
    template_use: Counter[str] = Counter()

    for idx in range(args.count):
        label = "".join(rng.choice("0123456789") for _ in range(3))
        template = rng.choice(templates)

        image, chosen, positions = generate_one(
            label=label,
            template=template,
            by_digit_position=by_digit_position,
            by_digit=by_digit,
            cache=cache,
            rng=rng,
            top_k=args.top_k,
            scale_jitter=args.scale_jitter,
            x_jitter=args.x_jitter,
            y_jitter=args.y_jitter,
            recolor_strength=args.recolor_strength,
            underline_probability=args.underline_probability,
        )

        filename = f"{idx:07d}_{label}.png"
        path = images_dir / filename
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")

        record: dict[str, object] = {
            "image_path": path.relative_to(output).as_posix(),
            "label": label,
            "template_source": template.source_tile,
        }

        for position, (glyph, (x, y)) in enumerate(
            zip(chosen, positions),
            start=1,
        ):
            record[f"glyph_{position}"] = glyph.path.as_posix()
            record[f"x_{position}"] = x
            record[f"y_{position}"] = y
            glyph_use[glyph.path.as_posix()] += 1

        records.append(record)
        paths.append(path)
        labels.append(label)
        digit_counts.update(label)
        template_use[template.source_tile] += 1

        done = idx + 1
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
        "top_k": args.top_k,
        "scale_jitter": args.scale_jitter,
        "x_jitter": args.x_jitter,
        "y_jitter": args.y_jitter,
        "recolor_strength": args.recolor_strength,
        "underline_probability": args.underline_probability,
        "digit_counts": {
            digit: digit_counts[digit]
            for digit in "0123456789"
        },
        "unique_glyphs_used": len(glyph_use),
        "unique_templates_used": len(template_use),
    }

    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))
    print(f"Preview: {output / 'preview.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Style-matched real-glyph CAPTCHA compositor."
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("data/real_assets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/composed_preview_v3"),
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=100)
    parser.add_argument("--min-quality", type=float, default=0.25)
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Randomly sample among the closest style-matched glyphs.",
    )
    parser.add_argument("--scale-jitter", type=float, default=0.015)
    parser.add_argument("--x-jitter", type=float, default=1.0)
    parser.add_argument("--y-jitter", type=float, default=0.75)
    parser.add_argument("--recolor-strength", type=float, default=0.92)
    parser.add_argument(
        "--underline-probability",
        type=float,
        default=1.0,
        help="Use the template underline when one exists.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if not args.assets.exists():
        parser.error(f"Assets directory does not exist: {args.assets}")
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not 0 <= args.min_quality <= 1:
        parser.error("--min-quality must be in [0, 1]")
    if not 0 <= args.underline_probability <= 1:
        parser.error("--underline-probability must be in [0, 1]")

    return args


if __name__ == "__main__":
    build(parse_args())
