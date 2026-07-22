#!/usr/bin/env python3
"""
Render coherent synthetic CAPTCHA tiles from clean digit prototypes.

One real template controls the whole CAPTCHA:
- background
- layout
- baseline
- three slot heights
- three slot colors
- underline style

All digits use the same prototype weight family.

Run:
python synthetic_renderer.py \
  --assets data/real_assets \
  --prototypes data/digit_prototypes \
  --output data/rendered_preview \
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
class GlyphSlot:
    path: Path
    position: int
    x: int
    y: int
    width: int
    height: int
    source_tile: str


@dataclass(frozen=True)
class Template:
    source_tile: str
    background_path: Path
    slots: tuple[GlyphSlot, GlyphSlot, GlyphSlot]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_bgra(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Invalid BGRA image: {path}")
    return image


def alpha_bounds(image: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(image[:, :, 3] > 16)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def trim(image: np.ndarray, padding: int = 2) -> np.ndarray:
    bounds = alpha_bounds(image)
    if bounds is None:
        return image
    x0, y0, x1, y1 = bounds
    h, w = image.shape[:2]
    return image[
        max(0, y0 - padding):min(h, y1 + padding),
        max(0, x0 - padding):min(w, x1 + padding),
    ]


def weighted_mean_lab(image: np.ndarray) -> np.ndarray:
    alpha = image[:, :, 3].astype(np.float32) / 255.0
    valid = alpha > 0.08
    if not np.any(valid):
        return np.array([128, 128, 128], dtype=np.float32)

    lab = cv2.cvtColor(
        image[:, :, :3],
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    return np.average(
        lab[valid],
        axis=0,
        weights=alpha[valid],
    ).astype(np.float32)


def colorize_prototype(
    prototype: np.ndarray,
    target_style: np.ndarray,
) -> np.ndarray:
    alpha = prototype[:, :, 3]
    target_lab = weighted_mean_lab(target_style)

    solid_lab = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    solid_lab[:, :, 0] = np.uint8(np.clip(target_lab[0], 0, 255))
    solid_lab[:, :, 1] = np.uint8(np.clip(target_lab[1], 0, 255))
    solid_lab[:, :, 2] = np.uint8(np.clip(target_lab[2], 0, 255))
    bgr = cv2.cvtColor(solid_lab, cv2.COLOR_LAB2BGR)

    return np.dstack([bgr, alpha])


def resize_to_height(
    image: np.ndarray,
    target_height: int,
    width_scale: float,
) -> np.ndarray:
    image = trim(image)
    h, w = image.shape[:2]
    scale = target_height / max(1, h)
    target_w = max(2, int(round(w * scale * width_scale)))

    return cv2.resize(
        image,
        (target_w, target_height),
        interpolation=cv2.INTER_CUBIC,
    )


def alpha_composite(
    canvas: np.ndarray,
    foreground: np.ndarray,
    x: int,
    y: int,
) -> None:
    h, w = foreground.shape[:2]
    roi = canvas[y:y + h, x:x + w]
    alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0

    canvas[y:y + h, x:x + w] = np.clip(
        foreground[:, :, :3].astype(np.float32) * alpha
        + roi.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def build_templates(assets: Path) -> list[Template]:
    glyph_rows = read_csv(assets / "manifests" / "glyphs.csv")
    background_rows = read_csv(
        assets / "manifests" / "backgrounds.csv"
    )

    grouped: dict[str, dict[int, GlyphSlot]] = defaultdict(dict)
    for row in glyph_rows:
        source = row["source_tile"]
        grouped[source][int(row["position"])] = GlyphSlot(
            path=resolve_path(assets, row["glyph_path"]),
            position=int(row["position"]),
            x=int(row["x"]),
            y=int(row["y"]),
            width=int(row["width"]),
            height=int(row["height"]),
            source_tile=source,
        )

    backgrounds = {
        row["source_tile"]: resolve_path(assets, row["background_path"])
        for row in background_rows
    }

    templates = []
    for source, slots in grouped.items():
        if set(slots) != {1, 2, 3} or source not in backgrounds:
            continue
        templates.append(
            Template(
                source_tile=source,
                background_path=backgrounds[source],
                slots=(slots[1], slots[2], slots[3]),
            )
        )

    if not templates:
        raise RuntimeError("No templates found")
    return templates


def choose_weight(template: Template) -> str:
    mean_height = np.mean([slot.height for slot in template.slots])
    mean_width = np.mean([slot.width for slot in template.slots])
    ratio = mean_width / max(1.0, mean_height)

    if ratio < 0.38:
        return "thin"
    if ratio > 0.52:
        return "bold"
    return "medium"


def extract_template_underline(template: Template) -> list[tuple[np.ndarray, int, int]]:
    layers = []

    for slot in template.slots:
        image = load_bgra(slot.path)
        alpha = image[:, :, 3]
        h, w = alpha.shape

        binary = (alpha > 16).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(8, int(round(w * 0.42))), 1),
        )
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        lower = np.zeros_like(lines)
        lower[int(h * 0.52):, :] = 255
        lines = cv2.bitwise_and(lines, lower)

        if np.count_nonzero(lines) < 4:
            continue

        underline = image.copy()
        underline[:, :, 3] = cv2.bitwise_and(alpha, lines)
        layers.append((underline, slot.x, slot.y))

    return layers


def render_one(
    label: str,
    template: Template,
    prototype_root: Path,
    rng: random.Random,
    x_jitter: float,
    y_jitter: float,
    scale_jitter: float,
) -> np.ndarray:
    canvas = cv2.imread(str(template.background_path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise ValueError(template.background_path)

    weight = choose_weight(template)
    rendered = []

    for digit, slot in zip(label, template.slots):
        prototype = load_bgra(prototype_root / digit / f"{weight}.png")
        target_style = load_bgra(slot.path)
        colored = colorize_prototype(prototype, target_style)

        target_height = max(
            10,
            int(round(
                slot.height
                * rng.uniform(1 - scale_jitter, 1 + scale_jitter)
            )),
        )

        target_aspect = slot.width / max(1, slot.height)
        proto_bounds = alpha_bounds(prototype)
        if proto_bounds is None:
            raise RuntimeError("Empty prototype")
        x0, y0, x1, y1 = proto_bounds
        proto_aspect = (x1 - x0) / max(1, y1 - y0)
        width_scale = np.clip(
            target_aspect / max(0.1, proto_aspect),
            0.78,
            1.28,
        )

        rendered.append(
            resize_to_height(colored, target_height, float(width_scale))
        )

    positions: list[list[int]] = []
    for image, slot in zip(rendered, template.slots):
        h, w = image.shape[:2]
        center = slot.x + slot.width / 2
        baseline = slot.y + slot.height
        positions.append([
            int(round(center - w / 2 + rng.uniform(-x_jitter, x_jitter))),
            int(round(baseline - h + rng.uniform(-y_jitter, y_jitter))),
        ])

    # Enforce readable but template-like gaps.
    for index in (1, 2):
        previous_right = positions[index - 1][0] + rendered[index - 1].shape[1]
        original_gap = (
            template.slots[index].x
            - (template.slots[index - 1].x + template.slots[index - 1].width)
        )
        safe_gap = max(1, min(7, original_gap))
        positions[index][0] = max(
            positions[index][0],
            previous_right + safe_gap,
        )

    canvas_h, canvas_w = canvas.shape[:2]
    left = min(p[0] for p in positions)
    right = max(
        p[0] + img.shape[1]
        for p, img in zip(positions, rendered)
    )

    if left < 2:
        shift = 2 - left
        for p in positions:
            p[0] += shift
    if right > canvas_w - 2:
        shift = right - (canvas_w - 2)
        for p in positions:
            p[0] -= shift

    for image, (x, y) in zip(rendered, positions):
        h, w = image.shape[:2]
        x = max(0, min(canvas_w - w, x))
        y = max(0, min(canvas_h - h, y))
        alpha_composite(canvas, image, x, y)

    # Paste original template underlines as CAPTCHA-level decoration.
    for underline, x, y in extract_template_underline(template):
        h, w = underline.shape[:2]
        if 0 <= x <= canvas_w - w and 0 <= y <= canvas_h - h:
            alpha_composite(canvas, underline, x, y)

    return canvas


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
    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    templates = build_templates(args.assets.resolve())
    records = []
    paths = []
    labels = []
    digit_counts: Counter[str] = Counter()

    for index in range(args.count):
        label = "".join(rng.choice("0123456789") for _ in range(3))
        template = rng.choice(templates)

        image = render_one(
            label=label,
            template=template,
            prototype_root=args.prototypes.resolve(),
            rng=rng,
            x_jitter=args.x_jitter,
            y_jitter=args.y_jitter,
            scale_jitter=args.scale_jitter,
        )

        filename = f"{index:07d}_{label}.png"
        path = images_dir / filename
        cv2.imwrite(str(path), image)

        records.append({
            "image_path": path.relative_to(output).as_posix(),
            "label": label,
            "template_source": template.source_tile,
            "weight": choose_weight(template),
        })
        paths.append(path)
        labels.append(label)
        digit_counts.update(label)

        done = index + 1
        if done % args.log_every == 0 or done == args.count:
            print(f"Generated {done:,}/{args.count:,}")

    with (output / "labels.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "template_source", "weight"],
        )
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
        "template_count": len(templates),
        "digit_counts": {
            digit: digit_counts[digit]
            for digit in "0123456789"
        },
    }

    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Preview: {output / 'preview.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("data/real_assets"),
    )
    parser.add_argument(
        "--prototypes",
        type=Path,
        default=Path("data/digit_prototypes"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rendered_preview"),
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=100)
    parser.add_argument("--scale-jitter", type=float, default=0.02)
    parser.add_argument("--x-jitter", type=float, default=0.75)
    parser.add_argument("--y-jitter", type=float, default=0.75)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if not args.assets.exists():
        parser.error(f"Missing assets: {args.assets}")
    if not args.prototypes.exists():
        parser.error(f"Missing prototypes: {args.prototypes}")
    return args


if __name__ == "__main__":
    build(parse_args())
