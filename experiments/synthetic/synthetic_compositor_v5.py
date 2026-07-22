#!/usr/bin/env python3
"""
Style-family medoid CAPTCHA compositor v5.

Properties:
- authentic glyph pixels only
- all three digits drawn from the same learned style family
- medoid glyphs only
- uniform scaling only
- template layout/background/palette
- generation-time rejection for overlap, bad spacing, height mismatch,
  stroke mismatch, and low contrast

Run:
python synthetic_compositor_v5.py \
  --style-bank data/style_bank \
  --output data/composed_preview_v5 \
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
class Slot:
    path: Path
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Template:
    source_tile: str
    style_cluster: int
    background_path: Path
    slots: tuple[Slot, Slot, Slot]
    mean_stroke: float
    stroke_std: float
    mean_height: float
    height_std: float
    spacing_1: float
    spacing_2: float
    baseline_std: float


@dataclass(frozen=True)
class Medoid:
    style_cluster: int
    digit: str
    position: int
    rank: int
    glyph_path: Path
    source_tile: str
    aspect: float
    density: float
    stroke_width: float
    edge_softness: float
    mean_l: float
    mean_a: float
    mean_b: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def recolor_to_slot(
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

    source_lab[:, :, 0] += shift[0] * 0.45
    source_lab[:, :, 1] += shift[1]
    source_lab[:, :, 2] += shift[2]
    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)

    output = source.copy()
    output[:, :, :3] = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
    return output


def resize_uniform(
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
    roi = canvas[y:y + h, x:x + w]
    alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    canvas[y:y + h, x:x + w] = np.clip(
        foreground[:, :, :3].astype(np.float32) * alpha
        + roi.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def load_style_bank(
    root: Path,
) -> tuple[
    list[Template],
    dict[tuple[int, str, int], list[Medoid]],
]:
    templates = []
    for row in read_csv(root / "templates.csv"):
        templates.append(
            Template(
                source_tile=row["source_tile"],
                style_cluster=int(row["style_cluster"]),
                background_path=Path(row["background_path"]),
                slots=(
                    Slot(
                        path=Path(row["slot_1_path"]),
                        x=int(row["x_1"]),
                        y=int(row["y_1"]),
                        width=int(row["width_1"]),
                        height=int(row["height_1"]),
                    ),
                    Slot(
                        path=Path(row["slot_2_path"]),
                        x=int(row["x_2"]),
                        y=int(row["y_2"]),
                        width=int(row["width_2"]),
                        height=int(row["height_2"]),
                    ),
                    Slot(
                        path=Path(row["slot_3_path"]),
                        x=int(row["x_3"]),
                        y=int(row["y_3"]),
                        width=int(row["width_3"]),
                        height=int(row["height_3"]),
                    ),
                ),
                mean_stroke=float(row["mean_stroke"]),
                stroke_std=float(row["stroke_std"]),
                mean_height=float(row["mean_height"]),
                height_std=float(row["height_std"]),
                spacing_1=float(row["spacing_1"]),
                spacing_2=float(row["spacing_2"]),
                baseline_std=float(row["baseline_std"]),
            )
        )

    pools: dict[tuple[int, str, int], list[Medoid]] = defaultdict(list)
    for row in read_csv(root / "medoids.csv"):
        medoid = Medoid(
            style_cluster=int(row["style_cluster"]),
            digit=row["digit"],
            position=int(row["position"]),
            rank=int(row["rank"]),
            glyph_path=Path(row["glyph_path"]),
            source_tile=row["source_tile"],
            aspect=float(row["aspect"]),
            density=float(row["density"]),
            stroke_width=float(row["stroke_width"]),
            edge_softness=float(row["edge_softness"]),
            mean_l=float(row["mean_l"]),
            mean_a=float(row["mean_a"]),
            mean_b=float(row["mean_b"]),
        )
        pools[(medoid.style_cluster, medoid.digit, medoid.position)].append(
            medoid
        )

    return templates, dict(pools)


def choose_medoid(
    template: Template,
    digit: str,
    position: int,
    pools: dict[tuple[int, str, int], list[Medoid]],
    rng: random.Random,
) -> Medoid | None:
    pool = pools.get((template.style_cluster, digit, position), [])
    if not pool:
        return None

    different_source = [
        medoid for medoid in pool
        if medoid.source_tile != template.source_tile
    ]
    if different_source:
        pool = different_source

    compatible = [
        medoid for medoid in pool
        if abs(medoid.stroke_width - template.mean_stroke)
        <= max(1.0, template.stroke_std + 0.9)
    ]
    if compatible:
        pool = compatible

    return rng.choice(pool[: min(3, len(pool))])


def placement(
    rendered: list[np.ndarray],
    template: Template,
    rng: random.Random,
    x_jitter: float,
    y_jitter: float,
) -> list[tuple[int, int]] | None:
    positions: list[list[int]] = []

    for image, slot in zip(rendered, template.slots):
        h, w = image.shape[:2]
        center = slot.x + slot.width / 2
        baseline = slot.y + slot.height
        positions.append([
            int(round(center - w / 2 + rng.uniform(-x_jitter, x_jitter))),
            int(round(baseline - h + rng.uniform(-y_jitter, y_jitter))),
        ])

    real_gaps = [template.spacing_1, template.spacing_2]
    for index in (1, 2):
        previous_right = (
            positions[index - 1][0]
            + rendered[index - 1].shape[1]
        )
        allowed_gap = int(round(np.clip(real_gaps[index - 1], 1, 8)))
        positions[index][0] = max(
            positions[index][0],
            previous_right + allowed_gap,
        )

    left = min(position[0] for position in positions)
    right = max(
        position[0] + image.shape[1]
        for position, image in zip(positions, rendered)
    )

    canvas_width = 200
    if right - left > canvas_width - 4:
        return None

    if left < 2:
        shift = 2 - left
        for position in positions:
            position[0] += shift
    if right > canvas_width - 2:
        shift = right - (canvas_width - 2)
        for position in positions:
            position[0] -= shift

    return [(int(x), int(y)) for x, y in positions]


def validate_composite(
    canvas: np.ndarray,
    rendered: list[np.ndarray],
    positions: list[tuple[int, int]],
    medoids: list[Medoid],
    template: Template,
) -> tuple[bool, str]:
    heights = np.array([image.shape[0] for image in rendered], dtype=np.float32)
    strokes = np.array([medoid.stroke_width for medoid in medoids])

    if heights.max() / max(1.0, heights.min()) > 1.35:
        return False, "height mismatch"
    if strokes.max() - strokes.min() > max(1.4, template.stroke_std + 1.0):
        return False, "stroke mismatch"

    for index in (1, 2):
        previous_right = positions[index - 1][0] + rendered[index - 1].shape[1]
        gap = positions[index][0] - previous_right
        if gap < 1 or gap > 12:
            return False, "spacing outside range"

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    contrast_values = []
    for image, (x, y) in zip(rendered, positions):
        h, w = image.shape[:2]
        alpha = image[:, :, 3] > 32
        if not np.any(alpha):
            return False, "empty glyph"
        roi = gray[y:y + h, x:x + w]
        foreground_mean = float(roi[alpha].mean())
        border_mean = float(np.median(roi))
        contrast_values.append(abs(foreground_mean - border_mean))

    if min(contrast_values) < 7.0:
        return False, "low contrast"

    return True, ""


def generate_one(
    label: str,
    template: Template,
    pools: dict[tuple[int, str, int], list[Medoid]],
    rng: random.Random,
    scale_jitter: float,
    x_jitter: float,
    y_jitter: float,
    recolor_strength: float,
) -> tuple[np.ndarray, list[Medoid], list[tuple[int, int]]] | None:
    canvas = cv2.imread(str(template.background_path), cv2.IMREAD_COLOR)
    if canvas is None:
        return None

    chosen: list[Medoid] = []
    rendered: list[np.ndarray] = []

    for position, (digit, slot) in enumerate(
        zip(label, template.slots),
        start=1,
    ):
        medoid = choose_medoid(
            template,
            digit,
            position,
            pools,
            rng,
        )
        if medoid is None:
            return None

        source = load_bgra(medoid.glyph_path)
        target = load_bgra(slot.path)
        source = recolor_to_slot(source, target, recolor_strength)

        target_height = max(
            10,
            int(round(
                slot.height
                * rng.uniform(1 - scale_jitter, 1 + scale_jitter)
            )),
        )
        rendered.append(resize_uniform(source, target_height))
        chosen.append(medoid)

    positions = placement(
        rendered,
        template,
        rng,
        x_jitter,
        y_jitter,
    )
    if positions is None:
        return None

    candidate = canvas.copy()
    for image, (x, y) in zip(rendered, positions):
        h, w = image.shape[:2]
        if x < 0 or y < 0 or x + w > candidate.shape[1] or y + h > candidate.shape[0]:
            return None
        alpha_composite(candidate, image, x, y)

    okay, _ = validate_composite(
        candidate,
        rendered,
        positions,
        chosen,
        template,
    )
    if not okay:
        return None

    return candidate, chosen, positions


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

    templates, pools = load_style_bank(args.style_bank.resolve())

    paths: list[Path] = []
    labels: list[str] = []
    rows: list[dict[str, object]] = []
    digit_counts: Counter[str] = Counter()
    rejections = 0
    attempts = 0

    while len(paths) < args.count:
        attempts += 1
        if attempts > args.count * args.max_attempt_multiplier:
            raise RuntimeError(
                "Too many rejected generations. "
                "Try fewer clusters or a larger medoid bank."
            )

        label = "".join(rng.choice("0123456789") for _ in range(3))
        template = rng.choice(templates)

        result = generate_one(
            label=label,
            template=template,
            pools=pools,
            rng=rng,
            scale_jitter=args.scale_jitter,
            x_jitter=args.x_jitter,
            y_jitter=args.y_jitter,
            recolor_strength=args.recolor_strength,
        )
        if result is None:
            rejections += 1
            continue

        image, medoids, positions = result
        index = len(paths)
        filename = f"{index:07d}_{label}.png"
        path = images_dir / filename
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")

        row: dict[str, object] = {
            "image_path": path.relative_to(output).as_posix(),
            "label": label,
            "template_source": template.source_tile,
            "style_cluster": template.style_cluster,
        }
        for position, (medoid, (x, y)) in enumerate(
            zip(medoids, positions),
            start=1,
        ):
            row[f"glyph_{position}"] = medoid.glyph_path.as_posix()
            row[f"x_{position}"] = x
            row[f"y_{position}"] = y

        rows.append(row)
        paths.append(path)
        labels.append(label)
        digit_counts.update(label)

        if len(paths) % args.log_every == 0 or len(paths) == args.count:
            print(
                f"Accepted {len(paths):,}/{args.count:,} | "
                f"rejected {rejections:,}"
            )

    with (output / "labels.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_preview(
        paths,
        labels,
        output / "preview.png",
        args.preview_count,
    )

    summary = {
        "count": args.count,
        "attempts": attempts,
        "rejections": rejections,
        "acceptance_rate": round(args.count / attempts, 4),
        "seed": args.seed,
        "template_count": len(templates),
        "pool_group_count": len(pools),
        "digit_counts": {
            digit: digit_counts[digit]
            for digit in "0123456789"
        },
    }

    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Preview: {output / 'preview.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style-bank",
        type=Path,
        default=Path("data/style_bank"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/composed_preview_v5"),
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=100)
    parser.add_argument("--scale-jitter", type=float, default=0.015)
    parser.add_argument("--x-jitter", type=float, default=0.75)
    parser.add_argument("--y-jitter", type=float, default=0.75)
    parser.add_argument("--recolor-strength", type=float, default=0.86)
    parser.add_argument("--max-attempt-multiplier", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if not args.style_bank.exists():
        parser.error(f"Missing style bank: {args.style_bank}")
    if args.count <= 0:
        parser.error("--count must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
