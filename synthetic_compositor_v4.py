#!/usr/bin/env python3
"""
Synthetic CAPTCHA compositor v4 — readability-first.

This intentionally avoids destructive glyph editing.

Rules:
- use authentic inpainted backgrounds
- use authentic RGBA glyphs unchanged
- use one real tile as the layout and color template
- reject visibly damaged/corrupt candidate glyph masks
- match candidates by position, aspect ratio, height, density, and stroke width
- preserve readable spacing and baseline alignment
- do NOT attempt underline removal yet

Run:
python synthetic_compositor_v4.py \
  --assets data/real_assets \
  --output data/composed_preview_v4 \
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


@dataclass(frozen=True)
class VisualFeatures:
    aspect: float
    density: float
    stroke: float
    component_count: int
    foreground_fraction: float
    height_fraction: float
    width_fraction: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_bgra(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected BGRA image: {path}")
    return image


def alpha_bounds(image: np.ndarray) -> tuple[int, int, int, int] | None:
    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def visual_features(image: np.ndarray) -> VisualFeatures:
    alpha = image[:, :, 3]
    binary = (alpha > 32).astype(np.uint8)
    bounds = alpha_bounds(image)

    if bounds is None:
        return VisualFeatures(0, 0, 0, 0, 0, 0, 0)

    x0, y0, x1, y1 = bounds
    crop = binary[y0:y1, x0:x1]
    h, w = binary.shape
    ch, cw = crop.shape

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        crop,
        connectivity=8,
    )
    substantial = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        comp_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area >= 5 and comp_h >= 3:
            substantial += 1

    distance = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    values = distance[crop > 0] * 2.0
    stroke = float(values.mean()) if values.size else 0.0

    return VisualFeatures(
        aspect=float(cw / max(1, ch)),
        density=float(crop.mean()),
        stroke=stroke,
        component_count=substantial,
        foreground_fraction=float(binary.mean()),
        height_fraction=float(ch / max(1, h)),
        width_fraction=float(cw / max(1, w)),
    )


def is_readable_candidate(
    glyph: Glyph,
    image: np.ndarray,
    features: VisualFeatures,
) -> bool:
    if glyph.quality < 0.35:
        return False

    bounds = alpha_bounds(image)
    if bounds is None:
        return False

    x0, y0, x1, y1 = bounds
    width = x1 - x0
    height = y1 - y0
    foreground = int(np.count_nonzero(image[:, :, 3] > 16))

    if width < 3 or height < 10 or foreground < 18:
        return False
    if features.component_count == 0 or features.component_count > 5:
        return False
    if not 0.03 <= features.foreground_fraction <= 0.72:
        return False
    if not 0.16 <= features.height_fraction <= 1.0:
        return False
    if features.aspect > 1.65:
        return False
    if features.stroke <= 0.2:
        return False

    return True


def load_assets(
    root: Path,
    min_quality: float,
) -> tuple[
    list[Glyph],
    list[Background],
    dict[Path, np.ndarray],
    dict[Path, VisualFeatures],
    list[dict[str, str]],
]:
    glyph_rows = read_csv(root / "manifests" / "glyphs.csv")
    background_rows = read_csv(root / "manifests" / "backgrounds.csv")

    cache: dict[Path, np.ndarray] = {}
    feature_cache: dict[Path, VisualFeatures] = {}
    rejected: list[dict[str, str]] = []
    glyphs: list[Glyph] = []

    for row in glyph_rows:
        quality = float(row.get("mask_quality", "1") or 1)
        if quality < min_quality:
            continue

        path = resolve_path(root, row["glyph_path"])
        glyph = Glyph(
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

        try:
            image = load_bgra(path)
            features = visual_features(image)
            if not is_readable_candidate(glyph, image, features):
                rejected.append({
                    "glyph_path": path.as_posix(),
                    "digit": glyph.digit,
                    "reason": "failed readability filter",
                })
                continue
            cache[path] = image
            feature_cache[path] = features
            glyphs.append(glyph)
        except Exception as exc:
            rejected.append({
                "glyph_path": path.as_posix(),
                "digit": glyph.digit,
                "reason": str(exc),
            })

    backgrounds: list[Background] = []
    for row in background_rows:
        path = resolve_path(root, row["background_path"])
        if path.exists():
            backgrounds.append(
                Background(path=path, source_tile=row["source_tile"])
            )

    if not glyphs or not backgrounds:
        raise RuntimeError("No usable assets found")

    return glyphs, backgrounds, cache, feature_cache, rejected


def build_templates(
    glyphs: list[Glyph],
    backgrounds: list[Background],
) -> list[Template]:
    grouped: dict[str, dict[int, Glyph]] = defaultdict(dict)
    for glyph in glyphs:
        grouped[glyph.source_tile][glyph.position] = glyph

    background_by_source = {
        background.source_tile: background
        for background in backgrounds
    }

    templates: list[Template] = []
    for source, slots in grouped.items():
        if set(slots) != {1, 2, 3}:
            continue
        background = background_by_source.get(source)
        if background is None:
            continue
        templates.append(
            Template(
                source_tile=source,
                background=background,
                slots=(slots[1], slots[2], slots[3]),
            )
        )

    if not templates:
        raise RuntimeError("No complete readable templates found")
    return templates


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

    missing = [digit for digit in "0123456789" if not by_digit[digit]]
    if missing:
        raise RuntimeError(f"Readability filter removed all: {missing}")

    return dict(by_digit_position), dict(by_digit)


def candidate_distance(
    candidate: Glyph,
    slot: Glyph,
    features: dict[Path, VisualFeatures],
) -> float:
    c = features[candidate.path]
    s = features[slot.path]

    height_ratio = candidate.height / max(1, slot.height)
    width_ratio = candidate.width / max(1, slot.width)

    return (
        2.3 * abs(c.aspect - s.aspect)
        + 1.0 * abs(c.density - s.density)
        + 0.30 * abs(c.stroke - s.stroke)
        + 0.75 * abs(height_ratio - 1.0)
        + 0.35 * abs(width_ratio - 1.0)
        + 0.20 * abs(c.component_count - s.component_count)
    )


def choose_candidate(
    digit: str,
    position: int,
    template_source: str,
    slot: Glyph,
    by_digit_position: dict[tuple[str, int], list[Glyph]],
    by_digit: dict[str, list[Glyph]],
    features: dict[Path, VisualFeatures],
    rng: random.Random,
    top_k: int,
) -> Glyph:
    pool = by_digit_position.get((digit, position), []) or by_digit[digit]
    different_source = [
        glyph for glyph in pool
        if glyph.source_tile != template_source
    ]
    if different_source:
        pool = different_source

    ranked = sorted(
        pool,
        key=lambda glyph: candidate_distance(glyph, slot, features),
    )
    selection = ranked[:max(1, min(top_k, len(ranked)))]
    return rng.choice(selection)


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


def recolor_to_template(
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
    output[:, :, :3] = cv2.cvtColor(
        source_lab,
        cv2.COLOR_LAB2BGR,
    )
    return output


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


def resize_for_slot(
    image: np.ndarray,
    slot: Glyph,
    rng: random.Random,
    scale_jitter: float,
) -> np.ndarray:
    image = trim(image)
    h, w = image.shape[:2]

    target_height = max(
        10,
        int(round(
            slot.height
            * rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        )),
    )
    scale = target_height / max(1, h)
    target_width = max(2, int(round(w * scale)))

    max_width = max(10, int(round(slot.width * 1.55)))
    if target_width > max_width:
        scale *= max_width / target_width
        target_width = max_width
        target_height = max(10, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )


def place_readably(
    rendered: list[np.ndarray],
    template: Template,
    canvas_shape: tuple[int, int],
    rng: random.Random,
    x_jitter: float,
    y_jitter: float,
) -> list[tuple[int, int]]:
    canvas_h, canvas_w = canvas_shape
    positions: list[list[int]] = []

    for image, slot in zip(rendered, template.slots):
        h, w = image.shape[:2]
        center = slot.x + slot.width / 2
        baseline = slot.y + slot.height
        positions.append([
            int(round(center - w / 2 + rng.uniform(-x_jitter, x_jitter))),
            int(round(baseline - h + rng.uniform(-y_jitter, y_jitter))),
        ])

    # Minimum readable gap. Never allow the severe overlap seen in v3.
    for index in (1, 2):
        previous_right = (
            positions[index - 1][0]
            + rendered[index - 1].shape[1]
        )
        original_gap = (
            template.slots[index].x
            - (
                template.slots[index - 1].x
                + template.slots[index - 1].width
            )
        )
        safe_gap = max(1, min(7, original_gap))
        positions[index][0] = max(
            positions[index][0],
            previous_right + safe_gap,
        )

    left = min(item[0] for item in positions)
    right = max(
        item[0] + image.shape[1]
        for item, image in zip(positions, rendered)
    )

    if left < 2:
        shift = 2 - left
        for item in positions:
            item[0] += shift

    if right > canvas_w - 2:
        shift = right - (canvas_w - 2)
        for item in positions:
            item[0] -= shift

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
        rgb * alpha
        + roi.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def generate_one(
    label: str,
    template: Template,
    pools_by_position: dict[tuple[str, int], list[Glyph]],
    pools_by_digit: dict[str, list[Glyph]],
    cache: dict[Path, np.ndarray],
    features: dict[Path, VisualFeatures],
    rng: random.Random,
    top_k: int,
    scale_jitter: float,
    x_jitter: float,
    y_jitter: float,
    recolor_strength: float,
) -> tuple[np.ndarray, list[Glyph], list[tuple[int, int]]]:
    canvas = cv2.imread(
        str(template.background.path),
        cv2.IMREAD_COLOR,
    )
    if canvas is None:
        raise ValueError(template.background.path)

    chosen: list[Glyph] = []
    rendered: list[np.ndarray] = []

    for position, (digit, slot) in enumerate(
        zip(label, template.slots),
        start=1,
    ):
        candidate = choose_candidate(
            digit=digit,
            position=position,
            template_source=template.source_tile,
            slot=slot,
            by_digit_position=pools_by_position,
            by_digit=pools_by_digit,
            features=features,
            rng=rng,
            top_k=top_k,
        )

        image = recolor_to_template(
            cache[candidate.path],
            cache[slot.path],
            recolor_strength,
        )
        image = resize_for_slot(
            image,
            slot,
            rng,
            scale_jitter,
        )
        chosen.append(candidate)
        rendered.append(image)

    positions = place_readably(
        rendered,
        template,
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
    output: Path,
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

    cv2.imwrite(str(output), sheet)


def write_rejected(
    rejected: list[dict[str, str]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["glyph_path", "digit", "reason"],
        )
        writer.writeheader()
        writer.writerows(rejected)


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    glyphs, backgrounds, cache, features, rejected = load_assets(
        args.assets.resolve(),
        args.min_quality,
    )
    templates = build_templates(glyphs, backgrounds)
    by_position, by_digit = build_pools(glyphs)

    records: list[dict[str, object]] = []
    paths: list[Path] = []
    labels: list[str] = []
    digit_counts: Counter[str] = Counter()
    glyph_use: Counter[str] = Counter()
    template_use: Counter[str] = Counter()

    for index in range(args.count):
        label = "".join(rng.choice("0123456789") for _ in range(3))
        template = rng.choice(templates)

        image, chosen, positions = generate_one(
            label=label,
            template=template,
            pools_by_position=by_position,
            pools_by_digit=by_digit,
            cache=cache,
            features=features,
            rng=rng,
            top_k=args.top_k,
            scale_jitter=args.scale_jitter,
            x_jitter=args.x_jitter,
            y_jitter=args.y_jitter,
            recolor_strength=args.recolor_strength,
        )

        filename = f"{index:07d}_{label}.png"
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
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    write_rejected(rejected, output / "rejected_glyphs.csv")
    make_preview(
        paths,
        labels,
        output / "preview.png",
        args.preview_count,
    )

    summary = {
        "count": args.count,
        "seed": args.seed,
        "input_glyphs": len(glyphs) + len(rejected),
        "accepted_glyphs": len(glyphs),
        "rejected_glyphs": len(rejected),
        "template_count": len(templates),
        "top_k": args.top_k,
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
    }

    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))
    print(f"Preview: {output / 'preview.png'}")
    print(f"Rejected: {output / 'rejected_glyphs.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Readability-first real-glyph compositor."
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("data/real_assets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/composed_preview_v4"),
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=100)
    parser.add_argument("--min-quality", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--scale-jitter", type=float, default=0.015)
    parser.add_argument("--x-jitter", type=float, default=0.75)
    parser.add_argument("--y-jitter", type=float, default=0.75)
    parser.add_argument("--recolor-strength", type=float, default=0.88)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if not args.assets.exists():
        parser.error(f"Assets directory does not exist: {args.assets}")
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    return args


if __name__ == "__main__":
    build(parse_args())
