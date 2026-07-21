#!/usr/bin/env python3
"""
Build a style-aware bank of authentic CAPTCHA glyphs.

Key design:
- Never average glyphs.
- Never synthesize glyph shapes.
- Cluster complete CAPTCHA templates by measurable visual style.
- For each style cluster, digit, and position, select authentic medoid glyphs.
- Detect underline contamination and record it separately.
- Save manifests consumed by synthetic_compositor_v5.py.

Run:
python build_style_bank.py \
  --assets data/real_assets \
  --output data/style_bank \
  --clusters 8
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
class GlyphFeatures:
    aspect: float
    density: float
    stroke_width: float
    edge_softness: float
    mean_l: float
    mean_a: float
    mean_b: float
    underline_score: float
    component_count: int


@dataclass(frozen=True)
class TemplateFeatures:
    mean_height: float
    height_std: float
    mean_width: float
    width_std: float
    mean_stroke: float
    stroke_std: float
    mean_edge_softness: float
    mean_density: float
    spacing_1: float
    spacing_2: float
    baseline_std: float
    underline_fraction: float
    mean_l: float
    mean_chroma: float


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
    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def underline_mask(alpha: np.ndarray) -> np.ndarray:
    binary = (alpha > 16).astype(np.uint8) * 255
    h, w = binary.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(8, int(round(w * 0.42))), 1),
    )
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    lower = np.zeros_like(lines)
    lower[int(h * 0.52):, :] = 255
    return cv2.bitwise_and(lines, lower)


def glyph_features(image: np.ndarray) -> GlyphFeatures:
    alpha = image[:, :, 3]
    bounds = alpha_bounds(image)
    if bounds is None:
        return GlyphFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0)

    x0, y0, x1, y1 = bounds
    crop_alpha = alpha[y0:y1, x0:x1]
    binary = (crop_alpha > 32).astype(np.uint8)
    h, w = binary.shape

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_values = distance[binary > 0] * 2.0
    stroke = float(stroke_values.mean()) if stroke_values.size else 0.0

    fractional = crop_alpha[(crop_alpha > 0) & (crop_alpha < 255)]
    edge_softness = float(
        len(fractional) / max(1, np.count_nonzero(crop_alpha))
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    substantial = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        ch = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area >= 5 and ch >= 3:
            substantial += 1

    valid = crop_alpha > 16
    bgr = image[y0:y1, x0:x1, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    weights = crop_alpha[valid].astype(np.float32) / 255.0
    mean_lab = (
        np.average(lab[valid], axis=0, weights=weights)
        if np.any(valid)
        else np.array([128, 128, 128], dtype=np.float32)
    )

    line = underline_mask(alpha)
    underline_score = float(
        np.count_nonzero(line) / max(1, np.count_nonzero(alpha))
    )

    return GlyphFeatures(
        aspect=float(w / max(1, h)),
        density=float(binary.mean()),
        stroke_width=stroke,
        edge_softness=edge_softness,
        mean_l=float(mean_lab[0]),
        mean_a=float(mean_lab[1]),
        mean_b=float(mean_lab[2]),
        underline_score=underline_score,
        component_count=substantial,
    )


def valid_glyph(
    glyph: Glyph,
    features: GlyphFeatures,
) -> tuple[bool, str]:
    if glyph.quality < 0.35:
        return False, "low mask quality"
    if features.stroke_width <= 0.25:
        return False, "invalid stroke width"
    if not 0.02 <= features.density <= 0.75:
        return False, "implausible density"
    if not 0.08 <= features.aspect <= 1.65:
        return False, "implausible aspect ratio"
    if not 1 <= features.component_count <= 5:
        return False, "implausible component count"
    if features.underline_score > 0.42:
        return False, "underline dominates glyph"
    return True, ""


def template_features(
    slots: tuple[Glyph, Glyph, Glyph],
    glyph_feature_map: dict[Path, GlyphFeatures],
) -> TemplateFeatures:
    heights = np.array([slot.height for slot in slots], dtype=np.float32)
    widths = np.array([slot.width for slot in slots], dtype=np.float32)
    baselines = np.array(
        [slot.y + slot.height for slot in slots],
        dtype=np.float32,
    )

    styles = [glyph_feature_map[slot.path] for slot in slots]
    strokes = np.array([style.stroke_width for style in styles])
    softness = np.array([style.edge_softness for style in styles])
    density = np.array([style.density for style in styles])
    underline = np.array([style.underline_score for style in styles])
    lightness = np.array([style.mean_l for style in styles])
    chroma = np.array([
        np.hypot(style.mean_a - 128, style.mean_b - 128)
        for style in styles
    ])

    spacing_1 = slots[1].x - (slots[0].x + slots[0].width)
    spacing_2 = slots[2].x - (slots[1].x + slots[1].width)

    return TemplateFeatures(
        mean_height=float(heights.mean()),
        height_std=float(heights.std()),
        mean_width=float(widths.mean()),
        width_std=float(widths.std()),
        mean_stroke=float(strokes.mean()),
        stroke_std=float(strokes.std()),
        mean_edge_softness=float(softness.mean()),
        mean_density=float(density.mean()),
        spacing_1=float(spacing_1),
        spacing_2=float(spacing_2),
        baseline_std=float(baselines.std()),
        underline_fraction=float(np.mean(underline > 0.03)),
        mean_l=float(lightness.mean()),
        mean_chroma=float(chroma.mean()),
    )


def vectorize_template(features: TemplateFeatures) -> np.ndarray:
    return np.array([
        features.mean_height,
        features.height_std,
        features.mean_width,
        features.width_std,
        features.mean_stroke,
        features.stroke_std,
        features.mean_edge_softness * 10.0,
        features.mean_density * 10.0,
        features.spacing_1,
        features.spacing_2,
        features.baseline_std,
        features.underline_fraction * 5.0,
        features.mean_l / 10.0,
        features.mean_chroma / 10.0,
    ], dtype=np.float32)


def glyph_distance(
    a: GlyphFeatures,
    b: GlyphFeatures,
) -> float:
    return (
        2.2 * abs(a.aspect - b.aspect)
        + 1.5 * abs(a.density - b.density)
        + 0.35 * abs(a.stroke_width - b.stroke_width)
        + 1.0 * abs(a.edge_softness - b.edge_softness)
        + 0.025 * abs(a.mean_l - b.mean_l)
        + 0.015 * abs(a.mean_a - b.mean_a)
        + 0.015 * abs(a.mean_b - b.mean_b)
        + 0.45 * abs(a.component_count - b.component_count)
    )


def select_medoids(
    glyphs: list[Glyph],
    features: dict[Path, GlyphFeatures],
    max_medoids: int,
) -> list[Glyph]:
    if len(glyphs) <= max_medoids:
        return glyphs

    matrix = np.zeros((len(glyphs), len(glyphs)), dtype=np.float32)
    for i, left in enumerate(glyphs):
        for j in range(i + 1, len(glyphs)):
            right = glyphs[j]
            distance = glyph_distance(
                features[left.path],
                features[right.path],
            )
            matrix[i, j] = distance
            matrix[j, i] = distance

    total_distance = matrix.sum(axis=1)
    first = int(np.argmin(total_distance))
    chosen = [first]

    while len(chosen) < max_medoids:
        nearest = np.min(matrix[:, chosen], axis=1)
        nearest[chosen] = -1
        next_index = int(np.argmax(nearest))
        chosen.append(next_index)

    return [glyphs[index] for index in chosen]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    assets = args.assets.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    glyph_rows = read_csv(assets / "manifests" / "glyphs.csv")
    background_rows = read_csv(
        assets / "manifests" / "backgrounds.csv"
    )

    backgrounds = {
        row["source_tile"]: resolve_path(assets, row["background_path"])
        for row in background_rows
    }

    all_glyphs: list[Glyph] = []
    feature_map: dict[Path, GlyphFeatures] = {}
    rejected_rows: list[dict[str, object]] = []

    for row in glyph_rows:
        glyph = Glyph(
            path=resolve_path(assets, row["glyph_path"]),
            digit=row["digit"],
            position=int(row["position"]),
            x=int(row["x"]),
            y=int(row["y"]),
            width=int(row["width"]),
            height=int(row["height"]),
            source_tile=row["source_tile"],
            quality=float(row.get("mask_quality", "1") or 1),
        )

        try:
            image = load_bgra(glyph.path)
            features = glyph_features(image)
            okay, reason = valid_glyph(glyph, features)
            if not okay:
                rejected_rows.append({
                    "glyph_path": glyph.path.as_posix(),
                    "digit": glyph.digit,
                    "source_tile": glyph.source_tile,
                    "reason": reason,
                })
                continue
        except Exception as exc:
            rejected_rows.append({
                "glyph_path": glyph.path.as_posix(),
                "digit": glyph.digit,
                "source_tile": glyph.source_tile,
                "reason": str(exc),
            })
            continue

        all_glyphs.append(glyph)
        feature_map[glyph.path] = features

    grouped: dict[str, dict[int, Glyph]] = defaultdict(dict)
    for glyph in all_glyphs:
        grouped[glyph.source_tile][glyph.position] = glyph

    template_sources = [
        source
        for source, slots in grouped.items()
        if set(slots) == {1, 2, 3} and source in backgrounds
    ]

    if len(template_sources) < args.clusters:
        raise RuntimeError("Too few complete templates for requested clusters")

    template_feature_map: dict[str, TemplateFeatures] = {}
    vectors = []
    for source in template_sources:
        slots = (
            grouped[source][1],
            grouped[source][2],
            grouped[source][3],
        )
        features = template_features(slots, feature_map)
        template_feature_map[source] = features
        vectors.append(vectorize_template(features))

    scaled = StandardScaler().fit_transform(np.stack(vectors))
    model = KMeans(
        n_clusters=args.clusters,
        random_state=args.seed,
        n_init=30,
    )
    cluster_labels = model.fit_predict(scaled)

    source_to_cluster = {
        source: int(cluster)
        for source, cluster in zip(template_sources, cluster_labels)
    }

    templates_manifest: list[dict[str, object]] = []
    for source in template_sources:
        slots = grouped[source]
        tf = template_feature_map[source]
        templates_manifest.append({
            "source_tile": source,
            "style_cluster": source_to_cluster[source],
            "background_path": backgrounds[source].as_posix(),
            "slot_1_path": slots[1].path.as_posix(),
            "slot_2_path": slots[2].path.as_posix(),
            "slot_3_path": slots[3].path.as_posix(),
            "x_1": slots[1].x,
            "y_1": slots[1].y,
            "width_1": slots[1].width,
            "height_1": slots[1].height,
            "x_2": slots[2].x,
            "y_2": slots[2].y,
            "width_2": slots[2].width,
            "height_2": slots[2].height,
            "x_3": slots[3].x,
            "y_3": slots[3].y,
            "width_3": slots[3].width,
            "height_3": slots[3].height,
            **asdict(tf),
        })

    medoid_manifest: list[dict[str, object]] = []
    by_cluster_digit_position: dict[
        tuple[int, str, int],
        list[Glyph],
    ] = defaultdict(list)

    for glyph in all_glyphs:
        cluster = source_to_cluster.get(glyph.source_tile)
        if cluster is None:
            continue
        by_cluster_digit_position[
            (cluster, glyph.digit, glyph.position)
        ].append(glyph)

    for key, candidates in sorted(by_cluster_digit_position.items()):
        cluster, digit, position = key
        medoids = select_medoids(
            candidates,
            feature_map,
            args.medoids_per_group,
        )
        for rank, glyph in enumerate(medoids):
            gf = feature_map[glyph.path]
            medoid_manifest.append({
                "style_cluster": cluster,
                "digit": digit,
                "position": position,
                "rank": rank,
                "glyph_path": glyph.path.as_posix(),
                "source_tile": glyph.source_tile,
                **asdict(gf),
            })

    write_csv(templates_manifest, output / "templates.csv")
    write_csv(medoid_manifest, output / "medoids.csv")
    write_csv(rejected_rows, output / "rejected_glyphs.csv")

    counts = Counter(source_to_cluster.values())
    summary = {
        "input_glyphs": len(glyph_rows),
        "accepted_glyphs": len(all_glyphs),
        "rejected_glyphs": len(rejected_rows),
        "complete_templates": len(template_sources),
        "clusters": args.clusters,
        "cluster_counts": {
            str(cluster): counts[cluster]
            for cluster in sorted(counts)
        },
        "medoid_count": len(medoid_manifest),
        "medoids_per_group": args.medoids_per_group,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Output: {output}")


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
        default=Path("data/style_bank"),
    )
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--medoids-per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.assets.exists():
        parser.error(f"Missing assets: {args.assets}")
    if args.clusters < 2:
        parser.error("--clusters must be at least 2")
    if args.medoids_per_group < 1:
        parser.error("--medoids-per-group must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
