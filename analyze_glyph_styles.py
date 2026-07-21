#!/usr/bin/env python3
"""
Analyze whether the CAPTCHA glyph bank contains one dominant font/style
or several distinct glyph families.

Expected input
--------------
data/real_assets/
├── glyph_bank/
└── manifests/
    └── glyphs.csv

Run
---
python analyze_glyph_styles.py \
    --assets data/real_assets \
    --output data/glyph_style_analysis \
    --clusters 4

Outputs
-------
data/glyph_style_analysis/
├── summary.json
├── glyph_features.csv
├── digit_0_clusters.png
├── digit_1_clusters.png
├── ...
├── digit_9_clusters.png
└── all_digit_prototypes.png

Dependencies
------------
numpy
opencv-python
scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


NORMALIZED_SIZE = 64


@dataclass(frozen=True)
class GlyphRow:
    path: Path
    digit: str
    position: int
    source_tile: str
    quality: float


@dataclass(frozen=True)
class FeatureRecord:
    glyph_path: str
    digit: str
    position: int
    source_tile: str
    quality: float
    cluster: int
    width_ratio: float
    foreground_fraction: float
    stroke_mean: float
    stroke_std: float
    centroid_x: float
    centroid_y: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_glyph_rows(
    assets_root: Path,
    min_quality: float,
) -> list[GlyphRow]:
    manifest = assets_root / "manifests" / "glyphs.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    rows: list[GlyphRow] = []
    for row in read_csv(manifest):
        quality = float(row.get("mask_quality", "1") or 1)
        if quality < min_quality:
            continue

        path = resolve_path(assets_root, row["glyph_path"])
        if not path.exists():
            raise FileNotFoundError(path)

        rows.append(
            GlyphRow(
                path=path,
                digit=row["digit"],
                position=int(row["position"]),
                source_tile=row["source_tile"],
                quality=quality,
            )
        )

    if not rows:
        raise RuntimeError("No glyphs loaded")
    return rows


def load_alpha(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected BGRA glyph: {path}")
    return image[:, :, 3]


def remove_underlines(mask: np.ndarray) -> np.ndarray:
    """
    Remove long horizontal components so clustering focuses on digit shape.
    """
    binary = (mask > 16).astype(np.uint8) * 255
    h, w = binary.shape
    kernel_width = max(8, round(w * 0.38))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1),
    )
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.subtract(binary, lines)

    # Keep the largest plausible digit components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    if count <= 1:
        return binary

    components = []
    for index in range(1, count):
        x, y, cw, ch, area = stats[index]
        if area >= 6 and ch >= max(4, round(h * 0.08)):
            components.append((area, index))

    if not components:
        return binary

    # Keep all substantial components because digits such as 4 can split.
    largest = max(area for area, _ in components)
    output = np.zeros_like(binary)
    for area, index in components:
        if area >= max(5, largest * 0.08):
            output[labels == index] = 255
    return output


def tight_crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    return mask[
        int(ys.min()):int(ys.max() + 1),
        int(xs.min()):int(xs.max() + 1),
    ]


def normalize_mask(mask: np.ndarray, size: int = NORMALIZED_SIZE) -> np.ndarray:
    mask = remove_underlines(mask)
    mask = tight_crop(mask)

    h, w = mask.shape
    if h < 2 or w < 1:
        return np.zeros((size, size), dtype=np.uint8)

    target_h = size - 8
    scale = target_h / h
    target_w = max(1, min(size - 8, int(round(w * scale))))

    resized = cv2.resize(
        mask,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - target_w) // 2
    y = (size - target_h) // 2
    canvas[y:y + target_h, x:x + target_w] = resized
    return canvas


def mask_features(mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    binary = (mask > 32).astype(np.uint8)
    ys, xs = np.where(binary > 0)

    if len(xs) == 0:
        handcrafted = {
            "width_ratio": 0.0,
            "foreground_fraction": 0.0,
            "stroke_mean": 0.0,
            "stroke_std": 0.0,
            "centroid_x": 0.5,
            "centroid_y": 0.5,
        }
        return mask.flatten().astype(np.float32) / 255.0, handcrafted

    width_ratio = (xs.max() - xs.min() + 1) / NORMALIZED_SIZE
    foreground_fraction = float(binary.mean())
    centroid_x = float(xs.mean() / NORMALIZED_SIZE)
    centroid_y = float(ys.mean() / NORMALIZED_SIZE)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_values = distance[binary > 0] * 2.0
    stroke_mean = float(stroke_values.mean())
    stroke_std = float(stroke_values.std())

    # Downsampled shape pixels dominate clustering; handcrafted metrics add
    # explicit width/stroke information.
    shape = cv2.resize(
        mask,
        (24, 24),
        interpolation=cv2.INTER_AREA,
    ).flatten().astype(np.float32) / 255.0

    handcrafted = {
        "width_ratio": float(width_ratio),
        "foreground_fraction": foreground_fraction,
        "stroke_mean": stroke_mean,
        "stroke_std": stroke_std,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
    }

    extra = np.array(
        [
            width_ratio * 3.0,
            foreground_fraction * 5.0,
            stroke_mean / 8.0,
            stroke_std / 5.0,
            centroid_x,
            centroid_y,
        ],
        dtype=np.float32,
    )
    return np.concatenate([shape, extra]), handcrafted


def choose_cluster_count(
    features: np.ndarray,
    max_clusters: int,
    seed: int,
) -> tuple[int, dict[int, float]]:
    count = len(features)
    if count < 6:
        return 1, {}

    scores: dict[int, float] = {}
    upper = min(max_clusters, count - 1)

    for clusters in range(2, upper + 1):
        model = KMeans(
            n_clusters=clusters,
            random_state=seed,
            n_init=20,
        )
        labels = model.fit_predict(features)
        if len(set(labels)) < 2:
            continue
        scores[clusters] = float(silhouette_score(features, labels))

    if not scores:
        return 1, {}

    best_clusters = max(scores, key=scores.get)
    best_score = scores[best_clusters]

    # Weak separation means one dominant family with rendering variation.
    if best_score < 0.16:
        return 1, scores
    return best_clusters, scores


def representative_indices(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster: int,
    count: int,
) -> list[int]:
    indices = np.where(labels == cluster)[0]
    if len(indices) == 0:
        return []

    distances = np.linalg.norm(
        features[indices] - centers[cluster][None, :],
        axis=1,
    )
    order = indices[np.argsort(distances)]
    return [int(index) for index in order[:count]]


def make_cluster_sheet(
    digit: str,
    masks: list[np.ndarray],
    rows: list[GlyphRow],
    labels: np.ndarray,
    features: np.ndarray,
    centers: np.ndarray,
    output_path: Path,
) -> None:
    clusters = sorted(set(int(value) for value in labels))
    cell = 96
    caption = 24
    examples_per_cluster = 8
    columns = examples_per_cluster
    rows_count = len(clusters)

    sheet = np.full(
        (rows_count * (cell + caption), columns * cell, 3),
        245,
        dtype=np.uint8,
    )

    for output_row, cluster in enumerate(clusters):
        indices = representative_indices(
            features,
            labels,
            centers,
            cluster,
            examples_per_cluster,
        )

        for column, index in enumerate(indices):
            mask = masks[index]
            tile = np.full((cell, cell, 3), 255, dtype=np.uint8)
            display = cv2.resize(
                mask,
                (cell - 12, cell - 12),
                interpolation=cv2.INTER_NEAREST,
            )
            display = 255 - display
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            tile[6:cell - 6, 6:cell - 6] = display

            x = column * cell
            y = output_row * (cell + caption)
            sheet[y:y + cell, x:x + cell] = tile

            cv2.putText(
                sheet,
                f"C{cluster}",
                (x + 4, y + cell + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        sheet,
        f"Digit {digit}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), sheet)


def make_prototype_sheet(
    prototypes: dict[str, list[np.ndarray]],
    output_path: Path,
) -> None:
    cell = 96
    max_clusters = max(len(items) for items in prototypes.values())
    columns = max_clusters
    rows = 10

    sheet = np.full(
        (rows * cell, columns * cell, 3),
        245,
        dtype=np.uint8,
    )

    for digit in "0123456789":
        row = int(digit)
        for column, prototype in enumerate(prototypes[digit]):
            display = np.clip(prototype * 255, 0, 255).astype(np.uint8)
            display = cv2.resize(
                display,
                (cell - 12, cell - 12),
                interpolation=cv2.INTER_NEAREST,
            )
            display = 255 - display
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

            y = row * cell
            x = column * cell
            sheet[y + 6:y + cell - 6, x + 6:x + cell - 6] = display
            cv2.putText(
                sheet,
                f"{digit}/C{column}",
                (x + 4, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_path), sheet)


def write_feature_csv(records: list[FeatureRecord], path: Path) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(records[0]).keys()),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def analyze(args: argparse.Namespace) -> None:
    assets = args.assets.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = load_glyph_rows(assets, args.min_quality)
    by_digit: dict[str, list[GlyphRow]] = defaultdict(list)
    for row in rows:
        by_digit[row.digit].append(row)

    all_records: list[FeatureRecord] = []
    summary: dict[str, object] = {
        "glyph_count": len(rows),
        "min_quality": args.min_quality,
        "digits": {},
    }
    prototypes: dict[str, list[np.ndarray]] = {}

    for digit in "0123456789":
        digit_rows = by_digit[digit]
        masks: list[np.ndarray] = []
        feature_vectors: list[np.ndarray] = []
        handcrafted_rows: list[dict[str, float]] = []

        for row in digit_rows:
            normalized = normalize_mask(load_alpha(row.path))
            features, handcrafted = mask_features(normalized)
            masks.append(normalized)
            feature_vectors.append(features)
            handcrafted_rows.append(handcrafted)

        raw_features = np.stack(feature_vectors)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(raw_features)

        cluster_count, silhouette_scores = choose_cluster_count(
            scaled_features,
            args.clusters,
            args.seed,
        )

        if cluster_count == 1:
            labels = np.zeros(len(digit_rows), dtype=np.int32)
            centers = np.mean(
                scaled_features,
                axis=0,
                keepdims=True,
            )
        else:
            model = KMeans(
                n_clusters=cluster_count,
                random_state=args.seed,
                n_init=30,
            )
            labels = model.fit_predict(scaled_features)
            centers = model.cluster_centers_

        counts = Counter(int(value) for value in labels)
        dominant_fraction = max(counts.values()) / len(labels)

        digit_prototypes: list[np.ndarray] = []
        for cluster in sorted(counts):
            cluster_masks = np.stack(
                [
                    masks[index].astype(np.float32) / 255.0
                    for index, label in enumerate(labels)
                    if int(label) == cluster
                ]
            )
            digit_prototypes.append(cluster_masks.mean(axis=0))
        prototypes[digit] = digit_prototypes

        make_cluster_sheet(
            digit,
            masks,
            digit_rows,
            labels,
            scaled_features,
            centers,
            output / f"digit_{digit}_clusters.png",
        )

        for row, label, handcrafted in zip(
            digit_rows,
            labels,
            handcrafted_rows,
        ):
            all_records.append(
                FeatureRecord(
                    glyph_path=row.path.as_posix(),
                    digit=digit,
                    position=row.position,
                    source_tile=row.source_tile,
                    quality=round(row.quality, 4),
                    cluster=int(label),
                    width_ratio=round(handcrafted["width_ratio"], 5),
                    foreground_fraction=round(
                        handcrafted["foreground_fraction"],
                        5,
                    ),
                    stroke_mean=round(handcrafted["stroke_mean"], 5),
                    stroke_std=round(handcrafted["stroke_std"], 5),
                    centroid_x=round(handcrafted["centroid_x"], 5),
                    centroid_y=round(handcrafted["centroid_y"], 5),
                )
            )

        summary["digits"][digit] = {
            "count": len(digit_rows),
            "selected_clusters": cluster_count,
            "cluster_counts": {
                str(cluster): counts[cluster]
                for cluster in sorted(counts)
            },
            "dominant_cluster_fraction": round(
                dominant_fraction,
                4,
            ),
            "silhouette_scores": {
                str(key): round(value, 4)
                for key, value in silhouette_scores.items()
            },
            "mean_width_ratio": round(
                float(np.mean([
                    item["width_ratio"]
                    for item in handcrafted_rows
                ])),
                4,
            ),
            "mean_stroke_width": round(
                float(np.mean([
                    item["stroke_mean"]
                    for item in handcrafted_rows
                ])),
                4,
            ),
        }

        print(
            f"Digit {digit}: {len(digit_rows)} glyphs, "
            f"{cluster_count} selected cluster(s), "
            f"dominant {dominant_fraction:.1%}"
        )

    make_prototype_sheet(
        prototypes,
        output / "all_digit_prototypes.png",
    )
    write_feature_csv(
        all_records,
        output / "glyph_features.csv",
    )

    selected_counts = [
        summary["digits"][digit]["selected_clusters"]
        for digit in "0123456789"
    ]
    multi_cluster_digits = sum(count > 1 for count in selected_counts)

    if multi_cluster_digits <= 2:
        conclusion = (
            "Strong evidence for one dominant glyph family/font with "
            "rendering, scale, and extraction variation."
        )
    elif multi_cluster_digits <= 5:
        conclusion = (
            "Mixed result: likely one dominant font plus several style or "
            "rendering subfamilies."
        )
    else:
        conclusion = (
            "Evidence for multiple distinct glyph families; a single-font "
            "renderer is unlikely to cover the dataset well."
        )

    summary["overall"] = {
        "digits_with_multiple_clusters": multi_cluster_digits,
        "conclusion": conclusion,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(conclusion)
    print(f"Output: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster authentic CAPTCHA glyphs by shape/style."
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("data/real_assets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/glyph_style_analysis"),
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Maximum clusters tested per digit.",
    )
    parser.add_argument("--min-quality", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.assets.exists():
        parser.error(f"Assets directory does not exist: {args.assets}")
    if args.clusters < 2:
        parser.error("--clusters must be at least 2")
    if not 0 <= args.min_quality <= 1:
        parser.error("--min-quality must be between 0 and 1")

    return args


if __name__ == "__main__":
    analyze(parse_args())
