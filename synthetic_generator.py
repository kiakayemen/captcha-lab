#!/usr/bin/env python3
"""
Synthetic CAPTCHA generator using authentic background texture patches.

It scans real 200x200 CAPTCHA tiles, automatically extracts the cleanest corner
patches, and builds synthetic backgrounds by mirror-tiling those real patches.
Only the digits are rendered synthetically.

Examples:
    # Quick test
    python synthetic_generator.py \
      --real-tiles extracted \
      --output data/synthetic_test \
      --count 100 \
      --seed 42

    # Full dataset
    python synthetic_generator.py \
      --real-tiles extracted \
      --output data/synthetic \
      --count 20000 \
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


TILE_SIZE = 200
PATCH_SIZE = 72

TEXT_COLORS = [
    (42, 40, 132), (53, 64, 155), (25, 82, 121),
    (0, 132, 164), (33, 180, 155), (64, 205, 176),
    (27, 139, 93), (74, 205, 45), (134, 213, 36),
    (211, 215, 26), (239, 212, 25), (194, 166, 34),
    (145, 101, 29), (118, 63, 21), (113, 25, 34),
    (167, 24, 28), (215, 15, 102), (194, 29, 177),
    (126, 42, 197), (86, 42, 170), (92, 96, 84),
    (129, 125, 102), (178, 170, 148), (220, 210, 196),
]


@dataclass(frozen=True)
class BackgroundPatch:
    image: np.ndarray
    source: str
    corner: str


@dataclass(frozen=True)
class SampleMetadata:
    path: str
    label: str
    split: str
    seed: int
    background_source: str
    background_corner: str
    font_name: str
    font_size: int
    text_color: str
    underline_mode: str


def rgb_string(rgb: tuple[int, int, int]) -> str:
    return ",".join(map(str, rgb))


def discover_real_tiles(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(root.rglob(pattern))

    tiles: list[Path] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                if image.size == (TILE_SIZE, TILE_SIZE):
                    tiles.append(path)
        except (OSError, ValueError):
            continue

    return sorted(set(tiles))


def patch_score(patch: np.ndarray) -> float:
    """
    Lower score means a patch is more likely to be clean background.

    Text creates strong edges and localized color variation, while the woven
    background has many small but relatively uniform repeating edges.
    """
    arr = patch.astype(np.float32)
    gray = (
        0.299 * arr[..., 0]
        + 0.587 * arr[..., 1]
        + 0.114 * arr[..., 2]
    )

    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()

    # Penalize unusually saturated or spatially uneven regions.
    channel_spread = (arr.max(axis=2) - arr.min(axis=2))
    high_saturation = np.percentile(channel_spread, 95)
    block_means = []
    for y in range(0, PATCH_SIZE, PATCH_SIZE // 3):
        for x in range(0, PATCH_SIZE, PATCH_SIZE // 3):
            block = gray[
                y:min(y + PATCH_SIZE // 3, PATCH_SIZE),
                x:min(x + PATCH_SIZE // 3, PATCH_SIZE),
            ]
            if block.size:
                block_means.append(float(block.mean()))
    unevenness = float(np.std(block_means))

    return gx + gy + 0.12 * high_saturation + 0.35 * unevenness


def extract_clean_patch(path: Path) -> BackgroundPatch:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)

    p = PATCH_SIZE
    margin = 4
    candidates = {
        "top_left": arr[margin:margin+p, margin:margin+p],
        "top_right": arr[margin:margin+p, -margin-p:-margin],
        "bottom_left": arr[-margin-p:-margin, margin:margin+p],
        "bottom_right": arr[-margin-p:-margin, -margin-p:-margin],
    }

    corner, patch = min(candidates.items(), key=lambda item: patch_score(item[1]))
    return BackgroundPatch(
        image=np.ascontiguousarray(patch),
        source=path.name,
        corner=corner,
    )


def build_background_library(real_tiles_root: Path) -> list[BackgroundPatch]:
    tiles = discover_real_tiles(real_tiles_root)
    if not tiles:
        raise RuntimeError(
            f"No {TILE_SIZE}x{TILE_SIZE} tile images found under {real_tiles_root}"
        )

    patches = [extract_clean_patch(path) for path in tiles]
    print(f"Loaded {len(tiles):,} real tiles.")
    print(f"Extracted {len(patches):,} authentic background patches.")
    return patches


def mirrored_texture(
    patch: np.ndarray,
    size: tuple[int, int],
    rng: random.Random,
) -> Image.Image:
    """
    Mirror-tile the real texture patch. Alternating flips avoid hard repeated
    edges while preserving the source weave instead of inventing a new one.
    """
    height, width = size[1], size[0]
    ph, pw = patch.shape[:2]

    rows = height // ph + 4
    cols = width // pw + 4
    canvas = np.empty((rows * ph, cols * pw, 3), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            tile = patch
            if col % 2:
                tile = tile[:, ::-1]
            if row % 2:
                tile = tile[::-1, :]
            canvas[row*ph:(row+1)*ph, col*pw:(col+1)*pw] = tile

    max_x = canvas.shape[1] - width
    max_y = canvas.shape[0] - height
    x = rng.randint(0, max_x)
    y = rng.randint(0, max_y)
    crop = canvas[y:y+height, x:x+width].copy()

    image = Image.fromarray(crop, "RGB")

    # Tiny real-world tint/brightness variation; texture remains authentic.
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.97, 1.03))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.95, 1.05))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.97, 1.04))
    return image


def discover_fonts(extra_dirs: Sequence[Path] = ()) -> list[Path]:
    roots = [
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("C:/Windows/Fonts"),
        *extra_dirs,
    ]

    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.ttf", "*.otf"):
            paths.extend(root.rglob(pattern))

    blocked = (
        "emoji", "symbol", "dingbat", "icon", "wingding", "webding",
        "arabic", "hebrew", "devanagari", "telugu", "thai", "khmer",
        "cjk", "japanese", "korean", "chinese",
    )
    paths = [
        path for path in paths
        if not any(token in path.name.lower() for token in blocked)
    ]

    preferred = (
        "arial", "helvetica", "roboto", "inter", "lato",
        "dejavu", "liberation", "ubuntu", "noto", "free",
        "condensed", "narrow", "sans",
    )
    paths.sort(
        key=lambda path: (
            not any(token in path.name.lower() for token in preferred),
            path.name.lower(),
        )
    )
    return paths


def choose_text_color(background: Image.Image, rng: random.Random) -> tuple[int, int, int]:
    bg = np.asarray(background.resize((1, 1), Image.Resampling.BOX))[0, 0].astype(float)

    # Include real low-contrast cases, but keep most samples readable.
    if rng.random() < 0.18:
        target = np.array(rng.choice(TEXT_COLORS), dtype=float)
        color = 0.50 * bg + 0.50 * target
        return tuple(np.clip(color, 0, 255).astype(int))

    # Usually choose a color with useful luminance contrast.
    candidates = TEXT_COLORS[:]
    rng.shuffle(candidates)
    bg_luma = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]

    for color in candidates:
        fg = np.array(color, dtype=float)
        fg_luma = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
        if abs(bg_luma - fg_luma) >= rng.uniform(35, 75):
            return color

    return rng.choice(TEXT_COLORS)


def fit_font(
    font_path: Path,
    text: str,
    max_width: int,
    target_height: int,
    start_size: int,
) -> tuple[ImageFont.FreeTypeFont, int]:
    size = start_size
    while size >= 48:
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= target_height:
            return font, size
        size -= 2

    return ImageFont.truetype(str(font_path), size=48), 48


def render_text(
    label: str,
    font_path: Path,
    color: tuple[int, int, int],
    rng: random.Random,
) -> tuple[Image.Image, int, str]:
    scale = 3
    size = TILE_SIZE * scale
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Derived from the report: most boxes are around 0.36-0.69 wide and
    # 0.34-0.59 high, with a median around 0.54 x 0.49.
    max_width = int(size * rng.uniform(0.48, 0.68))
    target_height = int(size * rng.uniform(0.40, 0.58))
    start_size = int(size * rng.uniform(0.60, 0.82))

    font, actual_size = fit_font(
        font_path, label, max_width, target_height, start_size
    )

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Stable placement with only small real-world shifts.
    center_x = size * rng.uniform(0.47, 0.53)
    center_y = size * rng.uniform(0.45, 0.52)
    x = int(center_x - text_w / 2 - bbox[0])
    y = int(center_y - text_h / 2 - bbox[1])

    alpha = rng.randint(215, 255)
    draw.text((x, y), label, font=font, fill=(*color, alpha))

    underline_mode = rng.choices(
        ["none", "below", "overlap"],
        weights=[0.76, 0.16, 0.08],
        k=1,
    )[0]

    if underline_mode != "none":
        left = x + int(text_w * rng.uniform(-0.01, 0.04))
        right = x + text_w + int(text_w * rng.uniform(-0.04, 0.02))
        thickness = rng.randint(3, 6) * scale

        if underline_mode == "below":
            line_y = y + text_h + int(size * rng.uniform(0.005, 0.025))
        else:
            line_y = y + int(text_h * rng.uniform(0.73, 0.86))

        draw.line(
            (left, line_y, right, line_y),
            fill=(*color, rng.randint(210, 255)),
            width=thickness,
        )

    return (
        layer.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS),
        actual_size // scale,
        underline_mode,
    )


def generate_sample(
    label: str,
    background_patch: BackgroundPatch,
    font_path: Path,
    seed: int,
) -> tuple[Image.Image, dict]:
    rng = random.Random(seed)

    background = mirrored_texture(
        background_patch.image,
        (TILE_SIZE, TILE_SIZE),
        rng,
    )
    color = choose_text_color(background, rng)
    text, font_size, underline_mode = render_text(
        label, font_path, color, rng
    )

    image = Image.alpha_composite(
        background.convert("RGBA"),
        text,
    ).convert("RGB")

    # The real images mostly show browser-style antialiasing, not blur.
    if rng.random() < 0.12:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.18)))

    return image, {
        "font_name": font_path.name,
        "font_size": font_size,
        "text_color": rgb_string(color),
        "underline_mode": underline_mode,
    }


def write_manifest(rows: Iterable[SampleMetadata], path: Path) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def generate_dataset(args: argparse.Namespace) -> None:
    backgrounds = build_background_library(args.real_tiles)

    fonts = discover_fonts([Path(path).expanduser() for path in args.font_dir])
    if not fonts:
        raise RuntimeError("No usable fonts found.")

    # Restrict variation instead of using every installed font.
    fonts = fonts[: args.max_fonts]
    print(f"Using {len(fonts)} fonts.")

    output = args.output.resolve()
    train_dir = output / "images" / "train"
    val_dir = output / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    shuffled_indices = list(range(args.count))
    rng.shuffle(shuffled_indices)
    val_count = round(args.count * args.val_ratio)
    val_indices = set(shuffled_indices[:val_count])

    labels = [f"{index % 1000:03d}" for index in range(args.count)]
    rng.shuffle(labels)

    rows: list[SampleMetadata] = []

    for index, label in enumerate(labels):
        split = "val" if index in val_indices else "train"
        directory = val_dir if split == "val" else train_dir

        seed = rng.randrange(2**31)
        sample_rng = random.Random(seed)
        background = sample_rng.choice(backgrounds)
        font = sample_rng.choice(fonts)

        image, metadata = generate_sample(
            label=label,
            background_patch=background,
            font_path=font,
            seed=seed,
        )

        filename = f"{index:07d}_{label}.png"
        path = directory / filename
        image.save(path, "PNG", optimize=True)

        rows.append(
            SampleMetadata(
                path=path.relative_to(output).as_posix(),
                label=label,
                split=split,
                seed=seed,
                background_source=background.source,
                background_corner=background.corner,
                font_name=metadata["font_name"],
                font_size=metadata["font_size"],
                text_color=metadata["text_color"],
                underline_mode=metadata["underline_mode"],
            )
        )

        if (index + 1) % args.log_every == 0 or index + 1 == args.count:
            print(f"Generated {index + 1:,}/{args.count:,}")

    write_manifest(rows, output / "manifest.csv")
    print(f"Done: {output}")
    print(f"Train: {args.count - val_count:,}")
    print(f"Validation: {val_count:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CAPTCHA tiles using authentic backgrounds."
    )
    parser.add_argument(
        "--real-tiles",
        type=Path,
        required=True,
        help="Directory containing real 200x200 extracted CAPTCHA tiles.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-dir", action="append", default=[])
    parser.add_argument(
        "--max-fonts",
        type=int,
        default=12,
        help="Maximum number of preferred fonts to use.",
    )
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    if not args.real_tiles.exists():
        parser.error(f"--real-tiles does not exist: {args.real_tiles}")
    return args


if __name__ == "__main__":
    generate_dataset(parse_args())
