#!/usr/bin/env python3
"""
Synthetic 3-digit CAPTCHA generator.

Outputs:
    output/
      images/
        train/
        val/
      manifest.csv

Each manifest row contains:
    path,label,split,seed,font_name,text_color,bg_color,rotation,scale,blur,underline

Example:
    python synthetic_generator.py --output data/synthetic --count 20000 --val-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


# Adjust these defaults to match the real tile dimensions.
DEFAULT_WIDTH = 96
DEFAULT_HEIGHT = 48

TEXT_COLORS = [
    (28, 96, 190),   # blue
    (24, 120, 165),
    (37, 137, 92),   # green
    (47, 150, 111),
    (35, 112, 150),
]

BACKGROUND_COLORS = [
    (236, 244, 250),
    (242, 247, 239),
    (245, 241, 248),
    (249, 245, 235),
    (235, 247, 244),
    (242, 242, 248),
]


@dataclass(frozen=True)
class SampleMetadata:
    path: str
    label: str
    split: str
    seed: int
    font_name: str
    text_color: str
    bg_color: str
    rotation: float
    scale: float
    blur: float
    underline: bool


def discover_fonts(extra_dirs: Sequence[Path] = ()) -> list[Path]:
    """Find usable TTF/OTF fonts on macOS, Linux, and Windows."""
    candidates = [
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("C:/Windows/Fonts"),
        *extra_dirs,
    ]

    fonts: list[Path] = []
    for root in candidates:
        if not root.exists():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            fonts.extend(root.rglob(pattern))

    blocked_tokens = {
        "emoji",
        "symbol",
        "dingbat",
        "icons",
        "wingdings",
        "webdings",
    }

    filtered = [
        path
        for path in fonts
        if not any(token in path.name.lower() for token in blocked_tokens)
    ]

    # Prefer ordinary sans/serif fonts before unusual display fonts.
    preference_tokens = (
        "arial",
        "helvetica",
        "verdana",
        "tahoma",
        "times",
        "georgia",
        "dejavu",
        "liberation",
        "noto",
        "roboto",
        "sf",
    )

    filtered.sort(
        key=lambda p: (
            not any(token in p.name.lower() for token in preference_tokens),
            p.name.lower(),
        )
    )
    return filtered


def rgb_to_string(rgb: tuple[int, int, int]) -> str:
    return ",".join(str(x) for x in rgb)


def add_background_texture(
    image: Image.Image,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Image.Image:
    """Add low-contrast paper-like texture, blobs, and sparse lines."""
    arr = np.asarray(image).astype(np.int16)

    # Fine-grained luminance noise.
    sigma = rng.uniform(1.5, 5.5)
    noise = np_rng.normal(0, sigma, size=arr.shape[:2])
    arr = np.clip(arr + noise[..., None], 0, 255).astype(np.uint8)
    textured = Image.fromarray(arr, mode="RGB")

    overlay = Image.new("RGBA", textured.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = textured.size

    # Soft pastel blobs.
    for _ in range(rng.randint(2, 7)):
        x = rng.randint(-width // 4, width)
        y = rng.randint(-height // 3, height)
        rx = rng.randint(max(4, width // 12), max(6, width // 3))
        ry = rng.randint(max(3, height // 10), max(5, height // 2))
        tint = rng.choice(BACKGROUND_COLORS)
        alpha = rng.randint(5, 20)
        draw.ellipse((x, y, x + rx, y + ry), fill=(*tint, alpha))

    # Sparse background lines.
    for _ in range(rng.randint(0, 4)):
        y = rng.randint(0, height - 1)
        shade = rng.choice([(80, 110, 120), (90, 130, 105), (100, 115, 145)])
        alpha = rng.randint(5, 18)
        draw.line(
            (
                rng.randint(-10, width // 4),
                y,
                rng.randint(3 * width // 4, width + 10),
                y + rng.randint(-3, 3),
            ),
            fill=(*shade, alpha),
            width=rng.randint(1, 2),
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))
    return Image.alpha_composite(textured.convert("RGBA"), overlay).convert("RGB")


def fit_font(
    font_path: Path,
    text: str,
    target_height: int,
    max_width: int,
    rng: random.Random,
) -> ImageFont.FreeTypeFont:
    """Pick a font size that fits within the requested box."""
    size = max(12, int(target_height * rng.uniform(0.72, 1.02)))

    while size >= 10:
        try:
            font = ImageFont.truetype(str(font_path), size=size)
        except OSError:
            size -= 1
            continue

        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= target_height:
            return font
        size -= 1

    raise RuntimeError(f"Could not load a usable font from {font_path}")


def render_text_layer(
    text: str,
    font_path: Path,
    canvas_size: tuple[int, int],
    rng: random.Random,
) -> tuple[Image.Image, float, bool]:
    """Render text, optional underline, and return the RGBA layer."""
    width, height = canvas_size
    oversample = 3

    big_width = width * oversample
    big_height = height * oversample
    layer = Image.new("RGBA", (big_width, big_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    target_height = int(big_height * rng.uniform(0.55, 0.78))
    max_width = int(big_width * rng.uniform(0.72, 0.93))
    font = fit_font(
        font_path=font_path,
        text=text,
        target_height=target_height,
        max_width=max_width,
        rng=rng,
    )

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x_center = big_width / 2 + rng.uniform(-0.08, 0.08) * big_width
    y_center = big_height / 2 + rng.uniform(-0.09, 0.09) * big_height

    x = int(x_center - text_width / 2 - bbox[0])
    y = int(y_center - text_height / 2 - bbox[1])

    color = rng.choice(TEXT_COLORS)
    alpha = rng.randint(215, 255)

    stroke_width = rng.choices([0, 1, 2], weights=[0.72, 0.22, 0.06], k=1)[0]
    stroke_fill = tuple(max(0, c - rng.randint(5, 25)) for c in color) + (alpha,)

    draw.text(
        (x, y),
        text,
        font=font,
        fill=(*color, alpha),
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    underline = rng.random() < 0.34
    if underline:
        underline_y = min(
            big_height - 2,
            int(y + text_height + rng.uniform(0.02, 0.12) * big_height),
        )
        line_width = rng.randint(max(2, oversample), max(3, oversample * 2))
        start_x = int(x + rng.uniform(-0.04, 0.08) * text_width)
        end_x = int(x + text_width + rng.uniform(-0.08, 0.05) * text_width)
        draw.line(
            (start_x, underline_y, end_x, underline_y + rng.randint(-2, 2)),
            fill=(*color, rng.randint(150, 235)),
            width=line_width,
        )

    scale = rng.uniform(0.92, 1.08)
    scaled_size = (
        max(1, int(big_width * scale)),
        max(1, int(big_height * scale)),
    )
    layer = layer.resize(scaled_size, Image.Resampling.BICUBIC)

    rotation = rng.uniform(-5.5, 5.5)
    layer = layer.rotate(
        rotation,
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )

    return layer, rotation, underline


def composite_centered(
    background: Image.Image,
    foreground: Image.Image,
    rng: random.Random,
) -> Image.Image:
    """Place foreground near the center with slight random translation."""
    bg = background.convert("RGBA")
    fg = foreground.convert("RGBA")

    max_shift_x = max(1, background.width // 18)
    max_shift_y = max(1, background.height // 12)

    x = (background.width - fg.width) // 2 + rng.randint(-max_shift_x, max_shift_x)
    y = (background.height - fg.height) // 2 + rng.randint(-max_shift_y, max_shift_y)

    bg.alpha_composite(fg, (x, y))
    return bg.convert("RGB")


def postprocess(image: Image.Image, rng: random.Random) -> tuple[Image.Image, float]:
    """Apply mild blur, contrast, sharpness, and occasional resampling artifacts."""
    blur_radius = rng.choices(
        [rng.uniform(0.0, 0.35), rng.uniform(0.35, 0.85), rng.uniform(0.85, 1.25)],
        weights=[0.45, 0.45, 0.10],
        k=1,
    )[0]

    if blur_radius > 0.02:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.10))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.96, 1.04))

    # Simulate browser/image scaling.
    if rng.random() < 0.25:
        downscale = rng.uniform(0.78, 0.94)
        small = image.resize(
            (
                max(8, int(image.width * downscale)),
                max(8, int(image.height * downscale)),
            ),
            Image.Resampling.BILINEAR,
        )
        image = small.resize(image.size, Image.Resampling.BICUBIC)

    return image, blur_radius


def generate_sample(
    label: str,
    font_path: Path,
    width: int,
    height: int,
    sample_seed: int,
) -> tuple[Image.Image, dict]:
    rng = random.Random(sample_seed)
    np_rng = np.random.default_rng(sample_seed)

    bg_color = rng.choice(BACKGROUND_COLORS)
    background = Image.new("RGB", (width, height), bg_color)
    background = add_background_texture(background, rng, np_rng)

    text_layer, rotation, underline = render_text_layer(
        text=label,
        font_path=font_path,
        canvas_size=(width, height),
        rng=rng,
    )

    image = composite_centered(background, text_layer, rng)
    image, blur_radius = postprocess(image, rng)

    metadata = {
        "font_name": font_path.name,
        "text_color": "mixed_palette",
        "bg_color": rgb_to_string(bg_color),
        "rotation": round(rotation, 4),
        "scale": "embedded",
        "blur": round(blur_radius, 4),
        "underline": underline,
    }
    return image, metadata


def choose_label(index: int, rng: random.Random, balanced: bool) -> str:
    if balanced:
        # Cycles uniformly over all 1,000 labels before reshuffling through the next block.
        block = index // 1000
        local = index % 1000
        block_rng = random.Random(rng.randint(0, 2**31 - 1) + block)
        labels = list(range(1000))
        block_rng.shuffle(labels)
        return f"{labels[local]:03d}"

    return f"{rng.randint(0, 999):03d}"


def write_manifest(rows: Iterable[SampleMetadata], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty manifest.")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def generate_dataset(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    train_dir = output_dir / "images" / "train"
    val_dir = output_dir / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    extra_dirs = [Path(p).expanduser().resolve() for p in args.font_dir]
    fonts = discover_fonts(extra_dirs)
    if not fonts:
        raise RuntimeError(
            "No usable fonts found. Pass one or more directories with --font-dir."
        )

    if args.max_fonts is not None:
        fonts = fonts[: args.max_fonts]

    master_rng = random.Random(args.seed)
    indices = list(range(args.count))
    master_rng.shuffle(indices)

    val_count = int(round(args.count * args.val_ratio))
    val_indices = set(indices[:val_count])

    manifest_rows: list[SampleMetadata] = []

    for index in range(args.count):
        split = "val" if index in val_indices else "train"
        destination_dir = val_dir if split == "val" else train_dir

        sample_seed = master_rng.randint(0, 2**31 - 1)
        sample_rng = random.Random(sample_seed)
        font_path = sample_rng.choice(fonts)
        label = choose_label(index=index, rng=master_rng, balanced=not args.unbalanced)

        image, meta = generate_sample(
            label=label,
            font_path=font_path,
            width=args.width,
            height=args.height,
            sample_seed=sample_seed,
        )

        filename = f"{index:07d}_{label}.png"
        output_path = destination_dir / filename
        image.save(output_path, format="PNG", optimize=True)

        relative_path = output_path.relative_to(output_dir).as_posix()
        manifest_rows.append(
            SampleMetadata(
                path=relative_path,
                label=label,
                split=split,
                seed=sample_seed,
                font_name=meta["font_name"],
                text_color=meta["text_color"],
                bg_color=meta["bg_color"],
                rotation=meta["rotation"],
                scale=1.0,
                blur=meta["blur"],
                underline=meta["underline"],
            )
        )

        if (index + 1) % args.log_every == 0 or index + 1 == args.count:
            print(f"Generated {index + 1:,}/{args.count:,}")

    write_manifest(manifest_rows, output_dir / "manifest.csv")

    train_count = args.count - val_count
    print()
    print(f"Done.")
    print(f"Fonts used: {len(fonts)}")
    print(f"Train images: {train_count:,}")
    print(f"Validation images: {val_count:,}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic three-digit CAPTCHA images."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20_000,
        help="Total number of images.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction reserved for synthetic validation.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--font-dir",
        action="append",
        default=[],
        help="Additional font directory. May be supplied more than once.",
    )
    parser.add_argument(
        "--max-fonts",
        type=int,
        default=40,
        help="Maximum number of discovered fonts to use.",
    )
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        help="Sample labels independently instead of approximately balancing 000-999.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print progress every N images.",
    )

    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive.")
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1).")
    if args.width < 16 or args.height < 16:
        parser.error("--width and --height must both be at least 16.")
    if args.max_fonts is not None and args.max_fonts <= 0:
        parser.error("--max-fonts must be positive.")

    return args


if __name__ == "__main__":
    generate_dataset(parse_args())
