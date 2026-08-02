from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from extract_tiles import (
    bounding_rectangle,
    crop_individual_tiles,
    draw_detection_debug,
    expand_box,
    find_square_candidates,
    select_grid_boxes,
)
from ocr import build_reader
from solver import solve_tile


TARGET_PATTERN = re.compile(
    r"please\s+select\s+all\s+boxes\s+with\s+number\s+(\d{3})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TileDecision:
    tile: int
    prediction: str
    score: float
    votes: int
    supporting_variants: tuple[str, ...]
    matches_target: bool
    uncertain: bool


@dataclass(frozen=True)
class CaptchaDecision:
    target: str
    status: str
    selected_tiles: tuple[int, ...]
    uncertain_tiles: tuple[int, ...]
    tiles: tuple[TileDecision, ...]


def validate_target(target: str) -> str:
    target = target.strip()
    if len(target) != 3 or not target.isdigit():
        raise ValueError("target must be exactly three digits")
    return target


def extract_tiles_from_screenshot(
    image: np.ndarray,
) -> tuple[
    list[np.ndarray],
    list[tuple[int, int, int, int]],
    tuple[int, int, int, int],
    np.ndarray,
]:
    if image is None or image.size == 0:
        raise ValueError("Cannot solve an empty screenshot")

    candidates, _edges = find_square_candidates(image)
    boxes = select_grid_boxes(candidates)
    tiles = crop_individual_tiles(image, boxes)

    if len(tiles) != 9:
        raise RuntimeError(f"Expected 9 CAPTCHA tiles, found {len(tiles)}")

    grid_box = bounding_rectangle(boxes)
    panel_box = expand_box(
        grid_box,
        image.shape,
        left_ratio=0.04,
        top_ratio=0.23,
        right_ratio=0.04,
        bottom_ratio=0.04,
    )
    debug = draw_detection_debug(image, boxes, grid_box, panel_box)
    return tiles, boxes, grid_box, debug


def extract_target_from_prompt(
    image: np.ndarray,
    grid_box: tuple[int, int, int, int],
    reader: Any,
) -> str:
    """
    OCR the instruction above the detected 3x3 grid.

    This is primarily a CLI fallback. In browser automation, reading the
    target from the visible DOM label is more reliable and should be preferred.
    """
    grid_x, grid_y, grid_w, _grid_h = grid_box

    if grid_y <= 0:
        raise RuntimeError("No prompt area exists above the detected grid")

    horizontal_padding = max(8, int(grid_w * 0.04))
    x1 = max(0, grid_x - horizontal_padding)
    x2 = min(image.shape[1], grid_x + grid_w + horizontal_padding)
    prompt = image[0:grid_y, x1:x2]

    if prompt.size == 0:
        raise RuntimeError("The CAPTCHA prompt crop is empty")

    results = reader.readtext(
        prompt,
        detail=0,
        paragraph=True,
    )
    text = " ".join(str(item) for item in results).strip()

    match = TARGET_PATTERN.search(text)
    if match:
        return validate_target(match.group(1))

    # Fallback for OCR that reads the sentence poorly but still sees the number.
    three_digit_numbers = re.findall(r"(?<!\d)\d{3}(?!\d)", text)
    if len(three_digit_numbers) == 1:
        return validate_target(three_digit_numbers[0])

    raise RuntimeError(
        "Could not identify one unambiguous three-digit target from the "
        f"CAPTCHA prompt. OCR text was: {text!r}"
    )


def solve_tiles(
    tiles: list[np.ndarray],
    target: str,
    reader: Any,
) -> CaptchaDecision:
    target = validate_target(target)

    decisions: list[TileDecision] = []
    selected: list[int] = []
    uncertain: list[int] = []

    for tile_number, tile in enumerate(tiles, start=1):
        result = solve_tile(tile, reader, target=target)
        decision = TileDecision(
            tile=tile_number,
            prediction=str(result["prediction"]),
            score=float(result["score"]),
            votes=int(result["votes"]),
            supporting_variants=tuple(result["supporting_variants"]),
            matches_target=bool(result["matches_target"]),
            uncertain=bool(result["uncertain"]),
        )
        decisions.append(decision)

        if decision.matches_target:
            selected.append(tile_number)
        if decision.uncertain:
            uncertain.append(tile_number)

    status = "confident" if not uncertain else "uncertain"

    return CaptchaDecision(
        target=target,
        status=status,
        selected_tiles=tuple(selected),
        uncertain_tiles=tuple(uncertain),
        tiles=tuple(decisions),
    )


def solve_captcha_image(
    image: np.ndarray,
    *,
    target: str | None = None,
    reader: Any | None = None,
    gpu: bool = False,
) -> tuple[
    CaptchaDecision,
    list[np.ndarray],
    list[tuple[int, int, int, int]],
    np.ndarray,
]:
    """
    Solve a CAPTCHA image.

    Browser code should pass the target extracted from the true visible DOM
    label. When target is omitted, the solver OCRs the prompt as a fallback.
    """
    tiles, boxes, grid_box, debug = extract_tiles_from_screenshot(image)

    if reader is None:
        reader = build_reader(gpu=gpu)

    resolved_target = (
        validate_target(target)
        if target is not None
        else extract_target_from_prompt(image, grid_box, reader)
    )

    decision = solve_tiles(tiles, resolved_target, reader)
    return decision, tiles, boxes, debug


def write_outputs(
    output_dir: Path,
    image: np.ndarray,
    debug: np.ndarray,
    tiles: list[np.ndarray],
    decision: CaptchaDecision,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_dir / "screenshot.png"), image):
        raise OSError(f"Could not write {output_dir / 'screenshot.png'}")
    if not cv2.imwrite(str(output_dir / "detected.png"), debug):
        raise OSError(f"Could not write {output_dir / 'detected.png'}")

    for tile_number, tile in enumerate(tiles, start=1):
        tile_path = output_dir / f"tile_{tile_number}.png"
        if not cv2.imwrite(str(tile_path), tile):
            raise OSError(f"Could not write {tile_path}")

    payload = {
        "target": decision.target,
        "status": decision.status,
        "selected_tiles": list(decision.selected_tiles),
        "uncertain_tiles": list(decision.uncertain_tiles),
        "tiles": [asdict(tile) for tile in decision.tiles],
    }
    (output_dir / "decision.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def print_decision(decision: CaptchaDecision) -> None:
    for tile in decision.tiles:
        marker = "SELECT" if tile.matches_target else "skip"
        print(
            f"Tile {tile.tile}: {tile.prediction or '<blank>'} "
            f"score={tile.score:.3f} votes={tile.votes} -> {marker}"
        )

    print()
    print(f"Target: {decision.target}")
    print(f"Status: {decision.status}")
    print(f"Selected tiles: {list(decision.selected_tiles)}")
    print(f"Uncertain tiles: {list(decision.uncertain_tiles)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve one 3x3 CAPTCHA screenshot. The target is automatically "
            "OCRed from the instruction unless --target is supplied."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the CAPTCHA screenshot",
    )
    parser.add_argument(
        "--target",
        type=validate_target,
        help=(
            "Optional three-digit target override. Browser automation should "
            "normally obtain this from the true visible DOM label."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/live_solver"),
        help="Output directory",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Ask EasyOCR to use a supported GPU",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    print("Loading EasyOCR...")
    reader = build_reader(gpu=args.gpu)

    decision, tiles, _boxes, debug = solve_captcha_image(
        image,
        target=args.target,
        reader=reader,
    )

    print_decision(decision)
    write_outputs(args.output, image, debug, tiles, decision)

    print(f"Decision file: {args.output / 'decision.json'}")
    return 0 if decision.status == "confident" else 2


if __name__ == "__main__":
    raise SystemExit(main())
