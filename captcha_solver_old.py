from __future__ import annotations

import argparse
import json
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
from target_extractor import TargetResult, extract_target


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
    target_confidence: float
    target_variant: str
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
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]], np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("Cannot solve an empty screenshot")

    candidates, _edges = find_square_candidates(image)
    boxes = select_grid_boxes(candidates)
    tiles = crop_individual_tiles(image, boxes)

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
    return tiles, boxes, debug


def solve_captcha(
    image: np.ndarray,
    target: str | None,
    reader: Any,
) -> CaptchaDecision:
    tiles, boxes, _debug = extract_tiles_from_screenshot(image)
    grid_box = bounding_rectangle(boxes)
    target_result = (
        TargetResult(validate_target(target), 1.0, "manual", np.empty((0, 0), dtype=np.uint8))
        if target is not None
        else extract_target(reader, image, grid_box)
    )
    target = target_result.target

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

    return CaptchaDecision(
        target=target,
        target_confidence=target_result.confidence,
        target_variant=target_result.variant,
        selected_tiles=tuple(selected),
        uncertain_tiles=tuple(uncertain),
        tiles=tuple(decisions),
    )


def write_debug_outputs(
    output_dir: Path,
    image: np.ndarray,
    debug: np.ndarray,
    tiles: list[np.ndarray],
    decision: CaptchaDecision,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "screenshot.png"), image)
    cv2.imwrite(str(output_dir / "detected.png"), debug)

    for tile_number, tile in enumerate(tiles, start=1):
        cv2.imwrite(str(output_dir / f"tile_{tile_number}.png"), tile)

    payload = {
        "target": decision.target,
        "target_confidence": decision.target_confidence,
        "target_variant": decision.target_variant,
        "selected_tiles": list(decision.selected_tiles),
        "uncertain_tiles": list(decision.uncertain_tiles),
        "tiles": [asdict(tile) for tile in decision.tiles],
    }
    (output_dir / "decision.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve a 3x3 CAPTCHA screenshot and optionally extract its target automatically."
    )
    parser.add_argument("image", type=Path, help="Path to the CAPTCHA screenshot")
    parser.add_argument(
        "target",
        nargs="?",
        help="Optional three-digit target. Omit to extract it automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/live_solver"),
        help="Debug output directory",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Ask EasyOCR to use a supported GPU",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    tiles, boxes, debug = extract_tiles_from_screenshot(image)

    print("Loading EasyOCR...")
    reader = build_reader(gpu=args.gpu)

    grid_box = bounding_rectangle(boxes)
    if args.target is None:
        target_result = extract_target(reader, image, grid_box)
        target = target_result.target
        print(
            f"Detected target: {target} "
            f"confidence={target_result.confidence:.3f} "
            f"variant={target_result.variant}"
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(args.output_dir / "instruction.png"),
            target_result.prompt_crop,
        )
    else:
        target = validate_target(args.target)
        target_result = TargetResult(
            target=target,
            confidence=1.0,
            variant="manual",
            prompt_crop=np.empty((0, 0), dtype=np.uint8),
        )

    # Reuse the already extracted tiles rather than detecting twice.
    tile_decisions: list[TileDecision] = []
    selected: list[int] = []
    uncertain: list[int] = []

    for tile_number, tile in enumerate(tiles, start=1):
        result = solve_tile(tile, reader, target=target)
        item = TileDecision(
            tile=tile_number,
            prediction=str(result["prediction"]),
            score=float(result["score"]),
            votes=int(result["votes"]),
            supporting_variants=tuple(result["supporting_variants"]),
            matches_target=bool(result["matches_target"]),
            uncertain=bool(result["uncertain"]),
        )
        tile_decisions.append(item)
        if item.matches_target:
            selected.append(tile_number)
        if item.uncertain:
            uncertain.append(tile_number)

        marker = "SELECT" if item.matches_target else "skip"
        print(
            f"Tile {tile_number}: {item.prediction or '<blank>'} "
            f"score={item.score:.3f} votes={item.votes} -> {marker}"
        )

    decision = CaptchaDecision(
        target=target,
        target_confidence=target_result.confidence,
        target_variant=target_result.variant,
        selected_tiles=tuple(selected),
        uncertain_tiles=tuple(uncertain),
        tiles=tuple(tile_decisions),
    )
    write_debug_outputs(args.output_dir, image, debug, tiles, decision)

    print()
    print(
        f"Target: {decision.target} "
        f"(confidence={decision.target_confidence:.3f}, "
        f"variant={decision.target_variant})"
    )
    print(f"Selected tiles: {list(decision.selected_tiles)}")
    print(f"Uncertain tiles: {list(decision.uncertain_tiles)}")
    print(f"Decision file: {args.output_dir / 'decision.json'}")


if __name__ == "__main__":
    main()
