from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("extracted")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

NORMALIZED_PANEL_SIZE = (720, 900)
NORMALIZED_GRID_SIZE = (600, 600)
NORMALIZED_TILE_SIZE = (200, 200)


Box = tuple[int, int, int, int]


def box_iou(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    intersection_left = max(ax, bx)
    intersection_top = max(ay, by)
    intersection_right = min(ax + aw, bx + bw)
    intersection_bottom = min(ay + ah, by + bh)

    intersection_width = max(
        0,
        intersection_right - intersection_left,
    )

    intersection_height = max(
        0,
        intersection_bottom - intersection_top,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = aw * ah
    area_b = bw * bh

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def remove_duplicate_boxes(
    boxes: list[Box],
    threshold: float = 0.65,
) -> list[Box]:
    """
    Remove inner and outer contours belonging to the same tile.
    """

    ordered = sorted(
        boxes,
        key=lambda box: box[2] * box[3],
        reverse=True,
    )

    kept: list[Box] = []

    for candidate in ordered:
        overlaps_existing = any(
            box_iou(candidate, existing) >= threshold
            for existing in kept
        )

        if not overlaps_existing:
            kept.append(candidate)

    return kept


def find_square_candidates(
    image: np.ndarray,
) -> tuple[list[Box], np.ndarray]:
    """
    Find rectangular objects that could be CAPTCHA tiles.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        35,
        120,
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width

    minimum_area = image_area * 0.003
    maximum_area = image_area * 0.08

    candidates: list[Box] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if width < 40 or height < 40:
            continue

        box_area = width * height

        if not minimum_area <= box_area <= maximum_area:
            continue

        aspect_ratio = width / height

        if not 0.72 <= aspect_ratio <= 1.28:
            continue

        contour_area = cv2.contourArea(contour)

        if contour_area <= 0:
            continue

        rectangularity = contour_area / box_area

        if rectangularity < 0.45:
            continue

        candidates.append(
            (x, y, width, height)
        )

    candidates = remove_duplicate_boxes(candidates)

    return candidates, closed


def find_dominant_size_group(
    candidates: list[Box],
) -> list[Box]:
    """
    The nine actual tiles should have almost identical dimensions.

    For every candidate, create a group containing boxes with
    similar width and height. Keep the largest group.
    """

    if len(candidates) < 9:
        raise RuntimeError(
            f"Only found {len(candidates)} square candidates."
        )

    best_group: list[Box] = []

    for anchor in candidates:
        _, _, anchor_width, anchor_height = anchor

        group: list[Box] = []

        for candidate in candidates:
            _, _, width, height = candidate

            width_difference = (
                abs(width - anchor_width) / anchor_width
            )

            height_difference = (
                abs(height - anchor_height) / anchor_height
            )

            if (
                width_difference <= 0.22
                and height_difference <= 0.22
            ):
                group.append(candidate)

        if len(group) > len(best_group):
            best_group = group

    if len(best_group) < 9:
        raise RuntimeError(
            "Could not find nine similarly sized tiles. "
            f"Largest size group contained {len(best_group)} boxes."
        )

    return best_group


def cluster_axis(
    values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide x or y center coordinates into three clusters.

    Returns labels ordered from left-to-right or top-to-bottom,
    plus the ordered cluster centers.
    """

    if len(values) < 3:
        raise ValueError(
            "At least three values are required for clustering."
        )

    data = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1, 1)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.1,
    )

    _, labels, centers = cv2.kmeans(
        data,
        3,
        None,
        criteria,
        20,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers[:, 0]

    center_order = np.argsort(centers)

    label_mapping = {
        int(original_label): ordered_label
        for ordered_label, original_label
        in enumerate(center_order)
    }

    ordered_labels = np.array(
        [
            label_mapping[int(label)]
            for label in labels[:, 0]
        ],
        dtype=np.int32,
    )

    ordered_centers = centers[center_order]

    return ordered_labels, ordered_centers


def select_grid_boxes(
    candidates: list[Box],
) -> list[Box]:
    """
    Select one box for every cell of a 3×3 arrangement.
    """

    size_group = find_dominant_size_group(candidates)

    x_centers = [
        x + width / 2
        for x, _, width, _ in size_group
    ]

    y_centers = [
        y + height / 2
        for _, y, _, height in size_group
    ]

    column_labels, column_centers = cluster_axis(x_centers)
    row_labels, row_centers = cluster_axis(y_centers)

    widths = np.asarray(
        [box[2] for box in size_group],
        dtype=np.float32,
    )

    heights = np.asarray(
        [box[3] for box in size_group],
        dtype=np.float32,
    )

    median_width = float(np.median(widths))
    median_height = float(np.median(heights))

    selected: dict[tuple[int, int], tuple[float, Box]] = {}

    for index, box in enumerate(size_group):
        x, y, width, height = box

        row = int(row_labels[index])
        column = int(column_labels[index])

        center_x = x + width / 2
        center_y = y + height / 2

        position_error = (
            abs(center_x - column_centers[column])
            / max(median_width, 1)
            + abs(center_y - row_centers[row])
            / max(median_height, 1)
        )

        size_error = (
            abs(width - median_width)
            / max(median_width, 1)
            + abs(height - median_height)
            / max(median_height, 1)
        )

        score = position_error + size_error

        cell = (row, column)

        previous = selected.get(cell)

        if previous is None or score < previous[0]:
            selected[cell] = (score, box)

    missing_cells = [
        (row, column)
        for row in range(3)
        for column in range(3)
        if (row, column) not in selected
    ]

    if missing_cells:
        raise RuntimeError(
            "Could not fill every grid position. "
            f"Missing cells: {missing_cells}"
        )

    ordered_boxes = [
        selected[(row, column)][1]
        for row in range(3)
        for column in range(3)
    ]

    validate_grid_geometry(ordered_boxes)

    return ordered_boxes


def validate_grid_geometry(
    boxes: list[Box],
) -> None:
    if len(boxes) != 9:
        raise RuntimeError(
            f"Expected 9 boxes, received {len(boxes)}."
        )

    widths = np.asarray(
        [box[2] for box in boxes],
        dtype=np.float32,
    )

    heights = np.asarray(
        [box[3] for box in boxes],
        dtype=np.float32,
    )

    median_width = float(np.median(widths))
    median_height = float(np.median(heights))

    for index, (_, _, width, height) in enumerate(
        boxes,
        start=1,
    ):
        width_ratio = width / median_width
        height_ratio = height / median_height

        if not 0.75 <= width_ratio <= 1.25:
            raise RuntimeError(
                f"Tile {index} width is inconsistent."
            )

        if not 0.75 <= height_ratio <= 1.25:
            raise RuntimeError(
                f"Tile {index} height is inconsistent."
            )


def bounding_rectangle(
    boxes: list[Box],
) -> Box:
    left = min(x for x, _, _, _ in boxes)
    top = min(y for _, y, _, _ in boxes)

    right = max(
        x + width
        for x, _, width, _ in boxes
    )

    bottom = max(
        y + height
        for _, y, _, height in boxes
    )

    return (
        left,
        top,
        right - left,
        bottom - top,
    )


def expand_box(
    box: Box,
    image_shape: tuple[int, ...],
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> Box:
    x, y, width, height = box

    image_height, image_width = image_shape[:2]

    left_padding = int(width * left_ratio)
    top_padding = int(height * top_ratio)
    right_padding = int(width * right_ratio)
    bottom_padding = int(height * bottom_ratio)

    left = max(0, x - left_padding)
    top = max(0, y - top_padding)

    right = min(
        image_width,
        x + width + right_padding,
    )

    bottom = min(
        image_height,
        y + height + bottom_padding,
    )

    return (
        left,
        top,
        right - left,
        bottom - top,
    )


def crop_box(
    image: np.ndarray,
    box: Box,
) -> np.ndarray:
    x, y, width, height = box

    crop = image[
        y:y + height,
        x:x + width,
    ]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop produced from box {box}."
        )

    return crop


def crop_individual_tiles(
    image: np.ndarray,
    boxes: list[Box],
) -> list[np.ndarray]:
    tiles: list[np.ndarray] = []

    for x, y, width, height in boxes:
        margin_x = max(
            2,
            int(width * 0.035),
        )

        margin_y = max(
            2,
            int(height * 0.035),
        )

        left = x + margin_x
        top = y + margin_y
        right = x + width - margin_x
        bottom = y + height - margin_y

        tile = image[
            top:bottom,
            left:right,
        ]

        if tile.size == 0:
            raise ValueError(
                f"An empty tile was produced from "
                f"x={x}, y={y}, w={width}, h={height}."
            )

        normalized = cv2.resize(
            tile,
            NORMALIZED_TILE_SIZE,
            interpolation=cv2.INTER_AREA,
        )

        tiles.append(normalized)

    return tiles


def draw_detection_debug(
    image: np.ndarray,
    boxes: list[Box],
    grid_box: Box,
    panel_box: Box,
) -> np.ndarray:
    debug = image.copy()

    panel_x, panel_y, panel_width, panel_height = panel_box

    cv2.rectangle(
        debug,
        (panel_x, panel_y),
        (
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        (255, 0, 0),
        3,
    )

    grid_x, grid_y, grid_width, grid_height = grid_box

    cv2.rectangle(
        debug,
        (grid_x, grid_y),
        (
            grid_x + grid_width,
            grid_y + grid_height,
        ),
        (0, 255, 255),
        3,
    )

    for tile_number, box in enumerate(
        boxes,
        start=1,
    ):
        x, y, width, height = box

        cv2.rectangle(
            debug,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            debug,
            str(tile_number),
            (x + 8, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return debug


def process_image(
    image_path: Path,
) -> dict[str, object]:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"OpenCV could not read {image_path.name}."
        )

    candidates, edge_image = find_square_candidates(image)

    tile_boxes = select_grid_boxes(candidates)

    grid_box = bounding_rectangle(tile_boxes)

    # The instruction text sits above the detected grid.
    # The panel itself has only small side/bottom margins.
    panel_box = expand_box(
        grid_box,
        image.shape,
        left_ratio=0.04,
        top_ratio=0.23,
        right_ratio=0.04,
        bottom_ratio=0.04,
    )

    grid = crop_box(
        image,
        grid_box,
    )

    panel = crop_box(
        image,
        panel_box,
    )

    normalized_grid = cv2.resize(
        grid,
        NORMALIZED_GRID_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    normalized_panel = cv2.resize(
        panel,
        NORMALIZED_PANEL_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    tiles = crop_individual_tiles(
        image,
        tile_boxes,
    )

    debug = draw_detection_debug(
        image,
        tile_boxes,
        grid_box,
        panel_box,
    )

    image_output_dir = (
        OUTPUT_DIR / image_path.stem
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(image_output_dir / "edges.png"),
        edge_image,
    )

    cv2.imwrite(
        str(image_output_dir / "detected.png"),
        debug,
    )

    cv2.imwrite(
        str(image_output_dir / "panel.png"),
        panel,
    )

    cv2.imwrite(
        str(image_output_dir / "panel_normalized.png"),
        normalized_panel,
    )

    cv2.imwrite(
        str(image_output_dir / "grid.png"),
        grid,
    )

    cv2.imwrite(
        str(image_output_dir / "grid_normalized.png"),
        normalized_grid,
    )

    for tile_number, tile in enumerate(
        tiles,
        start=1,
    ):
        cv2.imwrite(
            str(
                image_output_dir
                / f"tile_{tile_number}.png"
            ),
            tile,
        )

    return {
        "image": image_path.name,
        "status": "ok",
        "candidates": len(candidates),
        "tiles": len(tiles),
        "grid_x": grid_box[0],
        "grid_y": grid_box[1],
        "grid_width": grid_box[2],
        "grid_height": grid_box[3],
        "error": "",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder does not exist: "
            f"{DATASET_DIR.resolve()}"
        )

    image_paths = sorted(
        path
        for path in DATASET_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in "
            f"{DATASET_DIR.resolve()}"
        )

    results: list[dict[str, object]] = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:
            result = process_image(image_path)

            print(
                "  Detected 9 tiles successfully."
            )

        except (
            RuntimeError,
            ValueError,
            cv2.error,
        ) as error:
            print(
                f"  Failed: {error}"
            )

            result = {
                "image": image_path.name,
                "status": "failed",
                "candidates": "",
                "tiles": 0,
                "grid_x": "",
                "grid_y": "",
                "grid_width": "",
                "grid_height": "",
                "error": str(error),
            }

        results.append(result)

    report_path = OUTPUT_DIR / "extraction_report.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "image",
            "status",
            "candidates",
            "tiles",
            "grid_x",
            "grid_y",
            "grid_width",
            "grid_height",
            "error",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    successful = sum(
        result["status"] == "ok"
        for result in results
    )

    print()
    print(
        f"Successful: {successful}/{len(results)}"
    )
    print(
        f"Output folder: {OUTPUT_DIR.resolve()}"
    )
    print(
        f"Report: {report_path.resolve()}"
    )


if __name__ == "__main__":
    main()
