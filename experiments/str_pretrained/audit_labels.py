#!/usr/bin/env python3
"""
Interactive label audit for the captcha digit-tile dataset.

Workflow:
    - Shows one labeled tile at a time.
    - Press ENTER if the current 3-digit label is correct.
    - Type the corrected 3-digit number if the label is wrong.
    - q = save and quit.
    - b = go back one tile.
    - s = skip without changing.
    - Changes are written back to the original split CSVs.
    - A timestamped backup is created before any CSV is modified.

Default source:
    experiments/convnext/run_001/train_split.csv
    experiments/convnext/run_001/val_split.csv
    experiments/convnext/run_001/test_split.csv

IMPORTANT:
    This does NOT regenerate or reshuffle the locked split.
    It only edits labels inside the existing split CSVs.

Examples:
    # Audit the whole 513-tile dataset
    python experiments/str_pretrained/audit_labels.py

    # Audit validation only
    python experiments/str_pretrained/audit_labels.py --split val

    # Start at a particular row
    python experiments/str_pretrained/audit_labels.py --start 100

Controls:
    ENTER  keep current label
    123    replace current label with 123
    b      go back one item
    s      skip item
    q      save and quit
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


DEFAULT_SPLITS = {
    "train": Path("experiments/convnext/run_001/train_split.csv"),
    "val": Path("experiments/convnext/run_001/val_split.csv"),
    "test": Path("experiments/convnext/run_001/test_split.csv"),
}

LABEL_COLUMNS = [
    "label",
    "ground_truth",
    "groundtruth",
    "gt",
    "target",
    "truth",
    "text",
]

IMAGE_COLUMNS = [
    "tile_path",
    "crop_path",
    "image_path",
    "path",
    "file",
    "filename",
    "image",
]

SOURCE_COLUMNS = [
    "source_image",
    "source_path",
    "screenshot_path",
    "screenshot",
    "image",
]

TILE_COLUMNS = [
    "tile",
    "tile_index",
    "tile_idx",
    "position",
    "tile_position",
]

# Common bounding-box schemas.
BBOX_SCHEMAS = [
    ("x1", "y1", "x2", "y2"),
    ("left", "top", "right", "bottom"),
    ("xmin", "ymin", "xmax", "ymax"),
]

XYWH_SCHEMAS = [
    ("x", "y", "w", "h"),
    ("x", "y", "width", "height"),
    ("left", "top", "width", "height"),
]


def lower_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cmap = lower_map(df)
    for candidate in candidates:
        if candidate in cmap:
            return cmap[candidate]
    return None


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return "".join(ch for ch in text if ch.isdigit())


def resolve_path(raw_value, csv_path: Path, project_root: Path) -> Optional[Path]:
    if pd.isna(raw_value):
        return None

    raw = str(raw_value).strip()
    if not raw:
        return None

    p = Path(raw).expanduser()

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                project_root / p,
                csv_path.parent / p,
                p,
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    # Filename-only fallback: search a few likely dataset roots.
    filename = p.name
    search_roots = [
        project_root / "images",
        project_root / "data",
        project_root / "dataset",
        project_root / "datasets",
        project_root / "experiments",
    ]

    matches = []
    for root in search_roots:
        if not root.exists():
            continue
        try:
            matches.extend(root.rglob(filename))
        except PermissionError:
            pass
        if matches:
            break

    if len(matches) == 1:
        return matches[0].resolve()

    return None


def bbox_from_row(row: pd.Series, columns: List[str]) -> Optional[Tuple[int, int, int, int]]:
    cmap = {str(c).strip().lower(): c for c in columns}

    for schema in BBOX_SCHEMAS:
        if all(name in cmap for name in schema):
            try:
                x1 = int(float(row[cmap[schema[0]]]))
                y1 = int(float(row[cmap[schema[1]]]))
                x2 = int(float(row[cmap[schema[2]]]))
                y2 = int(float(row[cmap[schema[3]]]))
                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2
            except (TypeError, ValueError):
                pass

    for schema in XYWH_SCHEMAS:
        if all(name in cmap for name in schema):
            try:
                x = int(float(row[cmap[schema[0]]]))
                y = int(float(row[cmap[schema[1]]]))
                w = int(float(row[cmap[schema[2]]]))
                h = int(float(row[cmap[schema[3]]]))
                if w > 0 and h > 0:
                    return x, y, x + w, y + h
            except (TypeError, ValueError):
                pass

    return None


def equal_grid_crop(image: Image.Image, tile_number: int) -> Image.Image:
    """
    Last-resort 3x3 crop.

    This is only correct when the stored source image is already the captcha
    grid itself. The script announces when this fallback is used.
    """
    if tile_number < 1 or tile_number > 9:
        raise ValueError(f"Tile number must be 1..9, got {tile_number}")

    w, h = image.size
    col = (tile_number - 1) % 3
    row = (tile_number - 1) // 3

    x1 = round(col * w / 3)
    x2 = round((col + 1) * w / 3)
    y1 = round(row * h / 3)
    y2 = round((row + 1) * h / 3)

    return image.crop((x1, y1, x2, y2))


def load_tile_image(
    row: pd.Series,
    csv_path: Path,
    project_root: Path,
    columns: List[str],
) -> Tuple[Image.Image, str]:
    image_col = find_column(pd.DataFrame(columns=columns), IMAGE_COLUMNS)
    source_col = find_column(pd.DataFrame(columns=columns), SOURCE_COLUMNS)
    tile_col = find_column(pd.DataFrame(columns=columns), TILE_COLUMNS)

    # First try the most direct image/path field.
    if image_col is not None:
        direct_path = resolve_path(row[image_col], csv_path, project_root)
        if direct_path is not None:
            img = Image.open(direct_path).convert("RGB")
            bbox = bbox_from_row(row, columns)
            if bbox is not None:
                return img.crop(bbox), f"{direct_path.name} | bbox crop"

            # If a tile index exists, this could be a source screenshot.
            # If no tile index exists, we assume this is already a tile crop.
            if tile_col is None:
                return img, direct_path.name

            # If the image path's filename looks tile-specific, use it directly.
            name_lower = direct_path.stem.lower()
            try:
                tile_num = int(float(row[tile_col]))
            except (TypeError, ValueError):
                tile_num = None

            if tile_num is not None and (
                f"tile{tile_num}" in name_lower
                or f"tile_{tile_num}" in name_lower
                or f"-{tile_num}" in name_lower
            ):
                return img, direct_path.name

    # Then try explicit source screenshot + crop metadata / tile index.
    if source_col is not None:
        source_path = resolve_path(row[source_col], csv_path, project_root)
        if source_path is not None:
            img = Image.open(source_path).convert("RGB")

            bbox = bbox_from_row(row, columns)
            if bbox is not None:
                return img.crop(bbox), f"{source_path.name} | bbox crop"

            if tile_col is not None:
                try:
                    tile_num = int(float(row[tile_col]))
                    cropped = equal_grid_crop(img, tile_num)
                    return cropped, f"{source_path.name} | 3x3 fallback tile {tile_num}"
                except (TypeError, ValueError):
                    pass

            return img, source_path.name

    # If direct image resolved earlier and wasn't clearly a source, use it.
    if image_col is not None:
        direct_path = resolve_path(row[image_col], csv_path, project_root)
        if direct_path is not None:
            return Image.open(direct_path).convert("RGB"), direct_path.name

    raise FileNotFoundError(
        "Could not locate an image for this row.\n"
        f"CSV: {csv_path}\n"
        f"Columns: {columns}\n"
        "Expected an image/path column, or source screenshot information."
    )


def show_tile(
    image: Image.Image,
    split_name: str,
    index_in_audit: int,
    total: int,
    label: str,
    descriptor: str,
    row: pd.Series,
    tile_col: Optional[str],
):
    plt.clf()

    # Nearest-neighbor display keeps hard digit edges visible.
    plt.imshow(image.resize((max(300, image.width * 5), max(120, image.height * 5))))
    plt.axis("off")

    tile_info = ""
    if tile_col is not None:
        tile_info = f" | tile {row[tile_col]}"

    plt.title(
        f"[{index_in_audit + 1}/{total}] {split_name}{tile_info}\n"
        f"Current label: {label}\n"
        f"{descriptor}\n"
        f"ENTER=correct | type 3 digits=fix | b=back | s=skip | q=save+quit",
        fontsize=11,
    )

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def backup_csv(path: Path, backup_root: Path) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / path.name
    shutil.copy2(path, destination)
    return destination


def save_changed_splits(
    split_frames: Dict[str, pd.DataFrame],
    split_paths: Dict[str, Path],
    changed_splits: set,
    backup_root: Path,
):
    if not changed_splits:
        print("\nNo label changes to save.")
        return

    print("\nSaving corrected labels...")

    for split_name in sorted(changed_splits):
        path = split_paths[split_name]
        backup = backup_csv(path, backup_root)
        split_frames[split_name].to_csv(path, index=False)
        print(f"  {split_name:5s} updated: {path}")
        print(f"        backup: {backup}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Which existing split to audit. Default: all",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start at this zero-based item in the audit sequence.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root. Default: current working directory.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()

    requested = (
        ["train", "val", "test"]
        if args.split == "all"
        else [args.split]
    )

    split_paths: Dict[str, Path] = {}
    split_frames: Dict[str, pd.DataFrame] = {}
    label_cols: Dict[str, str] = {}

    print("Loading existing locked split CSVs...\n")

    for split_name in requested:
        path = (project_root / DEFAULT_SPLITS[split_name]).resolve()

        if not path.exists():
            raise SystemExit(f"Missing split CSV: {path}")

        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        label_col = find_column(df, LABEL_COLUMNS)

        if label_col is None:
            raise SystemExit(
                f"Could not identify label column in {path}\n"
                f"Columns: {list(df.columns)}"
            )

        split_paths[split_name] = path
        split_frames[split_name] = df
        label_cols[split_name] = label_col

        print(
            f"{split_name:5s}: {len(df):3d} rows | "
            f"label column='{label_col}' | {path}"
        )

    audit_items = []
    for split_name in requested:
        df = split_frames[split_name]
        for row_index in range(len(df)):
            audit_items.append((split_name, row_index))

    print(f"\nAudit items: {len(audit_items)}")
    print("The train/val/test assignment will NOT be regenerated or changed.")
    print("Only label values can be corrected.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = (
        project_root
        / "experiments"
        / "label_audit_backups"
        / timestamp
    )

    corrections = []
    changed_splits = set()

    plt.ion()
    fig = plt.figure(figsize=(8, 5))

    i = max(0, min(args.start, len(audit_items) - 1))

    try:
        while i < len(audit_items):
            split_name, row_index = audit_items[i]
            df = split_frames[split_name]
            row = df.iloc[row_index]
            label_col = label_cols[split_name]

            current_label = normalize_label(row[label_col])

            tile_col = find_column(df, TILE_COLUMNS)

            try:
                tile_image, descriptor = load_tile_image(
                    row=row,
                    csv_path=split_paths[split_name],
                    project_root=project_root,
                    columns=list(df.columns),
                )
            except Exception as exc:
                plt.close(fig)
                print("\nCould not display the current tile.")
                print(exc)
                print("\nTip: paste the CSV column names / first row and I can adapt the loader.")
                sys.exit(1)

            show_tile(
                image=tile_image,
                split_name=split_name,
                index_in_audit=i,
                total=len(audit_items),
                label=current_label,
                descriptor=descriptor,
                row=row,
                tile_col=tile_col,
            )

            answer = input(
                f"[{i + 1}/{len(audit_items)}] "
                f"{split_name} row={row_index} label={current_label} > "
            ).strip()

            if answer.lower() == "q":
                break

            if answer.lower() == "b":
                i = max(0, i - 1)
                continue

            if answer.lower() == "s":
                i += 1
                continue

            if answer == "":
                # Current label confirmed.
                i += 1
                continue

            if len(answer) == 3 and answer.isdigit():
                old = current_label
                new = answer

                if old != new:
                    df.at[row_index, label_col] = new
                    changed_splits.add(split_name)
                    corrections.append(
                        {
                            "split": split_name,
                            "row": row_index,
                            "old_label": old,
                            "new_label": new,
                        }
                    )
                    print(f"  corrected: {old} -> {new}")
                else:
                    print("  same label; kept unchanged.")

                i += 1
                continue

            print("  Invalid input. Press ENTER, type exactly 3 digits, or b/s/q.")

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving completed corrections.")
    finally:
        plt.close("all")

    save_changed_splits(
        split_frames=split_frames,
        split_paths=split_paths,
        changed_splits=changed_splits,
        backup_root=backup_root,
    )

    if corrections:
        corrections_path = backup_root / "corrections.csv"
        pd.DataFrame(corrections).to_csv(corrections_path, index=False)

        print(f"\nCorrections made: {len(corrections)}")
        print(f"Correction log:   {corrections_path}")

        print("\nChanges:")
        for item in corrections:
            print(
                f"  {item['split']:5s} row {item['row']:3d}: "
                f"{item['old_label']} -> {item['new_label']}"
            )
    else:
        print("\nCorrections made: 0")

    print("\nLabel audit complete.")


if __name__ == "__main__":
    main()
