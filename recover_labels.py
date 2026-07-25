from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATES = [
    Path("fusion_predictions.csv"),
    Path("output/predictions.csv"),
    Path("ocr_results.csv"),
    Path("fusion_results.csv"),
    Path("preprocessing_results.csv"),
]

IMAGE_COLUMN_NAMES = [
    "image",
    "image_path",
    "filename",
    "file",
    "tile_path",
    "path",
]

TILE_COLUMN_NAMES = [
    "tile",
    "tile_number",
    "tile_index",
    "index",
]

LABEL_COLUMN_NAMES = [
    "ground_truth",
    "truth",
    "label",
    "target",
    "expected",
]


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    normalized = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def normalize_image(value: object) -> str:
    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    return Path(text).name


def normalize_tile(value: object) -> int | None:
    if pd.isna(value):
        return None

    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None

    if 1 <= number <= 9:
        return number

    return None


def normalize_ground_truth(value: object) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        text = str(int(value)) if float(value).is_integer() else str(value)
    else:
        text = str(value).strip()

    text = text.removesuffix(".0").strip()

    if len(text) == 3 and text.isdigit():
        return text

    return ""


def recover_from_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Missing: {path}")
        return None

    dataframe = pd.read_csv(path, low_memory=False)

    print()
    print(f"===== {path} =====")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {list(dataframe.columns)}")

    image_column = find_column(dataframe, IMAGE_COLUMN_NAMES)
    tile_column = find_column(dataframe, TILE_COLUMN_NAMES)
    label_column = find_column(dataframe, LABEL_COLUMN_NAMES)

    print(f"Image column: {image_column}")
    print(f"Tile column: {tile_column}")
    print(f"Label column: {label_column}")

    if image_column is None or label_column is None:
        print("Cannot recover from this file.")
        return None

    recovered = pd.DataFrame(
        {
            "image": dataframe[image_column].map(normalize_image),
            "ground_truth": dataframe[label_column].map(
                normalize_ground_truth
            ),
        }
    )

    if tile_column is not None:
        recovered["tile"] = dataframe[tile_column].map(normalize_tile)
    else:
        # Try extracting the tile number from names such as tile_4.png.
        extracted_tile = (
            recovered["image"]
            .str.extract(r"tile[_-]?(\d+)", expand=False)
        )
        recovered["tile"] = extracted_tile.map(normalize_tile)

    recovered = recovered[
        (recovered["image"] != "")
        & recovered["tile"].notna()
        & (recovered["ground_truth"] != "")
    ].copy()

    recovered["tile"] = recovered["tile"].astype(int)

    recovered = recovered.drop_duplicates(
        subset=["image", "tile", "ground_truth"]
    )

    conflicting = (
        recovered.groupby(["image", "tile"])["ground_truth"]
        .nunique()
        .loc[lambda values: values > 1]
    )

    if not conflicting.empty:
        print(
            f"Rejected: {len(conflicting)} image/tile pairs "
            "have conflicting ground-truth labels."
        )
        return None

    recovered = recovered.drop_duplicates(
        subset=["image", "tile"],
        keep="first",
    )

    print(f"Recovered unique labels: {len(recovered)}")

    return recovered[["image", "tile", "ground_truth"]]


def main() -> None:
    viable: list[tuple[Path, pd.DataFrame]] = []

    for path in CANDIDATES:
        recovered = recover_from_file(path)

        if recovered is not None:
            viable.append((path, recovered))

    exact = [
        (path, dataframe)
        for path, dataframe in viable
        if len(dataframe) == 513
    ]

    if not exact:
        print()
        print("No candidate produced exactly 513 unique labels.")
        print("Viable results:")

        for path, dataframe in viable:
            print(f"  {path}: {len(dataframe)} labels")

        raise SystemExit(1)

    source_path, labels = exact[0]

    labels = labels.sort_values(
        ["image", "tile"],
        kind="stable",
    ).reset_index(drop=True)

    labels.to_csv("labels.csv", index=False)

    print()
    print(f"Recovered labels from: {source_path}")
    print(f"Created labels.csv with {len(labels)} rows")
    print()
    print(labels.head(12).to_string(index=False))


if __name__ == "__main__":
    main()