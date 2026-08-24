#!/usr/bin/env python3
"""
Prediction-overlap analysis for STR validation benchmarks.

This version is robust to prediction CSVs where a column such as "tile"
contains only the tile position (1..9) and therefore repeats across source
screenshots.

Strategy for row identity:
1. Prefer a genuinely unique ID-like column if one exists.
2. Try useful composite keys (e.g. screenshot/image + tile position).
3. Otherwise use validation row order as the alignment key.
4. Always verify ground truth agrees across models before comparing results.

Usage:
    python experiments/str_pretrained/analyze_prediction_overlap.py

Default prediction directory:
    experiments/str_pretrained/parseq_zero_shot/

Outputs:
    prediction_overlap_by_tile.csv
    prediction_overlap_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_MODELS = {
    "parseq": "parseq_val_predictions.csv",
    "parseq_tiny": "parseq_tiny_val_predictions.csv",
    "parseq_patch16_224": "parseq_patch16_224_val_predictions.csv",
    "vitstr": "vitstr_val_predictions.csv",
    "abinet": "abinet_val_predictions.csv",
    "trba": "trba_val_predictions.csv",
}


# Columns that might be globally unique.
UNIQUE_ID_CANDIDATES = [
    "tile_id",
    "sample_id",
    "uid",
    "id",
    "image_path",
    "path",
    "filename",
    "file",
    "image",
    "sample",
]

# A tile-position column is often only 1..9, so it is NOT assumed unique.
TILE_POSITION_CANDIDATES = [
    "tile",
    "tile_index",
    "tile_idx",
    "position",
    "tile_position",
]

# Columns that might identify the original screenshot/source image.
SOURCE_CANDIDATES = [
    "source",
    "source_image",
    "source_path",
    "screenshot",
    "screenshot_path",
    "captcha",
    "captcha_path",
    "parent",
    "parent_image",
]

GT_CANDIDATES = [
    "ground_truth",
    "groundtruth",
    "gt",
    "target",
    "label",
    "truth",
    "text",
]

PRED_CANDIDATES = [
    "prediction",
    "pred",
    "predicted",
    "raw_prediction",
    "decoded",
    "output",
]

DIGITS_PRED_CANDIDATES = [
    "digits_only_prediction",
    "digits_prediction",
    "prediction_digits_only",
    "pred_digits",
    "digits_only",
]

CONF_CANDIDATES = [
    "confidence",
    "conf",
    "score",
    "mean_confidence",
    "probability",
    "prob",
]


def lower_column_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cmap = lower_column_map(df)
    for candidate in candidates:
        if candidate in cmap:
            return cmap[candidate]
    return None


def normalize_digits(value) -> str:
    if pd.isna(value):
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_unique_nonempty(series: pd.Series) -> bool:
    s = series.astype(str).str.strip()
    return (
        len(s) > 0
        and not (s == "").any()
        and not s.duplicated().any()
    )


def choose_identity(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Return (identity_series, description).

    Important: a column named "tile" is not considered globally unique
    unless combined with a source/screenshot identifier.
    """
    cmap = lower_column_map(df)

    # 1) First look for genuinely unique ID-like columns.
    for candidate in UNIQUE_ID_CANDIDATES:
        col = cmap.get(candidate)
        if col is not None and is_unique_nonempty(df[col]):
            return df[col].astype(str).str.strip(), f"unique column '{col}'"

    # 2) Try source/screenshot + tile-position composites.
    source_col = None
    for candidate in SOURCE_CANDIDATES:
        if candidate in cmap:
            source_col = cmap[candidate]
            break

    tile_col = None
    for candidate in TILE_POSITION_CANDIDATES:
        if candidate in cmap:
            tile_col = cmap[candidate]
            break

    if source_col is not None and tile_col is not None:
        composite = (
            df[source_col].astype(str).str.strip()
            + "::tile="
            + df[tile_col].astype(str).str.strip()
        )
        if is_unique_nonempty(composite):
            return composite, f"composite '{source_col}' + '{tile_col}'"

    # 3) Try pairs of plausible identifier columns.
    possible_cols: List[str] = []
    for candidate in (
        SOURCE_CANDIDATES
        + UNIQUE_ID_CANDIDATES
        + TILE_POSITION_CANDIDATES
    ):
        col = cmap.get(candidate)
        if col is not None and col not in possible_cols:
            possible_cols.append(col)

    for i, col_a in enumerate(possible_cols):
        for col_b in possible_cols[i + 1:]:
            composite = (
                df[col_a].astype(str).str.strip()
                + "::"
                + df[col_b].astype(str).str.strip()
            )
            if is_unique_nonempty(composite):
                return composite, f"composite '{col_a}' + '{col_b}'"

    # 4) Safe fallback for benchmark outputs generated over the exact same
    # validation split/order. Ground-truth verification later protects us
    # from silently aligning unrelated rows.
    row_ids = pd.Series(
        [f"row_{i:03d}" for i in range(len(df))],
        index=df.index,
    )
    return row_ids, "row order fallback"


def detect_schema(
    df: pd.DataFrame,
    model_name: str,
) -> Tuple[str, str, Optional[str]]:
    gt_col = find_column(df, GT_CANDIDATES)
    digits_pred_col = find_column(df, DIGITS_PRED_CANDIDATES)
    raw_pred_col = find_column(df, PRED_CANDIDATES)
    conf_col = find_column(df, CONF_CANDIDATES)

    if gt_col is None:
        raise ValueError(
            f"[{model_name}] Could not identify ground-truth column.\n"
            f"Columns found: {list(df.columns)}"
        )

    pred_col = digits_pred_col or raw_pred_col
    if pred_col is None:
        raise ValueError(
            f"[{model_name}] Could not identify prediction column.\n"
            f"Columns found: {list(df.columns)}"
        )

    return gt_col, pred_col, conf_col


def load_model_csv(
    path: Path,
    model_name: str,
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)

    gt_col, pred_col, conf_col = detect_schema(df, model_name)
    identity, identity_description = choose_identity(df)

    out = pd.DataFrame(index=df.index)
    out["tile_id"] = identity.astype(str)
    out["ground_truth_raw"] = df[gt_col].map(normalize_text)
    out["ground_truth"] = df[gt_col].map(normalize_digits)
    out[f"{model_name}_prediction_raw"] = df[pred_col].map(normalize_text)
    out[f"{model_name}_prediction"] = df[pred_col].map(normalize_digits)
    out[f"{model_name}_correct"] = (
        out[f"{model_name}_prediction"] == out["ground_truth"]
    )

    if conf_col is not None:
        out[f"{model_name}_confidence"] = pd.to_numeric(
            df[conf_col],
            errors="coerce",
        )

    if out["tile_id"].duplicated().any():
        # This should only happen if an unexpected schema defeats the identity
        # logic above.
        raise ValueError(
            f"[{model_name}] Internal error: chosen identity is still duplicated."
        )

    return out, identity_description


def merge_models(
    frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge models by tile_id and verify that the ground truth is identical.

    If all files had to fall back to row-order IDs, this still gives a safe
    alignment as long as they were benchmarked on the same ordered validation
    split. The ground-truth check detects most accidental misalignment.
    """
    merged: Optional[pd.DataFrame] = None

    for model_name, frame in frames.items():
        if merged is None:
            merged = frame.copy()
            continue

        gt_check = frame[["tile_id", "ground_truth"]].rename(
            columns={"ground_truth": f"{model_name}_ground_truth_check"}
        )

        model_specific_cols = [
            c
            for c in frame.columns
            if c not in {"ground_truth_raw", "ground_truth"}
        ]

        merged = merged.merge(
            frame[model_specific_cols],
            on="tile_id",
            how="outer",
            validate="one_to_one",
        )

        merged = merged.merge(
            gt_check,
            on="tile_id",
            how="left",
            validate="one_to_one",
        )

        check_col = f"{model_name}_ground_truth_check"

        mismatch = (
            merged[check_col].notna()
            & merged["ground_truth"].notna()
            & (merged[check_col] != merged["ground_truth"])
        )

        if mismatch.any():
            bad = merged.loc[
                mismatch,
                ["tile_id", "ground_truth", check_col],
            ]

            raise ValueError(
                "\nGround-truth mismatch detected while aligning "
                f"{model_name}.\n"
                "This means the prediction CSVs are not in the same tile "
                "order or do not come from the same validation split.\n\n"
                f"{bad.head(10).to_string(index=False)}"
            )

        merged.drop(columns=[check_col], inplace=True)

    assert merged is not None
    return merged


def add_overlap_columns(
    df: pd.DataFrame,
    model_names: List[str],
) -> pd.DataFrame:
    correct_cols = [f"{m}_correct" for m in model_names]

    for col in correct_cols:
        df[col] = df[col].fillna(False).astype(bool)

    df["num_models_correct"] = df[correct_cols].sum(axis=1)
    df["all_models_correct"] = (
        df["num_models_correct"] == len(model_names)
    )
    df["all_models_wrong"] = df["num_models_correct"] == 0
    df["at_least_one_correct"] = df["num_models_correct"] > 0

    def correct_model_names(row) -> str:
        return ",".join(
            m for m in model_names if bool(row[f"{m}_correct"])
        )

    def wrong_model_names(row) -> str:
        return ",".join(
            m for m in model_names if not bool(row[f"{m}_correct"])
        )

    df["correct_models"] = df.apply(correct_model_names, axis=1)
    df["wrong_models"] = df.apply(wrong_model_names, axis=1)

    pred_cols = [f"{m}_prediction" for m in model_names]
    df["unique_prediction_count"] = df[pred_cols].nunique(
        axis=1,
        dropna=True,
    )
    df["models_disagree"] = df["unique_prediction_count"] > 1

    return df


def print_model_summary(
    df: pd.DataFrame,
    model_names: List[str],
) -> None:
    total = len(df)

    print("\n" + "=" * 80)
    print("MODEL ACCURACY")
    print("=" * 80)

    ranking = []
    for model in model_names:
        correct = int(df[f"{model}_correct"].sum())
        accuracy = correct / total * 100 if total else 0.0
        ranking.append((correct, accuracy, model))

    ranking.sort(reverse=True)

    for correct, accuracy, model in ranking:
        print(
            f"{model:22s} "
            f"{correct:3d}/{total:<3d} = {accuracy:6.2f}%"
        )

    any_correct = int(df["at_least_one_correct"].sum())
    all_wrong = int(df["all_models_wrong"].sum())
    disagreements = int(df["models_disagree"].sum())

    print("\n" + "=" * 80)
    print("COLLECTIVE COVERAGE")
    print("=" * 80)
    print(
        f"At least one model correct: "
        f"{any_correct}/{total} = {any_correct / total * 100:.2f}%"
    )
    print(f"All models wrong:          {all_wrong}/{total}")
    print(f"Prediction disagreements: {disagreements}/{total}")

    if any_correct == total:
        print(
            "\n*** IMPORTANT: The model pool collectively covers "
            "EVERY validation tile. ***"
        )
        print(
            "*** Perfect-oracle ensemble ceiling = "
            "100% validation accuracy.         ***"
        )
    else:
        print(
            f"\nPerfect-oracle ensemble ceiling: "
            f"{any_correct}/{total} = "
            f"{any_correct / total * 100:.2f}%"
        )


def print_wrong_tiles(
    df: pd.DataFrame,
    model: str,
) -> None:
    wrong = df[~df[f"{model}_correct"]]

    print(f"\n{model} wrong tiles: {len(wrong)}")

    if wrong.empty:
        return

    cols = [
        "tile_id",
        "ground_truth",
        f"{model}_prediction",
        "correct_models",
    ]

    conf_col = f"{model}_confidence"
    if conf_col in df.columns:
        cols.insert(3, conf_col)

    print(wrong[cols].to_string(index=False))


def print_key_error_analysis(
    df: pd.DataFrame,
    model_names: List[str],
) -> None:
    print("\n" + "=" * 80)
    print("KEY ERROR OVERLAP")
    print("=" * 80)

    preferred = [
        "parseq_tiny",
        "parseq",
        "parseq_patch16_224",
        "vitstr",
    ]
    preferred = [m for m in preferred if m in model_names]

    for model in preferred:
        print_wrong_tiles(df, model)

    if "parseq_tiny" in model_names:
        tiny_wrong = df[~df["parseq_tiny_correct"]].copy()

        print("\n" + "-" * 80)
        print("WHO FIXES PARSEQ-TINY'S ERRORS?")
        print("-" * 80)

        if tiny_wrong.empty:
            print("PARSeq-Tiny is already perfect on validation.")
        else:
            show_cols = [
                "tile_id",
                "ground_truth",
                "parseq_tiny_prediction",
            ]

            if "parseq_tiny_confidence" in tiny_wrong.columns:
                show_cols.append("parseq_tiny_confidence")

            for model in model_names:
                if model == "parseq_tiny":
                    continue

                show_cols.extend(
                    [
                        f"{model}_prediction",
                        f"{model}_correct",
                    ]
                )

                conf_col = f"{model}_confidence"
                if conf_col in tiny_wrong.columns:
                    show_cols.append(conf_col)

            print(tiny_wrong[show_cols].to_string(index=False))

    strongest = [
        m
        for m in [
            "parseq",
            "parseq_tiny",
            "parseq_patch16_224",
            "vitstr",
        ]
        if m in model_names
    ]

    if strongest:
        strong_correct_cols = [f"{m}_correct" for m in strongest]
        strong_all_wrong = df[
            ~df[strong_correct_cols].any(axis=1)
        ]

        print("\n" + "-" * 80)
        print("ALL FOUR STRONG MODELS WRONG")
        print("-" * 80)

        if strong_all_wrong.empty:
            print(
                "None. The strongest models collectively cover "
                "all validation tiles."
            )
        else:
            cols = ["tile_id", "ground_truth"] + [
                f"{m}_prediction" for m in strongest
            ]
            print(strong_all_wrong[cols].to_string(index=False))


def build_pairwise_summary(
    df: pd.DataFrame,
    model_names: List[str],
) -> pd.DataFrame:
    rows = []

    for i, a in enumerate(model_names):
        for b in model_names[i + 1:]:
            a_correct = df[f"{a}_correct"]
            b_correct = df[f"{b}_correct"]

            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "a_correct_b_wrong": int(
                        (a_correct & ~b_correct).sum()
                    ),
                    "a_wrong_b_correct": int(
                        (~a_correct & b_correct).sum()
                    ),
                    "both_correct": int(
                        (a_correct & b_correct).sum()
                    ),
                    "both_wrong": int(
                        (~a_correct & ~b_correct).sum()
                    ),
                    "prediction_disagreements": int(
                        (
                            df[f"{a}_prediction"]
                            != df[f"{b}_prediction"]
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path(
            "experiments/str_pretrained/parseq_zero_shot"
        ),
        help="Directory containing validation prediction CSVs.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --pred-dir.",
    )

    args = parser.parse_args()

    pred_dir = args.pred_dir
    output_dir = args.output_dir or pred_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {}
    missing = []

    print("")

    for model, filename in DEFAULT_MODELS.items():
        path = pred_dir / filename

        if not path.exists():
            missing.append((model, path))
            continue

        print(f"Loading {model}: {path}")

        frame, identity_description = load_model_csv(
            path,
            model,
        )

        print(f"  identity: {identity_description}")
        print(f"  rows:     {len(frame)}")

        frames[model] = frame

    if not frames:
        raise SystemExit(
            f"No prediction CSVs found under: {pred_dir}\n"
            "Check --pred-dir or the expected filenames."
        )

    model_names = list(frames.keys())

    print(f"\nModels included: {', '.join(model_names)}")

    if missing:
        print("\nMissing optional prediction files:")
        for model, path in missing:
            print(f"  - {model}: {path}")

    lengths = {m: len(frame) for m, frame in frames.items()}
    if len(set(lengths.values())) != 1:
        print("\nWARNING: Prediction CSV row counts differ:")
        for model, count in lengths.items():
            print(f"  {model}: {count}")

    merged = merge_models(frames)

    required_pred_cols = [
        f"{m}_prediction" for m in model_names
    ]

    before = len(merged)

    merged = merged.dropna(
        subset=required_pred_cols
    ).copy()

    after = len(merged)

    if before != after:
        print(
            f"\nWARNING: Dropped {before - after} rows not shared "
            "across every included model."
        )

    merged = add_overlap_columns(
        merged,
        model_names,
    )

    print_model_summary(
        merged,
        model_names,
    )

    print_key_error_analysis(
        merged,
        model_names,
    )

    pairwise = build_pairwise_summary(
        merged,
        model_names,
    )

    by_tile_path = (
        output_dir / "prediction_overlap_by_tile.csv"
    )

    pairwise_path = (
        output_dir / "prediction_overlap_summary.csv"
    )

    merged.to_csv(
        by_tile_path,
        index=False,
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("PAIRWISE COMPLEMENTARITY")
    print("=" * 80)

    if pairwise.empty:
        print("Only one model available.")
    else:
        print(pairwise.to_string(index=False))

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(by_tile_path)
    print(pairwise_path)

    print("\nPrimary question:")
    print(
        "Do PARSeq, PARSeq-Tiny, Patch16-224, and ViTSTR "
        "collectively cover 81/81?"
    )


if __name__ == "__main__":
    main()
