from __future__ import annotations

import csv
from pathlib import Path

import cv2
import pandas as pd


RESULTS_PATH = Path("ocr_results.csv")


def save_results(dataframe: pd.DataFrame) -> None:
    dataframe.to_csv(
        RESULTS_PATH,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RESULTS_PATH.resolve()}"
        )

    dataframe = pd.read_csv(
        RESULTS_PATH,
        dtype={
            "image": str,
            "prediction": str,
            "ground_truth": str,
            "correct": str,
            "tile_path": str,
        },
        keep_default_na=False,
    )

    if "ground_truth" not in dataframe.columns:
        dataframe["ground_truth"] = ""

    if "correct" not in dataframe.columns:
        dataframe["correct"] = ""

    total = len(dataframe)

    for index, row in dataframe.iterrows():
        existing_label = str(row["ground_truth"]).strip()

        if existing_label:
            continue

        tile_path = Path(str(row["tile_path"]))

        image = cv2.imread(str(tile_path))

        if image is None:
            print(f"Could not read: {tile_path}")
            continue

        prediction = str(row["prediction"]).strip()

        enlarged = cv2.resize(
            image,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_NEAREST,
        )

        title = (
            f"{index + 1}/{total} | "
            f"{row['image']} | "
            f"tile {row['tile']} | "
            f"prediction: {prediction or '<blank>'}"
        )

        cv2.imshow(title, enlarged)
        cv2.waitKey(1)

        print()
        print(
            f"[{index + 1}/{total}] "
            f"{row['image']} tile {row['tile']}"
        )
        print(f"OCR prediction: {prediction!r}")
        print(
            "Press Enter if correct, type the correct 3 digits, "
            "'s' to skip, or 'q' to quit."
        )

        answer = input("> ").strip()

        cv2.destroyWindow(title)

        if answer.lower() == "q":
            save_results(dataframe)
            print("Progress saved.")
            break

        if answer.lower() == "s":
            continue

        ground_truth = prediction if answer == "" else answer

        if not ground_truth.isdigit() or len(ground_truth) != 3:
            print("Invalid label. It must contain exactly three digits.")
            continue

        dataframe.at[index, "ground_truth"] = ground_truth
        dataframe.at[index, "correct"] = str(
            prediction == ground_truth
        )

        # Save after every answer so progress is never lost.
        save_results(dataframe)

    cv2.destroyAllWindows()

    labeled = dataframe[
        dataframe["ground_truth"].astype(str).str.len() == 3
    ]

    if labeled.empty:
        print("No tiles labeled yet.")
        return

    correct = (
        labeled["prediction"].astype(str)
        == labeled["ground_truth"].astype(str)
    )

    accuracy = correct.mean() * 100

    print()
    print(f"Labeled tiles: {len(labeled)}")
    print(f"Correct predictions: {int(correct.sum())}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
