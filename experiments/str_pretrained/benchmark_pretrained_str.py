from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODEL = "parseq"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_SPLIT_DIR = Path("experiments/convnext/run_001")
DEFAULT_RUN_DIR = Path("experiments/str_pretrained/parseq_zero_shot")

DIGITS_ONLY = re.compile(r"\D")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def clean_prediction(text: str) -> str:
    return DIGITS_ONLY.sub("", text)


def resolve_tile_path(
    row: dict[str, str],
    output_dir: Path,
) -> Path:
    saved_path = row.get("path", "").strip()

    if saved_path:
        candidate = Path(saved_path)

        if candidate.exists():
            return candidate

    image_name = row["image"]
    tile_number = int(row["tile"])
    stem = Path(image_name).stem

    candidate = (
        output_dir
        / f"{stem}_tile_{tile_number}.png"
    )

    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not locate tile for "
            f"{image_name} tile {tile_number}: "
            f"{candidate}"
        )

    return candidate


def load_split(
    split_path: Path,
    output_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    with split_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "image",
            "tile",
            "ground_truth",
        }

        if not required.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                f"{split_path} must contain "
                f"{sorted(required)}"
            )

        for row in reader:
            truth = row["ground_truth"].strip()

            if (
                len(truth) != 3
                or not truth.isdigit()
            ):
                raise ValueError(
                    f"Invalid ground truth "
                    f"{truth!r} in {split_path}"
                )

            tile_path = resolve_tile_path(
                row,
                output_dir,
            )

            rows.append(
                {
                    "image": row["image"],
                    "tile": int(row["tile"]),
                    "ground_truth": truth,
                    "path": tile_path,
                    "group": row.get(
                        "group",
                        Path(row["image"]).stem,
                    ),
                }
            )

    return rows


class TileDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        transform,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        sample = self.samples[index]

        image = Image.open(
            sample["path"]
        ).convert("RGB")

        image = self.transform(image)

        return (
            image,
            sample["image"],
            sample["tile"],
            sample["ground_truth"],
        )


def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def benchmark(
    model,
    loader,
    device,
) -> tuple[list[dict], dict]:
    model.eval()

    prediction_rows: list[dict] = []

    raw_exact_correct = 0
    cleaned_exact_correct = 0
    total = 0

    for (
        images,
        image_names,
        tile_numbers,
        truths,
    ) in loader:
        images = images.to(device)

        logits = model(images)

        probabilities = logits.softmax(-1)

        labels, confidences = (
            model.tokenizer.decode(
                probabilities
            )
        )

        for index in range(
            len(truths)
        ):
            raw_prediction = str(
                labels[index]
            )

            cleaned_prediction = (
                clean_prediction(
                    raw_prediction
                )
            )

            truth = truths[index]

            raw_correct = (
                raw_prediction == truth
            )

            cleaned_correct = (
                cleaned_prediction == truth
            )

            raw_exact_correct += int(
                raw_correct
            )

            cleaned_exact_correct += int(
                cleaned_correct
            )

            total += 1

            confidence_value = (
                confidences[index]
            )

            if torch.is_tensor(
                confidence_value
            ):
                confidence_value = (
                    confidence_value
                    .detach()
                    .cpu()
                    .float()
                )

                if (
                    confidence_value
                    .numel()
                    == 1
                ):
                    confidence_value = float(
                        confidence_value.item()
                    )
                else:
                    confidence_value = float(
                        confidence_value.mean().item()
                    )

            else:
                confidence_value = float(
                    confidence_value
                )

            prediction_rows.append(
                {
                    "image": image_names[index],
                    "tile": int(
                        tile_numbers[index]
                    ),
                    "ground_truth": truth,
                    "raw_prediction": (
                        raw_prediction
                    ),
                    "cleaned_prediction": (
                        cleaned_prediction
                    ),
                    "raw_correct": (
                        raw_correct
                    ),
                    "cleaned_correct": (
                        cleaned_correct
                    ),
                    "confidence": round(
                        confidence_value,
                        6,
                    ),
                }
            )

    metrics = {
        "total": total,
        "raw_exact_correct": (
            raw_exact_correct
        ),
        "raw_exact_accuracy": (
            raw_exact_correct / total
            if total
            else 0.0
        ),
        "cleaned_exact_correct": (
            cleaned_exact_correct
        ),
        "cleaned_exact_accuracy": (
            cleaned_exact_correct / total
            if total
            else 0.0
        ),
    }

    return prediction_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=[
            "parseq",
            "parseq_tiny",
            "parseq_patch16_224",
            "abinet",
            "trba",
            "crnn",
            "vitstr",
        ],
    )

    parser.add_argument(
        "--eval-split",
        choices=[
            "val",
            "test",
        ],
        default="val",
        help=(
            "Existing locked split to evaluate. "
            "Default: val"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    device = get_device()

    print(
        f"Device: {device}"
    )

    split_filename = (
        f"{args.eval_split}_split.csv"
    )

    split_path = (
        args.split_dir
        / split_filename
    )

    samples = load_split(
        split_path,
        args.output_dir,
    )

    split_display = (
        "Validation"
        if args.eval_split == "val"
        else "Test"
    )

    print(
        f"{split_display} tiles: "
        f"{len(samples)}"
    )

    print()
    print(
        f"Loading pretrained "
        f"{args.model}..."
    )

    model = torch.hub.load(
        "baudm/parseq",
        args.model,
        pretrained=True,
        trust_repo=True,
    )

    model = model.eval().to(
        device
    )

    # Import after torch.hub has downloaded/loaded
    # the PARSeq repository.
    from strhub.data.module import (
        SceneTextDataModule,
    )

    transform = (
        SceneTextDataModule
        .get_transform(
            model.hparams.img_size
        )
    )

    print(
        f"Model image size: "
        f"{model.hparams.img_size}"
    )

    dataset = TileDataset(
        samples,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "Running zero-shot "
        f"{args.eval_split} benchmark..."
    )

    rows, metrics = benchmark(
        model,
        loader,
        device,
    )

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        args.run_dir
        / (
            f"{args.model}_"
            f"{args.eval_split}_predictions.csv"
        )
    )

    save_csv(
        predictions_path,
        rows,
    )

    summary_rows = [
        {
            "model": args.model,
            "split": args.eval_split,
            "tiles": (
                metrics["total"]
            ),
            "raw_exact_correct": (
                metrics[
                    "raw_exact_correct"
                ]
            ),
            "raw_exact_accuracy": round(
                metrics[
                    "raw_exact_accuracy"
                ],
                6,
            ),
            "cleaned_exact_correct": (
                metrics[
                    "cleaned_exact_correct"
                ]
            ),
            "cleaned_exact_accuracy": round(
                metrics[
                    "cleaned_exact_accuracy"
                ],
                6,
            ),
        }
    ]

    summary_path = (
        args.run_dir
        / (
            f"{args.model}_"
            f"{args.eval_split}_summary.csv"
        )
    )

    save_csv(
        summary_path,
        summary_rows,
    )

    print()
    print("=" * 72)
    print(
        f"{args.model.upper()} "
        f"ZERO-SHOT "
        f"{args.eval_split.upper()} RESULT"
    )
    print("=" * 72)

    print(
        "Raw exact accuracy: "
        f"{metrics['raw_exact_accuracy']:.2%} "
        f"("
        f"{metrics['raw_exact_correct']}/"
        f"{metrics['total']}"
        f")"
    )

    print(
        "Digits-only exact accuracy: "
        f"{metrics['cleaned_exact_accuracy']:.2%} "
        f"("
        f"{metrics['cleaned_exact_correct']}/"
        f"{metrics['total']}"
        f")"
    )

    print()
    print(
        f"Predictions: "
        f"{predictions_path}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )

    if args.eval_split == "val":
        print(
            "Locked test set was "
            "NOT evaluated."
        )
    else:
        print(
            "Corrected locked test set "
            "WAS evaluated."
        )


if __name__ == "__main__":
    main()
