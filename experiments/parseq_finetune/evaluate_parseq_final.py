from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_SPLIT_DIR = Path("experiments/convnext/run_001")
DEFAULT_FINETUNE_RUN = Path("experiments/parseq_finetune/run_001")
DEFAULT_RUN_DIR = Path("experiments/parseq_final_test")

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
            f"Missing tile for {image_name} "
            f"tile {tile_number}: {candidate}"
        )

    return candidate


def load_split(
    split_path: Path,
    output_dir: Path,
) -> list[dict]:
    samples: list[dict] = []

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

            samples.append(
                {
                    "image": row["image"],
                    "tile": int(row["tile"]),
                    "ground_truth": truth,
                    "path": resolve_tile_path(
                        row,
                        output_dir,
                    ),
                }
            )

    return samples


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


def parse_confidence(value) -> float:
    if torch.is_tensor(value):
        value = (
            value
            .detach()
            .cpu()
            .float()
        )

        if value.numel() == 1:
            return float(
                value.item()
            )

        return float(
            value.mean().item()
        )

    return float(value)


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
) -> dict:
    model.eval()

    rows: list[dict] = []

    correct = 0
    total = 0
    confidence_sum = 0.0

    for (
        images,
        image_names,
        tile_numbers,
        truths,
    ) in loader:
        images = images.to(device)

        logits = model(
            images,
            max_length=3,
        )

        probabilities = (
            logits.softmax(-1)
        )

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

            prediction = clean_prediction(
                raw_prediction
            )

            truth = truths[index]

            is_correct = (
                prediction == truth
            )

            confidence = (
                parse_confidence(
                    confidences[index]
                )
            )

            correct += int(
                is_correct
            )

            total += 1

            confidence_sum += (
                confidence
            )

            rows.append(
                {
                    "image": (
                        image_names[index]
                    ),
                    "tile": int(
                        tile_numbers[index]
                    ),
                    "ground_truth": truth,
                    "raw_prediction": (
                        raw_prediction
                    ),
                    "prediction": prediction,
                    "correct": (
                        is_correct
                    ),
                    "confidence": round(
                        confidence,
                        6,
                    ),
                }
            )

    return {
        "accuracy": (
            correct / total
            if total
            else 0.0
        ),
        "correct": correct,
        "total": total,
        "mean_confidence": (
            confidence_sum / total
            if total
            else 0.0
        ),
        "rows": rows,
    }


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


def load_parseq(device):
    model = torch.hub.load(
        "baudm/parseq",
        "parseq",
        pretrained=True,
        trust_repo=True,
    )

    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()

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
        "--finetune-run",
        type=Path,
        default=DEFAULT_FINETUNE_RUN,
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

    test_samples = load_split(
        args.split_dir
        / "test_split.csv",
        args.output_dir,
    )

    print(
        f"Locked test tiles: "
        f"{len(test_samples)}"
    )

    print()
    print(
        "Loading zero-shot PARSeq..."
    )

    zero_shot_model = (
        load_parseq(device)
    )

    from strhub.data.module import (
        SceneTextDataModule,
    )

    transform = (
        SceneTextDataModule
        .get_transform(
            zero_shot_model
            .hparams
            .img_size
        )
    )

    loader = DataLoader(
        TileDataset(
            test_samples,
            transform,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Evaluating zero-shot..."
    )

    zero_metrics = evaluate(
        zero_shot_model,
        loader,
        device,
    )

    # Free the first instance before creating the second.
    del zero_shot_model

    if device.type == "mps":
        torch.mps.empty_cache()

    print()
    print(
        "Loading fine-tuned PARSeq..."
    )

    finetuned_model = (
        load_parseq(device)
    )

    checkpoint_path = (
        args.finetune_run
        / "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing fine-tuned checkpoint: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    finetuned_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "Evaluating fine-tuned..."
    )

    fine_metrics = evaluate(
        finetuned_model,
        loader,
        device,
    )

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        args.run_dir
        / "zero_shot_test_predictions.csv",
        zero_metrics["rows"],
    )

    save_csv(
        args.run_dir
        / "finetuned_test_predictions.csv",
        fine_metrics["rows"],
    )

    comparison_rows: list[dict] = []

    zero_map = {
        (
            row["image"],
            row["tile"],
        ): row
        for row in zero_metrics["rows"]
    }

    fine_map = {
        (
            row["image"],
            row["tile"],
        ): row
        for row in fine_metrics["rows"]
    }

    for key in zero_map:
        zero_row = zero_map[key]
        fine_row = fine_map[key]

        comparison_rows.append(
            {
                "image": (
                    zero_row["image"]
                ),
                "tile": (
                    zero_row["tile"]
                ),
                "ground_truth": (
                    zero_row[
                        "ground_truth"
                    ]
                ),
                "zero_shot_prediction": (
                    zero_row[
                        "prediction"
                    ]
                ),
                "zero_shot_correct": (
                    zero_row[
                        "correct"
                    ]
                ),
                "zero_shot_confidence": (
                    zero_row[
                        "confidence"
                    ]
                ),
                "finetuned_prediction": (
                    fine_row[
                        "prediction"
                    ]
                ),
                "finetuned_correct": (
                    fine_row[
                        "correct"
                    ]
                ),
                "finetuned_confidence": (
                    fine_row[
                        "confidence"
                    ]
                ),
                "prediction_changed": (
                    zero_row[
                        "prediction"
                    ]
                    != fine_row[
                        "prediction"
                    ]
                ),
            }
        )

    save_csv(
        args.run_dir
        / "comparison.csv",
        comparison_rows,
    )

    summary = [
        {
            "model": "parseq_zero_shot",
            "correct": (
                zero_metrics["correct"]
            ),
            "total": (
                zero_metrics["total"]
            ),
            "accuracy": (
                zero_metrics["accuracy"]
            ),
            "mean_confidence": (
                zero_metrics[
                    "mean_confidence"
                ]
            ),
        },
        {
            "model": "parseq_finetuned",
            "correct": (
                fine_metrics["correct"]
            ),
            "total": (
                fine_metrics["total"]
            ),
            "accuracy": (
                fine_metrics["accuracy"]
            ),
            "mean_confidence": (
                fine_metrics[
                    "mean_confidence"
                ]
            ),
        },
    ]

    save_csv(
        args.run_dir
        / "summary.csv",
        summary,
    )

    changed = sum(
        int(
            row[
                "prediction_changed"
            ]
        )
        for row in comparison_rows
    )

    print()
    print("=" * 72)
    print(
        "FINAL LOCKED TEST — PARSEQ"
    )
    print("=" * 72)

    print(
        "Zero-shot: "
        f"{zero_metrics['accuracy']:.2%} "
        f"("
        f"{zero_metrics['correct']}/"
        f"{zero_metrics['total']}"
        f") | conf "
        f"{zero_metrics['mean_confidence']:.4f}"
    )

    print(
        "Fine-tuned: "
        f"{fine_metrics['accuracy']:.2%} "
        f"("
        f"{fine_metrics['correct']}/"
        f"{fine_metrics['total']}"
        f") | conf "
        f"{fine_metrics['mean_confidence']:.4f}"
    )

    print(
        f"Predictions changed on "
        f"{changed}/"
        f"{len(comparison_rows)} "
        f"test tiles."
    )

    print()
    print(
        "This locked test set should now "
        "be considered consumed for "
        "PARSeq model selection."
    )


if __name__ == "__main__":
    main()
