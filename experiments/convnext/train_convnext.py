from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
IMAGE_SIZE = 224

DEFAULT_LABELS = Path("labels.csv")
DEFAULT_OUTPUT = Path("output")
DEFAULT_RUN_DIR = Path("experiments/convnext/run_001")

SCREENSHOT_RE = re.compile(r"(Screenshot-\d+)")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def screenshot_group(image_name: str) -> str:
    match = SCREENSHOT_RE.search(image_name)

    if not match:
        raise ValueError(
            f"Could not determine screenshot group from {image_name!r}"
        )

    return match.group(1)


def resolve_tile_path(
    row: dict[str, str],
    output_dir: Path,
) -> Path:
    image_name = row["image"]
    tile = int(row["tile"])

    stem = Path(image_name).stem

    return output_dir / f"{stem}_tile_{tile}.png"


def load_samples(
    labels_path: Path,
    output_dir: Path,
) -> list[dict]:
    samples = []

    with labels_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "image",
            "tile",
            "ground_truth",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{labels_path} must contain columns: "
                f"{sorted(required)}"
            )

        for row in reader:
            ground_truth = row["ground_truth"].strip()

            if (
                len(ground_truth) != 3
                or not ground_truth.isdigit()
            ):
                continue

            tile_path = resolve_tile_path(
                row,
                output_dir,
            )

            if not tile_path.exists():
                raise FileNotFoundError(
                    f"Missing tile: {tile_path}"
                )

            samples.append(
                {
                    "image": row["image"],
                    "tile": int(row["tile"]),
                    "path": tile_path,
                    "label": tuple(
                        int(digit)
                        for digit in ground_truth
                    ),
                    "ground_truth": ground_truth,
                    "group": screenshot_group(
                        row["image"]
                    ),
                }
            )

    if not samples:
        raise RuntimeError("No valid labeled samples found.")

    return samples


def grouped_split(
    samples: list[dict],
    seed: int,
):
    groups = [sample["group"] for sample in samples]

    # 80% train+validation, 20% test
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_val_idx, test_idx = next(
        outer.split(
            samples,
            groups=groups,
        )
    )

    train_val = [
        samples[index]
        for index in train_val_idx
    ]

    test = [
        samples[index]
        for index in test_idx
    ]

    train_val_groups = [
        sample["group"]
        for sample in train_val
    ]

    # 20% of remaining data becomes validation.
    # Overall approximately:
    #
    # train = 64%
    # val   = 16%
    # test  = 20%
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed + 1,
    )

    train_idx, val_idx = next(
        inner.split(
            train_val,
            groups=train_val_groups,
        )
    )

    train = [
        train_val[index]
        for index in train_idx
    ]

    val = [
        train_val[index]
        for index in val_idx
    ]

    return train, val, test


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class DigitTileDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        transform,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(
            sample["path"]
        ).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return (
            image,
            label,
            sample["ground_truth"],
            sample["image"],
            sample["tile"],
        )


# ---------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------

def build_transforms():
    weights = ConvNeXt_Tiny_Weights.DEFAULT

    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomAffine(
                degrees=7,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
                shear=3,
            ),

            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.03,
            ),

            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 1.2),
                    )
                ],
                p=0.20,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    return train_transform, eval_transform


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class ConvNeXtThreeDigit(nn.Module):
    def __init__(self):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT

        backbone = convnext_tiny(
            weights=weights
        )

        feature_dim = (
            backbone.classifier[2].in_features
        )

        # Keep ConvNeXt's feature extractor.
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Preserve its final LayerNorm/flatten behavior.
        self.norm = backbone.classifier[0]
        self.flatten = backbone.classifier[1]

        self.dropout = nn.Dropout(p=0.25)

        self.head_1 = nn.Linear(
            feature_dim,
            10,
        )
        self.head_2 = nn.Linear(
            feature_dim,
            10,
        )
        self.head_3 = nn.Linear(
            feature_dim,
            10,
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.dropout(x)

        return (
            self.head_1(x),
            self.head_2(x),
            self.head_3(x),
        )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def predictions_from_logits(outputs):
    predictions = []

    for output in outputs:
        predictions.append(
            output.argmax(dim=1)
        )

    return torch.stack(
        predictions,
        dim=1,
    )


def calculate_batch_metrics(
    predictions,
    targets,
):
    digit_correct = (
        predictions == targets
    )

    exact_correct = (
        digit_correct.all(dim=1)
    )

    return (
        digit_correct.sum().item(),
        exact_correct.sum().item(),
        targets.numel(),
        targets.shape[0],
    )


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------

def compute_loss(
    outputs,
    targets,
    criterion,
):
    return (
        criterion(outputs[0], targets[:, 0])
        + criterion(outputs[1], targets[:, 1])
        + criterion(outputs[2], targets[:, 2])
    )


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    exact_correct = 0
    sample_count = 0

    for images, targets, *_ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = compute_loss(
            outputs,
            targets,
            criterion,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        predictions = predictions_from_logits(
            outputs
        )

        _, batch_exact, _, batch_samples = (
            calculate_batch_metrics(
                predictions,
                targets,
            )
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        exact_correct += batch_exact
        sample_count += batch_samples

    return {
        "loss": total_loss / sample_count,
        "exact_accuracy": (
            exact_correct / sample_count
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    collect_predictions=False,
):
    model.eval()

    total_loss = 0.0
    digit_correct = 0
    exact_correct = 0
    digit_count = 0
    sample_count = 0

    prediction_rows = []

    for (
        images,
        targets,
        truths,
        image_names,
        tile_numbers,
    ) in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        loss = compute_loss(
            outputs,
            targets,
            criterion,
        )

        predictions = predictions_from_logits(
            outputs
        )

        (
            batch_digit_correct,
            batch_exact,
            batch_digits,
            batch_samples,
        ) = calculate_batch_metrics(
            predictions,
            targets,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        digit_correct += batch_digit_correct
        exact_correct += batch_exact
        digit_count += batch_digits
        sample_count += batch_samples

        if collect_predictions:
            probabilities = [
                torch.softmax(output, dim=1)
                for output in outputs
            ]

            predictions_cpu = (
                predictions.cpu()
            )

            for index in range(
                len(truths)
            ):
                predicted_digits = (
                    predictions_cpu[index]
                    .tolist()
                )

                prediction = "".join(
                    str(digit)
                    for digit in predicted_digits
                )

                digit_confidences = []

                for position in range(3):
                    predicted_digit = (
                        predicted_digits[position]
                    )

                    confidence = (
                        probabilities[position][
                            index,
                            predicted_digit,
                        ]
                        .item()
                    )

                    digit_confidences.append(
                        confidence
                    )

                prediction_rows.append(
                    {
                        "image": image_names[index],
                        "tile": int(
                            tile_numbers[index]
                        ),
                        "ground_truth": truths[index],
                        "prediction": prediction,
                        "correct": (
                            prediction
                            == truths[index]
                        ),
                        "confidence_1": round(
                            digit_confidences[0],
                            6,
                        ),
                        "confidence_2": round(
                            digit_confidences[1],
                            6,
                        ),
                        "confidence_3": round(
                            digit_confidences[2],
                            6,
                        ),
                        "min_confidence": round(
                            min(
                                digit_confidences
                            ),
                            6,
                        ),
                    }
                )

    return {
        "loss": total_loss / sample_count,
        "digit_accuracy": (
            digit_correct / digit_count
        ),
        "exact_accuracy": (
            exact_correct / sample_count
        ),
        "correct": exact_correct,
        "total": sample_count,
        "predictions": prediction_rows,
    }


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_split(
    path: Path,
    samples: list[dict],
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "tile",
                "ground_truth",
                "path",
                "group",
            ],
        )

        writer.writeheader()

        for sample in samples:
            writer.writerow(
                {
                    "image": sample["image"],
                    "tile": sample["tile"],
                    "ground_truth": (
                        sample["ground_truth"]
                    ),
                    "path": sample["path"],
                    "group": sample["group"],
                }
            )


def save_predictions(
    path: Path,
    rows: list[dict],
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Device: {device}")

    samples = load_samples(
        args.labels,
        args.output,
    )

    print(
        f"Loaded {len(samples)} labeled tiles."
    )

    train_samples, val_samples, test_samples = (
        grouped_split(
            samples,
            args.seed,
        )
    )

    print()
    print("Dataset split:")
    print(
        f"  Train: {len(train_samples)} tiles / "
        f"{len(set(x['group'] for x in train_samples))} screenshots"
    )
    print(
        f"  Val:   {len(val_samples)} tiles / "
        f"{len(set(x['group'] for x in val_samples))} screenshots"
    )
    print(
        f"  Test:  {len(test_samples)} tiles / "
        f"{len(set(x['group'] for x in test_samples))} screenshots"
    )

    train_groups = {
        x["group"]
        for x in train_samples
    }

    val_groups = {
        x["group"]
        for x in val_samples
    }

    test_groups = {
        x["group"]
        for x in test_samples
    }

    assert train_groups.isdisjoint(
        val_groups
    )
    assert train_groups.isdisjoint(
        test_groups
    )
    assert val_groups.isdisjoint(
        test_groups
    )

    save_split(
        args.run_dir / "train_split.csv",
        train_samples,
    )

    save_split(
        args.run_dir / "val_split.csv",
        val_samples,
    )

    save_split(
        args.run_dir / "test_split.csv",
        test_samples,
    )

    train_transform, eval_transform = (
        build_transforms()
    )

    train_dataset = DigitTileDataset(
        train_samples,
        train_transform,
    )

    val_dataset = DigitTileDataset(
        val_samples,
        eval_transform,
    )

    test_dataset = DigitTileDataset(
        test_samples,
        eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "Loading pretrained ConvNeXt-Tiny..."
    )

    model = ConvNeXtThreeDigit().to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    best_val_accuracy = -1.0
    patience = 10
    epochs_without_improvement = 0

    history = []

    checkpoint_path = (
        args.run_dir / "best_model.pt"
    )

    print()
    print("Training...")
    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_metrics["loss"]
                ),
                "train_exact_accuracy": (
                    train_metrics[
                        "exact_accuracy"
                    ]
                ),
                "val_loss": (
                    val_metrics["loss"]
                ),
                "val_digit_accuracy": (
                    val_metrics[
                        "digit_accuracy"
                    ]
                ),
                "val_exact_accuracy": (
                    val_metrics[
                        "exact_accuracy"
                    ]
                ),
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss "
            f"{train_metrics['loss']:.4f} | "
            f"train exact "
            f"{train_metrics['exact_accuracy']:.2%} | "
            f"val digit "
            f"{val_metrics['digit_accuracy']:.2%} | "
            f"val exact "
            f"{val_metrics['exact_accuracy']:.2%}"
        )

        current_accuracy = (
            val_metrics["exact_accuracy"]
        )

        if (
            current_accuracy
            > best_val_accuracy
        ):
            best_val_accuracy = (
                current_accuracy
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "epoch": epoch,
                    "val_exact_accuracy": (
                        best_val_accuracy
                    ),
                    "seed": args.seed,
                },
                checkpoint_path,
            )

            print(
                "  ↳ new best checkpoint"
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print()
            print(
                "Early stopping: "
                f"no improvement for "
                f"{patience} epochs."
            )
            break

    # Save training history.
    history_path = (
        args.run_dir / "history.csv"
    )

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)

    print()
    print(
        f"Loading best checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Test set is touched exactly once here.
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        collect_predictions=True,
    )

    save_predictions(
        args.run_dir / "test_predictions.csv",
        test_metrics["predictions"],
    )

    print()
    print("=" * 60)
    print("CONVNEXT-TINY FINAL TEST RESULT")
    print("=" * 60)

    print(
        f"Best validation exact accuracy: "
        f"{best_val_accuracy:.2%}"
    )

    print(
        f"Test digit accuracy: "
        f"{test_metrics['digit_accuracy']:.2%}"
    )

    print(
        f"Test exact 3-digit accuracy: "
        f"{test_metrics['exact_accuracy']:.2%}"
    )

    print(
        f"Correct tiles: "
        f"{test_metrics['correct']}/"
        f"{test_metrics['total']}"
    )

    print("=" * 60)

    baseline = 0.9103

    if (
        test_metrics["exact_accuracy"]
        > baseline
    ):
        print(
            "🔥 BEATS EXISTING 91.03% BASELINE"
        )
    else:
        difference = (
            baseline
            - test_metrics[
                "exact_accuracy"
            ]
        )

        print(
            "Does not beat baseline yet. "
            f"Gap: {difference:.2%}"
        )

    print()
    print(
        f"Everything saved under: "
        f"{args.run_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
