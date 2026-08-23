from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


SEED = 42
IMAGE_SIZE = 160

DEFAULT_OUTPUT = Path("output")
DEFAULT_SPLIT_DIR = Path("experiments/convnext/run_001")
DEFAULT_RUN_DIR = Path("experiments/digit_resnet/run_001")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def locate_tile(
    image_name: str,
    tile_number: int,
    output_dir: Path,
    saved_path: str,
) -> Path:
    if saved_path:
        candidate = Path(saved_path)

        if candidate.exists():
            return candidate

    stem = Path(image_name).stem
    candidate = (
        output_dir
        / f"{stem}_tile_{tile_number}.png"
    )

    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing tile: {candidate}"
        )

    return candidate


def load_tile_split(
    path: Path,
    output_dir: Path,
) -> list[dict]:
    samples: list[dict] = []

    with path.open(
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
                f"{path} must contain "
                f"{sorted(required)}"
            )

        for row in reader:
            truth = (
                row["ground_truth"]
                .strip()
            )

            if (
                len(truth) != 3
                or not truth.isdigit()
            ):
                raise ValueError(
                    f"Invalid truth "
                    f"{truth!r} in {path}"
                )

            tile_number = int(
                row["tile"]
            )

            tile_path = locate_tile(
                row["image"],
                tile_number,
                output_dir,
                row.get(
                    "path",
                    "",
                ).strip(),
            )

            samples.append(
                {
                    "image": row["image"],
                    "tile": tile_number,
                    "path": tile_path,
                    "ground_truth": truth,
                    "group": row.get(
                        "group",
                        Path(
                            row["image"]
                        ).stem,
                    ),
                }
            )

    return samples


def crop_digit_region(
    image: Image.Image,
    position: int,
) -> Image.Image:
    """
    Split a tile into three horizontal digit regions.

    A small overlap is kept between adjacent thirds so digits near a
    boundary are not accidentally clipped.
    """
    width, height = image.size

    third = width / 3.0
    overlap = int(
        round(width * 0.035)
    )

    left = int(
        round(
            position * third
        )
    )

    right = int(
        round(
            (position + 1)
            * third
        )
    )

    if position > 0:
        left -= overlap

    if position < 2:
        right += overlap

    left = max(
        0,
        left,
    )

    right = min(
        width,
        right,
    )

    return image.crop(
        (
            left,
            0,
            right,
            height,
        )
    )


class DigitCropDataset(Dataset):
    def __init__(
        self,
        tile_samples: list[dict],
        transform,
    ):
        self.transform = transform

        self.samples: list[dict] = []

        for tile_sample in tile_samples:
            truth = (
                tile_sample[
                    "ground_truth"
                ]
            )

            for position in range(3):
                self.samples.append(
                    {
                        **tile_sample,
                        "position": position,
                        "digit": int(
                            truth[position]
                        ),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        sample = (
            self.samples[index]
        )

        image = Image.open(
            sample["path"]
        ).convert("RGB")

        digit_image = (
            crop_digit_region(
                image,
                sample["position"],
            )
        )

        digit_image = (
            self.transform(
                digit_image
            )
        )

        return (
            digit_image,
            torch.tensor(
                sample["digit"],
                dtype=torch.long,
            ),
            sample["image"],
            sample["tile"],
            sample["position"],
            sample[
                "ground_truth"
            ],
        )


def build_transforms():
    weights = (
        ResNet18_Weights.DEFAULT
    )

    preprocessing = (
        weights.transforms()
    )

    train_transform = (
        transforms.Compose(
            [
                transforms.Resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    )
                ),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(
                        0.04,
                        0.04,
                    ),
                    scale=(
                        0.94,
                        1.06,
                    ),
                    shear=2,
                    fill=255,
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.06,
                    hue=0.015,
                ),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=3,
                            sigma=(
                                0.1,
                                0.7,
                            ),
                        )
                    ],
                    p=0.10,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(
                        preprocessing.mean
                    ),
                    std=(
                        preprocessing.std
                    ),
                ),
            ]
        )
    )

    eval_transform = (
        transforms.Compose(
            [
                transforms.Resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(
                        preprocessing.mean
                    ),
                    std=(
                        preprocessing.std
                    ),
                ),
            ]
        )
    )

    return (
        train_transform,
        eval_transform,
    )


def build_model() -> nn.Module:
    model = resnet18(
        weights=(
            ResNet18_Weights.DEFAULT
        )
    )

    in_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(
        nn.Dropout(
            p=0.20
        ),
        nn.Linear(
            in_features,
            10,
        ),
    )

    return model


def freeze_backbone(
    model: nn.Module,
) -> None:
    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    for parameter in (
        model.fc.parameters()
    ):
        parameter.requires_grad = True


def unfreeze_late_layers(
    model: nn.Module,
) -> None:
    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    for block in [
        model.layer3,
        model.layer4,
        model.fc,
    ]:
        for parameter in (
            block.parameters()
        ):
            parameter.requires_grad = True


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for (
        images,
        targets,
        *_,
    ) in loader:
        images = images.to(
            device
        )

        targets = targets.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p
                in model.parameters()
                if p.requires_grad
            ],
            max_norm=1.0,
        )

        optimizer.step()

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        batch_size = (
            targets.shape[0]
        )

        loss_sum += (
            loss.item()
            * batch_size
        )

        correct += (
            predictions.eq(
                targets
            )
            .sum()
            .item()
        )

        total += (
            batch_size
        )

    return {
        "loss": (
            loss_sum / total
        ),
        "digit_accuracy": (
            correct / total
        ),
    }


@torch.no_grad()
def evaluate_digits(
    model,
    loader,
    criterion,
    device,
    collect: bool = False,
):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    rows: list[dict] = []

    for (
        images,
        targets,
        image_names,
        tile_numbers,
        positions,
        truths,
    ) in loader:
        images = images.to(
            device
        )

        targets = targets.to(
            device
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            targets,
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        batch_size = (
            targets.shape[0]
        )

        loss_sum += (
            loss.item()
            * batch_size
        )

        correct += (
            predictions.eq(
                targets
            )
            .sum()
            .item()
        )

        total += (
            batch_size
        )

        if collect:
            predictions_cpu = (
                predictions.cpu()
            )

            probabilities_cpu = (
                probabilities.cpu()
            )

            for index in range(
                batch_size
            ):
                predicted = int(
                    predictions_cpu[
                        index
                    ]
                )

                confidence = float(
                    probabilities_cpu[
                        index,
                        predicted,
                    ]
                )

                rows.append(
                    {
                        "image": (
                            image_names[
                                index
                            ]
                        ),
                        "tile": int(
                            tile_numbers[
                                index
                            ]
                        ),
                        "position": int(
                            positions[
                                index
                            ]
                        ),
                        "ground_truth": (
                            truths[index]
                        ),
                        "digit_truth": int(
                            targets[
                                index
                            ].cpu()
                        ),
                        "digit_prediction": (
                            predicted
                        ),
                        "correct": (
                            predicted
                            == int(
                                targets[
                                    index
                                ].cpu()
                            )
                        ),
                        "confidence": round(
                            confidence,
                            6,
                        ),
                    }
                )

    return {
        "loss": (
            loss_sum / total
        ),
        "digit_accuracy": (
            correct / total
        ),
        "correct": correct,
        "total": total,
        "rows": rows,
    }


def reconstruct_tiles(
    digit_rows: list[dict],
) -> dict:
    grouped: dict[
        tuple[str, int],
        dict,
    ] = {}

    for row in digit_rows:
        key = (
            row["image"],
            row["tile"],
        )

        group = (
            grouped.setdefault(
                key,
                {
                    "image": (
                        row["image"]
                    ),
                    "tile": (
                        row["tile"]
                    ),
                    "ground_truth": (
                        row[
                            "ground_truth"
                        ]
                    ),
                    "digits": {},
                    "confidences": {},
                },
            )
        )

        position = int(
            row["position"]
        )

        group["digits"][
            position
        ] = str(
            row[
                "digit_prediction"
            ]
        )

        group[
            "confidences"
        ][position] = float(
            row["confidence"]
        )

    tile_rows = []

    for group in (
        grouped.values()
    ):
        prediction = "".join(
            group["digits"][
                position
            ]
            for position
            in range(3)
        )

        confidences = [
            group[
                "confidences"
            ][position]
            for position
            in range(3)
        ]

        tile_rows.append(
            {
                "image": (
                    group["image"]
                ),
                "tile": (
                    group["tile"]
                ),
                "ground_truth": (
                    group[
                        "ground_truth"
                    ]
                ),
                "prediction": (
                    prediction
                ),
                "correct": (
                    prediction
                    == group[
                        "ground_truth"
                    ]
                ),
                "min_confidence": round(
                    min(
                        confidences
                    ),
                    6,
                ),
                "mean_confidence": round(
                    sum(
                        confidences
                    )
                    / 3.0,
                    6,
                ),
            }
        )

    exact_correct = sum(
        int(
            row["correct"]
        )
        for row
        in tile_rows
    )

    total = len(
        tile_rows
    )

    return {
        "rows": tile_rows,
        "exact_correct": (
            exact_correct
        ),
        "total": total,
        "exact_accuracy": (
            exact_correct
            / total
            if total
            else 0.0
        ),
    }


def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    stage: str,
    val_digit_accuracy: float,
    val_exact_accuracy: float,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "epoch": epoch,
            "stage": stage,
            "val_digit_accuracy": (
                val_digit_accuracy
            ),
            "val_exact_accuracy": (
                val_exact_accuracy
            ),
            "val_loss": val_loss,
        },
        path,
    )


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
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
        default=32,
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    seed_everything(
        args.seed
    )

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        get_device()
    )

    print(
        f"Device: {device}"
    )

    train_tiles = (
        load_tile_split(
            args.split_dir
            / "train_split.csv",
            args.output,
        )
    )

    val_tiles = (
        load_tile_split(
            args.split_dir
            / "val_split.csv",
            args.output,
        )
    )

    test_tiles = (
        load_tile_split(
            args.split_dir
            / "test_split.csv",
            args.output,
        )
    )

    print()
    print(
        "Locked tile split:"
    )
    print(
        f"  Train tiles: "
        f"{len(train_tiles)} "
        f"→ digits: "
        f"{len(train_tiles) * 3}"
    )
    print(
        f"  Val tiles:   "
        f"{len(val_tiles)} "
        f"→ digits: "
        f"{len(val_tiles) * 3}"
    )
    print(
        f"  Test tiles:  "
        f"{len(test_tiles)} "
        f"→ digits: "
        f"{len(test_tiles) * 3}"
    )

    (
        train_transform,
        eval_transform,
    ) = build_transforms()

    train_loader = DataLoader(
        DigitCropDataset(
            train_tiles,
            train_transform,
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        DigitCropDataset(
            val_tiles,
            eval_transform,
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        DigitCropDataset(
            test_tiles,
            eval_transform,
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "Loading pretrained "
        "ResNet18 digit classifier..."
    )

    model = (
        build_model()
        .to(device)
    )

    criterion = (
        nn.CrossEntropyLoss(
            label_smoothing=0.02
        )
    )

    checkpoint_path = (
        args.run_dir
        / "best_model.pt"
    )

    history: list[dict] = []

    best_digit_accuracy = -1.0
    best_exact_accuracy = -1.0
    best_val_loss = float(
        "inf"
    )

    best_epoch = 0
    global_epoch = 0

    print()
    print("=" * 72)
    print(
        "STAGE 1 — DIGIT HEAD WARMUP"
    )
    print("=" * 72)

    freeze_backbone(
        model
    )

    optimizer = (
        torch.optim.AdamW(
            model.fc.parameters(),
            lr=(
                args.warmup_lr
            ),
            weight_decay=1e-4,
        )
    )

    for stage_epoch in range(
        1,
        args.warmup_epochs + 1,
    ):
        global_epoch += 1

        train_metrics = (
            train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
        )

        val_metrics = (
            evaluate_digits(
                model,
                val_loader,
                criterion,
                device,
                collect=True,
            )
        )

        val_tiles_result = (
            reconstruct_tiles(
                val_metrics["rows"]
            )
        )

        row = {
            "epoch": (
                global_epoch
            ),
            "stage": (
                "head_warmup"
            ),
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_digit_accuracy": (
                train_metrics[
                    "digit_accuracy"
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
                val_tiles_result[
                    "exact_accuracy"
                ]
            ),
        }

        history.append(
            row
        )

        print(
            f"Epoch "
            f"{global_epoch:02d} | "
            f"train digit "
            f"{train_metrics['digit_accuracy']:.2%} | "
            f"val digit "
            f"{val_metrics['digit_accuracy']:.2%} | "
            f"val exact "
            f"{val_tiles_result['exact_accuracy']:.2%} | "
            f"val loss "
            f"{val_metrics['loss']:.4f}"
        )

        improved = (
            val_metrics[
                "digit_accuracy"
            ]
            > best_digit_accuracy
            or (
                val_metrics[
                    "digit_accuracy"
                ]
                == best_digit_accuracy
                and val_tiles_result[
                    "exact_accuracy"
                ]
                > best_exact_accuracy
            )
            or (
                val_metrics[
                    "digit_accuracy"
                ]
                == best_digit_accuracy
                and val_tiles_result[
                    "exact_accuracy"
                ]
                == best_exact_accuracy
                and val_metrics[
                    "loss"
                ]
                < best_val_loss
            )
        )

        if improved:
            best_digit_accuracy = (
                val_metrics[
                    "digit_accuracy"
                ]
            )

            best_exact_accuracy = (
                val_tiles_result[
                    "exact_accuracy"
                ]
            )

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = (
                global_epoch
            )

            save_checkpoint(
                checkpoint_path,
                model,
                global_epoch,
                "head_warmup",
                best_digit_accuracy,
                best_exact_accuracy,
                best_val_loss,
            )

            print(
                "  ↳ new best "
                "checkpoint"
            )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()
    print("=" * 72)
    print(
        "STAGE 2 — LATE RESNET FINE-TUNING"
    )
    print("=" * 72)

    unfreeze_late_layers(
        model
    )

    backbone_parameters = []

    classifier_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if (
            not parameter.requires_grad
        ):
            continue

        if name.startswith(
            "fc."
        ):
            classifier_parameters.append(
                parameter
            )
        else:
            backbone_parameters.append(
                parameter
            )

    optimizer = (
        torch.optim.AdamW(
            [
                {
                    "params": (
                        backbone_parameters
                    ),
                    "lr": (
                        args.backbone_lr
                    ),
                },
                {
                    "params": (
                        classifier_parameters
                    ),
                    "lr": (
                        args.head_lr
                    ),
                },
            ],
            weight_decay=1e-4,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=(
                args.finetune_epochs
            ),
        )
    )

    no_improvement = 0

    for stage_epoch in range(
        1,
        args.finetune_epochs + 1,
    ):
        global_epoch += 1

        train_metrics = (
            train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
        )

        val_metrics = (
            evaluate_digits(
                model,
                val_loader,
                criterion,
                device,
                collect=True,
            )
        )

        val_tiles_result = (
            reconstruct_tiles(
                val_metrics["rows"]
            )
        )

        scheduler.step()

        row = {
            "epoch": (
                global_epoch
            ),
            "stage": (
                "finetune"
            ),
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_digit_accuracy": (
                train_metrics[
                    "digit_accuracy"
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
                val_tiles_result[
                    "exact_accuracy"
                ]
            ),
        }

        history.append(
            row
        )

        print(
            f"Epoch "
            f"{global_epoch:02d} | "
            f"train digit "
            f"{train_metrics['digit_accuracy']:.2%} | "
            f"val digit "
            f"{val_metrics['digit_accuracy']:.2%} | "
            f"val exact "
            f"{val_tiles_result['exact_accuracy']:.2%} | "
            f"val loss "
            f"{val_metrics['loss']:.4f}"
        )

        improved = (
            val_metrics[
                "digit_accuracy"
            ]
            > best_digit_accuracy
            or (
                val_metrics[
                    "digit_accuracy"
                ]
                == best_digit_accuracy
                and val_tiles_result[
                    "exact_accuracy"
                ]
                > best_exact_accuracy
            )
            or (
                val_metrics[
                    "digit_accuracy"
                ]
                == best_digit_accuracy
                and val_tiles_result[
                    "exact_accuracy"
                ]
                == best_exact_accuracy
                and val_metrics[
                    "loss"
                ]
                < best_val_loss
            )
        )

        if improved:
            best_digit_accuracy = (
                val_metrics[
                    "digit_accuracy"
                ]
            )

            best_exact_accuracy = (
                val_tiles_result[
                    "exact_accuracy"
                ]
            )

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = (
                global_epoch
            )

            no_improvement = 0

            save_checkpoint(
                checkpoint_path,
                model,
                global_epoch,
                "finetune",
                best_digit_accuracy,
                best_exact_accuracy,
                best_val_loss,
            )

            print(
                "  ↳ new best "
                "checkpoint"
            )

        else:
            no_improvement += 1

        if (
            no_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping after "
                f"{args.patience} "
                "epochs without "
                "validation improvement."
            )
            break

    save_csv(
        args.run_dir
        / "history.csv",
        history,
    )

    print()
    print("=" * 72)
    print(
        "VALIDATION RESULT"
    )
    print("=" * 72)

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        "Best validation "
        "digit accuracy: "
        f"{best_digit_accuracy:.2%}"
    )

    print(
        "Best validation "
        "exact accuracy: "
        f"{best_exact_accuracy:.2%}"
    )

    print(
        "Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    if not args.evaluate_test:
        print()
        print(
            "Locked test set was "
            "NOT evaluated."
        )
        return

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_metrics = (
        evaluate_digits(
            model,
            test_loader,
            criterion,
            device,
            collect=True,
        )
    )

    test_tiles_result = (
        reconstruct_tiles(
            test_metrics["rows"]
        )
    )

    save_csv(
        args.run_dir
        / "test_digit_predictions.csv",
        test_metrics["rows"],
    )

    save_csv(
        args.run_dir
        / "test_tile_predictions.csv",
        test_tiles_result[
            "rows"
        ],
    )

    print()
    print("=" * 72)
    print(
        "LOCKED TEST RESULT"
    )
    print("=" * 72)

    print(
        "Test digit accuracy: "
        f"{test_metrics['digit_accuracy']:.2%}"
    )

    print(
        "Test exact 3-digit "
        "accuracy: "
        f"{test_tiles_result['exact_accuracy']:.2%}"
    )

    print(
        "Correct tiles: "
        f"{test_tiles_result['exact_correct']}/"
        f"{test_tiles_result['total']}"
    )


if __name__ == "__main__":
    main()
