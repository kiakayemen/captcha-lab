from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


SEED = 42
IMAGE_SIZE = 224

DEFAULT_OUTPUT = Path("output")
DEFAULT_SPLIT_DIR = Path("experiments/convnext/run_001")
DEFAULT_RUN_DIR = Path("experiments/convnext/run_003_spatial")


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


def load_split_csv(path: Path, output_dir: Path) -> list[dict]:
    samples: list[dict] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {"image", "tile", "ground_truth"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns: {sorted(required)}"
            )

        for row in reader:
            truth = row["ground_truth"].strip()

            if len(truth) != 3 or not truth.isdigit():
                raise ValueError(
                    f"Invalid ground truth {truth!r} in {path}"
                )

            tile_number = int(row["tile"])

            candidate = Path(row.get("path", "").strip())

            if not candidate.exists():
                stem = Path(row["image"]).stem
                candidate = (
                    output_dir
                    / f"{stem}_tile_{tile_number}.png"
                )

            if not candidate.exists():
                raise FileNotFoundError(
                    f"Missing tile for {row['image']} "
                    f"tile {tile_number}: {candidate}"
                )

            samples.append(
                {
                    "image": row["image"],
                    "tile": tile_number,
                    "path": candidate,
                    "ground_truth": truth,
                    "label": tuple(int(d) for d in truth),
                    "group": row.get(
                        "group",
                        Path(row["image"]).stem,
                    ),
                }
            )

    return samples


class DigitTileDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        transform,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(
            sample["path"]
        ).convert("RGB")

        image = self.transform(image)

        target = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return (
            image,
            target,
            sample["ground_truth"],
            sample["image"],
            sample["tile"],
        )


def build_transforms():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    preprocessing = weights.transforms()

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.RandomAffine(
                degrees=4,
                translate=(0.025, 0.025),
                scale=(0.96, 1.04),
                shear=1.5,
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
                        sigma=(0.1, 0.6),
                    )
                ],
                p=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing.mean,
                std=preprocessing.std,
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
                mean=preprocessing.mean,
                std=preprocessing.std,
            ),
        ]
    )

    return train_transform, eval_transform


class SpatialConvNeXtThreeDigit(nn.Module):
    """
    ConvNeXt feature extractor that PRESERVES horizontal position.

    Instead of global average pooling the entire feature map into one
    vector, pool the final feature map into three horizontal bins:

        [ left digit | middle digit | right digit ]

    Each classifier head receives features from its own spatial region.
    """

    def __init__(self):
        super().__init__()

        backbone = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT
        )

        self.features = backbone.features

        # ConvNeXt-Tiny final feature width.
        feature_dim = backbone.classifier[2].in_features

        # Keep the pretrained LayerNorm from the original classifier.
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    feature_dim,
                    eps=1e-6,
                )
                for _ in range(3)
            ]
        )

        self.dropout = nn.Dropout(0.15)

        self.heads = nn.ModuleList(
            [
                nn.Linear(feature_dim, 10),
                nn.Linear(feature_dim, 10),
                nn.Linear(feature_dim, 10),
            ]
        )

    def forward(self, x):
        # ConvNeXt feature map:
        # B x C x H x W
        x = self.features(x)

        # MPS cannot adaptive-pool width 7 -> 3.
        # Split the width into 3 approximately equal regions instead.

        # For width=7 this becomes:
        #   left   = columns 0,1,2
        #   middle = columns 3,4
        #   right  = columns 5,6
        regions = torch.tensor_split(
            x,
            3,
            dim=3,
        )

        outputs = []

        for position, region in enumerate(regions):
            # Average over spatial dimensions while preserving
            # which horizontal region the features came from.
            #
            # B x C x H x W_region
            #       ↓
            # B x C
            region = region.mean(
                dim=(2, 3)
            )

            region = self.norms[position](
                region
            )

            region = self.dropout(
                region
            )

            outputs.append(
                self.heads[position](
                    region
                )
            )

        return tuple(outputs)

def freeze_backbone(
    model: SpatialConvNeXtThreeDigit,
) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.norms.parameters():
        parameter.requires_grad = True

    for parameter in model.heads.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(
    model: SpatialConvNeXtThreeDigit,
    num_blocks: int = 3,
) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    children = list(
        model.features.children()
    )

    for block in children[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.norms.parameters():
        parameter.requires_grad = True

    for parameter in model.heads.parameters():
        parameter.requires_grad = True


def predictions_from_logits(outputs):
    return torch.stack(
        [
            output.argmax(dim=1)
            for output in outputs
        ],
        dim=1,
    )


def compute_loss(
    outputs,
    targets,
    criterion,
):
    losses = [
        criterion(
            outputs[position],
            targets[:, position],
        )
        for position in range(3)
    ]

    return sum(losses) / 3.0


def calculate_metrics(
    predictions,
    targets,
):
    matches = predictions.eq(targets)

    return {
        "digit_correct": matches.sum().item(),
        "digit_total": targets.numel(),
        "exact_correct": (
            matches.all(dim=1)
            .sum()
            .item()
        ),
        "sample_total": targets.shape[0],
    }


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    loss_sum = 0.0
    digit_correct = 0
    digit_total = 0
    exact_correct = 0
    sample_total = 0

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
            [
                parameter
                for parameter
                in model.parameters()
                if parameter.requires_grad
            ],
            max_norm=1.0,
        )

        optimizer.step()

        predictions = (
            predictions_from_logits(
                outputs
            )
        )

        metrics = calculate_metrics(
            predictions,
            targets,
        )

        batch_size = targets.shape[0]

        loss_sum += (
            loss.item()
            * batch_size
        )

        digit_correct += (
            metrics["digit_correct"]
        )
        digit_total += (
            metrics["digit_total"]
        )
        exact_correct += (
            metrics["exact_correct"]
        )
        sample_total += (
            metrics["sample_total"]
        )

    return {
        "loss": (
            loss_sum / sample_total
        ),
        "digit_accuracy": (
            digit_correct / digit_total
        ),
        "exact_accuracy": (
            exact_correct / sample_total
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    collect_predictions: bool = False,
):
    model.eval()

    loss_sum = 0.0
    digit_correct = 0
    digit_total = 0
    exact_correct = 0
    sample_total = 0

    rows: list[dict] = []

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

        predictions = (
            predictions_from_logits(
                outputs
            )
        )

        metrics = calculate_metrics(
            predictions,
            targets,
        )

        batch_size = targets.shape[0]

        loss_sum += (
            loss.item()
            * batch_size
        )

        digit_correct += (
            metrics["digit_correct"]
        )
        digit_total += (
            metrics["digit_total"]
        )
        exact_correct += (
            metrics["exact_correct"]
        )
        sample_total += (
            metrics["sample_total"]
        )

        if collect_predictions:
            probabilities = [
                torch.softmax(
                    output,
                    dim=1,
                )
                for output in outputs
            ]

            predictions_cpu = (
                predictions.cpu()
            )

            for index in range(
                batch_size
            ):
                predicted_digits = (
                    predictions_cpu[
                        index
                    ].tolist()
                )

                prediction = "".join(
                    str(digit)
                    for digit
                    in predicted_digits
                )

                confidences = [
                    probabilities[
                        position
                    ][
                        index,
                        predicted_digits[
                            position
                        ],
                    ].item()
                    for position
                    in range(3)
                ]

                rows.append(
                    {
                        "image": (
                            image_names[index]
                        ),
                        "tile": int(
                            tile_numbers[index]
                        ),
                        "ground_truth": (
                            truths[index]
                        ),
                        "prediction": (
                            prediction
                        ),
                        "correct": (
                            prediction
                            == truths[index]
                        ),
                        "confidence_1": round(
                            confidences[0],
                            6,
                        ),
                        "confidence_2": round(
                            confidences[1],
                            6,
                        ),
                        "confidence_3": round(
                            confidences[2],
                            6,
                        ),
                        "min_confidence": round(
                            min(confidences),
                            6,
                        ),
                        "mean_confidence": round(
                            sum(confidences)
                            / 3.0,
                            6,
                        ),
                    }
                )

    return {
        "loss": (
            loss_sum / sample_total
        ),
        "digit_accuracy": (
            digit_correct / digit_total
        ),
        "exact_accuracy": (
            exact_correct / sample_total
        ),
        "correct": exact_correct,
        "total": sample_total,
        "predictions": rows,
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
        writer.writerows(rows)


def copy_split(
    destination: Path,
    samples: list[dict],
) -> None:
    rows = [
        {
            "image": sample["image"],
            "tile": sample["tile"],
            "ground_truth": (
                sample["ground_truth"]
            ),
            "path": str(
                sample["path"]
            ),
            "group": sample["group"],
        }
        for sample in samples
    ]

    save_csv(
        destination,
        rows,
    )


def build_warmup_optimizer(
    model,
    learning_rate: float,
):
    parameters = (
        list(
            model.norms.parameters()
        )
        + list(
            model.heads.parameters()
        )
    )

    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=1e-4,
    )


def build_finetune_optimizer(
    model,
    backbone_lr: float,
    head_lr: float,
):
    backbone_parameters = []
    classifier_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if (
            name.startswith("heads.")
            or name.startswith("norms.")
        ):
            classifier_parameters.append(
                parameter
            )
        else:
            backbone_parameters.append(
                parameter
            )

    return torch.optim.AdamW(
        [
            {
                "params": (
                    backbone_parameters
                ),
                "lr": backbone_lr,
            },
            {
                "params": (
                    classifier_parameters
                ),
                "lr": head_lr,
            },
        ],
        weight_decay=1e-4,
    )


def checkpoint_is_better(
    digit_accuracy: float,
    exact_accuracy: float,
    val_loss: float,
    best_digit_accuracy: float,
    best_exact_accuracy: float,
    best_val_loss: float,
) -> bool:
    if (
        digit_accuracy
        > best_digit_accuracy
    ):
        return True

    if (
        digit_accuracy
        == best_digit_accuracy
        and exact_accuracy
        > best_exact_accuracy
    ):
        return True

    if (
        digit_accuracy
        == best_digit_accuracy
        and exact_accuracy
        == best_exact_accuracy
        and val_loss
        < best_val_loss
    ):
        return True

    return False


def main():
    parser = argparse.ArgumentParser()

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
        default=16,
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=45,
    )

    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
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

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    train_samples = load_split_csv(
        args.split_dir
        / "train_split.csv",
        args.output,
    )

    val_samples = load_split_csv(
        args.split_dir
        / "val_split.csv",
        args.output,
    )

    test_samples = load_split_csv(
        args.split_dir
        / "test_split.csv",
        args.output,
    )

    print()
    print(
        "Locked dataset split:"
    )
    print(
        f"  Train: "
        f"{len(train_samples)}"
    )
    print(
        f"  Val:   "
        f"{len(val_samples)}"
    )
    print(
        f"  Test:  "
        f"{len(test_samples)}"
    )

    train_groups = {
        sample["group"]
        for sample
        in train_samples
    }

    val_groups = {
        sample["group"]
        for sample
        in val_samples
    }

    test_groups = {
        sample["group"]
        for sample
        in test_samples
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

    copy_split(
        args.run_dir
        / "train_split.csv",
        train_samples,
    )

    copy_split(
        args.run_dir
        / "val_split.csv",
        val_samples,
    )

    copy_split(
        args.run_dir
        / "test_split.csv",
        test_samples,
    )

    (
        train_transform,
        eval_transform,
    ) = build_transforms()

    train_loader = DataLoader(
        DigitTileDataset(
            train_samples,
            train_transform,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        DigitTileDataset(
            val_samples,
            eval_transform,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        DigitTileDataset(
            test_samples,
            eval_transform,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "Loading pretrained "
        "ConvNeXt-Tiny "
        "with spatial heads..."
    )

    model = (
        SpatialConvNeXtThreeDigit()
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
    best_val_loss = (
        float("inf")
    )
    best_epoch = 0

    global_epoch = 0

    # ---------------------------------------------------------
    # Stage 1
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "STAGE 1 — SPATIAL HEAD WARMUP"
    )
    print("=" * 72)

    freeze_backbone(
        model
    )

    optimizer = (
        build_warmup_optimizer(
            model,
            args.warmup_lr,
        )
    )

    for stage_epoch in range(
        1,
        args.warmup_epochs + 1,
    ):
        global_epoch += 1

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

        row = {
            "epoch": global_epoch,
            "stage": (
                "spatial_head_warmup"
            ),
            "stage_epoch": stage_epoch,
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_digit_accuracy": (
                train_metrics[
                    "digit_accuracy"
                ]
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

        history.append(
            row
        )

        print(
            f"Epoch "
            f"{global_epoch:02d} | "
            f"train digit "
            f"{train_metrics['digit_accuracy']:.2%} | "
            f"train exact "
            f"{train_metrics['exact_accuracy']:.2%} | "
            f"val digit "
            f"{val_metrics['digit_accuracy']:.2%} | "
            f"val exact "
            f"{val_metrics['exact_accuracy']:.2%} | "
            f"val loss "
            f"{val_metrics['loss']:.4f}"
        )

        improved = checkpoint_is_better(
            val_metrics[
                "digit_accuracy"
            ],
            val_metrics[
                "exact_accuracy"
            ],
            val_metrics[
                "loss"
            ],
            best_digit_accuracy,
            best_exact_accuracy,
            best_val_loss,
        )

        if improved:
            best_digit_accuracy = (
                val_metrics[
                    "digit_accuracy"
                ]
            )

            best_exact_accuracy = (
                val_metrics[
                    "exact_accuracy"
                ]
            )

            best_val_loss = (
                val_metrics[
                    "loss"
                ]
            )

            best_epoch = (
                global_epoch
            )

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "epoch": (
                        global_epoch
                    ),
                    "stage": (
                        "spatial_head_warmup"
                    ),
                    "val_digit_accuracy": (
                        best_digit_accuracy
                    ),
                    "val_exact_accuracy": (
                        best_exact_accuracy
                    ),
                    "val_loss": (
                        best_val_loss
                    ),
                    "seed": (
                        args.seed
                    ),
                },
                checkpoint_path,
            )

            print(
                "  ↳ new best "
                "checkpoint"
            )

    # Restore strongest warmup state before fine-tuning.
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

    # ---------------------------------------------------------
    # Stage 2
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "STAGE 2 — SPATIAL "
        "LATE-BLOCK FINE-TUNING"
    )
    print("=" * 72)

    unfreeze_last_blocks(
        model,
        num_blocks=3,
    )

    optimizer = (
        build_finetune_optimizer(
            model,
            backbone_lr=(
                args.backbone_lr
            ),
            head_lr=(
                args.head_lr
            ),
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

    epochs_without_improvement = 0

    for stage_epoch in range(
        1,
        args.finetune_epochs + 1,
    ):
        global_epoch += 1

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

        row = {
            "epoch": global_epoch,
            "stage": "finetune",
            "stage_epoch": stage_epoch,
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_digit_accuracy": (
                train_metrics[
                    "digit_accuracy"
                ]
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

        history.append(
            row
        )

        print(
            f"Epoch "
            f"{global_epoch:02d} | "
            f"train digit "
            f"{train_metrics['digit_accuracy']:.2%} | "
            f"train exact "
            f"{train_metrics['exact_accuracy']:.2%} | "
            f"val digit "
            f"{val_metrics['digit_accuracy']:.2%} | "
            f"val exact "
            f"{val_metrics['exact_accuracy']:.2%} | "
            f"val loss "
            f"{val_metrics['loss']:.4f}"
        )

        improved = checkpoint_is_better(
            val_metrics[
                "digit_accuracy"
            ],
            val_metrics[
                "exact_accuracy"
            ],
            val_metrics[
                "loss"
            ],
            best_digit_accuracy,
            best_exact_accuracy,
            best_val_loss,
        )

        if improved:
            best_digit_accuracy = (
                val_metrics[
                    "digit_accuracy"
                ]
            )

            best_exact_accuracy = (
                val_metrics[
                    "exact_accuracy"
                ]
            )

            best_val_loss = (
                val_metrics[
                    "loss"
                ]
            )

            best_epoch = (
                global_epoch
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "epoch": (
                        global_epoch
                    ),
                    "stage": (
                        "finetune"
                    ),
                    "val_digit_accuracy": (
                        best_digit_accuracy
                    ),
                    "val_exact_accuracy": (
                        best_exact_accuracy
                    ),
                    "val_loss": (
                        best_val_loss
                    ),
                    "seed": (
                        args.seed
                    ),
                },
                checkpoint_path,
            )

            print(
                "  ↳ new best "
                "checkpoint"
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping after "
                f"{args.patience} "
                "fine-tuning epochs "
                "without improvement."
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

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        collect_predictions=True,
    )

    save_csv(
        args.run_dir
        / "test_predictions.csv",
        test_metrics[
            "predictions"
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
        "Test exact 3-digit accuracy: "
        f"{test_metrics['exact_accuracy']:.2%}"
    )

    print(
        f"Correct tiles: "
        f"{test_metrics['correct']}/"
        f"{test_metrics['total']}"
    )


if __name__ == "__main__":
    main()
