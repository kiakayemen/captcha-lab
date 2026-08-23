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
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


SEED = 42
IMAGE_SIZE = 224

DEFAULT_LABELS = Path("labels.csv")
DEFAULT_OUTPUT = Path("output")
DEFAULT_SPLIT_DIR = Path("experiments/convnext/run_001")
DEFAULT_RUN_DIR = Path("experiments/convnext/run_002")


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
    rows: list[dict] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {"image", "tile", "ground_truth"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain at least columns: {sorted(required)}"
            )

        for row in reader:
            truth = row["ground_truth"].strip()

            if len(truth) != 3 or not truth.isdigit():
                raise ValueError(
                    f"Invalid ground truth {truth!r} in {path}"
                )

            tile_number = int(row["tile"])

            saved_path = row.get("path", "").strip()
            if saved_path:
                candidate = Path(saved_path)
            else:
                stem = Path(row["image"]).stem
                candidate = output_dir / f"{stem}_tile_{tile_number}.png"

            # Old split CSVs may contain absolute paths from another machine/run.
            if not candidate.exists():
                stem = Path(row["image"]).stem
                candidate = output_dir / f"{stem}_tile_{tile_number}.png"

            if not candidate.exists():
                raise FileNotFoundError(
                    f"Could not locate tile for {row['image']} tile {tile_number}: "
                    f"{candidate}"
                )

            rows.append(
                {
                    "image": row["image"],
                    "tile": tile_number,
                    "path": candidate,
                    "ground_truth": truth,
                    "label": tuple(int(d) for d in truth),
                    "group": row.get("group", Path(row["image"]).stem),
                }
            )

    return rows


class DigitTileDataset(Dataset):
    def __init__(self, samples: list[dict], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(sample["path"]).convert("RGB")
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
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=2,
                fill=255,
            ),
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 0.8),
                    )
                ],
                p=0.12,
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
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing.mean,
                std=preprocessing.std,
            ),
        ]
    )

    return train_transform, eval_transform


class ConvNeXtThreeDigit(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT
        )

        feature_dim = backbone.classifier[2].in_features

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]
        self.flatten = backbone.classifier[1]

        self.dropout = nn.Dropout(0.20)

        self.heads = nn.ModuleList(
            [
                nn.Linear(feature_dim, 10),
                nn.Linear(feature_dim, 10),
                nn.Linear(feature_dim, 10),
            ]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.dropout(x)

        return tuple(head(x) for head in self.heads)


def freeze_backbone(model: ConvNeXtThreeDigit) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.norm.parameters():
        parameter.requires_grad = False

    for parameter in model.heads.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(
    model: ConvNeXtThreeDigit,
    num_blocks: int = 2,
) -> None:
    # Keep early ConvNeXt representation frozen.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    children = list(model.features.children())

    for block in children[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.norm.parameters():
        parameter.requires_grad = True

    for parameter in model.heads.parameters():
        parameter.requires_grad = True


def predictions_from_logits(outputs):
    return torch.stack(
        [output.argmax(dim=1) for output in outputs],
        dim=1,
    )


def compute_loss(outputs, targets, criterion):
    losses = [
        criterion(outputs[position], targets[:, position])
        for position in range(3)
    ]

    return sum(losses) / 3.0


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
    exact_correct = 0
    digit_total = 0
    sample_total = 0

    for images, targets, *_ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = compute_loss(outputs, targets, criterion)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )

        optimizer.step()

        predictions = predictions_from_logits(outputs)
        matches = predictions.eq(targets)

        batch_size = targets.shape[0]

        loss_sum += loss.item() * batch_size
        digit_correct += matches.sum().item()
        exact_correct += matches.all(dim=1).sum().item()

        digit_total += targets.numel()
        sample_total += batch_size

    return {
        "loss": loss_sum / sample_total,
        "digit_accuracy": digit_correct / digit_total,
        "exact_accuracy": exact_correct / sample_total,
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
    exact_correct = 0
    digit_total = 0
    sample_total = 0

    rows = []

    for images, targets, truths, image_names, tile_numbers in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = compute_loss(outputs, targets, criterion)

        predictions = predictions_from_logits(outputs)
        matches = predictions.eq(targets)

        batch_size = targets.shape[0]

        loss_sum += loss.item() * batch_size
        digit_correct += matches.sum().item()
        exact_correct += matches.all(dim=1).sum().item()

        digit_total += targets.numel()
        sample_total += batch_size

        if collect_predictions:
            probabilities = [
                torch.softmax(output, dim=1)
                for output in outputs
            ]

            predictions_cpu = predictions.cpu()

            for index in range(batch_size):
                predicted_digits = predictions_cpu[index].tolist()

                prediction = "".join(
                    str(digit)
                    for digit in predicted_digits
                )

                confidences = [
                    probabilities[position][
                        index,
                        predicted_digits[position],
                    ].item()
                    for position in range(3)
                ]

                rows.append(
                    {
                        "image": image_names[index],
                        "tile": int(tile_numbers[index]),
                        "ground_truth": truths[index],
                        "prediction": prediction,
                        "correct": prediction == truths[index],
                        "confidence_1": round(confidences[0], 6),
                        "confidence_2": round(confidences[1], 6),
                        "confidence_3": round(confidences[2], 6),
                        "min_confidence": round(min(confidences), 6),
                        "mean_confidence": round(
                            sum(confidences) / 3.0,
                            6,
                        ),
                    }
                )

    return {
        "loss": loss_sum / sample_total,
        "digit_accuracy": digit_correct / digit_total,
        "exact_accuracy": exact_correct / sample_total,
        "correct": exact_correct,
        "total": sample_total,
        "predictions": rows,
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

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


def save_split_copy(path: Path, samples: list[dict]) -> None:
    rows = [
        {
            "image": sample["image"],
            "tile": sample["tile"],
            "ground_truth": sample["ground_truth"],
            "path": str(sample["path"]),
            "group": sample["group"],
        }
        for sample in samples
    ]

    save_csv(path, rows)


def build_head_optimizer(
    model: ConvNeXtThreeDigit,
    learning_rate: float,
):
    return torch.optim.AdamW(
        model.heads.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )


def build_finetune_optimizer(
    model: ConvNeXtThreeDigit,
    backbone_lr: float,
    head_lr: float,
):
    backbone_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("heads."):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
            },
            {
                "params": head_parameters,
                "lr": head_lr,
            },
        ],
        weight_decay=1e-4,
    )


def is_better_checkpoint(
    current_digit_accuracy: float,
    current_loss: float,
    best_digit_accuracy: float,
    best_loss: float,
) -> bool:
    if current_digit_accuracy > best_digit_accuracy:
        return True

    if (
        current_digit_accuracy == best_digit_accuracy
        and current_loss < best_loss
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
        default=12,
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=8e-4,
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=8e-6,
    )
    parser.add_argument(
        "--finetune-head-lr",
        type=float,
        default=8e-5,
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
        help="Evaluate the locked test set after training. Leave off while tuning.",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()
    print(f"Device: {device}")

    train_samples = load_split_csv(
        args.split_dir / "train_split.csv",
        args.output,
    )
    val_samples = load_split_csv(
        args.split_dir / "val_split.csv",
        args.output,
    )
    test_samples = load_split_csv(
        args.split_dir / "test_split.csv",
        args.output,
    )

    print()
    print("Locked dataset split:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Test:  {len(test_samples)}")

    train_groups = {sample["group"] for sample in train_samples}
    val_groups = {sample["group"] for sample in val_samples}
    test_groups = {sample["group"] for sample in test_samples}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    save_split_copy(
        args.run_dir / "train_split.csv",
        train_samples,
    )
    save_split_copy(
        args.run_dir / "val_split.csv",
        val_samples,
    )
    save_split_copy(
        args.run_dir / "test_split.csv",
        test_samples,
    )

    train_transform, eval_transform = build_transforms()

    train_loader = DataLoader(
        DigitTileDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        DigitTileDataset(val_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        DigitTileDataset(test_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print()
    print("Loading pretrained ConvNeXt-Tiny...")
    model = ConvNeXtThreeDigit().to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.03
    )

    checkpoint_path = (
        args.run_dir / "best_model.pt"
    )

    history: list[dict] = []

    best_digit_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    global_epoch = 0

    # -------------------------------------------------------------
    # Stage 1: learn only the randomly initialized digit heads.
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("STAGE 1 — HEAD WARMUP")
    print("=" * 72)

    freeze_backbone(model)

    optimizer = build_head_optimizer(
        model,
        args.head_lr,
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
            "stage": "head_warmup",
            "stage_epoch": stage_epoch,
            "train_loss": train_metrics["loss"],
            "train_digit_accuracy": train_metrics["digit_accuracy"],
            "train_exact_accuracy": train_metrics["exact_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_digit_accuracy": val_metrics["digit_accuracy"],
            "val_exact_accuracy": val_metrics["exact_accuracy"],
        }

        history.append(row)

        print(
            f"Epoch {global_epoch:02d} | "
            f"train digit {train_metrics['digit_accuracy']:.2%} | "
            f"train exact {train_metrics['exact_accuracy']:.2%} | "
            f"val digit {val_metrics['digit_accuracy']:.2%} | "
            f"val exact {val_metrics['exact_accuracy']:.2%} | "
            f"val loss {val_metrics['loss']:.4f}"
        )

        if is_better_checkpoint(
            val_metrics["digit_accuracy"],
            val_metrics["loss"],
            best_digit_accuracy,
            best_val_loss,
        ):
            best_digit_accuracy = val_metrics["digit_accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = global_epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": global_epoch,
                    "stage": "head_warmup",
                    "val_digit_accuracy": best_digit_accuracy,
                    "val_loss": best_val_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

            print("  ↳ new best checkpoint")

    # Start fine-tuning from the strongest warmup checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -------------------------------------------------------------
    # Stage 2: fine-tune only the later pretrained ConvNeXt blocks.
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("STAGE 2 — LATE-BLOCK FINE-TUNING")
    print("=" * 72)

    unfreeze_last_blocks(
        model,
        num_blocks=2,
    )

    optimizer = build_finetune_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.finetune_head_lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.finetune_epochs,
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
            "train_loss": train_metrics["loss"],
            "train_digit_accuracy": train_metrics["digit_accuracy"],
            "train_exact_accuracy": train_metrics["exact_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_digit_accuracy": val_metrics["digit_accuracy"],
            "val_exact_accuracy": val_metrics["exact_accuracy"],
        }

        history.append(row)

        print(
            f"Epoch {global_epoch:02d} | "
            f"train digit {train_metrics['digit_accuracy']:.2%} | "
            f"train exact {train_metrics['exact_accuracy']:.2%} | "
            f"val digit {val_metrics['digit_accuracy']:.2%} | "
            f"val exact {val_metrics['exact_accuracy']:.2%} | "
            f"val loss {val_metrics['loss']:.4f}"
        )

        improved = is_better_checkpoint(
            val_metrics["digit_accuracy"],
            val_metrics["loss"],
            best_digit_accuracy,
            best_val_loss,
        )

        if improved:
            best_digit_accuracy = val_metrics["digit_accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = global_epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": global_epoch,
                    "stage": "finetune",
                    "val_digit_accuracy": best_digit_accuracy,
                    "val_loss": best_val_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

            print("  ↳ new best checkpoint")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print()
            print(
                f"Early stopping after {args.patience} "
                "fine-tuning epochs without validation improvement."
            )
            break

    save_csv(
        args.run_dir / "history.csv",
        history,
    )

    print()
    print("=" * 72)
    print("VALIDATION RESULT")
    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation digit accuracy: "
        f"{best_digit_accuracy:.2%}"
    )
    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print()
    print(
        "The locked test set is NOT evaluated automatically in run_002."
    )
    print(
        "Use --evaluate-test only after the training recipe is accepted."
    )

    if args.evaluate_test:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            collect_predictions=True,
        )

        save_csv(
            args.run_dir / "test_predictions.csv",
            test_metrics["predictions"],
        )

        print()
        print("=" * 72)
        print("LOCKED TEST RESULT")
        print("=" * 72)
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
            f"{test_metrics['correct']}/{test_metrics['total']}"
        )


if __name__ == "__main__":
    main()
