from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader

from cnn.dataset import CaptchaTileDataset, Sample, load_samples
from cnn.model import CaptchaCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_samples(samples: list[Sample], validation_fraction: float, seed: int):
    groups = [sample.screenshot for sample in samples]
    indices = np.arange(len(samples))
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(indices, groups=groups))
    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
    )


def calculate_loss(outputs, labels, criterion):
    return sum(criterion(outputs[i], labels[:, i]) for i in range(3))


def batch_metrics(outputs, labels):
    predictions = torch.stack([output.argmax(dim=1) for output in outputs], dim=1)
    exact = int((predictions == labels).all(dim=1).sum().item())
    digits = [
        int((predictions[:, i] == labels[:, i]).sum().item())
        for i in range(3)
    ]
    return exact, digits


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_exact = 0
    total_digits = [0, 0, 0]

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = calculate_loss(outputs, labels, criterion)

        batch_size = images.size(0)
        exact, digits = batch_metrics(outputs, labels)

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        total_exact += exact
        total_digits = [total_digits[i] + digits[i] for i in range(3)]

    return (
        total_loss / total_samples,
        total_exact / total_samples,
        [value / total_samples for value in total_digits],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("preprocessing_results.csv"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--model-output", type=Path, default=Path("models/captcha_cnn.pt"))
    parser.add_argument("--history-output", type=Path, default=Path("cnn_training_history.csv"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device()
    print(f"Device: {device}")

    samples = load_samples(args.csv, args.project_root)

    if args.overfit_samples:
        selected = samples[:args.overfit_samples]
        train_samples = selected
        validation_samples = selected
        use_augmentation = False
        dropout = 0.0
        print(f"OVERFIT TEST: same {len(selected)} samples used for train and validation")
    else:
        train_samples, validation_samples = split_samples(
            samples, args.validation_fraction, args.seed
        )
        use_augmentation = True
        dropout = 0.20

    print(f"All samples: {len(samples)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")

    train_loader = DataLoader(
        CaptchaTileDataset(train_samples, training=use_augmentation),
        batch_size=min(args.batch_size, len(train_samples)),
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        CaptchaTileDataset(validation_samples, training=False),
        batch_size=min(args.batch_size, len(validation_samples)),
        shuffle=False,
        num_workers=0,
    )

    model = CaptchaCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0 if args.overfit_samples else 1e-4,
    )

    best_accuracy = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        sample_total = 0
        exact_total = 0
        digit_totals = [0, 0, 0]

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = calculate_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            exact, digits = batch_metrics(outputs, labels)
            loss_total += float(loss.item()) * batch_size
            sample_total += batch_size
            exact_total += exact
            digit_totals = [digit_totals[i] + digits[i] for i in range(3)]

        train_loss = loss_total / sample_total
        train_exact = exact_total / sample_total
        val_loss, val_exact, val_digits = evaluate(
            model, validation_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_exact_accuracy": train_exact,
            "validation_loss": val_loss,
            "validation_exact_accuracy": val_exact,
            "validation_digit_1_accuracy": val_digits[0],
            "validation_digit_2_accuracy": val_digits[1],
            "validation_digit_3_accuracy": val_digits[2],
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train exact {train_exact:.2%} | "
            f"val exact {val_exact:.2%} | "
            f"val digits {val_digits[0]:.1%}/{val_digits[1]:.1%}/{val_digits[2]:.1%}"
        )

        if val_exact > best_accuracy:
            best_accuracy = val_exact
            best_state = copy.deepcopy(model.state_dict())

        if args.overfit_samples and val_exact == 1.0:
            print("Sanity test passed: model memorized the tiny dataset.")
            break

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "validation_accuracy": best_accuracy,
        "seed": args.seed,
        "image_width": 96,
        "image_height": 32,
    }, args.model_output)

    with args.history_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print()
    print(f"Best validation exact accuracy: {best_accuracy:.2%}")
    print(f"Saved model: {args.model_output}")
    print(f"Saved history: {args.history_output}")


if __name__ == "__main__":
    main()
