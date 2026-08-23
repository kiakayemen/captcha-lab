from __future__ import annotations

import argparse
import csv
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SEED = 42

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_SPLIT_DIR = Path("experiments/convnext/run_001")
DEFAULT_RUN_DIR = Path("experiments/parseq_finetune/run_001")

DIGITS_ONLY = re.compile(r"\D")


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

            tile_path = resolve_tile_path(
                row,
                output_dir,
            )

            samples.append(
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
            sample["ground_truth"],
            sample["image"],
            sample["tile"],
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


def parse_confidence(
    confidence,
) -> float:
    if torch.is_tensor(confidence):
        confidence = (
            confidence
            .detach()
            .cpu()
            .float()
        )

        if confidence.numel() == 1:
            return float(
                confidence.item()
            )

        return float(
            confidence.mean().item()
        )

    return float(confidence)


def parseq_training_loss(
    model,
    images: torch.Tensor,
    labels: list[str],
) -> torch.Tensor:
    """
    Reproduce PARSeq's official permutation-training loss directly,
    without requiring Lightning Trainer / Hydra / LMDB.
    """
    device = images.device

    tgt = model.tokenizer.encode(
        labels,
        device,
    )

    memory = model.model.encode(images)

    tgt_perms = model.gen_tgt_perms(tgt)

    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    tgt_padding_mask = (
        (tgt_in == model.pad_id)
        | (tgt_in == model.eos_id)
    )

    total_loss = torch.zeros(
        (),
        device=device,
    )

    loss_numel = 0

    n = (
        tgt_out != model.pad_id
    ).sum().item()

    for index, perm in enumerate(
        tgt_perms
    ):
        tgt_mask, query_mask = (
            model.generate_attn_masks(
                perm
            )
        )

        decoded = model.model.decode(
            tgt_in,
            memory,
            tgt_mask,
            tgt_padding_mask,
            tgt_query_mask=query_mask,
        )

        logits = model.model.head(
            decoded
        ).flatten(
            end_dim=1
        )

        permutation_loss = (
            F.cross_entropy(
                logits,
                tgt_out.flatten(),
                ignore_index=model.pad_id,
            )
        )

        total_loss = (
            total_loss
            + n * permutation_loss
        )

        loss_numel += n

        # Matches the official PARSeq training logic:
        # after canonical + reverse permutations, do not
        # repeatedly train EOS under all remaining permutations.
        if index == 1:
            tgt_out = torch.where(
                tgt_out == model.eos_id,
                model.pad_id,
                tgt_out,
            )

            n = (
                tgt_out
                != model.pad_id
            ).sum().item()

    return (
        total_loss
        / max(loss_numel, 1)
    )


def set_decoder_only_trainable(
    model,
) -> None:
    """
    Torch Hub returns the PARSeq Lightning/system wrapper.
    The actual recognizer modules live under model.model in
    current strhub versions.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    recognizer = model.model

    for parameter in recognizer.decoder.parameters():
        parameter.requires_grad = True

    for parameter in recognizer.head.parameters():
        parameter.requires_grad = True

    for parameter in recognizer.text_embed.parameters():
        parameter.requires_grad = True

    recognizer.pos_queries.requires_grad = True


def set_all_trainable(
    model,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def trainable_parameters(
    model,
):
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def train_epoch(
    model,
    loader,
    optimizer,
    device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for (
        images,
        labels,
        _,
        _,
    ) in loader:
        images = images.to(device)

        labels = list(labels)

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = parseq_training_loss(
            model,
            images,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss: "
                f"{loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = images.shape[0]

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
) -> dict:
    model.eval()

    rows: list[dict] = []

    exact_correct = 0
    total = 0

    confidence_sum = 0.0

    for (
        images,
        truths,
        image_names,
        tile_numbers,
    ) in loader:
        images = images.to(device)

        # We know the desired text length is 3.
        # max_length=3 means PARSeq predicts 3 chars + EOS.
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

            correct = (
                prediction == truth
            )

            confidence = (
                parse_confidence(
                    confidences[index]
                )
            )

            exact_correct += int(
                correct
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
                    "correct": correct,
                    "confidence": round(
                        confidence,
                        6,
                    ),
                }
            )

    accuracy = (
        exact_correct / total
        if total
        else 0.0
    )

    mean_confidence = (
        confidence_sum / total
        if total
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "correct": exact_correct,
        "total": total,
        "mean_confidence": (
            mean_confidence
        ),
        "rows": rows,
    }


def checkpoint_is_better(
    accuracy: float,
    mean_confidence: float,
    best_accuracy: float,
    best_mean_confidence: float,
) -> bool:
    if accuracy > best_accuracy:
        return True

    if (
        math.isclose(
            accuracy,
            best_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and mean_confidence
        > best_mean_confidence
    ):
        return True

    return False


def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    stage: str,
    val_accuracy: float,
    val_mean_confidence: float,
) -> None:
    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "epoch": epoch,
            "stage": stage,
            "val_accuracy": (
                val_accuracy
            ),
            "val_mean_confidence": (
                val_mean_confidence
            ),
        },
        path,
    )


def build_loader(
    samples: list[dict],
    transform,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TileDataset(
            samples,
            transform,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


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
        "--decoder-epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--full-epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--full-lr",
        type=float,
        default=2e-6,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=6,
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

    train_samples = load_split(
        args.split_dir
        / "train_split.csv",
        args.output_dir,
    )

    val_samples = load_split(
        args.split_dir
        / "val_split.csv",
        args.output_dir,
    )

    test_samples = load_split(
        args.split_dir
        / "test_split.csv",
        args.output_dir,
    )

    print()
    print(
        "Locked split:"
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

    print()
    print(
        "Loading pretrained PARSeq..."
    )

    model = torch.hub.load(
        "baudm/parseq",
        "parseq",
        pretrained=True,
        trust_repo=True,
    )

    model = model.to(device)

    # Current strhub Torch Hub models are system wrappers.
    # Validate the API once here so future upstream changes fail
    # with a useful message rather than deep inside training.
    if not hasattr(model, "model"):
        raise RuntimeError(
            "Unexpected PARSeq API: loaded model has no '.model' recognizer wrapper."
        )

    required_recognizer_attrs = (
        "encoder",
        "decoder",
        "head",
        "text_embed",
        "pos_queries",
        "encode",
        "decode",
    )

    missing_attrs = [
        name
        for name in required_recognizer_attrs
        if not hasattr(model.model, name)
    ]

    if missing_attrs:
        raise RuntimeError(
            "Unexpected PARSeq recognizer API. Missing: "
            + ", ".join(missing_attrs)
        )

    print(
        "PARSeq recognizer API: "
        "wrapper.model encoder/decoder/head confirmed"
    )

    from strhub.data.module import (
        SceneTextDataModule,
    )

    transform = (
        SceneTextDataModule
        .get_transform(
            model.hparams.img_size
        )
    )

    train_loader = build_loader(
        train_samples,
        transform,
        args.batch_size,
        shuffle=True,
    )

    val_loader = build_loader(
        val_samples,
        transform,
        args.batch_size,
        shuffle=False,
    )

    test_loader = build_loader(
        test_samples,
        transform,
        args.batch_size,
        shuffle=False,
    )

    checkpoint_path = (
        args.run_dir
        / "best_model.pt"
    )

    history: list[dict] = []

    # ---------------------------------------------------------
    # Epoch 0: untouched pretrained PARSeq.
    # This is the incumbent and MUST be allowed to win.
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "EPOCH 0 — ZERO-SHOT INCUMBENT"
    )
    print("=" * 72)

    baseline_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    best_accuracy = (
        baseline_metrics[
            "accuracy"
        ]
    )

    best_mean_confidence = (
        baseline_metrics[
            "mean_confidence"
        ]
    )

    best_epoch = 0
    best_stage = "zero_shot"

    save_checkpoint(
        checkpoint_path,
        model,
        epoch=0,
        stage="zero_shot",
        val_accuracy=best_accuracy,
        val_mean_confidence=(
            best_mean_confidence
        ),
    )

    save_csv(
        args.run_dir
        / "epoch_000_val_predictions.csv",
        baseline_metrics["rows"],
    )

    history.append(
        {
            "epoch": 0,
            "stage": "zero_shot",
            "train_loss": "",
            "val_accuracy": (
                best_accuracy
            ),
            "val_correct": (
                baseline_metrics[
                    "correct"
                ]
            ),
            "val_total": (
                baseline_metrics[
                    "total"
                ]
            ),
            "val_mean_confidence": (
                best_mean_confidence
            ),
            "best": True,
        }
    )

    print(
        "Validation accuracy: "
        f"{best_accuracy:.2%} "
        f"("
        f"{baseline_metrics['correct']}/"
        f"{baseline_metrics['total']}"
        f")"
    )

    print(
        "Mean confidence: "
        f"{best_mean_confidence:.4f}"
    )

    # ---------------------------------------------------------
    # Stage 1: decoder/text components only.
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "STAGE 1 — DECODER-ONLY FINE-TUNING"
    )
    print("=" * 72)

    set_decoder_only_trainable(
        model
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.decoder_lr,
        weight_decay=args.weight_decay,
    )

    global_epoch = 0
    no_improvement = 0

    for stage_epoch in range(
        1,
        args.decoder_epochs + 1,
    ):
        global_epoch += 1

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        improved = checkpoint_is_better(
            val_metrics["accuracy"],
            val_metrics[
                "mean_confidence"
            ],
            best_accuracy,
            best_mean_confidence,
        )

        if improved:
            best_accuracy = (
                val_metrics[
                    "accuracy"
                ]
            )

            best_mean_confidence = (
                val_metrics[
                    "mean_confidence"
                ]
            )

            best_epoch = (
                global_epoch
            )

            best_stage = (
                "decoder_only"
            )

            no_improvement = 0

            save_checkpoint(
                checkpoint_path,
                model,
                epoch=global_epoch,
                stage=best_stage,
                val_accuracy=(
                    best_accuracy
                ),
                val_mean_confidence=(
                    best_mean_confidence
                ),
            )

            save_csv(
                args.run_dir
                / (
                    f"epoch_"
                    f"{global_epoch:03d}"
                    f"_val_predictions.csv"
                ),
                val_metrics["rows"],
            )

        else:
            no_improvement += 1

        history.append(
            {
                "epoch": (
                    global_epoch
                ),
                "stage": (
                    "decoder_only"
                ),
                "train_loss": (
                    train_loss
                ),
                "val_accuracy": (
                    val_metrics[
                        "accuracy"
                    ]
                ),
                "val_correct": (
                    val_metrics[
                        "correct"
                    ]
                ),
                "val_total": (
                    val_metrics[
                        "total"
                    ]
                ),
                "val_mean_confidence": (
                    val_metrics[
                        "mean_confidence"
                    ]
                ),
                "best": improved,
            }
        )

        print(
            f"Epoch "
            f"{global_epoch:02d} | "
            f"loss "
            f"{train_loss:.4f} | "
            f"val "
            f"{val_metrics['accuracy']:.2%} "
            f"("
            f"{val_metrics['correct']}/"
            f"{val_metrics['total']}"
            f") | "
            f"conf "
            f"{val_metrics['mean_confidence']:.4f}"
            + (
                " | ↳ new best"
                if improved
                else ""
            )
        )

    # Restore the best model before the more dangerous
    # full-network fine-tune.
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
    # Stage 2: entire network, extremely low LR.
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "STAGE 2 — FULL MODEL MICRO-FINE-TUNING"
    )
    print("=" * 72)

    set_all_trainable(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.full_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=max(
                args.full_epochs,
                1,
            ),
        )
    )

    no_improvement = 0

    for stage_epoch in range(
        1,
        args.full_epochs + 1,
    ):
        global_epoch += 1

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        improved = checkpoint_is_better(
            val_metrics["accuracy"],
            val_metrics[
                "mean_confidence"
            ],
            best_accuracy,
            best_mean_confidence,
        )

        if improved:
            best_accuracy = (
                val_metrics[
                    "accuracy"
                ]
            )

            best_mean_confidence = (
                val_metrics[
                    "mean_confidence"
                ]
            )

            best_epoch = (
                global_epoch
            )

            best_stage = (
                "full_model"
            )

            no_improvement = 0

            save_checkpoint(
                checkpoint_path,
                model,
                epoch=global_epoch,
                stage=best_stage,
                val_accuracy=(
                    best_accuracy
                ),
                val_mean_confidence=(
                    best_mean_confidence
                ),
            )

            save_csv(
                args.run_dir
                / (
                    f"epoch_"
                    f"{global_epoch:03d}"
                    f"_val_predictions.csv"
                ),
                val_metrics["rows"],
            )

        else:
            no_improvement += 1

        history.append(
            {
                "epoch": (
                    global_epoch
                ),
                "stage": (
                    "full_model"
                ),
                "train_loss": (
                    train_loss
                ),
                "val_accuracy": (
                    val_metrics[
                        "accuracy"
                    ]
                ),
                "val_correct": (
                    val_metrics[
                        "correct"
                    ]
                ),
                "val_total": (
                    val_metrics[
                        "total"
                    ]
                ),
                "val_mean_confidence": (
                    val_metrics[
                        "mean_confidence"
                    ]
                ),
                "best": improved,
            }
        )

        print(
            f"Epoch "
            f"{global_epoch:02d} | "
            f"loss "
            f"{train_loss:.4f} | "
            f"val "
            f"{val_metrics['accuracy']:.2%} "
            f"("
            f"{val_metrics['correct']}/"
            f"{val_metrics['total']}"
            f") | "
            f"conf "
            f"{val_metrics['mean_confidence']:.4f}"
            + (
                " | ↳ new best"
                if improved
                else ""
            )
        )

        if (
            no_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping after "
                f"{args.patience} "
                "full-model epochs "
                "without improvement."
            )
            break

    save_csv(
        args.run_dir
        / "history.csv",
        history,
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

    final_val_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    save_csv(
        args.run_dir
        / "best_val_predictions.csv",
        final_val_metrics["rows"],
    )

    summary = [
        {
            "best_epoch": (
                best_epoch
            ),
            "best_stage": (
                best_stage
            ),
            "best_val_accuracy": (
                final_val_metrics[
                    "accuracy"
                ]
            ),
            "best_val_correct": (
                final_val_metrics[
                    "correct"
                ]
            ),
            "best_val_total": (
                final_val_metrics[
                    "total"
                ]
            ),
            "best_val_mean_confidence": (
                final_val_metrics[
                    "mean_confidence"
                ]
            ),
            "zero_shot_accuracy": (
                baseline_metrics[
                    "accuracy"
                ]
            ),
            "zero_shot_correct": (
                baseline_metrics[
                    "correct"
                ]
            ),
        }
    ]

    save_csv(
        args.run_dir
        / "summary.csv",
        summary,
    )

    print()
    print("=" * 72)
    print(
        "PARSEQ FINE-TUNING RESULT"
    )
    print("=" * 72)

    print(
        f"Zero-shot: "
        f"{baseline_metrics['accuracy']:.2%} "
        f"("
        f"{baseline_metrics['correct']}/"
        f"{baseline_metrics['total']}"
        f")"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best stage: "
        f"{best_stage}"
    )

    print(
        f"Best validation: "
        f"{final_val_metrics['accuracy']:.2%} "
        f"("
        f"{final_val_metrics['correct']}/"
        f"{final_val_metrics['total']}"
        f")"
    )

    if (
        final_val_metrics["accuracy"]
        > baseline_metrics["accuracy"]
    ):
        print(
            "🔥 Fine-tuning BEAT "
            "zero-shot PARSeq."
        )
    elif (
        final_val_metrics["accuracy"]
        == baseline_metrics["accuracy"]
    ):
        print(
            "Fine-tuning tied zero-shot "
            "on exact accuracy."
        )
    else:
        print(
            "Zero-shot remains the winner. "
            "Its weights were preserved."
        )

    if not args.evaluate_test:
        print()
        print(
            "Locked test set was "
            "NOT evaluated."
        )
        return

    test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    save_csv(
        args.run_dir
        / "test_predictions.csv",
        test_metrics["rows"],
    )

    print()
    print("=" * 72)
    print(
        "LOCKED TEST RESULT"
    )
    print("=" * 72)

    print(
        f"Test exact accuracy: "
        f"{test_metrics['accuracy']:.2%} "
        f"("
        f"{test_metrics['correct']}/"
        f"{test_metrics['total']}"
        f")"
    )

    print(
        f"Mean confidence: "
        f"{test_metrics['mean_confidence']:.4f}"
    )


if __name__ == "__main__":
    main()
