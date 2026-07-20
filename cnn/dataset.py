from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_WIDTH = 96
IMAGE_HEIGHT = 32


@dataclass(frozen=True)
class Sample:
    screenshot: str
    tile: int
    tile_path: Path
    label_text: str


def load_samples(csv_path: Path, project_root: Path) -> list[Sample]:
    """Load one unique sample per screenshot/tile from preprocessing_results.csv."""
    samples: list[Sample] = []
    seen: set[tuple[str, int]] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {"image", "tile", "ground_truth", "tile_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            screenshot = row["image"].strip()
            tile = int(row["tile"])
            key = (screenshot, tile)

            if key in seen:
                continue
            seen.add(key)

            label_text = row["ground_truth"].strip()
            if len(label_text) != 3 or not label_text.isdigit():
                raise ValueError(
                    f"Invalid label {label_text!r} on CSV row {row_number}"
                )

            tile_path = project_root / row["tile_path"].strip()
            samples.append(
                Sample(
                    screenshot=screenshot,
                    tile=tile,
                    tile_path=tile_path,
                    label_text=label_text,
                )
            )

    samples.sort(key=lambda sample: (sample.screenshot, sample.tile))

    if not samples:
        raise RuntimeError(f"No samples found in {csv_path}")

    return samples


def make_transform(training: bool) -> transforms.Compose:
    operations: list[object] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    ]

    if training:
        # Small changes only. Strong augmentation could alter the digits.
        operations.append(
            transforms.RandomAffine(
                degrees=3,
                translate=(0.03, 0.05),
                scale=(0.95, 1.05),
                fill=255,
            )
        )

    operations.extend(
        [
            transforms.ToTensor(),
            # Map pixel values from [0, 1] to approximately [-1, 1].
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    return transforms.Compose(operations)


class CaptchaTileDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        training: bool,
    ) -> None:
        self.samples = samples
        self.transform = make_transform(training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        if not sample.tile_path.exists():
            raise FileNotFoundError(
                f"Tile image does not exist: {sample.tile_path}"
            )

        with Image.open(sample.tile_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        # "350" becomes tensor([3, 5, 0]).
        label_tensor = torch.tensor(
            [int(character) for character in sample.label_text],
            dtype=torch.long,
        )

        return image_tensor, label_tensor
