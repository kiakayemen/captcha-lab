from __future__ import annotations

import torch
from torch import nn


class CaptchaCNN(nn.Module):
    def __init__(self, dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 12, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(256, 10) for _ in range(3)])

    def forward(self, images: torch.Tensor):
        shared = self.shared(self.features(images))
        return tuple(head(shared) for head in self.heads)
