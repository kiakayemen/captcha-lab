from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from preprocess import ensure_bgr

DIGITS_ONLY = re.compile(r"\D")
EXPECTED_DIGITS = 3
DEFAULT_PARSEQ_CHECKPOINT = Path(__file__).resolve().parent / "experiments/parseq_finetune/run_001/best_model.pt"


@dataclass(frozen=True)
class OCRResult:
    variant: str
    prediction: str
    confidence: float

    @property
    def valid(self) -> bool:
        return len(self.prediction) == EXPECTED_DIGITS and self.prediction.isdigit()


def clean_prediction(text: str) -> str:
    return DIGITS_ONLY.sub("", text)


def add_white_padding(image: np.ndarray, padding: int = 15) -> np.ndarray:
    if padding < 0:
        raise ValueError("padding must be non-negative")
    return cv2.copyMakeBorder(
        ensure_bgr(image),
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _device_for(gpu: bool) -> torch.device:
    if gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PARSeqReader:
    """Adapter exposing the shared OCR boundary over fine-tuned PARSeq."""

    def __init__(self, model: Any, transform: Any, device: torch.device) -> None:
        self.model = model.eval()
        self.transform = transform
        self.device = device

    @torch.inference_mode()
    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        prepared = add_white_padding(image, padding=15)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        logits = self.model(tensor, max_length=EXPECTED_DIGITS)
        labels, confidences = self.model.tokenizer.decode(logits.softmax(-1))
        prediction = clean_prediction(str(labels[0]))
        confidence = confidences[0]
        if torch.is_tensor(confidence):
            confidence = confidence.detach().float().mean().item()
        return prediction, float(confidence)

    def readtext(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        raise RuntimeError(
            "PARSeqReader recognizes three-digit CAPTCHA tiles only. "
            "Read the target from the CAPTCHA DOM or pass --target."
        )


def build_reader(
    gpu: bool = False,
    checkpoint_path: Path = DEFAULT_PARSEQ_CHECKPOINT,
) -> PARSeqReader:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PARSeq checkpoint was not found: {checkpoint_path}")
    device = _device_for(gpu)
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).to(device)
    try:
        from strhub.data.module import SceneTextDataModule
    except ImportError as error:
        raise RuntimeError("PARSeq dependencies are not installed. Run: pip install -r requirements.txt") from error
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    return PARSeqReader(model, transform, device)


def recognize(reader: Any, image: np.ndarray, variant: str) -> OCRResult:
    prediction, confidence = reader.recognize(image)
    return OCRResult(variant=variant, prediction=prediction, confidence=confidence)
