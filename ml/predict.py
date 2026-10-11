from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ml.checkpoint import load_checkpoint
from ml.dataset import CLASS_TO_INDEX, INDEX_TO_CLASS
from ml.model import CandlestickCNN, MODEL_VERSION
from ml.preprocessing import preprocess_image_bytes


@dataclass(frozen=True)
class PredictionResult:
    prediction: str
    confidence: float
    model_version: str
    raw_class: str
    low_confidence: bool


class Predictor:
    def __init__(
        self,
        model: CandlestickCNN,
        device: torch.device,
        image_size: int,
        confidence_threshold: float,
        model_version: str,
    ) -> None:
        self.model = model
        self.device = device
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.model_version = model_version

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image_tensor = preprocess_image_bytes(image_bytes, self.image_size)
        with torch.inference_mode():
            logits = self.model(image_tensor.unsqueeze(0).to(self.device))
            if logits.shape != (1, len(INDEX_TO_CLASS)):
                raise RuntimeError("Model returned an unexpected output shape.")
            probabilities = torch.softmax(logits, dim=1)[0]
            winning_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[winning_index].item())
        raw_class = INDEX_TO_CLASS[winning_index]
        return PredictionResult(
            prediction=raw_class,
            confidence=confidence,
            model_version=self.model_version,
            raw_class=raw_class,
            low_confidence=confidence < self.confidence_threshold,
        )


def load_predictor(
    checkpoint_path: str | Path,
    *,
    image_size: int,
    confidence_threshold: float,
) -> Predictor:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint does not exist: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(path, map_location=device, image_size=image_size)
    model_version = checkpoint["model_version"]

    model = CandlestickCNN(num_classes=len(INDEX_TO_CLASS))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return Predictor(
        model,
        device,
        image_size,
        confidence_threshold,
        model_version,
    )
