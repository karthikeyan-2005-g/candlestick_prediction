from __future__ import annotations

import pickle

from ml.predict import Predictor, PredictionResult, load_predictor


class PredictionService:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        image_size: int,
        confidence_threshold: float,
    ) -> None:
        self.predictor: Predictor | None = None
        self.load_error: str | None = None
        try:
            self.predictor = load_predictor(
                checkpoint_path,
                image_size=image_size,
                confidence_threshold=confidence_threshold,
            )
        except (
            FileNotFoundError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            self.load_error = str(exc)

    @property
    def is_available(self) -> bool:
        return self.predictor is not None

    def predict(self, image_bytes: bytes) -> PredictionResult:
        if self.predictor is None:
            raise RuntimeError("No compatible trained checkpoint is available.")
        return self.predictor.predict(image_bytes)
