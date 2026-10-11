from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, Index, String, func
from sqlalchemy.orm import validates

from backend.extensions import db

PREDICTION_VALUES = ("Bullish", "Bearish", "Uncertain", "Down", "Up")
ACTUAL_VALUES = ("Bullish", "Bearish")


class Prediction(db.Model):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "prediction IN ('Bullish', 'Bearish', 'Uncertain', 'Down', 'Up')",
            name="ck_predictions_prediction",
        ),
        CheckConstraint(
            "actual_direction IS NULL OR actual_direction IN ('Bullish', 'Bearish')",
            name="ck_predictions_actual_direction",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_predictions_confidence",
        ),
        CheckConstraint(
            "length(image_sha256) = 64",
            name="ck_predictions_image_sha256_length",
        ),
        Index("ix_predictions_created_at_id", "created_at", "id"),
        Index("ix_predictions_symbol_created_at", "symbol", "created_at"),
    )

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(String(24), nullable=True)
    timeframe = db.Column(String(24), nullable=True)
    prediction = db.Column(String(16), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_version = db.Column(String(64), nullable=False)
    image_sha256 = db.Column(String(64), nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    actual_direction = db.Column(String(16), nullable=True)
    observed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    @validates("symbol")
    def validate_symbol(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized or len(normalized) > 24:
            raise ValueError("Symbol must contain between 1 and 24 characters.")
        return normalized

    @validates("timeframe")
    def validate_timeframe(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 24:
            raise ValueError("Timeframe must contain between 1 and 24 characters.")
        return normalized

    @validates("prediction")
    def validate_prediction(self, _key: str, value: str) -> str:
        if value not in PREDICTION_VALUES:
            raise ValueError("Prediction must be Bullish, Bearish, Uncertain, Down, or Up.")
        return value

    @validates("actual_direction")
    def validate_actual_direction(self, _key: str, value: str | None) -> str | None:
        if value is not None and value not in ACTUAL_VALUES:
            raise ValueError("Actual direction must be Bullish or Bearish.")
        return value

    @validates("confidence")
    def validate_confidence(self, _key: str, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        return float(value)

    def to_dict(self) -> dict[str, object]:
        created_at = self.created_at
        if created_at is not None and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        observed_at = self.observed_at
        if observed_at is not None and observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "image_sha256": self.image_sha256,
            "created_at": created_at.isoformat() if created_at else None,
            "actual_direction": self.actual_direction,
            "observed_at": observed_at.isoformat() if observed_at else None,
        }
