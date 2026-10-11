from __future__ import annotations

import pytest

from backend.database.models import Prediction
from backend.extensions import db


def test_prediction_persists_and_serializes(app):
    with app.app_context():
        record = Prediction(
            symbol="msft",
            timeframe="1d",
            prediction="Bullish",
            confidence=0.72,
            model_version="cnn_test",
            image_sha256="a" * 64,
        )
        db.session.add(record)
        db.session.commit()
        loaded = db.session.get(Prediction, record.id)
        assert loaded is not None
        assert loaded.symbol == "MSFT"
        assert loaded.to_dict()["prediction"] == "Bullish"


@pytest.mark.parametrize("dataset_class", ["Down", "Up"])
def test_prediction_database_accepts_archive_class_names(app, dataset_class):
    with app.app_context():
        record = Prediction(
            prediction=dataset_class,
            confidence=0.72,
            model_version="cnn_test",
            image_sha256="c" * 64,
        )
        db.session.add(record)
        db.session.commit()
        assert db.session.get(Prediction, record.id).prediction == dataset_class


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prediction", "Sideways"),
        ("actual_direction", "Uncertain"),
        ("confidence", 1.1),
    ],
)
def test_prediction_model_validation(app, field, value):
    with app.app_context():
        kwargs = {
            "prediction": "Bearish",
            "confidence": 0.5,
            "model_version": "cnn_test",
            "image_sha256": "b" * 64,
        }
        kwargs[field] = value
        with pytest.raises(ValueError):
            Prediction(**kwargs)
