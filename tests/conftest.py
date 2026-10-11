from __future__ import annotations

import io

import pytest
from PIL import Image

from app import create_application
from backend.extensions import db


class StubPredictor:
    def __init__(self, prediction="Bullish", confidence=0.81):
        self.prediction = prediction
        self.confidence = confidence
        self.model_version = "test_model"

    def predict(self, _image_bytes):
        from ml.predict import PredictionResult

        return PredictionResult(
            prediction=self.prediction,
            confidence=self.confidence,
            model_version=self.model_version,
            raw_class="Bullish",
            low_confidence=False,
        )


class StubService:
    is_available = True
    load_error = None

    def __init__(self):
        self.predictor = StubPredictor()

    def predict(self, image_bytes):
        return self.predictor.predict(image_bytes)


@pytest.fixture()
def app(tmp_path):
    class TestSettings:
        TESTING = True
        SECRET_KEY = "test-only-secret"
        SQLALCHEMY_DATABASE_URI = "sqlite://"
        SQLALCHEMY_TRACK_MODIFICATIONS = False
        MAX_CONTENT_LENGTH = 1024 * 1024
        MODEL_PATH = str(tmp_path / "missing.pth")
        IMAGE_SIZE = 128
        CONFIDENCE_THRESHOLD = 0.6
        METRICS_PATH = str(tmp_path / "metrics.json")
        LOG_LEVEL = "CRITICAL"

    application = create_application(TestSettings)
    with application.app_context():
        db.create_all()
        application.extensions["prediction_service"] = StubService()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 180, 90)).save(buffer, format="PNG")
    return buffer.getvalue()
