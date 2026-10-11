from __future__ import annotations

import os
import math
import secrets
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _integer_setting(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://localhost/candlestick_db",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = _integer_setting("MAX_CONTENT_LENGTH", 5 * 1024 * 1024)
    MODEL_PATH = str(
        (PROJECT_ROOT / os.getenv("MODEL_PATH", "models/candlestick_cnn.pth"))
        .resolve()
    )
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    IMAGE_SIZE = _integer_setting("IMAGE_SIZE", 128)
    METRICS_PATH = str(
        (PROJECT_ROOT / os.getenv("METRICS_PATH", "models/evaluation_metrics.json"))
        .resolve()
    )
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    JSON_SORT_KEYS = False

    if not math.isfinite(CONFIDENCE_THRESHOLD) or not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1.")


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    MODEL_PATH = str(PROJECT_ROOT / "models" / "test-checkpoint-does-not-exist.pth")
