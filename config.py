from __future__ import annotations

import math
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables from the project root
load_dotenv(PROJECT_ROOT / ".env")


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _integer_setting(name: str, default: int) -> int:
    """Read a positive integer from environment variables."""
    value = int(os.getenv(name, str(default)))

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

    return value


def _float_setting(name: str, default: float) -> float:
    """Read a floating-point setting from environment variables."""
    value = float(os.getenv(name, str(default)))

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")

    return value


# ==========================================
# MAIN CONFIGURATION
# ==========================================

class Config:

    # Flask security
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)

    # Database configuration
    # SQLite is used by default.
    # Set DATABASE_URL in .env to use another database.
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(PROJECT_ROOT / 'candlestick.db').as_posix()}",
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload configuration: 5 MB by default
    MAX_CONTENT_LENGTH = _integer_setting(
        "MAX_CONTENT_LENGTH",
        5 * 1024 * 1024,
    )

    # Machine learning model checkpoint
    MODEL_PATH = str(
        (
            PROJECT_ROOT
            / os.getenv(
                "MODEL_PATH",
                "models/candlestick_cnn.pth",
            )
        ).resolve()
    )

    # Prediction confidence threshold
    CONFIDENCE_THRESHOLD = _float_setting(
        "CONFIDENCE_THRESHOLD",
        0.60,
    )

    if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        raise ValueError(
            "CONFIDENCE_THRESHOLD must be between 0 and 1."
        )

    # Image preprocessing size
    IMAGE_SIZE = _integer_setting(
        "IMAGE_SIZE",
        128,
    )

    # Model evaluation metrics
    METRICS_PATH = str(
        (
            PROJECT_ROOT
            / os.getenv(
                "METRICS_PATH",
                "models/evaluation_metrics.json",
            )
        ).resolve()
    )

    # Application logging
    LOG_LEVEL = os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).upper()

    JSON_SORT_KEYS = False


# ==========================================
# TEST CONFIGURATION
# ==========================================

class TestConfig(Config):

    TESTING = True

    # In-memory SQLite database for tests
    SQLALCHEMY_DATABASE_URI = "sqlite://"

    # Deliberately missing model checkpoint for tests
    MODEL_PATH = str(
        PROJECT_ROOT
        / "models"
        / "test-checkpoint-does-not-exist.pth"
    )