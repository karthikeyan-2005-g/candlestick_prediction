from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask

from backend.extensions import db
from backend.services.prediction_service import PredictionService
from backend.services.web_search_service import WebSearchService


def create_app(config_object: type | object | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object or "config.Config")
    app.config["FRONTEND_DIR"] = str(Path(__file__).resolve().parent.parent / "frontend")
    db.init_app(app)

    app.extensions["prediction_service"] = PredictionService(
        checkpoint_path=app.config["MODEL_PATH"],
        image_size=app.config["IMAGE_SIZE"],
        confidence_threshold=app.config["CONFIDENCE_THRESHOLD"],
    )

    app.extensions["web_search_service"] = WebSearchService()

    from backend.database import models as _models  # noqa: F401
    from backend.routes.health_routes import health_bp
    from backend.routes.history_routes import history_bp
    from backend.routes.prediction_routes import prediction_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")
    app.register_blueprint(prediction_bp, url_prefix="/api")
    return app
