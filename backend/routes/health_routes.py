from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify
from sqlalchemy import text

from backend.extensions import db

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    database_status = "ok"
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        db.session.rollback()
        database_status = "unavailable"

    prediction_service = current_app.extensions["prediction_service"]
    model_status = "available" if prediction_service.is_available else "unavailable"
    overall_status = (
        "ok" if database_status == "ok" and model_status == "available" else "degraded"
    )
    return jsonify(
        {
            "success": True,
            "status": overall_status,
            "database": database_status,
            "model": model_status,
            "model_message": (
                None
                if prediction_service.is_available
                else "A compatible trained model checkpoint is not available."
            ),
        }
    )


@health_bp.get("/metrics")
def metrics():
    path = Path(current_app.config["METRICS_PATH"])
    if not path.is_file():
        return jsonify(
            {
                "success": False,
                "error": {
                    "code": "metrics_unavailable",
                    "message": "No independent test evaluation has been saved yet.",
                },
            }
        ), 404
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current_app.logger.exception("Could not read evaluation metrics")
        return jsonify(
            {
                "success": False,
                "error": {
                    "code": "metrics_unavailable",
                    "message": "Saved evaluation metrics could not be read.",
                },
            }
        ), 503
    return jsonify({"success": True, "data": result})
