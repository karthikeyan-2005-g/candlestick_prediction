from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from sqlalchemy import func, select

from backend.database.models import ACTUAL_VALUES, Prediction
from backend.extensions import db

history_bp = Blueprint("history", __name__)


@history_bp.get("/history")
def list_history():
    try:
        page = int(request.args.get("page", "1"))
        per_page = int(request.args.get("per_page", "20"))
    except ValueError:
        return _error("page and per_page must be integers.", 400)
    if page < 1 or not 1 <= per_page <= 100:
        return _error("page must be positive and per_page must be between 1 and 100.", 400)

    query = select(Prediction)
    symbol = request.args.get("symbol")
    if symbol is not None:
        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 24:
            return _error("symbol must contain between 1 and 24 characters.", 400)
        query = query.where(Prediction.symbol == symbol)

    total = db.session.scalar(
        select(func.count()).select_from(query.subquery())
    ) or 0
    entries = db.session.scalars(
        query.order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    ).all()
    return jsonify(
        {
            "success": True,
            "data": [entry.to_dict() for entry in entries],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page,
            },
        }
    )


@history_bp.get("/history/<int:prediction_id>")
def get_prediction(prediction_id: int):
    record = db.session.get(Prediction, prediction_id)
    if record is None:
        return _error("Prediction was not found.", 404, "not_found")
    return jsonify({"success": True, "data": record.to_dict()})


@history_bp.patch("/history/<int:prediction_id>/outcome")
def record_outcome(prediction_id: int):
    record = db.session.get(Prediction, prediction_id)
    if record is None:
        return _error("Prediction was not found.", 404, "not_found")
    if not request.is_json:
        return _error("Request body must be JSON.", 400)

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error("Request body must be a JSON object.", 400)
    direction = body.get("actual_direction")
    if direction not in ACTUAL_VALUES:
        return _error("actual_direction must be Bullish or Bearish.", 400)

    observed_at_value = body.get("observed_at")
    if observed_at_value is None:
        observed_at = datetime.now(timezone.utc)
    elif isinstance(observed_at_value, str):
        try:
            observed_at = datetime.fromisoformat(
                observed_at_value.replace("Z", "+00:00")
            )
        except ValueError:
            return _error("observed_at must be an ISO 8601 timestamp.", 400)
        if observed_at.tzinfo is None:
            return _error("observed_at must include a timezone.", 400)
    else:
        return _error("observed_at must be an ISO 8601 timestamp.", 400)

    record.actual_direction = direction
    record.observed_at = observed_at
    db.session.commit()
    return jsonify({"success": True, "data": record.to_dict()})


def _error(message: str, status: int, code: str = "invalid_request"):
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status
