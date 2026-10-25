from __future__ import annotations

import logging
from flask import Blueprint, current_app, jsonify, request
from sqlalchemy.exc import SQLAlchemyError

from backend.database.models import Prediction
from backend.extensions import db
from backend.services.image_service import ImageValidationError, validate_image_upload
from backend.utils.validators import optional_text_field

LOGGER = logging.getLogger(__name__)

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(
            {"success": False, "error": {"code": "missing_image", "message": "Select an image file."}}
        ), 400

    uploaded = request.files["image"]
    try:
        image_bytes, image_sha256 = validate_image_upload(
            uploaded,
            max_bytes=current_app.config["MAX_CONTENT_LENGTH"],
            image_size=current_app.config["IMAGE_SIZE"],
        )
        symbol = optional_text_field(request.form.get("symbol"), "symbol", 24)
        timeframe = optional_text_field(request.form.get("timeframe"), "timeframe", 24)
        enable_web_search = request.form.get("web_search", "").lower() in ("true", "1", "yes", "on")
    except ValueError as exc:
        return jsonify({"success": False, "error": {"code": "invalid_request", "message": str(exc)}}), 400

    service = current_app.extensions["prediction_service"]
    if not service.is_available:
        return jsonify(
            {
                "success": False,
                "error": {
                    "code": "model_unavailable",
                    "message": "A compatible trained model checkpoint is required before predicting.",
                },
            }
        ), 503

    try:
        result = service.predict(image_bytes)
        record = Prediction(
            symbol=symbol,
            timeframe=timeframe,
            prediction=result.prediction,
            confidence=result.confidence,
            model_version=result.model_version,
            image_sha256=image_sha256,
        )
        db.session.add(record)
        db.session.commit()
    except ImageValidationError as exc:
        db.session.rollback()
        return jsonify({"success": False, "error": {"code": "invalid_image", "message": str(exc)}}), 400
    except (RuntimeError, ValueError) as exc:
        db.session.rollback()
        current_app.logger.warning("Inference failed: %s", exc)
        return jsonify(
            {"success": False, "error": {"code": "inference_failed", "message": "The image could not be evaluated."}}
        ), 503
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception("Could not save prediction")
        return jsonify(
            {"success": False, "error": {"code": "database_unavailable", "message": "Prediction could not be saved."}}
        ), 503

    # Optional Web Search market analysis
    web_analysis = None
    if enable_web_search:
        search_service = current_app.extensions.get("web_search_service")
        if search_service:
            try:
                web_analysis = search_service.search_and_analyze(
                    symbol=symbol,
                    timeframe=timeframe,
                    predicted_direction=result.prediction,
                )
            except Exception as exc:
                LOGGER.warning("Web search error during prediction: %s", exc)
                web_analysis = {
                    "query": symbol or result.prediction,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "sentiment": "Neutral",
                    "sentiment_score": 0.50,
                    "summary": "Web search is currently unavailable. Displaying chart prediction only.",
                    "sources": [],
                }

    response_payload = {
        "success": True,
        "prediction": result.prediction,
        "confidence": result.confidence,
        "low_confidence": result.low_confidence,
        "symbol": record.symbol,
        "timeframe": record.timeframe,
        "model_version": result.model_version,
        "prediction_id": record.id,
        "created_at": record.to_dict()["created_at"],
        "message": (
            "Experimental image-classification output. Dataset labels "
            "have not been verified as future candle direction; not financial advice."
        ),
    }

    if web_analysis is not None:
        response_payload["web_search"] = web_analysis

    return jsonify(response_payload), 201


@prediction_bp.route("/web-search", methods=["GET", "POST"])
def search_market():
    """Standalone endpoint for searching market news and technical context."""
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        prediction = data.get("prediction")
    else:
        symbol = request.args.get("symbol")
        timeframe = request.args.get("timeframe")
        prediction = request.args.get("prediction")

    search_service = current_app.extensions.get("web_search_service")
    if not search_service:
        return jsonify({"success": False, "error": "Search service is unavailable."}), 503

    try:
        analysis = search_service.search_and_analyze(
            symbol=symbol,
            timeframe=timeframe,
            predicted_direction=prediction,
        )
        return jsonify({"success": True, "data": analysis}), 200
    except Exception as exc:
        LOGGER.exception("Market web search failed")
        return jsonify({"success": False, "error": str(exc)}), 500
