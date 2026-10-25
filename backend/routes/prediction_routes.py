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

        # Screen Analysis & Technical Pattern Recognition
        screen_service = current_app.extensions.get("screen_analysis_service")
        screen_result = None
        if screen_service:
            try:
                screen_result = screen_service.analyze_chart(image_bytes)
            except Exception as exc:
                LOGGER.warning("Screen analysis failed: %s", exc)

        # Confluence Fusion (CNN + Screen Analysis S/R, Pattern, Trend)
        p_cnn_up = result.confidence if result.prediction == "Up" else (1.0 - result.confidence)
        if screen_result:
            pattern_bias = screen_result.neighbour_pattern["bias"]
            sr_bias = screen_result.support_resistance["bias"]
            trend_bias = screen_result.market_trend["bias"]
            confluence_up = (
                0.40 * p_cnn_up
                + 0.35 * (pattern_bias + 0.5)
                + 0.15 * (sr_bias + 0.5)
                + 0.10 * (trend_bias + 0.5)
            )
            confluence_up = max(0.05, min(0.95, confluence_up))
            if confluence_up >= 0.51:
                final_prediction = "Up"
                final_confidence = round(confluence_up, 4)
            else:
                final_prediction = "Down"
                final_confidence = round(1.0 - confluence_up, 4)
        else:
            final_prediction = result.prediction
            final_confidence = result.confidence

        record = Prediction(
            symbol=symbol,
            timeframe=timeframe,
            prediction=final_prediction,
            confidence=final_confidence,
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
                    predicted_direction=final_prediction,
                )
            except Exception as exc:
                LOGGER.warning("Web search error during prediction: %s", exc)
                web_analysis = {
                    "query": symbol or final_prediction,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "sentiment": "Neutral",
                    "sentiment_score": 0.50,
                    "summary": "Web search is currently unavailable. Displaying chart prediction only.",
                    "sources": [],
                }

    # Quotex Binary Trade Signal Evaluation (>70% Precision Strategy)
    binary_service = current_app.extensions.get("binary_trade_service")
    quotex_signal = None
    if binary_service:
        try:
            quotex_signal = binary_service.evaluate_binary_trade(
                cnn_prediction=result.prediction,
                cnn_confidence=result.confidence,
                screen_analysis=screen_result.to_dict() if screen_result else None,
            )
        except Exception as exc:
            LOGGER.warning("Binary trade evaluation error: %s", exc)

    response_payload = {
        "success": True,
        "prediction": final_prediction,
        "confidence": final_confidence,
        "low_confidence": final_confidence < current_app.config["CONFIDENCE_THRESHOLD"],
        "symbol": record.symbol,
        "timeframe": record.timeframe,
        "model_version": result.model_version,
        "prediction_id": record.id,
        "created_at": record.to_dict()["created_at"],
        "screen_analysis": screen_result.to_dict() if screen_result else None,
        "annotated_image": screen_result.annotated_image if screen_result else None,
        "market_trend": screen_result.market_trend if screen_result else None,
        "support_resistance": screen_result.support_resistance if screen_result else None,
        "neighbour_pattern": screen_result.neighbour_pattern if screen_result else None,
        "technical_indicators": screen_result.technical_indicators if screen_result else None,
        "binary_trade": quotex_signal.to_dict() if quotex_signal else None,
        "message": (
            "Technical and image-classification output combining computer vision S/R analysis, "
            "neighbour candlestick patterns, and trained neural network features. Not financial advice."
        ),
    }

    if web_analysis is not None:
        response_payload["web_search"] = web_analysis

    return jsonify(response_payload), 201


@prediction_bp.post("/binary-trade")
def binary_trade_analysis():
    """Dedicated endpoint for Quotex binary options (CALL / PUT >70% accuracy)."""
    if "image" not in request.files:
        return jsonify(
            {"success": False, "error": {"code": "missing_image", "message": "Select an image file."}}
        ), 400

    uploaded = request.files["image"]
    try:
        image_bytes, _ = validate_image_upload(
            uploaded,
            max_bytes=current_app.config["MAX_CONTENT_LENGTH"],
            image_size=current_app.config["IMAGE_SIZE"],
        )
    except (ValueError, ImageValidationError) as exc:
        return jsonify({"success": False, "error": {"code": "invalid_image", "message": str(exc)}}), 400

    pred_service = current_app.extensions["prediction_service"]
    screen_service = current_app.extensions.get("screen_analysis_service")
    binary_service = current_app.extensions.get("binary_trade_service")

    try:
        cnn_res = pred_service.predict(image_bytes)
        sa_res = screen_service.analyze_chart(image_bytes) if screen_service else None
        signal = binary_service.evaluate_binary_trade(
            cnn_prediction=cnn_res.prediction,
            cnn_confidence=cnn_res.confidence,
            screen_analysis=sa_res.to_dict() if sa_res else None,
        )
        return jsonify({
            "success": True,
            "data": signal.to_dict(),
            "annotated_image": sa_res.annotated_image if sa_res else None,
        }), 200
    except Exception as exc:
        LOGGER.exception("Binary trade analysis failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@prediction_bp.post("/screen-analysis")
def screen_analysis_only():
    """Analyze chart screen directly for S/R levels, neighbour patterns, and trend."""
    if "image" not in request.files:
        return jsonify(
            {"success": False, "error": {"code": "missing_image", "message": "Select an image file."}}
        ), 400

    uploaded = request.files["image"]
    try:
        image_bytes, _ = validate_image_upload(
            uploaded,
            max_bytes=current_app.config["MAX_CONTENT_LENGTH"],
            image_size=current_app.config["IMAGE_SIZE"],
        )
    except (ValueError, ImageValidationError) as exc:
        return jsonify({"success": False, "error": {"code": "invalid_image", "message": str(exc)}}), 400

    screen_service = current_app.extensions.get("screen_analysis_service")
    if not screen_service:
        return jsonify({"success": False, "error": "Screen analysis service is unavailable."}), 503

    try:
        analysis = screen_service.analyze_chart(image_bytes)
        return jsonify({"success": True, "data": analysis.to_dict()}), 200
    except Exception as exc:
        LOGGER.exception("Screen analysis failed")
        return jsonify({"success": False, "error": str(exc)}), 500


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
