from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy.exc import SQLAlchemyError

from backend.database.models import Prediction
from backend.extensions import db
from backend.services.image_service import ImageValidationError, validate_image_upload
from backend.utils.validators import optional_text_field

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

    return jsonify(
        {
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
    ), 201
