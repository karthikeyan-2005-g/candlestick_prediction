from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image

from backend.services.prediction_service import PredictionService
from ml.dataset import CLASS_TO_INDEX, INDEX_TO_CLASS
from ml.model import CandlestickCNN, MODEL_VERSION
from ml.preprocessing import preprocessing_config


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["database"] == "ok"
    assert payload["model"] == "available"


def test_frontend_pages_and_assets_are_served_same_origin(client):
    assert client.get("/").status_code == 200
    assert client.get("/history.html").status_code == 200
    assert client.get("/prediction.html").status_code == 200
    assert client.get("/about.html").status_code == 200
    assert client.get("/frontend/js/app.js").status_code == 200


def test_prediction_endpoint_saves_record(client, png_bytes):
    response = client.post(
        "/api/predict",
        data={
            "image": (BytesIO(png_bytes), "chart.png"),
            "symbol": "aapl",
            "timeframe": "1d",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["prediction"] == "Bullish"
    assert payload["confidence"] == 0.81
    assert payload["symbol"] == "AAPL"
    assert payload["prediction_id"] == 1


def test_temporary_checkpoint_inference_api_and_history_smoke(
    client, app, png_bytes, tmp_path
):
    """Integration smoke test with untrained temporary weights, not a forecast."""
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": CandlestickCNN().state_dict(),
            "class_to_idx": CLASS_TO_INDEX,
            "classes": list(INDEX_TO_CLASS),
            "image_size": 128,
            "preprocessing": preprocessing_config(128),
            "model_version": MODEL_VERSION,
        },
        checkpoint_path,
    )
    app.extensions["prediction_service"] = PredictionService(
        checkpoint_path=str(checkpoint_path),
        image_size=128,
        confidence_threshold=0.6,
    )
    assert app.extensions["prediction_service"].is_available

    response = client.post(
        "/api/predict",
        data={"image": (BytesIO(png_bytes), "chart.png"), "symbol": "TEST"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    prediction = response.get_json()
    assert prediction["prediction"] in {"Down", "Up"}
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["model_version"] == MODEL_VERSION

    history = client.get(f"/api/history/{prediction['prediction_id']}")
    assert history.status_code == 200
    saved = history.get_json()["data"]
    assert saved["image_sha256"]
    assert saved["symbol"] == "TEST"
    assert saved["prediction"] == prediction["prediction"]


def test_prediction_rejects_missing_image(client):
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "missing_image"


def test_prediction_rejects_empty_upload(client):
    response = client.post(
        "/api/predict",
        data={"image": (BytesIO(b""), "empty.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_prediction_rejects_unsupported_and_corrupt_uploads(client):
    unsupported = client.post(
        "/api/predict",
        data={"image": (BytesIO(b"data"), "image.gif")},
        content_type="multipart/form-data",
    )
    corrupt = client.post(
        "/api/predict",
        data={"image": (BytesIO(b"not an image"), "image.png")},
        content_type="multipart/form-data",
    )
    assert unsupported.status_code == 400
    assert corrupt.status_code == 400


def test_prediction_rejects_oversized_upload(client):
    response = client.post(
        "/api/predict",
        data={"image": (BytesIO(b"x" * (2 * 1024 * 1024)), "large.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413


def test_missing_model_returns_service_unavailable(client, app):
    class MissingService:
        is_available = False
        load_error = "checkpoint missing"

    app.extensions["prediction_service"] = MissingService()
    response = client.post(
        "/api/predict",
        data={"image": (BytesIO(_small_png()), "chart.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "model_unavailable"


def test_history_pagination_and_symbol_filter(client, png_bytes):
    _create_prediction(client, png_bytes, "AAPL")
    _create_prediction(client, png_bytes, "MSFT")
    response = client.get("/api/history?page=1&per_page=1&symbol=aapl")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["pagination"] == {"page": 1, "per_page": 1, "total": 1, "pages": 1}
    assert payload["data"][0]["symbol"] == "AAPL"


def test_history_detail_missing_id(client):
    response = client.get("/api/history/999")
    assert response.status_code == 404


def test_record_actual_outcome(client, png_bytes):
    created = _create_prediction(client, png_bytes, "AAPL").get_json()
    response = client.patch(
        f"/api/history/{created['prediction_id']}/outcome",
        json={"actual_direction": "Bullish", "observed_at": "2025-01-02T15:00:00Z"},
    )
    assert response.status_code == 200
    assert response.get_json()["data"]["actual_direction"] == "Bullish"
    assert response.get_json()["data"]["observed_at"].endswith("+00:00")


def test_reject_invalid_outcome(client, png_bytes):
    created = _create_prediction(client, png_bytes, "AAPL").get_json()
    response = client.patch(
        f"/api/history/{created['prediction_id']}/outcome",
        json={"actual_direction": "Uncertain"},
    )
    assert response.status_code == 400


def test_metrics_endpoint_reports_not_evaluated(client):
    response = client.get("/api/metrics")
    assert response.status_code == 404
    assert response.get_json()["success"] is False


def _create_prediction(client, image_bytes, symbol):
    return client.post(
        "/api/predict",
        data={
            "image": (BytesIO(image_bytes), "chart.png"),
            "symbol": symbol,
            "timeframe": "1d",
        },
        content_type="multipart/form-data",
    )


def _small_png():
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()
