from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from backend.services.screen_analysis_service import ScreenAnalysisService


def create_mock_chart(bullish: bool = True) -> bytes:
    img = Image.new("RGB", (300, 200), (245, 247, 250))
    draw = ImageDraw.Draw(img)

    # Draw 5 candles
    for i in range(5):
        x = 30 + i * 50
        is_green = bullish if i >= 3 else not bullish
        color = (29, 150, 99) if is_green else (211, 73, 79)

        y_top = 50 + (10 - i * 8) if bullish else 50 + (i * 8)
        y_bot = y_top + 30
        draw.line([(x, y_top - 10), (x, y_bot + 10)], fill=color, width=1)
        draw.rectangle([(x - 8, y_top), (x + 8, y_bot)], fill=color)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_screen_analysis_detects_candles_and_patterns():
    svc = ScreenAnalysisService()
    img_bytes = create_mock_chart(bullish=True)
    res = svc.analyze_chart(img_bytes)

    assert res.candles_detected >= 2
    assert "name" in res.neighbour_pattern
    assert "pattern_type" in res.neighbour_pattern
    assert "direction" in res.market_trend
    assert "strength_pct" in res.market_trend
    assert "support_levels" in res.support_resistance
    assert "resistance_levels" in res.support_resistance
    assert "rsi_estimate" in res.technical_indicators
    assert "fibonacci_status" in res.technical_indicators
    assert 0.0 <= res.confluence_score <= 1.0
    assert res.annotated_image.startswith("data:image/png;base64,")


def test_screen_analysis_endpoint(client):
    img_bytes = create_mock_chart(bullish=False)
    data = {"image": (io.BytesIO(img_bytes), "test_chart.png")}
    response = client.post("/api/screen-analysis", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "neighbour_pattern" in json_data["data"]
    assert "support_resistance" in json_data["data"]
    assert "market_trend" in json_data["data"]
    assert "annotated_image" in json_data["data"]


def test_predict_returns_screen_analysis_metadata(client):
    img_bytes = create_mock_chart(bullish=True)
    data = {
        "image": (io.BytesIO(img_bytes), "sample.png"),
        "symbol": "BTCUSDT",
        "timeframe": "1d",
        "web_search": "false",
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["prediction"] in ("Up", "Down")
    assert "screen_analysis" in json_data
    assert "annotated_image" in json_data
    assert "market_trend" in json_data
    assert "neighbour_pattern" in json_data
    assert "support_resistance" in json_data
