from __future__ import annotations

import io
from PIL import Image, ImageDraw

from backend.services.binary_trade_service import BinaryTradeService


def create_mock_chart(bullish: bool = True) -> bytes:
    img = Image.new("RGB", (300, 200), (245, 247, 250))
    draw = ImageDraw.Draw(img)
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


def test_binary_trade_service_generates_quotex_signals():
    service = BinaryTradeService()
    screen_mock = {
        "confluence_score": 0.82,
        "neighbour_pattern": {"name": "Hammer", "bias": 0.40, "reliability": "High"},
        "market_trend": {"direction": "Bullish Uptrend", "strength_pct": 75, "bias": 0.25},
        "support_resistance": {"bias": 0.25},
        "technical_indicators": {"rsi_estimate": 28.0, "golden_pocket_near": True, "bias": 0.20},
    }

    signal = service.evaluate_binary_trade(
        cnn_prediction="Up",
        cnn_confidence=0.75,
        screen_analysis=screen_mock,
    )

    assert "CALL" in signal.action
    assert signal.win_probability_pct >= 70.0
    assert signal.is_tradable is True
    assert "Quotex" in signal.platform
    assert "1 Minute" in signal.expiry_recommendation or "Minutes" in signal.expiry_recommendation
    assert len(signal.checklist) >= 1


def test_binary_trade_service_filters_chop_to_wait():
    service = BinaryTradeService()
    screen_chop = {
        "confluence_score": 0.50,
        "neighbour_pattern": {"name": "Doji", "bias": 0.0, "reliability": "Low"},
        "market_trend": {"direction": "Sideways Consolidation", "strength_pct": 40, "bias": 0.0},
        "support_resistance": {"bias": 0.0},
        "technical_indicators": {"rsi_estimate": 50.0, "golden_pocket_near": False, "bias": 0.0},
    }

    signal = service.evaluate_binary_trade(
        cnn_prediction="Up",
        cnn_confidence=0.51,
        screen_analysis=screen_chop,
    )

    assert signal.action == "WAIT / NO TRADE ⏸"
    assert signal.is_tradable is False
    assert "Capital Guard" in signal.setup_grade


def test_binary_trade_api_endpoint(client):
    img_bytes = create_mock_chart(bullish=True)
    data = {"image": (io.BytesIO(img_bytes), "quotex_test.png")}
    response = client.post("/api/binary-trade", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "action" in json_data["data"]
    assert "win_probability_pct" in json_data["data"]
    assert "expiry_recommendation" in json_data["data"]
    assert "setup_grade" in json_data["data"]
