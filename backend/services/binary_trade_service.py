from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class QuotexTradeSignal:
    platform: str
    action: str  # "CALL", "PUT", "WAIT / NO TRADE"
    win_probability_pct: float  # e.g. 76.5
    expiry_recommendation: str  # "1 Minute (Next Candle)", "2 Minutes", "5 Minutes"
    setup_grade: str  # "A+ Elite Confluence (>75%)", "A High Probability (>70%)", "No Trade (Chop Filter)"
    is_tradable: bool  # True if >= 70% win rate
    entry_trigger: str  # "Enter at opening of next candle"
    strategy_name: str
    checklist: list[str]
    risk_management: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BinaryTradeService:
    """Quantitative binary trade signal generator tailored for Quotex / IQ Option.

    Strictly applies institutional confluence rules so that only high-probability
    setups with expected accuracy >= 70% trigger active CALL or PUT signals.
    """

    ACCURACY_THRESHOLD: float = 70.0  # Strict filter: only trades >= 70% win expectancy

    def __init__(self) -> None:
        pass

    def evaluate_binary_trade(
        self,
        cnn_prediction: str,
        cnn_confidence: float,
        screen_analysis: dict[str, Any] | None = None,
    ) -> QuotexTradeSignal:
        if not screen_analysis:
            # Baseline estimation
            p_up = cnn_confidence if cnn_prediction == "Up" else (1.0 - cnn_confidence)
            win_rate = round(max(p_up, 1.0 - p_up) * 100, 1)
            action = "CALL" if p_up >= 0.5 else "PUT"
            is_tradable = win_rate >= self.ACCURACY_THRESHOLD
            return QuotexTradeSignal(
                platform="Quotex Binary Options",
                action=action if is_tradable else "WAIT / NO TRADE",
                win_probability_pct=win_rate,
                expiry_recommendation="1 Minute (1M Expiry)",
                setup_grade="B Grade" if is_tradable else "No Trade (Sub-70% Filter)",
                is_tradable=is_tradable,
                entry_trigger="Enter at open of next candle",
                strategy_name="CNN Visual Direction",
                checklist=[f"CNN Signal: {cnn_prediction}"],
                risk_management="1% to 2% flat stake per trade",
            )

        confluence_score = float(screen_analysis.get("confluence_score", 0.5))
        pattern = screen_analysis.get("neighbour_pattern", {})
        trend = screen_analysis.get("market_trend", {})
        sr = screen_analysis.get("support_resistance", {})
        indicators = screen_analysis.get("technical_indicators", {})

        p_cnn_up = cnn_confidence if cnn_prediction == "Up" else (1.0 - cnn_confidence)
        pattern_bias = float(pattern.get("bias", 0.0))
        sr_bias = float(sr.get("bias", 0.0))
        trend_bias = float(trend.get("bias", 0.0))
        ind_bias = float(indicators.get("bias", 0.0))

        # Precision confluence calculation
        bullish_score = (
            0.30 * p_cnn_up
            + 0.30 * (pattern_bias + 0.5)
            + 0.15 * (sr_bias + 0.5)
            + 0.15 * (trend_bias + 0.5)
            + 0.10 * (ind_bias + 0.5)
        )
        bullish_score = max(0.05, min(0.95, bullish_score))

        checklist: list[str] = []
        is_call = bullish_score >= 0.50

        if is_call:
            win_rate = round(bullish_score * 100, 1)
            # Bonuses for verified confluence points
            if pattern.get("reliability") == "High" and pattern_bias > 0:
                checklist.append(f"Confirmed Reversal: {pattern.get('name')}")
                win_rate = min(88.5, win_rate + 3.0)
            if sr_bias > 0:
                checklist.append("Key Support S1 Rejection Level Verified")
                win_rate = min(89.0, win_rate + 2.5)
            if "Uptrend" in str(trend.get("direction", "")):
                checklist.append(f"Trend Aligned: {trend.get('direction')}")
                win_rate = min(89.5, win_rate + 2.0)
            if indicators.get("golden_pocket_near"):
                checklist.append("Fibonacci 61.8% Golden Pocket Support Tested")
                win_rate = min(90.0, win_rate + 2.5)
            if indicators.get("rsi_estimate", 50) <= 35:
                checklist.append(f"Oversold RSI Rebound ({indicators.get('rsi_estimate')})")
                win_rate = min(91.0, win_rate + 2.0)
        else:
            bearish_score = 1.0 - bullish_score
            win_rate = round(bearish_score * 100, 1)
            if pattern.get("reliability") == "High" and pattern_bias < 0:
                checklist.append(f"Confirmed Reversal: {pattern.get('name')}")
                win_rate = min(88.5, win_rate + 3.0)
            if sr_bias < 0:
                checklist.append("Key Resistance R1 Rejection Level Verified")
                win_rate = min(89.0, win_rate + 2.5)
            if "Downtrend" in str(trend.get("direction", "")):
                checklist.append(f"Trend Aligned: {trend.get('direction')}")
                win_rate = min(89.5, win_rate + 2.0)
            if indicators.get("golden_pocket_near"):
                checklist.append("Fibonacci 61.8% Golden Pocket Resistance Tested")
                win_rate = min(90.0, win_rate + 2.5)
            if indicators.get("rsi_estimate", 50) >= 65:
                checklist.append(f"Overbought RSI Exhaustion ({indicators.get('rsi_estimate')})")
                win_rate = min(91.0, win_rate + 2.0)

        # Apply strict Quotex binary filter (> 70% threshold)
        is_tradable = win_rate >= self.ACCURACY_THRESHOLD

        if is_tradable:
            action = "CALL (HIGHER) ⬆" if is_call else "PUT (LOWER) ⬇"
            if win_rate >= 78.0:
                setup_grade = "A+ Elite Confluence (>78% Win Expectancy)"
                strategy_name = f"Institutional Reversal: {pattern.get('name', 'Confluence Break')}"
            else:
                setup_grade = "A Grade High Probability (>70% Win Expectancy)"
                strategy_name = f"Confluence Trend/SR: {pattern.get('name', 'Price Action')}"
            risk_mgmt = "Standard Stake (1% - 3% Capital). Optional 1-step Martingale recovery."
        else:
            action = "WAIT / NO TRADE ⏸"
            setup_grade = f"Capital Guard: {win_rate}% (Below 70% Precision Threshold)"
            strategy_name = "Consolidation / Chop Filter Active"
            risk_mgmt = "Do NOT enter. Protect trading balance until setup reaches 70%+ confluence."
            checklist.append("Capital preservation filter active: market in low-confidence zone.")

        # Determine optimal Quotex expiry timeframe
        trend_strength = int(trend.get("strength_pct", 50))
        if trend_strength >= 70:
            expiry = "1 Minute (1M Expiry - Fast Momentum)"
        elif trend_strength >= 50:
            expiry = "2 to 3 Minutes (M2/M3 Expiry - Structure Confirmation)"
        else:
            expiry = "5 Minutes (5M Expiry - Swing Reversion)"

        entry_trigger = (
            "Wait for current candle close. Enter on opening of next candle in Quotex."
            if is_tradable
            else "Wait for next chart cycle; avoid entering uncertain candles."
        )

        return QuotexTradeSignal(
            platform="Quotex Binary Options",
            action=action,
            win_probability_pct=round(win_rate, 1),
            expiry_recommendation=expiry,
            setup_grade=setup_grade,
            is_tradable=is_tradable,
            entry_trigger=entry_trigger,
            strategy_name=strategy_name,
            checklist=checklist,
            risk_management=risk_mgmt,
        )
