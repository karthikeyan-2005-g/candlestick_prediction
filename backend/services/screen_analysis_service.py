from __future__ import annotations

import base64
import io
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class CandleInfo:
    x_center: int
    x_left: int
    x_right: int
    y_high: int
    y_low: int
    y_open: int
    y_close: int
    is_bullish: bool
    body_height: float
    upper_wick: float
    lower_wick: float
    range_height: float


@dataclass
class NeighbourPattern:
    name: str
    pattern_type: str
    reliability: str
    bias: float
    description: str


@dataclass
class MarketTrend:
    direction: str
    strength_pct: int
    slope: float
    bias: float


@dataclass
class SupportResistanceInfo:
    support_levels: list[int]
    resistance_levels: list[int]
    nearest_support: int | None
    nearest_resistance: int | None
    distance_to_support_pct: float | None
    distance_to_resistance_pct: float | None
    position_summary: str
    bias: float


@dataclass
class TechnicalIndicators:
    rsi_estimate: float
    rsi_condition: str
    ema_trend: str
    golden_pocket_near: bool
    fibonacci_status: str
    fibonacci_levels: dict[str, int]
    bias: float


@dataclass
class ScreenAnalysisResult:
    candles_detected: int
    neighbour_pattern: dict[str, Any]
    market_trend: dict[str, Any]
    support_resistance: dict[str, Any]
    technical_indicators: dict[str, Any]
    confluence_score: float
    confluence_direction: str
    annotated_image: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candles_detected": self.candles_detected,
            "neighbour_pattern": self.neighbour_pattern,
            "market_trend": self.market_trend,
            "support_resistance": self.support_resistance,
            "technical_indicators": self.technical_indicators,
            "confluence_score": round(self.confluence_score, 4),
            "confluence_direction": self.confluence_direction,
            "annotated_image": self.annotated_image,
        }


class ScreenAnalysisService:
    def __init__(self) -> None:
        pass

    def analyze_chart(self, image_bytes: bytes) -> ScreenAnalysisResult:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_img.size
        arr = np.array(pil_img)

        # 1. Segment candles
        candles = self._extract_candles(arr, w, h)

        # Fallback if no candles detected (e.g. non-standard format)
        if len(candles) < 2:
            candles = self._synthetic_projection_candles(arr, w, h)

        # 2. Support & Resistance Analysis
        sr_info = self._analyze_support_resistance(candles, h)

        # 3. Neighbour Candlestick Pattern Recognition
        pattern = self._detect_neighbour_patterns(candles)

        # 4. Market Trend Analysis
        trend = self._analyze_market_trend(candles, h)

        # 5. Technical Indicators Analysis (Fibonacci Retracement, RSI Proxy, EMA Alignment)
        indicators = self._analyze_technical_indicators(candles, trend, h)

        # 6. Confluence Score Calculation
        # Combines pattern (+0.50 to -0.50), SR (+0.25 to -0.25), trend (+0.25 to -0.25), and indicators (+0.25 to -0.25)
        raw_confluence = 0.5 + (0.35 * pattern.bias) + (0.25 * sr_info.bias) + (0.20 * trend.bias) + (0.20 * indicators.bias)
        confluence_score = max(0.05, min(0.95, raw_confluence))

        if confluence_score >= 0.53:
            confluence_direction = "Bullish (Up)"
        elif confluence_score <= 0.47:
            confluence_direction = "Bearish (Down)"
        else:
            confluence_direction = "Neutral / Rangebound"

        # 7. Generate Annotated Chart
        annotated_b64 = self._generate_annotated_chart(
            pil_img, candles, sr_info, pattern, trend, indicators, confluence_direction
        )

        return ScreenAnalysisResult(
            candles_detected=len(candles),
            neighbour_pattern=asdict(pattern),
            market_trend=asdict(trend),
            support_resistance=asdict(sr_info),
            technical_indicators=asdict(indicators),
            confluence_score=confluence_score,
            confluence_direction=confluence_direction,
            annotated_image=annotated_b64,
        )

    def _extract_candles(self, arr: np.ndarray, w: int, h: int) -> list[CandleInfo]:
        r = arr[:, :, 0].astype(int)
        g = arr[:, :, 1].astype(int)
        b = arr[:, :, 2].astype(int)

        # Mask green and red candles
        green_mask = (g > r + 22) & (g > b + 22)
        red_mask = (r > g + 22) & (r > b + 22)
        candle_mask = green_mask | red_mask

        col_counts = np.sum(candle_mask, axis=0)
        active_cols = np.where(col_counts > 2)[0]
        if len(active_cols) == 0:
            return []

        # Group contiguous active columns
        clusters: list[list[int]] = []
        cur: list[int] = [active_cols[0]]
        for col in active_cols[1:]:
            if col - cur[-1] <= 3:
                cur.append(col)
            else:
                if len(cur) >= 2:
                    clusters.append(cur)
                cur = [col]
        if len(cur) >= 2:
            clusters.append(cur)

        candles: list[CandleInfo] = []
        for cl in clusters:
            x1, x2 = cl[0], cl[-1]
            x_center = (x1 + x2) // 2
            sub_mask = candle_mask[:, x1 : x2 + 1]
            y_indices, _ = np.where(sub_mask)
            if len(y_indices) < 2:
                continue

            y_high = int(np.min(y_indices))
            y_low = int(np.max(y_indices))
            range_h = max(1.0, float(y_low - y_high))

            # Detect body vs wick
            g_count = np.sum(green_mask[:, x1 : x2 + 1])
            r_count = np.sum(red_mask[:, x1 : x2 + 1])
            is_bullish = bool(g_count >= r_count)

            row_counts = np.sum(sub_mask, axis=1)
            body_rows = np.where(row_counts >= max(2, (x2 - x1 + 1) * 0.4))[0]
            if len(body_rows) > 0:
                y_top_body = int(np.min(body_rows))
                y_bot_body = int(np.max(body_rows))
            else:
                y_top_body = y_high
                y_bot_body = y_low

            body_h = max(1.0, float(y_bot_body - y_top_body))
            upper_w = max(0.0, float(y_top_body - y_high))
            lower_w = max(0.0, float(y_low - y_bot_body))

            if is_bullish:
                y_open = y_bot_body
                y_close = y_top_body
            else:
                y_open = y_top_body
                y_close = y_bot_body

            candles.append(
                CandleInfo(
                    x_center=x_center,
                    x_left=x1,
                    x_right=x2,
                    y_high=y_high,
                    y_low=y_low,
                    y_open=y_open,
                    y_close=y_close,
                    is_bullish=is_bullish,
                    body_height=body_h,
                    upper_wick=upper_w,
                    lower_wick=lower_w,
                    range_height=range_h,
                )
            )

        candles.sort(key=lambda c: c.x_center)
        return candles

    def _synthetic_projection_candles(self, arr: np.ndarray, w: int, h: int) -> list[CandleInfo]:
        """Fallback candle generator using equal horizontal slices."""
        n_slices = 15
        step = w / n_slices
        candles: list[CandleInfo] = []
        for i in range(n_slices):
            x1 = int(i * step + step * 0.2)
            x2 = int((i + 1) * step - step * 0.2)
            x_center = (x1 + x2) // 2
            sub = arr[:, x1:x2, :]
            gray = np.mean(sub, axis=2)
            dark_y = np.where(gray < 220)[0]
            if len(dark_y) > 0:
                y_high = int(np.min(dark_y))
                y_low = int(np.max(dark_y))
            else:
                y_high = int(h * 0.3)
                y_low = int(h * 0.7)

            range_h = max(1.0, float(y_low - y_high))
            body_h = range_h * 0.6
            is_bullish = bool(i % 2 == 0)
            candles.append(
                CandleInfo(
                    x_center=x_center,
                    x_left=x1,
                    x_right=x2,
                    y_high=y_high,
                    y_low=y_low,
                    y_open=int(y_high + range_h * 0.7 if is_bullish else y_high + range_h * 0.2),
                    y_close=int(y_high + range_h * 0.2 if is_bullish else y_high + range_h * 0.7),
                    is_bullish=is_bullish,
                    body_height=body_h,
                    upper_wick=range_h * 0.2,
                    lower_wick=range_h * 0.2,
                    range_height=range_h,
                )
            )
        return candles

    def _analyze_support_resistance(self, candles: list[CandleInfo], chart_h: int) -> SupportResistanceInfo:
        if not candles:
            return SupportResistanceInfo([], [], None, None, None, None, "No S/R Detected", 0.0)

        # Highs (Resistance candidates) and Lows (Support candidates)
        # Remember: lower y in pixel coords means higher price
        high_levels = [c.y_high for c in candles]
        low_levels = [c.y_low for c in candles]

        # Cluster levels within a threshold (e.g. 4% of chart height)
        threshold = max(8, int(chart_h * 0.04))

        def cluster_levels(levels: list[int]) -> list[int]:
            levels = sorted(levels)
            clusters: list[list[int]] = []
            for lvl in levels:
                if not clusters or abs(lvl - clusters[-1][-1]) > threshold:
                    clusters.append([lvl])
                else:
                    clusters[-1].append(lvl)
            # Pick representative level with high density (2 or more touches preferred)
            clustered = [int(np.mean(cl)) for cl in clusters if len(cl) >= 2]
            return clustered if clustered else [int(np.mean(cl)) for cl in clusters]

        resistance_clusters = cluster_levels(high_levels)
        support_clusters = cluster_levels(low_levels)

        latest = candles[-1]
        latest_y = latest.y_close

        # Support levels must be below latest price in real space (i.e. y_support > latest_y in pixel coords)
        valid_supports = [s for s in support_clusters if s >= latest_y]
        valid_resistances = [r for r in resistance_clusters if r <= latest_y]

        nearest_support = min(valid_supports, key=lambda s: s - latest_y) if valid_supports else None
        nearest_resistance = min(valid_resistances, key=lambda r: latest_y - r) if valid_resistances else None

        dist_supp_pct = (
            round((nearest_support - latest_y) / chart_h * 100, 1) if nearest_support is not None else None
        )
        dist_res_pct = (
            round((latest_y - nearest_resistance) / chart_h * 100, 1) if nearest_resistance is not None else None
        )

        bias = 0.0
        summary = "Mid-Range / Consolidation"

        if dist_supp_pct is not None and dist_supp_pct < 3.5:
            summary = "Near Key Support Zone (High-Probability Bounce)"
            bias = +0.28
        elif dist_res_pct is not None and dist_res_pct < 3.5:
            summary = "Near Key Resistance Zone (Rejection / Resistance Risk)"
            bias = -0.28
        elif dist_supp_pct is not None and dist_supp_pct < 7.0:
            summary = "Approaching Support Area (Favorable Risk/Reward for Up)"
            bias = +0.15
        elif dist_res_pct is not None and dist_res_pct < 7.0:
            summary = "Approaching Resistance Area (Caution on Longs)"
            bias = -0.15

        return SupportResistanceInfo(
            support_levels=sorted(support_clusters, reverse=True)[:3],
            resistance_levels=sorted(resistance_clusters)[:3],
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            distance_to_support_pct=dist_supp_pct,
            distance_to_resistance_pct=dist_res_pct,
            position_summary=summary,
            bias=bias,
        )

    def _detect_neighbour_patterns(self, candles: list[CandleInfo]) -> NeighbourPattern:
        if not candles:
            return NeighbourPattern("Standard Bar", "Neutral", "Low", 0.0, "Insufficient candle data.")

        c0 = candles[-1]
        c1 = candles[-2] if len(candles) >= 2 else None
        c2 = candles[-3] if len(candles) >= 3 else None

        # Check 3-candle patterns
        if c1 is not None and c2 is not None:
            # Morning Star: c2 Bearish, c1 small indecision gapped down, c0 strong Bullish closing > 50% c2 body
            if (
                not c2.is_bullish
                and c2.body_height >= c2.range_height * 0.5
                and c1.body_height <= c1.range_height * 0.35
                and c1.y_high >= c2.y_close
                and c0.is_bullish
                and c0.y_close <= (c2.y_open + c2.y_close) / 2
            ):
                return NeighbourPattern(
                    name="Morning Star",
                    pattern_type="Bullish Reversal",
                    reliability="High",
                    bias=+0.50,
                    description="Strong 3-candle bullish reversal pattern: sellers exhausted followed by decisive bullish impulse.",
                )

            # Evening Star: c2 Bullish, c1 small indecision gapped up, c0 strong Bearish closing < 50% c2 body
            if (
                c2.is_bullish
                and c2.body_height >= c2.range_height * 0.5
                and c1.body_height <= c1.range_height * 0.35
                and c1.y_low <= c2.y_close
                and not c0.is_bullish
                and c0.y_close >= (c2.y_open + c2.y_close) / 2
            ):
                return NeighbourPattern(
                    name="Evening Star",
                    pattern_type="Bearish Reversal",
                    reliability="High",
                    bias=-0.50,
                    description="Strong 3-candle bearish reversal pattern: buyers exhausted followed by decisive bearish impulse.",
                )

            # Three White Soldiers: 3 consecutive bullish candles closing near highs
            if (
                c2.is_bullish
                and c1.is_bullish
                and c0.is_bullish
                and c0.y_close < c1.y_close < c2.y_close
                and c0.upper_wick < c0.range_height * 0.25
            ):
                return NeighbourPattern(
                    name="Three White Soldiers",
                    pattern_type="Bullish Continuation",
                    reliability="High",
                    bias=+0.45,
                    description="Three consecutive advancing bullish candles with sustained buying pressure.",
                )

            # Three Black Crows: 3 consecutive bearish candles closing near lows
            if (
                not c2.is_bullish
                and not c1.is_bullish
                and not c0.is_bullish
                and c0.y_close > c1.y_close > c2.y_close
                and c0.lower_wick < c0.range_height * 0.25
            ):
                return NeighbourPattern(
                    name="Three Black Crows",
                    pattern_type="Bearish Continuation",
                    reliability="High",
                    bias=-0.45,
                    description="Three consecutive declining bearish candles with sustained selling pressure.",
                )

        # Check 2-candle patterns
        if c1 is not None:
            # Bullish Engulfing: c1 Bearish, c0 Bullish engulfing c1 body
            if (
                not c1.is_bullish
                and c0.is_bullish
                and c0.y_close <= c1.y_open
                and c0.y_open >= c1.y_close
                and c0.body_height > c1.body_height
            ):
                return NeighbourPattern(
                    name="Bullish Engulfing",
                    pattern_type="Bullish Reversal",
                    reliability="High",
                    bias=+0.45,
                    description="Bullish candle completely engulfs previous bearish candle, signaling aggressive buyer dominance.",
                )

            # Bearish Engulfing: c1 Bullish, c0 Bearish engulfing c1 body
            if (
                c1.is_bullish
                and not c0.is_bullish
                and c0.y_close >= c1.y_open
                and c0.y_open <= c1.y_close
                and c0.body_height > c1.body_height
            ):
                return NeighbourPattern(
                    name="Bearish Engulfing",
                    pattern_type="Bearish Reversal",
                    reliability="High",
                    bias=-0.45,
                    description="Bearish candle completely engulfs previous bullish candle, signaling strong seller dominance.",
                )

            # Piercing Line: c1 Bearish, c0 Bullish closing > 50% into c1 body
            if (
                not c1.is_bullish
                and c0.is_bullish
                and c0.y_open >= c1.y_close
                and c0.y_close <= (c1.y_open + c1.y_close) / 2
            ):
                return NeighbourPattern(
                    name="Piercing Line",
                    pattern_type="Bullish Reversal",
                    reliability="Medium",
                    bias=+0.35,
                    description="Bullish candle pierces and closes deep inside preceding bearish candle body.",
                )

            # Dark Cloud Cover: c1 Bullish, c0 Bearish closing < 50% into c1 body
            if (
                c1.is_bullish
                and not c0.is_bullish
                and c0.y_open <= c1.y_close
                and c0.y_close >= (c1.y_open + c1.y_close) / 2
            ):
                return NeighbourPattern(
                    name="Dark Cloud Cover",
                    pattern_type="Bearish Reversal",
                    reliability="Medium",
                    bias=-0.35,
                    description="Bearish candle opens high and penetrates deep inside preceding bullish body.",
                )

            # Tweezer Bottom: Matching lows near support
            if abs(c0.y_low - c1.y_low) <= max(2, c0.range_height * 0.05) and c0.is_bullish and not c1.is_bullish:
                return NeighbourPattern(
                    name="Tweezer Bottom",
                    pattern_type="Bullish Reversal",
                    reliability="Medium",
                    bias=+0.35,
                    description="Double test of the exact same low price level with bullish rejection.",
                )

            # Tweezer Top: Matching highs near resistance
            if abs(c0.y_high - c1.y_high) <= max(2, c0.range_height * 0.05) and not c0.is_bullish and c1.is_bullish:
                return NeighbourPattern(
                    name="Tweezer Top",
                    pattern_type="Bearish Reversal",
                    reliability="Medium",
                    bias=-0.35,
                    description="Double test of the exact same high price level with bearish rejection.",
                )

        # Single candle patterns on c0
        # Hammer: Lower wick >= 1.8x body, minimal upper wick
        if c0.lower_wick >= 1.8 * c0.body_height and c0.upper_wick <= 0.3 * c0.body_height:
            return NeighbourPattern(
                name="Hammer",
                pattern_type="Bullish Reversal",
                reliability="High",
                bias=+0.40,
                description="Long lower shadow demonstrates vigorous buying rejection at price lows.",
            )

        # Shooting Star: Upper wick >= 1.8x body, minimal lower wick
        if c0.upper_wick >= 1.8 * c0.body_height and c0.lower_wick <= 0.3 * c0.body_height:
            return NeighbourPattern(
                name="Shooting Star",
                pattern_type="Bearish Reversal",
                reliability="High",
                bias=-0.40,
                description="Long upper shadow demonstrates strong seller rejection at price highs.",
            )

        # Doji: Body <= 10% of total range
        if c0.body_height <= 0.12 * c0.range_height:
            if c0.lower_wick >= 2.0 * c0.upper_wick:
                return NeighbourPattern(
                    name="Dragonfly Doji",
                    pattern_type="Bullish Reversal",
                    reliability="Medium",
                    bias=+0.25,
                    description="Dragonfly Doji signals aggressive buying rejection off lows.",
                )
            elif c0.upper_wick >= 2.0 * c0.lower_wick:
                return NeighbourPattern(
                    name="Gravestone Doji",
                    pattern_type="Bearish Reversal",
                    reliability="Medium",
                    bias=-0.25,
                    description="Gravestone Doji signals aggressive selling rejection off highs.",
                )
            return NeighbourPattern(
                name="Doji (Indecision)",
                pattern_type="Indecision / Neutral",
                reliability="Moderate",
                bias=0.0,
                description="Equilibrium between buyers and sellers, often preceding a volatility expansion.",
            )

        # Marubozu / Strong Impulses
        if c0.body_height >= 0.82 * c0.range_height:
            if c0.is_bullish:
                return NeighbourPattern(
                    name="Bullish Marubozu",
                    pattern_type="Bullish Momentum",
                    reliability="Medium",
                    bias=+0.35,
                    description="Decisive solid green candle with minimal shadows, showing buyers in control from open to close.",
                )
            else:
                return NeighbourPattern(
                    name="Bearish Marubozu",
                    pattern_type="Bearish Momentum",
                    reliability="Medium",
                    bias=-0.35,
                    description="Decisive solid red candle with minimal shadows, showing sellers in control from open to close.",
                )

        # Default standard candle
        if c0.is_bullish:
            return NeighbourPattern(
                name="Bullish Candlestick",
                pattern_type="Bullish",
                reliability="Moderate",
                bias=+0.15,
                description="Standard bullish candle with positive closing price momentum.",
            )
        else:
            return NeighbourPattern(
                name="Bearish Candlestick",
                pattern_type="Bearish",
                reliability="Moderate",
                bias=-0.15,
                description="Standard bearish candle with negative closing price momentum.",
            )

    def _analyze_market_trend(self, candles: list[CandleInfo], chart_h: int) -> MarketTrend:
        if len(candles) < 3:
            return MarketTrend("Sideways Consolidation", 50, 0.0, 0.0)

        # Price midpoints (remember lower y in pixels = higher price in market)
        x_vals = np.array([float(c.x_center) for c in candles])
        y_vals = np.array([float((c.y_high + c.y_low) / 2.0) for c in candles])

        # Normalize x and y to [0, 1]
        x_norm = (x_vals - x_vals.min()) / max(1.0, x_vals.max() - x_vals.min())
        y_norm = (chart_h - y_vals) / max(1.0, chart_h)  # now higher value = higher price!

        # Linear regression slope: dy / dx
        slope, intercept = np.polyfit(x_norm, y_norm, deg=1)
        # R^2 correlation
        y_pred = slope * x_norm + intercept
        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
        ss_res = np.sum((y_norm - y_pred) ** 2)
        r2 = 1.0 - (ss_res / max(1e-6, ss_tot)) if ss_tot > 1e-6 else 0.0
        strength_pct = int(max(20, min(95, (abs(slope) * 60 + r2 * 40))))

        if slope > 0.08:
            direction = "Bullish Uptrend"
            bias = +0.25 * min(1.0, abs(slope) / 0.25)
        elif slope < -0.08:
            direction = "Bearish Downtrend"
            bias = -0.25 * min(1.0, abs(slope) / 0.25)
        else:
            direction = "Sideways Consolidation"
            bias = 0.0
            strength_pct = max(30, int((1.0 - abs(slope) / 0.08) * 70))

        return MarketTrend(
            direction=direction,
            strength_pct=strength_pct,
            slope=round(float(slope), 4),
            bias=round(float(bias), 4),
        )

    def _analyze_technical_indicators(
        self, candles: list[CandleInfo], trend: MarketTrend, chart_h: int
    ) -> TechnicalIndicators:
        if len(candles) < 2:
            return TechnicalIndicators(
                rsi_estimate=50.0,
                rsi_condition="Neutral Momentum",
                ema_trend="Neutral EMA",
                golden_pocket_near=False,
                fibonacci_status="Insufficient candle data",
                fibonacci_levels={},
                bias=0.0,
            )

        # 1. Fibonacci Retracements
        y_highs = [c.y_high for c in candles]
        y_lows = [c.y_low for c in candles]
        swing_high_y = min(y_highs)  # lowest y = highest price
        swing_low_y = max(y_lows)   # highest y = lowest price
        span_h = max(10, swing_low_y - swing_high_y)

        fib_ratios = {"23.6%": 0.236, "38.2%": 0.382, "50.0%": 0.500, "61.8%": 0.618, "78.6%": 0.786}
        fib_levels: dict[str, int] = {}
        for name, ratio in fib_ratios.items():
            if "Uptrend" in trend.direction:
                lvl = int(swing_high_y + ratio * span_h)
            elif "Downtrend" in trend.direction:
                lvl = int(swing_low_y - ratio * span_h)
            else:
                lvl = int(swing_high_y + ratio * span_h)
            fib_levels[name] = lvl

        latest_y = candles[-1].y_close
        golden_618 = fib_levels.get("61.8%", swing_high_y + int(0.618 * span_h))
        golden_50 = fib_levels.get("50.0%", swing_high_y + int(0.500 * span_h))
        dist_to_golden = min(abs(latest_y - golden_618), abs(latest_y - golden_50))
        golden_pocket_near = dist_to_golden <= max(8, int(span_h * 0.05))

        fib_bias = 0.0
        if golden_pocket_near:
            if "Uptrend" in trend.direction:
                fib_status = "Testing 61.8% Golden Pocket Support (Bullish Reversal Setup)"
                fib_bias = +0.22
            elif "Downtrend" in trend.direction:
                fib_status = "Testing 61.8% Golden Pocket Resistance (Bearish Rejection Setup)"
                fib_bias = -0.22
            else:
                fib_status = "At 50-61.8% Equilibrium Pivot Zone"
                fib_bias = +0.10
        elif latest_y < fib_levels.get("38.2%", 0):
            fib_status = "Upper Fibonacci Bullish Extension Zone"
            fib_bias = +0.10
        elif latest_y > fib_levels.get("78.6%", chart_h):
            fib_status = "Deep Fibonacci Retracement / Lower Band Risk"
            fib_bias = -0.10
        else:
            fib_status = "Mid-Range Fibonacci Zone"
            fib_bias = 0.0

        # 2. RSI Proxy Oscillator Estimate
        gains = []
        losses = []
        for i in range(1, len(candles)):
            # diff in price = prev_y - curr_y (positive means price rose)
            price_change = candles[i - 1].y_close - candles[i].y_close
            if price_change > 0:
                gains.append(price_change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(price_change))

        avg_gain = float(np.mean(gains)) if gains else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        if avg_loss == 0.0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_estimate = round(max(5.0, min(95.0, float(rsi))), 1)
        rsi_bias = 0.0
        if rsi_estimate <= 32:
            rsi_condition = f"Oversold ({rsi_estimate}) - Strong Buying Exhaustion Reversal"
            rsi_bias = +0.20
        elif rsi_estimate >= 68:
            rsi_condition = f"Overbought ({rsi_estimate}) - Overextended Momentum"
            rsi_bias = -0.20
        else:
            rsi_condition = f"Neutral Momentum ({rsi_estimate})"
            rsi_bias = (rsi_estimate - 50.0) / 100.0 * 0.20

        # 3. EMA Trend Alignment
        alpha_fast = 2.0 / (3 + 1.0)
        alpha_slow = 2.0 / (7 + 1.0)
        fast_ema = None
        slow_ema = None
        for c in candles:
            price = float((c.y_high + c.y_low) / 2.0)
            if fast_ema is None:
                fast_ema = price
                slow_ema = price
            else:
                fast_ema = alpha_fast * price + (1.0 - alpha_fast) * fast_ema
                slow_ema = alpha_slow * price + (1.0 - alpha_slow) * slow_ema

        ema_bias = 0.0
        if fast_ema is not None and slow_ema is not None:
            if fast_ema < slow_ema - 2:  # fast EMA price is higher than slow EMA
                ema_trend = "Bullish EMA Alignment (Fast EMA > Slow EMA)"
                ema_bias = +0.15
            elif fast_ema > slow_ema + 2:
                ema_trend = "Bearish EMA Alignment (Fast EMA < Slow EMA)"
                ema_bias = -0.15
            else:
                ema_trend = "Converging EMAs (Consolidation Squeeze)"
                ema_bias = 0.0
        else:
            ema_trend = "Neutral EMA"

        combined_indicator_bias = round(0.40 * fib_bias + 0.35 * rsi_bias + 0.25 * ema_bias, 4)

        return TechnicalIndicators(
            rsi_estimate=rsi_estimate,
            rsi_condition=rsi_condition,
            ema_trend=ema_trend,
            golden_pocket_near=golden_pocket_near,
            fibonacci_status=fib_status,
            fibonacci_levels=fib_levels,
            bias=combined_indicator_bias,
        )

    def _generate_annotated_chart(
        self,
        base_img: Image.Image,
        candles: list[CandleInfo],
        sr_info: SupportResistanceInfo,
        pattern: NeighbourPattern,
        trend: MarketTrend,
        indicators: TechnicalIndicators,
        confluence_dir: str,
    ) -> str:
        annotated = base_img.copy()
        draw = ImageDraw.Draw(annotated, "RGBA")
        w, h = annotated.size

        # 1. Draw Support Lines (Green horizontal lines)
        for i, s_y in enumerate(sr_info.support_levels):
            draw.line([(0, s_y), (w, s_y)], fill=(38, 166, 154, 210), width=2)
            label = f"SUPPORT [S{i+1}]"
            draw.rectangle([(w - 110, s_y - 16), (w - 10, s_y + 4)], fill=(38, 166, 154, 230))
            draw.text((w - 105, s_y - 14), label, fill=(255, 255, 255))

        # 2. Draw Resistance Lines (Red horizontal lines)
        for i, r_y in enumerate(sr_info.resistance_levels):
            draw.line([(0, r_y), (w, r_y)], fill=(239, 83, 80, 210), width=2)
            label = f"RESIST [R{i+1}]"
            draw.rectangle([(w - 105, r_y - 16), (w - 10, r_y + 4)], fill=(239, 83, 80, 230))
            draw.text((w - 100, r_y - 14), label, fill=(255, 255, 255))

        # 3. Draw Fibonacci Golden Pocket Level (61.8% Amber Dashed Line)
        if "61.8%" in indicators.fibonacci_levels:
            fib_y = indicators.fibonacci_levels["61.8%"]
            draw.line([(0, fib_y), (w, fib_y)], fill=(255, 193, 7, 200), width=1)
            draw.rectangle([(w - 175, fib_y - 14), (w - 10, fib_y + 4)], fill=(255, 193, 7, 220))
            draw.text((w - 170, fib_y - 13), "FIB 61.8% GOLDEN POCKET", fill=(20, 25, 35))

        # 4. Draw Fast Dynamic EMA Curve (Cyan Line)
        if len(candles) >= 3:
            alpha = 2.0 / (4 + 1.0)
            ema_val = None
            ema_points = []
            for c in candles:
                c_mid = (c.y_open + c.y_close) / 2.0
                ema_val = c_mid if ema_val is None else alpha * c_mid + (1.0 - alpha) * ema_val
                ema_points.append((c.x_center, int(ema_val)))
            for p1, p2 in zip(ema_points[:-1], ema_points[1:]):
                draw.line([p1, p2], fill=(3, 169, 244, 210), width=2)

        # 5. Highlight Neighbour Candlestick Pattern on latest candles
        if candles:
            target_candles = candles[-3:] if len(candles) >= 3 else candles
            min_x = max(0, min(c.x_left for c in target_candles) - 4)
            max_x = min(w, max(c.x_right for c in target_candles) + 4)
            min_y = max(0, min(c.y_high for c in target_candles) - 6)
            max_y = min(h, max(c.y_low for c in target_candles) + 6)

            box_color = (41, 98, 255, 240) if pattern.bias >= 0 else (213, 0, 0, 240)
            draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=box_color, width=2)

            pattern_badge = f"{pattern.name} ({pattern.reliability})"
            b_width = min(220, max(120, len(pattern_badge) * 8))
            draw.rectangle([(min_x, max(4, min_y - 20)), (min_x + b_width, min_y)], fill=box_color)
            draw.text((min_x + 6, max(6, min_y - 17)), pattern_badge, fill=(255, 255, 255))

        # 6. Top Information Banner
        banner_h = 32
        banner_bg = (18, 24, 38, 235)
        draw.rectangle([(0, 0), (w, banner_h)], fill=banner_bg)

        # Confluence direction tag
        conf_color = (38, 166, 154) if "Bullish" in confluence_dir else ((239, 83, 80) if "Bearish" in confluence_dir else (255, 179, 0))
        draw.rectangle([(10, 5), (145, 27)], fill=conf_color)
        draw.text((16, 8), f"AI: {confluence_dir[:9]}", fill=(255, 255, 255))

        # Trend & RSI tag
        draw.text((155, 8), f"Trend: {trend.direction} | RSI: {indicators.rsi_estimate}", fill=(220, 225, 235))

        # Pattern tag
        draw.text((w - 220, 8), f"Pattern: {pattern.name[:18]}", fill=(255, 235, 59))

        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
