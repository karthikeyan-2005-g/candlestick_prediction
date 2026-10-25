from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_binance_klines(symbol: str, interval: str = "1d", limit: int = 500) -> list[Candle]:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    raw = response.json()
    candles: list[Candle] = []
    for item in raw:
        candles.append(
            Candle(
                timestamp=int(item[0]),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
            )
        )
    return candles


def render_candlestick_chart(
    candles: list[Candle],
    width: int = 700,
    height: int = 500,
    padding_x: int = 30,
    padding_y: int = 30,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (248, 249, 252))
    draw = ImageDraw.Draw(image)

    min_price = min(c.low for c in candles)
    max_price = max(c.high for c in candles)
    price_span = max_price - min_price
    if price_span <= 0:
        price_span = max(abs(max_price) * 0.01, 1e-4)

    chart_w = width - 2 * padding_x
    chart_h = height - 2 * padding_y
    n = len(candles)
    step = chart_w / n
    body_w = max(2, int(step * 0.65))

    def price_to_y(p: float) -> int:
        ratio = (max_price - p) / price_span
        return int(padding_y + ratio * chart_h)

    # Subtle grid lines
    for i in range(1, 5):
        gy = int(padding_y + (i / 5.0) * chart_h)
        draw.line([(padding_x, gy), (width - padding_x, gy)], fill=(235, 238, 242), width=1)

    for i, c in enumerate(candles):
        center_x = int(padding_x + (i + 0.5) * step)
        is_bullish = c.close >= c.open
        candle_color = (29, 150, 99) if is_bullish else (211, 73, 79)

        y_high = price_to_y(c.high)
        y_low = price_to_y(c.low)
        y_open = price_to_y(c.open)
        y_close = price_to_y(c.close)

        # Draw wick
        draw.line([(center_x, y_high), (center_x, y_low)], fill=candle_color, width=1)

        # Draw body
        top_y = min(y_open, y_close)
        bot_y = max(y_open, y_close)
        if bot_y - top_y < 1:
            bot_y = top_y + 1

        left_x = center_x - body_w // 2
        right_x = left_x + body_w
        draw.rectangle([left_x, top_y, right_x, bot_y], fill=candle_color, outline=candle_color)

    return image


def build_and_save_dataset(
    output_archive_dir: Path,
    raw_csv_dir: Path | None = None,
    samples_per_symbol: int = 120,
    window_size: int = 25,
    seed: int = 42,
) -> dict[str, int]:
    random.seed(seed)
    symbols = [
        ("BTCUSDT", "BTC"),
        ("ETHUSDT", "ETH"),
        ("SOLUSDT", "SOL"),
        ("BNBUSDT", "BNB"),
        ("ADAUSDT", "ADA"),
        ("XRPUSDT", "XRP"),
        ("AVAXUSDT", "AVAX"),
        ("LINKUSDT", "LINK"),
        ("DOTUSDT", "DOT"),
    ]

    intervals = ["1d", "4h"]
    counts = {"Train_Up": 0, "Train_Down": 0, "Test_Up": 0, "Test_Down": 0}

    for partition in ("Train", "Test"):
        for cls_name in ("Up", "Down"):
            (output_archive_dir / partition / cls_name).mkdir(parents=True, exist_ok=True)

    if raw_csv_dir:
        raw_csv_dir.mkdir(parents=True, exist_ok=True)

    for api_symbol, prefix in symbols:
        all_candles: list[Candle] = []
        for interval in intervals:
            try:
                c_list = fetch_binance_klines(api_symbol, interval=interval, limit=500)
                all_candles.extend(c_list)
            except Exception as e:
                print(f"Warning: Failed to fetch {api_symbol} {interval}: {e}")

        if not all_candles:
            continue

        # Save raw CSV
        if raw_csv_dir:
            csv_path = raw_csv_dir / f"{prefix}_raw.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
                for c in all_candles:
                    writer.writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])

        # Generate windows
        up_windows: list[list[Candle]] = []
        down_windows: list[list[Candle]] = []

        for idx in range(window_size, len(all_candles) - 1, 2):
            window = all_candles[idx - window_size : idx]
            next_candle = all_candles[idx]
            current_close = window[-1].close
            next_close = next_candle.close
            ret = (next_close - current_close) / current_close

            # Clear movement threshold (>= 0.25%)
            if ret > 0.0025:
                up_windows.append(window)
            elif ret < -0.0025:
                down_windows.append(window)

        # Equalize classes per symbol
        n_samples = min(len(up_windows), len(down_windows), samples_per_symbol)
        random.shuffle(up_windows)
        random.shuffle(down_windows)
        selected_up = up_windows[:n_samples]
        selected_down = down_windows[:n_samples]

        n_train = int(n_samples * 0.8)

        # Save images
        for i, window in enumerate(selected_up):
            partition = "Train" if i < n_train else "Test"
            counts[f"{partition}_Up"] += 1
            filename = f"{prefix}_{counts[f'{partition}_Up'] + 1000}.png"
            dest = output_archive_dir / partition / "Up" / filename
            img = render_candlestick_chart(window)
            img.save(dest, format="PNG")

        for i, window in enumerate(selected_down):
            partition = "Train" if i < n_train else "Test"
            counts[f"{partition}_Down"] += 1
            filename = f"{prefix}_{counts[f'{partition}_Down'] + 1000}.png"
            dest = output_archive_dir / partition / "Down" / filename
            img = render_candlestick_chart(window)
            img.save(dest, format="PNG")

        print(f"[{prefix}] Generated {n_train} Train pairs, {n_samples - n_train} Test pairs.")

    print("Completed multi-asset dataset generation:", counts)
    return counts


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    archive_dir = project_root / "archive"
    raw_dir = project_root / "dataset" / "raw_crypto"
    build_and_save_dataset(archive_dir, raw_dir, samples_per_symbol=80)
