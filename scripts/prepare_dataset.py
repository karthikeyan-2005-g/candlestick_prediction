from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

PARTITIONS = ("train", "validation", "test")
CLASS_NAMES = ("bearish", "bullish")


@dataclass(frozen=True)
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_timestamp(value: str) -> datetime:
    timestamp_value = value.strip()
    if timestamp_value.endswith("Z"):
        timestamp_value = timestamp_value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp_value)
    except ValueError as exc:
        raise ValueError("Timestamps must use ISO-8601 date/time format.") from exc
    if parsed.tzinfo is None:
        raise ValueError("Timestamps must include a timezone offset.")
    return parsed


def read_bars(csv_path: Path) -> list[Bar]:
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain these columns: {sorted(required)}")
        bars: list[Bar] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                bar = Bar(
                    timestamp=row["timestamp"].strip(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid OHLCV values on CSV line {line_number}.") from exc
            if not bar.timestamp or not all(
                math.isfinite(value)
                for value in (bar.open, bar.high, bar.low, bar.close, bar.volume)
            ):
                raise ValueError(f"Invalid timestamp or non-finite value on line {line_number}.")
            if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
                raise ValueError(f"Inconsistent OHLC values on line {line_number}.")
            bars.append(bar)
    parsed_timestamps = [(parse_timestamp(bar.timestamp), bar) for bar in bars]
    parsed_timestamps = [
        (timestamp.astimezone(timezone.utc), bar)
        for timestamp, bar in parsed_timestamps
    ]
    parsed_timestamps.sort(key=lambda item: item[0])
    bars = [bar for _, bar in parsed_timestamps]
    normalized_times = [timestamp for timestamp, _ in parsed_timestamps]
    if len(set(normalized_times)) != len(normalized_times):
        raise ValueError("CSV contains duplicate timestamps.")
    return bars


def draw_candlestick_window(bars: list[Bar], image_size: int = 128) -> Image.Image:
    padding = 8
    image = Image.new("RGB", (image_size, image_size), (248, 249, 252))
    draw = ImageDraw.Draw(image)
    minimum = min(bar.low for bar in bars)
    maximum = max(bar.high for bar in bars)
    span = maximum - minimum or max(abs(maximum) * 0.01, 1e-6)
    chart_height = image_size - 2 * padding
    candle_step = (image_size - 2 * padding) / len(bars)
    body_width = max(1, int(candle_step * 0.55))
    for index, bar in enumerate(bars):
        x = padding + (index + 0.5) * candle_step

        def y(price: float) -> float:
            return padding + (maximum - price) / span * chart_height

        color = (29, 150, 99) if bar.close >= bar.open else (211, 73, 79)
        draw.line((x, y(bar.high), x, y(bar.low)), fill=color, width=1)
        top = min(y(bar.open), y(bar.close))
        bottom = max(y(bar.open), y(bar.close))
        if bottom - top < 1:
            bottom = top + 1
        draw.rectangle(
            (x - body_width / 2, top, x + body_width / 2, bottom),
            fill=color,
        )
    return image


def partition_bounds(total: int, train_ratio: float, validation_ratio: float):
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, total),
    }


def prepare_dataset(
    csv_path: Path,
    output_dir: Path,
    window: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> int:
    if window < 2:
        raise ValueError("window must be at least 2.")
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1.")
    bars = read_bars(csv_path)
    if len(bars) < window + 3:
        raise ValueError("Not enough OHLCV rows for the requested window and splits.")

    bounds = partition_bounds(len(bars), train_ratio, validation_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    count = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=(
                "partition",
                "prediction_timestamp",
                "target_timestamp",
                "label",
                "image_path",
            ),
        )
        writer.writeheader()
        for partition, (start, end) in bounds.items():
            for class_name in CLASS_NAMES:
                (output_dir / partition / class_name).mkdir(parents=True, exist_ok=True)

            first_target_index = start + window
            for target_index in range(first_target_index, end):
                input_bars = bars[target_index - window : target_index]
                target_bar = bars[target_index]
                if target_bar.close == target_bar.open:
                    continue
                label = "bullish" if target_bar.close > target_bar.open else "bearish"
                timestamp_token = target_bar.timestamp.replace(":", "-").replace("/", "-")
                image_path = (
                    output_dir
                    / partition
                    / label
                    / f"{partition}_{target_index:08d}_{timestamp_token}.png"
                )
                draw_candlestick_window(input_bars).save(image_path, format="PNG")
                writer.writerow(
                    {
                        "partition": partition,
                        "prediction_timestamp": input_bars[-1].timestamp,
                        "target_timestamp": target_bar.timestamp,
                        "label": label,
                        "image_path": image_path.as_posix(),
                    }
                )
                count += 1

    print(f"Generated {count} non-doji samples under {output_dir}.")
    print(
        "Chronological bar splits were made before window generation; no sample "
        "window or next-candle label crosses a split boundary."
    )
    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create chronologically split candlestick images from OHLCV CSV."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        args.csv,
        args.output,
        args.window,
        args.train_ratio,
        args.validation_ratio,
    )
