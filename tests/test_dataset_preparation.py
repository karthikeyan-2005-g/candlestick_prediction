from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

from scripts.prepare_dataset import prepare_dataset, read_bars


def test_dataset_is_split_before_windows_and_excludes_doji(tmp_path):
    csv_path = tmp_path / "prices.csv"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("timestamp", "open", "high", "low", "close", "volume"))
        for index in range(30):
            timestamp = start + timedelta(days=index)
            open_price = 100 + index
            close_price = open_price + (0 if index == 29 else (1 if index % 2 else -1))
            writer.writerow(
                (
                    timestamp.isoformat(),
                    open_price,
                    max(open_price, close_price) + 1,
                    min(open_price, close_price) - 1,
                    close_price,
                    1000 + index,
                )
            )

    output = tmp_path / "dataset"
    count = prepare_dataset(csv_path, output, window=3)
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert count == len(rows) == 20
    assert {row["partition"] for row in rows} == {"train", "validation", "test"}
    assert all(row["prediction_timestamp"] < row["target_timestamp"] for row in rows)

    bars = read_bars(csv_path)
    timestamp_to_index = {bar.timestamp: index for index, bar in enumerate(bars)}
    bounds = {"train": (0, 21), "validation": (21, 25), "test": (25, 30)}
    for row in rows:
        target_index = timestamp_to_index[row["target_timestamp"]]
        start_index, end_index = bounds[row["partition"]]
        assert start_index <= target_index < end_index
        assert target_index - 3 >= start_index
        assert target_index != 29


def test_timestamp_sort_uses_timezone_aware_chronology(tmp_path):
    csv_path = tmp_path / "offsets.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T01:00:00+01:00,1,2,0,1.5,1\n"
        "2024-01-01T00:00:00+00:00,1,2,0,1.5,1\n",
        encoding="utf-8",
    )
    try:
        read_bars(csv_path)
    except ValueError as error:
        assert "duplicate timestamps" in str(error)
    else:
        raise AssertionError("Equivalent timestamps with different offsets must be duplicates.")
