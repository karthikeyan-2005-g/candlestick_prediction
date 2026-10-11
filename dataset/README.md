# Candlestick image dataset

No sample or fabricated dataset is bundled. Create licensed OHLCV data as a
CSV with `timestamp,open,high,low,close,volume` columns, then run:

```bat
python scripts\prepare_dataset.py --csv data\historical_ohlcv.csv --output dataset --window 40
```

The preparation script orders bars by timestamp, partitions the chronological
bar series into 70% training, 15% validation, and 15% test first, then makes
windows within each partition. The last visible bar ends at the prediction
timestamp; the subsequent bar is the label target and never appears in the
input chart. Labels are `bullish` when next close > next open and `bearish`
when next close < next open. Exact dojis are excluded. Timestamps are used as
strings for ordering; use ISO-8601 dates/times with timezone information.

Do not mix instruments or timeframes without recording those attributes and
carefully choosing a split policy. Adjacent chart windows are highly
correlated, so the chronological split is essential. The independent test
period must remain untouched during training and model selection.
