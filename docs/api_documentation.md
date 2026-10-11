# API documentation

All responses are JSON. Errors use `{"success":false,"error":{"code":"...","message":"..."}}`.

## `GET /api/health`

Returns HTTP 200 with app `status`, `database` (`ok`/`unavailable`), `model`
(`available`/`unavailable`), and model-load error detail. A degraded service is
reported in the response body rather than as a successful prediction.

## `POST /api/predict`

Multipart form: required `image` (PNG/JPG/JPEG), optional `symbol` (up to 24
characters), optional `timeframe` (up to 24 characters). Max request size is
configured by `MAX_CONTENT_LENGTH` (default 5 MiB).

Success is HTTP 201:

```json
{
  "success": true,
  "prediction": "Up",
  "confidence": 0.72,
  "low_confidence": false,
  "symbol": "EXAMPLE",
  "timeframe": "1d",
  "model_version": "cnn_v1",
  "prediction_id": 12,
  "created_at": "2026-01-01T12:00:00+00:00",
  "message": "Experimental model output. Not financial advice."
}
```

The `prediction` field is the archive class `Down` or `Up`. The confidence is
the winning softmax score; `low_confidence` flags scores below the configured
threshold without changing the class label. Class meaning and dataset split
provenance are unverified, so this output is not established future direction.
Missing model gives HTTP 503;
missing, malformed, corrupt, or invalid uploads return HTTP 400; request over
the configured limit returns HTTP 413; database save failure returns HTTP 503.
Prediction is not returned as successful unless the database insert commits.

Example Windows `curl`:

```bat
curl -X POST -F "image=@chart.png" -F "symbol=AAPL" -F "timeframe=1d" http://127.0.0.1:5000/api/predict
```

## `GET /api/history`

Query parameters: `page` (default 1), `per_page` (default 20, maximum 100),
and optional `symbol`. Response includes `data` and pagination `page`,
`per_page`, `total`, `pages`. Newest timestamps and IDs are first.

```json
{"success":true,"data":[],"pagination":{"page":1,"per_page":20,"total":0,"pages":0}}
```

## `GET /api/history/<prediction_id>`

Returns `{"success":true,"data":{...prediction row...}}`. A nonexistent ID
returns 404.

## `PATCH /api/history/<prediction_id>/outcome`

JSON body requires `actual_direction` equal to `Bullish` or `Bearish`.
Optional `observed_at` must be an ISO-8601 timestamp with a timezone; if
omitted, current UTC time is recorded.

```json
{"actual_direction":"Bearish","observed_at":"2026-01-01T13:00:00Z"}
```

## `GET /api/metrics`

Returns saved data from `METRICS_PATH` after `python -m ml.evaluate`. Until
independent test evaluation exists, returns 404 `metrics_unavailable`; it does
not invent metric values.
