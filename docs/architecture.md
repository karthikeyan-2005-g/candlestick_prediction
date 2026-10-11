# Architecture

## Runtime

`app.py` is the WSGI/development-server entry point. The application factory
in `backend/__init__.py` applies configuration, initializes Flask-SQLAlchemy,
registers API blueprints, and loads a `PredictionService` once. Model load
errors are retained as health information; a missing checkpoint does not stop
the dashboard from opening, but `/api/predict` returns 503 instead of using
random model weights.

Flask serves the static HTML pages at `/`, `/history.html`,
`/prediction.html`, and `/about.html`, with assets under `/frontend/...`. The
same-origin browser code calls `/api/...`, so CORS is intentionally absent.

## Request path

1. Browser posts an image with `multipart/form-data`.
2. Route validates metadata and fields.
3. Image service checks the filename extension, bounded byte size, actual
   image format, image dimensions, and decodability. It computes SHA-256 and
   keeps bytes in memory; it never writes the client filename to disk.
4. Shared preprocessing decodes with Pillow/OpenCV, converts BGR to RGB,
   resizes, scales pixels to `[0, 1]`, and produces a CHW tensor.
5. Predictor runs `eval()` model inference under `torch.inference_mode()`,
   applies softmax and the configured uncertainty threshold.
6. SQLAlchemy commits model version, score, symbol/timeframe, image hash, and
   timestamp. The original image is not persisted.
7. API serializes the saved row. History and outcome routes query/update rows.

## Training path

`scripts/prepare_dataset.py` sorts raw OHLCV bars, splits the bar timeline
chronologically, then forms input windows and next-candle labels wholly inside
each split. The target candle is not drawn. Generated directories are consumed
by Torchvision `ImageFolder` with the same shared image preprocessing used at
inference. `ml/train.py` selects a best validation checkpoint and stores
training metrics. `ml/evaluate.py` evaluates the independent test split once,
compares the model with a majority-class baseline, and writes metrics JSON.

## Configuration and boundaries

`config.py` loads `.env` values; `.env.example` contains safe placeholders.
The database layer uses bound SQLAlchemy parameters. Upload bytes, image size,
confidence threshold, model location, and logging level are configurable. This
local development application has no user identity/authentication, distributed
rate limiter, migrations, or public deployment hardening.
