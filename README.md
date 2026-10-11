# AI-Based Candlestick Prediction System

A local, educational full-stack project that accepts candlestick chart images,
uses a trained PyTorch CNN to classify the supplied dataset's **Down** or **Up**
image class, and records results in PostgreSQL. The archive's label rule and
chronological split provenance are undocumented, so these labels must not be
interpreted as verified next-candle directions. Flask serves both the REST API
and the same-origin HTML/CSS/JavaScript dashboard.

> **Financial risk notice:** This is an experimental image classifier, not a
> trading system, investment advice, or a reliable guarantee of future market
> direction. Model scores are not calibrated probabilities. No accuracy or
> profitability is claimed. A useful model requires a correctly labeled,
> chronological dataset and independent evaluation.

## Features

- Single-origin responsive dashboard, upload preview, prediction, and history.
- Flask JSON API with validation, pagination, status, metrics, and outcome
  recording.
- OpenCV image decoding and shared 128 × 128 RGB preprocessing.
- PyTorch CNN training, best-validation checkpointing, early stopping, and
  chronological test evaluation.
- PostgreSQL persistence through SQLAlchemy; SQLite is supported for tests.
- Historical OHLCV dataset generator that excludes the target candle from each
  chart and splits raw time periods before making overlapping samples.
- No fake predictions: inference returns HTTP 503 until a compatible trained
  checkpoint is available.

## Architecture

```text
Browser (HTML/CSS/vanilla JS)
       | same-origin fetch/FormData
Flask app + API blueprints
   |         |             |
OpenCV    PyTorch CNN   SQLAlchemy ORM ---> PostgreSQL
                          |
                   prediction history

Historical OHLCV CSV -> chronological dataset generator -> ImageFolder dataset
                                                     -> train -> checkpoint
                                                     -> evaluate -> metrics JSON
```

## Folder structure

```text
candlestick project/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── backend/
│   ├── __init__.py
│   ├── extensions.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health_routes.py
│   │   ├── prediction_routes.py
│   │   └── history_routes.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_service.py
│   │   └── prediction_service.py
│   └── utils/
│       ├── __init__.py
│       └── validators.py
├── ml/
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── archive/
│   ├── Train/{Down,Up}/
│   └── Test/{Down,Up}/
├── dataset/                   # optional generated OHLCV-derived alternative
├── models/.gitkeep
├── uploads/.gitkeep
├── scripts/
│   ├── init_db.py
│   ├── check_environment.py
│   └── prepare_dataset.py
├── frontend/
│   ├── index.html
│   ├── prediction.html
│   ├── history.html
│   ├── about.html
│   ├── css/style.css
│   └── js/{app.js,prediction.js,history.js}
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_database.py
└── docs/
    ├── architecture.md
    ├── api_documentation.md
    ├── database_schema.md
    └── model_training.md
```

The package markers are named `__init__.py` (the Python convention, rather
than `init.py`). The dataset preparation utility is included as an additional
script because it turns historical OHLCV rows into the labeled image folders
consumed by training.

## Windows Command Prompt: install and run

### 1. Install tools

Install Python 3.11 or 3.12 (64-bit), PostgreSQL, and VS Code. During Python
installation, enable **Add Python to PATH**. Open the project folder in VS Code
using **File → Open Folder**. The commands below are for **Command Prompt
(cmd.exe)**, not PowerShell.

### 2. Create the environment and install packages

```bat
cd /d "C:\Users\YOUR_NAME\Desktop\candlestick project"
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch and Torchvision versions are paired in `requirements.txt`; the CPU
wheels are sufficient for this project. If pip cannot find a matching wheel,
use the official PyTorch install selector for your Python and CUDA combination,
then install the remaining packages from `requirements.txt`.

### 3. Create and configure PostgreSQL

In pgAdmin or `psql`, create the local database and login (use a strong local
password):

```sql
CREATE USER candlestick_user WITH PASSWORD 'replace_this_local_password';
CREATE DATABASE candlestick_db OWNER candlestick_user;
```

Copy the example configuration and edit `.env` with your local values:

```bat
copy .env.example .env
notepad .env
```

For a local development URL, SQLAlchemy accepts:

```text
DATABASE_URL=postgresql+psycopg://candlestick_user:your-password@localhost:5432/candlestick_db
```

Never commit `.env`; `.gitignore` excludes it. URL-encode special characters in
the password. The application does not require CORS because Flask serves the
frontend and APIs from the same origin.

### 4. Create the schema and verify setup

```bat
python scripts\init_db.py
python scripts\check_environment.py
```

`init_db.py` creates tables from the ORM. It does not overwrite or drop data.
For later production schema changes, use migrations (see
[`docs/database_schema.md`](docs/database_schema.md)).

### 5. Audit the supplied image archive

The provided `archive/Train/Down`, `archive/Train/Up`, `archive/Test/Down`,
and `archive/Test/Up` image counts and file integrity can be checked before
training. The training script uses only `archive/Train`; it makes a stratified
random validation holdout from those training files. The `archive/Test`
partition is loaded only by the separate evaluation script.

The archive does not include its source, label-generation rule, timestamp
metadata, or split methodology. Filename stem overlap is reported by
evaluation. Because that provenance is unknown, test metrics are agreement
with the supplied folder labels, not verified future-candle accuracy.

### 6. Optional: prepare OHLCV-derived data

Collect historical OHLCV data from a source whose terms permit your use. Make a
CSV with columns `timestamp,open,high,low,close,volume`, ordered or sortable by
timestamp. The script splits the raw ordered bars into non-overlapping
chronological train/validation/test periods **before** producing windows.

```bat
python scripts\prepare_dataset.py --csv data\historical_ohlcv.csv --output dataset --window 40
```

The chart shows only the preceding `window` candles. The label is the next
bar's close versus its open; doji candles are excluded. A generated dataset
contains the input images plus a `manifest.csv` of timestamps, labels, and
image paths. These are examples of a reproducible labeling process, not a
claim that data or training results are already available.

### 7. Train and evaluate the provided archive

```bat
python -m ml.train --data-dir archive --train-split Train --output models\candlestick_cnn.pth --epochs 30 --batch-size 32 --learning-rate 0.001 --validation-fraction 0.2
python -m ml.evaluate --data-dir archive --train-split Train --test-split Test --checkpoint models\candlestick_cnn.pth --output models\evaluation_metrics.json
```

The trainer never loads `archive/Test`; it saves the best validation checkpoint
as `models/candlestick_cnn.pth` and logs train/validation loss. Evaluation
reports accuracy, per-class precision/recall/F1, confusion matrix, and an
overlap audit. Do not tune the model using Test results. The output classes are
exactly the supplied `Down` and `Up` folder names; no claim is made about what
those labels mean.

### 8. Start and use the web application

```bat
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Upload a PNG/JPG/JPEG, optionally
enter a symbol and timeframe, then select **Predict**. Use **History** to view
stored predictions. Once the next candle is known, open a prediction detail
and save its actual direction. The UI also shows whether the API, database,
and model are available.

### 9. Run tests

```bat
pytest
```

Tests use an in-memory SQLite database and stub inference, so they do not need
the local PostgreSQL server or a trained checkpoint.

## Environment variables

See [`.env.example`](.env.example). `MODEL_PATH` defaults to
`models/candlestick_cnn.pth`; `CONFIDENCE_THRESHOLD` defaults to `0.60`.
`MAX_CONTENT_LENGTH` defaults to 5 MiB. The secret key is used for Flask
sessions; if `SECRET_KEY` is absent, the process generates an ephemeral
development key. Set a persistent random key in `.env` for a stable deployment.
The application does not store credentials in source code.

## API quick reference

- `GET /api/health` — service, database, and model status.
- `POST /api/predict` — multipart `image`, optional `symbol`, `timeframe`.
- `GET /api/history?page=1&per_page=20&symbol=...` — paginated records.
- `GET /api/history/<id>` — one prediction.
- `PATCH /api/history/<id>/outcome` — JSON actual candle direction.
- `GET /api/metrics` — saved independent test-set metrics, if available.

Full examples and error shapes are in
[`docs/api_documentation.md`](docs/api_documentation.md).

## Security and limitations

- Uploads are size-limited, extension-checked, decoded from memory and hashed;
  client filenames are never used as paths or saved on disk.
- SQL values go through SQLAlchemy parameters; secrets come from `.env`.
- Unexpected API exceptions return generic JSON errors; server details are
  logged without returning tracebacks to clients.
- The application does not authenticate users or rate-limit requests. Do not
  expose this development server to the public internet without adding
  authentication, HTTPS, rate limits, CSRF controls appropriate to deployment,
  and operational monitoring.
- Image layout, chart scaling, and renderer differences affect CNN behavior.
  The score is not necessarily calibrated and does not establish future
  accuracy. Backtesting, chronological out-of-sample tests, and financial
  domain review are still required.

## GitHub without leaking secrets

```bat
git init
git add .
git status
git commit -m "Build candlestick prediction application"
```

Confirm `.env`, personal data, database dumps, uploads, checkpoints, and large
datasets are not staged before pushing. `.gitignore` excludes them by default.
Push only source and documentation; distribute large model files and data
through an access-controlled artifact store if needed.

## Troubleshooting

- **`DATABASE_URL` missing / connection refused:** start PostgreSQL, verify
  database/user/port, and check `.env` spelling.
- **No model available:** prepare a real dataset, train, then verify
  `MODEL_PATH` points to `models/candlestick_cnn.pth`.
- **Incompatible checkpoint:** train using this repository's CNN, Down/Up label order,
  and 128 × 128 RGB scaling pipeline.
- **Torchvision wheel error:** use Python 3.11/3.12 and select compatible
  official PyTorch/Torchvision CPU or CUDA wheels.
- **Upload rejected:** only PNG, JPG, and JPEG are accepted, up to the
  configured request size; invalid image bytes are rejected even if the
  filename looks valid.
- **Test dependency on local DB:** tests explicitly override the app database
  to in-memory SQLite; do not put production initialization in test setup.

## Documentation

- [Architecture](docs/architecture.md)
- [API](docs/api_documentation.md)
- [Database schema](docs/database_schema.md)
- [Training and evaluation](docs/model_training.md)
