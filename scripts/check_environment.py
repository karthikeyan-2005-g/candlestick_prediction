from __future__ import annotations

import importlib
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_application
from backend.extensions import db


def main() -> int:
    required = (
        "flask",
        "flask_sqlalchemy",
        "psycopg",
        "cv2",
        "numpy",
        "PIL",
        "torch",
        "torchvision",
        "sklearn",
    )
    missing: list[str] = []
    for module_name in required:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name}")
        except ImportError:
            missing.append(module_name)
            print(f"[MISSING] {module_name}")

    print(f"Python: {sys.version.split()[0]}")
    if missing:
        print("Install dependencies with: pip install -r requirements.txt")
        return 1

    application = create_application()
    with application.app_context():
        try:
            db.session.execute(text("SELECT 1"))
            print("[OK] Database connection")
        except Exception as exc:
            db.session.rollback()
            print(
                f"[ERROR] Database connection failed ({type(exc).__name__}). "
                "Check DATABASE_URL and confirm PostgreSQL is running."
            )
            return 1
        service = application.extensions["prediction_service"]
        if service.is_available:
            print("[OK] Trained model checkpoint")
        else:
            print(f"[INFO] Model unavailable until trained: {service.load_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
