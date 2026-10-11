from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_application
from backend.extensions import db


def main() -> None:
    application = create_application()
    with application.app_context():
        db.create_all()
        print("Database tables are ready.")


if __name__ == "__main__":
    main()
