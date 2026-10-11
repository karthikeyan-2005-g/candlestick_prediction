from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException

from backend import create_app


def configure_logging(app: Flask) -> None:
    logging.basicConfig(
        level=getattr(logging, app.config["LOG_LEVEL"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": error.name.lower().replace(" ", "_"),
                        "message": error.description,
                    },
                }
            ),
            error.code or 500,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception) -> tuple[Any, int]:
        app.logger.exception("Unhandled application error", exc_info=error)
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": "internal_server_error",
                        "message": "An unexpected server error occurred.",
                    },
                }
            ),
            500,
        )


def create_application(config_object: type | object | None = None) -> Flask:
    app = create_app(config_object)
    configure_logging(app)
    register_error_handlers(app)

    @app.get("/")
    def index() -> Any:
        return send_from_directory(app.config["FRONTEND_DIR"], "index.html")

    @app.get("/<page>.html")
    def frontend_page(page: str) -> Any:
        if page not in {"prediction", "history", "about"}:
            return jsonify({"success": False, "error": {"message": "Page not found."}}), 404
        return send_from_directory(app.config["FRONTEND_DIR"], f"{page}.html")

    @app.get("/frontend/<path:asset_path>")
    def frontend_asset(asset_path: str) -> Any:
        return send_from_directory(app.config["FRONTEND_DIR"], asset_path)

    return app


application = create_application()


if __name__ == "__main__":
    application.run(host="127.0.0.1", port=5000, debug=False)
