from __future__ import annotations

from pathlib import PurePath

ALLOWED_EXTENSIONS = frozenset({"png", "jpg", "jpeg"})


def allowed_image_filename(filename: str) -> bool:
    if "/" in filename or "\\" in filename:
        return False
    name = PurePath(filename).name
    suffix = name.rsplit(".", 1)
    return len(suffix) == 2 and suffix[-1].lower() in ALLOWED_EXTENSIONS


def optional_text_field(value: str | None, name: str, maximum: int) -> str | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters.")
    return normalized.upper() if name == "symbol" else normalized
