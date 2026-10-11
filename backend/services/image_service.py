from __future__ import annotations

import hashlib

from werkzeug.datastructures import FileStorage

from backend.utils.validators import allowed_image_filename
from ml.preprocessing import ImageDecodeError, preprocess_image_bytes


class ImageValidationError(ValueError):
    """An uploaded payload is not a valid supported image."""


def validate_image_upload(
    uploaded: FileStorage,
    *,
    max_bytes: int,
    image_size: int,
) -> tuple[bytes, str]:
    if not uploaded.filename:
        raise ImageValidationError("Choose an image file.")
    if not allowed_image_filename(uploaded.filename):
        raise ImageValidationError("Only PNG, JPG, and JPEG files are supported.")

    image_bytes = uploaded.stream.read(max_bytes + 1)
    if not image_bytes:
        raise ImageValidationError("The uploaded image is empty.")
    if len(image_bytes) > max_bytes:
        raise ImageValidationError("The uploaded image exceeds the size limit.")

    try:
        preprocess_image_bytes(image_bytes, image_size=image_size)
    except ImageDecodeError as exc:
        raise ImageValidationError(str(exc)) from exc
    return image_bytes, hashlib.sha256(image_bytes).hexdigest()
