from __future__ import annotations

from io import BytesIO
import warnings

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_PIXELS = 25_000_000


def preprocessing_config(image_size: int = 128) -> dict[str, str | int]:
    return {
        "backend": "opencv",
        "color_space": "RGB",
        "image_size": image_size,
        "interpolation": "INTER_AREA",
        "normalization": "uint8_to_float32_0_1",
    }


class ImageDecodeError(ValueError):
    """Raised when image bytes cannot be safely decoded."""


def rgb_array_to_tensor(rgb_image: np.ndarray, image_size: int = 128) -> torch.Tensor:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ImageDecodeError("Image must have three RGB color channels.")
    if rgb_image.shape[0] < 1 or rgb_image.shape[1] < 1:
        raise ImageDecodeError("Image has invalid dimensions.")
    if rgb_image.shape[0] * rgb_image.shape[1] > MAX_IMAGE_PIXELS:
        raise ImageDecodeError("Image dimensions exceed the allowed pixel count.")

    resized = cv2.resize(
        rgb_image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    normalized = resized.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(normalized)).permute(2, 0, 1)


def preprocess_image_bytes(image_bytes: bytes, image_size: int = 128) -> torch.Tensor:
    if not image_bytes:
        raise ImageDecodeError("Image data is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(image_bytes))
        with image:
            if image.format not in {"PNG", "JPEG"}:
                raise ImageDecodeError("Only PNG, JPG, and JPEG image data is supported.")
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise ImageDecodeError("Image dimensions exceed the allowed pixel count.")
            image.verify()
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageDecodeError("Image data could not be decoded.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb_array_to_tensor(rgb, image_size)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        cv2.error,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        if isinstance(exc, ImageDecodeError):
            raise
        raise ImageDecodeError("The uploaded file is not a valid supported image.") from exc


def preprocess_pil_image(image: Image.Image, image_size: int = 128) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb_array_to_tensor(rgb, image_size)
