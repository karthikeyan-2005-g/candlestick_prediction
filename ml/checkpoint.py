from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ml.dataset import CLASS_TO_INDEX, INDEX_TO_CLASS
from ml.model import MODEL_VERSION
from ml.preprocessing import preprocessing_config

SUPPORTED_MODEL_VERSIONS = frozenset({MODEL_VERSION})


def validate_checkpoint(
    checkpoint: object,
    *,
    image_size: int | None = None,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary with model metadata.")

    class_to_idx = checkpoint.get("class_to_idx")
    if (
        not isinstance(class_to_idx, dict)
        or set(class_to_idx) != set(CLASS_TO_INDEX)
        or any(type(index) is not int for index in class_to_idx.values())
        or class_to_idx != CLASS_TO_INDEX
    ):
        raise ValueError("Checkpoint class mapping is incompatible with Down/Up.")

    classes = checkpoint.get("classes")
    if type(classes) is not list or classes != list(INDEX_TO_CLASS):
        raise ValueError("Checkpoint classes are missing or incompatible with Down/Up.")

    checkpoint_image_size = checkpoint.get("image_size")
    if type(checkpoint_image_size) is not int or checkpoint_image_size <= 0:
        raise ValueError("Checkpoint has an invalid image_size.")
    if image_size is not None:
        if type(image_size) is not int or image_size <= 0:
            raise ValueError("Requested image_size must be a positive integer.")
        if checkpoint_image_size != image_size:
            raise ValueError("Checkpoint image size does not match configured IMAGE_SIZE.")

    preprocessing = checkpoint.get("preprocessing")
    if (
        not isinstance(preprocessing, dict)
        or preprocessing != preprocessing_config(checkpoint_image_size)
    ):
        raise ValueError("Checkpoint preprocessing configuration is incompatible.")

    model_version = checkpoint.get("model_version")
    if type(model_version) is not str or model_version not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(
            f"Unsupported or missing model_version; supported versions are "
            f"{', '.join(sorted(SUPPORTED_MODEL_VERSIONS))}."
        )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")

    return checkpoint


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device,
    image_size: int | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    return validate_checkpoint(checkpoint, image_size=image_size)