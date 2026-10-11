from __future__ import annotations

import io
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from ml.checkpoint import validate_checkpoint
from ml.dataset import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    CandlestickImageFolder,
    archive_instrument_group,
)
from ml.model import CandlestickCNN, MODEL_VERSION
from ml.predict import load_predictor
from ml.preprocessing import (
    ImageDecodeError,
    preprocess_image_bytes,
    preprocess_pil_image,
    preprocessing_config,
)
from ml.train import split_training_validation


def make_png(width=64, height=32):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocessing_dimensions_and_normalization():
    tensor = preprocess_image_bytes(make_png(), image_size=128)
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32
    assert 0 <= float(tensor.min()) <= float(tensor.max()) <= 1


def test_training_and_inference_preprocessing_produce_identical_tensor(tmp_path):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (64, 32), (30, 120, 200)).save(image_path)
    image_bytes = image_path.read_bytes()
    training_tensor = preprocess_pil_image(Image.open(image_path), image_size=128)
    inference_tensor = preprocess_image_bytes(image_bytes, image_size=128)
    assert torch.equal(training_tensor, inference_tensor)


def test_dataset_class_mapping_matches_checkpoint_label_order(tmp_path):
    for class_name in CLASS_TO_INDEX:
        class_folder = tmp_path / class_name
        class_folder.mkdir()
        Image.new("RGB", (16, 16), "white").save(class_folder / "sample.png")
    dataset = CandlestickImageFolder(tmp_path)
    assert dataset.dataset.class_to_idx == CLASS_TO_INDEX
    assert len(dataset) == 2


def test_dataset_loader_accepts_archive_case_class_folder_names(tmp_path):
    for split in ("Train", "Test"):
        for class_name in ("Down", "Up"):
            class_folder = tmp_path / split / class_name
            class_folder.mkdir(parents=True)
            Image.new("RGBA", (20, 15), (50, 120, 200, 128)).save(
                class_folder / "sample.png"
            )
        dataset = CandlestickImageFolder(tmp_path / split, image_size=64)
        image, label = dataset[0]
        assert dataset.dataset.class_to_idx == {"Down": 0, "Up": 1}
        assert image.shape == (3, 64, 64)
        assert image.dtype == torch.float32
        assert label in (0, 1)


def test_preprocessing_rejects_empty_and_invalid_bytes():
    with pytest.raises(ImageDecodeError):
        preprocess_image_bytes(b"")
    with pytest.raises(ImageDecodeError):
        preprocess_image_bytes(b"not image")


def test_model_output_dimensions():
    model = CandlestickCNN(num_classes=2).eval()
    with torch.inference_mode():
        output = model(torch.zeros((2, 3, 128, 128)))
    assert output.shape == (2, 2)


def test_train_validation_split_is_stratified_and_reproducible():
    labels = [0] * 50 + [1] * 50
    train_a, validation_a = split_training_validation(
        labels,
        validation_fraction=0.2,
        seed=17,
    )
    train_b, validation_b = split_training_validation(
        labels,
        validation_fraction=0.2,
        seed=17,
    )
    assert train_a == train_b
    assert validation_a == validation_b
    assert not set(train_a) & set(validation_a)
    assert len(validation_a) == 20
    assert {labels[index] for index in train_a} == {0, 1}
    assert {labels[index] for index in validation_a} == {0, 1}


def test_archive_instrument_group_uses_prefix_not_numeric_suffix():
    assert archive_instrument_group("archive/Train/Up/QQQ_0.png") == "QQQ"
    assert archive_instrument_group("archive/Train/Down/SPY_149.png") == "SPY"
    with pytest.raises(ValueError, match="Expected an archive filename"):
        archive_instrument_group("unknown_149.png")


def test_grouped_training_validation_split_is_reproducible_and_separates_groups():
    targets = []
    groups = []
    for group, down_count, up_count in (("A", 30, 30), ("B", 13, 12), ("C", 7, 8)):
        targets.extend([0] * down_count + [1] * up_count)
        groups.extend([group] * (down_count + up_count))

    train_a, validation_a = split_training_validation(
        targets,
        validation_fraction=0.2,
        seed=17,
        groups=groups,
    )
    train_b, validation_b = split_training_validation(
        targets,
        validation_fraction=0.2,
        seed=17,
        groups=groups,
    )

    assert train_a == train_b
    assert validation_a == validation_b
    assert {groups[index] for index in train_a}.isdisjoint(
        {groups[index] for index in validation_a}
    )
    assert len(validation_a) / len(targets) == 0.25
    assert Counter(targets[index] for index in train_a) == {0: 37, 1: 38}
    assert Counter(targets[index] for index in validation_a) == {0: 13, 1: 12}


def test_checkpoint_loader_and_predictor_forward_pass_smoke(tmp_path):
    """Exercise checkpoint wiring only; these temporary weights are untrained."""
    checkpoint_path = tmp_path / "trained.pth"
    model = CandlestickCNN()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": CLASS_TO_INDEX,
            "classes": list(INDEX_TO_CLASS),
            "image_size": 128,
            "preprocessing": preprocessing_config(128),
            "model_version": MODEL_VERSION,
        },
        checkpoint_path,
    )
    predictor = load_predictor(
        checkpoint_path,
        image_size=128,
        confidence_threshold=0.6,
    )
    result = predictor.predict(make_png())
    assert predictor.model.training is False
    assert result.prediction in {"Down", "Up"}
    assert result.raw_class in {"Down", "Up"}
    assert 0 <= result.confidence <= 1
    assert isinstance(result.low_confidence, bool)
    assert result.model_version == MODEL_VERSION


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_to_idx", {"Down": False, "Up": 1}),
        ("classes", ["Up", "Down"]),
        ("image_size", True),
        ("image_size", 0),
        ("preprocessing", {**preprocessing_config(128), "backend": "pillow"}),
        ("model_version", "cnn_v999"),
        ("model_version", None),
        ("model_state_dict", None),
    ],
)
def test_checkpoint_validator_rejects_malformed_or_unsupported_metadata(field, value):
    checkpoint = {
        "model_state_dict": {},
        "class_to_idx": CLASS_TO_INDEX.copy(),
        "classes": list(INDEX_TO_CLASS),
        "image_size": 128,
        "preprocessing": preprocessing_config(128),
        "model_version": MODEL_VERSION,
    }
    checkpoint[field] = value

    with pytest.raises(ValueError):
        validate_checkpoint(checkpoint, image_size=128)


def test_checkpoint_validator_rejects_configured_image_size_mismatch():
    checkpoint = {
        "model_state_dict": {},
        "class_to_idx": CLASS_TO_INDEX.copy(),
        "classes": list(INDEX_TO_CLASS),
        "image_size": 128,
        "preprocessing": preprocessing_config(128),
        "model_version": MODEL_VERSION,
    }

    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint(checkpoint, image_size=224)


def test_missing_checkpoint_is_not_replaced_with_random_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictor(
            tmp_path / "missing.pth",
            image_size=128,
            confidence_threshold=0.6,
        )
