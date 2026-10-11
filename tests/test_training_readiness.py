from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.dataset import CLASS_TO_INDEX, INDEX_TO_CLASS
from ml.evaluate import audit_train_test_overlap, evaluate
from ml.model import CandlestickCNN, MODEL_VERSION
from ml.preprocessing import preprocessing_config
from ml.train import resolve_training_partition, train


def test_training_stops_without_dataset_and_does_not_write_checkpoint(tmp_path):
    data_dir = tmp_path / "empty_dataset"
    for partition in ("Train",):
        for class_name in ("Down", "Up"):
            (data_dir / partition / class_name).mkdir(parents=True)
    checkpoint = tmp_path / "models" / "candlestick_cnn.pth"
    args = Namespace(
        data_dir=str(data_dir),
        output=str(checkpoint),
        train_split="Train",
        validation_fraction=0.2,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        image_size=128,
        patience=1,
        seed=42,
        model_version=MODEL_VERSION,
    )

    with pytest.raises(ValueError, match="no supported images"):
        train(args)
    assert not checkpoint.exists()


def test_training_rejects_test_partition_as_training_input(tmp_path):
    data_dir = tmp_path / "dataset"
    for class_name in CLASS_TO_INDEX:
        (data_dir / "Test" / class_name).mkdir(parents=True)
    checkpoint = tmp_path / "models" / "candlestick_cnn.pth"
    args = Namespace(
        data_dir=str(data_dir),
        output=str(checkpoint),
        train_split="Test",
        validation_fraction=0.2,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        image_size=128,
        patience=1,
        seed=42,
        model_version=MODEL_VERSION,
    )

    with pytest.raises(ValueError, match="reserved for final evaluation"):
        train(args)
    assert not checkpoint.exists()


def test_training_rejects_unsupported_model_version_before_creating_output(tmp_path):
    checkpoint = tmp_path / "models" / "candlestick_cnn.pth"
    args = Namespace(
        data_dir=str(tmp_path / "missing_dataset"),
        output=str(checkpoint),
        train_split="Train",
        validation_fraction=0.2,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        image_size=128,
        patience=1,
        seed=42,
        model_version="cnn_v999",
    )

    with pytest.raises(ValueError, match="Unsupported model_version"):
        train(args)
    assert not checkpoint.parent.exists()


def test_training_path_validation_rejects_test_aliases_and_accepts_train(tmp_path):
    data_dir = tmp_path / "archive"
    train_dir = data_dir / "Train"
    test_dir = data_dir / "Test"
    for partition in (train_dir, test_dir):
        for class_name in CLASS_TO_INDEX:
            (partition / class_name).mkdir(parents=True)

    resolved_root, resolved_train = resolve_training_partition(data_dir, "Train")
    assert resolved_root == data_dir.resolve()
    assert resolved_train == train_dir.resolve()
    assert resolve_training_partition(data_dir, str(train_dir.resolve())) == (
        resolved_root,
        resolved_train,
    )

    rejected_paths = (
        (data_dir, "Test"),
        (data_dir, "."),
        (data_dir, "Test/."),
        (data_dir, "Train/../Test"),
        (data_dir, str(test_dir.resolve())),
        (test_dir, "."),
        (data_dir / "Train" / ".." / "Test", "."),
    )
    for requested_data_dir, requested_split in rejected_paths:
        with pytest.raises(ValueError):
            resolve_training_partition(requested_data_dir, requested_split)

    test_alias = data_dir / "TestAlias"
    try:
        test_alias.symlink_to(test_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="Test partition"):
        resolve_training_partition(data_dir, "TestAlias")


def test_evaluation_stops_without_independent_test_data_and_writes_no_metrics(tmp_path):
    data_dir = tmp_path / "empty_dataset"
    for partition in ("Train", "Test"):
        for class_name in ("Down", "Up"):
            (data_dir / partition / class_name).mkdir(parents=True)
    metrics_path = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="no supported images"):
        evaluate(data_dir, tmp_path / "absent-checkpoint.pth", metrics_path)
    assert not metrics_path.exists()


def test_evaluation_rejects_identical_resolved_train_test_roots(tmp_path):
    data_dir = tmp_path / "dataset"
    train_root = data_dir / "Train"
    train_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="different directories"):
        evaluate(
            data_dir,
            tmp_path / "absent-checkpoint.pth",
            tmp_path / "metrics.json",
            train_split="Train",
            test_split=str(train_root.resolve()),
        )


def test_evaluation_reports_cross_entropy_for_test_images(tmp_path):
    data_dir = tmp_path / "dataset"
    for partition in ("Train", "Test"):
        for class_name in CLASS_TO_INDEX:
            class_dir = data_dir / partition / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(class_dir / "sample.png")

    checkpoint_path = tmp_path / "checkpoint.pth"
    model = CandlestickCNN()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": CLASS_TO_INDEX,
            "classes": list(INDEX_TO_CLASS),
            "image_size": 16,
            "preprocessing": preprocessing_config(16),
            "model_version": MODEL_VERSION,
        },
        checkpoint_path,
    )

    result = evaluate(data_dir, checkpoint_path, tmp_path / "metrics.json")

    assert result["test_samples"] == 2
    assert isinstance(result["loss"], float)
    assert result["loss"] >= 0


def test_evaluation_does_not_overwrite_existing_report_by_default(tmp_path):
    output_path = tmp_path / "metrics.json"
    output_path.write_text("keep this report", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate(
            tmp_path / "missing-data",
            tmp_path / "missing-checkpoint.pth",
            output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "keep this report"


def test_evaluation_overwrite_requires_explicit_opt_in(tmp_path):
    data_dir = tmp_path / "dataset"
    for partition in ("Train", "Test"):
        for class_name in CLASS_TO_INDEX:
            class_dir = data_dir / partition / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(class_dir / "sample.png")

    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": CandlestickCNN().state_dict(),
            "class_to_idx": CLASS_TO_INDEX,
            "classes": list(INDEX_TO_CLASS),
            "image_size": 16,
            "preprocessing": preprocessing_config(16),
            "model_version": MODEL_VERSION,
        },
        checkpoint_path,
    )
    output_path = tmp_path / "metrics.json"
    output_path.write_text("old report", encoding="utf-8")

    result = evaluate(data_dir, checkpoint_path, output_path, overwrite=True)

    assert result["test_samples"] == 2
    assert output_path.read_text(encoding="utf-8").startswith("{")


def test_overlap_audit_preserves_multiple_classes_for_a_stem(tmp_path):
    train_root = tmp_path / "Train"
    test_root = tmp_path / "Test"
    (train_root / "Down").mkdir(parents=True)
    (train_root / "Up").mkdir(parents=True)
    (test_root / "Down").mkdir(parents=True)
    (train_root / "Down" / "SPY_149.png").write_bytes(b"train-down")
    (train_root / "Up" / "SPY_149.png").write_bytes(b"train-up")
    (test_root / "Down" / "SPY_149.png").write_bytes(b"test-down")

    audit = audit_train_test_overlap(train_root, test_root)

    assert audit["train_multi_class_stem_count"] == 1
    assert audit["test_multi_class_stem_count"] == 0
    assert audit["shared_stem_label_set_mismatches"] == 1
    assert audit["shared_stem_label_set_mismatch_examples"] == [
        {
            "stem": "SPY_149",
            "train_classes": ["Down", "Up"],
            "test_classes": ["Down"],
        }
    ]
