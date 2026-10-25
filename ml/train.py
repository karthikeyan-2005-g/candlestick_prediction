from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from ml.checkpoint import SUPPORTED_MODEL_VERSIONS
from ml.dataset import (
    CLASS_TO_INDEX,
    CandlestickImageFolder,
    archive_instrument_group,
)
from ml.model import CandlestickCNN, MODEL_VERSION
from ml.preprocessing import preprocessing_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    if not targets:
        raise ValueError("Dataset partition contains no images.")
    return (
        total_loss / len(targets),
        accuracy_score(targets, predictions),
        balanced_accuracy_score(targets, predictions),
    )


def require_partition_images(data_dir: Path, partition: str) -> None:
    supported = {".png", ".jpg", ".jpeg"}
    missing_or_empty = [
        class_name
        for class_name in CLASS_TO_INDEX
        if not any(
            path.is_file() and path.suffix.lower() in supported
            for path in (data_dir / partition / class_name).glob("*")
        )
    ]
    if missing_or_empty:
        raise ValueError(
            f"Dataset partition '{partition}' has no supported images for: "
            f"{', '.join(missing_or_empty)}. Prepare a labeled OHLCV dataset "
            "with scripts/prepare_dataset.py before training."
        )


def split_training_validation(
    targets: list[int],
    *,
    validation_fraction: float,
    seed: int,
    groups: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    if len(targets) < 4:
        raise ValueError("At least four training images are needed for a validation split.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    class_counts = Counter(targets)
    if len(class_counts) != len(CLASS_TO_INDEX) or min(class_counts.values()) < 2:
        raise ValueError("Each training class needs at least two images for stratification.")
    indices = list(range(len(targets)))
    if groups is not None:
        if len(groups) != len(targets):
            raise ValueError("groups must contain one group key per training image.")
        if any(not group for group in groups):
            raise ValueError("Every training image must have a non-empty group key.")
        group_names = sorted(set(groups))
        if len(group_names) < 2:
            raise ValueError("At least two groups are needed for a group-aware split.")

        random.Random(seed).shuffle(group_names)
        group_order = {group: order for order, group in enumerate(group_names)}
        candidates: list[tuple[float, float, int, list[int], list[int]]] = []
        for validation_group in group_names:
            validation_indices = [
                index for index, group in enumerate(groups) if group == validation_group
            ]
            training_indices = [
                index for index, group in enumerate(groups) if group != validation_group
            ]
            training_counts = Counter(targets[index] for index in training_indices)
            validation_counts = Counter(targets[index] for index in validation_indices)
            if any(
                training_counts[class_index] == 0 or validation_counts[class_index] == 0
                for class_index in CLASS_TO_INDEX.values()
            ):
                continue

            size_error = abs(len(validation_indices) / len(targets) - validation_fraction)
            class_distribution_error = sum(
                abs(
                    validation_counts[class_index] / len(validation_indices)
                    - class_counts[class_index] / len(targets)
                )
                for class_index in CLASS_TO_INDEX.values()
            )
            candidates.append(
                (
                    size_error,
                    class_distribution_error,
                    group_order[validation_group],
                    training_indices,
                    validation_indices,
                )
            )

        if not candidates:
            raise ValueError(
                "No group-aware split keeps every class in both training and validation."
            )
        _, _, _, training_indices, validation_indices = min(
            candidates, key=lambda candidate: candidate[:3]
        )
        return training_indices, validation_indices

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=targets,
    )
    return list(train_indices), list(validation_indices)


def _path_is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def resolve_training_partition(
    data_dir: str | Path,
    train_split: str | Path,
) -> tuple[Path, Path]:
    try:
        resolved_data_dir = Path(data_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Dataset root does not exist or cannot be resolved: {data_dir}") from exc
    if not resolved_data_dir.is_dir():
        raise ValueError(f"Dataset root is not a directory: {resolved_data_dir}")

    if resolved_data_dir.name.casefold() == "test" or any(
        ancestor.name.casefold() == "test"
        and (ancestor.parent / "Train").is_dir()
        for ancestor in resolved_data_dir.parents
    ):
        raise ValueError("The Test partition cannot be used as a dataset root.")

    test_path = (resolved_data_dir / "Test").resolve(strict=False)
    try:
        requested_path = Path(train_split).expanduser()
        requested_training_path = (resolved_data_dir / requested_path).resolve(
            strict=False
        )
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"Training path cannot be resolved under {resolved_data_dir}."
        ) from exc

    if (
        requested_training_path.name.casefold() == "test"
        or _path_is_within(requested_training_path, test_path)
    ):
        raise ValueError("The Test partition is reserved for final evaluation.")

    try:
        resolved_training_path = requested_training_path.resolve(strict=True)
        approved_training_path = (resolved_data_dir / "Train").resolve(
            strict=True
        )
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"The approved Train partition cannot be resolved under {resolved_data_dir}."
        ) from exc

    if (
        resolved_training_path.name.casefold() == "test"
        or _path_is_within(resolved_training_path, test_path)
    ):
        raise ValueError("The Test partition is reserved for final evaluation.")
    if not _path_is_within(approved_training_path, resolved_data_dir):
        raise ValueError("The approved Train partition must remain inside the dataset root.")
    if resolved_training_path != approved_training_path:
        raise ValueError(
            "Training must use the approved Train partition at "
            f"{approved_training_path}; received {resolved_training_path}."
        )
    if not resolved_training_path.is_dir():
        raise ValueError(f"The approved Train partition is not a directory: {resolved_training_path}")

    return resolved_data_dir, resolved_training_path


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.model_version not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(
            f"Unsupported model_version {args.model_version!r}; supported versions are "
            f"{', '.join(sorted(SUPPORTED_MODEL_VERSIONS))}."
        )
    data_dir, training_path = resolve_training_partition(
        args.data_dir,
        args.train_split,
    )
    seed_everything(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    require_partition_images(data_dir, "Train")
    full_train_data = CandlestickImageFolder(training_path, args.image_size)
    if not full_train_data:
        raise ValueError(f"Training partition '{args.train_split}' contains no images.")
    groups = [archive_instrument_group(path) for path, _ in full_train_data.samples]
    split_groups = None if getattr(args, "stratified", False) else groups
    train_indices, validation_indices = split_training_validation(
        full_train_data.targets,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        groups=split_groups,
    )
    train_data = Subset(full_train_data, train_indices)
    validation_data = Subset(full_train_data, validation_indices)
    train_counts = Counter(full_train_data.targets[index] for index in train_indices)
    validation_counts = Counter(
        full_train_data.targets[index] for index in validation_indices
    )
    training_groups = sorted({groups[index] for index in train_indices})
    validation_groups = sorted({groups[index] for index in validation_indices})
    actual_validation_fraction = len(validation_data) / len(full_train_data)
    validation_strategy = (
        "stratified_random" if getattr(args, "stratified", False) else "instrument_prefix_group_holdout"
    )
    print(f"Validation strategy: {validation_strategy}")
    print(f"Training instrument groups: {training_groups}")
    print(f"Validation instrument groups: {validation_groups}")
    print(f"Actual validation fraction: {actual_validation_fraction:.3f}")
    print(
        "Training class counts:",
        {name: train_counts[index] for name, index in CLASS_TO_INDEX.items()},
    )
    print(
        "Validation class counts:",
        {name: validation_counts[index] for name, index in CLASS_TO_INDEX.items()},
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = CandlestickCNN().to(device)
    total_train = len(train_data)
    weight_down = total_train / (2.0 * max(1, train_counts.get(CLASS_TO_INDEX["Down"], 1)))
    weight_up = total_train / (2.0 * max(1, train_counts.get(CLASS_TO_INDEX["Up"], 1)))
    class_weights = torch.tensor([weight_down, weight_up], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    best_validation_score = -1.0
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    wait = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch:03d}/{args.epochs}...", flush=True)
        train_loss, train_accuracy, train_balanced = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy, validation_balanced = run_epoch(
            model, validation_loader, criterion, device
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": float(train_accuracy),
                "train_balanced_accuracy": float(train_balanced),
                "validation_loss": validation_loss,
                "validation_accuracy": float(validation_accuracy),
                "validation_balanced_accuracy": float(validation_balanced),
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train loss={train_loss:.4f} acc={train_accuracy:.4f} bal={train_balanced:.4f}; "
            f"val loss={validation_loss:.4f} acc={validation_accuracy:.4f} bal={validation_balanced:.4f}",
            flush=True,
        )

        val_score = float(validation_balanced)
        improved = (
            val_score > best_validation_score
            or (
                val_score == best_validation_score
                and (
                    validation_accuracy > best_validation_accuracy
                    or (validation_accuracy == best_validation_accuracy and validation_loss < best_validation_loss)
                )
            )
        )
        if improved:
            best_validation_score = val_score
            best_validation_accuracy = float(validation_accuracy)
            best_validation_loss = validation_loss
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": CLASS_TO_INDEX,
                    "classes": list(CLASS_TO_INDEX),
                    "image_size": args.image_size,
                    "preprocessing": preprocessing_config(args.image_size),
                    "model_version": args.model_version,
                    "validation_accuracy": best_validation_accuracy,
                    "validation_loss": best_validation_loss,
                    "training_config": {
                        "data_directory": str(data_dir.resolve()),
                        "training_split": "Train",
                        "validation_fraction": args.validation_fraction,
                        "actual_validation_fraction": actual_validation_fraction,
                        "validation_strategy": validation_strategy,
                        "training_groups": training_groups,
                        "validation_groups": validation_groups,
                        "seed": args.seed,
                        "epochs_requested": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "train_samples": len(train_data),
                        "validation_samples": len(validation_data),
                        "train_class_counts": dict(train_counts),
                        "validation_class_counts": dict(validation_counts),
                    },
                },
                output_path,
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"Epochs completed: {len(history)}", flush=True)
    history_path = output_path.with_name("training_metrics.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint saved to {output_path}")
    print(f"Training history saved to {history_path}")
    print("Class imbalance should be considered when interpreting accuracy.")
    return {"history": history, "checkpoint": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the candlestick CNN.")
    parser.add_argument("--data-dir", default="archive")
    parser.add_argument("--train-split", default="Train")
    parser.add_argument("--output", default="models/candlestick_cnn.pth")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified", action="store_true", help="Use stratified split across instruments.")
    parser.add_argument(
        "--model-version",
        default=MODEL_VERSION,
        choices=sorted(SUPPORTED_MODEL_VERSIONS),
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size, args.patience) <= 0:
        parser.error("epochs, batch size, image size, and patience must be positive.")
    if args.learning_rate <= 0:
        parser.error("learning rate must be positive.")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation fraction must be between 0 and 1.")
    return args


if __name__ == "__main__":
    train(parse_args())
