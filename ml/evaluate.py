from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from ml.checkpoint import load_checkpoint
from ml.dataset import CLASS_TO_INDEX, INDEX_TO_CLASS, CandlestickImageFolder
from ml.model import CandlestickCNN

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SEQUENCE_FILENAME = re.compile(r"(QQQ|SPY)_(\d+)", re.IGNORECASE)


def require_partition_images(data_dir: Path, partition: str) -> None:
    missing_or_empty = [
        class_name
        for class_name in CLASS_TO_INDEX
        if not any(
            path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            for path in (data_dir / partition / class_name).glob("*")
        )
    ]
    if missing_or_empty:
        raise ValueError(
            f"Dataset partition '{partition}' has no supported images for: "
            f"{', '.join(missing_or_empty)}."
        )


def audit_train_test_overlap(
    train_root: Path,
    test_root: Path,
) -> dict[str, Any]:
    """Inspect exact file duplicates and filename sequence overlap."""
    train_files = [
        path
        for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    test_files = [
        path
        for path in test_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    train_digests: dict[str, list[str]] = {}
    for path in train_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        train_digests.setdefault(digest, []).append(path.name)
    test_digests: dict[str, list[str]] = {}
    for path in test_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        test_digests.setdefault(digest, []).append(path.name)
    exact_duplicate_hashes = set(train_digests) & set(test_digests)

    train_sequences: dict[tuple[str, int], set[str]] = {}
    test_sequences: dict[tuple[str, int], set[str]] = {}
    for path, target in [
        *((path, train_sequences) for path in train_files),
        *((path, test_sequences) for path in test_files),
    ]:
        match = SEQUENCE_FILENAME.fullmatch(path.stem)
        if match:
            key = (match.group(1).upper(), int(match.group(2)))
            target.setdefault(key, set()).add(path.parent.name)

    shared_sequences = set(train_sequences) & set(test_sequences)
    conflicting_labels = sorted(
        (
            symbol,
            index,
            sorted(train_sequences[(symbol, index)]),
            sorted(test_sequences[(symbol, index)]),
        )
        for symbol, index in shared_sequences
        if train_sequences[(symbol, index)] != test_sequences[(symbol, index)]
    )
    train_multi_class_stems = sorted(
        f"{symbol}_{index}"
        for (symbol, index), labels in train_sequences.items()
        if len(labels) > 1
    )
    test_multi_class_stems = sorted(
        f"{symbol}_{index}"
        for (symbol, index), labels in test_sequences.items()
        if len(labels) > 1
    )
    train_repeated_hashes = sum(
        len(paths) - 1 for paths in train_digests.values() if len(paths) > 1
    )
    test_repeated_hashes = sum(
        len(paths) - 1 for paths in test_digests.values() if len(paths) > 1
    )
    return {
        "train_image_count": len(train_files),
        "test_image_count": len(test_files),
        "duplicate_file_count_within_train": train_repeated_hashes,
        "duplicate_file_count_within_test": test_repeated_hashes,
        "exact_duplicate_file_hashes": len(exact_duplicate_hashes),
        "shared_symbol_index_stems": len(shared_sequences),
        "shared_symbol_index_examples": [
            f"{symbol}_{index}"
            for symbol, index in sorted(shared_sequences)[:20]
        ],
        "train_multi_class_stem_count": len(train_multi_class_stems),
        "train_multi_class_stem_examples": train_multi_class_stems[:20],
        "test_multi_class_stem_count": len(test_multi_class_stems),
        "test_multi_class_stem_examples": test_multi_class_stems[:20],
        "shared_stem_label_set_mismatches": len(conflicting_labels),
        "shared_stem_label_set_mismatch_examples": [
            {
                "stem": f"{symbol}_{index}",
                "train_classes": train_labels,
                "test_classes": test_labels,
            }
            for symbol, index, train_labels, test_labels in conflicting_labels[:20]
        ],
        "split_provenance_verified": False,
        "warning": (
            "The archive includes no provenance or split documentation. Shared "
            "symbol/index filename stems are not proof of identical images or "
            "temporal leakage, but Test independence cannot be established."
        ),
    }


def evaluate(
    data_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    train_split: str = "Train",
    test_split: str = "Test",
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Evaluation report already exists: {output_path}. "
            "Choose a new --output path or pass --overwrite explicitly."
        )

    train_root = (Path(data_dir) / train_split).resolve(strict=True)
    test_root = (Path(data_dir) / test_split).resolve(strict=True)
    if not train_root.is_dir() or not test_root.is_dir():
        raise ValueError("Train and Test split paths must both be directories.")
    if train_root == test_root:
        raise ValueError("Train and Test splits must resolve to different directories.")

    require_partition_images(train_root.parent, train_root.name)
    require_partition_images(test_root.parent, test_root.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    image_size = checkpoint["image_size"]

    overlap_audit = audit_train_test_overlap(train_root, test_root)
    LOGGER.warning(
        "Test split provenance is unverified: %s matching filename stems; "
        "%s shared exact file hashes; %s shared stems with different label sets.",
        overlap_audit["shared_symbol_index_stems"],
        overlap_audit["exact_duplicate_file_hashes"],
        overlap_audit["shared_stem_label_set_mismatches"],
    )

    test_data = CandlestickImageFolder(test_root, image_size=image_size)
    train_data = CandlestickImageFolder(train_root, image_size=image_size)
    if not test_data:
        raise ValueError(f"Test partition '{test_root}' contains no images.")
    if not train_data:
        raise ValueError(f"Training partition '{train_root}' contains no images.")

    model = CandlestickCNN(num_classes=len(INDEX_TO_CLASS))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    truth: list[int] = []
    predicted: list[int] = []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if logits.ndim != 2 or logits.shape[1] != len(INDEX_TO_CLASS):
                raise ValueError("Model returned an unexpected number of output classes.")
            total_loss += float(criterion(logits, labels.to(device)).item())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            truth.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(range(len(INDEX_TO_CLASS))),
        zero_division=0,
    )
    train_counts = Counter(train_data.targets)
    majority_class = train_counts.most_common(1)[0][0]
    result: dict[str, object] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": checkpoint.get("model_version"),
        "class_to_idx": CLASS_TO_INDEX,
        "class_names": list(INDEX_TO_CLASS),
        "test_samples": len(truth),
        "loss": total_loss / len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": {
            INDEX_TO_CLASS[i]: float(precision[i])
            for i in range(len(INDEX_TO_CLASS))
        },
        "recall": {
            INDEX_TO_CLASS[i]: float(recall[i])
            for i in range(len(INDEX_TO_CLASS))
        },
        "f1": {
            INDEX_TO_CLASS[i]: float(f1[i])
            for i in range(len(INDEX_TO_CLASS))
        },
        "confusion_matrix": confusion_matrix(
            truth,
            predicted,
            labels=list(range(len(INDEX_TO_CLASS))),
        ).tolist(),
        "training_majority_baseline_test_accuracy": float(
            sum(actual == majority_class for actual in truth) / len(truth)
        ),
        "training_majority_baseline_class": INDEX_TO_CLASS[majority_class],
        "dataset_split_audit": overlap_audit,
        "interpretation_warning": (
            "These metrics measure agreement with the archive's folder labels. "
            "The Down/Up label meaning and test split provenance are unverified; "
            "they must not be interpreted as future-market accuracy."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_result = json.dumps(result, indent=2)
    if overwrite:
        output_path.write_text(serialized_result, encoding="utf-8")
    else:
        try:
            with output_path.open("x", encoding="utf-8") as report_file:
                report_file.write(serialized_result)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Evaluation report already exists: {output_path}. "
                "Choose a new --output path or pass --overwrite explicitly."
            ) from exc
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN against archive/Test (never used in training)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("archive"))
    parser.add_argument("--train-split", default="Train")
    parser.add_argument("--test-split", default="Test")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/candlestick_cnn.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/evaluation_metrics.json"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evaluation report at --output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    evaluate(
        args.data_dir,
        args.checkpoint,
        args.output,
        train_split=args.train_split,
        test_split=args.test_split,
        overwrite=args.overwrite,
    )
