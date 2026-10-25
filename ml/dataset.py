from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from ml.preprocessing import rgb_array_to_tensor

CLASS_TO_INDEX = {"Down": 0, "Up": 1}
INDEX_TO_CLASS = ("Down", "Up")
ARCHIVE_INSTRUMENT_FILENAME = re.compile(
    r"(?P<instrument>QQQ|SPY|BTC|ETH|SOL|BNB|ADA|XRP|AVAX|DOGE|LINK|DOT)_\d+",
    re.IGNORECASE,
)


def archive_instrument_group(path: str | Path) -> str:
    """Return the instrument prefix used to group supplied archive images."""
    match = ARCHIVE_INSTRUMENT_FILENAME.fullmatch(Path(path).stem)
    if match is None:
        raise ValueError(
            "Expected an archive filename with a QQQ_<index> or SPY_<index> stem; "
            f"got {Path(path).name!r}."
        )
    return match.group("instrument").upper()


class CandlestickImageFolder(Dataset):
    """ImageFolder dataset for the archive's Down/Up classes.

    Pillow decodes image files to RGB, then OpenCV performs the shared resize
    and normalization used by inference.
    """

    def __init__(self, root: str | Path, image_size: int = 128) -> None:
        self.image_size = image_size
        self.dataset = ImageFolder(
            str(root),
            transform=self._preprocess,
            allow_empty=True,
        )
        discovered = {name.lower(): index for name, index in self.dataset.class_to_idx.items()}
        if discovered != {"down": 0, "up": 1}:
            raise ValueError(
                f"Expected Down/ and Up/ folders with labels "
                f"{CLASS_TO_INDEX}; found {self.dataset.class_to_idx} in {root}."
            )
        self.dataset.class_to_idx = CLASS_TO_INDEX.copy()
        self.dataset.classes = list(CLASS_TO_INDEX)

    def _preprocess(self, image) -> torch.Tensor:
        rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return rgb_array_to_tensor(rgb_image, self.image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    @property
    def targets(self) -> list[int]:
        return list(self.dataset.targets)

    @property
    def samples(self):
        return self.dataset.samples
