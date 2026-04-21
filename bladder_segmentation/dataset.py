from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path | None


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def resolve_mask_path(mask_root: Path, split: str, image_path: Path) -> Path:
    return mask_root / split / f"{image_path.stem}.png"


def build_samples(
    image_root: str | Path,
    split: str,
    mask_root: str | Path | None = None,
    require_masks: bool = True,
) -> list[Sample]:
    image_dir = Path(image_root) / split
    image_paths = list_images(image_dir)

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    samples: list[Sample] = []
    missing_masks: list[Path] = []

    for image_path in image_paths:
        mask_path = None
        if mask_root is not None:
            candidate = resolve_mask_path(Path(mask_root), split, image_path)
            if candidate.exists():
                mask_path = candidate
            elif require_masks:
                missing_masks.append(candidate)
        elif require_masks:
            missing_masks.append(Path("<mask_root_not_provided>") / split / f"{image_path.stem}.png")

        samples.append(Sample(image_path=image_path, mask_path=mask_path))

    if missing_masks:
        preview = "\n".join(str(p) for p in missing_masks[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} mask files for split '{split}'. "
            f"Expected examples:\n{preview}"
        )

    return samples


class UltrasoundSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[Sample],
        image_size: int = 256,
        augment: bool = False,
        return_paths: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment
        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        return image

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        return mask

    def _augment(self, image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()

        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-0.08, 0.08)
            image = np.clip(image * alpha + beta, 0.0, 1.0)

        return image, mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        mask = self._load_mask(sample.mask_path) if sample.mask_path else None

        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image).unsqueeze(0)

        if mask is None:
            item = {"image": image_tensor}
        else:
            item = {
                "image": image_tensor,
                "mask": torch.from_numpy(mask).unsqueeze(0),
            }

        if self.return_paths:
            item["image_path"] = str(sample.image_path)
            if sample.mask_path is not None:
                item["mask_path"] = str(sample.mask_path)

        return item
