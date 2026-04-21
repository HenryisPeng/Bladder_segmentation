from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bladder_segmentation.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize predicted masks against ground-truth masks.")
    parser.add_argument("--image-root", type=str, required=True, help="Directory with test images.")
    parser.add_argument("--gt-mask-root", type=str, required=True, help="Directory with ground-truth masks.")
    parser.add_argument("--pred-mask-root", type=str, required=True, help="Directory with predicted masks.")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations/test_compare")
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 127).astype(np.uint8)


def make_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(base, 0.65, overlay, 0.35, 0.0)


def make_error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    error = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)

    true_positive = (gt_mask == 1) & (pred_mask == 1)
    false_positive = (gt_mask == 0) & (pred_mask == 1)
    false_negative = (gt_mask == 1) & (pred_mask == 0)

    error[true_positive] = (0, 255, 0)
    error[false_positive] = (255, 0, 0)
    error[false_negative] = (0, 0, 255)
    return error


def main() -> None:
    args = parse_args()

    image_root = Path(args.image_root)
    gt_mask_root = Path(args.gt_mask_root)
    pred_mask_root = Path(args.pred_mask_root)
    output_dir = ensure_dir(args.output_dir)

    image_paths = list_images(image_root)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_root}")

    for image_path in tqdm(image_paths, desc="Visualizing"):
        gt_mask_path = gt_mask_root / f"{image_path.stem}.png"
        pred_mask_path = pred_mask_root / f"{image_path.stem}.png"

        if not gt_mask_path.exists():
            raise FileNotFoundError(f"Missing ground-truth mask: {gt_mask_path}")
        if not pred_mask_path.exists():
            raise FileNotFoundError(f"Missing predicted mask: {pred_mask_path}")

        image = load_grayscale(image_path)
        gt_mask = to_binary(load_grayscale(gt_mask_path))
        pred_mask = to_binary(load_grayscale(pred_mask_path))

        if gt_mask.shape != image.shape:
            gt_mask = cv2.resize(gt_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if pred_mask.shape != image.shape:
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_overlay = make_overlay(image, gt_mask, (0, 255, 0))
        pred_overlay = make_overlay(image, pred_mask, (0, 165, 255))
        error_map = make_error_map(gt_mask, pred_mask)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Image")
        axes[0, 1].imshow(gt_mask, cmap="gray")
        axes[0, 1].set_title("Ground Truth Mask")
        axes[0, 2].imshow(pred_mask, cmap="gray")
        axes[0, 2].set_title("Predicted Mask")

        axes[1, 0].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Ground Truth Overlay")
        axes[1, 1].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Prediction Overlay")
        axes[1, 2].imshow(error_map)
        axes[1, 2].set_title("Error Map")

        for ax in axes.ravel():
            ax.axis("off")

        fig.suptitle(
            f"{image_path.name} | TP=green FP=red FN=blue",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{image_path.stem}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved comparison figures to: {output_dir}")


if __name__ == "__main__":
    main()
