from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bladder_segmentation.dataset import Sample, UltrasoundSegmentationDataset, list_images
from bladder_segmentation.model import UNet
from bladder_segmentation.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation inference on bladder ultrasound images.")
    parser.add_argument("--image-root", type=str, required=True, help="Folder that contains images, or a split folder like bladdercancer/test.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-overlay", action="store_true", help="Save contour overlay images.")
    return parser.parse_args()


def build_inference_samples(image_root: str | Path) -> list[Sample]:
    image_dir = Path(image_root)
    image_paths = list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return [Sample(image_path=path, mask_path=None) for path in image_paths]


def load_checkpoint(model: UNet, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)


def save_overlay_image(image_path: Path, mask: np.ndarray, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), color_image)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    samples = build_inference_samples(args.image_root)
    dataset = UltrasoundSegmentationDataset(
        samples=samples,
        image_size=args.image_size,
        augment=False,
        return_paths=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    output_dir = ensure_dir(args.output_dir)
    mask_dir = ensure_dir(output_dir / "masks")
    overlay_dir = ensure_dir(output_dir / "overlays") if args.save_overlay else None

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    for batch in tqdm(loader, desc="Predicting"):
        images = batch["image"].to(device)
        image_paths = batch["image_path"]

        logits = model(images)
        probs = torch.sigmoid(logits)
        masks = (probs > args.threshold).float().cpu().numpy()

        for i, image_path_str in enumerate(image_paths):
            image_path = Path(image_path_str)
            mask = (masks[i, 0] * 255).astype(np.uint8)

            save_mask(mask, mask_dir / f"{image_path.stem}.png")

            if overlay_dir is not None:
                save_overlay_image(image_path, mask, overlay_dir / f"{image_path.stem}_overlay.png")

    print(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
