from __future__ import annotations

import argparse
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create mask folder scaffold for segmentation labels.")
    parser.add_argument("--image-root", type=str, default="data/bladdercancer")
    parser.add_argument("--mask-root", type=str, default="data/bladder_masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    mask_root = Path(args.mask_root)

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    print(f"Image root: {image_root.resolve()}")
    print(f"Mask root: {mask_root.resolve()}")

    for split in SPLITS:
        image_dir = image_root / split
        output_dir = mask_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0
        if image_dir.exists():
            image_count = len([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

        print(f"{split}: created {output_dir}  |  images to label: {image_count}")

    print("\nMask naming rule:")
    print("  image: data/bladdercancer/train/10.jpg")
    print("  mask : data/bladder_masks/train/10.png")
    print("\nMask pixel values should be 0 for background and 255 for bladder region.")


if __name__ == "__main__":
    main()
