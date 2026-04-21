from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bladder_segmentation.dataset import UltrasoundSegmentationDataset, build_samples
from bladder_segmentation.metrics import dice_score, iou_score
from bladder_segmentation.model import UNet
from bladder_segmentation.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained bladder segmentation model on a dataset split.")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory that contains split folders for images.")
    parser.add_argument("--mask-root", type=str, required=True, help="Root directory that contains split folders for masks.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(model: UNet, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    samples = build_samples(args.image_root, args.split, args.mask_root, require_masks=True)
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

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    sample_results: list[dict[str, float | str]] = []
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        image_paths = batch["image_path"]

        logits = model(images)
        batch_size = images.size(0)

        for i in range(batch_size):
            sample_logits = logits[i : i + 1]
            sample_masks = masks[i : i + 1]
            sample_dice = dice_score(sample_logits, sample_masks, args.threshold)
            sample_iou = iou_score(sample_logits, sample_masks, args.threshold)
            image_path = Path(image_paths[i])

            sample_results.append(
                {
                    "image": image_path.name,
                    "dice": sample_dice,
                    "iou": sample_iou,
                }
            )
            total_dice += sample_dice
            total_iou += sample_iou
            total_samples += 1

    mean_dice = total_dice / total_samples
    mean_iou = total_iou / total_samples

    output_dir = ensure_dir(Path(args.output_dir) / args.split)
    summary = {
        "split": args.split,
        "num_samples": total_samples,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "image_size": args.image_size,
        "results": sample_results,
    }
    save_json(summary, output_dir / "metrics.json")

    print(f"Split: {args.split}")
    print(f"Samples: {total_samples}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
