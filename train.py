from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from bladder_segmentation.dataset import UltrasoundSegmentationDataset, build_samples
from bladder_segmentation.losses import BCEDiceLoss
from bladder_segmentation.metrics import dice_score, iou_score
from bladder_segmentation.model import UNet
from bladder_segmentation.utils import ensure_dir, plot_training_curves, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN U-Net for bladder ultrasound segmentation.")
    parser.add_argument("--image-root", type=str, default="data/bladdercancer", help="Root directory for train/val/test images.")
    parser.add_argument("--mask-root", type=str, default="data/bladder_masks", help="Root directory for train/val/test masks.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: Adam,
    criterion: BCEDiceLoss,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    device: torch.device,
    threshold: float,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        running_dice += dice_score(logits, masks, threshold) * images.size(0)
        running_iou += iou_score(logits, masks, threshold) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_dice / n, running_iou / n


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    plots_dir = ensure_dir(output_dir / "plots")

    train_samples = build_samples(args.image_root, "train", args.mask_root, require_masks=True)
    val_samples = build_samples(args.image_root, "val", args.mask_root, require_masks=True)

    train_dataset = UltrasoundSegmentationDataset(
        train_samples,
        image_size=args.image_size,
        augment=True,
    )
    val_dataset = UltrasoundSegmentationDataset(
        val_samples,
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    criterion = BCEDiceLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_dice = -1.0
    wait = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device, args.threshold)
        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_dice={val_dice:.4f}  "
            f"val_iou={val_iou:.4f}  "
            f"lr={lr:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_dice": val_dice,
            "val_iou": val_iou,
        }
        torch.save(checkpoint, checkpoints_dir / "last_model.pt")

        if val_dice > best_dice:
            best_dice = val_dice
            wait = 0
            torch.save(checkpoint, checkpoints_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    plot_training_curves(history, plots_dir / "training_curves.png")
    save_json(
        {
            "best_val_dice": best_dice,
            "history": history,
            "config": vars(args),
        },
        output_dir / "training_summary.json",
    )
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
