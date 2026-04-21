from __future__ import annotations

import torch


def _flatten_predictions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    truth = targets.view(targets.size(0), -1)
    return preds, truth


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    preds, truth = _flatten_predictions(logits, targets, threshold)
    intersection = (preds * truth).sum(dim=1)
    denominator = preds.sum(dim=1) + truth.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice.mean().item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    preds, truth = _flatten_predictions(logits, targets, threshold)
    intersection = (preds * truth).sum(dim=1)
    union = preds.sum(dim=1) + truth.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
