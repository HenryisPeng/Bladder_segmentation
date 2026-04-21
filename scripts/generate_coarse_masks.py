from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate coarse bladder masks from ultrasound images using classical image processing."
    )
    parser.add_argument("--image-root", type=str, default="data/bladdercancer", help="Root folder containing train/val/test.")
    parser.add_argument("--output-root", type=str, default="data/bladder_masks_auto", help="Output folder for coarse masks.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument("--save-overlay", action="store_true", help="Save contour overlays for visual inspection.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated masks.")
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])


def preprocess(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    return cv2.equalizeHist(blurred)


def centrality_score(cx: float, cy: float, w: int, h: int) -> float:
    nx = (cx - w / 2.0) / (w / 2.0)
    ny = (cy - h / 2.0) / (h / 2.0)
    distance = np.sqrt(nx * nx + ny * ny)
    return float(max(0.0, 1.0 - distance))


def border_touching(x: int, y: int, bw: int, bh: int, w: int, h: int) -> bool:
    margin_x = max(8, int(0.02 * w))
    margin_y = max(8, int(0.02 * h))
    return x <= margin_x or y <= margin_y or (x + bw) >= (w - margin_x) or (y + bh) >= (h - margin_y)


def component_score(
    component_mask: np.ndarray,
    processed_gray: np.ndarray,
) -> float:
    h, w = component_mask.shape
    ys, xs = np.where(component_mask > 0)
    if len(xs) == 0:
        return -1e9

    x, y, bw, bh = cv2.boundingRect(np.column_stack([xs, ys]))
    area = float(component_mask.sum() / 255.0)
    area_ratio = area / float(h * w)

    if area_ratio < 0.003 or area_ratio > 0.45:
        return -1e9

    cx = float(xs.mean())
    cy = float(ys.mean())
    central = centrality_score(cx, cy, w, h)
    darkness = 1.0 - float(processed_gray[component_mask > 0].mean()) / 255.0

    perimeter = cv2.arcLength(cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
    circularity = 0.0 if perimeter <= 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
    circularity = max(0.0, min(circularity, 1.0))

    aspect = min(bw, bh) / max(bw, bh)
    border_penalty = 0.4 if border_touching(x, y, bw, bh, w, h) else 0.0

    return (
        2.8 * darkness
        + 2.2 * central
        + 1.2 * circularity
        + 0.8 * aspect
        + 1.0 * min(area_ratio / 0.08, 1.0)
        - border_penalty
    )


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    mask = cv2.medianBlur(mask, 5)
    return mask


def pick_best_component(candidate_mask: np.ndarray, processed_gray: np.ndarray) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(candidate_mask)
    best_score = -1e9
    best_mask = np.zeros_like(candidate_mask)

    for label in range(1, num_labels):
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        score = component_score(component, processed_gray)
        if score > best_score:
            best_score = score
            best_mask = component

    return best_mask


def generate_mask(gray: np.ndarray) -> np.ndarray:
    processed = preprocess(gray)

    candidate_masks: list[np.ndarray] = []
    percentiles = [8, 12, 16, 20, 24, 28]
    for percentile in percentiles:
        threshold = int(np.percentile(processed, percentile))
        candidate = np.where(processed <= threshold, 255, 0).astype(np.uint8)
        candidate = clean_binary_mask(candidate)
        candidate_masks.append(candidate)

    _, otsu_inv = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidate_masks.append(clean_binary_mask(otsu_inv))

    best_score = -1e9
    best_mask = np.zeros_like(gray)

    for candidate in candidate_masks:
        component = pick_best_component(candidate, processed)
        score = component_score(component, processed)
        if score > best_score:
            best_score = score
            best_mask = component

    if best_score < -1e8:
        return np.zeros_like(gray)

    return clean_binary_mask(best_mask)


def save_overlay(image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), color)


def process_split(
    image_root: Path,
    output_root: Path,
    split: str,
    save_overlay_flag: bool,
    overwrite: bool,
) -> None:
    image_dir = image_root / split
    if not image_dir.exists():
        print(f"Skip split '{split}': directory not found -> {image_dir}")
        return

    images = list_images(image_dir)
    if not images:
        print(f"Skip split '{split}': no images found in {image_dir}")
        return

    mask_dir = output_root / split
    overlay_dir = output_root / f"{split}_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay_flag:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for image_path in tqdm(images, desc=f"Generating {split} masks"):
        output_mask_path = mask_dir / f"{image_path.stem}.png"
        output_overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"

        if output_mask_path.exists() and not overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: failed to read image -> {image_path}")
            continue

        mask = generate_mask(image)
        cv2.imwrite(str(output_mask_path), mask)

        if save_overlay_flag:
            save_overlay(image, mask, output_overlay_path)

        generated += 1

    print(f"{split}: generated={generated}, skipped={skipped}, output={mask_dir}")


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    print(f"Image root: {image_root.resolve()}")
    print(f"Output root: {output_root.resolve()}")

    for split in args.splits:
        process_split(
            image_root=image_root,
            output_root=output_root,
            split=split,
            save_overlay_flag=args.save_overlay,
            overwrite=args.overwrite,
        )

    print("\nThese masks are coarse pseudo-labels. Review and correct them before formal training.")


if __name__ == "__main__":
    main()
