from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read polygon annotations from txt files and export mask PNG + JSON files."
    )
    parser.add_argument("--input-dir", type=str, default="data/seg_test", help="Folder containing image files and txt annotations.")
    parser.add_argument("--output-dir", type=str, default="data/bladder_masks", help="Folder to save generated mask PNG and JSON files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def find_image_path(input_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = input_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_annotation(txt_path: Path) -> dict:
    with txt_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_polygons(annotation: dict) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for box in annotation.get("boxes", []):
        points = box.get("points")
        if box.get("type") == "polygon" and points:
            polygons.append([(float(x), float(y)) for x, y in points])
    return polygons


def fill_polygons(width: int, height: int, polygons: list[list[tuple[float, float]]]) -> list[list[int]]:
    mask = [[0 for _ in range(width)] for _ in range(height)]

    for polygon in polygons:
        if polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]

        min_y = max(0, int(min(y for _, y in polygon)))
        max_y = min(height - 1, int(max(y for _, y in polygon)))

        for yi in range(min_y, max_y + 1):
            scan_y = yi + 0.5
            intersections: list[float] = []
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:]):
                if y1 == y2:
                    continue
                if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
                    x = x1 + (scan_y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(x)

            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 >= len(intersections):
                    break
                x_start = max(0, int(round(intersections[i])))
                x_end = min(width - 1, int(round(intersections[i + 1])))
                for xi in range(x_start, x_end + 1):
                    mask[yi][xi] = 255

    return mask


def png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    return (
        struct.pack("!I", len(chunk_data))
        + chunk_type
        + chunk_data
        + struct.pack("!I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
    )


def write_grayscale_png(mask: list[list[int]], output_path: Path) -> None:
    height = len(mask)
    width = len(mask[0]) if height > 0 else 0

    raw = bytearray()
    for row in mask:
        raw.append(0)
        raw.extend(bytes(row))

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    png += png_chunk(b"IEND", b"")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(png)


def export_one(txt_path: Path, output_dir: Path, overwrite: bool) -> tuple[bool, str]:
    stem = txt_path.stem
    try:
        annotation = load_annotation(txt_path)
    except json.JSONDecodeError:
        return False, f"skip {stem}: invalid json content"

    image_path = find_image_path(txt_path.parent, stem)
    if image_path is None:
        return False, f"skip {stem}: matching image not found"

    width = int(annotation["width"])
    height = int(annotation["height"])
    polygons = extract_polygons(annotation)
    if not polygons:
        return False, f"skip {stem}: no polygon annotations found"

    png_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"

    if not overwrite and png_path.exists() and json_path.exists():
        return False, f"skip {stem}: outputs already exist"

    mask = fill_polygons(width, height, polygons)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    write_grayscale_png(mask, png_path)

    return True, f"ok {stem}: wrote {png_path.name}, {json_path.name}"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt" and not path.name.startswith(".")
    )
    if not txt_files:
        raise FileNotFoundError(f"No txt annotation files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for txt_path in txt_files:
        wrote, message = export_one(txt_path, output_dir, args.overwrite)
        print(message)
        if wrote:
            created += 1
        else:
            skipped += 1

    print(f"\nDone. created={created}, skipped={skipped}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
