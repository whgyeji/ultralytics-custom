"""Run YOLOE inference after fixed ratio cropping every image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ultralytics import YOLOE


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/home/data/wanghonggang/dataset/drive_test/data_new/ck_20260302/images")
DEFAULT_WEIGHTS = ROOT / "runs/detect/train86/weights/last.pt"
DEFAULT_OUTPUT = ROOT / "runs/detect/ck_20260513_ratio_crop"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CROP_LEFT = 1 / 3
CROP_RIGHT = 1 / 3
CROP_TOP = 440 / 1080
CROP_BOTTOM = 0
import os
import shutil
if os.path.exists(DEFAULT_OUTPUT):
    shutil.rmtree(DEFAULT_OUTPUT)
    os.mkdir(DEFAULT_OUTPUT)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLOE on fixed-ratio crops of all images in a directory."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Input image directory")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLOE weights")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="Inference device, for example 0 or cpu")
    return parser.parse_args()


def fixed_ratio_crop(image):
    """Crop left/right/top by configured fractions."""
    height, width = image.shape[:2]
    left = int(round(width * CROP_LEFT))
    right = width - int(round(width * CROP_RIGHT))
    top = int(round(height * CROP_TOP))
    bottom = height - int(round(height * CROP_BOTTOM))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {source}")
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {weights}")

    image_paths = sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {source}")

    crop_dir = output / "crops"
    result_dir = output / "results"
    crop_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOE(str(weights))
    model.set_classes(["road_obstacle"])

    processed = 0
    skipped = 0
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{index}/{len(image_paths)}] Skip unreadable image: {image_path}")
            skipped += 1
            continue

        crop = fixed_ratio_crop(image)
        if crop is None:
            height, width = image.shape[:2]
            print(f"[{index}/{len(image_paths)}] Skip invalid crop: {image_path} ({width}x{height})")
            skipped += 1
            continue

        # Preserve subdirectories and use PNG to avoid adding another lossy JPEG encoding.
        relative_stem = image_path.relative_to(source).with_suffix("")
        crop_path = (crop_dir / relative_stem).with_suffix(".png")
        result_path = (result_dir / relative_stem).with_suffix(".png")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(crop_path), crop)
        predict_kwargs = {"source": crop, "imgsz": max(crop.shape[:2]), "conf": args.conf, "verbose": False}
        if args.device is not None:
            predict_kwargs["device"] = args.device
        result = model.predict(**predict_kwargs)[0]
        cv2.imwrite(str(result_path), result.plot())
        processed += 1
        print(f"[{index}/{len(image_paths)}] Saved: {result_path}")

    print(f"Done. Processed: {processed}, skipped: {skipped}, output: {output}")


if __name__ == "__main__":
    main()
