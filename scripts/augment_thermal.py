import argparse
import os
from physaug.augment.thermal import apply_thermal_augmentation
from physaug.augment.grain import apply_grain_noise
from physaug.utils.io import load_image_folder, save_image
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Apply thermal and grain noise augmentations")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save augmented images")
    parser.add_argument("--mode", type=str, choices=["rgb", "gray"], default="rgb", help="Image mode")
    parser.add_argument("--apply_grain", action="store_true", help="Apply grain noise")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    images, filenames = load_image_folder(args.input_dir, mode=args.mode)

    for img, name in tqdm(zip(images, filenames), total=len(images)):
        thermal = apply_thermal_augmentation(img)
        if args.apply_grain:
            thermal = apply_grain_noise(thermal)
        save_path = os.path.join(args.output_dir, name)
        save_image(thermal, save_path, mode=args.mode)


if __name__ == "__main__":
    main()
