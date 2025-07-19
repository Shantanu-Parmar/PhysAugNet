import os
from physaug.augment.thermal import apply_thermal_augmentation
from physaug.utils.io import load_image_folder, save_image
from physaug.utils.logger import setup_logger
from physaug.utils.config import load_config

def main(input_dir, output_dir, config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    logger = setup_logger("augment_thermal", cfg["log_dir"])
    os.makedirs(output_dir, exist_ok=True)
    images, names = load_image_folder(input_dir, cfg["vqvae"]["image_size"])
    for img, name in zip(images, names):
        aug_img = apply_thermal_augmentation(img)
        save_image(aug_img, f"{output_dir}/{name}")
    logger.info(f"Thermal augmentations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.config)