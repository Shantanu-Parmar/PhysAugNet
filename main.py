# physaug/main.py

import argparse
import os
import torch

from physaug.utils.config import load_config_with_overrides
from physaug.vqvae.train import train_vqvae
from physaug.vqvae.infer import reconstruct_folder
from physaug.augment.thermal_grain import apply_thermal_grain
from physaug.augment.combined import apply_combined_augmentation


def main():
    parser = argparse.ArgumentParser(description="PhysAugNet CLI: VQ-VAE & Thermal-Grain Augmentation")

    parser.add_argument("--mode", type=str, choices=["train_vqvae", "reconstruct", "thermal_grain", "combined"],
                        required=False, help="Mode to run")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")

    # Optional CLI overrides
    parser.add_argument("--input_dir", type=str, help="Input directory for images")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--vqvae_path", type=str, help="Path to trained VQ-VAE model (.pth)")
    parser.add_argument("--image_size", type=int, nargs=2, help="Resize image to size, e.g., 128 128")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device override (cpu or cuda)")

    args = parser.parse_args()

    config = load_config_with_overrides(args.config, overrides={
        "mode": args.mode,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "vqvae_path": args.vqvae_path,
        "image_size": args.image_size,
        "device": args.device,
    })

    mode = config["mode"]

    if mode == "train_vqvae":
        train_vqvae(config)

    elif mode == "reconstruct":
        reconstruct_folder(
            input_dir=config["input_dir"],
            output_dir=config["output_dir"],
            model_path=config["vqvae_path"],
            image_size=tuple(config["image_size"]),
            device=torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        )

    elif mode == "thermal_grain":
        apply_thermal_grain(config["input_dir"], config["output_dir"])

    elif mode == "combined":
        apply_combined_augmentation(config["input_dir"], config["output_dir"])

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
