import os
import torch
from physaug.vqvae.vqvae import VQVAE
from physaug.augment.thermal import apply_thermal_augmentation
from physaug.augment.grain import add_grain
from physaug.utils.io import load_image_folder, save_image
from physaug.utils.logger import setup_logger
from physaug.utils.config import load_config

def main(input_dir, output_dir, checkpoint, config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    logger = setup_logger("augment_combined", cfg["log_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQVAE().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    images, names = load_image_folder(input_dir, cfg["vqvae"]["image_size"])
    for img, name in zip(images, names):
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _, _ = model(img)
        recon = recon.squeeze(0).cpu()
        aug = apply_thermal_augmentation(recon)
        aug = add_grain(aug)
        save_image(aug, f"{output_dir}/{name}")
    logger.info(f"Combined augmentations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.vqvae_ckpt, args.config)