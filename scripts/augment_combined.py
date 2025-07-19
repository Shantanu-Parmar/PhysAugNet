import os
import argparse
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from physaug.vqvae.vqvae import VQVAE
from physaug.augment.thermal import apply_thermal_augmentation
from physaug.augment.grain import apply_grain_noise
from physaug.utils.io import load_images_from_folder
from physaug.utils.logger import get_logger


def load_vqvae_model(checkpoint_path, device):
    model = VQVAE(img_channels=3, hidden_channels=128, embedding_dim=64, num_embeddings=512)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def augment_images(model, input_dir, output_dir, device, apply_grain):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.ToTensor()
    image_paths = load_images_from_folder(input_dir)

    for img_path in tqdm(image_paths, desc="Processing VQ+Thermal(+Grain)"):
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            recon, _ = model(tensor)

        # Remove batch
        recon = recon.squeeze(0).cpu().clamp(0, 1)

        # Apply thermal
        thermal = apply_thermal_augmentation(recon)

        # Apply optional grain
        if apply_grain:
            thermal = apply_grain_noise(thermal)

        save_image(thermal, os.path.join(output_dir, img_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQ-VAE + Thermal + Grain Augmentation")
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder')
    parser.add_argument('--vqvae_ckpt', type=str, required=True, help='Path to trained VQ-VAE checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--apply_grain', action='store_true', help='Apply grain noise after thermal')
    args = parser.parse_args()

    logger = get_logger("augment_combined")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"VQ-VAE Checkpoint: {args.vqvae_ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_vqvae_model(args.vqvae_ckpt, device)

    augment_images(model, args.input_dir, args.output_dir, device, args.apply_grain)

    logger.info("✅ Combined augmentation completed.")
