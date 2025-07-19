import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from .vqvae import VQVAE
from physaug.utils.logger import setup_logger
from physaug.utils.config import load_config

def reconstruct_folder(input_dir, output_dir, model_path, image_size=(128, 128), config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    logger = setup_logger("infer", cfg["log_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    os.makedirs(output_dir, exist_ok=True)
    for fname in files:
        img = Image.open(f"{input_dir}/{fname}").convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _, _ = model(x)
        save_image(recon.clamp(0, 1), f"{output_dir}/{fname}")
    logger.info(f"Reconstructed images saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    reconstruct_folder(args.input_dir, args.output_dir, args.model_path, tuple(args.image_size), args.config)