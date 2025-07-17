import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from .vqvae import VQVAE
from physaug.utils.logger import get_logger

logger = get_logger("infer")


def reconstruct_folder(input_dir, output_dir, model_path, image_size=(128, 128), device=None):
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = VQVAE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Loaded VQ-VAE model from {model_path}")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    logger.info(f"Found {len(files)} images in {input_dir}")

    for fname in tqdm(files, desc="Reconstructing with VQ-VAE"):
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            x_recon, _, _ = model(x)

        base_name, _ = os.path.splitext(fname)
        save_path = os.path.join(output_dir, base_name + ".png")
        save_image(x_recon.clamp(0, 1), save_path)

    logger.info(f"Reconstruction complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct images from folder using trained VQ-VAE.")
    parser.add_argument("--input_dir", required=True, help="Path to input image directory.")
    parser.add_argument("--output_dir", required=True, help="Path to save reconstructed images.")
    parser.add_argument("--model_path", required=True, help="Path to trained VQ-VAE .pth file.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(128, 128), help="Resize image to this size (W H).")
    parser.add_argument("--device", default=None, help="Device to use: cuda or cpu. Default: auto-detect.")

    args = parser.parse_args()
    reconstruct_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        image_size=tuple(args.image_size),
        device=args.device,
    )
