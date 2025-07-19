from .thermal import apply_thermal_augmentation
from .grain import add_grain
from torchvision.utils import save_image
from ..utils.io import load_image_folder

def apply_combined_augmentation(input_dir, output_dir, checkpoint, config_path="configs/default.yaml"):
    from .vqvae import VQVAE
    import torch
    from physaug.utils.config import load_config
    from physaug.utils.logger import setup_logger
    cfg = load_config(config_path)
    logger = setup_logger("combined", cfg["log_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQVAE().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
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
