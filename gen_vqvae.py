import os
import torch
import logging
import datetime
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from vqvae_model.vqvae import VQVAE

# ========== Logging Setup ==========
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
log_path = f"logs/generate_vqvae_{timestamp}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# ========== Configuration ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (1024,1024)
CHECKPOINT = "vqvae_outputs/20250713_2057/vqvae_epoch34.pth" 

INPUT_DIRS = {
    "train": "dataset/images/train",
    "val": "dataset/images/val",
    "test": "dataset/images/test"
}
OUTPUT_DIR = "aug_images_vqvae"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Load Model ==========
model = VQVAE()
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
log.info(f"Loaded VQ-VAE from {CHECKPOINT}")

# ========== Preprocessing ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE[0], IMG_SIZE[1])),
    transforms.ToTensor()
])

# ========== Run Inference ==========
with torch.no_grad():
    for split, input_path in INPUT_DIRS.items():
        out_path = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_path, exist_ok=True)

        img_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        log.info(f"Processing {split}: {len(img_files)} images")

        for img_file in tqdm(img_files, desc=f"{split}"):
            in_file = os.path.join(input_path, img_file)
            out_file = os.path.join(out_path, img_file.replace(".png", "_vq.png"))

            img = Image.open(in_file).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            recon, _ = model(img_tensor)
            save_image(recon, out_file)

log.info("✅ VQ-VAE reconstructions generated and saved.")
