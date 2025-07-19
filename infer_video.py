import os
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
from physaug.vqvae.vqvae import VQVAE
from physaug.utils.logger import setup_logger
from physaug.utils.config import load_config

def infer_video(video_path, output_path, checkpoint, config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    logger = setup_logger("infer_video", cfg["log_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VQVAE().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(cfg["vqvae"]["image_size"]), transforms.ToTensor()])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (cfg["vqvae"]["image_size"][1], cfg["vqvae"]["image_size"][0])
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _, _ = model(tensor)
        recon = recon.squeeze(0).cpu().mul(255).byte().permute(1, 2, 0).numpy()
        recon = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
        out.write(recon)
    
    cap.release()
    out.release()
    logger.info(f"Reconstructed video saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct video using VQ-VAE")
    parser.add_argument("--video_path", required=True, help="Input video file")
    parser.add_argument("--output_path", required=True, help="Output video file")
    parser.add_argument("--checkpoint", required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    infer_video(args.video_path, args.output_path, args.checkpoint, args.config)