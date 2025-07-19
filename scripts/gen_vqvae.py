import argparse
import os
from physaug.vqvae.infer import reconstruct_folder
from physaug.utils.io import load_image_folder, save_image
from tqdm import tqdm
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VQ-VAE reconstructions")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save reconstructed images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VQ-VAE model checkpoint")
    parser.add_argument("--mode", type=str, choices=["rgb", "gray"], default="rgb", help="Image mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    images, filenames = load_image_folder(args.input_dir, mode=args.mode)

    reconstructions = reconstruct_folder(images, args.checkpoint, device=args.device)

    for rec, name in tqdm(zip(reconstructions, filenames), total=len(images)):
        save_path = os.path.join(args.output_dir, name)
        save_image(rec, save_path, mode=args.mode)


if __name__ == "__main__":
    main()
