import os
import torch
from physaug.vqvae.infer import reconstruct_folder
from physaug.utils.logger import setup_logger

def main(input_dir, output_dir, checkpoint, config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    logger = setup_logger("gen_vqvae", cfg["log_dir"])
    reconstruct_folder(input_dir, output_dir, checkpoint, cfg["vqvae"]["image_size"])
    logger.info(f"Reconstructed images saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.checkpoint, args.config)