import argparse
import yaml
import torch
import os
from physaug.vqvae.train import VQVAETrainer
from physaug.vqvae.vqvae import VQVAE
from physaug.utils.logger import setup_logging
from physaug.utils.config import load_config
from physaug.utils.io import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE on metal defect images")

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    logger, writer = setup_logging(cfg)
    logger.info("Starting VQ-VAE training")

    train_loader, val_loader = get_dataloaders(cfg)
    model = VQVAE(in_channels=3 if cfg["mode"] == "rgb" else 1)
    model.to(args.device)

    trainer = VQVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        writer=writer,
        logger=logger,
        resume_path=args.resume,
        device=args.device
    )

    trainer.train()


if __name__ == "__main__":
    main()
