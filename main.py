import argparse
from physaug.vqvae.train import VQVAETrainer
from physaug.vqvae.infer import reconstruct_folder
from physaug.augment.thermal import apply_thermal_augmentation
from physaug.augment.combined import apply_combined_augmentation
from physaug.utils.config import load_config
from physaug.utils.io import load_image_folder, save_image
from physaug.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="PhysAugNet CLI")
    parser.add_argument("mode", choices=["train_vqvae", "reconstruct", "augment_tg", "augment_combined"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("main", cfg["log_dir"])

    if args.mode == "train_vqvae":
        trainer = VQVAETrainer(cfg)
        trainer.train()
    elif args.mode == "reconstruct":
        reconstruct_folder(cfg["input_dir"], cfg["output_dir"], cfg["vqvae_path"], cfg["vqvae"]["image_size"])
        logger.info(f"Reconstructed images saved to {cfg['output_dir']}")
    elif args.mode == "augment_tg":
        images, names = load_image_folder(cfg["input_dir"])
        for img, name in zip(images, names):
            aug_img = apply_thermal_augmentation(img)
            save_image(aug_img, f"{cfg['output_dir']}/{name}")
        logger.info(f"Thermal augmentations saved to {cfg['output_dir']}")
    elif args.mode == "augment_combined":
        apply_combined_augmentation(cfg["input_dir"], cfg["output_dir"], cfg["vqvae_path"])
        logger.info(f"Combined augmentations saved to {cfg['output_dir']}")

if __name__ == "__main__":
    main()