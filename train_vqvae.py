import torch
from physaug.vqvae.vqvae import VQVAE
from physaug.utils.config import load_config
from physaug.utils.io import get_dataloaders
from physaug.utils.logger import setup_logger

def main(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    logger = setup_logger("train_vqvae", cfg["log_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _ = get_dataloaders(cfg)
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["vqvae"]["learning_rate"])

    for epoch in range(cfg["vqvae"]["num_epochs"]):
        model.train()
        running_loss = 0
        for batch in train_loader:
            imgs = batch[0].to(device)
            recon, vq_loss, _ = model(imgs)
            loss = torch.nn.MSELoss()(recon, imgs) + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{cfg['vqvae']['num_epochs']}: Loss={running_loss/len(train_loader):.4f}")
        if (epoch + 1) % cfg["vqvae"]["save_interval"] == 0:
            torch.save(model.state_dict(), f"{cfg['vqvae']['checkpoint_dir']}/vqvae_{epoch+1}.pth")
            logger.info(f"Saved checkpoint: {cfg['vqvae']['checkpoint_dir']}/vqvae_{epoch+1}.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)