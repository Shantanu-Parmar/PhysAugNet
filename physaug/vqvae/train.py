import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from .vqvae import VQVAE
from physaug.utils.logger import setup_logger
from physaug.utils.io import save_image

class VQVAETrainer:
    def __init__(self, config):
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = setup_logger("vqvae_trainer", config["log_dir"])
        self.model = VQVAE().to(self.device)
        lr = float(config["vqvae"]["learning_rate"])  # Convert string to float
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def get_dataloader(self):
        transform = transforms.Compose([transforms.Resize(tuple(self.cfg["vqvae"]["image_size"])), transforms.ToTensor()])
        dataset = datasets.ImageFolder(root=self.cfg["dataset_dir"], transform=transform)
        return DataLoader(dataset, batch_size=self.cfg["vqvae"]["batch_size"], shuffle=True, num_workers=self.cfg["vqvae"]["num_workers"])

    def train(self):
        dataloader = self.get_dataloader()
        self.model.train()
        for epoch in range(self.cfg["vqvae"]["num_epochs"]):
            running_loss = 0
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                recon, vq_loss, _ = self.model(imgs)
                recon_loss = self.criterion(recon, imgs)
                loss = recon_loss + vq_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1}/{self.cfg['vqvae']['num_epochs']}: Loss={avg_loss:.4f}")
            if (epoch + 1) % self.cfg["vqvae"]["save_interval"] == 0:
                ckpt_path = f"{self.cfg['vqvae']['checkpoint_dir']}/vqvae_{epoch+1}.pth"
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"Saved checkpoint: {ckpt_path}")