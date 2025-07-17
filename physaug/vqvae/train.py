import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from physaug.vqvae.vqvae import VQVAE
from physaug.utils.logger import get_logger
from physaug.utils.io import save_model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = get_logger("VQ-VAE-Train")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=args.dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = VQVAE(in_channels=3, hidden_dims=args.hidden_dim, embedding_dim=args.embedding_dim, num_embeddings=args.num_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            x_recon, vq_loss = model(imgs)
            recon_loss = criterion(x_recon, imgs)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"vqvae_epoch_{epoch+1}.pt")
        save_model(model, ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VQ-VAE on image folder")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint output path')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_embeddings', type=int, default=512)
    args = parser.parse_args()

    train(args)