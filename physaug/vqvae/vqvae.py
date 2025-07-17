# physaug/vqvae/vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        B, C, H, W = x.shape
        flat_x = x.permute(0, 2, 3, 1).contiguous()
        flat_x = flat_x.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_x, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight ** 2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight).view(B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        return quantized, loss, encoding_indices.view(B, H, W)


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 3, 1, 1),
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, quantized