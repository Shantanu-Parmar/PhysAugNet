import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
        distances = torch.cdist(flat_x, self.embedding.weight)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.embedding.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)
        loss = F.mse_loss(quantized.detach(), x) * self.commitment_cost + F.mse_loss(quantized, x.detach())
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices.view(x.shape[0], x.shape[2], x.shape[3])

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, embedding_dim, 4, 2, 1), nn.ReLU(),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        recon = self.decoder(quantized)
        return recon, vq_loss, quantized