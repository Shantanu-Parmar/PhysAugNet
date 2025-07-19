import torch

def add_grain(image, intensity=0.05):
    if not torch.is_tensor(image):
        image = torch.tensor(image)
    noise = torch.randn_like(image) * intensity
    return torch.clamp(image + noise, 0, 1)