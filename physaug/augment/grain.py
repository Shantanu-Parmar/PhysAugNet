# physaug/augment/grain.py
import torch
import numpy as np
import torchvision.transforms.functional as TF

def add_grain(image, intensity=0.05):
    """
    Adds grain noise to an image tensor (C, H, W), values in [0,1].
    Args:
        image: torch.Tensor of shape (C, H, W)
        intensity: float, standard deviation of Gaussian noise
    Returns:
        torch.Tensor: noisy image
    """
    if not torch.is_tensor(image):
        raise TypeError("Input must be a torch.Tensor")

    noise = torch.randn_like(image) * intensity
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)
