# physaug/augment/thermal.py
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

def thermal_augment(pil_image):
    """
    Applies thermal-style augmentation to a PIL image.
    Includes blur, hue shift, contrast change.
    Args:
        pil_image: PIL.Image
    Returns:
        PIL.Image
    """
    if not isinstance(pil_image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    # Apply random Gaussian blur
    if random.random() < 0.5:
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Convert to tensor for additional effects
    img_tensor = TF.to_tensor(pil_image)

    # Random contrast and hue
    img_tensor = T.ColorJitter(
        contrast=(0.8, 1.2),
        hue=(-0.1, 0.1),
        brightness=(0.9, 1.1),
        saturation=(0.8, 1.2)
    )(img_tensor)

    return TF.to_pil_image(torch.clamp(img_tensor, 0, 1))
