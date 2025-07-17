# physaug/augment/combined.py
from PIL import Image
import torchvision.transforms.functional as TF

from .grain import add_grain
from .thermal import thermal_augment


def apply_combined_augmentation(pil_image, grain_intensity=0.05):
    """
    Applies thermal and grain augmentation to a PIL image.
    Args:
        pil_image: PIL.Image
        grain_intensity: float, std of Gaussian noise
    Returns:
        PIL.Image
    """
    # Apply thermal effect
    thermal_img = thermal_augment(pil_image)

    # Convert to tensor and add grain
    tensor_img = TF.to_tensor(thermal_img)
    grain_img = add_grain(tensor_img, intensity=grain_intensity)

    return TF.to_pil_image(grain_img)
