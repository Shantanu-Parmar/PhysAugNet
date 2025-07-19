from PIL import Image
import torchvision.transforms as T
import random
from torchvision.transforms import functional as TF

def apply_thermal_augmentation(image):
    if not isinstance(image, Image.Image):
        image = TF.to_pil_image(image)
    img = T.ColorJitter(contrast=(0.8, 1.2), hue=(-0.1, 0.1))(image)
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    return TF.to_tensor(img)