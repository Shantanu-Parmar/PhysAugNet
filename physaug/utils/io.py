import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

def load_image_folder(folder, image_size=(128, 128)):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    images, names = [], []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('jpg', 'png', 'jpeg')):
            img = Image.open(f"{folder}/{fname}").convert('RGB')
            images.append(transform(img))
            names.append(fname)
    return images, names

def save_image(tensor, path):
    save_image(tensor.clamp(0, 1), path)