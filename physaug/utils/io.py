
# physaug/utils/io.py
import os
from PIL import Image
from torchvision import transforms

def load_images_from_folder(folder, image_size=(128, 128)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    images = []
    names = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img = transform(img)
            images.append(img)
            names.append(filename)
    return images, names

def save_image_tensor(tensor, path):
    from torchvision.utils import save_image
    save_image(tensor, path)


def make_output_folder(base, name):
    output_path = os.path.join(base, name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

