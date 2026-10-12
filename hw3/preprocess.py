"""Подготовка ImageNet-изображений без зависимости клиента от PyTorch."""
import numpy as np
from PIL import Image


def prepare_image(path):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    if width < height:
        size = (256, int(256 * height / width))
    else:
        size = (int(256 * width / height), 256)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = round((image.width - 224) / 2), round((image.height - 224) / 2)
    image = image.crop((left, top, left + 224, top + 224))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))
