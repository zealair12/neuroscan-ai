from PIL import Image
import numpy as np

def preprocess_image(image, size=(128, 128)):
    image = image.convert('L')
    image = image.resize(size)
    image_array = np.array(image) / 255.0
    flattened = image_array.flatten()
    return flattened
