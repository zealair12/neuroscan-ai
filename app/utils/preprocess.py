from PIL import Image
import numpy as np

def preprocess_image(image, size=(224, 224)):
    """
    Preprocess the input image to match the model's expected shape.
    """
    image = image.convert("RGB")
    image = image.resize(size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array  # Final shape: (1, 224, 224, 3)
