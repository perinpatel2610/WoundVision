import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained segmentation model
unet_model = load_model(r'models/best_segmentation_model.h5', compile=False)

def segment_image(image_input, image_size=(512, 512)):
    """
    Segment an image using the trained U-Net model and return a binary mask.
    """
    if isinstance(image_input, str):
        original_image = cv2.imread(image_input)
        if original_image is None:
            raise FileNotFoundError(f"Image not found or invalid path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        original_image = image_input
    else:
        raise ValueError(f"Expected a file path or numpy array, got {type(image_input)}")

    resized_image = cv2.resize(original_image, image_size)
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    predicted_mask = unet_model.predict(input_image)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]))

    binary_mask = (predicted_mask_resized > 0.5).astype(np.uint8)

    return binary_mask

def compute_wound_area(binary_mask, pixel_area_mm2=0.01):
    """
    Computes wound area from a binary mask.

    Returns:
        area_pixels (int): Total wound pixels
        area_mm2 (float): Converted area in square millimeters
    """
    area_pixels = np.sum(binary_mask)
    area_mm2 = area_pixels * pixel_area_mm2
    return area_pixels, area_mm2
