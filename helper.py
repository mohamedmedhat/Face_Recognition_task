import cv2
import numpy as np

def processing_img(img):
    """
    This function processes a single image or video frame for emotion detection.
    It resizes it to 48x48 pixels, converts it to grayscale, normalizes, and reshapes for the model.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img