import os

import cv2


def load_image(file_path):
    """Load an image file and return it as a numpy array"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return None

    try:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Failed to load image {file_path}.")
            return None
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


def save_image(image, file_path):
    """Save the image to the specified path"""
    try:
        result = cv2.imwrite(file_path, image)
        return result
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False