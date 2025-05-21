import cv2


def convert_to_binary_watermark(image):
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Make transparent pixels white
        mask = image[:,:,3] == 0
        image_copy = image.copy()  # Create a copy to avoid modifying the original
        image_copy[mask] = [255, 255, 255, 255]
        # Convert to grayscale, dropping alpha channel
        grayscale = cv2.cvtColor(image_copy, cv2.COLOR_BGRA2GRAY)
    elif len(image.shape) == 3:
        # Convert BGR to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        grayscale = image.copy()

    _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    return binary
