import cv2
import numpy as np


def resize_image(image, max_width=1000):
    """
    Resize image while maintaining aspect ratio.
    """

    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

    return resized


def crop_black_area(image):
    """
    Remove unnecessary black borders from the panorama.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    return image[y:y+h, x:x+w]