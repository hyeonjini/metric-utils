from PIL import Image
import numpy as np
import cv2


def read_image(image_path: str = None, opencv: bool = False):
    if opencv:
        image = cv2.imread("/path/to/image.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    image = Image.open(image_path).convert("rgb")
    image = np.array(image)
    return image
