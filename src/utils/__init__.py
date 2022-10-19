from PIL import Image
import numpy as np
import cv2


def read_image(image_path: str = None, opencv: bool = False):
    if opencv:
        image = cv2.imread("/path/to/image.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    return image

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))

    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range

def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image

def draw_rectangle_by_class(image, label, color):
    image_height, image_width, _ = image.shape
    color = color.lstrip('#')
    color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=2)
    return image
    