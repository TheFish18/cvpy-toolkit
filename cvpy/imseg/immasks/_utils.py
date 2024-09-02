import cv2
import numpy as np


def _scale_image(image, factor=0.5):
    factor = round(factor, 2)

    if factor == 1.:
        return image
    else:
        width = int(image.shape[1] * factor)
        height = int(image.shape[0] * factor)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _color_gen():
    while True:
        b, g, r = np.random.randint(0, 256, 3, dtype=np.uint8)
        yield b, g, r


_COLOR_GEN = _color_gen()
