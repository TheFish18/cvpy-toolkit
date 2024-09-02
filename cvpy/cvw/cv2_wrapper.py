from pathlib import Path
from collections import namedtuple

import cv2
import numpy as np


def show_image(image: np.ndarray, wait: int = -1, window_name: str | None = None):
    """
    Show image.

    wraps cv2.imshow, shows and waits.
    Args:
        image: image array
        wait: optional how long to wait, default -1
        window_name: optional name of window, default ""

    Returns:
        cv2.waitKey(wait)
    """
    window_name = window_name if not None else ""
    cv2.imshow(window_name, image)
    return cv2.waitKey(wait)


def scale_image(image: np.ndarray, factor=0.5):
    """
    Proportionally scale an image.
    Args:
        image: image array
        factor: scaling factor, default is 0.5 i.e. half the image size.

    Returns:

    """
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def imread(path: Path | str, flags=None) -> np.ndarray:
    """
    Read an image from filesystem.

    Checks that path exists, raises error if it doesn't
    Args:
        path: path to file
        flags: imread flags

    Returns:
        np.ndarray

    Raises:
        FileNotFoundError: if path doesn't exist in filesystem
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"could not find image at {path}")
    image = cv2.imread(str(path), flags)
    return image


def rotate(image: np.ndarray, angle: int, pad_val: int) -> np.ndarray:
    """
    Rotate image by angle degrees ccw and pad with y
    Args:
        image:
        angle: in degrees
        pad_val:

    Returns:
        an image: np.ndarray
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=pad_val, flags=cv2.INTER_NEAREST)


def color_transfer(source, target):
    """
    Transfers color from source to target
    Args:
        source:
        target:

    Returns:

    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype('float32')

    l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src = image_stats(source)
    l_mean_tar, l_std_tar, a_mean_tar, a_std_tar, b_mean_tar, b_std_tar = image_stats(target)

    # subtract means from target image
    l, a, b = cv2.split(target)
    l -= l_mean_tar
    a -= a_mean_tar
    b -= b_mean_tar

    # scale by sd
    l = (l_std_tar / l_std_src) * l
    a = (a_std_tar / a_std_src) * a
    b = (b_std_tar / b_std_src) * b

    # add in the source mean
    l += l_mean_src
    a += a_mean_src
    b += b_mean_src

    # clip between 0, 255
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge channels and convert back to rgb
    image = cv2.merge([l, a, b])
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return image


def brightness_transfer(source, target):
    grey_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype('float32')
    grey_tar = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype('float32')

    mean_src = grey_src.mean()
    std_src = grey_src.std()

    mean_tar = grey_tar.mean()
    std_tar = grey_tar.std()

    grey_tar -= mean_tar
    grey_tar = (std_tar/ std_src) * grey_tar
    grey_tar += mean_src

    grey_tar = np.clip(grey_tar, 0, 255)

    image = cv2.cvtColor(grey_tar.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return image


def image_stats(image):
    """
    Compute Mean and Std of each channel
    Args:
        image:

    Returns:

    """
    c1, c2, c3 = cv2.split(image)

    Stats = namedtuple("stats", "c1_mean c1_std c2_mean c2_std c3_mean c3_std")
    return Stats(c1.mean(), c1.std(), c2.mean(), c2.std(), c3.mean(), c3.std())
