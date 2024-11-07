"""cv2 wrapper."""

from pathlib import Path

import cv2
import numpy as np


def show_image(image: np.ndarray, wait: int = -1, window_name: str | None = None) -> int:
    """
    Show image.

    wraps cv2.imshow, shows and waits.

    Args:
    ----
        image: image array
        wait: optional how long to wait, default -1
        window_name: optional name of window, default ""

    Returns:
    -------
        cv2.waitKey(wait)

    """
    window_name = window_name if not None else ""
    cv2.imshow(window_name, image)
    return cv2.waitKey(wait)


def scale_image(image: np.ndarray, factor=0.5):
    """
    Proportionally scale an image.

    Args:
    ----
        image: image array
        factor: scaling factor, default is 0.5 i.e. half the image size.

    Returns:
    -------
        resized image

    """
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def imread(path: Path | str, flags: int | None = None) -> np.ndarray:
    """
    Read an image from filesystem.

    Checks that path exists, raises error if it doesn't
    Args:
        path: path to file
        flags: imread flags, default is to read image unchanged

    Returns
    -------
        np.ndarray

    Raises
    ------
        FileNotFoundError: if path doesn't exist in filesystem

    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"could not find image at {path}")

    if flags is None:
        flags = cv2.IMREAD_UNCHANGED

    return cv2.imread(str(path), flags)


def rotate(image: np.ndarray, angle: int, pad_val: int) -> np.ndarray:
    """
    Rotate image by angle (degrees), and pad with pad_val is necessary.

    Args:
    ----
        image: Either grayscale or color image
        angle: degrees to rotate image, positive is ccw and negative is cw
        pad_val: value to pad edges

    Returns:
    -------
        an image: np.ndarray

    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (c_x, c_y) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rot_m = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
    cos = np.abs(rot_m[0, 0])
    sin = np.abs(rot_m[0, 1])

    # compute the new bounding dimensions of the image
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    rot_m[0, 2] += (n_w / 2) - c_x
    rot_m[1, 2] += (n_h / 2) - c_y

    # perform the actual rotation and return the image
    return cv2.warpAffine(
        image,
        rot_m,
        (n_w, n_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=pad_val,
        flags=cv2.INTER_NEAREST
    )
