from typing import Tuple, Any
from abc import abstractmethod, ABC

import cv2
import numpy as np

from cvpy.cvw import scale_image
from cvpy.imseg import BBox


class CoordinateClicker(ABC):
    """
    Abstract base class for doing mouse based manipulations. Left clicks are stored to self.coords,
    coords can then be used in key_binds to draw bboxes, crop images, etc.

    to use, subclass ClickCropper, implement key_binds and define events
    """

    def __init__(self):
        self.coords = []  # (x, y)

    def __call__(self, image: np.ndarray, scale=0.5, *args, **kwargs) -> Any:
        """
        Shows image, logs left clicks to self.coords, on keypress calls key_binds.
        Args:
            image: image
            scale: factor to scale down image when showing image, does not affect coords
            *args: passed through to key_binds
            **kwargs: passed through to key_binds

        Returns:
            Returns whatever is returned from key_binds
        """
        param = {'scale': 0.5}
        cv2.namedWindow("interactive")
        cv2.setMouseCallback('interactive', self.click_event, param=param)

        while True:
            cv2.imshow("interactive", scale_image(image, scale))
            k = cv2.waitKey(1)
            (done, result) = self.key_binds(image=image, k=k, *args, **kwargs)
            if done:
                return result

    def click_event(self, event, x, y, flags, param: dict):
        """
        Defines what happends on click events
        Args:
            event: cv2 event
            x: x position
            y: y position
            flags: nothing, used to keep same signature as
            param: dictionary of params

        Returns:
        """
        scale = param['scale']
        if event == cv2.EVENT_LBUTTONDOWN:
            # unscale points
            x = int((1 / scale) * x)
            y = int((1 / scale) * y)
            print(f'{x}, {y}')
            self.coords.append((x, y))

    @abstractmethod
    def key_binds(self, image: np.ndarray, k: int, *args, **kwargs) -> Tuple[bool, Any]:
        """
        Key binding logic.
        Args:
            images: image
            k: keyboard input code

        Returns:
            False if not done, Anything else will terminate program and return result
        """
        raise NotImplementedError


class BoundingBoxClicker(CoordinateClicker):
    """
    Pulls up image, specify coords of bbox (top left, bottom right), returns BBox
    """

    def key_binds(self, image: np.ndarray, k: int, *args, **kwargs) -> Tuple[bool, Any]:
        if k == ord('s'):
            xi, yi = self.coords[-2]
            xf, yf = self.coords[-1]
            return True, BBox(x=xi, y=yi, xf=xf, yf=yf)
        else:
            return False, None
