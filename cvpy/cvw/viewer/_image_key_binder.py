from abc import ABC
from typing import Callable, Any

import cv2
import numpy as np

from cvpy import cvw
from cvpy.imseg import BBox


class ImageKeyBinder(ABC):
    """
    Conveniently add key bindings and functions to image viewer.

    Adding key bind and function. To add a function to a key-bind.
    You must create that function with the following signature:

    def func(image: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, Any]

    The function takes in the image and any arguments (passed along from __call__)
    and returns the transformed image along with any data, which will be appended to self.data.

    Examples:

    # "t": terminate is auto added on img_key_binder initialization
    img_key_binder = ImageKeyBinder()

    # to add the ability to rotate, we can define:
    def rotate_ccw(image: np.ndarray, *args, **kwargs) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), []

    # and add it to img_key_binder
    img_key_binder.add_key_bind("a", rotate_ccw)

    # then we can view our image and rotate ccw by pressing "a"
    image = cv2.imread(...)
    image = img_key_binder(image)
    """

    def __init__(self, show_fn=None):
        self._show_fn = None
        self.show_fn = show_fn  # set w/ setter

        self.data = []
        self.data_log = []
        self._key_binds = {}

        # bind [esc] to terminate
        self.add_key_bind(chr(27), self.terminate)

    @property
    def show_fn(self):
        return self._show_fn

    @show_fn.setter
    def show_fn(self, value):
        if value is None:
            self._show_fn = self.default_show_fn
        else:
            if not callable(value):
                raise ValueError('show function must be callable')
            self._show_fn = value

    @staticmethod
    def terminate(image: np.ndarray, *args, **kwargs) -> tuple[None, None]:
        return None, None

    @staticmethod
    def rotate_cw(image: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, Any]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 'cw'

    @staticmethod
    def rotate_ccw(image: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, Any]:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 'ccw'

    @staticmethod
    def mirror(image, *args, **kwargs):
        return np.fliplr(image), 'm'

    def reset_data(self, image, *args, **kwargs):
        self.data.clear()
        return image, []

    def click_event(self, event: int, x: int, y: int, flags: int, param: Any | None):
        """
        Defines what happends on click events

        function signature is defined by cv2
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
            print(f'x: {x}, y: {y}')
            self.data.append((x, y))

    def _show_helper(self, image: np.ndarray, **kwargs) -> tuple[np.ndarray, str]:
        window_name = kwargs.get('window_name', "image_key_binder")
        cv2.namedWindow(window_name)
        kwargs['scale'] = kwargs.get('scale', 1.)

        cv2.setMouseCallback(window_name, self.click_event, param=kwargs)
        image = cvw.scale_image(image, kwargs['scale'])
        return image, window_name

    def default_show_fn(self, image: np.ndarray, **kwargs):
        image, window_name = self._show_helper(image, **kwargs)
        return cvw.show_image(image, 1, window_name=window_name)

    def bounding_box_show_fn(self, image: np.ndarray, **kwargs):
        kwargs['window_name'] = kwargs.get('window_name', 'bbox_key_binder')
        image, window_name = self._show_helper(image, **kwargs)
        scale = kwargs.get('scale')
        # TODO(josh) definitely a better way to do this, but this works for now
        try:
            x, y = self.data[-2]
            xf, yf = self.data[-1]
            if scale is not None:
                def _fn(val: int):
                    return int(val * scale)
                x, y, xf, yf = _fn(x), _fn(y), _fn(xf), _fn(yf)
            bbox = BBox(x, y, xf=xf, yf=yf)
            if len(image.shape) == 3:
                color = (255, 255, 255)
            else:
                color = 255
            img_copy = bbox.draw(image.copy(), color, 2)
            self.data_log.append(bbox)
            return cvw.show_image(img_copy, 1, window_name)
        except Exception as e:
            pass

        return cvw.show_image(image, 1, window_name)

    @staticmethod
    def scaled_show_fn(image: np.ndarray, **kwargs):
        scale = kwargs.get('scale', 0.5)
        window_name = "image_key_binder"
        return cvw.show_image(cvw.scale_image(image, scale), 1, window_name=window_name)

    def add_key_bind(
        self,
        k: str,
        fn: Callable[[np.ndarray, tuple[Any, ...], dict[str, Any]], tuple[np.ndarray, Any]]
    ) -> None:
        """Add key bind."""
        if not isinstance(k, str):
            raise ValueError(f"{k} must be a string")
        if not callable(fn):
            raise ValueError("fn must be callable")

        self._key_binds[k] = fn

    def __call__(self, *images: np.ndarray, **kwargs):
        """
        Show image, applies key bindings
        Args:
            *image: First image is shown key_binds are applied to all images
            **kwargs: kwargs to be passed to show_fn, key bindings, etc.

        Returns:
            image
        """
        last = [im.copy() for im in images]
        self.data.clear()

        while True:
            image = images[0]
            k = self.show_fn(image, **kwargs)
            if k != -1:
                k = chr(k)
                fn = self._key_binds.get(k)
                if fn is not None:
                    images, data = zip(*[fn(im, **kwargs) for im in last])
                    self.data.append(data)

                    if images[0] is None:  # indicates to stop loop
                        if len(last) == 1:
                            return last[0]
                        elif len(last) > 1:
                            return tuple(last)
                        else:
                            raise ValueError('idk')
                    elif isinstance(image, np.ndarray):
                        last = [im.copy() for im in images]
                    else:
                        raise ValueError("fn must return an np.ndarray")
