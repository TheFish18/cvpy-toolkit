import json
import math
import warnings
from typing import List, Dict, Tuple

import cv2
import numpy as np

from cvpy.imseg._utils import get_2d_rotation_matrix
from cvpy.imseg import Contours


class BBox:
    """
    Bounding Boxes

    Must pass bounding box top left (x, y) coordinates and well as
    bounding box width, height (w, h) or bottom right coordinates (xf, yf)
    """
    def __init__(
            self,
            x: int,
            y: int,
            w: int | None = None,
            h: int | None = None,
            xf: int | None = None,
            yf: int | None = None
    ):
        """
        BBox container class.
        Args:
            x: bounding box top left x coordinate, required
            y: bounding box top left y coordinate, required
            w: bounding box width
            h: bounding box height
            xf: bounding box bottom right x coordinate
            yf: bounding box bottom right y coordinate
        """
        def check_ints(**kwargs):
            for key, val in kwargs.items():
                if not isinstance(val, int):
                    raise ValueError(f"{key} must be an int, got type: {type(val)}")

        check_ints(x=x, y=y)

        self.x = x
        self.y = y

        if (w is not None and h is not None) and (xf is None and yf is None):
            check_ints(w=w, h=h)
            self.w = w
            self.h = h
            self.xf = self.x + self.w
            self.yf = self.y + self.h
        elif (xf is not None and yf is not None) and (w is None and h is None):
            check_ints(xf=xf, yf=yf)
            self.xf = xf
            self.yf = yf
            self.w = self.xf - self.x
            self.h = self.yf - self.y
        else:
            raise ValueError(
                "Both height and width must be specified or both xf and yf must be specified, "
                "but passing some combination of both is un-allowed as the dimensions can be ambiguous"
            )

        self.area = self.w * self.h

    @property
    def centroid(self) -> np.ndarray:
        """
        Get centroid of bounding box.

        Returns:
            np.ndarray w/ shape (2, 1) where 2 corresponds to (xc, yc)
        """
        return np.array([
            [(self.x + self.xf) / 2], [(self.y + self.yf) / 2]
        ])

    def __repr__(self):
        keep_keys = ['x', 'xf', 'w', 'y', 'yf', 'h']
        d = {k: self.__dict__[k] for k in keep_keys}
        return str(d)

    def __str__(self):
        return repr(self)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        if not self.overlaps_with(other):
            return False
        overlap = self.intersection(other)
        if overlap == self:
            return True
        else:
            return False

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        if not self.overlaps_with(other):
            return False
        overlap = self.intersection(other)
        if overlap == other:
            return True
        else:
            return False

    def __gt__(self, other):
        return self >= other and self != other

    def intensity(self, image):
        contour_img = self.crop(image)
        inten = np.sum(contour_img) / np.sum(np.where(contour_img > 0, 1, 0))
        return inten

    def prop_in(self, other):
        """Answers: What proportion of self is contained in other"""
        if not self.overlaps_with(other):
            return 0.0
        overlap = self.intersection(other)
        return overlap.a / self.a

    def draw(self, image, color, border):
        # idk why, cv2 draws self.xf + 1 col of pixels and self.yf + 1 row of pxs
        return cv2.rectangle(image, (self.x, self.y), (self.xf - 1, self.yf - 1), color, border)

    def crop(self, image):
        """
        Crops according to bbox
        Args:
            image: Image to crop

        Returns:
            cropped image
        """
        return image[self.y: self.yf, self.x: self.xf]

    def union(self, other):
        assert self.overlaps_with(other), "BBoxes don't overlap"

        xi = min(self.x, other.x)
        yi = min(self.y, other.y)

        xf = max(self.xf, other.xf)
        yf = max(self.yf, other.yf)
        return BBox(x=xi, y=yi, xf=xf, yf=yf)

    def intersection(self, other):
        assert self.overlaps_with(other), "BBoxes don't overlap"

        xi = max(self.x, other.x)
        yi = max(self.y, other.y)

        xf = min(self.xf, other.xf)
        yf = min(self.yf, other.yf)
        return BBox(x=xi, y=yi, xf=xf, yf=yf)

    def iou(self, other) -> float:
        if not self.overlaps_with(other):
            return 0.0

        intersect = self.intersection(other)
        return intersect.a / (self.a + other.a - intersect.a)

    def overlaps_with(self, other) -> bool:
        xi = min(self.x, other.x)
        yi = min(self.y, other.y)

        xf = max(self.xf, other.xf)
        yf = max(self.yf, other.yf)

        x_overlap = (self.w + other.w) - (xf - xi)
        y_overlap = (self.h + other.h) - (yf - yi)

        return x_overlap > 0 and y_overlap > 0

    def _able_shift(self, boundary):
        x_able_neg_shift = self.x - boundary.x
        x_able_pos_shift = boundary.xf - self.xf

        y_able_neg_shift = self.y - boundary.y
        y_able_pos_shift = boundary.yf - self.yf

        return x_able_neg_shift, x_able_pos_shift, y_able_neg_shift, y_able_pos_shift

    def reshape(self, w, h, boundary):
        """ Expand current BBox to w and h in a boundary while keeping view centered """
        assert w <= boundary.w, "reshape w > boundary.w"
        assert h <= boundary.h, "reshape h > bounary.h"

        (x_able_neg_shift, x_able_pos_shift,
         y_able_neg_shift, y_able_pos_shift) = self._able_shift(boundary)

        # the amount to be expanded is split between left and right
        # if left and right are able to shift enough: subtract split for xi and add to xf
        # if left side cant accommodate shift, shift as far as possible and add rest to xf
        x = (w - self.w) / 2
        if x_able_neg_shift < x:
            xi = self.x - x_able_neg_shift
            temp_x = w - self.w - x_able_neg_shift
            xf = self.xf + temp_x
        elif x_able_pos_shift < x:
            xf = self.xf + x_able_pos_shift
            temp_x = w - self.w - x_able_pos_shift
            xi = self.x - temp_x
        else:
            xi = round(self.x - x)
            xf = round(self.xf + x)

        y = (h - self.h) / 2
        if y_able_neg_shift < y:
            yi = self.y - y_able_neg_shift
            temp_y = h - self.h - y_able_neg_shift
            yf = self.yf + temp_y
        elif y_able_pos_shift < y:
            yf = self.yf + y_able_pos_shift
            temp_y = h - self.h - y_able_pos_shift
            yi = self.y - temp_y
        else:
            yi = round(self.y - y)
            yf = round(self.yf + y)

        out_bbox = BBox(x=xi, y=yi, xf=xf, yf=yf)
        assert out_bbox.w == w and out_bbox.h == h, 'out_bbox dims arent correct. ' \
                                                    f'\nout_bbox: {repr(out_bbox)}' \
                                                    f'\nself: {repr(self)}' \
                                                    f'\nboundary: {repr(boundary)}' \
                                                    f'\nw, h: {w, h}'

        return BBox(x=xi, y=yi, xf=xf, yf=yf)

    def random_shift(self, boundary, max_shift: int):
        """
        Randomly shift a bbox while keeping it constrained within the image and same size
        Args:
            boundary: BBox containing boundary points
            max_shift: max shift to keep object in frame

        Returns:
            BBox of new coords
        """
        assert self <= boundary, "self bbox must be fully contained within the boundary coordinates." \
                                 f"\nself: {repr(self)}" \
                                 f"\nboundary: {repr(boundary)}"

        x_able_neg_shift = min(max_shift, self.x - boundary.x)
        x_able_pos_shift = min(max_shift, boundary.xf - self.xf)

        y_able_neg_shift = min(max_shift, self.y - boundary.y)
        y_able_pos_shift = min(max_shift, boundary.yf - self.yf)

        x_shift = np.random.randint(-x_able_neg_shift, x_able_pos_shift)
        y_shift = np.random.randint(-y_able_neg_shift, y_able_pos_shift)

        xi = self.x + x_shift
        xf = self.xf + x_shift
        yi = self.y + y_shift
        yf = self.yf + y_shift

        return BBox(xi, yi, xf=xf, yf=yf)

    @staticmethod
    def get_weldline_bbox(mask, weldline_px):
        mask = np.where(mask == weldline_px, 1, 0).astype(np.uint8)
        mask_contours = Contours.get_contours_from_mask(mask)
        mask_bboxes = mask_contours.bboxes
        return mask_bboxes.max()

    def write(self, p):
        ks = ['x', 'y', 'xf', 'yf']
        out_d = {k: self.__dict__[k] for k in ks}
        with open(p, 'w') as f:
            json.dump(out_d, f)

    @staticmethod
    def read(p, x='x', y='y', xf='xf', yf='yf'):
        with open(p) as f:
            d = json.load(f)

        return BBox(x=d[x], y=d[y], xf=d[xf], yf=d[yf])

    def get_windows(self, kernel_size: int, stride: int):
        """

        Args:
            bbox: Bounds to work in
            kernel_size: Size of window
            stride: step from window_n to window_{n+1}

        Returns:

        """

        if self.h < kernel_size or self.w < kernel_size:
            raise ValueError("Height and Width of bounds (bbox) must be >= to kernel size")
        h_exact = self.h % kernel_size == 0
        w_exact = self.w % kernel_size == 0

        bboxes = []

        for y in range(self.y, self.yf - kernel_size + 1, stride):
            for x in range(self.x, self.xf - kernel_size + 1, stride):
                bboxes.append(
                    BBox(x=x, y=y, w=kernel_size, h=kernel_size)
                )
        if not h_exact:  # need to get bottom
            # gets the bottom edge
            for x in range(self.x, self.xf - kernel_size + 1, stride):
                bboxes.append(
                    BBox(x=x, y=self.yf - kernel_size, w=kernel_size, h=kernel_size)
                )
        if not w_exact:
            # gets right edge
            for y in range(self.y, self.yf - kernel_size + 1, stride):
                bboxes.append(
                    BBox(x=self.xf - kernel_size, y=y, w=kernel_size, h=kernel_size)
                )

        # append bottom right bounding box
        bboxes.append(
            BBox(x=self.xf - kernel_size, y=self.yf - kernel_size, w=kernel_size, h=kernel_size)
        )

        return BBoxes(bboxes)

    def get_jitter_windows(self, centroids, window_size, jitter_xy, n_samples) -> 'BBoxes':
        """
        Get jittered windows w/in current bbox scope.

        For each centroid in centroids, draw a bbox with shape (window_size, window_size)
        such that center of bbox == centroid, then shift bbox between (-jitter_xy, jitter_xy)
        in the x and y dimensions.

        Args:
            centroids: np.ndarray[n, 2], n is number of centroids and 2 is (x, y)
            window_size: int, size of output windows
            jitter_xy: int, amount to randomly (jitter ~ U(-jitter_xy, jitter_xy)) jitter window in x, y direction.
            n_samples: int, n of boxes to take for each centroid

        Returns:
            BBoxes
        """

        if isinstance(centroids, np.ndarray):
            if len(centroids.shape) != 2 or centroids.shape[1] != 2:
                raise ValueError(
                    "if centroids is an np.ndarray then it must have shape "
                    "(N, 2) where N is number of centroids and 2 is (x, y)"
                )
        else:
            raise ValueError(
                "centroid must be list[tuple[int, int]] | np.ndarray"
            )

        if self.h < window_size or self.w < window_size:
            raise ValueError("Height and Width of bounds (bbox) must be >= to kernel size")

        canvas_xy = np.array([self.x, self.y])
        canvas_xfyf = np.array([self.xf, self.yf])

        if np.any(centroids < canvas_xy) or np.any(centroids > canvas_xfyf):
            raise ValueError("Some centroids are outside the working bbox")

        bbox_xy = centroids - (window_size // 2)
        bbox_xy = bbox_xy.repeat(n_samples, axis=0)

        n = bbox_xy.shape[0]
        bbox_xy = bbox_xy + np.random.randint(-jitter_xy, jitter_xy, (n, 1))
        # fix min oob
        bbox_xy = bbox_xy + np.clip(canvas_xy - bbox_xy, 0, np.inf)
        bbox_xfyf = bbox_xy + window_size
        # fix max oob
        bbox_xfyf = bbox_xfyf - np.clip(bbox_xfyf - canvas_xfyf, 0, np.inf)
        bbox_xy = bbox_xfyf - window_size

        if np.any(bbox_xy < canvas_xy):
            raise ValueError("unable to reconcile out of bounds bboxes due to jittering process ")

        windows = []
        for i in range(n):
            x, y = bbox_xy[i]
            xf, yf = bbox_xfyf[i]

            windows.append(BBox(x=int(x), y=int(y), xf=int(xf), yf=int(yf)))

        return BBoxes(windows)

    def to_json(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "xf": self.xf, "yf": self.yf}

    @staticmethod
    def from_json(d: Dict[str, int]):
        try:
            x, y, xf, yf = d["x"], d["y"], d["xf"], d["yf"]
        except KeyError:
            raise KeyError("d must contain keys: x, y, xf, yf")

        return BBox(x, y, xf=xf, yf=yf)

    def to_array(self):
        """
        Gets array view of bbox
        Returns:

        """
        arr = np.array([
            [self.x, self.xf],
            [self.y, self.yf]
        ])
        return arr

    @staticmethod
    def from_array(arr):
        """
        Creates a bbox from a matrix of column vector data points
        Args:
            array: (2, 2) [
                [xi, xf],
                [yi, yf]
            ]

        Returns:
            BBox from array
        """
        x, xf = arr[0, 0], arr[0, 1]
        y, yf = arr[1, 0], arr[1, 1]
        return BBox(x=x, y=y, xf=xf, yf=yf)

    def to_rotated_bbox(self, degrees: float = None, radians: float = None, about: Tuple[float, float] = None):
        return RotatedBBox(self, degrees=degrees, radians=radians, about=about)


class BBoxes(list):
    def __init__(self, bboxes: List[BBox]):
        super().__init__(bboxes)
        self.bboxes = bboxes

    def max(self):
        """ Returns BBox with max area """
        a = 0
        mx = 0
        for i, bbox in enumerate(self.bboxes):
            temp_a = bbox.area
            if temp_a > a:
                a = temp_a
                mx = i

        return self.bboxes[mx]

    def min(self):
        """ Returns BBox with min area """
        a = np.inf
        mn = 0
        for i, bbox in enumerate(self.bboxes):
            temp_a = bbox.area
            if temp_a < a:
                a = temp_a
                mn = i

        return self.bboxes[mn]

    def draw(self, image: np.ndarray, color, border=-1):
        """
        Draws all BBox in BBoxes on image
        Args:
            image: image array
            color: <tuple[int] or int> should correspond to channel dim in image
            border: -1: fill; >0: line weight

        Returns:
            image with bboxes drawn on
        """
        for bbox in self.bboxes:
            image = bbox.draw(image, color, border)
        return image

    def reconstruct_image(self, out_shape, img_windows: np.ndarray) -> np.ndarray:
        """
        Reconstruct an image from sequence of bboxes and img_windows
        Args:
            out_shape: shape of output image
            img_windows: (N, H, W) or (N, H, W, C), N must equal len(bboxes)

        Returns:
            np.ndarray
        """
        out_image = np.zeros(out_shape)
        for i, bbox in enumerate(self):
            window = img_windows[i]
            x, y, xf, yf = bbox.x, bbox.y, bbox.xf, bbox.yf
            out_image[y: yf, x: xf] = np.maximum(out_image[y: yf, x: xf], window)

        return out_image

    def to_json(self) -> List[Dict[str, int]]:
        """ Returns json serializable version of object """
        return [bbox.to_json() for bbox in self.bboxes]

    @staticmethod
    def from_json(l: List[Dict[str, int]]):
        """
        Can recreate object from json serializable

        Args:
            l: [{"x": x, "y": y, "xf": xf, "yf": yf}, . . . ]

        Returns:

        """
        bboxes = [BBox.from_json(d) for d in l]
        return BBoxes(bboxes)

    def to_array(self) -> np.ndarray:
        """
        returns bboxes as a numpy array
        Returns:
            (N, 2, 2): where each (2, 2) is [[xi, xf], [yi, yf]]
        """
        arr = np.concatenate([bbox.to_array() for bbox in self], axis=0)
        return arr.reshape((-1, 2, 2))

    @staticmethod
    def from_array(arr: np.ndarray):
        """
        Takes an array of bboxes and converts to BBoxes
        Args:
            arr: (N, 2, 2): where each (2, 2) is [[xi, xf], [yi, yf]]

        Returns:
            BBoxes
        """
        bboxes = []
        for i in range(arr.shape[0]):
            bboxes.append(BBox.from_array(arr[i, :, :]))

        return BBoxes(bboxes)

    @staticmethod
    def get_bboxes_from_contours(contours: Contours):
        """ Gets bboxes from Contours """
        return BBoxes([contour.bbox for contour in contours])

    @staticmethod
    def pad_for_exact_windows(image, px_val, kernel_size=256, xi=0, yi=0, xf=None, yf=None, split: float = 1.0):
        """
        pads images for exact windows, extends image frame to w % kernel_size = 0 if possible else padds with px_val
        Args:
            image:
            px_val:
            kernel_size:
            stride:
            xi:
            yi:
            xf:
            yf:
            split:
                if 1: only pad right and bottom,
                if 0.5 padding will be split equally between left/right top/bottom

        Returns:

        """
        xf = image.shape[1] if xf is None else xf
        yf = image.shape[0] if yf is None else yf

        w = xf - xi
        h = yf - yi

        pad_w = (kernel_size * math.ceil(w / kernel_size)) - w + 1
        pad_h = (kernel_size * math.ceil(h / kernel_size)) - h + 1

        pad_right = int(pad_w * split)
        pad_left = pad_w - pad_right

        pad_bottom = int(pad_h * split)
        pad_top = pad_h - pad_bottom

        return cv2.copyMakeBorder(image, bottom=pad_bottom, right=pad_right, top=pad_top, left=pad_left,
                                  borderType=cv2.BORDER_CONSTANT, value=px_val)



class RotatedBBox:
    def __init__(self, bbox: BBox, degrees=None, radians=None, about=None):
        """
        Rotated Bounding Boxes, + is cw, - is ccw
        Args:
            bbox: BBox
            rotation: rotation in degrees
            about: rotate about which point, default is centroid
        """
        self.bbox = bbox

        if degrees is None and radians is None:
            raise TypeError('RotatedBBox.__init__() missing 1 required positional argument: "degrees" OR "radians"')
        elif degrees is not None and radians is not None:
            raise TypeError("RotatedBBox take either 'degrees' or 'radians' not both")
        elif degrees is not None:
            self.degrees = degrees
        elif radians is not None:
            self.radians = radians

        if about is not None:
            self.about = about
        else:
            self._about = self.bbox.centroid

    def __str__(self):
        return f"bbox: {self.bbox}, degrees: {self.degrees}, radians: {self.radians}, about: {tuple(self.about[:, 0])}"

    @property
    def radians(self):
        return self._radians

    @radians.setter
    def radians(self, radians: float):
        self._radians = radians
        self._degrees = np.rad2deg(radians)

    @property
    def degrees(self):
        return self._degrees

    @degrees.setter
    def degrees(self, degrees: float):
        self._degrees = degrees
        self._radians = np.deg2rad(degrees)

    @property
    def centroid(self):
        arr = self.to_array()
        return np.mean(arr, axis=1, keepdims=True)

    @property
    def rotation_matrix(self):
        return get_2d_rotation_matrix(self.radians)

    @property
    def about(self):
        return self._about

    @about.setter
    def about(self, about: Tuple[float, float]):
        x, y = about
        self._about = np.array([[x], [y]])

    def draw(self, image: np.ndarray, color: Tuple[int, int, int] or int, thickness: int):
        ar = self.to_array().round().astype(int)
        xy, x2y, x2y2, xy2 = tuple(ar[:, 0]), tuple(ar[:, 1]), tuple(ar[:, 2]), tuple(ar[:, 3])

        if thickness > 0:
            image = cv2.line(image, pt1=xy, pt2=x2y, color=color, thickness=thickness)
            image = cv2.line(image, pt1=x2y, pt2=x2y2, color=color, thickness=thickness)
            image = cv2.line(image, pt1=x2y2, pt2=xy2, color=color, thickness=thickness)
            image = cv2.line(image, pt1=xy2, pt2=xy, color=color, thickness=thickness)
        else:
            end_pts = np.array([xy, x2y, x2y2, xy2])
            image = cv2.fillPoly(image, pts=[end_pts], color=color)

        return image

    def to_json(self):
        ar = self.to_array()
        xy, x2y, x2y2, xy2 = ar[:, 0], ar[:, 1], ar[:, 2], ar[:, 3]

        return {"xy": xy.tolist(), "x2y": x2y.tolist(), "x2y2": x2y2.tolist(), "xy2": xy2.tolist()}

    @staticmethod
    def from_json(d: dict, about: Tuple[float, float] = None):
        """ expects {"xy": [x, y], "x2y": [x2, y], "x2y2": [xy2], "xy2": [x2y2]} """
        xy, x2y, x2y2, xy2 = d["xy"], d["x2y"], d["x2y2"], d["xy2"]
        arr = np.array([
            xy, x2y, x2y2, xy2
        ])
        return RotatedBBox.from_array(arr.T, about)

    def to_array(self):
        """
        Outputs RotatedBBox as numpy array
        Returns:
            numpy array in format:
            [
                [x, x2, x2, x],
                [y, y, y2, y2]
            ]
        """
        # xy, x2y, x2y2, xy2
        arr = np.array([
            [self.bbox.x, self.bbox.xf, self.bbox.xf, self.bbox.x],
            [self.bbox.y, self.bbox.y, self.bbox.yf, self.bbox.yf]
        ])
        rot_mat = get_2d_rotation_matrix(self.radians)

        return (rot_mat @ (arr - self.about)) + self.about

    @staticmethod
    def from_array(arr: np.ndarray, about: Tuple[float, float] = None):
        """
        Creates a RotatedBBox from array
        Args:
            arr: [
                [x, x2, x2, x],
                [y, y, y2, y2]
            ]

        Returns:
            RotatedBBox from array
        """
        if about is None:
            about = np.mean(arr, axis=1, keepdims=True)
        else:
            x, y = about
            about = np.array([[x], [y]])

        # get angle
        d = arr[:, 2] - arr[:, 3]
        dx, dy = d[0], d[1]
        rads = np.arctan(dy/dx)
        # negative, b/c we need to rotate in opposite direction
        R = get_2d_rotation_matrix(-rads)
        straight_bbox = R@(arr - about) + about

        x_min, y_min = np.min(straight_bbox, axis=1).tolist()
        x_max, y_max = np.max(straight_bbox, axis=1).tolist()

        x, y, xf, yf = round(x_min), round(y_min), round(x_max), round(y_max)
        if x != x_min or y != y_min or xf != x_max or yf != y_max:
            warnings.warn("unable to perfectly straighten bbox, using rounded values")

        bbox = BBox(x=x, y=y, xf=xf, yf=yf)
        rot_bbox = RotatedBBox(bbox, radians=rads)
        rot_bbox._about = about
        return rot_bbox
