import ast
from typing import List, Dict

import numpy as np
import cv2
import pandas as pd


class Contour:
    """ cv2 returns these as [n, 1, 2], where n is number of points and 2 is (x, y) """

    def __init__(self, contour: np.ndarray):
        assert type(contour) is np.ndarray, f"Contour must be a numpy array, but a {type(contour)} was passed"
        self.contour = contour
        self._moment = None
        self._area = None
        self._perimeter = None
        self._bbox = None
        self._cx = None
        self._cy = None
        self._roundness = None
        self._m00 = None
        self._m10 = None
        self._m01 = None
        self._max = np.max(contour, axis=0)
        self._min = np.min(contour, axis=0)
        mn = self._min[0]
        mx = self._max[0]
        self.x = mn[0]
        self.y = mn[1]
        self.xf = mx[0]
        self.yf = mx[1]

    def __str__(self):
        arr = self.to_array()
        xs = arr[0, :]
        ys = arr[1, :]
        return str({"xs": xs.tolist(), 'ys': ys.tolist()})

    def __repr__(self):
        return str(self)

    def draw(self, image, color, thickness=1):
        return cv2.drawContours(image, [self.contour], 0, color, thickness)

    def add_weighted(self, src1, alpha, src2, beta, gamma=0):
        """
        Args:
            src1:
            alpha:
            src2: Crops contour from this image and overlays onto src1
            beta:
            gamma:

        Returns:
        """
        src2 = self.extract(src2)
        return cv2.addWeighted(src1, alpha, src2, beta, gamma)

    def _min_black_box(self, other):
        xf = max(self.bbox.xf, other.bbox.xf)
        yf = max(self.bbox.yf, other.bbox.yf)
        temp_image = np.zeros((yf, xf))
        return temp_image

    def intersection(self, other) -> np.ndarray:
        temp_image = self._min_black_box(other)
        this_contour_img = self.draw(temp_image.copy(), 1, -1)
        other_contour_img = other.draw(temp_image, 1, -1)
        mask = cv2.bitwise_and(this_contour_img, other_contour_img)
        return mask

    def union(self, other):
        temp_image = self._min_black_box(other)
        this_contour_img = self.draw(temp_image.copy(), 1, -1)
        other_contour_img = other.draw(temp_image, 1, -1)
        mask = cv2.bitwise_or(this_contour_img, other_contour_img)
        return mask

    def iou(self, other):
        """ IOU = intersection / union """

        intersection_mask = self.intersection(other)
        union_mask = self.union(other)

        return np.sum(intersection_mask) / np.sum(union_mask)

    @staticmethod
    def recall(true_contour, pred_contour):
        "recall = intersection / true_contour.area"
        intersection_mask = true_contour.intersection(pred_contour)
        return np.sum(intersection_mask) / true_contour.area

    @staticmethod
    def precision(true_contour, pred_contour):
        "precision = intersection / pred_contour.area"
        intersection_mask = true_contour.intersection(pred_contour)
        return np.sum(intersection_mask) / pred_contour.area

    def tolist(self) -> List[List[int]]:
        return self.contour.tolist()

    def extract(self, image):
        """ turns image all black except for inside contour """
        temp_image = np.zeros_like(image)
        temp_image = self.draw(temp_image, 1, -1)
        temp_image = np.where(temp_image == 1, image, 0)
        return temp_image

    def crop(self, image):
        return self.bbox.crop(self.extract(image))

    def intensity(self, image):
        contour_img = self.crop(image)
        inten = np.sum(contour_img) / np.sum(np.where(contour_img > 0, 1, 0))
        return inten

    def standardize(self):
        contour = self.contour - self._min
        return Contour(contour)

    def _contours(self):
        return Contours([self])

    def to_json(self) -> List[List[int]]:
        """ Return json serializable version of object that can be read in using Contour.from_json() """
        ar = self.contour[:, 0, :]
        return ar.tolist()

    @staticmethod
    def from_json(l: List[List[int]]):
        """ Returns a Contour instance [[x1, y1], [x2, y2], . . .]"""
        ar = np.array(l)[:, None, :]
        return Contour(ar)

    def to_array(self) -> np.ndarray:
        """
        contour to numpy array (2, N)
        Returns:
            (2, N): [
                [x1, x2, ..., xN],
                [y1, y2, ..., yN]
            ]
        """
        """ Returns a numpy array version of the contour in format (N, 2)"""
        return self.contour[:, 0, :].transpose((1, 0))

    @staticmethod
    def from_array(arr: np.ndarray):
        """ Takes array in format (2, N) """
        return Contour(arr.transpose((1, 0))[:, None, :])

    @property
    def m00(self):
        if self._m00 is None:
            self._m00 = self.moment['m00']
        return self._m00

    @property
    def m10(self):
        if self._m10 is None:
            self._m10 = self.moment['m10']
        return self._m10

    @property
    def m01(self):
        if self._m01 is None:
            self._m01 = self.moment['m01']
        return self._m01

    @property
    def roundness(self):
        if self.perimeter == 0:
            self._roundness = 0
        elif self._roundness is None:
            self._roundness = (4 * np.pi * self.area) / (self.perimeter ** 2)
        return self._roundness

    @property
    def moment(self):
        if self._moment is None:
            self._moment = cv2.moments(self.contour)
        return self._moment

    @property
    def cx(self):
        if self._cx is None:
            self._cx = self.m10 / self.m00
        return self._cx

    @property
    def cy(self):
        if self._cy is None:
            self._cy = self.m01 / self.m00
        return self._cy

    @property
    def area(self):
        if self._area is None:
            temp_image = np.zeros((self.bbox.yf + 1, self.bbox.xf + 1))
            self._area = np.sum(self.draw(temp_image, 1, -1))
        return self._area

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = cv2.arcLength(self.contour, True)
        return self._perimeter

    @property
    def bbox(self):
        from joshpy.images.bbox_utils import BBox
        if self._bbox is None:
            self._bbox = BBox(*cv2.boundingRect(self.contour))
        return self._bbox

    @staticmethod
    def get_contour_from_str(contour):
        return Contour(np.array(ast.literal_eval(contour)))


class Contours(list):
    def __init__(self, contours: List[Contour]):
        if len(contours) > 0:
            assert type(
                contours[0]) is Contour, f'Items in array must be type Contour, but {type(contours[0])} was passed'
        super().__init__(contours)

        self._area = None
        self._bboxes = None

    @property
    def bboxes(self):
        from joshpy.images.bbox_utils import BBoxes
        if self._bboxes is None:
            self._bboxes = [contour.bbox for contour in self]
        return BBoxes(self._bboxes)

    @property
    def area(self):
        if self._area is None:
            areas = [contour.area for contour in self]
            self._area = sum(areas)
        return self._area

    def draw(self, image, color, thickness=1):
        for contour in self:
            image = contour.draw(image, color, thickness)
        return image

    def draw_weighted(self, image, alpha, beta, color, thickness, gamma=0):
        temp_image = self._min_black_box()
        temp_image = self.draw(temp_image, color, thickness)
        out_image = cv2.addWeighted(image, alpha, temp_image, beta, gamma)
        return out_image

    def _min_black_box(self, other=None):
        xf = max([contour.bbox.xf for contour in self])
        yf = max([contour.bbox.yf for contour in self])
        if other.__class__ == Contours:
            xf = max(xf, *[contour.bbox.xf for contour in other])
            yf = max(yf, *[contour.bbox.yf for contour in other])
        elif other.__class__ == Contour:
            xf = max(xf, other.bbox.xf)
            yf = max(yf, other.bbox.yf)
        elif other is None:
            pass
        else:
            raise "Unsupported Class"

        return np.zeros((yf, xf), dtype=np.uint8)

    def intersection(self, other):
        temp_img = self._min_black_box(other)
        this_mask = self.draw(temp_img.copy(), 1, -1)
        other_mask = other.draw(temp_img, 1, -1)
        mask = cv2.bitwise_and(this_mask, other_mask)
        return mask

    def union(self, other):
        temp_img = self._min_black_box(other)
        mask = self.draw(temp_img, 1, -1)
        mask = other.draw(mask, 1, -1)
        return mask

    def iou(self, other):
        intersect = self.intersection(other)
        union_ = self.union(other)
        return np.sum(intersect) / np.sum(union_)

    def _contours(self):
        return self

    def to_json(self) -> List[List[List[int]]]:
        """ List of contours, each contour has a list of [x, y] """
        return [contour.to_json() for contour in self]

    @staticmethod
    def from_json(l: List[List[List[int]]]):
        l = [Contour.from_json(x) for x in l]
        return Contours(l)

    @staticmethod
    def recall(true_contours, pred_contours):
        "recall = intersection / true_contour.area"
        true_contours = true_contours._contours()
        pred_contours = pred_contours._contours()

        intersection_mask = true_contours.intersection(pred_contours)
        return np.sum(intersection_mask) / true_contours.area

    @staticmethod
    def precision(true_contours, pred_contours):
        "precision = intersection / pred_contour.area"
        true_contours = true_contours._contours()
        pred_contours = pred_contours._contours()

        intersection_mask = true_contours.intersection(pred_contours)
        return np.sum(intersection_mask) / pred_contours.area

    @staticmethod
    def get_contours_from_mask(mask):
        contours_out = []
        # TODO(joshfisher): allow option for cv2.CHAIN_APPROX method
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            contours_out.append(Contour(contour))

        return Contours(contours_out)

    @staticmethod
    def contours_from_series(s: pd.Series):
        contours = []
        for contour in s:
            contour = Contour.get_contour_from_str(contour)
            contours.append(contour)
        return Contours(contours)
