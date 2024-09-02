import os
import glob
from typing import List, Iterator

import cv2
import numpy as np

import joshpy.images.cv2_wrapper as cvw


class ImPath(str):
    """
    Gracefully handle image paths

    Attributes:
        - path: image path, e.g. Users/TheFish18/Projects/SomeDir/test_image.png
        - directory: image's directory, e.g. Users/TheFish18/Projects/SomeDir
        - root_directory: image's directory's directory, e.g. Users/TheFish18/Projects
        - directory_name: name of directory, e.g. SomeDir
        - filename: test_image.png
        - image_name: test_image
        - file_ext: .png
    """

    def __init__(self, path):
        """
        Convenient access image path attributes or image
        Args:
            path: image file path
        """
        self.path = path
        self.directory = os.path.dirname(path)
        self.root_directory, self.subdir = os.path.split(self.directory)
        self.filename = os.path.basename(path)
        self.image_name, self.file_ext = os.path.splitext(self.filename)

    def __str__(self):
        return self.path

    def imread(self, cv2_flag: int, dirname=None) -> np.ndarray:
        """
        Reads in image from ImPath
        Args:
            cv2_flag: cv2 flag, e.g. cv2.IMREAD_GRAYSCALE
            dirname: optional, read image with same root_directory and filename from different directory

        Returns:
            np.ndarray
        """
        # If dirname is not None then read same file in different subdir
        if dirname is not None:
            return cvw.imread(os.path.join(self.root_directory, dirname, self.filename), cv2_flag)
        else:
            return cvw.imread(self.path, cv2_flag)

    def imwrite(self, dirname, image: np.ndarray) -> None:
        """ write image to new directory with same root_directory and filename, directory must exist """
        cv2.imwrite(os.path.join(self.root_directory, dirname, self.filename), image)


class ImageGenerator:
    """ Dynamically yield images from list/iterator of ImPath/np.ndarray """

    def __init__(
            self,
            data: List[np.ndarray] or List[ImPath] or Iterator[np.ndarray] or Iterator[ImPath],
            cv2_flag: int = None
    ):
        """
        Args:
            data: List/Iterator of np.ndarray/ImPath
            cv2_flag: If instance of data is ImPath, cv2 flag that image will be read with. e.g. cv2.IMREAD_GRAYSCALE
        """
        self.data = data
        self.cv2_flag = cv2_flag

    def __iter__(self):
        for datum in self.data:
            if isinstance(datum, np.ndarray):
                yield datum
            elif isinstance(datum, ImPath):
                if self.cv2_flag is None:
                    raise ValueError("If Data is composed of ImPaths, a cv2_flag must be specified")
                yield datum.imread(self.cv2_flag)
            else:
                raise TypeError("Accepted types for individual instances are either np.ndarray or ImPath")


def glob_impath(dir: str) -> Iterator[ImPath]:
    """
    Takes directory and yields ImPaths
    Args:
        dir: directory path w/ shell style wildcards (e.g. /usr/PyCharmProjects/*.png)

    Returns:
        ImPaths
    """
    for path in glob.iglob(dir):
        yield ImPath(path)


def glob_img(dir: str, cv2_flag: int) -> ImageGenerator:
    """
    Takes directory and yields ImageGenerator
    Args:
        dir: directory path w/ shell style wildcards (e.g. /usr/PyCharmProjects/*.png)
        cv2_flag: If instance of data is ImPath, cv2 flag that image will be read with. e.g. cv2.IMREAD_GRAYSCALE

    Returns:
        ImageGenerator
    """
    img_paths = glob_impath(dir)
    return ImageGenerator(img_paths, cv2_flag=cv2_flag)
