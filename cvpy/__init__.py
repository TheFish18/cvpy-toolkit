"""
Computer Vision for Python Toolkit

Welcome to `cvpy`, a Python library designed to simplify and streamline common computer vision tasks,
with a particular focus on segmentation.

## Features

### 1. **Image Segmentation Toolkit (`cvpy.imseg`)**
   The `cvpy.imseg` package provides robust tools for creating and manipulating:
   - **Bounding Boxes**: Easily define and adjust bounding boxes within images.
   - **Contours**: Tools to work with image contours, simplifying the process of identifying and working with object boundaries.

Within this package, you'll also find `cvpy.imseg.immasks`, which includes:
   - **Image Segmentation Masks**: Support for various representations of image segmentations (masks),
   enabling flexible and efficient manipulation of mask data. Including
    - **BinaryMask**
    - **SpareMask**
    - **CategoricalMask**
    - **CompoundMask**

### 2. **OpenCV Wrapper and Extender (`cvpy.cvw`)**
   The `cvpy.cvw` package acts as a wrapper and extender for OpenCV,
   adding new functionality and simplifying common OpenCV workflows,
   while maintaining compatibility with the broader OpenCV ecosystem.

   Highlights include
    - a simple GUI generator.
    - FileNotFound Errors for cv2.imread

## Installation
To install `cvpy`, simply run:
```bash
pip install cvpy-toolkit
```

## Getting Started
# TODO(add some examples)

## Contributing
We welcome contributions to `cvpy`. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

---

Let me know if you'd like any changes or additions!
"""

from . import cvw
from . import imseg
