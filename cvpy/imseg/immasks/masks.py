import os
import json
import warnings

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Sequence

import cv2
import numpy as np

from cvpy.imseg.immasks._utils import _scale_image, _COLOR_GEN


class _Mask(ABC):
    """
    Abstract Base Class for Masks:
    - for user consistency, any functions returning a mask should return a _Mask subclass object
    """
    def __init__(self):
        self._mask = None
        self._class_map = None

    @property

    @abstractmethod
    def mask(self):
        """ returns mask """

    @mask.setter
    @abstractmethod
    def mask(self, mask):
        """ handle paths, asserts valid """

    @property
    @abstractmethod
    def class_map(self):
        """ returns class map """

    @class_map.setter
    @abstractmethod
    def class_map(self, class_map):
        """ handles paths, asserts valid """

    @abstractmethod
    def __getitem__(self, item: str):
        """ Gets a class from class_map and returns new _Mask subclass object """
        
    def __iter__(self):
        for cls in self.class_map:
            yield self.__getitem__(cls)

    @abstractmethod
    def show(self, scale=1.) -> None:
        """ Shows mask/masks to user, destroy windows afterward """

    @abstractmethod
    def rename_classes(self, rename_d: Dict[str, str]):
        """
        Renames classes in self.class_map according to rename_d and coerces mask into proper format.

        Can be used to consolidate classes: {cls1: cls1, cls2: cls1}
        or drop classes: {cls1: None}
        classes not present in rename_d will be preserved

        Args:
            rename_d: {old class: new class}

        Returns:
            Instance of self with changes made
        """

    def update_classes(self, rename_d: dict):
        """
        THIS IS BEING DEPRECATED FOR RENAME_CLASSES

        renames classes in self.class_map according to rename_d and coerces mask into proper format.
        Can be used to consolidate classes: ex {cls1: cls2, cls3: cls2}
        Args:
            rename_d: {old_class_name: new_class_name}

        Returns:
            _Mask subclass object
        """
        warnings.warn(
            "update_classes is deprecated and will be removed in a future version. Use rename_classes instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.rename_classes(rename_d)

    @abstractmethod
    def transpose_values(self, new_class_map):
        """
        Reorder class map values and coerce mask into correct format
        Args:
            new_class_map: {cls: new_val}

        Returns:
            _Mask subclass object
        """


class CompositeMask(_Mask):
    """ Composite Mask has dims (H, W, 3) where three corresponds to (blue, green, red) channels. """

    def __init__(self, mask: np.ndarray | str, class_map: dict | str):
        super().__init__()
        self.mask = mask
        self.class_map = class_map

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, np.ndarray):
            s = mask.shape
            assert len(s) == 3 and s[-1] == 3, f"mask dimensions must be (H, W, 3), got {s}"
            self._mask = mask
        elif isinstance(mask, str):
            self._mask = cv2.imread(mask)
        else:
            raise TypeError(f"mask must either be a np.ndarray or a path to an image, got {type(mask)}")

    @property
    def class_map(self):
        return self._class_map

    @class_map.setter
    def class_map(self, class_map):
        if isinstance(class_map, dict):
            self._class_map = class_map
        elif isinstance(class_map, str):
            with open(class_map) as f:
                self._class_map = json.load(f)
        else:
            raise TypeError(f'class map must be a dict or a path, got {type(class_map)}')

    def __getitem__(self, item: str):
        # colors of instances that belong to class
        vals = np.array(self.class_map[item])[None, None, :, :]
        bool_mask = np.all(self.mask[..., None, :] == vals, axis=-1)
        bool_mask = np.any(bool_mask, axis=-1)
        mask = np.where(bool_mask[:, :, None], self.mask, 0)
        return CompositeMask(mask, {item: vals})

    def get_instance(self, color: Tuple[int, int, int]):
        """
        Get binary mask of a single defect instance
        Args:
            color: the color of defect instance to extract from composite mask, (r, g, b)

        Returns:
            BinaryMask
        """
        # mask = np.all(np.where(self.mask == color, 1, 0), axis=-1).astype(np.uint8)  # 10 @ 2.03
        # mask = np.all(self.mask == color, axis=-1).astype(np.uint8)  # 10 @ 1.44

        mask_r, mask_g, mask_b = self.mask[:, :, 0], self.mask[:, :, 1], self.mask[:, :, 2]
        r, g, b = color
        mask = (  # 10 @ 0.28
                (mask_r == r) * (mask_g == g) * (mask_b == b)
        ).astype(np.uint8)

        return BinaryMask(mask, "instance")

    def show(self, scale: float = 1.):
        cv2.imshow('composite', _scale_image(self.mask, scale))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def rename_classes(self, rename_d: dict):
        """
        Renames classes in self.class_map according to rename_d and coerces mask into proper format.

        Can be used to consolidate classes: {cls1: cls1, cls2: cls1}
        or drop classes: {cls1: None}
        classes not present in rename_d will be preserved

        Args:
            rename_d: {old class: new class}

        Returns:
            CompositeMask
        """
        new_cm = {}
        for cls, vals in self.class_map.items():
            new_cls = rename_d.get(cls, cls)
            new_vals = new_cm.get(new_cls, [])

            new_vals.extend(vals)
            new_cm[new_cls] = new_vals

        return CompositeMask(self.mask, new_cm)

    def transpose_values(self, new_class_map):
        raise NotImplemented("CompositeMask does not implement transpose_values")

    def as_categorical(self, include_background=True):
        """ convert CompositeMask to CategoricalMask """
        cat_masks = []
        cat_cm = {}

        for i, (cls, vals) in enumerate(self.class_map.items()):
            cat_cm[cls] = i
            cat_mask = np.mean(self.__getitem__(cls).mask, axis=-1, keepdims=True)
            cat_mask = np.where(cat_mask > 0, 1, 0).astype(np.uint8)
            cat_masks.append(cat_mask)

        cat_mask = np.concatenate(cat_masks, axis=-1)
        if include_background:
            cat_cm = {k: v + 1 for k, v in cat_cm.items()}
            cat_cm['background'] = 0
            background = np.where(cat_mask.sum(axis=-1, keepdims=True) == 0, 1, 0)
            cat_mask = np.concatenate((background, cat_mask), axis=-1)

        return CategoricalMask(cat_mask, cat_cm)

    def save(self, mask_name, mask_dir, class_map_dir, img_ext='.png'):
        """
        Args:
            mask_name:
            mask_dir:
            class_map_dir:
            img_ext:

        Returns:

        """
        mask_name = os.path.splitext(mask_name)[0]

        cm_ext = ".json"
        mask_path = os.path.join(mask_dir, f"{mask_name}{img_ext}")
        cm_path = os.path.join(class_map_dir, f"{mask_name}{cm_ext}")

        cv2.imwrite(mask_path, self.mask)
        with open(cm_path, 'w') as f:
            json.dump(self.class_map, f, indent=4)


class CategoricalMask(_Mask):
    """
    Categorical masks have dims (H, W, C) where C are classes.
    Class map have structure {cls: cls_idx} such that mask[:, :, cls_idx] will return binary mask for cls
    """

    def __init__(self, mask: np.ndarray | str, class_map: dict | str):
        super().__init__()
        self.mask = mask
        self.class_map = class_map

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, np.ndarray):
            s = mask.shape
            assert len(s) == 3, f"categorical masks have rank 3 (ie (H, W, C), got {s}"
            self._mask = mask
        elif isinstance(mask, str):
            ext = os.path.splitext(mask)[-1]
            if os.path.isfile(mask) and ext == ".npy":
                with open(mask, 'rb') as f:
                    self._mask = np.load(f)
        else:
            raise TypeError("mask must either by np.ndarray or path with ext '.npy'")

    @property
    def class_map(self):
        return self._class_map

    @class_map.setter
    def class_map(self, class_map):
        if isinstance(class_map, dict):
            cm = class_map
        elif isinstance(class_map, str):
            if os.path.isfile(class_map):
                with open(class_map) as f:
                    cm = json.load(f)
        else:
            raise TypeError("class map must either be dict or path to dict")

        self._class_map = dict(sorted(cm.items(), key=lambda item: item[1]))


    def __getitem__(self, item: str):
        """ gets binary mask of cls """
        cls_idx = self.class_map[item]
        return BinaryMask(self.mask[:, :, cls_idx], item)

    def show(self, scale=1.):
        for cls, cls_idx in self.class_map.items():
            mask = self.mask[:, :, cls_idx]
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            mask = _scale_image(mask, scale)
            cv2.imshow(cls, mask)
            cv2.waitKey()
        cv2.destroyAllWindows()

    def show_color(self, scale=1.):
        mask = self.as_sparse()
        mask.show_color(scale)


    def rename_classes(self, rename_d: dict):
        """
        Renames classes in self.class_map according to rename_d and coerces mask into proper format.

        Can be used to consolidate classes: ex {cls1: cls2, cls3: cls2}.
        Classes not in rename_d will be preserved in output
        Classes can be dropped by setting new_class_name to None, e.g. {old_class_name: None}

        Args:
            rename_d: {old_class_name: new_class_name}

        Returns:
            CategoricalMask
        """
        old_masks_d = {k: self.mask[:, :, v] for k, v in self.class_map.items() if k not in rename_d}
        new_masks_d = {}
        default_mask = np.zeros((self.mask.shape[:2]), dtype=np.uint8)

        for old_k, new_k in rename_d.items():
            if new_k is None: continue
            old_idx = self.class_map[old_k]
            new_mask = new_masks_d.get(new_k, default_mask)
            new_mask = np.bitwise_or(new_mask, self.mask[:, :, old_idx])
            new_masks_d[new_k] = new_mask

        new_cm = {k: i for i, k in enumerate([*old_masks_d.keys(), *new_masks_d.keys()])}
        masks = np.array([*list(old_masks_d.values()), *list(new_masks_d.values())])
        masks = masks.transpose((1, 2, 0))

        return CategoricalMask(masks, new_cm)

    def transpose_values(self, new_class_map: Dict[str, int]):
        """
        mask will be coerced into format defined by new_class_map.

        keys in new_class_map but not self.class_map will be added as empty dim
        keys in self.class_map but not in new_class_map will raise error
            - if desire is to drop, use rename_classes
        multiple keys sharing same value will raise error
            - if desire is to combine classes, use rename_classes
        keys in both will be transposed

        Args:
            new_class_map: {classes: class_index}
                all keys in self.class_map must be present in new_class_map

        Raises:
            IndexError: if key in new_class_map not used
            ValueError: if multiple keys share same class

        Returns:
            CategoricalMask
        """
        old_keys = set(self.class_map.keys())
        new_keys = set(new_class_map.keys())
        if not old_keys <= new_keys:
            raise IndexError(
            f'all keys in old class map must be accounted for in new class map, '
            f'new class map missing: {old_keys - new_keys}'
            )

        new_cls_idxs = new_class_map.values()
        if len(new_cls_idxs) != len(set(new_cls_idxs)):
            raise ValueError(
                "No two keys can share the same value. If goal is to consolidate classes, use rename_classes."
            )
        if set(range(len(new_class_map))) != set(new_cls_idxs):
            raise ValueError(
                f"There must be no gaps in values (e.g. [0, 1, 2] is valid [0, 1, 3] is not, got {sorted(new_cls_idxs)}"
            )

        blank_mask = lambda: np.zeros_like(self.mask[:, :, 0])
        out_masks = [None] * len(new_cls_idxs)

        for new_cls, new_cls_idx in new_class_map.items():
            if new_cls not in old_keys:
                out_masks[new_cls_idx] = blank_mask()
            else:
                old_cls_idx = self.class_map[new_cls]
                out_masks[new_cls_idx] = self.mask[:, :, old_cls_idx]
        mask = np.array(out_masks, dtype=np.uint8).transpose((1, 2, 0))
        return CategoricalMask(mask, new_class_map)

        # new_idxs = self._transpose_order(self.class_map, new_class_map)
        # mask = self.mask[:, :, new_idxs]
        # return CategoricalMask(mask, new_class_map)

    @staticmethod
    def _transpose_order(old_cm, new_cm):
        old_keys = set(old_cm.keys())
        new_keys = set(new_cm.keys())

        assert old_keys <= new_keys, \
            f'all keys in old class map must be accounted for in new class map, ' \
            f'new class map missing: {old_keys - new_keys}'

        update_map = {new_cm[k]: old_cm[k] for k in old_keys}
        true_order = sorted(old_cm.values())
        transpose_order = [update_map[x] for x in true_order]
        return transpose_order

    def as_sparse(self, background_classes: str | List[str] = None):
        """
        Convert CategoricalMask to SparseMask, lossy when masks from multiple classes overlap.
        When multiple masks from multiple classes overlap, the highest cls_idx will be given priority.
            - For example, if self.class_map = {cls1: 1, cls2: 2, cls3: 3} and the masks for cls1, cls2, cls3 overlap,
            in the sparse representation, the overlap region will be classified as cls3 (ie 3 > 2 > 1)

        Args:
            background_classes: A class or classes representing background, i.e. {not_background: 0, background: 1}

        Returns:
            SparseMask
        """

        new_pxs_vals = np.arange(len(self.class_map)) + 1

        if background_classes is not None:
            if not isinstance(background_classes, list):
                background_classes = [background_classes]
            assert set(background_classes) < set(self.class_map.keys()), \
                "background classes must be a proper subset of self.class_map.keys()"
            background_idxs = [self.class_map[k] for k in background_classes]
            new_pxs_vals[background_idxs] = 0

        mask = self.mask * new_pxs_vals
        mask = np.max(mask, axis=-1)

        new_cm = {k: v for k, v in zip(self.class_map, new_pxs_vals) if v != 0}
        return SparseMask(mask, new_cm)

    def squeeze(self):
        """ transposes cm such that """
        n = len(self.class_map)
        new_cm = {k: v for k, v in self.class_map.items() if v <= n}

        poss_vals = set(range(n))
        used_vals = set(new_cm.values())
        new_vals = poss_vals - used_vals

        for cls in self.class_map:
            if cls in new_cm: continue
            new_val = new_vals.pop()
            new_cm[cls] = new_val

        return self.transpose_values(new_cm)

    def to_binary(self):
        mask = np.max(self.mask, axis=-1)
        mask = np.where(mask > 0, 1, 0)
        return BinaryMask(mask, 'binary')


class SparseMask(_Mask):
    """
    SparseMask have shape (H, W). Each pixel in mask corresponds to a class.
    """

    def __init__(self, mask: np.ndarray | str, class_map: dict | str):
        super().__init__()
        self._background = None

        self.mask = mask
        self.class_map = class_map

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, np.ndarray):
            s = mask.shape
            assert len(s) == 2, f"SparseMasks are rank 2 (ie (H, W)), got {s}"
            self._mask = mask
        elif isinstance(mask, str) and os.path.isfile(mask):
            self._mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        else:
            raise TypeError(f"mask must either by a np.ndarray or image path, got {type(mask)}")

    @property
    def class_map(self):
        return self._class_map

    @class_map.setter
    def class_map(self, class_map):
        if isinstance(class_map, dict):
            class_map = class_map
        elif isinstance(class_map, str) and os.path.isfile(class_map):
            with open(class_map) as f:
                class_map = json.load(f)
        else:
            raise TypeError("class map must either be a dict or path to a dict")

        self._class_map = dict(sorted(class_map.items(), key=lambda item: item[1]))

        # if user passed a background class, keep track of it
        cm_iter = iter(self.class_map)
        first_cls = next(cm_iter)
        first_px = self.class_map[first_cls]

        if first_px == 0:
            self.class_map.pop(first_cls)
            self._background = first_cls
        else:
            self._background = "background"

    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, value):
        self._background = value

    def __getitem__(self, item: str):
        """ returns binary mask of cls """
        px_val = self.class_map[item]
        mask = np.where(self.mask == px_val, 1, 0).astype(np.uint8)
        return BinaryMask(mask, item)

    def __next__(self):
        yield from self.class_map

    def show(self, scale=1.):
        for cls, px in self.class_map.items():
            temp = np.where(self.mask == px, 255, 0).astype(np.uint8)
            cv2.imshow(cls, _scale_image(temp, scale))
            cv2.waitKey()
        cv2.destroyAllWindows()

    def show_color(self, scale=1., color_map: Dict[str, Tuple[int, int, int] | str] = None):
        if color_map is None:
            color_map = {}
            for cls in self.class_map:
                color_map[cls] = next(_COLOR_GEN)

        out_image = np.zeros((*self.mask.shape, 3), dtype=np.uint8)

        for cls, color in color_map.items():
            px_val = self.class_map[cls]
            out_image = np.where(self.mask[:, :, None] == px_val, color, out_image)

        out_image = _scale_image(out_image, scale)
        cv2.imshow("color SparseMask", out_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def rename_classes(self, rename_d: dict):
        """
        Renames classes in self.class_map according to rename_d and coerces mask into proper format.
        Can be used to consolidate classes: ex {cls1: cls2, cls3: cls2}.
        Classes not in rename_d will be preserved in output.
        Classes can be dropped by setting new_class_name to None, e.g. {old_class_name: None}

        Args:
            rename_d: {old_class_name: new_class_name}

        Returns:
            CategoricalMask
        """
        consolidate_classes = {self.background: [0]}
        consolidate_classes.update({k: [v] for k, v in self.class_map.items() if k not in rename_d})

        for old_k, new_k in rename_d.items():
            new_k = self.background if new_k is None else new_k
            old_val = self.class_map[old_k]
            old_vals = consolidate_classes.get(new_k, [])
            old_vals.append(old_val)
            consolidate_classes[new_k] = old_vals

        out_mask = np.zeros_like(self.mask, dtype=np.uint8)
        new_cm = {}

        for new_val, (new_k, old_vals) in enumerate(consolidate_classes.items()):
            if new_val == 0: continue  # dont need to add 0 to 0 array
            out_mask = np.where(np.isin(self.mask, old_vals), new_val, out_mask)
            new_cm[new_k] = new_val

        return SparseMask(out_mask, new_cm)

    def transpose_values(self, new_class_map: dict):
        """ mask will be coerced into order defined by new_class_map. All classes must be present in self.class_map """
        new_mask = np.zeros_like(self.mask)
        for cls, new_px_val in new_class_map.items():
            if new_px_val == 0: raise ValueError("0 is reserved for background classes, cannot override")
            old_px_val = self.class_map[cls]
            new_mask = np.where(self.mask == old_px_val, new_px_val, new_mask)

        return SparseMask(new_mask.astype(np.uint8), new_class_map)

    def as_categorical(self, include_background=True):
        """
        Takes categorical and turns to one hot encoding
        Args:
            include_background: TODO add this

        Returns:
            One hot encoded array
        """
        y = self.mask
        cm = {self.background: 0}
        dtype = np.uint8

        cm.update(self.class_map)
        num_classes = max(cm.values()) + 1

        y = np.array(y, dtype="int")
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        # len of flattened array
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)

        if include_background:
            new_cm = {k: i for i, k in enumerate(cm)}
        else:
            cm.pop(self.background)
            new_cm = {k: i for i, k in enumerate(cm)}
            categorical = categorical[:, :, 1:]

        return CategoricalMask(categorical, new_cm)

    def squeeze(self):
        """ transposes cm such that """
        n = len(self.class_map)
        new_cm = {k: v for k, v in self.class_map.items() if v <= n}

        poss_vals = set(range(1, n+1, 1))
        used_vals = set(new_cm.values())
        new_vals = poss_vals - used_vals

        for cls in self.class_map:
            if cls in new_cm: continue
            new_val = new_vals.pop()
            new_cm[cls] = new_val

        # TODO can make this more efficient by not looping over classes that stay the same
        return self.transpose_values(new_cm)

    def to_binary(self):
        mask = np.where(self.mask > 0, 1, 0)
        return BinaryMask(mask, 'binary')


class BinaryMask:
    """ Special case of SparseMask """
    def __init__(self, mask: np.ndarray, cls: str):
        self._cls = cls
        self._sparse_mask = SparseMask(mask, {cls: 1})

    @property
    def mask(self):
        return self._sparse_mask.mask

    @mask.setter
    def mask(self, mask: np.ndarray):
        self._sparse_mask.mask = mask

    @property
    def class_map(self):
        return self._sparse_mask.class_map

    @class_map.setter
    def class_map(self, class_map: dict):
        self._sparse_mask.class_map = class_map

    @property
    def cls(self):
        return self._cls

    @cls.setter
    def cls(self, cls):
        self.class_map = {cls: 1}
        self._cls = cls

    def show(self, scale=1.):
        self._sparse_mask.show(scale)

def binaries_to_categorical(binaries: Sequence[BinaryMask]) -> CategoricalMask:
    cat_mask = np.array([binary.mask for binary in binaries])
    cat_mask = cat_mask.transpose((1, 2, 0))
    cm = {binary.cls: i for i, binary in enumerate(binaries)}

    return CategoricalMask(cat_mask, cm)


def binaries_to_sparse(binaries: Sequence[BinaryMask]) -> SparseMask:
    cat_mask = binaries_to_categorical(binaries)
    return cat_mask.as_sparse()
