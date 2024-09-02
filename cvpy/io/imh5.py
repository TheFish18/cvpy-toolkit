from typing import Iterable

import h5py
import numpy as np
import tqdm
from tqdm import tqdm

from joshpy.images.mapper import ClassMap
from joshpy.gen_utils import handle_kwarg

class H5(h5py.File):
    def __init__(self, name, mode=None):
        super().__init__(name, mode)
        self._class_map = None

    @property
    def class_map(self):
        if self._class_map is None:
            self._class_map = self.get_class_map()
        return self._class_map

    def save_class_map(self, class_map: dict):
        for k in self.get_class_map():
            del self.attrs[k]
        for k, v in class_map.items():
            self.attrs[k] = v
        return self

    def get_class_map(self):
        ks = self.attrs.keys()
        class_map = {}
        for k in ks:
            class_map[k] = self.attrs[k]
        return ClassMap(class_map, "")

    def _transfer_class_map(self, new):
        """ Transfers class map from self to new h5 """
        assert type(new) is H5, "New h5 must be type H5"
        new.save_class_map(self.get_class_map())
        return new

    def shuffle(self, save_path, mode='w', exclude: Iterable=None, keys=None):
        """
        Shuffles entire dataset while maintaining relative ordering between keys. Saves to path.
        Args:
            exclude: Keys to exclude from being shuffled

        Returns:
            None
        """
        if keys is not None:
            ks = keys
        else:
            exclude = set() if exclude is None else set(exclude)
            ks = list(set(self.keys()) - set(exclude))

        n = self.get(ks[0]).shape[0]

        for k in ks:
            assert self.get(k).shape[0] == n, 'All keys being shuffled must be same size'

        idxs = np.random.choice(n, n, replace=False)

        f = H5(save_path, mode)
        f.save_class_map(self.get_class_map())

        print(f'All keys: {ks}')
        for k in ks:
            print(f'Currently shuffling: {k}')
            curr_imgs = self.get(k)
            out_imgs = []
            for i in tqdm(idxs):
                img = curr_imgs[i]
                out_imgs.append(img)
            print(f'Writing key: {k}')
            out_imgs = np.array(out_imgs)
            f.create_dataset(
                name=k,
                dtype=curr_imgs.dtype,
                shape=curr_imgs.shape,
                data=out_imgs
            )
        self._transfer_class_map(f)
        f.close()

    def transpose_like(self, other):
        """
        Transpose f2[k2] such that the class maps of each match, keys in class map must match
        Args:
            f1: H5 File 1
            k1:
            f2: H5 File 2
            k2:

        Returns:
            numpy array
        """
        self_class_map = self.get_class_map()
        other_class_map = other.get_class_map()

        assert set(self_class_map.keys()) == set(other_class_map.keys()), 'f1.class_map must match f2.class_map'

        update_map = self_class_map.get_update_map(other_class_map)
        true_order = sorted(self_class_map.values())
        transpose_order = [update_map[x] for x in true_order]
        return(transpose_order)

def _save_dataset(f, key: str, data: np.ndarray, compression=None, chunks=None):
    """ Saves a dataset for appending """
    maxshape = (None, *data.shape[1:])
    f.create_dataset(
        name=key,
        maxshape=maxshape,
        dtype=data.dtype,
        chunks=chunks,
        compression=compression,
        data=data
    )

def append_hdf5(save_path: str, key: str, data: np.ndarray, **kwargs):
    """
    Args:
        save_path: path of h5, if not exists -> will create
        key: key to append to/add
        data: data
        kwargs:
            - chunks: default None
            - compression: default None

    Returns:
        None
    """
    chunks = handle_kwarg(key="chunks", default=None, **kwargs)
    compression = handle_kwarg(key='compression', default=None, **kwargs)

    with h5py.File(save_path, 'a') as f:
        keys = f.keys()
        if key not in keys:
            _save_dataset(f, key, data, compression, chunks)
        else:
            dataset = f.get(key)
            curr_len = dataset.shape[0]
            new_len = curr_len + data.shape[0]
            dataset.resize(new_len, axis=0)
            dataset[-data.shape[0]:] = data
