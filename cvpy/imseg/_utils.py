import numpy as np


def get_2d_rotation_matrix(radians):
    return np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
