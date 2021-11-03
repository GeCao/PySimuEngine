from enum import Enum

import numpy
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from .Constants import *


class AutoEnum(Enum):
    """ Auto generate index of enum """

    def __new__(cls):
        value = int(len(cls.__members__))
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    @classmethod
    def convert_index_to_enum(cls, index):
        for obj in cls.__members__.values():
            if obj.value == index:
                return obj
        return None


class MessageAttribute(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2


def degToRad(theta):
    theta_calmp = theta - 360 * math.floor(theta / 360)
    return theta_calmp * math.pi / 180.0


def RadToDeg(theta):
    return math.degrees(theta)


def normalize(v):
    m = math.sqrt(v.dot(v))
    if m == 0:
        return v
    return v / m


def cross(v1, v2):
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], -(v1[0] * v2[2] - v1[2] * v2[0]), v1[0] * v2[1] - v1[1] * v2[0]])


def get_rotation_mat_from_euler_angle(vec, theta, my_dtype=default_dtype):
    normalized_vec = normalize(vec)
    x, y, z = normalized_vec[0], normalized_vec[1], normalized_vec[2]
    cos_theta, sin_theta = math.cos(degToRad(theta)), math.sin(degToRad(theta))
    return numpy.array([[cos_theta + (1 - cos_theta) * x * x, x * y * (1 - cos_theta) - z * sin_theta,
                         x * z * (1 - cos_theta) + y * sin_theta],
                        [x * y * (1 - cos_theta) + z * sin_theta, cos_theta + (1 - cos_theta) * y * y,
                         y * z * (1 - cos_theta) - x * sin_theta],
                        [x * z * (1 - cos_theta) - y * sin_theta, y * z * (1 - cos_theta) + x * sin_theta,
                         cos_theta + (1 - cos_theta) * z * z]],
                       dtype=my_dtype)


def get_quaternions_from_euler_angle(vec, theta, my_dtype=default_dtype):
    pass


def compute_ortho_mat(left, right, bottom, top, near, far, my_dtype=default_dtype):
    return np.array([[2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                     [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                     [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
                     [0.0, 0.0, 0.0, 1.0]], dtype=my_dtype).T


def compute_perspective_mat(fov, aspect, near, far, my_dtype=default_dtype):
    fov_cot = 1.0 / math.tan(degToRad(fov / 2.0))

    return np.array([[fov_cot / aspect, 0, 0, 0],
                     [0, fov_cot, 0, 0],
                     [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
                     [0, 0, -1.0, 0]], dtype=my_dtype).T


def compute_lookat_mat(eye, center, up, my_dtype=default_dtype):
    Z = normalize(eye - center)
    Y = normalize(up)
    X = np.cross(Y, Z)
    Y = np.cross(Z, X)
    matrix = np.eye(4, dtype=my_dtype)
    matrix[0, 0:3] = X
    matrix[1, 0:3] = Y
    matrix[2, 0:3] = Z
    matrix[0:3, 3] = [-np.dot(X, eye), -np.dot(Y, eye), -np.dot(Z, eye)]
    return matrix.T


def _save_img_2d(array, filename, colormap='RGB'):
    array = _process_img_2d(array, colormap)
    img = Image.fromarray(array)
    img.save(filename)


def _read_img_2d(file_path, my_dtype=default_dtype):
    return np.array(imageio.imread(file_path), my_dtype)


def _process_img_2d(array, colormap='RGB'):
    array = np.squeeze(array)
    # deal with velocity
    if len(array.shape) == 3:
        # add a zero channel to the 2d velocity
        if array.shape[2] == 2:
            array = np.concatenate(
                (array, np.expand_dims(np.zeros_like(array[..., 0]), axis=2)),
                axis=2
            )
    max_ = np.absolute(array.max())
    min_ = array.min() if array.min() >= 0 else -max_
    array = (array - min_) / (max_ - min_)  # [0, 1]
    if colormap == 'RGB':
        array = array * 255.0
    elif colormap == 'RDBU':
        array = plt.cm.RdBu(array) * 255.0
    elif colormap == 'SEISMIC':
        array = plt.cm.seismic(array) * 255.0
    elif colormap == 'ORRD':
        cc = plt.cm.get_cmap("OrRd")
        array = np.minimum(1, cc(array) / cc(0)) * 255.0
    elif colormap == 'HOT':
        array = plt.cm.hot(array) * 255.0
    else:
        raise ValueError('Colormap {} not supported.'.format(colormap))
    array = np.uint8(array)
    return array
