import math
from typing import Union, Tuple, Optional, List

import numpy as np

Vec4Type = Union[Tuple[float, float, float, float], np.ndarray]
Vec3Type = Union[Tuple[float, float, float], np.ndarray, 'Vec3', List[float]]
Vec3TypeX = Union[float, Vec3Type]
FloatType = Union[float, np.ndarray]  # Can be single element array (as a result of a numpy operation)


# ==========================================================
# Helper methods for dealing with heterogeneous input types
# ==========================================================

def unpack_nd_params(
        n: int,
        *args,
        default: Optional = None) -> Tuple:
    try:
        if len(args[0]) == n:
            assert all([args[i] is None for i in range(1, n)]), \
                "When a container is passed, all other values must not be given"
            components = args[0]
        elif len(args[0]) == n - 1:
            assert n == 2 or all([args[i] is None for i in range(2, n)]), \
                "When a container is passed, all other values must not be given"
            components = list(args[0]) + [args[1]]
        else:
            raise ValueError(f"Passed container must contain {n} or {n - 1} values!")
    except TypeError:
        # x was actually a number
        components = [default if arg is None else arg for arg in args]
    return components


def unpack_3d_params(
        x: Optional[Vec3TypeX],
        y: float,
        z: float,
        default: Optional = None) -> Tuple[float, float, float]:
    return unpack_nd_params(3, x, y, z, default=default)


def unpack_single(value: FloatType) -> float:
    if isinstance(value, np.ndarray):
        assert value.size == 1, f"Expected array with exactly one element! Got {value.size}"
        return value.item()
    else:
        return value


# ==========================================================
# 3D utilities
# ==========================================================

def rotation_matrix_between_vectors(vec1: Vec3TypeX, vec2: Vec3TypeX) -> np.ndarray:
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1 = Vec3(vec1)
    vec2 = Vec3(vec2)

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


def angle_between_vectors(vec1: Vec3TypeX, vec2: Vec3TypeX) -> float:
    vec1 = Vec3(vec1)
    vec2 = Vec3(vec2)
    return math.acos(vec1.dot(vec2) / (vec1.length() * vec2.length()))


def offset_vector_between_line_and_point(offset: Vec3TypeX, direction: Vec3TypeX, point: Vec3TypeX):
    offset = Vec3(offset)
    direction = Vec3(direction).normalize()
    point = Vec3(point)

    direction_to_point = point - offset
    closest_point_on_line = offset + direction_to_point.dot(direction) * direction
    move = point - closest_point_on_line
    return move


# ==========================================================
# Numpy Vectors
# ==========================================================


class Vec2(np.ndarray):
    def __new__(cls,
                x: Union[float, Tuple[float, float], np.ndarray],
                y: Optional[float] = None):
        vec2 = super().__new__(cls, (2,), dtype=np.float32)
        try:
            assert len(x) == 2, "Passed list must contain exactly 3 values"
            assert y is None, "When a list is passed, y must not be given"
            x, y = x
        except TypeError:
            # x was actually a number
            pass
        vec2[:] = [x, y]
        return vec2

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @x.setter
    def x(self, x: float):
        self[0] = x

    @y.setter
    def y(self, y: float):
        self[1] = y

    def length(self):
        return np.linalg.norm(self)

    def normalize(self) -> 'Vec2':
        return self / self.length()

    def rotate(self, angle: float) -> 'Vec2':
        new_x = self.x * math.cos(angle) - self.y * math.sin(angle)
        new_y = self.x * math.sin(angle) + self.y * math.cos(angle)
        self.x = new_x
        self.y = new_y
        return self


class Vec3(np.ndarray):
    def __new__(cls,
                x: Optional[Vec3TypeX] = None,
                y: Optional[float] = None,
                z: Optional[float] = None):
        # TODO: Allow no arguments -> 0 vector
        vec3 = super().__new__(cls, (3,), dtype=np.float32)
        x, y, z = unpack_3d_params(x, y, z, default=0)
        vec3[:] = [x, y, z]
        return vec3

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    @x.setter
    def x(self, x: float):
        self[0] = x

    @y.setter
    def y(self, y: float):
        self[1] = y

    @z.setter
    def z(self, z: float):
        self[2] = z

    def length(self):
        return np.linalg.norm(self)

    def normalize(self) -> 'Vec3':
        return self / self.length()

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(np.cross(self, other))

    def homogenize(self) -> 'Vec4':
        return Vec4(self, 1)


class Vec4(np.ndarray):
    def __new__(cls,
                x: Union[float, Tuple[float, float, float, float], np.ndarray, 'Vec4', 'Vec3'],
                y: Optional[float] = None,
                z: Optional[float] = None,
                w: Optional[float] = None):
        vec4 = super().__new__(cls, (4,), dtype=np.float32)
        x, y, z, w = unpack_nd_params(4, x, y, z, w)
        vec4[:] = [x, y, z, w]
        return vec4

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    @property
    def w(self) -> float:
        return self[3]

    @x.setter
    def x(self, x: float):
        self[0] = x

    @y.setter
    def y(self, y: float):
        self[1] = y

    @z.setter
    def z(self, z: float):
        self[2] = z

    @w.setter
    def w(self, w: float):
        self[3] = w

    def length(self):
        return np.linalg.norm(self)

    def normalize(self) -> 'Vec4':
        return self / self.length()
