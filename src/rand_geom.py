from .attitude_lib import rv_to_dcm
from .math_utility import dot, hat
import numpy as np


def rand_unit_vectors(num: int) -> np.ndarray:
    return rand_cone_vectors(np.array([1, 0, 0]), np.pi, num)


def rand_cone_vectors(cone_axis: np.array, theta: float, num: int) -> np.array:
    r1 = np.random.rand(num)
    r2 = np.random.rand(num)
    z = (1 - np.cos(theta)) * r1 + np.cos(theta)
    phi = 2 * np.pi * r2

    ref = hat(np.array([0, 0, 1]))
    cone_vec = np.transpose(
        np.array(
            [np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z]
        )
    )
    if all(ref != cone_axis):
        rot_vector = -np.cross(cone_axis, ref)
        rot_angle = np.arccos(dot(cone_axis, ref))
        rotm = rv_to_dcm(rot_vector * rot_angle)
    else:
        rotm = np.eye(3)
    return cone_vec @ rotm
