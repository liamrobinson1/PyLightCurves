from .attitude_lib import rv_to_dcm
from .math_utility import dot, hat
import numpy as np


def rand_unit_vectors(num: int) -> np.ndarray:
    """Generates uniform random vectors on S^2

    Args:
        num (int): Number of unit vectors to generate

    Returns:
        np.ndarray num x 3: Sampled unit vectors

    """
    return rand_cone_vectors(np.array([1, 0, 0]), np.pi, num)


def rand_cone_vectors(
    cone_axis: np.array, cone_half_angle: float, num: int
) -> np.array:
    """Generates uniform random unit vectors in a cone

    Args:
        cone_axis (np.ndarray 1x3): Axis of symmetry for the cone
        cone_half_angle (float): Half-angle of the cone
        num (int): Number of vectors to sample

    Returns:
        np.ndarray num x 3: Sampled unit vectors

    """
    r1 = np.random.rand(num)
    r2 = np.random.rand(num)
    z = (1 - np.cos(cone_half_angle)) * r1 + np.cos(cone_half_angle)
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
