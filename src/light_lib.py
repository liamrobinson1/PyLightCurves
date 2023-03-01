import numpy as np
from .math_utility import hat, rdot, dot


def brdf_diffuse(cd: np.ndarray) -> np.ndarray:
    return cd / np.pi


def brdf_phong(L, O, N, cd, cs, n):
    R = hat(2 * dot(N, L) * N - L)
    fd = brdf_diffuse(cd)
    fs = cs * (n + 1) / (2 * np.pi) * rdot(R, O) ** n / dot(N, L)
    return fd + fs


class Brdf:
    def __init__(self, name: str, cd: float, cs: float, n: int):
        self.validate(name, cd, cs, n)
        self.name = name
        self.cd = cd
        self.cs = cs
        self.n = n

    def validate(self, name: str, cd: float, cs: float, n: int):
        assert cd != 0 and cs != 0, ValueError("your BRDF is boring")
        assert cd >= 0 and cs >= 0, ValueError(
            "cd and cs must be >= 0 for energy conservation"
        )
        assert cd + cs <= 1, ValueError("cd + cs must be <= 1 for energy conservation")
        assert n > 0, ValueError("n > 0 for a physical brdf")
        assert name == "phong", ValueError("only 'phong' supported as BRDF name")

    def eval(self, L: np.array, O: np.array, N: np.array) -> np.array:
        return brdf_diffuse(self.cd) + brdf_phong(L, O, N, self.cd, self.cs, self.n)

    def eval_normalized_brightness(
        self, L: np.array, O: np.array, N: np.array
    ) -> np.array:
        fr = np.abs(self.eval(L, O, N))
        lc = fr * rdot(N, O) * rdot(N, L)
        return lc

    def compute_reflection_matrix(
        self, L: np.array, O: np.array, N: np.array
    ) -> np.ndarray:
        assert L.shape[0] == O.shape[0], ValueError(
            "Size of illumination vectors must match observer vectors"
        )
        geom_count = L.shape[0]
        normal_count = N.shape[0]
        [v1_grid, v2_grid] = np.meshgrid(
            np.arange(0, geom_count), np.arange(0, normal_count)
        )
        # Grid of values for each time and facet
        ov_all = O[v1_grid.flatten(), :]  # Column vector of all observer vectors
        sv_all = L[v1_grid.flatten(), :]  # Column vector of all sun vectors
        nv_all = N[v2_grid.flatten(), :]  # Column vector of all normal vectors

        lc_all = self.eval_normalized_brightness(sv_all, ov_all, nv_all)
        return np.reshape(lc_all, (geom_count, normal_count), order="F")
