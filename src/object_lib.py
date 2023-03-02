import os
import numpy as np
import scipy
from typing import Tuple
import pyvista as pv

from .math_utility import *
from .rand_geom import *
from .light_lib import Brdf


class Object:
    def __init__(
        self,
        obj_file: str = None,
        obj_vf: Tuple[np.ndarray, np.ndarray] = None,
        is_dual: bool = False,
    ):
        """Initializes an instance of Object from either:
            - a path to a *.obj file
            - Vertex and face adjacency information

        Args:
            obj_file (str): *.obj file to read in $MODELDIR folder
            obj_vf (tuple[np.ndarray, np.ndarray]): Vertex and face triangle information
            is_dual (bool): Flag to prevent building dual's dual (leading to infinite recursion)

        """
        if obj_file is not None:
            self.file_name = obj_file
            self.path = f"{os.environ['MODELDIR']}/{obj_file}"
            self._mesh= pv.read(self.path)
            self.f = self._mesh.faces.reshape(-1, 4)[:, 1:]
            self.v = self._mesh.points
        elif obj_vf is not None:
            self.v = obj_vf[0]
            self.f = obj_vf[1]
            padded_f = np.hstack(
                (3*np.ones((self.f.shape[0],1), dtype=np.int64), self.f)
            ).flatten()
            self._mesh = pv.PolyData(self.v, padded_f)
        else:
            ValueError("Either obj_path or obj_vf must be input")
        self.is_dual = is_dual
        self.build_properties()
        self.flip_normals()

    def build_properties(self):
        """Builds all relevant object properties for shape inversion or visualization"""
        self.vertices_on_each_face()
        self.compute_face_normals()
        self.compute_face_areas()
        self.unique_areas_and_normals()
        self.compute_face_centroids()
        self.compute_supports()
        self.compute_volume()
        self.get_egi()
        self.get_inner_vertices()
        if not self.is_dual:
            self.get_dual()

    def vertices_on_each_face(self):
        """Computes a np.ndarray Nx3x3 of vertices on each face"""
        self.fv = self.v[self.f]

    def get_face_vertices(self) -> Tuple[np.ndarray]:
        """Reshapes face vertices into tuple of all 1st vertices, 2nd vertices, 3rd vertices"""
        return (self.fv[:, 0, :], self.fv[:, 1, :], self.fv[:, 2, :])

    def compute_face_centroids(self):
        """Computes the centroids of each face by averaging vertices"""
        (v1, v2, v3) = self.get_face_vertices()
        self.face_centroids = (v1 + v2 + v3) / 3

    def compute_face_normals(self):
        """Computes the normals of each face"""
        (v1, v2, v3) = self.get_face_vertices()
        self.face_normals = hat(np.cross(v2 - v1, v3 - v1))

    def compute_face_areas(self):
        """Computes the area of each face"""
        (v1, v2, v3) = self.get_face_vertices()
        self.face_areas = vecnorm(np.cross(v2 - v1, v3 - v1)).flatten() / 2

    def compute_supports(self):
        """Computs the support (perpendicular distance from plane containing face to origin)"""
        (v1, _, _) = self.get_face_vertices()
        self.supports = dot(v1[self.all_to_unique, :], self.unique_normals)

    def compute_volume(self):
        """Computs the volume via the supports and unique areas"""
        self.volume = 1 / 3 * np.sum(dot(self.supports, self.unique_areas))

    def unique_areas_and_normals(self):
        """Finds groups of unique normals and areas to save rows elsewhere"""
        (self.unique_normals, self.all_to_unique, self.unique_to_all) = unique_rows(
            self.face_normals, return_index=True, return_inverse=True
        )
        self.unique_areas = np.zeros((self.unique_normals.shape[0]))
        np.add.at(self.unique_areas, self.unique_to_all, self.face_areas)

    def get_egi(self):
        """Computes the Extended Gaussian Image (EGI)"""
        self.egi = np.expand_dims(self.unique_areas, 1) * self.unique_normals

    def get_dual(self):
        """Sets the dual object"""
        dual_v = self.unique_normals / self.supports
        dual_f = scipy.spatial.ConvexHull(dual_v).simplices
        self.dual = Object(obj_vf=(dual_v, dual_f), is_dual=True)

    def get_inner_vertices(self):
        """Computes which vertices are within the convex hull (
        not contained in any faces of the convhull)
        """
        nvert = self.v.shape[0]
        self.verts_within_convhull = np.setdiff1d(
            np.arange(0, nvert), np.unique(self.f)
        )

    def flip_normals(self):
        """For convex objects, flips any normal vectors pointing inside the object"""
        n_points_in = dot(self.face_centroids, self.face_normals).flatten() < 0
        if any(n_points_in):
            self.f[n_points_in, :] = np.array(
                [self.f[n_points_in, 0], self.f[n_points_in, 2], self.f[n_points_in, 1]]
            ).T
            self.build_properties()

    def render(self, pl: pv.Plotter = None, **kwargs):
        """Plots the object mesh using pyvista

        Args:
            pl (pyvista.Plotter): Plotter to render object to

        Returns:


        """
        if pl is None:
            pl = pv.Plotter()
        pl.add_mesh(self._mesh, **kwargs)
        pl.show()

    def compute_convex_light_curve(
        self, brdf: Brdf, svb: np.ndarray, ovb: np.ndarray
    ) -> np.ndarray:
        """Computes the light curve of a covex object with a given BRDF

        Args:
            brdf (Brdf): BRDF to render with
            svb (np.ndarray nx1): Sun unit vectors in the body frame
            ovb (np.ndarray nx1): Observer unit vectors in the body frame

        Returns:
            np.ndarray nx1: Observed irradiance at 1 unit distance

        """
        g = brdf.compute_reflection_matrix(svb, ovb, self.unique_normals)
        return g @ self.unique_areas


def build_dual(normals: np.array, supports: np.array) -> Object:
    """Computes the polyhedral dual of the set of normals and supports

    Args:
        normals (np.ndarray nx3): Outward-facing normal vectors
        supports (np.ndarray nx1): Support of each face (see get_supports above)

    Returns:
        Object: Dual object defined by the normals and supports

    """
    dual_v = normals / supports
    dual_f = scipy.spatial.ConvexHull(dual_v).simplices
    return Object(obj_vf=(dual_v, dual_f), is_dual=True)


def construct_from_egi_and_supports(egi: np.array, support: np.array) -> Object:
    """Constructs an object from an input Extended Gaussian Image (EGI) and support set

    Args:
        egi (np.ndarray nx3): EGI of the object
        support (np.ndarray nx1): Supports of each face in the EGI

    Returns:
        Object: Resulting convex object, with potentially fewer faces than
        the original EGI depending on supports selected

    """
    dual_obj = build_dual(hat(egi), np.expand_dims(support, axis=1))
    b = np.ones((dual_obj.f.shape[0], 3, 1))
    rec_obj_verts = np.reshape(
        np.linalg.solve(dual_obj.fv, b), (dual_obj.f.shape[0], 3)
    )
    rec_obj_faces = scipy.spatial.ConvexHull(rec_obj_verts).simplices
    rec_obj = Object(obj_vf=(rec_obj_verts, rec_obj_faces))
    return (rec_obj, dual_obj)


def support_reconstruction_error(egi: np.array, support: np.ndarray) -> float:
    """Computes the error in the reconstruction of an object
    from an Extended Gaussian Image (EGI) and support set

    Args:
        egi (np.ndarray nx3): EGI to attempt to reconstruct a convex polytope for
        support (np.ndarray nx1): Supports that may or may not correctly
            construct a polytope with the required EGI

    Returns:
        float: Support reconstruction error for optimization

    """
    small_tol = 1e-4
    (rec_obj, rec_dual) = construct_from_egi_and_supports(egi, support)
    nrec = rec_obj.unique_normals.shape[0]
    nref = egi.shape[0]
    assert nrec <= nref, ValueError("more normals reconstructed than in reference?!")
    (v1_grid, v2_grid) = np.meshgrid(np.arange(0, nref), np.arange(0, nrec))
    normal_diff = vecnorm(
        rec_obj.unique_normals[v2_grid.flatten(), :] - hat(egi[v1_grid.flatten(), :])
    )
    normal_diff = np.reshape(normal_diff, (nref, nrec), order="F")
    matching_normals = np.argwhere(normal_diff < small_tol)
    ref_normals_missing = np.setdiff1d(np.arange(0, nref), matching_normals[:, 0])
    area_error_missing = np.sum(vecnorm(egi[ref_normals_missing, :]))
    area_error_matching = np.sum(
        np.abs(
            vecnorm(egi[matching_normals[:, 0]]).flatten()
            - rec_obj.unique_areas[matching_normals[:, 1]]
        )
    )
    support_error = 0
    # Finding distance to dual convex hull
    if ref_normals_missing.size:
        nmiss = ref_normals_missing.size
        ndualface = rec_dual.unique_normals.shape[0]
        (v1_grid, v2_grid) = np.meshgrid(np.arange(0, nmiss), np.arange(0, ndualface))
        all_missing_dual_vertices = rec_dual.v[
            ref_normals_missing[v1_grid.flatten()], :
        ]
        all_dual_normals = rec_dual.unique_normals[v2_grid.flatten(), :]
        all_dual_supports = rec_dual.supports[v2_grid.flatten(), :]
        missing_pt_dists_to_hull = points_to_planes(
            pt=all_missing_dual_vertices,
            plane_n=all_dual_normals,
            support=all_dual_supports,
        )
        missing_pt_dists_to_hull = np.reshape(
            missing_pt_dists_to_hull, (nmiss, ndualface), order="F"
        )
        min_missing_dists_to_hull = np.min(
            np.abs(missing_pt_dists_to_hull), axis=1, keepdims=True
        )
        support_error = np.sum(min_missing_dists_to_hull)

    err = area_error_matching + 20 * support_error + area_error_missing
    print(
        f"{err:.2e} | [{support_error:.2e}, {area_error_missing:.2e}, {area_error_matching:.2e}] | {ref_normals_missing.size}/{nref} missing!"
    )
    return err


def optimize_supports(egi: np.array) -> Object:
    """Optimizes a support vector to construct
    the polytope with the given Extended Gaussian Image (EGI)

    Args:
        egi (np.ndarray nx3): EGI to find supports for
            (convergence only possible if the sum of rows is zero)

    Returns:
        Object: Produced from optimal set of supports

    """
    h0 = np.sqrt(np.sum(vecnorm(egi)) / (4 * np.pi)) * np.ones(egi.shape[0])
    fun = lambda h: support_reconstruction_error(egi, h)
    res = scipy.optimize.minimize(
        fun,
        h0,
        method="SLSQP",
        options={"ftol": 1e-9, "disp": True, "maxiter": int(1e3)},
        bounds=np.tile([1e-2, 10], (egi.shape[0], 1)),
    )
    (rec_obj, _) = construct_from_egi_and_supports(egi, res.x)
    return rec_obj


def optimize_egi(
    lc: np.ndarray,
    svb: np.ndarray,
    ovb: np.ndarray,
    brdf: Brdf,
    num_candidates: int = int(1e3),
    merge_iter: int = 1,
    merge_angle: float = np.pi / 10,
) -> np.ndarray:
    """Optimizes an Extended Gaussian Image (EGI) to fit observed irradiances

    Args:
        lc (np.ndarray nx1): Unit irradiance light curve
        svb (np.ndarray nx3): Sun unit vectors in the body frame for each observation
        ovb (np.ndarray nx3): Observer unit vectors in the body frame
        brdf (Brdf): BRDF to use
        num_candidates (int): Number of candidate normal vectors to fit
        merge_iter (int): Number of iterations for the merging step
        merge_angle (float): Angle between vectors to merge by

    Returns:
        np.ndarray nx3: Optimized EGI

    """
    normal_candidates = rand_cone_vectors(np.array([0, 0, 1]), np.pi, num_candidates)
    g_candidates = brdf.compute_reflection_matrix(svb, ovb, normal_candidates)
    a_candidates = np.expand_dims(
        scipy.optimize.nnls(g_candidates, lc.flatten())[0], axis=1
    )
    egi_candidate = normal_candidates * a_candidates
    egi_candidate = remove_zero_rows(egi_candidate)
    egi_candidate = merge_clusters(egi_candidate, merge_angle, miter=merge_iter)
    return close_egi(egi_candidate)
