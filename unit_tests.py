import unittest
from datetime import datetime, timezone
import os
from src.math_utility import *
from src.attitude_lib import *
from src.orbit_lib import *
from src.astro_const import AstroConstants
from src.observer_utility import *
from src.astro_time_utility import *
from src.astro_coordinates import *
from src.light_lib import *
from src.engine_lib import *
from src.rand_geom import *

v = np.array([[1, 2, 3], [2, 3, 4]])


class MyTestMethods(unittest.TestCase):
    def assertAlmostVectorEqual(self, v1: np.ndarray, v2: np.ndarray, **kwargs):
        self.assertAlmostEqual(np.sum(vecnorm(v1 - v2)), 0.0, **kwargs)

    def assertAlmostMatrixEqual(self, m1: np.ndarray, m2: np.ndarray, **kwargs):
        self.assertAlmostEqual(np.sum(np.abs((m1 - m2))), 0.0, **kwargs)


class TestAttitude(MyTestMethods):
    def test_template(self):
        self.assertTrue(True)

    def test_rv_quat(self):
        self.assertAlmostVectorEqual(v, quat_to_rv(rv_to_quat(v)))

    def test_mrp_quat(self):
        self.assertAlmostVectorEqual(v, quat_to_mrp(mrp_to_quat(v)))

    def test_identities(self):
        self.assertAlmostMatrixEqual(quat_to_dcm(np.array([[0, 0, 0, 1]])), np.eye(3))
        self.assertAlmostMatrixEqual(
            quat_to_mrp(np.array([[0, 0, 0, 1]])), np.zeros((3,))
        )
        self.assertAlmostMatrixEqual(
            quat_to_rv(np.array([[0, 0, 0, 1]])), np.zeros((3,))
        )

    def test_quat_inv(self):
        q = rv_to_quat(v)
        npo = np.tile([0, 0, 0, 1], (2, 1))
        o = quat_add(q, quat_inv(q))
        self.assertAlmostVectorEqual(o, npo)

    def test_addition(self):
        self.assertAlmostVectorEqual(rv_add(v, -v), np.zeros((2, 3)))
        self.assertAlmostVectorEqual(mrp_add(v, -v), np.zeros((2, 3)))

    def test_ang(self):
        q = rv_to_quat(v)
        self.assertAlmostVectorEqual(
            np.array([1.651890372817769, 1.651890372817769]),
            quat_ang(q[[0, 1], :], q[[1, 0], :]),
        )


class TestMathUtility(MyTestMethods):
    def test_hat(self):
        self.assertAlmostVectorEqual(hat(np.array([0, 0, 0])), np.array([0, 0, 0]))

    def test_cart_sph(self):
        (a, e, r) = cart_to_sph(v[:, 0], v[:, 1], v[:, 2])
        (x, y, z) = sph_to_cart(a, e, r)
        self.assertAlmostVectorEqual(np.squeeze(np.array([[x], [y], [z]]).T), v)

    def test_angle_wrapping(self):
        ls = np.linspace(0, 1e3, int(1e3))
        ls_wrapped = wrap_to_360(ls)
        self.assertAlmostVectorEqual(np.cos(ls_wrapped), np.cos(ls))
        self.assertAlmostVectorEqual(np.sin(ls_wrapped), np.sin(ls))

    def test_unique_rows(self):
        v2 = np.tile(v, (2, 1))
        self.assertEqual(unique_rows(v2).shape[0], 2)

    def test_points_to_plane(self):
        pt = np.array([[1, 0, 0]])
        support = 1 / np.sqrt(2)
        plane_n = np.array([[0, 1, 0]])
        d = points_to_planes(pt, plane_n, support)
        self.assertAlmostEqual(d, support)

    def test_close_egi(self):
        self.assertAlmostVectorEqual(np.sum(close_egi(v)), np.array([0, 0, 0]))

    def test_remove_zero_rows(self):
        v2 = np.vstack((v, np.array([[0, 0, 0], [0, 0, 0]])))
        self.assertEqual(remove_zero_rows(v2).shape[0], 2)

    def test_vecnorm(self):
        vm = vecnorm(v)
        self.assertTupleEqual(vm.shape, (2, 1))


rv = np.array([AstroConstants.earth_r_eq * 1.5, 0, 0, 0, 7.5, 0])
i = np.diag([1, 2, 3])
w = np.array([1, 1, 1])
m = np.array([1, -1, 2])


class TestOrbits(MyTestMethods):
    def test_two_body_acc(self):
        rvdot = two_body_dynamics(rv)
        rvdot_truth = np.array([0, 7.500000000000000, 0, -0.004354789430030, 0, 0])
        self.assertAlmostVectorEqual(rvdot, rvdot_truth)

    def test_j2_acc(self):
        a_j2 = j2_acceleration(rv[:3])
        a_j2_truth = np.array([-0.314307414848805e-5, 0, 0])
        self.assertAlmostVectorEqual(a_j2, a_j2_truth)

    def test_gg_torque(self):
        d = rv_to_dcm(v[0, :])
        irot = np.transpose(d) @ i @ d
        m_gg = gravity_gradient_torque(irot, rv[:3]) * 1e9
        m_gg_truth = np.array([0, -0.700039616569510, -0.068416322484417])
        self.assertAlmostVectorEqual(m_gg, m_gg_truth)

    def test_rotation_dynamics(self):
        m = lambda t: np.array([1, -1, 2])
        wdot = rigid_rotation_dynamics(0, w, i, m)
        wdot_truth = np.array([0, 1 / 2, 1 / 3])
        self.assertAlmostVectorEqual(wdot, wdot_truth)

    def test_quat_kinematics(self):
        q = rv_to_quat(v[0, :])
        qdot = quat_kinematics(q, w)
        qdot_truth = np.array(
            [
                -0.275436493769121,
                0.107546296298775,
                -0.275436493769121,
                -0.765965580135793,
            ]
        )
        self.assertAlmostVectorEqual(qdot, qdot_truth)


obs_lat_geod_rad = np.pi / 5
obs_lon_rad = np.pi / 3
obs_h_km = 10
date = datetime(2022, 12, 9, 12, 23, 34, tzinfo=timezone.utc)


class TestObserver(MyTestMethods):
    def test_lla_to_itrf(self):
        itrf_true = np.array(
            [2.587065541531238e3, 4.480928960442796e3, 3.734060588150026e3]
        )
        itrf = lla_to_itrf(obs_lat_geod_rad, obs_lon_rad, obs_h_km)
        self.assertAlmostVectorEqual(itrf, itrf_true, msg="lla -> itrf wrong")

    def test_lla_to_eci(self):
        eci_true = np.array(
            [4.192331656461110e3, -3.032488705164016e3, 3.734060588150026e3]
        )
        eci = lla_to_eci(obs_lat_geod_rad, obs_lon_rad, obs_h_km, date)
        self.assertAlmostVectorEqual(eci, eci_true, msg="lla -> eci wrong")


class TestCoordinates(MyTestMethods):
    def test_sun(self):
        sun_true = np.array(
            [-0.214815336337292, -0.881911527862012, -0.382300897809063]
        )
        sun_mine = sun(date_to_jd(date)).flatten()
        self.assertAlmostVectorEqual(sun_mine, sun_true, msg="Vallado Sun vector wrong")


class TestTime(MyTestMethods):
    def test_date_to_jd(self):
        jd_true = 2.459923016365741e06
        jd = date_to_jd(date)
        self.assertAlmostEqual(jd, jd_true, msg="Julian date wrong")

    def test_date_to_sidereal(self):
        sid_true = 2.050588850909319e06
        sid = date_to_sidereal(date)
        self.assertAlmostEqual(sid, sid_true, msg="Sidereal time wrong")


class TestEngine(MyTestMethods):
    def test_dir_exists(self):
        lce_dir = os.environ["LCEDIR"]
        self.assertTrue(
            os.path.exists(lce_dir),
            msg="Provided LightCurveEngine directory does not exist!",
        )

    def test_dir_contains_lce(self):
        lce_dir = os.environ["LCEDIR"]
        path_to_c = f"{lce_dir}/LightCurveEngine.c"
        path_to_exec = f"{lce_dir}/LightCurveEngine"
        self.assertTrue(
            os.path.exists(path_to_c),
            msg="LightCurveEngine.c not in provided directory!",
        )
        self.assertTrue(
            os.path.exists(path_to_exec),
            msg="LightCurveEngine executable not in provided directory!",
        )

    def test_engine_run(self):
        engine_res = run_engine(
            Brdf("phong", 0.5, 0.5, 100),
            "cube.obj",
            hat(np.array([[1, 1, 1]])),
            hat(np.array([[1, 1, 1]])),
        )
        self.assertAlmostEqual(engine_res[0], 0.1575300000)


class TestBRDF(MyTestMethods):
    def test_energy_conservation(self):
        self.assertRaises(AssertionError, lambda: Brdf("diffuse", 0.5, 0.6, 100))
        self.assertRaises(AssertionError, lambda: Brdf("diffuse", -0.5, 0.6, 100))
        self.assertRaises(AssertionError, lambda: Brdf("diffuse", 0.5, 0.5, -100))
        self.assertRaises(AssertionError, lambda: Brdf("diffuse", 0, 0, 1))

    def test_bad_brdf_name(self):
        self.assertRaises(AssertionError, lambda: Brdf("diffusee", 0.5, 0.5, 100))


if __name__ == "__main__":
    unittest.main()
