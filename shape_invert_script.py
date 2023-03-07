from src.light_lib import Brdf
from src.engine_lib import run_engine
from src.object_lib import *
from src.rand_geom import rand_unit_vectors
from src.math_utility import *
from src.attitude_lib import *
import pyvista as pv
from datetime import datetime, timezone
from src.astro_time_utility import *
from src.astro_coordinates import sun
from src.astro_const import AstroConstants

# Setup
obj = Object("cube.obj")
brdf = Brdf("phong", cd=0.5, cs=0.5, n=5)
w0 = np.array([1, 3, 1])
q0 = hat(np.array([1, -3, 2, 1]))
itensor = np.diag([1, 2, 3])
data_points = int(1e3)
teval = np.linspace(0, 100, data_points)
(q, _) = propagate_attitude_torque_free(q0, w0, itensor, teval)
jd0 = jd_now()
sun_inertial = sun(jd0 + teval / AstroConstants.earth_sec_in_day)
obs_inertial = rand_unit_vectors(teval.size)

sun_body = np.zeros_like(sun_inertial)
obs_body = np.zeros_like(sun_inertial)
for i in range(teval.size):
    dcm = quat_to_dcm(q[i, :])
    sun_body[i, :] = dcm @ sun_inertial[i, :].T
    obs_body[i, :] = dcm @ obs_inertial[i, :].T

# Simulation
lc = run_engine(brdf, obj.file_name, sun_body, obs_body)

# Inversion
egi_opt = optimize_egi(lc, sun_body, obs_body, brdf)
rec_obj = optimize_supports(egi_opt)

pl = pv.Plotter()
rec_obj.render(pl)
pl.set_background("white")
pl.show()
