import numpy as np
from typing import Tuple
from src.math_utility import *
from scipy.special import *
from src.attitude_lib import *
from src.vis_utility import *
from src.profile_utility import *
from src.rand_geom import *

w0 = np.array([-1, 1, -3])
q0 = hat(np.array([1, -3, 2, 1]))
itensor = np.diag([1, 2, 3])
teval = np.linspace(0, 10, int(1e3))

tic()
(quat_num, omega_num) = integrate_rigid_attitude_dynamics(q0, w0, itensor, teval)
t_num = toc()
tic()
(quat_anal, omega_anal) = propagate_attitude_torque_free(q0, w0, itensor, teval)
t_anal = toc()

print(f"Analytical time is {t_num / t_anal:.2f}x faster!")

# rv_anal = quat_to_rv(quat_upper_hemisphere(quat_anal))
# rv_num = quat_to_rv(quat_upper_hemisphere(quat_num))
# pl = pv.Plotter()
# plot3(pl, rv_anal, color="red", line_width=10)
# plot3(pl, rv_num, color="white", line_width=10)
# scatter3(pl, rv_anal-rv_num)
# pl.show_bounds(location='all')
# pl.show()

teval = np.linspace(0, 1000, int(1e6))
import cProfile

fun_call = """(quat_anal, omega_anal) = propagate_attitude_torque_free(q0, w0, itensor, teval)"""
cProfile.run(fun_call, sort=1)
