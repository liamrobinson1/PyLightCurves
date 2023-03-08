import numpy as np
from src.math_utility import *
from scipy.special import *
from src.attitude_lib import *
from src.vis_utility import *
from src.profile_utility import *
from src.rand_geom import *

w0 = np.array([1,1,0])
q0 = hat(np.array([1, -3, 2, 1]))
itensor = np.diag([3,2,1])
teval = np.linspace(0, 100, int(1e2))

tic()
(quat_num, omega_num) = integrate_rigid_attitude_dynamics(q0, w0, itensor, teval)
t_num = toc()
tic()
(quat_anal, omega_anal) = propagate_attitude_torque_free(q0, w0, itensor, teval)
t_anal = toc()

print(f"Analytical time is {t_num / t_anal:.2f}x faster!")

rv_anal = quat_to_rv(quat_upper_hemisphere(quat_anal))
rv_num = quat_to_rv(quat_upper_hemisphere(quat_num))
print(np.max(vecnorm(rv_anal - rv_num)))

# pl = pv.Plotter(shape=(1, 3))
# pl.subplot(0, 0)
# plot3(pl, rv_anal, scalars="arc_length", line_width=10, cmap="plasma")
# plot3(pl, rv_num, scalars="arc_length", line_width=10)
# pl.show_bounds(location="all")
# pl.subplot(0, 1)
# scatter3(pl, omega_anal - omega_num)
# pl.show_bounds(location="all")
# pl.subplot(0, 2)
# scatter3(pl, quat_anal[:, :3] - quat_num[:, :3])
# pl.show_bounds(location="all")
# pl.show()
