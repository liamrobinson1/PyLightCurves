from src.attitude_lib import *
from src.vis_utility import *
from src.orbit_lib import *
from src.astro_const import AstroConstants
from src.astro_time_utility import *

# Initial conditions
jd0 = date_to_jd(datetime.now(tz=timezone.utc))
itensor = np.diag([1, 2, 3])
q0 = np.array([0, 0, 0, 1])
w0 = np.array([0, 0, 0])
rv0 = np.concatenate(
    coe_to_rv(a=1.5 * AstroConstants.earth_r_eq, e=0.1, i=10, Om=100, om=100, ma=10)
)

# 3rd body orbital perturbations and body-frame torques
perturbations = lambda t, y: j2_acceleration(y[0:3]) + sun_acceleration(
    y[0:3], t / AstroConstants.earth_sec_in_day + jd0
)
torques = lambda t, y: gravity_gradient_torque(itensor, y)

# Integrate spacecraft state
teval = np.linspace(0, 90 * AstroConstants.earth_sec_in_day, int(1e5))
(r, v, q, w) = integrate_orbit_and_attitude(
    rv0, q0, w0, itensor, perturbations, torques, teval, int_tol=1e-6
)

# Plot resulting trajectory
pl = pv.Plotter()
pl.set_background("black")
plot3(pl, r, scalars="arc_length", line_width=10, cmap="plasma")
plot_earth(pl)
pl.show()
