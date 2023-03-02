from src.attitude_lib import *
from src.vis_utility import *
from src.orbit_lib import *
from src.astro_const import AstroConstants

re = AstroConstants.earth_r_eq
(r, v) = coe_to_rv(a=1.5 * re, e=0.1, i=10, Om=100, om=100, ma=10)
rv0 = np.concatenate((r, v))
perts = lambda t, y: np.zeros((6,))
teval = np.linspace(0, 1e4, int(1e3))

q0 = np.array([0, 0, 0, 1])
w0 = np.array([0, 0.01, 0])
i = np.diag([1, 2, 3])
m = lambda t, y: gravity_gradient_torque(i, y)
(r, v, q, w) = integrate_orbit_and_attitude(rv0, q0, w0, i, perts, m, teval)


pl = pv.Plotter()
pl.set_background('black')
scatter3(pl, r, scalars=r[:,0])
plot_earth(pl)
pl.show()
