import scipy
from src.attitude_lib import *
from src.vis_utility import *
from typing import Callable
from src.orbit_lib import *
from src.astro_const import AstroConstants

q0 = np.array([0,0,0,1])
w0 = np.array([1, 1, 1])
i = np.diag([1, 2, 3])
m = lambda t, y: gravity_gradient_torque(i, w0*1e6)
teval = np.linspace(0, 10, int(1e3))

attitude_res = integrate_rigid_attitude_dynamics(q0, w0, i, m, teval)
cdata = vec2cdata(attitude_res[:,6], 'viridis')
new3figure()
fig = scatter3(attitude_res[:,4:], cdata=cdata)
axis('equal')

re = AstroConstants.earth_r_eq
(r,v) = coe_to_rv(a=1.5*re, e=0.1, i=10, Om=100, om=100, ma=10)
rv0 = np.concatenate((r,v))
perts = lambda t, y: np.zeros((6,))
teval = np.linspace(0, 1e4, int(1e3))
orbit_res = integrate_orbit_dynamics(rv0, perts, teval)

cdata = vec2cdata(orbit_res[:,2], 'viridis')
new3figure()
print(orbit_res[:,3:].shape)
fig = scatter3(orbit_res[:,3:], cdata=cdata)
axis('equal')
shg()