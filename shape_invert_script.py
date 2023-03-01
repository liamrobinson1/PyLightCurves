from src.attitude_lib import *
from src.orbit_lib import *
from src.vis_utility import *
from src.astro_time_utility import *
from src.observer_utility import *
from src.astro_coordinates import *
from src.profile_utility import *
from src.astro_const import *
from src.light_lib import *
import scipy

p = np.array([[1, 2, 3], [2, 3, 4]])
q = rv_to_quat(p)

q0 = q[0, :]
w0 = np.array([1, 1, 1])
i = np.diag([1, 2, 3])
m = np.array([1, -1, 2])

fun = lambda t, y: np.concatenate(
    (quat_kinematics(y[:4], y[4:]), rigid_rotation_dynamics(i, y[4:], m))
)
tspan = [0, 10]
teval = np.linspace(tspan[0], tspan[1], int(1e3))
y0 = np.concatenate((q0, w0))
tol = 1e-13
ode_res = scipy.integrate.solve_ivp(fun, tspan, y0, t_eval=teval, rtol=tol, atol=tol)

# cdata = vec2cdata(ode_res.y[2,:], 'viridis')
# print(cdata)
# new3figure()
# fig = scatter3(ode_res.y[4:,:], cdata=cdata)
# shg()


a = 1e4
e = 0.5
i = 40
Om = 40
om = 55
ma = 191
(r, v) = rv_to_coe(a, e, i, Om, om, ma)

cd = 0.5
cs = 0.5
n = 5

from src.object_lib import *

obj = Object("models/cube.obj")

from src.rand_geom import *

ns = int(1e3)
o = rand_cone_vectors(np.array([0, 0, 1]), np.pi, ns)
normals = np.tile([0, 0, 1], (ns, 1))
l = rand_cone_vectors(np.array([0, 0, 1]), np.pi, ns)

b = Brdf("phong", cd, cs, n)
fr = b.eval(l, o, normals)
g = b.compute_reflection_matrix(l, o, obj.face_normals)
lc = g @ obj.face_areas

# Inverting EGI
ns = int(1e3)
normal_candidates = rand_cone_vectors(np.array([0, 0, 1]), np.pi, ns)
g_candidates = b.compute_reflection_matrix(l, o, normal_candidates)
a_candidates = np.expand_dims(
    scipy.optimize.nnls(g_candidates, lc.flatten())[0], axis=1
)
egi_candidate = normal_candidates * a_candidates
egi_candidate = remove_zero_rows(egi_candidate)
egi_candidate = merge_clusters(egi_candidate, np.pi / 10, miter=1)
egi_candidate = close_egi(egi_candidate)

# Optimizing supports
rec_obj = optimize_supports(egi_candidate)
new3figure()
rec_obj.render()
axis("equal")
shg()
