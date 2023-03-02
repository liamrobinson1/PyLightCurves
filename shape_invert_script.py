from src.light_lib import Brdf
from src.engine_lib import run_engine
from src.object_lib import Object, optimize_egi, optimize_supports
from src.rand_geom import rand_unit_vectors
import pyvista as pv

# Setup
obj = Object("cube.obj")
brdf = Brdf("phong", cd=0.5, cs=0.5, n=5)

# Simulation
data_points = int(1e3)
sun_vectors_body = rand_unit_vectors(data_points)
obs_vectors_body = rand_unit_vectors(data_points)
lc = run_engine(brdf, obj.file_name, sun_vectors_body, obs_vectors_body)

# Inversion
egi_opt = optimize_egi(lc, sun_vectors_body, obs_vectors_body, brdf)
rec_obj = optimize_supports(egi_opt)

pl = pv.Plotter()
rec_obj.render(pl)
pl.set_background("white")
pl.show()
