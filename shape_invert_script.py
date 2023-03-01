from src.vis_utility import *
from src.profile_utility import tic, toc
from src.light_lib import Brdf
from src.engine_lib import run_engine
from src.object_lib import Object, optimize_egi, optimize_supports
from src.rand_geom import rand_unit_vectors

obj = Object("gem.obj")

data_points = int(1e3)
sun_vectors_body = rand_unit_vectors(data_points)
obs_vectors_body = rand_unit_vectors(data_points)

brdf = Brdf("phong", cd=0.5, cs=0.5, n=5)
lc = run_engine(brdf, obj.file_name, sun_vectors_body, obs_vectors_body)

egi_opt = optimize_egi(lc, sun_vectors_body, obs_vectors_body, brdf)
rec_obj = optimize_supports(egi_opt)

new3figure()
rec_obj.render()
axis("equal")
shg()
