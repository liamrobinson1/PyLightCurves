from src.engine_lib import *
import numpy as np
from src.rand_geom import *

model_file = "cube.obj"
b = Brdf("phong", 0.5, 0.5, 100)
b.name = "oren-nayar"
brdf_ind = query_brdf_registry(b)
print(brdf_ind)

n = int(1e3)
svb = rand_unit_vectors(n)
ovb = rand_unit_vectors(n)
engine_res = run_engine(b, model_file, svb, ovb)
print(engine_res)
print_all_registered_brdfs()
