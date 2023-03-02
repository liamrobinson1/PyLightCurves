from src.light_lib import Brdf
from src.engine_lib import run_engine
from src.object_lib import Object
from src.rand_geom import spiral_sample_sphere
from src.vis_utility import *
from src.math_utility import *
import pyvista as pv
import numpy as np

# Setup
obj = Object("cube.obj")
brdf = Brdf("phong", cd=0.5, cs=0.5, n=5)

# Simulation
data_points = int(1e3)
t = np.reshape(np.linspace(0, 10, data_points), (data_points, 1))
sun_vectors_body = hat(np.hstack((np.sin(t), np.cos(t), t)))
obs_vectors_body = hat(np.hstack((np.cos(t), np.sin(t), t)))
lc = run_engine(brdf, obj.file_name, sun_vectors_body, obs_vectors_body)

ch = pv.Chart2D()
ch.line(t, lc, width=3, label="", style="-")
texit(ch, "Cube Light Curve Test", "Time [s]", "Normalized Irradiance")

show_and_copy(ch)
