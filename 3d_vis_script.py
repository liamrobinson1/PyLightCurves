import pyvista as pv
import numpy as np
from src.object_lib import Object
from src.vis_utility import *
from src.rand_geom import *
from src.math_utility import *

rv = rand_unit_vectors(int(1e7))

mesh = Object("deer.obj")
pl = pv.Plotter()
pl.set_background('black')
# mesh.render(pl)
scatter3(pl, rv, scalars=rv[:,0])
pl.show()