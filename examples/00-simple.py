"""Simple usage example.
"""

import numpy as np
from pyscf.pbc import gto
from gdf import CCGDF

cell = gto.Cell()
cell.atom = "He 0 0 0; He 1 1 1"
cell.basis = "6-31g"
cell.a = np.eye(3) * 3
cell.build()

kpts = cell.make_kpts([3, 3, 3])

gdf = CCGDF(cell, kpts)
gdf.build()
