"""Example of k-point mean-field using PySCF.
"""

import numpy as np
from pyscf.pbc import gto, scf
from gdf import CCGDF

cell = gto.Cell()
cell.atom = "He 0 0 0; He 1 1 1"
cell.basis = "6-31g"
cell.a = np.eye(3) * 3
cell.build()

kpts = cell.make_kpts([3, 2, 1])

gdf = CCGDF(cell, kpts)
gdf.build()

mf = scf.KRHF(cell, kpts)
mf.with_df = gdf
mf.kernel()
