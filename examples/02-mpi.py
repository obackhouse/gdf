"""
Example detailing the MPI distribution of the three-center integrals
in the density fitting objects.

Example should be run with different numbers of MPI processes to see
the different distributions.
"""

import numpy as np
from pyscf.pbc import gto
from pyscf.agf2 import mpi_helper
from gdf import CCGDF

cell = gto.Cell()
cell.atom = "He 0 0 0; He 1 1 1"
cell.basis = "6-31g"
cell.a = np.eye(3) * 3
cell.build()

kpts = cell.make_kpts([3, 2, 1])

gdf = CCGDF(cell, kpts)
gdf.build()

# The internal storage format of the three-center integrals is an
# array that is contiguous in all the pairs of k-points calculated
# on that MPI process.
#
# The `mpi_policy` method returns a dictionary that maps the k-point
# pairs to the indices in the array on each process.
#
# Each process only has access to the k-point pairs that it calculated.
# This not only ensures that the CPU time is parallelised efficiently,
# but also that the memory overhead is distributed.
#
# Post-HF methods using these integrals should use the `mpi_policy`
# method to determine which k-point pairs to calculate on each process,
# and treat them accordingly.
#
for i in range(mpi_helper.size):
    if i == mpi_helper.rank:
        print("MPI process %d:" % i)
        print("  3-center integrals: %s" % repr(gdf._cderi.shape))
        print("  k-point pairs for each index:")
        for key, val in gdf.mpi_policy().items():
            print("    %s: %s" % (val, key), flush=True)
    mpi_helper.barrier()
