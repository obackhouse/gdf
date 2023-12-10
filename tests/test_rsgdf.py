"""Tests for `gdf/rsgdf.py`.
"""

import unittest

import numpy as np
from pyscf.agf2 import mpi_helper
from pyscf.pbc import df as pyscf_df
from pyscf.pbc import gto, scf

from gdf import RSGDF


class TestRSGDF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 0 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.verbose = 0
        cell.precision = 1e-14
        cell.build()

        kpts = cell.make_kpts([3, 2, 1])

        df_ref = pyscf_df.RSGDF(cell, kpts)
        df_ref.auxbasis = "weigend"
        df_ref._prefer_ccdf = False
        df_ref.build()

        df = RSGDF(cell, kpts, auxbasis="weigend")
        df.build()

        self.df_ref, self.df, self.kpts = df_ref, df, kpts

    @classmethod
    def tearDownClass(self):
        del self.df_ref, self.df, self.kpts

    def test_sr_loop(self):
        policy = self.df.mpi_policy()

        for ki, kj, kk in self.df.kpts.loop(3):
            kl = self.df.kpts.conserve(ki, kj, kk)

            kpt_ij = self.kpts[[ki, kj]]
            kpt_kl = self.kpts[[kk, kl]]

            r1, i1, _ = list(self.df_ref.sr_loop(kpt_ij, compact=False))[0]
            v1 = r1 + i1 * 1j
            r1, i1, _ = list(self.df_ref.sr_loop(kpt_kl, compact=False))[0]
            u1 = r1 + i1 * 1j
            eri1 = np.dot(v1.T, u1)

            if (ki, kj) in policy:
                r2, i2, _ = list(self.df.sr_loop(kpt_ij, compact=False))[0]
                v2 = r2 + i2 * 1j
            else:
                v2 = np.zeros_like(v1)
            v2 = mpi_helper.allreduce(v2)
            if (kk, kl) in policy:
                r2, i2, _ = list(self.df.sr_loop(kpt_kl, compact=False))[0]
                u2 = r2 + i2 * 1j
            else:
                u2 = np.zeros_like(u1)
            u2 = mpi_helper.allreduce(u2)
            eri2 = np.dot(v2.T, u2)

            self.assertAlmostEqual(np.max(np.abs(eri1 - eri2)), 0, 7)

    def test_get_jk(self):
        dm = np.random.random((self.df.nkpts, self.df.nao, self.df.nao))
        dm = dm + np.random.random((self.df.nkpts, self.df.nao, self.df.nao)) * 1.0j
        dm = dm + dm.swapaxes(1, 2).conj()
        dm = mpi_helper.allreduce(dm) / self.df.nkpts

        j1, k1 = self.df_ref.get_jk(dm)
        j2, k2 = self.df.get_jk(dm)

        self.assertAlmostEqual(np.max(np.abs(j1 - j2)), 0, 7)
        self.assertAlmostEqual(np.max(np.abs(k1 - k2)), 0, 7)

    def test_pyscf_mf(self):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 0 1"
        cell.a = np.eye(3) * 3
        cell.basis = "6-31g"
        cell.verbose = 0
        cell.build()

        kpts = cell.make_kpts([3, 2, 1])

        df = RSGDF(cell, kpts)
        df.build()

        mf1 = scf.KRHF(cell, kpts)
        mf1 = mf1.rs_density_fit(auxbasis="weigend")
        mf1.with_df._prefer_ccdf = False
        mf1.kernel()

        mf2 = scf.KRHF(cell, kpts)
        mf2.with_df = df
        mf2.kernel()

        self.assertAlmostEqual(mf1.e_tot, mf2.e_tot, 8)


class TestRSGDF_exp_to_discard(TestRSGDF):
    @classmethod
    def setUpClass(self):
        cell = gto.Cell()
        cell.atom = "Si 0 0 0; Si {r} {r} {r}".format(r=5.31/2)
        cell.basis = "gth-szv"
        cell.pseudo = "gth-pade"
        cell.a = (np.ones((3, 3)) - np.eye(3)) * 5.31
        cell.verbose = 0
        cell.precision = 1e-14
        cell.exp_to_discard = 0.1
        cell.build()

        kpts = cell.make_kpts([2, 1, 1])

        df_ref = pyscf_df.RSGDF(cell, kpts)
        df_ref.auxbasis = "weigend"
        df_ref._prefer_ccdf = False
        df_ref.build()

        df = RSGDF(cell, kpts, auxbasis="weigend")
        df.build()

        self.df_ref, self.df, self.kpts = df_ref, df, kpts



if __name__ == "__main__":
    unittest.main()
