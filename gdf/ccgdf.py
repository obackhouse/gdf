"""Charge-compensated Gaussian density fitting.
"""

import numpy as np
import scipy.linalg
import copy
import ctypes

from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.ao2mo.outcore import balance_partition
from pyscf.pbc.tools import pbc
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.df.aft import weighted_coulG
from pyscf.pbc.df.ft_ao import ft_ao, ft_aopair_kpts
from pyscf.pbc.df.incore import aux_e2
from pyscf.pbc.df.gdf_builder import _guess_eta, estimate_ke_cutoff_for_eta, auxbar
from pyscf.pbc.df.rsdf_builder import _estimate_meshz

from gdf.base import BaseGDF
from gdf.cell import make_auxcell, make_chgcell, fuse_auxcell_chgcell

libpbc = lib.load_library("libpbc")


class CCGDF(BaseGDF):
    __doc__ = BaseGDF.__doc__.format(
        description="Charge-compensated Gaussian density fitting.",
        extra_attributes=(
            "eta : float"
            "        Charge compensation parameter. If `None`, determine from",
            "        `cell.precision`. Default value is `None`."
        )
    )

    # Extra attributes:
    eta = None

    _attributes = BaseGDF._attributes | {"eta"}


    def get_mesh_parameters(self, cell=None, auxcell=None, eta=None, mesh=None, precision=None):
        """
        Return the mesh, the η parameter, and the kinetic energy cutoff.
        Whether these are estimated from `cell.precision` depends on the
        attributes of the object.

        Parameters
        ----------
        cell : pyscf.pbc.gto.Cell, optional
            Cell object. If `None`, use `self.cell`. Default value is
            `None`.
        auxcell : pyscf.pbc.gto.Cell, optional
            Auxiliary cell object. Only required if `self.eta` is
            `None`. Default value is `None`.
        eta : float, optional
            Charge compensation parameter. If `None`, use `self.eta`.
            Default value is `None`.
        mesh : tuple of int, optional
            Mesh size along each direction. If `None`, use `self.mesh`.
            Default value is `None`.
        precision : float, optional
            Precision of the calculation. If `None`, use
            `cell.precision`. Default value is `None`.

        Returns
        -------
        mesh : tuple of int
            Mesh size along each direction.
        eta : float
            Charge compensation parameter.
        ke_cutoff : float
            Kinetic energy cutoff.
        """

        cell = cell if cell is not None else self.cell
        eta = eta if eta is not None else self.eta
        mesh = mesh if mesh is not None else self.mesh
        ke_cutoff = None

        if eta is None:
            eta, mesh, ke_cuttoff = _guess_eta(auxcell, kpts=self.kpts._kpts, mesh=mesh)

        if mesh is None:
            ke_cutoff = estimate_ke_cutoff_for_eta(cell, eta, precision=precision)
            mesh = cell.cutoff_to_mesh(ke_cutoff)

        if ke_cutoff is None:
            ke_cutoff = pbc.mesh_to_cutoff(cell.lattice_vectors(), mesh)
            ke_cutoff = np.min(ke_cutoff)

        if cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
            mesh[2] = _estimate_meshz(cell)
        elif cell.dimension < 2:
            mesh[cell.dimension:] = cell.mesh[cell.dimension:]

        mesh = cell.symmetrize_mesh(mesh)

        return mesh, eta, ke_cutoff


    def get_qpts(self, return_map=False, time_reversal_symmetry=False):
        """Get the q-points corresponding to the k-points.

        Parameters
        ----------
        return_map : bool, optional
            Whether to return the map between the q-points and the
            k-points. Default value is `False`.
        time_reversal_symmetry : bool, optional
            Whether to return the time-reversal symmetry status of the
            q-points.

        Returns
        -------
        qpts : KPoints
            The q-points.
        qmap : list of list of tuple of int
            The map between the q-points and the k-points. Only returned
            if `return_map` is `True`.
        conj : list of bool
            The time-reversal symmetry status of the q-points. Only
            returned if `time_reversal_symmetry` is `True`.
        """

        qpts = []
        qmap = []
        conj = []
        for qpt, ki, kj, cc in kpts_helper.kk_adapted_iter(self.cell, self.kpts._kpts, time_reversal_symmetry=True):
            qpts.append(-qpt)  # sign difference c.f. PySCF
            qmap.append(list(zip(ki, kj)))
            conj.append(cc)
            if not time_reversal_symmetry and not cc:
                qpts.append(qpt)
                qmap.append(list(zip(kj, ki)))

        qpts = self.kpts.__class__(self.cell, np.array(qpts))

        out = [qpts]
        if return_map:
            out.append(qmap)
        if time_reversal_symmetry:
            out.append(conj)

        return tuple(out)


    def build_int2c2e(self, fused_cell):
        """Build the bare 2-center 2-electron integrals.

        Parameters
        ----------
        fused_cell : pyscf.pbc.gto.Cell
            Fused cell.

        Returns
        -------
        int2c2e : list of ndarray
            List of 2-center 2-electron integrals for each q-point.
        """

        cput0 = (logger.process_clock(), logger.perf_counter())
        qpts, _ = self.get_qpts(time_reversal_symmetry=True)

        int2c2e = list(fused_cell.pbc_intor("int2c2e", hermi=0, kpts=-qpts._kpts))

        cput1 = logger.timer(self, "bare 2c2e", *cput0)

        return int2c2e


    def build_j2c(
        self,
        auxcell,
        fused_cell,
        fuse,
        eta=None,
    ):
        """Build the 2-center Coulomb integrals.

        Parameters
        ----------
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary cell.
        fused_cell : pyscf.pbc.gto.Cell
            Fused cell.
        fuse : callable
            Function to fuse the auxiliary cell with the fused cell.
        eta : float, optional
            Charge compensation parameter. If `None`, use `self.eta`.
            Default value is `None`.

        Returns
        -------
        j2c : list of ndarray
            List of 2-center Coulomb integrals for each q-point.
        """

        cput0 = (logger.process_clock(), logger.perf_counter())
        qpts, _ = self.get_qpts(time_reversal_symmetry=True)
        naux = auxcell.nao_nr()
        if auxcell.dimension == 0:
            return [auxcell.intor("int2c2e", hermi=1)]

        # Get the bare 2c2e integrals (eq. 32, first term)
        j2c = self.build_int2c2e(fused_cell)

        # The 2c2e integrals are sensitive to the mesh. Use a different
        # mesh here to achieve desired accuracy
        mesh, eta, ke_cutoff = self.get_mesh_parameters(
            cell=auxcell,
            eta=eta,
            precision=auxcell.precision ** 2,
        )
        logger.debug(self, "Using mesh %s for 2c2e integrals", mesh)

        # Get the G vectors
        vG, vGbase, _ = fused_cell.get_Gv_weights(mesh)
        ngrids = vG.shape[0]
        grid = lib.cartesian_prod([np.arange(len(x)) for x in vGbase])
        reciprocal_vectors = fused_cell.reciprocal_vectors()

        # Build the 2-center Coulomb integrals (eq. 32)
        for q in qpts.loop(1):
            cput1 = (logger.process_clock(), logger.perf_counter())

            G_chg = ft_ao(
                fused_cell,
                vG,
                b=reciprocal_vectors,
                gxyz=grid,
                Gvbase=vGbase,
                kpt=-qpts[q],
            ).T
            G_aux = G_chg[naux:] * weighted_coulG(self, -qpts[q], mesh=mesh)

            # Eq. 32 final three terms:
            j2c_comp = np.dot(G_aux.conj(), G_chg.T)
            if qpts.is_zero(qpts[q]):
                j2c_comp = j2c_comp.real
            j2c[q][naux:] -= j2c_comp
            j2c[q][:naux, naux:] = j2c[q][naux:, :naux].T.conj()

            j2c[q] = (j2c[q] + j2c[q].T.conj()) * 0.5
            j2c[q] = fuse(fuse(j2c[q], axis=1), axis=0)

            del G_chg, G_aux, j2c_comp

            logger.timer_debug1(self, "j2c [%d]" % q, *cput1)

        logger.timer_debug1(self, "j2c", *cput0)

        return j2c


    def build_j2c_chol(self, j2c, q, threshold=None):
        """
        Get the inverse Cholesky factorization of the 2-center Coulomb
        integral at a single q-point.

        Parameters
        ----------
        j2c : list of ndarray
            List of 2-center Coulomb integrals for each q-point.
        q : int
            Index of the q-point.
        threshold : float, optional
            Threshold for linear dependence. If `None`, use
            `self.linear_dep_threshold`. Default value is `None`.

        Returns
        -------
        j2c_chol : ndarray
            Inverse Cholesky factorization of the 2-center Coulomb
            integral.
        """

        if threshold is None:
            threshold = self.linear_dep_threshold

        w, v = scipy.linalg.eigh(j2c[q])

        mask = w > threshold
        w = w[mask]
        v = v[:, mask]

        j2c_chol = np.dot(v * (w ** -0.5)[None], v.T.conj())

        logger.debug1(
            self,
            "j2c [%d] condition number: %.3g, dropped %d auxiliary functions",
            q,
            np.max(w) / np.min(w),
            np.sum(~mask),
        )

        return j2c_chol


    def build_int3c2e(self, fused_cell):
        """Build the bare 3-center 2-electron integrals.

        Returns
        -------
        int3c2e : dict of ndarray
            Bare 3-center 2-electron integrals for each k-point pair.
        """

        cput0 = (logger.process_clock(), logger.perf_counter())
        ngrids = fused_cell.nao_nr()
        aux_loc = fused_cell.ao_loc_nr(fused_cell.cart)

        # Get the k-point pairs
        kpts = self.kpts
        kpt_pairs_idx = [(ki, kj) for ki in range(len(kpts)) for kj in range(len(kpts))]
        kpt_pairs = np.array([(kpts[ki], kpts[kj]) for ki, kj in kpt_pairs_idx])

        # Initialise arrays
        int3c2e = {idx: np.zeros((ngrids, self.nao_pair), dtype=np.complex128) for idx in kpt_pairs_idx}

        for p0, p1 in lib.prange(0, fused_cell.nbas, fused_cell.nbas):  # TODO MPI
            cput1 = (logger.process_clock(), logger.perf_counter())

            shls_slice = (0, self.cell.nbas, 0, self.cell.nbas, p0, p1)
            q0, q1 = aux_loc[p0], aux_loc[p1]

            int3c2e_part = aux_e2(
                self.cell,
                fused_cell,
                "int3c2e",
                aosym="s2",
                kptij_lst=kpt_pairs,
                shls_slice=shls_slice,
            )
            int3c2e_part = lib.transpose(int3c2e_part, axes=(0, 2, 1))
            int3c2e_part = int3c2e_part.reshape(-1, q1 - q0, self.nao_pair)

            for k, (ki, kj) in enumerate(kpt_pairs_idx):
                int3c2e[ki, kj][q0:q1] = int3c2e_part[k]

            logger.timer_debug1(self, "int3c2e [%d:%d]" % (p0, p1), *cput1)

        logger.timer_debug1(self, "int3c2e", *cput0)

        return int3c2e


    def build_j3c(
        self,
        auxcell,
        fused_cell,
        fuse,
        j2c,
        mesh=None,
        ke_cutoff=None,
    ):
        """Build the 3-center Coulomb integrals.

        Parameters
        ----------
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary cell.
        fused_cell : pyscf.pbc.gto.Cell
            Fused cell.
        fuse : callable
            Function to fuse the auxiliary cell with the fused cell.
        j2c : list of ndarray
            List of 2-center Coulomb integrals for each q-point.
        mesh : tuple of int, optional
            Mesh size along each direction. If `None`, use `self.mesh`.
            Default value is `None`.
        ke_cutoff : float, optional
            Cutoff for the kinetic energy. If `None`, use
            `self.ke_cutoff`. Default value is `None`.

        Returns
        -------
        j3c : list of ndarray
            Array of 3-center Coulomb integrals for each k-point pair.
        """

        cput0 = (logger.process_clock(), logger.perf_counter())

        kpts = self.kpts
        qpts, qpt_idx, conj = self.get_qpts(return_map=True, time_reversal_symmetry=True)
        naux = auxcell.nao_nr()

        # Get the G vectors
        vG, vGbase, _ = fused_cell.get_Gv_weights(mesh)
        ngrids = np.prod(mesh)
        grid = lib.cartesian_prod([np.arange(len(x)) for x in vGbase])
        reciprocal_vectors = cell.reciprocal_vectors()

        # Get the bare 3c2e integrals (eq. 31, first term)
        int3c2e = self.build_int3c2e(fused_cell)

        # Initialise arrays
        j3c_tri = {}

        for q in qpts.loop(1):
            # Take the inverse Cholesky factorisation of the 2-center
            # interaction
            j2c_chol = self.build_j2c_chol(j2c, q)

            # Handle the time-reversal symmetry - if the q-point is
            # self-conjugate, conj[q] is `True`, otherwise it is
            # `False`. The loop is therefore over [0] or [0, 1],
            # respectively. This index can then be used to swap the
            # corresponding k-points and the sign of the q-point.
            for time_reversal in range((not conj[q]) + 1):
                kis = [tup[time_reversal] for tup in qpt_idx[q]]
                kjs = [tup[1 - time_reversal] for tup in qpt_idx[q]]
                qpt = -qpts[q] if time_reversal == 0 else qpts[q]
                if time_reversal:
                    j2c_chol = j2c_chol.conj()

                # Eq. 33
                shls_slice = (auxcell.nbas, fused_cell.nbas)
                G_chg = ft_ao(
                    fused_cell,
                    vG,
                    shls_slice=shls_slice,
                    b=reciprocal_vectors,
                    gxyz=grid,
                    Gvbase=vGbase,
                    kpt=qpt,
                )
                G_chg *= weighted_coulG(self, qpt, mesh=mesh).ravel()[:, None]
                logger.debug1(self, "Norm of FT for fused cell: %.6g", np.linalg.norm(G_chg))

                # Eq. 26
                if qpts.is_zero(qpt):
                    logger.debug(self, "Including net charge of AO products")
                    vbar = fuse(auxbar(fused_cell))
                    ovlp = self.cell.pbc_intor("int1e_ovlp", hermi=0, kpts=kpts[kis])
                    ovlp = [lib.pack_tril(s) for s in ovlp]

                # Eq. 24
                p0, p1, pn = balance_partition(self.cell.ao_loc_nr() * self.nao, self.nao ** 2)[0]
                shls_slice = (p0, p1, 0, self.cell.nbas)
                G_ao = ft_aopair_kpts(
                    cell,
                    vG,
                    shls_slice=shls_slice,
                    b=reciprocal_vectors,
                    aosym="s2",
                    gxyz=grid,
                    Gvbase=vGbase,
                    q=qpt,
                    kptjs=kpts[kjs],
                )
                G_ao = G_ao.reshape(-1, ngrids, self.nao_pair)
                logger.debug1(self, "Norm of FT for AO cell: %.6g", np.linalg.norm(G_ao))

                for i, (ki, kj) in enumerate(zip(kis, kjs)):
                    # Eq. 31, first term
                    v = int3c2e[ki, kj].copy()  # FIXME copy needed?

                    # Eq. 31, second term
                    if qpts.is_zero(qpt):
                        mask = np.where(vbar != 0)[0]
                        v[mask] -= lib.einsum("i,j->ij", vbar[mask], ovlp[i])

                    # Eq. 31, third term
                    v[naux:] -= np.dot(G_chg.T.conj(), G_ao[i])

                    # Fused auxiliary and charge compensating parts
                    v = fuse(v)

                    # Eq. 29
                    v = np.dot(j2c_chol, v)

                    j3c_tri[ki, kj] = v + j3c_tri.get((ki, kj), 0.0)
                    logger.debug(self, "Filled j3c for kpt [%d, %d]", ki, kj)

        # Unpack the three-center integrals
        j3c = {}
        for ki, kj in self.kpts.loop(2):
            out = np.zeros((naux, self.nao, self.nao), dtype=np.complex128)
            libpbc.PBCunpack_tril_triu(
                out.ctypes.data_as(ctypes.c_void_p),
                j3c_tri[ki, kj].ctypes.data_as(ctypes.c_void_p),
                j3c_tri[kj, ki].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(self.nao),
            )
            j3c[ki, kj] = out

        logger.timer(self, "j3c", *cput0)

        return j3c


    def _build(self):
        """Build the density fitting integrals.
        """

        # Build the auxiliary cell
        auxcell = make_auxcell(
            self.cell,
            self.auxbasis,
            exp_to_discard=self.exp_to_discard,
        )

        # Get the parameters or estimate using the cell precision
        mesh, eta, ke_cutoff = self.get_mesh_parameters(auxcell=auxcell)
        logger.info(self, "mesh = %s", mesh)
        logger.info(self, "eta = %.10f", eta)
        logger.info(self, "ke_cutoff = %.10f", ke_cutoff)

        # Build the charge compensating cell
        chgcell = make_chgcell(auxcell, eta)

        # Build the fused cell
        fused_cell, fuse = fuse_auxcell_chgcell(auxcell, chgcell)

        # Get the 2-center Coulomb integrals
        j2c = self.build_j2c(auxcell, fused_cell, fuse, eta=eta)

        # Get the 3-center Coulomb integrals
        j3c = self.build_j3c(auxcell, fused_cell, fuse, j2c, mesh=mesh)

        return j3c


    @property
    def direct_scf_tol(self):
        exp_min = np.min(np.hstack(self.cell.bas_exps()))
        lattice_sum_factor = max((2 * self.cell.rcut) ** 3 / self.cell.vol / exp_min, 1)
        cutoff = self.cell.precision / lattice_sum_factor * 0.1
        return cutoff


if __name__ == "__main__":
    import pyscf.pbc

    cell = pyscf.pbc.gto.Cell()
    cell.atom = "He 0 0 0; He 1 1 1"
    cell.basis = "6-31g"
    cell.a = np.eye(3) * 3
    cell.verbose = 3
    cell.precision = 1e-14
    cell.build()

    kpts = cell.make_kpts([3, 2, 1])

    df1 = pyscf.pbc.df.DF(cell, kpts=kpts)
    df1.auxbasis = "weigend"
    df1._prefer_ccdf = True
    df1.build()

    df2 = CCGDF(cell, kpts, auxbasis="weigend")
    df2.build()

    import itertools
    #for ki, kj in itertools.product(range(len(kpts)), repeat=2):
    #    kpt = kpts[[ki, kj]]

    #    r1, i1, _ = list(df1.sr_loop(kpt, compact=False))[0]
    #    v1 = r1 + i1 * 1j
    #    eri1 = np.dot(v1.T, v1)

    #    r2, i2, _ = list(df2.sr_loop(kpt, compact=False))[0]
    #    v2 = r2 + i2 * 1j
    #    eri2 = np.dot(v2.T, v2)

    #    print(
    #        ki, kj,
    #        np.max(np.abs(eri1 - eri2)) < 1e-6,
    #        np.max(np.abs(v1) - np.abs(v2)) < 1e-6,
    #        np.max(np.abs(v1 - v2)) < 1e-6,
    #        np.max(np.abs(eri1 - eri2)),
    #    )
    #    #print([np.linalg.norm(np.abs(v1[i]) - np.abs(v2[i])) for i in range(len(v1))])
    #    if (ki, kj) == (0, 2) and 0:
    #        for x in range(len(v1)):
    #            if not np.allclose(v1[x], v2[x]):
    #                print(x)
    #                print(v1[x].reshape(4, 4).real)
    #                print(v2[x].reshape(4, 4).real)

    from pyscf.pbc.lib import kpts_helper
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    for ki, kj, kk in itertools.product(range(len(kpts)), repeat=3):
        kl = kconserv[ki, kj, kk]
        kpt_ij = kpts[[ki, kj]]
        kpt_kl = kpts[[kk, kl]]

        r1, i1, _ = list(df1.sr_loop(kpt_ij, compact=False))[0]
        v1 = r1 + i1 * 1j
        r1, i1, _ = list(df1.sr_loop(kpt_kl, compact=False))[0]
        u1 = r1 + i1 * 1j
        eri1 = np.dot(v1.T, u1)

        r2, i2, _ = list(df2.sr_loop(kpt_ij, compact=False))[0]
        v2 = r2 + i2 * 1j
        r2, i2, _ = list(df2.sr_loop(kpt_kl, compact=False))[0]
        u2 = r2 + i2 * 1j
        eri2 = np.dot(v2.T, u2)

        print(
            "%d %d %d %d %5s %.4g" % (
                ki, kj, kk, kl,
                np.max(np.abs(eri1 - eri2)) < 1e-6,
                np.max(np.abs(eri1 - eri2)),
            )
        )
