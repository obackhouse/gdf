"""Charge-compensated Gaussian density fitting.

Ref: J. Chem. Phys. 147, 164119 (2017)
"""

import copy
import ctypes

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.ao2mo.outcore import balance_partition
from pyscf.lib import logger
from pyscf.pbc.df.aft import weighted_coulG
from pyscf.pbc.df.ft_ao import ft_ao, ft_aopair_kpts
from pyscf.pbc.df.gdf_builder import _guess_eta, auxbar, estimate_ke_cutoff_for_eta
from pyscf.pbc.df.incore import aux_e2
from pyscf.pbc.df.rsdf_builder import _estimate_meshz
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools import pbc
from pyscf.scf import _vhf

from gdf.base import BaseGDF
from gdf.cell import fuse_auxcell_chgcell, make_auxcell, make_chgcell

libpbc = lib.load_library("libpbc")


class CCGDF(BaseGDF):
    __doc__ = BaseGDF.__doc__.format(
        description="Charge-compensated Gaussian density fitting.",
        extra_attributes=(
            "eta : float" "        Charge compensation parameter. If `None`, determine from",
            "        `cell.precision`. Default value is `None`.",
        ),
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
            mesh[cell.dimension :] = cell.mesh[cell.dimension :]

        mesh = cell.symmetrize_mesh(mesh)

        return mesh, eta, ke_cutoff

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
        int2c2e = self.build_int2c2e(fused_cell)

        # The 2c2e integrals are sensitive to the mesh. Use a different
        # mesh here to achieve desired accuracy
        mesh, eta, ke_cutoff = self.get_mesh_parameters(
            cell=auxcell,
            eta=eta,
            precision=auxcell.precision**2,
        )
        logger.debug(self, "Using mesh %s for 2c2e integrals", mesh)

        # Get the G vectors
        vG, vGbase, _ = fused_cell.get_Gv_weights(mesh)
        ngrids = vG.shape[0]
        grid = lib.cartesian_prod([np.arange(len(x)) for x in vGbase])
        reciprocal_vectors = fused_cell.reciprocal_vectors()

        # Initialise arrays
        j2c = [np.zeros((naux, naux), dtype=int2c2e_part.dtype) for int2c2e_part in int2c2e]

        # Build the 2-center Coulomb integrals
        for q in qpts.loop(1, mpi=True):
            cput1 = (logger.process_clock(), logger.perf_counter())

            # Eq. 32, first term
            v = int2c2e[q]

            G_chg = ft_ao(
                fused_cell,
                vG,
                b=reciprocal_vectors,
                gxyz=grid,
                Gvbase=vGbase,
                kpt=-qpts[q],
            ).T
            G_aux = G_chg[naux:] * weighted_coulG(self, -qpts[q], mesh=mesh)

            # Eq. 32, final three terms
            tmp = np.dot(G_aux.conj(), G_chg.T)
            if qpts.is_zero(qpts[q]):
                tmp = tmp.real
            v[naux:] -= tmp
            v[:naux, naux:] = v[naux:, :naux].T.conj()

            # Hermitise
            v = (v + v.T.conj()) * 0.5

            # Fuse auxiliary and charge compensating parts
            v = fuse(fuse(v, axis=1), axis=0)

            j2c[q] = v
            logger.timer_debug1(self, "j2c [%d]" % q, *cput1)

        # MPI reduce
        for q in qpts.loop(1):
            j2c[q] = mpi_helper.allreduce(j2c[q])

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

        # Inverse Cholesky decomposition
        j2c_chol = np.dot(v * (w**-0.5)[None], v.T.conj())

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
        # TODO can we exploit symmetry here?
        policy = self.mpi_policy()
        kpt_pairs = np.array([(self.kpts[ki], self.kpts[kj]) for ki, kj in policy])

        # Initialise arrays
        int3c2e = {idx: np.zeros((ngrids, self.nao_pair), dtype=np.complex128) for idx in policy}

        # Construct the integral
        int3c2e_part = aux_e2(
            self.cell,
            fused_cell,
            "int3c2e",
            aosym="s2",
            kptij_lst=kpt_pairs,
        )
        int3c2e_part = lib.transpose(int3c2e_part, axes=(0, 2, 1))
        int3c2e_part = int3c2e_part.reshape(-1, ngrids, self.nao_pair)

        # Fill the array
        for k, (ki, kj) in enumerate(policy):
            int3c2e[ki, kj] = int3c2e_part[k]

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
        policy = self.mpi_policy()

        # Get the G vectors
        vG, vGbase, _ = fused_cell.get_Gv_weights(mesh)
        ngrids = np.prod(mesh)
        grid = lib.cartesian_prod([np.arange(len(x)) for x in vGbase])
        reciprocal_vectors = self.cell.reciprocal_vectors()

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
                qpt = qpts[q] if time_reversal == 0 else -qpts[q]
                if time_reversal:
                    j2c_chol = j2c_chol.conj()

                # If this q-point won't contribute to the required
                # integrals on this rank, skip it
                if not policy.intersection({*zip(kis, kjs), *zip(kjs, kis)}):
                    continue

                # Eq. 33
                # TODO MPI
                shls_slice = (auxcell.nbas, fused_cell.nbas)
                G_chg = ft_ao(
                    fused_cell,
                    vG,
                    shls_slice=shls_slice,
                    b=reciprocal_vectors,
                    gxyz=grid,
                    Gvbase=vGbase,
                    kpt=-qpt,
                )
                G_chg *= weighted_coulG(self, -qpt, mesh=mesh).ravel()[:, None]
                logger.debug1(self, "Norm of FT for fused cell: %.6g", np.linalg.norm(G_chg))

                # Eq. 26
                if qpts.is_zero(qpt):
                    logger.debug(self, "Including net charge of AO products")
                    vbar = fuse(auxbar(fused_cell))
                    ovlp = self.cell.pbc_intor("int1e_ovlp", hermi=0, kpts=kpts[kis])
                    ovlp = [lib.pack_tril(s) for s in ovlp]

                # Eq. 24
                # TODO MPI
                p0, p1, pn = balance_partition(self.cell.ao_loc_nr() * self.nao, self.nao**2)[0]
                shls_slice = (p0, p1, 0, self.cell.nbas)
                G_ao = ft_aopair_kpts(
                    self.cell,
                    vG,
                    shls_slice=shls_slice,
                    b=reciprocal_vectors,
                    aosym="s2",
                    gxyz=grid,
                    Gvbase=vGbase,
                    q=-qpt,
                    kptjs=kpts[kjs],
                )
                G_ao = G_ao.reshape(-1, ngrids, self.nao_pair)
                logger.debug1(self, "Norm of FT for AO cell: %.6g", np.linalg.norm(G_ao))

                for i, (ki, kj) in enumerate(zip(kis, kjs)):
                    # If this k-point pair won't contribute to the
                    # required integrals on this rank, skip it
                    if (ki, kj) not in policy and (kj, ki) not in policy:
                        continue

                    # Eq. 31, first term
                    v = int3c2e[ki, kj].copy()

                    # Eq. 31, second term
                    if qpts.is_zero(qpt):
                        mask = np.where(vbar != 0)[0]
                        v[mask] -= lib.einsum("i,j->ij", vbar[mask], ovlp[i])

                    # Eq. 31, third term
                    v[naux:] -= np.dot(G_chg.T.conj(), G_ao[i])

                    # Fuse auxiliary and charge compensating parts
                    v = fuse(v)

                    # Eq. 29
                    v = np.dot(j2c_chol, v)

                    j3c_tri[ki, kj] = v + j3c_tri.get((ki, kj), 0.0)
                    logger.debug(self, "Filled j3c for kpt [%d, %d]", ki, kj)

        # Unpack the three-center integrals
        j3c = {}
        for ki, kj in policy:
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
        """Build the density fitting integrals."""

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
