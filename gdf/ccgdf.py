"""Charge-compensated Gaussian density fitting.

Ref: J. Chem. Phys. 147, 164119 (2017)
"""

import ctypes

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.lib import logger
from pyscf.pbc.df.aft import weighted_coulG
from pyscf.pbc.df.gdf_builder import auxbar
from pyscf.pbc.df.incore import aux_e2
from pyscf.pbc.tools import pbc

from gdf import ft, parameters
from gdf.base import BaseGDF
from gdf.cell import fuse_auxcell_chgcell, make_auxcell, make_chgcell

libpbc = lib.load_library("libpbc")


class CCGDF(BaseGDF):
    __doc__ = BaseGDF.__doc__.format(
        description="Charge-compensated Gaussian density fitting.",
        extra_attributes=(
            "eta : float, optional\n"
            "        Charge compensation parameter. If `None`, determine from\n",
            "        `cell.precision`. Default value is `None`.\n",
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
            eta, mesh, ke_cuttoff = parameters.estimate_eta(
                auxcell,
                kpts=self.kpts._kpts,
                mesh=mesh,
            )

        if mesh is None:
            ke_cutoff = parameters.estimate_ke_cutoff_for_eta(cell, eta, precision=precision)
            mesh = cell.cutoff_to_mesh(ke_cutoff)

        if ke_cutoff is None:
            ke_cutoff = pbc.mesh_to_cutoff(cell.lattice_vectors(), mesh)
            ke_cutoff = np.min(ke_cutoff)

        if cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
            mesh[2] = parameters.estimate_meshz(cell)
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

        logger.timer(self, "bare 2c2e", *cput0)

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
        # TODO keyword arguments
        mesh, eta, ke_cutoff = self.get_mesh_parameters(
            cell=auxcell,
            eta=eta,
            precision=auxcell.precision**2,
        )
        logger.debug(self, "Using mesh %s for 2c2e integrals", mesh)

        # Get the G vectors
        vG, vGbase, _ = fused_cell.get_Gv_weights(mesh)
        grid = lib.cartesian_prod([np.arange(len(x)) for x in vGbase])
        reciprocal_vectors = fused_cell.reciprocal_vectors()

        # Initialise arrays
        j2c = [np.zeros((naux, naux), dtype=int2c2e_part.dtype) for int2c2e_part in int2c2e]

        # Build the 2-center Coulomb integrals
        for q in qpts.loop(1, mpi=True):
            cput1 = (logger.process_clock(), logger.perf_counter())

            # Eq. 32, first term
            v = int2c2e[q]

            # Get the lattice sum
            G_chg = ft.ft_ao(
                fused_cell,
                vG,
                b=reciprocal_vectors,
                gxyz=grid,
                Gvbase=vGbase,
                qpt=-qpts[q],
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

    def build_int3c2e(self, fused_cell):
        """Build the bare 3-center 2-electron integrals.

        Returns
        -------
        int3c2e : dict of ndarray
            Bare 3-center 2-electron integrals for each k-point pair.
        """

        cput0 = (logger.process_clock(), logger.perf_counter())
        ngrids = fused_cell.nao_nr()

        # Get the k-point pairs
        # TODO can we exploit symmetry here?
        policy = self.mpi_policy()
        kpt_pairs = [None] * len(policy)
        for (ki, kj), idx in policy.items():
            kpt_pairs[idx] = (self.kpts[ki], self.kpts[kj])
        kpt_pairs = np.array(kpt_pairs)

        # Construct the integral
        # TODO rewrite and combine with RS stuff
        int3c2e = aux_e2(
            self.cell,
            fused_cell,
            "int3c2e",
            aosym="s2",
            kptij_lst=kpt_pairs,
        )
        int3c2e = lib.transpose(int3c2e, axes=(0, 2, 1))

        # Will be missing an index if we have a single k-point
        int3c2e = int3c2e.reshape(-1, ngrids, self.nao_pair)
        int3c2e = int3c2e.astype(np.complex128)

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

        # FIXME if mesh isn't defined?

        cput0 = (logger.process_clock(), logger.perf_counter())

        kpts = self.kpts
        qpts, qpt_idx, conj = self.get_qpts(return_map=True, time_reversal_symmetry=True)
        naux = auxcell.nao_nr()
        policy = self.mpi_policy()

        # Get the G vectors
        vG, vGbase, _ = fused_cell.get_Gv_weights(mesh)
        grid = lib.cartesian_prod([np.arange(len(x)) for x in vGbase])
        reciprocal_vectors = self.cell.reciprocal_vectors()

        # Get the Fourier transform kernel
        ft_aopair_kpts = ft.gen_ft_aopair_kpts(self.cell, kmesh=self.kpts.kmesh)

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
                kis = np.array([tup[time_reversal] for tup in qpt_idx[q]])
                kjs = np.array([tup[1 - time_reversal] for tup in qpt_idx[q]])
                qpt = qpts[q] if time_reversal == 0 else -qpts[q]
                if time_reversal:
                    # We'll only need this one more time, so just conj
                    # it in-place
                    j2c_chol = j2c_chol.conj()

                # Get the k-point pair indices that are needed for this
                # rank, others can be skipped
                policy_inds = [
                    i
                    for i, (ki, kj) in enumerate(zip(kis, kjs))
                    if (ki, kj) in policy or (kj, ki) in policy
                ]

                # If this q-point won't contribute at all to the
                # required integrals on this rank, skip it
                if not policy_inds:
                    # FIXME this basically never happens
                    continue

                # Eq. 33
                # TODO better MPI - precalculate?
                G_chg = ft.ft_ao(
                    fused_cell,
                    vG,
                    shls_slice=(auxcell.nbas, fused_cell.nbas),
                    b=reciprocal_vectors,
                    gxyz=grid,
                    Gvbase=vGbase,
                    qpt=-qpt,
                )
                G_chg *= weighted_coulG(self, -qpt, mesh=mesh).ravel()[:, None]
                logger.debug1(self, "Norm of FT for fused cell: %.6g", np.linalg.norm(G_chg))

                # Eq. 26
                if qpts.is_zero(qpt):
                    logger.debug(self, "Including net charge of AO products")
                    vbar = fuse(auxbar(fused_cell))
                    ovlp = self.cell.pbc_intor("int1e_ovlp", hermi=0, kpts=kpts[kis[policy_inds]])
                    ovlp = [lib.pack_tril(s) for s in ovlp]

                # Eq. 24
                G_ao = ft_aopair_kpts(
                    vG,
                    Gvbase=vGbase,
                    gxyz=grid,
                    qpt=-qpt,
                    kpts=kpts[kjs[policy_inds]],
                    aosym="s2",
                )
                logger.debug1(self, "Norm of FT for AO cell: %.6g", np.linalg.norm(G_ao))

                for i, (ki, kj) in enumerate(zip(kis[policy_inds], kjs[policy_inds])):
                    # Eq. 31, first term - note that we don't copy,
                    # but this array isn't needed anywhere else
                    v = int3c2e[policy[ki, kj]]

                    # Eq. 31, second term
                    if qpts.is_zero(qpt):
                        mask = np.where(vbar != 0)[0]
                        if np.any(mask):
                            v[mask] -= lib.einsum("i,j->ij", vbar[mask], ovlp[i])

                    # Eq. 31, third term
                    v[naux:] -= np.dot(G_chg.T.conj(), G_ao[i])

                    # Fuse auxiliary and charge compensating parts
                    v = fuse(v)

                    # Eq. 29
                    v = np.dot(j2c_chol, v)

                    # TODO DEBUG - remove
                    if (ki, kj) in j3c_tri:
                        raise ValueError("Oops")

                    j3c_tri[ki, kj] = v
                    logger.debug(self, "Filled j3c for kpt [%d, %d]", ki, kj)

        # We don't need the bare 3c2e integrals anymore
        del int3c2e

        # Unpack the three-center integrals
        j3c = np.zeros((len(policy), naux, self.nao, self.nao), dtype=np.complex128)
        while policy:
            ki, kj = next(iter(policy))

            # Build (Q|ij)
            idx = policy[ki, kj]
            libpbc.PBCunpack_tril_triu(
                j3c[idx].ctypes.data_as(ctypes.c_void_p),
                j3c_tri[ki, kj].ctypes.data_as(ctypes.c_void_p),
                j3c_tri[kj, ki].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(self.nao),
            )
            policy.pop((ki, kj))

            # Build (Q|ji)
            if ki != kj and (kj, ki) in policy:
                # TODO can we combine these?
                idx = policy[kj, ki]
                libpbc.PBCunpack_tril_triu(
                    j3c[idx].ctypes.data_as(ctypes.c_void_p),
                    j3c_tri[kj, ki].ctypes.data_as(ctypes.c_void_p),
                    j3c_tri[ki, kj].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(self.nao),
                )
                policy.pop((kj, ki))

            # We don't need j3c_tri[ki, kj] or j3c_tri[kj, ki] any more
            del j3c_tri[ki, kj]
            if (kj, ki) in policy:
                del j3c_tri[kj, ki]

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
