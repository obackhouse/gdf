"""Range-separated Gaussian density fitting.
"""

import ctypes

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.data.nist import BOHR
from pyscf.gto import moleintor
from pyscf.lib import logger
from pyscf.pbc.df.aft import weighted_coulG
from pyscf.pbc.df.ft_ao import _RangeSeparatedCell
from pyscf.pbc.df.rsdf_builder import RCUT_THRESHOLD, _ExtendedMoleFT
from pyscf.pbc.df.rsdf_helper import (
    _get_3c2e_Rcuts,
    _get_atom_Rcuts_3c,
    _get_Lsmin,
    _get_refuniq_map,
)
from pyscf.pbc.df.rsdf_helper import _get_schwartz_data as _get_schwarz_data
from pyscf.pbc.df.rsdf_helper import _get_schwartz_dcut as _get_schwarz_dcut
from pyscf.pbc.df.rsdf_helper import intor_j2c, wrap_int3c_nospltbas

from gdf import ft, parameters
from gdf.base import BaseGDF
from gdf.cell import make_auxcell

libpbc = lib.load_library("libpbc")


class RSGDF(BaseGDF):
    __doc__ = BaseGDF.__doc__.format(
        description="Range-separated Gaussian density fitting",
        extra_parameters=(
            "omega : float, optional\n"
            "        Range separation parameter. If `None`, estimate the\n"
            "        parameter using the cell precision. Default value is\n"
            "        `None`.\n"
            "    omega_j2c : float, optional\n"
            "        Range separation parameter for the two-center integral.\n"
            "        Default value is `0.4`.\n"
            "    mesh_j2c : tuple of int\n"
            "        Mesh size along each direction for the 2-electron integral. If\n"
            "        `None`, determine from `cell.precision`. Default value is `None`.\n"
            "    npw_max : int, optional\n"
            "        Maximum mesh size if `omega` is not specified. Default value\n"
            "        is `350`.\n"
            "    omega_min : float, optional\n"
            "        Minimum value of the range separation parameter. Default value\n"
            "        is `0.3`.\n"
            "    precision_j2c : float, optional\n"
            "        Precision for the two-center integral. Default value is `1e-14`.\n"
            "    real_space_precision_factor : float, optional\n"
            "        Factor to scale the precision for the real-space lattice sum.\n"
            "        Default value is `1e-2`.\n"
        ),
    )

    # Extra attributes:
    omega = None
    omega_j2c = 0.4
    mesh_j2c = None
    npw_max = 350
    omega_min = 0.3
    precision_j2c = 1e-14
    real_space_precision_factor = 1e-5  # FIXME this needs to be much tighter than PySCF

    _atributes = BaseGDF._attributes | {
        "omega",
        "omega_j2c",
        "mesh_j2c",
        "npw_max",
        "omega_min",
        "precision_j2c",
        "real_space_precision_factor",
    }

    def get_mesh_parameters(self, cell=None, omega=None, mesh=None, precision=None):
        """
        Return the mesh, the range separation parameters θ, and the
        kinetic energy cutoff.

        Parameters
        ----------
        cell : pyscf.pbc.gto.Cell, optional
            Cell object. If `None`, use `self.cell`. Default value is
            `None`.
        omega : float, optional
            Range separation parameter. If `None`, use `self.omega`, or
            estimate the parameter using the cell precision. Default
            value is `None`.
        mesh : tuple of int, optional
            Mesh size along each direction. If `None`, use `self.mesh`.
            Default value is `None`.
        precision : float, optional
            Precision of the calculation. If `None`, use
            `self.precision_R`. Default value is `None`.

        Returns
        -------
        mesh : tuple of int
            Mesh size along each direction.
        omega : float
            Range separation parameter.
        ke_cutoff : float
            Kinetic energy cutoff.
        """

        cell = cell if cell is not None else self.cell
        omega = omega if omega is not None else self.omega
        mesh = mesh if mesh is not None else self.mesh
        precision = precision if precision is not None else self.cell.precision
        ke_cutoff = None

        # TODO what is this?
        kpts = np.dot(self.kpts.get_scaled_kpts(self.kpts._kpts), cell.reciprocal_vectors())
        kmax = np.max(np.linalg.norm(kpts, axis=1))
        if kmax < 1e-3:
            kmax = 2 * np.pi * (0.75 / np.pi / cell.vol) ** (1 / 3)

        if omega is None:
            omega, ke_cutoff, mesh = parameters.estimate_omega_for_npw(
                cell,
                self.npw_max,
                precision=precision,
                kmax=kmax,
                round_to_odd=True,
            )

            if omega < self.omega_min:
                omega = self.omega_min
                ke_cutoff, mesh = parameters.estimate_mesh_for_omega(
                    cell,
                    omega,
                    precision=precision,
                    kmax=kmax,
                    round_to_odd=True,
                )

            if mesh is None:
                mesh = self.mesh

        if mesh is None:
            ke_cutoff, mesh = parameters.estimate_mesh_for_omega(
                cell,
                omega,
                precision=precision,
                kmax=kmax,
                round_to_odd=True,
            )

        if cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
            mesh[2] = parameters.estimate_meshz(cell)
        elif cell.dimension < 2:
            mesh[cell.dimension :] = cell.mesh[cell.dimension :]

        mesh = cell.symmetrize_mesh(mesh)

        return mesh, omega, ke_cutoff

    def get_mesh_parameters_j2c(self, auxcell, omega=None, mesh=None, precision=None):
        """
        Return the mesh, the range separation parameters θ, and the
        kinetic energy cutoff for the two-center integral.

        Parameters
        ----------
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary cell object.
        omega : float, optional
            Range separation parameter. If `None`, use `self.omega_j2c`.
            Default value is `None`.
        mesh : tuple of int, optional
            Mesh size along each direction. If `None`, use
            `self.mesh_j2c`. Default value is `None`.
        precision : float, optional
            Precision of the calculation. If `None`, use
            `self.precision_j2c`. Default value is `None`.

        Returns
        -------
        mesh : tuple of int
            Mesh size along each direction.
        omega : float
            Range separation parameter.
        ke_cutoff : float
            Kinetic energy cutoff.
        """

        omega = omega if omega is not None else self.omega_j2c
        mesh = mesh if mesh is not None else self.mesh_j2c
        precision = precision if precision is not None else self.precision_j2c
        ke_cutoff = None

        if mesh is None:
            ke_cutoff, mesh = parameters.estimate_mesh_for_omega(
                auxcell,
                omega,
                precision=precision,
                round_to_odd=True,
            )

        if self.cell.dimension == 2 and self.cell.low_dim_ft_type != "inf_vacuum":
            mesh[2] = parameters.estimate_meshz(self.cell)
        elif self.cell.dimension < 2:
            mesh[self.cell.dimension :] = self.cell.mesh[self.cell.dimension :]

        mesh = self.cell.symmetrize_mesh(mesh)

        return mesh, omega, ke_cutoff

    def build_int2c2e(self, auxcell, omega=None):
        """Build the bare 2-center 2-electron integrals.

        Parameters
        ----------
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary cell.
        omega : float, optional
            Range separation parameter. If `None`, use `self.omega_j2c`.
            Default value is `None`.

        Returns
        -------
        int2c2e : list of numpy.ndarray
            List of 2-center 2-electron integrals for each q-point.
        """

        cput0 = (logger.process_clock(), logger.perf_counter())
        qpts, _ = self.get_qpts(time_reversal_symmetry=True)
        if omega is None:
            omega = abs(self.omega_j2c)

        # FIXME
        int2c2e = intor_j2c(auxcell, omega, kpts=-qpts._kpts)

        logger.timer(self, "bare 2c2e", *cput0)

        return int2c2e

    def build_j2c(self, auxcell, omega=None):
        """Build the 2-center Coulomb integrals.

        Parameters
        ----------
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary cell.
        omega : float, optional
            Range separation parameter. If `None`, use `self.omega_j2c`.
            Default value is `None`.

        Returns
        -------
        j2c : list of numpy.ndarray
            List of 2-center Coulomb integrals for each q-point.
        """

        cput0 = (logger.process_clock(), logger.perf_counter())
        qpts, _ = self.get_qpts(time_reversal_symmetry=True)
        naux = auxcell.nao_nr()

        # Get the bare 2c2e integrals
        int2c2e = self.build_int2c2e(auxcell, omega=omega)

        # Get the mesh
        mesh, omega, ke_cutoff = self.get_mesh_parameters_j2c(auxcell, omega=omega)
        logger.debug(self, "Using mesh %s for 2c2e integrals", mesh)

        # Get the G vectors
        vG, vGbase, _ = self.cell.get_Gv_weights(mesh)
        grid = lib.cartesian_prod([np.arange(len(x)) for x in vGbase])
        reciprocal_vectors = self.cell.reciprocal_vectors()

        # Get the auxbasis charge
        if self.cell.dimension == 3:
            qaux = ft.ft_ao(auxcell, np.zeros((1, 3)))[0].real
        else:
            qaux = np.zeros((auxcell.nao_nr(),))
        qaux = np.outer(qaux, qaux)
        qaux *= np.pi / omega**2 / self.cell.vol

        # Initialise arrays
        j2c = [np.zeros((naux, naux), dtype=int2c2e_part.dtype) for int2c2e_part in int2c2e]

        # Build the 2-center Coulomb integrals
        # FIXME comment with eqn numbers
        for q in qpts.loop(1, mpi=True):
            cput1 = (logger.process_clock(), logger.perf_counter())

            v = int2c2e[q]

            if qpts.is_zero(qpts[q]):
                v -= qaux

            # Get the lattice sum
            G_aux = ft.ft_ao(
                auxcell,
                vG,
                b=reciprocal_vectors,
                gxyz=grid,
                Gvbase=vGbase,
                qpt=-qpts[q],
            ).T
            G_aux_coul = G_aux * weighted_coulG(self, -qpts[q], mesh=mesh, omega=omega)

            tmp = np.dot(G_aux_coul.conj(), G_aux.T)
            if qpts.is_zero(qpts[q]):
                tmp = tmp.real
            v += tmp  # FIXME sign difference from GDF?

            # Hermitise
            v = (v + v.T.conj()) * 0.5

            j2c[q] = v
            logger.timer_debug1(self, "j2c [%d]" % q, *cput1)

        # MPI reduce
        for q in qpts.loop(1):
            j2c[q] = mpi_helper.allreduce(j2c[q])

        logger.timer_debug1(self, "j2c", *cput0)

        return j2c

    def build_int3c2e(
        self,
        auxcell,
        omega=None,
        estimator="ME",
    ):
        """Build the bare 3-center 2-electron integrals.

        Parameters
        ----------
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary cell.
        omega : float, optional
            Range separation parameter. If `None`, use `self.omega`.
            Default value is `None`.

        Returns
        -------
        int3c2e : numpy.ndarray
            List of 3-center 2-electron integrals for each q-point.
        """

        # FIXME can we do aosym = s2?

        cput0 = (logger.process_clock(), logger.perf_counter())
        if omega is None:
            omega = self.omega
        if omega is None:
            omega = self.get_mesh_parameters()[1]
        precision = self.cell.precision * self.real_space_precision_factor

        # Screening data
        ref_uniq_shl_map, uniq_atms, ref_uniq_bas, ref_uniq_bas_loc = _get_refuniq_map(self.cell)
        aux_uniq_shl_map, uniq_atms, aux_uniq_bas, aux_uniq_bas_loc = _get_refuniq_map(auxcell)
        ref_exps = np.array([np.min(np.asarray(b[1:])[:, 0]) for b in ref_uniq_bas])

        # Integral screening
        dstep = 1.0 / BOHR
        q_aux = _get_schwarz_data(aux_uniq_bas, omega, keep1ctr=False, safe=True)
        dcuts = _get_schwarz_dcut(ref_uniq_bas, omega, precision / np.max(q_aux), r0=self.cell.rcut)
        dijs_lst = [np.arange(0, dcut, dstep) for dcut in dcuts]
        dijs_loc = np.cumsum([0] + [len(dij) for dij in dijs_lst]).astype(int)
        if estimator in {"ISFQ0", "ISFQL"}:
            qs = _get_schwarz_data(ref_uniq_bas, omega, dijs_lst, keep1ctr=True, safe=True)
        else:
            qs = [np.zeros_like(dijs) for dijs in dijs_lst]
        rcuts = _get_3c2e_Rcuts(
            ref_uniq_bas, aux_uniq_bas, dijs_lst, omega, precision, estimator, qs
        )
        atom_rcuts = _get_atom_Rcuts_3c(
            rcuts,
            dijs_lst,
            ref_exps,
            ref_uniq_bas_loc,
            aux_uniq_bas_loc,
        )
        ls = _get_Lsmin(self.cell, atom_rcuts, uniq_atms)
        prescreening_data = (
            ref_uniq_shl_map,
            aux_uniq_shl_map,
            len(aux_uniq_bas),
            ref_exps,
            dcuts**2,
            dstep,
            rcuts**2,
            dijs_loc,
            ls,
        )
        logger.debug(
            self, "j3c prescreening: cell rcut %.2f Bohr  keep %d imgs", np.max(atom_rcuts), len(ls)
        )
        cput1 = logger.timer_debug1(self, "j3c prescreening warmup", *cput0)

        # Get the intor
        intor, comp = moleintor._get_intor_and_comp(self.cell._add_suffix("int3c2e"))

        shls_slice = (0, self.cell.nbas, 0, self.cell.nbas, 0, auxcell.nbas)
        shlpr_mask = np.ones((self.cell.nbas, self.cell.nbas), dtype=np.int8, order="C")
        ao_loc = self.cell.ao_loc_nr()
        aux_loc = auxcell.ao_loc_nr(auxcell.cart or "ssc" in intor)[: auxcell.nbas + 1]

        # Get the k-point pairs
        # TODO can we exploit symmetry here?
        policy = self.mpi_policy()
        kpt_pairs = [None] * len(policy)
        for (ki, kj), idx in policy.items():
            kpt_pairs[idx] = (self.kpts[ki], self.kpts[kj])
        kpt_pairs = np.array(kpt_pairs)

        # FIXME rewrite and combine with CC
        # FIXME might fail with gamma point only
        # FIXME calculates as s1, but should be able to be s2?
        int3c = wrap_int3c_nospltbas(
            self.cell,
            auxcell,
            omega,
            shlpr_mask,
            prescreening_data,
            intor,
            "s2",
            comp,
            kpt_pairs,
            bvk_kmesh=self.kpts.kmesh,
        )

        if self.kpts.is_zero(kpt_pairs):
            int3c2e = np.zeros(
                (
                    len(kpt_pairs),
                    1,
                    self.nao_pair,
                    auxcell.nao_nr(),
                ),
                dtype=np.float64,
            )
        else:
            int3c2e = np.zeros(
                (
                    len(kpt_pairs),
                    1,
                    self.nao**2,
                    auxcell.nao_nr(),
                ),
                dtype=np.complex128,
            )
        int3c(shls_slice, int3c2e)
        int3c2e = lib.transpose(int3c2e[:, 0], axes=(0, 2, 1))

        # Pack if needed
        if int3c2e.shape[-1] != self.nao_pair:
            int3c2e = int3c2e.reshape(-1, self.nao, self.nao)
            int3c2e = lib.pack_tril(int3c2e)

        # Will be missing an index if we have a single k-point
        int3c2e = int3c2e.reshape(-1, auxcell.nao_nr(), self.nao_pair)
        int3c2e = int3c2e.astype(np.complex128)

        logger.timer_debug1(self, "int3c2e", *cput0)

        return int3c2e

    def build_j3c(
        self,
        auxcell,
        supmol_ft,
        j2c,
        omega=None,
        estimator="ME",
    ):
        """Build the 3-center Coulomb integrals.

        Parameters
        ----------
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary cell.
        j2c : list of ndarray
            List of 2-center Coulomb integrals for each q-point.
        omega : float, optional
            Range separation parameter. If `None`, use `self.omega`.
            Default value is `None`.

        Returns
        -------
        j3c : numpy.ndarray
            List of 3-center Coulomb integrals for each q-point.
        """
        # TODO supmol_ft doc

        cput0 = (logger.process_clock(), logger.perf_counter())

        kpts = self.kpts
        qpts, qpt_idx, conj = self.get_qpts(return_map=True, time_reversal_symmetry=True)
        naux = auxcell.nao_nr()
        policy = self.mpi_policy()

        # Get the mesh
        mesh, omega, ke_cutoff = self.get_mesh_parameters(omega=omega)

        # Get the G vectors
        vG, vGbase, _ = self.cell.get_Gv_weights(mesh)
        grid = lib.cartesian_prod([np.arange(len(x)) for x in vGbase])
        reciprocal_vectors = self.cell.reciprocal_vectors()

        # Get the Fourier transform kernel
        # FIXME can we drop rs_cell and supmol and just call gen_ft_aopair_kpts or not?
        ft_kern = ft.gen_ft_kernel(supmol_ft, return_complex=False)

        # Get the bare 3c2e integrals (eq. 31, first term)
        int3c2e = self.build_int3c2e(auxcell)

        # Initialise arrays
        j3c_tri = {}

        for q in qpts.loop(1):
            # Take inverse Cholesky factorisation of the 2-center
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
                    continue

                G_aux = ft.ft_ao(
                    auxcell,
                    vG,
                    shls_slice=(0, auxcell.nbas),
                    b=reciprocal_vectors,
                    gxyz=grid,
                    Gvbase=vGbase,
                    qpt=-qpt,
                )
                G_aux *= weighted_coulG(self, -qpt, mesh=mesh, omega=omega)[:, None]
                logger.debug1(self, "Norm of FT for auxiliary cell: %.6g", np.linalg.norm(G_aux))

                if qpts.is_zero(qpt) and self.cell.dimension == 3:
                    logger.debug(self, "Including net charge of AO products")
                    vbar = ft.ft_ao(auxcell, np.zeros((1, 3)))[0].real
                    vbar *= np.pi / omega**2 / self.cell.vol
                    ovlp = self.cell.pbc_intor("int1e_ovlp", hermi=0, kpts=kpts[kis[policy_inds]])
                    ovlp = [lib.pack_tril(s) for s in ovlp]

                G_ao = ft_kern(
                    vG,
                    Gvbase=vGbase,
                    gxyz=grid,
                    qpt=-qpt,
                    kpts=kpts[kjs[policy_inds]],
                    aosym="s2",
                )
                G_ao = G_ao[0] + G_ao[1] * 1.0j  # FIXME
                logger.debug1(self, "Norm of FT for AO cell: %.6g", np.linalg.norm(G_ao))

                for i, (ki, kj) in enumerate(zip(kis[policy_inds], kjs[policy_inds])):
                    v = int3c2e[policy[ki, kj]]

                    if qpts.is_zero(qpt):
                        mask = np.where(vbar != 0)[0]
                        if np.any(mask):
                            v[mask] -= lib.einsum("i,j->ij", vbar[mask], ovlp[i])

                    v += np.dot(G_aux.T.conj(), G_ao[i])

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
        # auxcell = make_auxcell(
        #    self.cell,
        #    self.auxbasis,
        #    exp_to_discard=self.exp_to_discard,
        # )
        from pyscf.df.addons import make_auxmol  # FIXME ???

        auxcell = make_auxmol(self.cell, self.auxbasis)
        drop_eta = self.exp_to_discard
        if drop_eta is not None and drop_eta > 0:
            logger.info(self, "Drop primitive fitting functions with exponent < %s", drop_eta)
            from pyscf.pbc.df.rsdf_helper import remove_exp_basis

            auxbasis = remove_exp_basis(auxcell._basis, amin=drop_eta)
            auxcellnew = make_auxmol(self.cell, auxbasis)
            auxcell = auxcellnew
        auxcell.precision = self.precision_j2c
        auxcell.rcut = max(auxcell.bas_rcut(ib, self.precision_j2c) for ib in range(auxcell.nbas))

        # Ge the parameters or estimate using the cell precision
        mesh, omega, ke_cutoff = self.get_mesh_parameters()
        logger.info(self, "mesh = %s", mesh)
        logger.info(self, "omega = %.10f", omega)
        logger.info(self, "ke_cutoff = %.10f", ke_cutoff)

        # Get the parameters or estimate using the cell precision for
        # the two-center integral
        mesh_j2c, omega_j2c, ke_cutoff_j2c = self.get_mesh_parameters_j2c(auxcell)
        logger.info(self, "mesh_j2c = %s", mesh_j2c)
        logger.info(self, "omega_j2c = %.10f", omega_j2c)
        logger.info(self, "ke_cutoff_j2c = %.10f", ke_cutoff_j2c)

        # Build the range-separated cell
        rs_cell = _RangeSeparatedCell.from_cell(
            self.cell,
            ke_cutoff,
            RCUT_THRESHOLD,
            verbose=self.verbose,
        )

        # Build the supmol
        rcut = parameters.estimate_ft_rcut(rs_cell, self.cell.precision, exclude_dd_block=False)
        supmol_ft = _ExtendedMoleFT.from_cell(
            rs_cell, self.kpts.kmesh, np.max(rcut), verbose=self.verbose
        )
        supmol_ft.exclude_dd_block = False
        supmol_ft = supmol_ft.strip_basis(rcut)
        logger.info(
            self,
            "supmol_ft nbas = %d cGTO = %d pGTO = %d",
            supmol_ft.nbas,
            supmol_ft.nao,
            supmol_ft.npgto_nr(),
        )

        # Get the 2-center Coulomb integrals
        j2c = self.build_j2c(auxcell)

        # Get the 3-center Coulomb integrals
        j3c = self.build_j3c(auxcell, supmol_ft, j2c)

        return j3c
