"""Base classes.
"""

import ctypes

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df.df import DF as _DF
from pyscf.pbc.lib import kpts_helper

from gdf import lib as libgdf
from gdf.kpts import KPoints
from gdf.util import cache


def needs_cderi(func):
    """
    Decorate a function such that `self.build` is called before the
    function is executed, if `self._cderi` is `None`.
    """

    def wrapper(self, *args, **kwargs):
        if self._cderi is None:
            self.build()
        return func(self, *args, **kwargs)

    return wrapper


class BaseGDF(lib.StreamObject):
    __doc__ = """
    {description}

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Cell object.
    kpts : KPoints
        K-points object.

    Attributes
    ----------
    auxbasis : str
        Auxiliary basis set. Default value is `"weigend"`.
    exp_to_discard : float
        Threshold for discarding auxiliary basis functions. Default
        value is `0.0`.
    linear_dep_threshold : float
        Threshold for linear dependence of auxiliary basis functions.
        Default value is `1e-10`.
    mesh : tuple of int
        Mesh size along each direction. If `None`, determine from
        `cell.precision`. Default value is `None`.
    {extra_attributes}
    """.format(
        description="Base class for Gaussian density fitting.",
        extra_attributes="",
    )

    # Attributes:
    auxbasis = "weigend"
    exp_to_discard = 0.0
    linear_dep_threshold = 1e-10
    mesh = None

    _attributes = {"auxbasis", "exp_to_discard", "linear_dep_threshold", "mesh"}
    _keys = property(lambda self: self._attributes | {"cell", "kpts", "stdout", "verbose"})

    def __init__(
        self,
        cell,
        kpts,
        auxbasis=None,
    ):
        # Parameters:
        self.cell = cell
        self.kpts = kpts if isinstance(kpts, KPoints) else KPoints(cell, kpts)
        self.auxbasis = auxbasis if auxbasis is not None else self.auxbasis

        # Logging:
        self.stdout = cell.stdout
        self.verbose = cell.verbose

        # Attributes:
        self._cderi = None

    def _build(self):
        """
        Hook for subclasses to build the density fitting integrals.
        Return value should be the 3-center array of integrals in the
        AO basis for each k-point pair in `self.mpi_policy()`.

        This internal representation ensures optimal caching for C
        routines.
        """
        raise NotImplementedError

    def build(self):
        """Build the density fitting integrals."""

        self.check_sanity()
        self.dump_flags()

        cput0 = (logger.process_clock(), logger.perf_counter())
        self._cderi = self._build()
        self._cderi = np.asarray(self._cderi, order="C")
        logger.timer_debug1(self, "build", *cput0)

        return self

    def dump_flags(self, verbose=None):
        """Dump flags to the logger."""
        log = logger.new_logger(self, verbose)
        log.info("\n******** %s ********", self.__class__)
        log.info("auxbasis = %s", self.auxbasis)
        log.info("exp_to_discard = %s", self.exp_to_discard)
        log.info("linear_dep_threshold = %s", self.linear_dep_threshold)
        log.info("len(kpts) = %s", len(self.kpts))
        log.debug1("kpts = %s", self.kpts)

    def reset(self, cell=None):
        """Reset the object."""
        if cell is not None:
            self.cell = cell
        self._cderi = None

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

    @needs_cderi
    def sr_loop(
        self,
        kpti_kptj=(0, 0),
        max_memory=None,
        compact=True,
        blksize=None,
        aux_slice=None,
    ):
        r"""
        Loop over density fitting integrals for a pair of k-points. This
        function is not recommended for generaluse, but is an important
        part of the interface for PySCF.

        Parameters
        ----------
        kpti_kptj : tuple of int or numpy.ndarray, optional
            Array of k-points or indices of the k-points to return the
            contribution at. Default value is `(0, 0)` (Γ-point).
        max_memory : float, optional
            Keyword argument only to satisfy interface with PySCF. Has
            no effect.
        compact : bool, optional
            Whether to return the integrals in compact form where
            symmetry permits. Default value is `True`.
        blksize : int, optional
            Size of auxiliary basis blocks to yield. If `None`, yield
            all integrals at once. Default value is `None`.
        aux_slice : tuple of int, optional
            Slice of auxiliary basis functions to return. If `None`,
            return the full slice. Default value is `None`.

        Yields
        ------
        LpqR : numpy.ndarray
            Real part of the density fitting integrals at the pair of
            k-points.
        LpqI : numpy.ndarray
            Imaginary part of the density fitting integrals at the pair
            of k-points.
        sign : int
            Sign of the integrals, such that the contribution to the
            4-center Coulomb integrals is

            .. math::

                \sum_{L} s_{pq} s_{rs} V_{Lpq} V_{Lrs}

            where :math:`s_{pq}` is the sign of the integrals for the
            k-point pair. This is always `1` in the case of 3D systems,
            however low-dimensional systems have a single negative
            contribution.
        """

        # Get the k-point indices
        ki, kj = kpti_kptj
        if not isinstance(ki, int):
            ki = self.kpts.index(self.kpts.wrap_around(ki))
        if not isinstance(kj, int):
            kj = self.kpts.index(self.kpts.wrap_around(kj))
        qpt = self.kpts.wrap_around(self.kpts[ki] - self.kpts[kj])

        # Check if this rank has the integral
        policy = self.mpi_policy()
        if (ki, kj) not in policy:
            yield None, None, 0
            return
        idx = policy[ki, kj]

        # Get the auxiliary slice and block size
        if blksize is None:
            blksize = self.naux
        if aux_slice is None:
            aux_slice = (0, self.naux)

        # Loop over the auxiliary basis
        for p0, p1 in lib.prange(*aux_slice, blksize):
            Lpq = self._cderi[idx][p0:p1]
            LpqR = Lpq.real
            LpqI = Lpq.imag
            if compact and self.kpts.is_zero(qpt):
                LpqR = lib.pack_tril(LpqR, axis=-1)
                LpqI = lib.pack_tril(LpqI, axis=-1)
            LpqR = np.asarray(LpqR.reshape(min(p1 - p0, blksize), -1), order="C")
            LpqI = np.asarray(LpqI.reshape(min(p1 - p0, blksize), -1), order="C")
            sign = 1
            yield LpqR, LpqI, sign

    def mpi_policy(self):
        """
        Return the k-point pairs that the current rank should compute
        integral contributions for.

        Returns
        -------
        policy : dict of (tuple of int, int)
            Dictionary whose keys are the k-point pairs to compute
            integrals for, and values the indices in the array on that
            rank for the given pair.
        """

        # TODO build just the policy for this rank
        # TODO benchmark vs. old method, this should cache better
        # TODO this is still far from optimal: prioritise cache locality!

        policies = [{} for _ in range(mpi_helper.size)]

        rank = 0
        i = 0
        idx = 0
        for ki in self.kpts.loop(1):
            for kj in range(ki + 1):
                policies[rank][ki, kj] = idx
                idx += 1
                if ki != kj:
                    policies[rank][kj, ki] = idx
                    idx += 1
                if len(policies[rank]) >= (self.nkpts**2 // mpi_helper.size):
                    rank = (rank + 1) % mpi_helper.size
                    idx = len(policies[rank])

        policy = policies[mpi_helper.rank]

        return policy

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
        # TODO move this gdf.kpts
        for qpt, ki, kj, cc in kpts_helper.kk_adapted_iter(
            self.cell, self.kpts._kpts, time_reversal_symmetry=True
        ):
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

    @needs_cderi
    def get_jk(
        self,
        dm,
        hermi=1,  # Compatibility
        kpts=None,  # Compatibility
        kpts_band=None,  # Compatibility
        with_j=True,
        with_k=True,
        omega=None,  # Compatibility
        exxdiv=None,
    ):
        """
        Build the J (Coulomb) and K (exchange) contributions to the Fock
        matrix due to a given density matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrices at each k-point.
        with_j : bool, optional
            Whether to compute the Coulomb matrix. Default value is
            `True`.
        with_k : bool, optional
            Whether to compute the exchange matrix. Default value is
            `True`.
        exxdiv : str, optional
            Exchange divergence treatment. Default value is `None`.

        Returns
        -------
        vj : numpy.ndarray
            Coulomb matrix, if `with_j` is `True`.
        vk : numpy.ndarray
            Exchange matrix, if `with_k` is `True`.
        """

        # Check compatibility options
        if int(hermi) != 1 or kpts_band is not None or omega is not None:
            raise ValueError(
                f"{self.__class__.__name__}.get_jk only supports the `hermi=1`, "
                "`kpts_band=None`, and `omega=None` arguments."
            )
        if kpts is not None and KPoints(self.cell, kpts) != self.kpts:
            raise ValueError(
                f"{self.__class__.__name__}.get_jk only supports the `kpts` argument "
                f"if it matches the k-points of the {self.__class__.__name__} object."
            )

        # Get the sizes
        nkpts = len(self.kpts)
        nao = self.cell.nao_nr()
        naux = self.get_naoaux(rank_max=True)
        policy = self.mpi_policy()

        # Reshape the density matrix
        dms = dm.reshape(-1, nkpts, nao, nao)
        dms = dms.astype(np.complex128)
        ndm = dms.shape[0]

        # Initialise arrays
        vj = vk = None
        if with_j:
            vj = np.zeros((ndm, nkpts, nao, nao), dtype=np.complex128)
        if with_k:
            vk = np.zeros((ndm, nkpts, nao, nao), dtype=np.complex128)

        for i in range(ndm):
            # J matrix
            if with_j:
                tmp = np.zeros((naux,), dtype=np.complex128)

                for (ki, kj), idx in policy.items():
                    if ki == kj:
                        # cderi(L, p, q) D(p, q) -> tmp(L)
                        scipy.linalg.blas.zgemv(
                            a=self._cderi[idx].reshape(naux, nao * nao),
                            x=dms[i, ki].ravel().conj(),
                            y=tmp,
                            alpha=1.0,
                            beta=1.0,
                            overwrite_y=True,
                        )

                tmp = mpi_helper.allreduce(tmp)

                for (ki, kj), idx in policy.items():
                    if ki == kj:
                        # tmp(L) cderi(L, p, q) -> vj(p, q)
                        scipy.linalg.blas.zgemv(
                            a=self._cderi[idx].reshape(naux, nao * nao),
                            x=tmp,
                            y=vj[i, ki].ravel(),
                            alpha=1.0,
                            beta=1.0,
                            trans=True,
                            overwrite_y=True,
                        )

                vj /= nkpts
                vj = mpi_helper.allreduce(vj)

                del tmp

            # K matrix
            if with_k:
                kis = np.array([ki for ki, _ in policy])
                kjs = np.array([kj for _, kj in policy])
                aux_slice = np.array([0, naux])
                libgdf.cderi_get_k(
                    ctypes.c_int(nkpts),
                    ctypes.c_int(len(policy)),
                    ctypes.c_int(nao),
                    ctypes.c_int(naux),
                    aux_slice.ctypes.data_as(ctypes.c_void_p),
                    self._cderi.ctypes.data_as(ctypes.c_void_p),
                    dms[i].ctypes.data_as(ctypes.c_void_p),
                    kis.ctypes.data_as(ctypes.c_void_p),
                    kjs.ctypes.data_as(ctypes.c_void_p),
                    vk[i].ctypes.data_as(ctypes.c_void_p),
                )

                # for (ki, kj), idx in policy.items():
                #    idx_flip = policy[kj, ki]

                #    # cderi(L, r, p) D(p, q) -> tmp(L, r, q)
                #    tmp = lib.einsum("Lrp,pq->Lrq", self._cderi[idx_flip], dms[i, ki])

                #    # tmp(L, r, q) -> tmp(r, L, q)
                #    # tmp(r, L, q) cderi(L, q, s) -> vk(r, s)
                #    vk[i, kj] += lib.einsum("Lrq,Lqs->rs", tmp, self._cderi[idx])

                # vk /= nkpts
                vk = mpi_helper.allreduce(vk)

        # Exchange divergence treatment
        if with_k and exxdiv == "ewald":
            s = self.get_ovlp()
            madelung = self.madelung
            for i in range(ndm):
                for ki in self.kpts.loop(1):
                    vk[i, ki] += madelung * np.linalg.multi_dot((s[ki], dms[i, ki], s[ki]))

        vj = vj.reshape(dm.shape)
        vk = vk.reshape(dm.shape)

        return vj, vk

    # @needs_cderi
    def get_naoaux(self, rank_max=False):
        """Get the maximum number of auxiliary basis functions."""
        if rank_max:
            op = getattr(mpi_helper.mpi, "MAX", None)
            return mpi_helper.allreduce(self._cderi.shape[1], op=op)
        return self._cderi.shape[1]

    @property
    def kpts_band(self):
        # Currently not supported
        return self.kpts

    @property
    def nao(self):
        """Number of atomic orbitals."""
        return self.cell.nao

    @property
    def nao_pair(self):
        """Number of AO pairs."""
        return self.nao * (self.nao + 1) // 2

    @property
    def naux(self):
        """Number of auxiliary functions."""
        return self.get_naoaux()

    @property
    def nkpts(self):
        """Number of k-points."""
        return len(self.kpts)

    # --- PySCF interface properties

    @property
    def direct_scf_tol(self):
        exp_min = np.min(np.hstack(self.cell.bas_exps()))
        lattice_sum_factor = max((2 * self.cell.rcut) ** 3 / self.cell.vol / exp_min, 1)
        cutoff = self.cell.precision / lattice_sum_factor * 0.1
        return cutoff

    @property
    def _prefer_ccdf(self):
        return True

    # --- Cached PySCF methods

    @cache
    def get_nuc(self, kpts=None):
        """Get the nuclear repulsion energy."""
        # TODO MPI?
        if kpts is None:
            kpts = self.kpts._kpts
        return _DF.get_nuc(self, kpts=kpts)

    @cache
    def get_pp(self, kpts=None):
        """Get the pseudopotential."""
        # TODO MPI?
        if kpts is None:
            kpts = self.kpts._kpts
        return _DF.get_pp(self, kpts=kpts)

    @cache
    def get_ovlp(self):
        return self.cell.pbc_intor("int1e_ovlp", hermi=1, kpts=self.kpts._kpts)

    @property
    @cache
    def madelung(self):
        """Get the Madelung constant."""
        return tools.pbc.madelung(self.cell, self.kpts._kpts)
