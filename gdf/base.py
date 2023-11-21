"""Base classes.
"""

import numpy as np
from functools import lru_cache
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.df.df import DF as _DF

from gdf.kpts import KPoints
from gdf.lib import cderi_jk


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
        Loop over density fitting integrals for a pair of k-points.

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

        # TODO MPI stuff

        # Get the k-point indices
        ki, kj = kpti_kptj
        if not isinstance(ki, int):
            ki = self.kpts.index(self.kpts.wrap_around(ki))
        if not isinstance(kj, int):
            kj = self.kpts.index(self.kpts.wrap_around(kj))
        qpt = self.kpts.wrap_around(self.kpts[ki] - self.kpts[kj])
        idx = self.mpi_policy()[ki, kj]

        if blksize is None:
            blksize = self.naux
        if aux_slice is None:
            aux_slice = (0, self.naux)

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

        i = 0
        idx = 0
        policy = {}
        for ki in self.kpts.loop(1):
            for kj in range(ki + 1):
                if i % mpi_helper.size == mpi_helper.rank:
                    policy[ki, kj] = idx
                    idx += 1
                    if ki != kj:
                        policy[kj, ki] = idx
                        idx += 1
                i += 1

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

    def get_jk(
        self,
        dm,
        with_j=True,
        with_k=True,
        exxdiv=None,
        # Compatibility options:
        hermi=1,
        kpts=None,
        kpts_band=None,
        omega=None,
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

        if hermi != 1 or kpts_band is not None or omega is not None:
            raise ValueError(
                f"{self.__class__.__name__}.get_jk only supports the `hermi=1`, "
                "`kpts_band=None`, and `omega=None` arguments."
            )
        if kpts is not None and KPoints(self.cell, kpts) != self.kpts:
            raise ValueError(
                f"{self.__class__.__name__}.get_jk only supports the `kpts` argument "
                f"if it matches the k-points of the {self.__class__.__name__} object."
            )

        nkpts = len(self.kpts)
        nao = self.cell.nao_nr()
        naux = self.get_naoaux()

        dms = dm.reshape(-1, nkpts, nao, nao)
        ndm = dms.shape[0]
        vj = vk = None

        policy = self.mpi_policy()

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
                        tmp += lib.einsum("Lpq,pq->L", self._cderi[idx], dms[i, ki].conj())
                for (ki, kj), idx in policy.items():
                    if ki == kj:
                        vj[i, ki] += lib.einsum("L,Lrs->rs", tmp, self._cderi[idx])

            # K matrix
            if with_k:
                for (ki, kj), idx in policy.items():
                    idx_flip = policy[kj, ki]
                    tmp = lib.einsum("Lrp,pq->rLq", self._cderi[idx_flip], dms[i, ki])
                    vk[i, kj] += lib.einsum("rLq,Lqs->rs", tmp, self._cderi[idx])

        vj = mpi_helper.allreduce(vj) / nkpts
        vk = mpi_helper.allreduce(vk) / nkpts

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

    def get_naoaux(self):
        """Get the maximum number of auxiliary basis functions."""
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

    @property
    def direct_scf_tol(self):
        """Direct SCF tolerance, to appease PySCF API."""
        exp_min = np.min(np.hstack(self.cell.bas_exps()))
        lattice_sum_factor = max((2 * self.cell.rcut) ** 3 / self.cell.vol / exp_min, 1)
        cutoff = self.cell.precision / lattice_sum_factor * 0.1
        return cutoff

    @lru_cache(maxsize=1)
    def get_nuc(self, kpts=None):
        """Get the nuclear repulsion energy."""
        # TODO MPI?
        if kpts is None:
            kpts = self.kpts._kpts
        return _DF.get_nuc(self, kpts=kpts)

    @lru_cache(maxsize=1)
    def get_pp(self, kpts=None):
        """Get the pseudopotential."""
        # TODO MPI?
        if kpts is None:
            kpts = self.kpts._kpts
        return _DF.get_pp(self, kpts=kpts)

    @lru_cache(maxsize=1)
    def get_ovlp(self):
        return self.cell.pbc_intor("int1e_ovlp", hermi=1, kpts=self.kpts._kpts)

    @property
    @lru_cache(maxsize=1)
    def madelung(self):
        """Get the Madelung constant."""
        return tools.pbc.madelung(self.cell, self.kpts._kpts)
