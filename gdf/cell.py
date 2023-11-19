"""Auxiliary cells for density fitting.
"""

import ctypes

import numpy as np
from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.pbc.df.incore import make_auxcell as _make_auxcell
from pyscf.pbc.df.rsdf_builder import RCUT_THRESHOLD, estimate_ft_rcut
from pyscf.pbc.gto.cell import _estimate_rcut


def make_auxcell(cell, auxbasis, exp_to_discard=0.0):
    """
    Build a cell with the density fitting `auxbasis` as the basis set.

    To simplify the charged-compensated algorithm, the auxiliary basis
    normalisation coefficients are

    .. math::
        \int (r^{l} e^{-\alpha r^{2}}) r^{2} dr

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Cell object.
    auxbasis : str
        Auxiliary basis set.
    exp_to_discard : float, optional
        Threshold for discarding auxiliary basis functions. Default
        value is `0.0`.

    Returns
    -------
    auxcell : pyscf.pbc.gto.Cell
        Auxiliary cell object.
    """

    # Build the auxilairy cell
    auxcell = _make_auxcell(cell, auxbasis)
    _env = auxcell._env.copy()

    # Set the normalisation coefficients
    ndrop = 0
    rcut = []
    steep = []
    for i in range(len(auxcell._bas)):
        l = auxcell.bas_angular(i)
        α = auxcell.bas_exp(i)

        nprim = auxcell.bas_nprim(i)
        ncont = auxcell.bas_nctr(i)

        pi = auxcell._bas[i, gto.PTR_COEFF]
        c = auxcell._env[pi : pi + nprim * ncont].reshape(ncont, nprim).T

        if exp_to_discard is not None and np.any(α < exp_to_discard):
            mask = α >= exp_to_discard
            α = α[mask]
            c = c[mask]
            nprim, ndrop = len(α), ndrop + nprim - len(α)

        if nprim > 0:
            pe = auxcell._bas[i, gto.PTR_EXP]
            auxcell._bas[i, gto.NPRIM_OF] = nprim
            _env[pe : pe + nprim] = α

            mp = gto.gaussian_int(l * 2 + 2, α)
            mp = lib.einsum("pi,p->i", c, mp)

            cs = lib.einsum("pi,i->pi", c, 1 / mp) * np.sqrt(0.25 / np.pi)
            _env[pi : pi + nprim * ncont] = cs.T.ravel()

            r = _estimate_rcut(α, l, np.max(np.abs(cs), axis=1), cell.precision)
            rcut.append(np.max(r))
            steep.append(i)

    auxcell._env = _env
    auxcell._bas = np.asarray(auxcell._bas[steep], order="C")
    auxcell.rcut = np.max(rcut)

    logger.info(cell, "Dropped %d primitive fitting functions", ndrop)
    logger.info(
        cell, "Auxiliary basis: num shells = %d, num cGTOs = %d", auxcell.nbas, auxcell.nao_nr()
    )
    logger.info(cell, "auxcell.rcut = %s", auxcell.rcut)

    return auxcell


def make_chgcell(auxcell, eta):
    """
    Build a cell with smooth Gaussian functions for each angular momenta
    to carry the compensating charge.

    Parameters
    ----------
    auxcell : pyscf.pbc.gto.Cell
        Cell object with the auxiliary basis.
    eta : float
        Charge compensation parameter.

    Returns
    -------
    chgcell : pyscf.pbc.gto.Cell
        Charge compensating cell.
    """

    # Build the charge compensating cell
    chgcell = auxcell.copy()
    _env = [eta]
    _bas = []

    # Set the normalisation coefficients
    p0 = auxcell._env.size
    p1 = p0 + 1
    l_max = np.max(auxcell._bas[:, gto.ANG_OF])
    norms = [
        1.0 / (np.sqrt(4.0 * np.pi) * gto.gaussian_int(l * 2 + 2, eta)) for l in range(l_max + 1)
    ]
    for i in range(auxcell.natm):
        for l in set(auxcell._bas[auxcell._bas[:, gto.ATOM_OF] == i, gto.ANG_OF]):
            _bas.append([i, l, 1, 1, 0, p0, p1, 0])
            _env.append(norms[l])
            p1 += 1

    chgcell._atm = auxcell._atm
    chgcell._bas = np.asarray(_bas, dtype=np.int32).reshape(-1, gto.BAS_SLOTS)
    chgcell._env = np.hstack((auxcell._env, _env))
    chgcell.rcut = _estimate_rcut(eta, l_max, 1.0, auxcell.precision)

    logger.debug1(
        auxcell,
        "Compensating basis: num shells = %d, num cGTOs = %d",
        chgcell.nbas,
        chgcell.nao_nr(),
    )
    logger.debug1(auxcell, "chgcell.rcut = %s", chgcell.rcut)

    return chgcell


def fuse_auxcell_chgcell(auxcell, chgcell):
    """
    Build a cell fusing the auxiliary and charge compensating basis
    sets. Also returns a function to adapt a matrix in the fused basis
    to the auxiliary basis along the given axis.

    Parameters
    ----------
    auxcell : pyscf.pbc.gto.Cell
        Cell object with the auxiliary basis.
    chgcell : pyscf.pbc.gto.Cell
        Cell object with the charge compensating basis.

    Returns
    -------
    fused_cell : pyscf.pbc.gto.Cell
        Cell object with the fused basis.
    fuse : callable
        Function to adapt a matrix in the fused basis to the auxiliary
        basis along the given axis.
    """

    # 0D systems have no charge compensation
    if auxcell.dimension == 0:
        return auxcell, lambda Lpq, axis=0: Lpq

    # Build the fused cell
    fused_cell = auxcell.copy()
    fused_cell._atm, fused_cell._bas, fused_cell._env = gto.conc_env(
        auxcell._atm,
        auxcell._bas,
        auxcell._env,
        chgcell._atm,
        chgcell._bas,
        chgcell._env,
    )
    fused_cell.rcut = max(auxcell.rcut, chgcell.rcut)

    # Get the offset for each charge compensating basis function
    aux_loc = auxcell.ao_loc_nr()
    naux = aux_loc[-1]
    if auxcell.cart:
        aux_loc_sph = auxcell.ao_loc_nr(cart=False)
        naux_sph = aux_loc_sph[-1]
    chg_loc = chgcell.ao_loc_nr()
    offset = -np.ones((chgcell.natm, 8), dtype=int)
    for i in range(chgcell.nbas):
        offset[chgcell.bas_atom(i), chgcell.bas_angular(i)] = chg_loc[i]

    def fuse(Lpq, axis=0):
        """Fusion function."""

        # FIXME may happen in-place?

        # If we need the final axis, tranpose
        if axis == 1 and Lpq.ndim == 2:
            Lpq = lib.transpose(Lpq)

        # Get the auxiliary and charge compensating parts
        Lpq_aux = Lpq[:naux]
        Lpq_chg = Lpq[naux:]

        # Initialise the fused matrix
        if auxcell.cart:
            if Lpq_aux.ndim == 1:
                npq = 1
                Lpq_out = np.empty((naux_sph,), dtype=Lpq.dtype)
            else:
                npq = Lpq.shape[1]
                Lpq_out = np.empty((naux_sph, npq), dtype=Lpq.dtype)
            if np.iscomplexobj(Lpq_aux):
                npq *= 2  # c2s supports double only
        else:
            Lpq_out = Lpq_aux

        for i in range(auxcell.nbas):
            l = auxcell.bas_angular(i)
            p0 = offset[auxcell.bas_atom(i), l]

            if p0 >= 0:
                if auxcell.cart:
                    nd = (l + 1) * (l + 2) // 2
                    s0, s1 = aux_loc_sph[i], aux_loc_sph[i + 1]
                else:
                    nd = 2 * l + 1
                c0, c1 = aux_loc[i], aux_loc[i + 1]

                # Subtract the charge compensating part
                for i0, i1 in lib.prange(c0, c1, nd):
                    Lpq_aux[i0:i1] -= Lpq_chg[p0 : p0 + nd]

                if auxcell.cart:
                    # Get the spherical contribution
                    if l < 2:
                        Lpq_out[s0:s1] = Lpq_aux[c0:c1]
                    else:
                        Lpq_cart = np.asarray(Lpq_aux[c0:c1], order="C")
                        gto.moleintor.libcgto.CINTc2s_ket_sph(
                            Lpq_out[s0:s1].ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(npq * auxcell.bas_nctr(i)),
                            Lpq_cart.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(l),
                        )

        # If we need the final axis, tranpose back
        if axis == 1 and Lpq_aux.ndim == 2:
            Lpq_out = lib.transpose(Lpq_out)

        return np.asarray(Lpq_out, order="A")

    return fused_cell, fuse
