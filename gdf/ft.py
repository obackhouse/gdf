"""Fourier transforms.
"""

import ctypes

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df.ft_ao import (
    KECUT_THRESHOLD,
    RCUT_THRESHOLD,
    ExtendedMole,
    _RangeSeparatedCell,
    estimate_rcut,
)
from pyscf.pbc.tools import k2gamma

libpbc = lib.load_library("libpbc")


def gen_ft_aopair_kpts(
    cell,
    kmesh=None,
    intor="GTO_ft_ovlp",
):
    """Generate Fourier transform AO pair for a group of k-points."""

    assert kmesh is not None

    log = logger.new_logger(cell)

    # Get the range-separated cell
    rs_cell = _RangeSeparatedCell.from_cell(
        cell,
        KECUT_THRESHOLD,
        RCUT_THRESHOLD,
        log,
    )
    rcut = estimate_rcut(rs_cell)

    # Get the supercell
    supmol = ExtendedMole.from_cell(rs_cell, kmesh, np.max(rcut), log)
    supmol.strip_basis(rcut)

    # Generate the FT kernel
    ft_kern = gen_ft_kernel(
        supmol,
        intor=intor,
        return_complex=True,
    )

    return ft_kern


def gen_ft_kernel(
    supmol,
    intor="GTO_ft_ovlp",
    return_complex=False,
):
    r"""
    Generate the analytical Fourier transform kernel for AO products.

    .. math::
        \sum_{x} e^{-i k_{j} x} \int e^{-i (G+q) r) i(r) j(r-x) dr^3
    """

    log = logger.new_logger(supmol)
    cput0 = (logger.process_clock(), logger.perf_counter())

    # Get number of a basis functions in original cell
    nbasp = supmol.rs_cell.ref_cell.nbas
    cell0_ao_loc = supmol.rs_cell.ref_cell.ao_loc
    bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape

    # Get the overlap masks
    ovlp_mask = supmol.get_ovlp_mask()
    bvk_ovlp_mask = lib.condense("np.any", ovlp_mask, supmol.rs_cell.sh_loc, supmol.sh_loc)
    cell0_ovlp_mask = np.any(bvk_ovlp_mask.reshape(nbasp, bvk_ncells, nbasp), axis=1)
    ovlp_mask = ovlp_mask.astype(np.int8)
    cell0_ovlp_mask = cell0_ovlp_mask.astype(np.int8)

    # Get the driver
    reciprocal_vectors = supmol.rs_cell.reciprocal_vectors()
    orth = np.allclose(reciprocal_vectors, np.diag(np.diag(reciprocal_vectors)))
    eval_gz = getattr(libpbc, f"GTO_Gv_{'non' if not orth else ''}orth")
    fdrv = getattr(libpbc, f"PBC_ft_bvk_drv")
    cintor = getattr(libpbc, supmol.rs_cell._add_suffix(intor))

    def ft_kernel(
        Gv,
        gxyz=None,
        Gvbase=None,
        qpt=None,
        kpts=None,
        aosym="s1",
        out=None,
    ):
        """Analytical FT for orbital products."""

        assert kpts is not None
        assert qpt is not None
        assert gxyz is not None
        assert Gvbase is not None
        assert qpt.ndim == 1

        cput0 = (logger.process_clock(), logger.perf_counter())

        kpts = np.asarray(kpts, order="C").reshape(-1, 3)
        nkpts = kpts.shape[0]

        expLk = np.exp(1.0j * np.dot(supmol.bvkmesh_Ls, kpts.T))
        expLkR = np.array(expLk.real, order="C")
        expLkI = np.array(expLk.imag, order="C")
        GvT = np.asarray(Gv.T + qpt[:, None], order="C")

        shls_slice = (0, nbasp, 0, nbasp)
        ni = cell0_ao_loc[shls_slice[1]] - cell0_ao_loc[shls_slice[0]]
        nj = cell0_ao_loc[shls_slice[3]] - cell0_ao_loc[shls_slice[2]]

        aosym = aosym[:2]
        if aosym == "s2":
            i0 = cell0_ao_loc[shls_slice[0]]
            i1 = cell0_ao_loc[shls_slice[1]]
            nij = i1 * (i1 + 1) // 2 - i0 * (i0 + 1) // 2
            shape = (nkpts, 1, nij, GvT.shape[1])
        else:
            shape = (nkpts, 1, ni, nj, GvT.shape[1])

        gxyzT = np.asarray(gxyz.T, order="C", dtype=np.int32)
        p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
        bqGv = np.hstack((reciprocal_vectors.ravel(), qpt) + Gvbase)
        p_b = bqGv.ctypes.data_as(ctypes.c_void_p)
        p_mesh = (ctypes.c_int * 3)(*[len(x) for x in Gvbase])

        fill = getattr(libpbc, f"PBC_ft_bvk_{'nk1' if nkpts == 1 else 'k'}{aosym}")
        fsort = getattr(libpbc, f"PBC_ft_{'z' if return_complex else 'd'}sort_{aosym}")

        if return_complex:
            out = np.ndarray(shape, dtype=np.complex128, buffer=out)
        else:
            out = np.ndarray((2,) + shape, buffer=out)

        if GvT.shape[1] > 0:
            fdrv(
                cintor,
                eval_gz,
                fill,
                fsort,
                out.ctypes.data_as(ctypes.c_void_p),
                expLkR.ctypes.data_as(ctypes.c_void_p),
                expLkI.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bvk_ncells),
                ctypes.c_int(nimgs),
                ctypes.c_int(nkpts),
                ctypes.c_int(nbasp),
                ctypes.c_int(1),
                supmol.seg_loc.ctypes.data_as(ctypes.c_void_p),
                supmol.seg2sh.ctypes.data_as(ctypes.c_void_p),
                cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 4)(*shls_slice),
                ovlp_mask.ctypes.data_as(ctypes.c_void_p),
                cell0_ovlp_mask.ctypes.data_as(ctypes.c_void_p),
                GvT.ctypes.data_as(ctypes.c_void_p),
                p_b,
                p_gxyzT,
                p_mesh,
                ctypes.c_int(GvT.shape[1]),
                supmol._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                supmol._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                supmol._env.ctypes.data_as(ctypes.c_void_p),
            )
            log.timer_debug1(f"ft_ao intor {intor}", *cput0)

        if return_complex:
            out = np.rollaxis(out, -1, 2)
            out = out[:, 0]
        else:
            out = np.rollaxis(out, -1, 3)
            out = out[:, :, 0]

        return out

    return ft_kernel
