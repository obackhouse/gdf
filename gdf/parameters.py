"""Parameter estimation functions.
"""

import numpy as np
from pyscf import __config__, gto
from pyscf.lib import logger
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df.rsdf_builder import _round_off_to_odd_mesh, estimate_ft_rcut
from pyscf.pbc.df.rsdf_helper import _binary_search
from pyscf.pbc.tools import pbc as pbctools

ETA_MIN = getattr(__config__, "pbc_df_aft_estimate_eta_min", 0.1)


def estimate_eta(cell, kpts=None, mesh=None, eta_min=ETA_MIN):
    """
    Find an optimal η, mesh, and kinetic energy cutoff for the given
    cell.
    """

    if cell.dimension == 0:
        if mesh is None:
            mesh = cell.mesh
        ke_cutoff = np.min(pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh))
        eta = estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision=cell.precision)
        return eta, mesh, ke_cutoff

    ke_cutoff_min = estimate_ke_cutoff_for_eta(cell, eta_min, precision=cell.precision)
    lattice_vectors = cell.lattice_vectors()

    if mesh is None:
        nkpts = len(kpts)
        ke_cutoff = 30.0 * nkpts ** (-1.0 / 3.0)
        ke_cutoff = max(ke_cutoff, ke_cutoff_min)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
    else:
        mesh = np.asarray(mesh)
        mesh_min = cell.cutoff_to_mesh(ke_cutoff_min)
        if np.any(mesh < mesh_min):
            logger.warn(
                cell,
                "mesh %s is not enough to converge to the required integral precision "
                "%g.\nRecommended mesh is %s.",
                mesh,
                cell.precision,
                mesh_min,
            )

    ke_cutoff = np.min(pbctools.mesh_to_cutoff(lattice_vectors, mesh)[: cell.dimension])
    eta = estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision=cell.precision)

    return eta, mesh, ke_cutoff


def estimate_ke_cutoff_for_eta(cell, eta, precision=None):
    """
    Given η, find the lower bound of the kinetic energy cutoff to attain
    the required precision in Coulomb integrals.
    """

    if precision is None:
        precision = cell.precision

    exp = np.max(np.hstack(cell.bas_exps())) * 2.0
    coef_exp = gto.gto_norm(0, exp)
    coef_eta = gto.gto_norm(0, eta)

    theta = 1.0 / (1.0 / exp + 1.0 / eta)
    norm = (4.0 * np.pi) ** -1.5

    factor = 32.0 * np.pi**5
    factor *= coef_exp**2
    factor *= coef_eta
    factor *= norm
    factor *= exp * 2
    factor /= (exp * eta) ** 1.5
    factor /= precision

    ke_cutoff = 20.0
    ke_cutoff = np.log(factor * (ke_cutoff * 2) ** -0.5) * theta * 2
    ke_cutoff = np.log(factor * (ke_cutoff * 2) ** -0.5) * theta * 2

    return ke_cutoff


def estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision=None):
    """
    Given a kinetic energy cutoff, find the required η to attain the
    required precision in Coulomb integrals.
    """

    if precision is None:
        precision = cell.precision

    exp = np.max(np.hstack(cell.bas_exps())) * 2.0
    coef_exp = gto.gto_norm(0, exp)

    norm = (4.0 * np.pi) ** -1.5

    factor = 64.0 * np.pi**5
    factor *= coef_exp**2
    factor *= norm
    factor *= (exp * ke_cutoff * 2) ** -0.5
    factor /= precision

    eta = 4.0
    eta = np.log(factor * eta**-1.5) * 2
    eta /= ke_cutoff
    eta -= 1.0 / exp
    eta = 1.0 / eta
    eta = min(4.0, max(0.0, eta))

    return eta


def estimate_meshz(cell, precision=None):
    """
    For a 2D system with a truncated Coulomb potential, estimate the
    necessary mesh size in the z-direction to attain the required
    precision to converge the Gaussian function.
    """

    if precision is None:
        precision = cell.precision

    exp = np.max(np.hstack(cell.bas_exps()))
    ke_cutoff = -np.log(precision) * exp * 2.0
    meshz = cell.cutoff_to_mesh(ke_cutoff)[2]

    logger.debug2(cell, "estimate_meshz %d", meshz)

    return max(meshz, cell.mesh[2])


def estimate_ke_cutoff_for_omega(cell, omega, kmax, precision=None):
    """
    Given a range separation parameter, find the lower bound of the
    kinetic energy cutoff to attain the required precision in the long
    range integrals.
    """

    if precision is None:
        precision = cell.precision

    factor = 32.0 * np.pi**2  # Qiming
    # factor = 4.0 * cell.vol / np.pi  # Hongzhou

    ke_cutoff = -2.0 * omega**2
    ke_cutoff *= np.log(precision / (factor * omega**2))
    ke_cutoff = ((ke_cutoff * 2) ** 0.5 + kmax) ** 2
    ke_cutoff *= 0.5

    return ke_cutoff


def estimate_mesh_for_omega(cell, omega, kmax=0.0, precision=None, round_to_odd=True):
    """
    Given a range separation parameter, find the necessary mesh size to
    attain the required precision in the long range integrals.
    """

    if precision is None:
        precision = cell.precision

    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega, precision=cell.precision, kmax=kmax)
    mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), ke_cutoff)

    if round_to_odd:
        mesh = _round_off_to_odd_mesh(mesh)

    return ke_cutoff, mesh


def estimate_omega_for_npw(cell, npw_max, precision=None, kmax=0.0, round_to_odd=True):
    """
    Find the largest range separation parameter where the corresponding
    mesh for achieving a given precision in the long range integrals
    does not exceed `npw_max` in size.
    """

    if precision is None:
        precision = cell.precision

    lattice_vectors = cell.lattice_vectors()

    def omega_to_parameters(omega):
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega, precision=precision, kmax=kmax)
        mesh = pbctools.cutoff_to_mesh(lattice_vectors, ke_cutoff)
        if round_to_odd:
            mesh = _round_off_to_odd_mesh(mesh)
        return ke_cutoff, mesh

    def fcheck(omega):
        return np.prod(omega_to_parameters(omega)[1]) > npw_max

    omega_range = np.array([0.05, 2.0])
    omega = _binary_search(*omega_range, 0.02, False, fcheck)

    ke_cutoff, mesh = omega_to_parameters(omega)

    return omega, ke_cutoff, mesh


del ETA_MIN
