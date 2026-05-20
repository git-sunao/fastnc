"""Opening-angle resummation utilities."""
from __future__ import annotations

import numpy as np

from .spin import SpinTriple, as_spin_triple


def opening_angle_phase_values(k_values: np.ndarray, sigma: tuple[int, int, int] | SpinTriple, *, phase: str = "nu") -> np.ndarray:
    """Return Fourier phase labels for opening-angle resummation.

    For cross-projection natural components, the manuscript uses
    ``exp(i nu_k DeltaPhi)`` with ``nu_k = k + (sigma2-sigma1)/2``.  Set
    ``phase='k'`` for the raw ``exp(i k DeltaPhi)`` decomposition.
    """
    spin = as_spin_triple(sigma)
    k_values = np.asarray(k_values, dtype=float)
    if phase == "nu":
        return spin.nu(k_values)
    if phase == "k":
        return k_values
    raise ValueError("phase must be either 'nu' or 'k'.")


def resummation_matrix(k_values: np.ndarray, delta_phi: np.ndarray, sigma: tuple[int, int, int] | SpinTriple, *, phase: str = "nu", bin_width: float | None = None) -> np.ndarray:
    labels = opening_angle_phase_values(k_values, sigma, phase=phase)
    delta_phi = np.asarray(delta_phi, dtype=float)
    mat = np.exp(1j * labels[:, None] * delta_phi[None, :])
    if bin_width is not None:
        width = float(bin_width)
        fac = np.ones_like(labels, dtype=complex)
        nz = np.abs(labels) > 0.0
        fac[nz] = (np.exp(1j * labels[nz] * width) - 1.0) / (1j * labels[nz] * width)
        mat = fac[:, None] * mat
    return mat


def resum_multipoles(zeta_k: np.ndarray, k_values: np.ndarray, delta_phi: np.ndarray, sigma: tuple[int, int, int] | SpinTriple, *, phase: str = "nu", normalization: float = 1.0, bin_width: float | None = None) -> np.ndarray:
    """Resum zeta_k(theta1,theta2) into zeta(theta1,theta2,DeltaPhi)."""
    zeta_k = np.asarray(zeta_k)
    if zeta_k.ndim != 3:
        raise ValueError("zeta_k must have shape (nk, ntheta1, ntheta2).")
    mat = resummation_matrix(k_values, delta_phi, sigma, phase=phase, bin_width=bin_width)
    out = np.tensordot(zeta_k, mat, axes=(0, 0))
    # tensordot gives (ntheta1, ntheta2, nphi)
    return normalization * out
