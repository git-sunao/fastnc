"""Minimal logarithmic power-law decomposition used by the fixed-z Slepian route."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FFTLogExpansion:
    k: np.ndarray
    coefficients: np.ndarray
    exponents: np.ndarray
    bias: float


def decompose_log_powerlaw(func, *, k_min, k_max, n, bias, z=None):
    """Approximate ``func(k,z)`` by ``sum c_j k**nu_j`` on a log-periodic grid."""
    n = int(n)
    dlog = np.log(float(k_max) / float(k_min)) / n
    k = float(k_min) * np.exp(dlog * np.arange(n))
    vals = np.asarray(func(k, z) if z is not None else func(k), dtype=complex)
    vals = np.broadcast_to(vals, k.shape)
    g = vals / k**float(bias)
    coeff_periodic = np.fft.fft(g) / n
    eta = 2.0 * np.pi * np.fft.fftfreq(n, d=dlog)
    exponents = float(bias) + 1j * eta
    coefficients = coeff_periodic * float(k_min) ** (-1j * eta)
    return FFTLogExpansion(k=k, coefficients=coefficients, exponents=exponents, bias=float(bias))


def reconstruct(expansion: FFTLogExpansion, k):
    k = np.asarray(k, dtype=float)
    return np.sum(expansion.coefficients[:, None] * k.ravel()[None, :] ** expansion.exponents[:, None], axis=0).reshape(k.shape)
