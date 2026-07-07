"""Analytic and semi-analytic bispectrum building blocks.

This module implements the tree-level matter bispectrum in the full Fourier
basis.  The linear spectrum is represented by a shared complex-power FFTLog
expansion.  Its third-side Fourier coefficients are computed in a single
batched angular FFT and then cached in a contracted ``(log k_>, r)`` table for
fast scalar and grid evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline

from .base import Bispectrum3D
from .multipole import BispectrumMultipole3D
from .support import Support3D
from fastnc.hankel.wrapper import PowerLawFFTLogConfig, power_law_fftlog_coefficients


def standard_linear_growth(z, cosmo: Mapping[str, float] | None = None):
    r"""Linear growth factor for a flat constant-``w`` background.

    The growing-mode solution is evaluated from

    .. math::
       D(a) \propto E(a)\int_0^a \frac{da'}{a'^3 E(a')^3},

    and normalized to ``D(z=0)=1``.  Radiation is intentionally neglected;
    this is the standard late-time growth prescription used for the notebook.
    """
    cosmo = {} if cosmo is None else dict(cosmo)
    Om0 = float(cosmo.get("Om0", 0.3))
    Ode0 = float(cosmo.get("Ode0", 1.0 - Om0))
    w0 = float(cosmo.get("w0", -1.0))
    if Om0 <= 0.0 or Ode0 < 0.0:
        raise ValueError("standard_linear_growth requires Om0>0 and Ode0>=0")

    z_arr = np.asarray(z, dtype=float)
    if np.any(z_arr < -0.999999):
        raise ValueError("z must satisfy z > -1")

    def E(a):
        return np.sqrt(Om0 * a ** -3.0 + Ode0 * a ** (-3.0 * (1.0 + w0)))

    def raw(a):
        value, _ = quad(lambda ap: 1.0 / (ap**3 * E(ap) ** 3), 0.0, float(a),
                        epsabs=1.0e-10, epsrel=1.0e-8, limit=200)
        return 2.5 * Om0 * E(float(a)) * value

    a_arr = 1.0 / (1.0 + z_arr)
    d0 = raw(1.0)
    out = np.array([raw(a) / d0 for a in np.ravel(a_arr)], dtype=float).reshape(a_arr.shape)
    return out.item() if np.isscalar(z) else out


def eisenstein_hu_no_wiggle_pklin(
    k,
    cosmo: Mapping[str, float] | None = None,
    *,
    amplitude: float = 1.0,
):
    """Eisenstein--Hu no-wiggle linear matter spectrum at ``z=0``.

    This is the common zero-baryon transfer-function form from
    Eisenstein & Hu (1998), multiplied by ``amplitude * k**ns``.  It is useful
    for validation and examples; scientific analyses should normally provide a
    CAMB/CLASS spectrum and its physical normalization.
    """
    cosmo = {} if cosmo is None else dict(cosmo)
    Om0 = float(cosmo.get("Om0", 0.3))
    h = float(cosmo.get("h", 0.7))
    ns = float(cosmo.get("ns", 0.965))
    theta = float(cosmo.get("Tcmb", 2.7255)) / 2.7
    k = np.asarray(k, dtype=float)
    if np.any(k <= 0.0):
        raise ValueError("k must be strictly positive")

    omhh = Om0 * h * h
    q = k * theta**2 / omhh
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    transfer = L0 / (L0 + C0 * q * q)
    return float(amplitude) * k**ns * transfer**2


def _validate_log_grid(k, pk):
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    if k.ndim != 1 or pk.ndim != 1 or k.shape != pk.shape:
        raise ValueError("k and pklin must be matching one-dimensional arrays")
    if k.size < 8 or np.any(k <= 0.0) or np.any(pk <= 0.0):
        raise ValueError("k and pklin must be positive; at least eight samples are required")
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("k must be strictly increasing")
    dln = np.diff(np.log(k))
    if not np.allclose(dln, dln[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("the FFTLog representation requires a logarithmically spaced k grid")
    return k, pk


def fourier_power_kernel(mode, nu, r, n_phi: int = 512):
    """Fourier coefficient of ``(1 + 2 r cos(phi) + r**2)**(nu/2)``.

    The kernel itself is universal.  It is evaluated by a periodic trapezoid
    rule (an FFT coefficient), not by evaluating a bispectrum at opening-angle
    samples.  For analytic functions at ``r<1`` this converges spectrally; the
    default resolution is deliberately conservative for the validation path.
    """
    mode = np.asarray(mode, dtype=int)
    r = np.asarray(r, dtype=float)
    if np.any(r < 0.0) or np.any(r > 1.0 + 1.0e-12):
        raise ValueError("r must lie in [0, 1]")
    r = np.minimum(r, np.nextafter(1.0, 0.0))
    phi = 2.0 * np.pi * np.arange(int(n_phi), dtype=float) / int(n_phi)
    # Broadcasting: r[...,None] is a batch of ratios, mode can be scalar or
    # match r.  Current callers use scalar mode, which keeps this compact.
    rr = r[..., None]
    shape = 1.0 + 2.0 * rr * np.cos(phi) + rr * rr
    values = np.exp(0.5 * complex(nu) * np.log(shape))
    phase = np.exp(-1j * np.asarray(mode)[..., None] * phi)
    return np.mean(values * phase, axis=-1)
