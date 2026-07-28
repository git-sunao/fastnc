"""Small cosmology helpers used by examples, presets, and validation code.

The bundled power-spectrum approximations are intended for diagnostics and
examples, not as replacements for physically normalized CAMB/CLASS spectra.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.integrate import quad


_DEFAULT_COSMO_WMAP_LIKE = {
    "Om0": 0.279,
    "Ode0": 0.721,
    "ns": 0.972,
    "w0": -1.0,
    "wa": 0.0,
    "fnu0": 0.0,
    "sigma8": 0.82,
    "h": 0.70,
    "Ob0": 0.046,
}

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



def simple_linear_growth(z, cosmo: Mapping[str, float] | None = None):
    """Simple debug growth factor, normalized to D(0)=1."""
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + z)


def simple_debug_pklin(
    k,
    cosmo: Mapping[str, float] | None = None,
    amplitude: float = 1.0e4,
    k_eq: float = 2.0e-2,
    transfer_power: float = 1.5,
):
    """Smooth positive debug linear power spectrum.

    This is EH/BBKS-like in spirit but intentionally chosen with a stable
    high-k tail for the bundled Halofit implementation.
    """
    cosmo = _DEFAULT_COSMO_WMAP_LIKE if cosmo is None else cosmo
    ns = float(cosmo.get("ns", 0.97))
    k = np.asarray(k, dtype=float)
    return amplitude * k**ns / (1.0 + (k / k_eq) ** 2) ** transfer_power


def eisenstein_hu_like_pklin(
    k,
    cosmo: Mapping[str, float] | None = None,
    amplitude: float = 1.0e4,
):
    """Crude Eisenstein-Hu/BBKS-like no-wiggle spectrum for diagnostics.

    This is not a replacement for CAMB/CLASS.  It is provided only as a
    convenient preset for debugging code paths.
    """
    cosmo = _DEFAULT_COSMO_WMAP_LIKE if cosmo is None else cosmo
    ns = float(cosmo.get("ns", 0.97))
    Om0 = float(cosmo.get("Om0", 0.279))
    h = float(cosmo.get("h", 0.70))
    theta = 2.7255 / 2.7
    gamma_eff = Om0 * h / theta**2
    q = np.asarray(k, dtype=float) / gamma_eff
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T = L0 / (L0 + C0 * q**2)
    return amplitude * np.asarray(k, dtype=float) ** ns * T**2
