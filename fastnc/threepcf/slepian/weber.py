"""Weber--Schafheitlin kernels for the fixed-z Slepian calculator.

The Gauss hypergeometric function is evaluated by the specialized vectorized
fastnc Weber evaluator in :mod:`fastnc.threepcf.slepian.hyp2f1`.  Geometry
preparation subsequently uses interpolation of the universal r-table; the
online term assembly contains no hypergeometric calls.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mpmath as mp
from scipy.special import loggamma, rgamma

from .hyp2f1 import hyp2f1_weber


def canonical_bessel_order(order: int):
    order = int(order)
    if order >= 0:
        return order, 1
    n = -order
    return n, -1 if n % 2 else 1


def single_bessel_factor(nu, order):
    """A_m(nu) for integral dl l^(nu+1) J_m(l x)."""
    m, sign = canonical_bessel_order(order)
    nu = np.asarray(nu, dtype=complex)
    loga = (nu + 1.0) * np.log(2.0) + loggamma((m + nu + 2.0) / 2.0) - loggamma((m - nu) / 2.0)
    return sign * np.exp(loga)


def _weber_unit_reference(nu: complex, order_small: int, order_big: int, r: float):
    """Original scalar mpmath Weber evaluator retained as a validation oracle."""
    mu, sign_s = canonical_bessel_order(order_small)
    nb, sign_b = canonical_bessel_order(order_big)
    if r == 0.0 and mu > 0:
        return 0j
    lam = -complex(nu) - 1.0
    A = (nb + mu - lam + 1.0) / 2.0
    B = (mu - nb - lam + 1.0) / 2.0
    C = mu + 1.0
    D = (nb - mu + lam + 1.0) / 2.0
    rr = mp.mpf(r)
    val = (rr**mu) * (mp.mpf(2) ** (-lam)) * mp.gamma(A) * mp.rgamma(D) * mp.rgamma(C)
    val *= mp.hyp2f1(A, B, C, rr * rr)
    return complex(sign_s * sign_b * val)


def _weber_table_fast(exponents, order_small: int, order_big: int, r):
    """Vectorized Weber table for one canonical order orientation."""
    exponents = np.asarray(exponents, dtype=np.complex128)
    r = np.asarray(r, dtype=float)
    mu, sign_s = canonical_bessel_order(order_small)
    nb, sign_b = canonical_bessel_order(order_big)

    eta = exponents[:, None]
    rr = r[None, :]
    lam = -eta - 1.0
    A = (nb + mu - lam + 1.0) / 2.0
    B = (mu - nb - lam + 1.0) / 2.0
    C = np.asarray(mu + 1.0, dtype=np.complex128)
    D = (nb - mu + lam + 1.0) / 2.0

    F = hyp2f1_weber(A, B, C, lam, rr * rr)
    pref = (
        np.power(rr, mu)
        * np.exp(-lam * np.log(2.0))
        * np.exp(loggamma(A))
        * rgamma(D)
        * rgamma(C)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        vals = (sign_s * sign_b) * pref * F

    # At r=0, 0**0 is correctly one for mu=0; for mu>0 it is zero.
    if np.any(r == 0.0) and mu > 0:
        vals[:, r == 0.0] = 0.0j

    # r=1 can contain a genuine singular/contact limit or a removable
    # zero-times-infinity cancellation in the prefactor.  Preserve the exact
    # pre-optimization semantics at this single table column.  This costs only
    # N_fftlog scalar reference calls per table instead of N_fftlog*N_r.
    endpoint = r == 1.0
    if np.any(endpoint):
        for j in np.flatnonzero(endpoint):
            vals[:, j] = np.asarray([
                _weber_unit_reference(complex(e), order_small, order_big, 1.0)
                for e in exponents
            ], dtype=np.complex128)
    return np.asarray(vals, dtype=np.complex128)


def _weber_unit(nu: complex, order_small: int, order_big: int, r: float):
    """Integral dy y^(nu+1) J_mu(r y) J_nu(y), 0<=r<=1."""
    return complex(
        _weber_table_fast(
            np.asarray([complex(nu)]), int(order_small), int(order_big), np.asarray([float(r)])
        )[0, 0]
    )


@dataclass
class WeberTableCache:
    r_points: int = 256

    def __post_init__(self):
        self._tables = {}
        self.hypergeom_evaluations = 0

    @property
    def n_tables(self):
        return len(self._tables)

    def _r_grid(self):
        # Concentrate points close to r=1, where the kernel varies most rapidly.
        u = np.linspace(0.0, 1.0, int(self.r_points))
        return 1.0 - (1.0 - u) ** 4

    def table(self, exponents, order_small, order_big):
        exponents = np.asarray(exponents, dtype=complex)
        key = (exponents.tobytes(), int(order_small), int(order_big), int(self.r_points))
        if key in self._tables:
            return self._tables[key]
        rg = self._r_grid()
        vals = _weber_table_fast(exponents, int(order_small), int(order_big), rg)
        self.hypergeom_evaluations += vals.size
        self._tables[key] = (rg, vals)
        return rg, vals

    def prepared_basis(self, exponents, order_x, order_theta, x, theta):
        """Return basis[j,itheta,ix] for the two-Bessel transform."""
        exponents = np.asarray(exponents, dtype=complex)
        x = np.asarray(x, dtype=float)
        theta = np.asarray(theta, dtype=float)
        X = x[None, :]
        T = theta[:, None]
        small_x = X <= T
        s = np.maximum(X, T)
        r = np.minimum(X, T) / s
        out = np.empty((exponents.size, theta.size, x.size), dtype=complex)
        # Two order orientations because the smaller geometric argument can be x or theta.
        for mask, osmall, obig in ((small_x, order_x, order_theta), (~small_x, order_theta, order_x)):
            rg, tab = self.table(exponents, osmall, obig)
            flat_r = r[mask]
            interp = np.empty((exponents.size, flat_r.size), dtype=complex)
            for j in range(exponents.size):
                interp[j] = np.interp(flat_r, rg, tab[j].real) + 1j*np.interp(flat_r, rg, tab[j].imag)
            scale = s[mask][None, :] ** (-exponents[:, None] - 2.0)
            tmp = interp * scale
            for j in range(exponents.size):
                out[j][mask] = tmp[j]
        return out
