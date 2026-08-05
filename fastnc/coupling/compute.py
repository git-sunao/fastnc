"""Numerical core for the Fourier-basis spin coupling kernel.

The current manuscript defines

    G_{Lk}(sigma; psi) = int_0^{2pi} dDelta
        exp[i (L - nu_k) Delta] exp[i sigma_1 bar_beta(psi, Delta)]

with

    nu_k = k + (sigma_3 - sigma_2)/2.

This is the ``X_1``-reference convention: the independent Fourier vectors are
``ell_2`` and ``ell_3``, and the reference field is the field at vertex 1.
Thus the coupling depends on L and k only through

    delta = L - nu_k,

and on the spin assignment only through sigma_1.  Writing

    h_1 = sigma_1 / 2,
    A(z;psi) = (cos(psi) z + sin(psi))/(cos(psi) + sin(psi) z),
    A(z;psi)^{h_3} = sum_p b_p^{(h_3)}(psi) z^p,

one has

    G_delta(sigma_3; psi) = 2*pi*b_{-delta}^{(h_3)}(psi).

Internally all half-integer labels are stored exactly with integer keys:

    two_q     = 2 h_1 = sigma_1,
    two_delta = 2 delta,
    two_p     = 2 p = -two_delta.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, cos, pi, sin
from typing import Dict, Iterable

import numpy as np
from scipy.integrate import quad

Number = float | int | np.floating | np.integer


@dataclass(frozen=True)
class CouplingIndex:
    """Allowed Bessel orders and opening-angle label for a given k."""

    Sigma: int
    k: float
    m: float
    n: float
    nu: float
    integer_orders: bool


def _as_sigma(sigma: Iterable[int]) -> tuple[int, int, int]:
    out = tuple(int(x) for x in sigma)
    if len(out) != 3:
        raise ValueError("sigma must contain exactly three entries: (sigma1, sigma2, sigma3).")
    return out  # type: ignore[return-value]


def _as_two_x(x: Number, *, name: str = "x", atol: float = 1e-12) -> int:
    """Return exact integer key two_x=2*x for integer/half-integer x."""
    xf = float(x)
    two_x = int(round(2.0 * xf))
    if abs(2.0 * xf - two_x) > atol:
        raise ValueError(f"{name} must be integer or half-integer, got {x}.")
    return two_x


def coupling_index(k: Number, sigma: Iterable[int], *, atol: float = 1e-12) -> CouplingIndex:
    """Return Sigma, m_k, n_k, and nu_k for a given k and sigma."""
    s1, s2, s3 = _as_sigma(sigma)
    Sigma = s1 + s2 + s3
    kf = float(k)
    m = 0.5 * Sigma + kf
    n = 0.5 * Sigma - kf
    nu = kf + 0.5 * (s3 - s2)
    integer_orders = abs(m - round(m)) < atol and abs(n - round(n)) < atol
    return CouplingIndex(Sigma=Sigma, k=kf, m=m, n=n, nu=nu, integer_orders=integer_orders)


def allowed_k(k: Number, sigma: Iterable[int], *, atol: float = 1e-12) -> bool:
    """True when both Bessel orders m_k and n_k are integers."""
    return coupling_index(k, sigma, atol=atol).integer_orders


def two_delta_from_L_k(L: int, k: Number, sigma: Iterable[int], *, atol: float = 1e-12) -> int | None:
    """Return two_delta = 2*(L - nu_k), or None if k is not allowed."""
    idx = coupling_index(k, sigma, atol=atol)
    if not idx.integer_orders:
        return None
    return _as_two_x(float(L) - idx.nu, name="L - nu_k", atol=atol)


def _is_endpoint(psi: float, target: float, atol: float) -> bool:
    return abs(float(psi) - target) <= atol


def bar_beta(psi: float, Delta: float) -> float:
    """bar_beta defined by beta_1 = beta + bar_beta in the X1 convention."""
    z = -(cos(psi) * np.exp(0.5j * Delta) + sin(psi) * np.exp(-0.5j * Delta))
    return float(np.angle(z))


def spin_phase_coeff(q: int, p: int, psi: float, *, atol: float = 1e-14) -> float:
    """Fourier coefficient b_p^{(q)}(psi) for integer q and integer p.

    A(z;psi) = (cos(psi) z + sin(psi))/(cos(psi) + sin(psi) z).
    The analytic one-sided expansions are used on the open intervals
    (0,pi/4) and (pi/4,pi/2).  At psi=0, pi/4, pi/2 the support collapses
    to p=q, 0, and -q respectively.
    """
    q = int(q)
    p = int(p)
    psi = float(psi)

    if q == 0:
        return 1.0 if p == 0 else 0.0

    if not (-atol <= psi <= pi / 2 + atol):
        raise ValueError("psi must be in [0, pi/2].")

    if _is_endpoint(psi, 0.0, atol):
        return 1.0 if p == q else 0.0
    if _is_endpoint(psi, pi / 4, atol):
        return 1.0 if p == 0 else 0.0
    if _is_endpoint(psi, pi / 2, atol):
        return 1.0 if p == -q else 0.0

    if 0.0 < psi < pi / 4:
        t = np.tan(psi)
        if q > 0:
            Q = q
            if p < 0:
                return 0.0
            total = 0.0
            for a in range(0, Q + 1):
                r = p - a
                if r < 0:
                    continue
                total += comb(Q, a) * t ** (Q - a) * ((-1) ** r) * comb(Q + r - 1, r) * t**r
            return float(total)
        Q = -q
        if p > 0:
            return 0.0
        total = 0.0
        for a in range(0, Q + 1):
            r = a - Q - p
            if r < 0:
                continue
            total += comb(Q, a) * t**a * ((-1) ** r) * comb(Q + r - 1, r) * t**r
        return float(total)

    if pi / 4 < psi < pi / 2:
        u = 1.0 / np.tan(psi)
        if q > 0:
            Q = q
            if p > 0:
                return 0.0
            total = 0.0
            for a in range(0, Q + 1):
                r = a - Q - p
                if r < 0:
                    continue
                total += comb(Q, a) * u**a * ((-1) ** r) * comb(Q + r - 1, r) * u**r
            return float(total)
        Q = -q
        if p < 0:
            return 0.0
        total = 0.0
        for a in range(0, Q + 1):
            r = p - Q + a
            if r < 0:
                continue
            total += comb(Q, a) * u**a * ((-1) ** r) * comb(Q + r - 1, r) * u**r
        return float(total)

    return spin_phase_coeff_quad(q, p, psi)


def spin_phase_coeff_quad(q: float, p: float, psi: float) -> float:
    """Quadrature definition of b_p^{(q)}(psi) for integer/half-integer q,p."""
    two_q = _as_two_x(q, name="q")
    two_p = _as_two_x(p, name="p")
    return spin_phase_coeff_from_keys(two_q, two_p, psi)


def spin_phase_coeff_from_keys(two_q: int, two_p: int, psi: float, *, atol: float = 1e-14) -> float:
    """Fourier coefficient using exact integer keys two_q=2q and two_p=2p."""
    two_q = int(two_q)
    two_p = int(two_p)
    if two_q % 2 == 0 and two_p % 2 == 0:
        return spin_phase_coeff(two_q // 2, two_p // 2, psi, atol=atol)

    p = 0.5 * two_p

    def integrand(x: float) -> float:
        val = np.exp(1j * two_q * bar_beta(float(psi), x)) * np.exp(-1j * p * x)
        return float(np.real(val))

    val, _ = quad(integrand, 0.0, 2 * pi, epsabs=1e-11, epsrel=1e-11, limit=300)
    out = float(val / (2 * pi))
    return 0.0 if abs(out) < 10 * np.finfo(float).eps else out


def spin_phase_coeff_from_two_q(two_q: int, p: int, psi: float, *, atol: float = 1e-14) -> float:
    """Backward-compatible wrapper for integer p."""
    return spin_phase_coeff_from_keys(two_q, 2 * int(p), psi, atol=atol)


def exact_zero_delta(two_delta: int, sigma1: int, psi, *, atol: float = 1e-14):
    """Return True if Fourier support implies G_delta(sigma1;psi)=0 exactly.

    Supports scalar or numpy-array psi.
    """
    two_delta = int(two_delta)
    sigma1 = int(sigma1)
    two_p = -two_delta

    x = np.asarray(psi, dtype=float)
    scalar_input = x.ndim == 0

    if sigma1 == 0:
        out = np.full_like(x, two_delta != 0, dtype=bool)
        return bool(out) if scalar_input else out

    # Analytic support rules require integer reference half-spin and integer p.
    if sigma1 % 2 != 0 or two_p % 2 != 0:
        out = np.zeros_like(x, dtype=bool)
        return bool(out) if scalar_input else out

    q = sigma1 // 2
    p = two_p // 2
    eta = 1 if sigma1 > 0 else -1

    out = np.zeros_like(x, dtype=bool)

    is_0 = np.isclose(x, 0.0, atol=atol, rtol=0.0)
    is_mid = np.isclose(x, pi / 4, atol=atol, rtol=0.0)
    is_pi2 = np.isclose(x, pi / 2, atol=atol, rtol=0.0)

    out |= is_0 & (p != q)
    out |= is_mid & (p != 0)
    out |= is_pi2 & (p != -q)

    interior = ~(is_0 | is_mid | is_pi2)

    left = interior & (0.0 < x) & (x < pi / 4)
    right = interior & (pi / 4 < x) & (x < pi / 2)

    out |= left & (eta * p < 0)
    out |= right & (eta * p > 0)

    return bool(out) if scalar_input else out


def coupling_delta(two_delta: int, sigma1: int, psi: float, *, atol: float = 1e-14) -> float:
    """Compute G_delta(sigma1;psi), where sigma1 is the reference-vertex spin."""
    two_delta = int(two_delta)
    sigma1 = int(sigma1)
    if exact_zero_delta(two_delta, sigma1, psi, atol=atol):
        return 0.0
    two_p = -two_delta
    val = 2 * pi * spin_phase_coeff_from_keys(sigma1, two_p, psi, atol=atol)
    return float(0.0 if abs(val) < 10 * np.finfo(float).eps else val)


def coupling_delta_float(delta: Number, sigma1: int, psi: float, *, atol: float = 1e-14) -> float:
    """Compute G_delta from delta=L-nu_k, accepting integer/half-integer delta."""
    return coupling_delta(_as_two_x(delta, name="delta", atol=atol), sigma1, psi, atol=atol)


def coupling_G(L: int, k: Number, sigma: Iterable[int], psi: float, *, atol: float = 1e-14) -> float:
    """Backward-compatible API: compute G_{Lk}(sigma;psi)."""
    two_delta = two_delta_from_L_k(int(L), k, sigma, atol=atol)
    if two_delta is None:
        return 0.0
    return coupling_delta(two_delta, _as_sigma(sigma)[0], psi, atol=atol)


def coupling_G_quad(L: int, k: Number, sigma: Iterable[int], psi: float) -> float:
    """Direct quadrature for validation."""
    idx = coupling_index(k, sigma)
    if not idx.integer_orders:
        return 0.0
    s1 = _as_sigma(sigma)[0]
    delta = float(L) - idx.nu

    def integrand(x: float) -> float:
        return float(np.real(np.exp(1j * delta * x + 1j * s1 * bar_beta(psi, x))))

    val, _ = quad(integrand, 0.0, 2 * pi, epsabs=1e-10, epsrel=1e-10, limit=300)
    return float(0.0 if abs(val) < 1e-14 else val)


def exact_zero_by_support(L: int, k: Number, sigma: Iterable[int], psi: float, *, atol: float = 1e-14) -> bool:
    """Backward-compatible support test for G_{Lk}."""
    two_delta = two_delta_from_L_k(int(L), k, sigma, atol=atol)
    if two_delta is None:
        return True
    return exact_zero_delta(two_delta, _as_sigma(sigma)[2], psi, atol=atol)


def b_array(q: float, p_values: Iterable[float], psi_values: Iterable[float], *, atol: float = 1e-14) -> np.ndarray:
    """Return array b[ipsi, ip] for requested q, p, and psi grids."""
    two_q = _as_two_x(q, name="q", atol=atol)
    two_p_values = [_as_two_x(p, name="p", atol=atol) for p in p_values]
    return b_array_from_two_q(two_q, two_p_values, psi_values, atol=atol)


def b_array_from_two_q(two_q: int, two_p_values: Iterable[int], psi_values: Iterable[float], *, atol: float = 1e-14) -> np.ndarray:
    """Return array b[ipsi, ip] using exact integer keys two_q=2q, two_p=2p."""
    two_ps = np.asarray(list(two_p_values), dtype=int)
    xs = np.asarray(list(psi_values), dtype=float)
    out = np.empty((xs.size, two_ps.size), dtype=float)
    for i, x in enumerate(xs):
        for j, two_p in enumerate(two_ps):
            out[i, j] = spin_phase_coeff_from_keys(int(two_q), int(two_p), float(x), atol=atol)
    out[np.abs(out) < 10 * np.finfo(float).eps] = 0.0
    return out


# Finite Laurent expansion of P_L((z+z^{-1})/2).  The 3PCF pipeline uses
# these coefficients to map inner-angle Legendre bispectrum multipoles onto
# the existing outer-angle Fourier coupling kernel.  They remain independent
# of spin and therefore contain no coupling-specific approximation.
@lru_cache(maxsize=None)
def legendre_laurent_coeffs(L: int) -> Dict[int, float]:
    if L < 0:
        raise ValueError("Legacy Legendre L must be non-negative.")
    if L == 0:
        return {0: 1.0}
    if L == 1:
        return {-1: 0.5, 1: 0.5}
    Pm1 = {0: 1.0}
    P0 = {-1: 0.5, 1: 0.5}
    x_coeff = {-1: 0.5, 1: 0.5}
    for ell in range(1, L):
        xP: Dict[int, float] = {}
        for a, ca in x_coeff.items():
            for b, cb in P0.items():
                xP[a + b] = xP.get(a + b, 0.0) + ca * cb
        P1: Dict[int, float] = {}
        for j, c in xP.items():
            P1[j] = P1.get(j, 0.0) + (2 * ell + 1) * c / (ell + 1)
        for j, c in Pm1.items():
            P1[j] = P1.get(j, 0.0) - ell * c / (ell + 1)
        Pm1, P0 = P0, {j: c for j, c in P1.items() if abs(c) > 0.0}
    return P0


def legendre_support(L: int) -> tuple[int, ...]:
    return tuple(range(-L, L + 1, 2))
