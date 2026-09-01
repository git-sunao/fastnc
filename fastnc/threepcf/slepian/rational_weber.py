"""Semi-analytic Weber kernels for BiHalofit rational radial factors.

The BiHalofit three-halo functions contain

    I(k,z) = 1 / (1 + e_n(z) r_sigma(z) k).

After k=ell/chi, define alpha=e_n r_sigma/chi.  This module evaluates the
same-canonical-order double-Bessel transform of 1/(1+alpha ell) without FFTLog.
It is the validated reference building block for future universal integrated
kernels K(q,rho), rho=alpha/Theta.

General unequal integer orders are intentionally not implemented yet: callers
must not silently use this class for general-spin BiHalofit until that
recurrence reduction is validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre, roots_laguerre

from .weber import canonical_bessel_order


@lru_cache(maxsize=None)
def _legendre_rule(n: int):
    u, w = roots_legendre(int(n))
    phi = 0.5 * np.pi * (u + 1.0)
    weight = 0.5 * w  # division by pi combined with dphi=pi/2 du
    return phi, weight


@lru_cache(maxsize=None)
def _laguerre_rule(n: int):
    return roots_laguerre(int(n))


@dataclass(frozen=True)
class SameOrderInfo:
    order: int
    sign: int


def same_canonical_order(order_x: int, order_theta: int) -> SameOrderInfo:
    ax, sx = canonical_bessel_order(order_x)
    at, st = canonical_bessel_order(order_theta)
    if ax != at:
        raise ValueError(
            "same-canonical-order rational Weber kernel required; "
            f"got ({order_x}, {order_theta}) -> ({ax}, {at})"
        )
    return SameOrderInfo(order=ax, sign=sx * st)


class RationalIWeberSameOrder:
    """Semi-analytic same-order transform of ``I=1/(1+a k)``.

    The dimensionless transform is

        I_hat_n(y,a,rho) = int dq q/(1+rho q) J_n(qy) J_n(qa).

    A Laplace representation turns the oscillatory q integral into a
    Gauss-Laguerre integral over a non-oscillatory Bessel-product Laplace
    transform, itself evaluated by Gauss-Legendre angular quadrature.
    """

    def __init__(self, n_laguerre: int = 32, n_phi: int = 96):
        self.n_laguerre = int(n_laguerre)
        self.n_phi = int(n_phi)
        self._phi, self._wphi = _legendre_rule(self.n_phi)
        self._s, self._ws = _laguerre_rule(self.n_laguerre)
        self._cos_cache = {}

    def _cosn(self, n: int):
        n = abs(int(n))
        out = self._cos_cache.get(n)
        if out is None:
            out = np.cos(n * self._phi)
            self._cos_cache[n] = out
        return out

    def I_hat(self, n: int, y, a: float, rho: float, regulator: float = 0.0):
        """Return the dimensionless rational-I double-Bessel transform."""
        y = np.asarray(y, dtype=float)
        flat = y.reshape(-1)
        a = float(a)
        rho = float(rho)
        eps = float(regulator)
        if rho <= 0.0:
            raise ValueError("rho must be positive")

        yy = flat[:, None, None]
        ss = self._s[None, :, None]
        cp = np.cos(self._phi)[None, None, :]
        base = yy * yy + a * a - 2.0 * yy * a * cp
        A = base + eps * eps
        B = base + (eps + rho * ss) ** 2
        sqrtA = np.sqrt(np.maximum(A, np.finfo(float).tiny))
        sqrtB = np.sqrt(np.maximum(B, np.finfo(float).tiny))
        # Stable evaluation of 1/sqrt(A)-1/sqrt(B).
        diff = (B - A) / (sqrtA * sqrtB * (sqrtA + sqrtB))
        angular = np.sum(
            diff * (self._wphi * self._cosn(n))[None, None, :], axis=2
        )
        out = (angular @ self._ws) / rho
        return out.reshape(y.shape)

    def I_physical(
        self, order_x: int, order_theta: int, x, theta: float,
        *, alpha: float, Theta: float,
    ):
        info = same_canonical_order(order_x, order_theta)
        y = np.asarray(x, dtype=float) / float(Theta)
        a = float(theta) / float(Theta)
        rho = float(alpha) / float(Theta)
        return info.sign * self.I_hat(info.order, y, a, rho) / float(Theta) ** 2

    def kI_regular_hat(self, n: int, y, a: float, rho: float):
        # Away from the genuine W^(0) contact, kI regular = -I_hat/rho.
        return -self.I_hat(n, y, a, rho) / float(rho)

    @staticmethod
    def kI_physical_contact_coefficient(
        order_x: int, order_theta: int, *, alpha: float, chi: float
    ):
        """C in ``C delta(x-theta)/x`` for the physical kI transform."""
        info = same_canonical_order(order_x, order_theta)
        return info.sign / (float(chi) * float(alpha))


__all__ = ["SameOrderInfo", "same_canonical_order", "RationalIWeberSameOrder"]
