"""One-dimensional multipole decomposers.

These classes implement the interpolation-coefficient projection used in
Sugiyama et al. rather than a naive Riemann sum.  The input samples are
interpreted as values of a piecewise-linear function on ``x``; the product of
that interpolant with the chosen basis is integrated analytically in each bin.
"""
from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre


class MultipoleBase:
    def __init__(self, x, max_mode: int, method: str = "gauss-legendre", verbose: bool = False):
        self.x = np.asarray(x, dtype=float)
        if self.x.ndim != 1 or self.x.size < 2:
            raise ValueError("x must be a one-dimensional grid with at least two points")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x must be strictly increasing")
        self.max_mode = int(max_mode)
        self.method = method
        self.verbose = verbose
        self._init_basis_function()

    def _init_basis_function(self):
        pass

    def _get_basis_function(self, modes):
        raise NotImplementedError

    def _linear_interp_coeffs(self, fx, axis=0):
        fx = np.asarray(fx)
        axis = np.core.numeric.normalize_axis_index(axis, fx.ndim)
        if self.x.size != fx.shape[axis]:
            raise ValueError("shape of x and fx must match along the decomposition axis")
        y = np.moveaxis(fx, axis, -1)
        dx = np.diff(self.x)
        a = np.diff(y, axis=-1) / dx
        b = (y[..., :-1] * self.x[1:] - y[..., 1:] * self.x[:-1]) / dx
        return np.moveaxis(a, -1, axis), np.moveaxis(b, -1, axis)

    def decompose(self, fx, modes, axis=0):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        if self.method in {"gauss-legendre", "linear"}:
            return self._decompose_piecewise_linear(fx, modes, axis=axis)
        if self.method == "riemann":
            return self._decompose_riemann(fx, modes, axis=axis)
        raise ValueError(f"unsupported method: {self.method}")

    def _decompose_piecewise_linear(self, fx, modes, axis=0):
        a, b = self._linear_interp_coeffs(fx, axis=axis)
        w0, w1 = self._get_basis_function(modes)
        out = np.tensordot(w1, a, axes=([1], [axis]))
        out += np.tensordot(w0, b, axes=([1], [axis]))
        return out

    def _decompose_riemann(self, fx, modes, axis=0):
        # Provided only for diagnostics; this is not recommended for production.
        w = self._basis_values(modes)
        dx = np.gradient(self.x)
        w = w * dx[None, :]
        return np.tensordot(w, fx, axes=([1], [axis]))

    def _basis_values(self, modes):
        raise NotImplementedError


class MultipoleLegendre(MultipoleBase):
    """Projection onto ``P_L(mu)`` with normalization ``(2L+1)/2``."""
    def _init_basis_function(self):
        self._p_table = {}
        for L in range(max(self.max_mode + 3, 3)):
            self._p_table[L] = eval_legendre(L, self.x)

    def _P(self, L):
        L = np.atleast_1d(np.asarray(L, dtype=int))
        out = np.zeros(L.shape + self.x.shape)
        for i, ell in enumerate(L):
            if ell >= 0:
                if ell not in self._p_table:
                    self._p_table[ell] = eval_legendre(ell, self.x)
                out[i] = self._p_table[ell]
        return out

    def _int_P(self, L):
        L = np.atleast_1d(np.asarray(L, dtype=int))
        out = self._P(L + 1) - self._P(L - 1)
        out[L == 0] = self.x
        return out

    def _int_xP(self, L):
        L = np.atleast_1d(np.asarray(L, dtype=int))
        x = self.x[None, :]
        pL = self._P(L)
        pLm1 = self._P(L - 1)
        pLm2 = self._P(L - 2)
        pLp1 = self._P(L + 1)
        pLp2 = self._P(L + 2)
        out = (pL - pLm2) / (2 * L[:, None] - 1) + x * (pLp1 - pLm1) + (pL - pLp2) / (2 * L[:, None] + 3)
        out[L == 0] = self.x**2 / 2
        out[L == 1] = self.x**3
        return out

    def _get_basis_function(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        norm = (2 * modes[:, None] + 1) / 2
        w0 = norm * np.diff(self._int_P(modes), axis=1)
        w1 = norm * np.diff(self._int_xP(modes), axis=1)
        return w0, w1

    def _basis_values(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        return (2 * modes[:, None] + 1) / 2 * self._P(modes)


class MultipoleFourier(MultipoleBase):
    """Projection against ``exp(i mode x)``.

    The normalization is intentionally not included.  Callers should multiply by
    ``1/(2*pi)`` or another convention-dependent factor.
    """
    def _get_basis_function(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        x = self.x[None, :]
        w0 = np.zeros(modes.shape + self.x.shape, dtype=complex)
        w1 = np.zeros_like(w0)
        zero = modes == 0
        w0[zero] = x
        w1[zero] = x**2 / 2
        nz = ~zero
        if np.any(nz):
            im = 1j * modes[nz, None]
            w0[nz] = np.exp(im * x) / im
            w1[nz] = (x / im - 1.0 / im**2) * np.exp(im * x)
        return np.diff(w0, axis=1), np.diff(w1, axis=1)

    def _basis_values(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        return np.exp(1j * modes[:, None] * self.x[None, :])


class MultipoleCosine(MultipoleBase):
    """Projection against ``cos(mode x)`` without normalization."""
    def _get_basis_function(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        x = self.x[None, :]
        w0 = np.zeros(modes.shape + self.x.shape)
        w1 = np.zeros_like(w0)
        zero = modes == 0
        w0[zero] = x
        w1[zero] = x**2 / 2
        nz = ~zero
        if np.any(nz):
            m = modes[nz, None]
            w0[nz] = np.sin(m * x) / m
            w1[nz] = x * np.sin(m * x) / m + np.cos(m * x) / m**2
        return np.diff(w0, axis=1), np.diff(w1, axis=1)

    def _basis_values(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        return np.cos(modes[:, None] * self.x[None, :])


class MultipoleSine(MultipoleBase):
    """Projection against ``sin(mode x)`` without normalization."""
    def _get_basis_function(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        x = self.x[None, :]
        w0 = np.zeros(modes.shape + self.x.shape)
        w1 = np.zeros_like(w0)
        nz = modes != 0
        if np.any(nz):
            m = modes[nz, None]
            w0[nz] = -np.cos(m * x) / m
            w1[nz] = -x * np.cos(m * x) / m + np.sin(m * x) / m**2
        return np.diff(w0, axis=1), np.diff(w1, axis=1)

    def _basis_values(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        return np.sin(modes[:, None] * self.x[None, :])
