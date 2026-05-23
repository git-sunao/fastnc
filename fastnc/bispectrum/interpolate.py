"""Interpolation helpers for angular bispectra."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .base import Bispectrum2D
from .support import Support2D
from .grids import sides_to_ellpsialpha


def sides_to_ruv(ell1, ell2, ell3):
    """TreeCorr-like unsigned triangle coordinates.

    The sides are sorted so that ``d1 >= d2 >= d3``.  We define
    ``r=d2``, ``u=d3/d2``, and ``v=(d1-d2)/d3``.  This removes permutation
    redundancy for symmetric bispectra.
    """
    arr = np.sort(np.stack([ell1, ell2, ell3], axis=0), axis=0)
    d3, d2, d1 = arr[0], arr[1], arr[2]
    r = d2
    u = np.divide(d3, d2, out=np.zeros_like(d2, dtype=float), where=d2 != 0)
    v = np.divide(d1 - d2, d3, out=np.zeros_like(d3, dtype=float), where=d3 != 0)
    return r, u, v


class RuvInterpolatedBispectrum2D(Bispectrum2D):
    def __init__(self, base: Bispectrum2D, r_grid, u_grid, v_grid, log_values,
                 method="linear", window=None):
        self.base = base
        self.r_grid = np.asarray(r_grid)
        self.u_grid = np.asarray(u_grid)
        self.v_grid = np.asarray(v_grid)
        self.log_values = np.asarray(log_values)
        self.interpolator = RegularGridInterpolator(
            (np.log(self.r_grid), np.log(self.u_grid), self.v_grid),
            self.log_values,
            method=method,
            bounds_error=False,
            fill_value=None,
        )
        self.support = base.support
        self.window = window

    @classmethod
    def from_bispectrum(cls, base: Bispectrum2D, r_grid, u_grid, v_grid, method="linear", floor=1.0e-300, **params):
        R, U, V = np.meshgrid(r_grid, u_grid, v_grid, indexing="ij")
        # Inverse of the convention above: d2=r, d3=u*r, d1=r+v*d3.
        d2 = R
        d3 = U * R
        d1 = R + V * d3
        vals = np.maximum(base(d1, d2, d3, **params), floor)
        return cls(base, r_grid, u_grid, v_grid, np.log(vals), method=method)

    def evaluate(self, ell1, ell2, ell3, **params):
        r, u, v = sides_to_ruv(ell1, ell2, ell3)
        pts = (np.log(r), np.log(u), v)
        val = np.exp(self.interpolator(pts))
        if self.window is not None:
            val = val * self.window(ell1, ell2, ell3)
        return val


# Backward-compatible alias.  New code should use RuvInterpolatedBispectrum2D.
RuvInterpolatedAngularBispectrum2D = RuvInterpolatedBispectrum2D
