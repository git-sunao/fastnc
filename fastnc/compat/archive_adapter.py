"""Compatibility adapter from new ``fourier-even`` multipoles to archive Fourier.

The new fourier-even convention stores real cosine coefficients ``c_L`` of

    B(Delta beta) = c_0 + sum_{L>0} c_L cos(L Delta beta),

where ``Delta beta`` is the outer angle.  The archived ``multipole_type=
'fourier'`` engine instead consumes the non-negative complex Fourier
coefficients with respect to the inner angle ``alpha = pi - Delta beta``:

    b_L = (1/pi) int_0^pi d alpha B(pi-alpha) exp(i L alpha).

Therefore

    b_0 = c_0,
    b_L = (-1)^L c_L / 2,  L > 0.

The archive engine subsequently applies its own ``(-1)^L`` factor in
``FastNaturalComponents.HM``.  Do not absorb that second factor here.
"""
from __future__ import annotations

import numpy as np


class ArchiveFourierEvenBispectrumAdapter:
    """Expose a new Fourier-even multipole through the archive bispectrum API."""

    multipole_type = "fourier"

    def __init__(self, multipole, *, ell_min: float, ell_max: float):
        basis = getattr(multipole, "basis", None)
        if basis != "fourier-even":
            raise ValueError(
                "ArchiveFourierEvenBispectrumAdapter requires "
                "basis='fourier-even'; got {!r}.".format(basis)
            )
        self.multipole = multipole
        self.ell1min = float(ell_min)
        self.ell1max = float(ell_max)
        if not (self.ell1min > 0.0 and self.ell1max > self.ell1min):
            raise ValueError("Require 0 < ell_min < ell_max.")

    @staticmethod
    def _ell12_from_ell_psi(ell, psi):
        ell = np.asarray(ell, dtype=float)
        psi = np.asarray(psi, dtype=float)
        return ell * np.cos(psi), ell * np.sin(psi)

    @staticmethod
    def _archive_fourier_factor(L: np.ndarray, ndim: int) -> np.ndarray:
        """Return ``1`` for ``L=0`` and ``(-1)^L/2`` otherwise.

        The returned factor broadcasts against arrays shaped
        ``(n_mode, *ell_shape)``.
        """
        L = np.asarray(L, dtype=int)
        factor = np.where(L == 0, 1.0, 0.5 * (-1.0) ** L)
        return factor.reshape((L.size,) + (1,) * ndim)

    def kappa_bispectrum_multipole(self, L, ell, psi, **kwargs):
        """Return archive Fourier coefficients ``b_L`` with mode axis first."""
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected archive multipole arguments: {unknown}")

        L = np.atleast_1d(np.asarray(L, dtype=int))
        if np.any(L < 0):
            raise ValueError("The archive comparison requests L >= 0.")

        ell1, ell2 = self._ell12_from_ell_psi(ell, psi)
        cL = np.asarray(self.multipole(L, ell1, ell2))
        return self._archive_fourier_factor(L, ell1.ndim) * cL

    def kappa_bispectrum_multipole_diag(self, L, ell1, **kwargs):
        """Return diagonal archive Fourier coefficients ``b_L(ell1, ell1)``."""
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected archive multipole arguments: {unknown}")

        L = np.atleast_1d(np.asarray(L, dtype=int))
        if np.any(L < 0):
            raise ValueError("The archive comparison requests L >= 0.")

        ell1 = np.asarray(ell1, dtype=float)
        cL = np.asarray(self.multipole(L, ell1, ell1))
        return self._archive_fourier_factor(L, ell1.ndim) * cL

class ArchiveLegendreBispectrumAdapter:
    """
    Expose a new Legendre multipole through the archive bispectrum API.

    Both new and archive Legendre multipoles use the inner angle
        alpha = pi - Delta beta
    and the expansion
        B(alpha) = sum_L b_L P_L(cos alpha).
    """

    multipole_type = "legendre"

    def __init__(self, multipole, *, ell_min, ell_max):
        basis = getattr(multipole, "basis", None)
        if basis != "legendre":
            raise ValueError(
                "ArchiveLegendreBispectrumAdapter requires "
                f"basis='legendre'; got {basis!r}."
            )

        self.multipole = multipole
        self.ell1min = float(ell_min)
        self.ell1max = float(ell_max)

    @staticmethod
    def _ell12_from_ell_psi(ell, psi):
        ell = np.asarray(ell, dtype=float)
        psi = np.asarray(psi, dtype=float)
        return ell * np.cos(psi), ell * np.sin(psi)

    def kappa_bispectrum_multipole(self, L, ell, psi, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        L = np.atleast_1d(np.asarray(L, dtype=int))
        if np.any(L < 0):
            raise ValueError("Archive Legendre comparison requires L >= 0.")

        ell1, ell2 = self._ell12_from_ell_psi(ell, psi)
        return np.asarray(self.multipole(L, ell1, ell2))

    def kappa_bispectrum_multipole_diag(self, L, ell1, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        L = np.atleast_1d(np.asarray(L, dtype=int))
        if np.any(L < 0):
            raise ValueError("Archive Legendre comparison requires L >= 0.")

        ell1 = np.asarray(ell1, dtype=float)
        return np.asarray(self.multipole(L, ell1, ell1))
