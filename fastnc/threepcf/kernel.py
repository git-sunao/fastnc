"""Construction of the angularly mixed kernel H_k(ell1, ell2)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .spin import SpinTriple, as_spin_triple


@dataclass
class HKernelBuilder:
    """Build H_k from a bispectrum multipole object and a coupling matrix.

    Parameters
    ----------
    sigma
        Effective spin triple.
    coupling
        Callable with signature ``coupling(L, k, psi)``.  This matches the new
        ``fastnc.coupling.CouplingMatrix`` API.
    Lmax
        Maximum absolute complex Fourier mode L in the H_k sum.
    bispectrum_basis
        Currently ``fourier-even`` and ``fourier`` are supported.  For
        ``fourier-even``, the bispectrum multipole object returns cosine
        coefficients c_L and this builder converts them to complex Fourier
        coefficients B_{+L}=B_{-L}=c_L/2 for L>0.
    """

    sigma: tuple[int, int, int] | SpinTriple
    coupling: object
    Lmax: int
    bispectrum_basis: str = "fourier-even"

    def __post_init__(self):
        self.spin = as_spin_triple(self.sigma)
        self.L_values = np.arange(-int(self.Lmax), int(self.Lmax) + 1, dtype=int)

    def bispectrum_complex_coefficient(self, bmultipole, L: int, ell1: np.ndarray, ell2: np.ndarray):
        """Return complex Fourier coefficient B_L from a bispectrum multipole object."""
        L = int(L)
        basis = getattr(bmultipole, "basis", self.bispectrum_basis)
        if basis == "fourier-even":
            coeff = bmultipole(abs(L), ell1, ell2)
            if L == 0:
                return coeff
            return 0.5 * coeff
        if basis == "fourier":
            return bmultipole(L, ell1, ell2)
        if basis in {"cosine", "legendre"}:
            raise ValueError(f"basis={basis!r} is not a complex Fourier basis for H_k. Use basis='fourier-even' or 'fourier'.")
        # Default: try direct complex Fourier mode lookup.
        return bmultipole(L, ell1, ell2)

    def compute(self, k: float, bmultipole, ell1: np.ndarray, ell2: np.ndarray) -> np.ndarray:
        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell1, ell2 = np.broadcast_arrays(ell1, ell2)
        psi = np.arctan2(ell2, ell1)
        out = np.zeros_like(ell1, dtype=complex)
        for L in self.L_values:
            B_L = self.bispectrum_complex_coefficient(bmultipole, int(L), ell1, ell2)
            G_Lk = self.coupling(int(L), float(k), psi)
            out = out + B_L * G_Lk
        return out
