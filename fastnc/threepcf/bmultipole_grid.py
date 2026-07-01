"""Bispectrum multipoles evaluated on the managed 3PCF FFT grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
import logging
import time
import numpy as np

from .grid import FFTGrid


@dataclass
class BMultipoleGrid:
    """In-memory storage of bispectrum multipoles on an :class:`FFTGrid`.

    The stored L range depends on the multipole basis.

    ``basis='fourier'``
        Store full complex outer-angle Fourier coefficients ``B_L`` for
        ``L = -Lmax, ..., Lmax``.

    ``basis='fourier-even'``
        Store outer-angle cosine coefficients ``c_L`` for ``L = 0, ..., Lmax``
        in the convention

        ``B(Delta beta) = c_0 + sum_{L>0} c_L cos(L Delta beta)``.

        These are *not* duplicated into negative-L storage.  Downstream
        H-kernel construction interprets them as
        ``B_0 = c_0`` and ``B_{+L} = B_{-L} = c_L/2`` for ``L > 0``.

    ``basis='legendre'``
        Store *inner-angle* Legendre coefficients ``B_ell^P`` for
        ``ell = 0, ..., Lmax`` in

        ``B(alpha) = sum_ell B_ell^P P_ell(cos(alpha))``,
        ``alpha = pi - Delta beta``.

        The H-kernel always uses the existing outer-angle Fourier coupling.
        Before coupling, each Legendre coefficient is expanded exactly into
        its finite Fourier superposition,

        ``P_ell(cos(alpha)) = (-1)^ell sum_m a_{ell m} exp(i m Delta beta)``.

        Consequently the coupling kernel itself, including its exact-zero
        support rules, is reused unchanged.

    The object is independent of spin, epsilon, and opening-angle mode k, and
    can therefore be shared by all downstream stages.
    """

    grid: FFTGrid
    Lmax: int
    Lmin: int = None
    basis: str = "fourier-even"
    values: np.ndarray | None = None
    logger: logging.Logger | None = field(default=None, repr=False, compare=False)
    L_values: np.ndarray = field(init=False)
    _legendre_fourier_cache: dict[int, np.ndarray] | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self):
        self.Lmax = int(self.Lmax)
        self.Lmin = self._make_L_min(self.basis, self.Lmin)
        self.basis = self._canonical_basis(self.basis)
        self.L_values = np.arange(self.Lmin, self.Lmax + 1, dtype=int)
        if self.values is not None:
            self.values = np.asarray(self.values)
            self._validate_values()

    @staticmethod
    def _canonical_basis(basis: str) -> str:
        basis = str(basis)
        if basis in {"fourier", "fourier-even", "legendre"}:
            return basis
        if basis in {"cosine", "sine"}:
            raise ValueError(
                f"basis={basis!r} is not supported by the spin-3PCF H_k pipeline. "
                "Use basis='fourier-even', 'fourier', or 'legendre'."
            )
        raise ValueError(
            f"Unsupported multipole basis {basis!r}. "
            "Supported bases are 'fourier', 'fourier-even', and 'legendre'."
        )
    
    def _make_L_min(self, basis: str, Lmin: int | None) -> int:
        if Lmin is not None:
            return int(Lmin)
        if basis in {"fourier-even", "legendre"}:
            return 0
        if basis == "fourier":
            return -self.Lmax
        raise ValueError(
            f"Unsupported multipole basis {basis!r}. "
            "Supported bases are 'fourier', 'fourier-even', and 'legendre'."
        )

    def _set_basis(self, basis: str) -> None:
        basis = self._canonical_basis(basis)
        if basis != self.basis:
            if self.values is not None:
                raise RuntimeError("Cannot change basis after values have been computed.")
            self.basis = basis
            self.L_values = self._make_L_values(self.basis)

    @property
    def computed(self) -> bool:
        return self.values is not None

    @property
    def ell2(self) -> np.ndarray:
        """Independent Fourier radius ``ell_2`` (X1-reference convention)."""
        return self.grid.ELL1

    @property
    def ell3(self) -> np.ndarray:
        """Independent Fourier radius ``ell_3`` (X1-reference convention)."""
        return self.grid.ELL2

    @property
    def ell1(self) -> np.ndarray:
        """Compatibility alias for the first stored independent Fourier axis."""
        return self.ell2

    @property
    def stored_mode_count(self) -> int:
        return int(self.L_values.size)

    @property
    def full_fourier_mode_count(self) -> int:
        return int(2 * self.Lmax + 1)

    def _validate_values(self) -> None:
        assert self.values is not None
        if self.values.shape[0] != self.L_values.size:
            raise ValueError(
                "values.shape[0] must match the number of stored L modes "
                f"for basis={self.basis!r}."
            )
        if self.values.shape[1:] != self.grid.shape_ell:
            raise ValueError("values must have shape (nL, n_ell, n_ell).")

    def compute(self, bmultipole, *, force: bool = False) -> "BMultipoleGrid":
        """Evaluate and store multipoles on the FFT grid.

        For ``fourier-even`` input only non-negative cosine modes are evaluated.
        This halves the number of stored/evaluated multipoles relative to a full
        complex Fourier representation.
        """
        if self.values is not None and not force:
            return self
        if force:
            self.values = None
        self._legendre_fourier_cache = None

        self._set_basis(getattr(bmultipole, "basis", self.basis))

        vals = []
        for L in self.L_values:
            t0 = time.perf_counter()
            vals.append(bmultipole(int(L), self.ell2, self.ell3))
            if self.logger is not None:
                self.logger.debug("3PCF bmultipoles L=%+d finished in %.3f s", int(L), time.perf_counter() - t0)
        self.values = np.asarray(vals)
        self._validate_values()
        return self

    @classmethod
    def from_bmultipole(cls, bmultipole, *, grid: FFTGrid, Lmax: int) -> "BMultipoleGrid":
        obj = cls(grid=grid, Lmax=int(Lmax), basis=getattr(bmultipole, "basis", "fourier-even"))
        return obj.compute(bmultipole)

    def require_computed(self) -> np.ndarray:
        if self.values is None:
            raise RuntimeError("BMultipoleGrid.compute() must be called before accessing values.")
        return self.values

    def _index_of_L(self, L: int) -> int:
        matches = np.where(self.L_values == int(L))[0]
        if matches.size == 0:
            raise KeyError(f"Stored multipole L={L} is not available for basis={self.basis!r}.")
        return int(matches[0])

    def get_stored_mode(self, L: int) -> np.ndarray:
        """Return the stored coefficient for L.

        For ``fourier-even`` this is the cosine coefficient ``c_L`` and only
        ``L >= 0`` is available.
        """
        values = self.require_computed()
        return values[self._index_of_L(int(L))]

    def iter_stored_modes(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over the basis-native stored coefficients."""
        values = self.require_computed()
        for i, L in enumerate(self.L_values):
            yield int(L), values[i]

    def _legendre_to_full_fourier(self) -> dict[int, np.ndarray]:
        """Return outer-angle Fourier coefficients from inner-angle Legendre data.

        The public Legendre convention is fixed by

        ``B(alpha) = sum_ell B_ell^P P_ell(cos(alpha))``,
        ``alpha = pi - Delta beta``.

        With ``P_ell(cos Delta beta) = sum_m a_{ell m} exp(i m Delta beta)``,
        this gives

        ``B_m = sum_ell (-1)^ell a_{ell m} B_ell^P``.

        The finite Laurent coefficients are evaluated once per requested
        Fourier mode; no angular quadrature is introduced in this conversion.
        """
        if self.basis != "legendre":
            raise RuntimeError("Legendre-to-Fourier conversion requires basis='legendre'.")

        from ..coupling.compute import legendre_laurent_coeffs

        if self._legendre_fourier_cache is not None:
            return self._legendre_fourier_cache

        values = self.require_computed()
        out: dict[int, np.ndarray] = {}
        for m in range(-self.Lmax, self.Lmax + 1):
            coeff_m = np.zeros(
                self.grid.shape_ell,
                dtype=np.result_type(values.dtype, np.float64),
            )
            for ell in range(abs(m), self.Lmax + 1, 2):
                a_ell_m = legendre_laurent_coeffs(int(ell)).get(int(m), 0.0)
                if a_ell_m == 0.0:
                    continue
                coeff_m = coeff_m + ((-1.0) ** ell) * a_ell_m * self.get_stored_mode(ell)
            out[int(m)] = coeff_m
        self._legendre_fourier_cache = out
        return out

    def iter_hkernel_terms(self) -> Iterator[tuple[complex, int, np.ndarray]]:
        """Iterate over the full-Fourier terms used by the H-kernel.

        Yields ``(weight, L, coeff)`` such that

        ``H_k = sum_L weight * coeff * G_{Lk}``.

        ``fourier-even`` storage is expanded as the two terms
        ``c_L/2`` at ``L=+L`` and ``L=-L`` for every ``L>0``.  ``legendre``
        storage is first converted from the package's inner-angle convention
        to a full outer-angle Fourier superposition.  In all cases the
        coupling is evaluated only through the existing Fourier kernel.
        """
        if self.basis == "fourier-even":
            c0 = self.get_stored_mode(0)
            yield 1.0, 0, c0
            for L in range(1, self.Lmax + 1):
                cL = self.get_stored_mode(L)
                yield 0.5, int(+L), cL
                yield 0.5, int(-L), cL
            return

        if self.basis == "legendre":
            for L, coeff in self._legendre_to_full_fourier().items():
                yield 1.0, int(L), coeff
            return

        # Full complex Fourier storage.
        for L, coeff in self.iter_stored_modes():
            yield 1.0, int(L), coeff

    def get_full_fourier_coefficient(self, L: int) -> np.ndarray:
        """Return the equivalent outer-angle complex Fourier coefficient ``B_L``."""
        L = int(L)
        if abs(L) > self.Lmax:
            raise KeyError(f"L={L} is outside Lmax={self.Lmax}.")
        if self.basis == "fourier-even":
            coeff = self.get_stored_mode(abs(L))
            return coeff if L == 0 else 0.5 * coeff
        if self.basis == "legendre":
            return self._legendre_to_full_fourier()[L]
        return self.get_stored_mode(L)

    def as_stored_cache(self) -> dict[int, np.ndarray]:
        """Return a dictionary of the basis-native stored coefficients."""
        return {int(L): coeff for L, coeff in self.iter_stored_modes()}

    def as_cache(self) -> dict[int, np.ndarray]:
        """Return a full-Fourier coefficient dictionary.

        The returned dictionary contains keys ``-Lmax, ..., Lmax``.  For
        ``fourier-even`` input, the values are reconstructed as
        ``B_0=c_0`` and ``B_±L=c_L/2``.  For ``legendre`` input, the returned
        coefficients use the outer-angle convention after the exact finite
        Legendre-to-Fourier conversion.
        """
        return {
            int(L): self.get_full_fourier_coefficient(int(L))
            for L in range(-self.Lmax, self.Lmax + 1)
        }
