"""Bispectrum multipoles evaluated on the managed 3PCF FFT grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
import numpy as np

from .grid import FFTGrid


@dataclass
class BMultipoleGrid:
    """In-memory storage of bispectrum multipoles on an :class:`FFTGrid`.

    The stored L range depends on the multipole basis.

    ``basis='fourier'``
        Store full complex Fourier coefficients ``B_L`` for
        ``L = -Lmax, ..., Lmax``.

    ``basis='fourier-even'``
        Store cosine coefficients ``c_L`` for ``L = 0, ..., Lmax`` in the
        convention

        ``B(delta) = c_0 + sum_{L>0} c_L cos(L delta)``.

        These are *not* duplicated into negative-L storage.  Downstream
        H-kernel construction must interpret them as
        ``B_0 = c_0`` and ``B_{+L} = B_{-L} = c_L/2`` for ``L > 0``.

    The object is independent of spin, epsilon, and opening-angle mode k, and
    can therefore be shared by all downstream stages.
    """

    grid: FFTGrid
    Lmax: int
    basis: str = "fourier-even"
    values: np.ndarray | None = None
    L_values: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Lmax = int(self.Lmax)
        self.basis = self._canonical_basis(self.basis)
        self.L_values = self._make_L_values(self.basis)
        if self.values is not None:
            self.values = np.asarray(self.values)
            self._validate_values()

    @staticmethod
    def _canonical_basis(basis: str) -> str:
        basis = str(basis)
        if basis in {"fourier", "fourier-even"}:
            return basis
        if basis in {"cosine", "sine", "legendre"}:
            raise ValueError(
                f"basis={basis!r} is not supported by the spin-3PCF H_k pipeline. "
                "Use basis='fourier-even' or basis='fourier'."
            )
        raise ValueError(
            f"Unsupported multipole basis {basis!r}. "
            "Supported bases are 'fourier' and 'fourier-even'."
        )

    def _make_L_values(self, basis: str) -> np.ndarray:
        if basis == "fourier-even":
            return np.arange(0, self.Lmax + 1, dtype=int)
        if basis == "fourier":
            return np.arange(-self.Lmax, self.Lmax + 1, dtype=int)
        raise ValueError(
            f"Unsupported multipole basis {self.basis!r}. "
            "Supported bases are 'fourier' and 'fourier-even'."
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
    def ell1(self) -> np.ndarray:
        return self.grid.ELL1

    @property
    def ell2(self) -> np.ndarray:
        return self.grid.ELL2

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

        self._set_basis(getattr(bmultipole, "basis", self.basis))

        vals = [
            bmultipole(int(L), self.grid.ELL1, self.grid.ELL2)
            for L in self.L_values
        ]
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

    def iter_hkernel_terms(self) -> Iterator[tuple[complex, int, np.ndarray]]:
        """Iterate over full-Fourier H-kernel terms.

        Yields ``(weight, L, coeff)`` such that the H-kernel may be built as

        ``sum weight * coeff * G_{Lk}``.

        For ``fourier-even`` storage, positive cosine modes are expanded as
        two full-Fourier contributions with weights ``1/2`` at ``+L`` and
        ``-L``.
        """
        if self.basis == "fourier-even":
            c0 = self.get_stored_mode(0)
            yield 1.0, 0, c0
            for L in range(1, self.Lmax + 1):
                cL = self.get_stored_mode(L)
                yield 0.5, int(+L), cL
                yield 0.5, int(-L), cL
            return

        # Full complex Fourier storage.
        for L, coeff in self.iter_stored_modes():
            yield 1.0, int(L), coeff

    def get_full_fourier_coefficient(self, L: int) -> np.ndarray:
        """Return the equivalent complex Fourier coefficient ``B_L``."""
        L = int(L)
        if abs(L) > self.Lmax:
            raise KeyError(f"L={L} is outside Lmax={self.Lmax}.")
        if self.basis == "fourier-even":
            coeff = self.get_stored_mode(abs(L))
            return coeff if L == 0 else 0.5 * coeff
        return self.get_stored_mode(L)

    def as_stored_cache(self) -> dict[int, np.ndarray]:
        """Return a dictionary of the basis-native stored coefficients."""
        return {int(L): coeff for L, coeff in self.iter_stored_modes()}

    def as_cache(self) -> dict[int, np.ndarray]:
        """Return a full-Fourier coefficient dictionary.

        The returned dictionary contains keys ``-Lmax, ..., Lmax``.  For
        ``fourier-even`` input, the values are reconstructed as
        ``B_0=c_0`` and ``B_±L=c_L/2``.
        """
        return {
            int(L): self.get_full_fourier_coefficient(int(L))
            for L in range(-self.Lmax, self.Lmax + 1)
        }
