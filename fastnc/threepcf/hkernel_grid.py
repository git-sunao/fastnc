"""Angularly mixed H_k kernels on the managed 3PCF FFT grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Iterable
import numpy as np

from .bmultipole_grid import BMultipoleGrid
from .grid import FFTGrid
from .spin import SpinSpec, as_effective_spin_triple


@dataclass(frozen=True)
class HKernelKey:
    """Deduplication key for ``H_k``.

    For fixed B_L values, the X1-reference angular mixing depends on the
    reference-vertex spin ``sigma1`` and ``nu_k``.  ``two_nu`` stores ``2*nu_k``
    as an integer to avoid
    floating-point dictionary keys.
    """

    sigma1: int
    two_nu: int


@dataclass
class HKernel:
    """One angularly mixed kernel ``H_k(ell1, ell2)``."""

    grid: FFTGrid
    key: HKernelKey
    value: np.ndarray
    source_k: float
    source_sigma: tuple[int, int, int]

    @property
    def ell2(self) -> np.ndarray:
        return self.grid.ELL1

    @property
    def ell3(self) -> np.ndarray:
        return self.grid.ELL2

    @property
    def ell1(self) -> np.ndarray:
        """Compatibility alias for the first stored independent Fourier axis."""
        return self.ell2


@dataclass
class HKernelGrid:
    """In-memory storage of deduplicated ``H_k`` kernels.

    The object is initialized by physical ``spin``, ``kmax``, and the common
    :class:`FFTGrid`.  It computes kernels from a :class:`BMultipoleGrid` for one
    epsilon or for all requested epsilons.
    """

    spin: tuple[int, int, int]
    kmax: float
    grid: FFTGrid
    kernels: dict[HKernelKey, HKernel] = field(default_factory=dict)
    aliases: dict[tuple[tuple[int, int, int], int], HKernelKey] = field(default_factory=dict)

    def __post_init__(self):
        self.spin = tuple(int(x) for x in self.spin)
        self.kmax = float(self.kmax)
        self.spin_spec = SpinSpec(self.spin)

    @staticmethod
    def key_from_sigma_k(sigma: tuple[int, int, int], k: float) -> HKernelKey:
        eff = as_effective_spin_triple(sigma)
        two_nu = int(round(2.0 * float(eff.nu(float(k)))))
        return HKernelKey(sigma1=int(eff.sigma1), two_nu=two_nu)

    @staticmethod
    def two_k(k: float) -> int:
        return int(round(2.0 * float(k)))

    def sigma_from_epsilon(self, epsilon: tuple[int, int, int]) -> tuple[int, int, int]:
        return self.spin_spec.sigma_from_epsilon(epsilon)

    def k_values(self, epsilon: tuple[int, int, int]) -> np.ndarray:
        return as_effective_spin_triple(self.sigma_from_epsilon(epsilon)).k_values(self.kmax)

    def _alias_key(self, epsilon: tuple[int, int, int], k: float) -> tuple[tuple[int, int, int], int]:
        return tuple(int(e) for e in epsilon), self.two_k(k)

    def add(self, kernel: HKernel) -> None:
        kernel.grid.validate_same(self.grid)
        self.kernels[kernel.key] = kernel

    def get(self, key: HKernelKey) -> HKernel:
        return self.kernels[key]

    def get_for_epsilon(self, epsilon: tuple[int, int, int], k: float) -> HKernel:
        key = self.aliases.get(self._alias_key(epsilon, k))
        if key is None:
            key = self.key_from_sigma_k(self.sigma_from_epsilon(epsilon), k)
        return self.kernels[key]

    def __contains__(self, key: HKernelKey) -> bool:
        return key in self.kernels

    def _compute_value(self, Bgrid: BMultipoleGrid, coupling, k: float) -> np.ndarray:
        """Compute one H_k using the basis-native BMultipoleGrid storage.

        Bgrid.iter_hkernel_terms() expands the stored basis into the
        full-Fourier terms required by

            H_k = sum_L B_L G_{Lk}.

        In particular, for basis='fourier-even' only L >= 0 cosine
        coefficients are stored, and this method evaluates

            c_0 G_{0k} + sum_{L>0} c_L/2 * (G_{+L,k} + G_{-L,k}),

        which is equivalent to the full complex Fourier sum but avoids
        storing/evaluating duplicate B multipoles.  For ``basis='legendre'``,
        ``Bgrid`` supplies the equivalent outer-angle Fourier coefficients
        obtained from the finite Legendre superposition before this loop.
        """
        Bgrid.require_computed()
        psi_uni, inv = np.unique(self.grid.psi_ell, return_inverse=True)
        shape = self.grid.shape_ell
        out = np.zeros(shape, dtype=complex)
        for weight, L, coeff in Bgrid.iter_hkernel_terms():
            G_uni = coupling(int(L), float(k), psi_uni)
            G = np.asarray(G_uni)[inv].reshape(shape)
            out = out + weight * coeff * G
        return out

    def compute_epsilon(
        self,
        Bgrid: BMultipoleGrid,
        epsilon: tuple[int, int, int],
        *,
        coupling_factory: Callable[[tuple[int, int, int]], object],
        force: bool = False,
    ) -> list[HKernel]:
        """Compute all allowed k modes for one epsilon label."""
        Bgrid.grid.validate_same(self.grid)
        eps = tuple(int(e) for e in epsilon)
        sigma = self.sigma_from_epsilon(eps)
        coupling = coupling_factory(sigma)
        out: list[HKernel] = []
        for k in self.k_values(eps):
            key = self.key_from_sigma_k(sigma, float(k))
            alias = self._alias_key(eps, float(k))
            if key not in self.kernels or force:
                value = self._compute_value(Bgrid, coupling, float(k))
                self.kernels[key] = HKernel(
                    grid=self.grid,
                    key=key,
                    value=value,
                    source_k=float(k),
                    source_sigma=sigma,
                )
            self.aliases[alias] = key
            out.append(self.kernels[key])
        return out

    def compute_all_epsilons(
        self,
        Bgrid: BMultipoleGrid,
        epsilons: Iterable[tuple[int, int, int]] | None = None,
        *,
        coupling_factory: Callable[[tuple[int, int, int]], object],
        force: bool = False,
    ) -> "HKernelGrid":
        if epsilons is None:
            epsilons = self.spin_spec.representative_epsilons()
        for eps in epsilons:
            self.compute_epsilon(Bgrid, tuple(eps), coupling_factory=coupling_factory, force=force)
        return self
