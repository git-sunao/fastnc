"""Spin and natural-component bookkeeping for projected spin-field 3PCFs.

The physical field spin ``s`` and the effective spin ``sigma`` are distinct:

    sigma_i = epsilon_i * s_i.

User-facing 3PCF objects should be configured by the physical spin triple.
For each independent natural component, this module provides the corresponding
representative conjugation vector ``epsilon`` and effective spin ``sigma``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass(frozen=True)
class EffectiveSpinTriple:
    """Effective spin triple sigma=(sigma1,sigma2,sigma3)."""

    sigma: tuple[int, int, int]
    atol: float = 1.0e-12

    def __post_init__(self):
        if len(tuple(self.sigma)) != 3:
            raise ValueError("sigma must contain exactly three entries.")
        object.__setattr__(self, "sigma", tuple(int(x) for x in self.sigma))

    @property
    def sigma1(self) -> int:
        return self.sigma[0]

    @property
    def sigma2(self) -> int:
        return self.sigma[1]

    @property
    def sigma3(self) -> int:
        return self.sigma[2]

    @property
    def Sigma(self) -> int:
        return int(sum(self.sigma))

    def nu(self, k: float | np.ndarray):
        """Return nu_k = k + (sigma2-sigma1)/2."""
        return np.asarray(k, dtype=float) + 0.5 * (self.sigma2 - self.sigma1)

    def bessel_orders(self, k: float) -> tuple[int, int]:
        """Return integer Bessel orders m_k and n_k for an allowed k."""
        kf = float(k)
        m = 0.5 * self.Sigma + kf
        n = 0.5 * self.Sigma - kf
        mr = int(round(m))
        nr = int(round(n))
        if abs(m - mr) > self.atol or abs(n - nr) > self.atol:
            raise ValueError(f"k={k} is not allowed for sigma={self.sigma}; m={m}, n={n} are not both integers.")
        return mr, nr

    def allowed(self, k: float) -> bool:
        try:
            self.bessel_orders(k)
            return True
        except ValueError:
            return False

    def k_values(self, kmax: float) -> np.ndarray:
        """Return all allowed integer/half-integer k with |k|<=kmax."""
        kmax = float(kmax)
        two_k_max = int(np.floor(2.0 * kmax + self.atol))
        vals = []
        for two_k in range(-two_k_max, two_k_max + 1):
            k = 0.5 * two_k
            if abs(k) <= kmax + self.atol and self.allowed(k):
                vals.append(k)
        return np.asarray(vals, dtype=float)


def as_effective_spin_triple(sigma: tuple[int, int, int] | EffectiveSpinTriple) -> EffectiveSpinTriple:
    return sigma if isinstance(sigma, EffectiveSpinTriple) else EffectiveSpinTriple(tuple(sigma))


@dataclass(frozen=True)
class ComponentSpec:
    """One independent natural-component representative."""

    index: int
    epsilon: tuple[int, int, int]
    sigma: tuple[int, int, int]
    spin: tuple[int, int, int]


@dataclass(frozen=True)
class SpinSpec:
    """Physical spin triple and its independent natural components.

    Parameters
    ----------
    spin
        Physical field-spin triple ``s=(s1,s2,s3)``.  Scalar entries have
        ``s_i=0`` and their epsilon labels are redundant.

    Notes
    -----
    The all-flipped epsilon labels are treated as complex-conjugate partners.
    Scalar epsilons are fixed to +1.  Among each conjugate pair, the
    representative is chosen to have the smaller number of minus signs on
    active spin entries.  For an even number of active spins, ties are broken
    by putting the first minus sign as early as possible.  The resulting order
    is all-plus first, followed by single-minus representatives in field order
    when ``spin=(2,2,2)``.
    """

    spin: tuple[int, int, int]

    def __post_init__(self):
        if len(tuple(self.spin)) != 3:
            raise ValueError("spin must contain exactly three entries.")
        object.__setattr__(self, "spin", tuple(int(x) for x in self.spin))

    @property
    def active_indices(self) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.spin) if s != 0)

    @property
    def nspin(self) -> int:
        return len(self.active_indices)

    @property
    def n_components(self) -> int:
        return 1 if self.nspin == 0 else 2 ** (self.nspin - 1)

    def sigma_from_epsilon(self, epsilon: tuple[int, int, int]) -> tuple[int, int, int]:
        eps = _validate_epsilon(epsilon)
        return tuple(int(e * s) for e, s in zip(eps, self.spin))

    def representative_epsilons(self) -> tuple[tuple[int, int, int], ...]:
        """Return independent epsilon representatives in component order.

        For ``spin=(2,2,2)``, the order is::

            component 0: (+1, +1, +1)
            component 1: (-1, +1, +1)
            component 2: (+1, -1, +1)
            component 3: (+1, +1, -1)

        More generally, scalar entries are fixed to +1.  For each conjugate
        pair ``epsilon ~ -epsilon`` on the active spin entries, the stored
        representative is the one with fewer minus signs.  If the number of
        active spin entries is even and the pair is tied, the representative
        with the earliest minus sign is chosen.
        """
        if self.nspin == 0:
            return ((1, 1, 1),)

        reps: set[tuple[int, int, int]] = set()
        active = self.active_indices
        for signs in product((1, -1), repeat=self.nspin):
            eps = [1, 1, 1]
            for i, e in zip(active, signs):
                eps[i] = int(e)
            rep, _ = self._canonicalize_active_epsilon(tuple(eps))
            reps.add(rep)

        return tuple(sorted(reps, key=self._component_order_key))

    def components(self) -> tuple[ComponentSpec, ...]:
        return tuple(
            ComponentSpec(index=i, epsilon=eps, sigma=self.sigma_from_epsilon(eps), spin=self.spin)
            for i, eps in enumerate(self.representative_epsilons())
        )

    def component(self, index: int) -> ComponentSpec:
        comps = self.components()
        i = int(index)
        if i < 0 or i >= len(comps):
            raise IndexError(f"component index {index} is out of range for spin={self.spin}; n_components={len(comps)}.")
        return comps[i]

    def canonicalize_epsilon(self, epsilon: tuple[int, int, int]) -> tuple[tuple[int, int, int], bool]:
        """Return representative epsilon and whether conjugation is needed.

        Scalar epsilon entries are set to +1.  The representative is then
        selected from the conjugate pair ``epsilon ~ -epsilon`` using the same
        ordering convention as :meth:`representative_epsilons`.  If the selected
        representative is the all-active flip of the requested epsilon, the
        requested component is the complex conjugate of the stored one and the
        returned boolean is ``True``.
        """
        eps = list(_validate_epsilon(epsilon))
        for i, s in enumerate(self.spin):
            if s == 0:
                eps[i] = 1

        if self.nspin == 0:
            return (1, 1, 1), False

        return self._canonicalize_active_epsilon(tuple(eps))

    def _canonicalize_active_epsilon(self, epsilon: tuple[int, int, int]) -> tuple[tuple[int, int, int], bool]:
        eps = tuple(epsilon)
        flipped = list(eps)
        for i in self.active_indices:
            flipped[i] *= -1
        flipped = tuple(flipped)

        eps_key = self._representative_choice_key(eps)
        flipped_key = self._representative_choice_key(flipped)
        if flipped_key < eps_key:
            return flipped, True
        return eps, False

    def _representative_choice_key(self, epsilon: tuple[int, int, int]) -> tuple[int, int]:
        active_signs = [epsilon[i] for i in self.active_indices]
        nminus = sum(e < 0 for e in active_signs)
        first_minus = next((j for j, e in enumerate(active_signs) if e < 0), self.nspin)
        return nminus, first_minus

    def _component_order_key(self, epsilon: tuple[int, int, int]) -> tuple[int, int]:
        active_signs = [epsilon[i] for i in self.active_indices]
        nminus = sum(e < 0 for e in active_signs)
        first_minus = next((j for j, e in enumerate(active_signs) if e < 0), self.nspin)
        return nminus, first_minus

    def component_index_from_epsilon(self, epsilon: tuple[int, int, int]) -> tuple[int, bool]:
        rep, conjugated = self.canonicalize_epsilon(epsilon)
        for comp in self.components():
            if comp.epsilon == rep:
                return comp.index, conjugated
        raise RuntimeError(f"Internal component bookkeeping error for epsilon={epsilon}, representative={rep}.")


def _validate_epsilon(epsilon: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(tuple(epsilon)) != 3:
        raise ValueError("epsilon must contain exactly three entries.")
    eps = tuple(int(e) for e in epsilon)
    if any(e not in (-1, 1) for e in eps):
        raise ValueError("epsilon entries must be +1 or -1.")
    return eps


def independent_epsilons(spin: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    return SpinSpec(spin).representative_epsilons()


def component_specs(spin: tuple[int, int, int]) -> tuple[ComponentSpec, ...]:
    return SpinSpec(spin).components()
