"""Spin and mode bookkeeping for projected spin-field 3PCFs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpinTriple:
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


def as_spin_triple(sigma: tuple[int, int, int] | SpinTriple) -> SpinTriple:
    return sigma if isinstance(sigma, SpinTriple) else SpinTriple(tuple(sigma))
