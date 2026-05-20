"""Support-range utilities for bispectrum models."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Support3D:
    k_min: float = 0.0
    k_max: float = np.inf
    z_min: float = 0.0
    z_max: float = np.inf
    policy: str = "zero"  # zero, raise, clip, ignore

    def contains(self, k1, k2, k3, z):
        return ((self.k_min <= k1) & (k1 <= self.k_max) &
                (self.k_min <= k2) & (k2 <= self.k_max) &
                (self.k_min <= k3) & (k3 <= self.k_max) &
                (self.z_min <= z) & (z <= self.z_max))


@dataclass(frozen=True)
class Support2D:
    ell_min: float = 0.0
    ell_max: float = np.inf
    alpha_min: float = 0.0
    alpha_max: float = np.pi
    policy: str = "zero"  # zero, raise, clip, ignore

    def contains_sides(self, ell1, ell2, ell3):
        return ((self.ell_min <= ell1) & (ell1 <= self.ell_max) &
                (self.ell_min <= ell2) & (ell2 <= self.ell_max) &
                (self.ell_min <= ell3) & (ell3 <= self.ell_max))
