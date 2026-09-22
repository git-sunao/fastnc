"""Configuration for numeric angular-bispectrum multipoles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NumericMultipoleConfig:
    """Angular quadrature and Fourier-basis conventions."""

    mode_max: int = 30
    n_angle: int = 257
    delta_beta_min: float = 5.0e-4
    delta_beta_max: float = np.pi - 5.0e-4
    basis: str = "fourier-even"
    decomposition_angle: str = "outer"
    method: str = "gauss-legendre"

    def __post_init__(self):
        if int(self.mode_max) < 0:
            raise ValueError("mode_max must be non-negative")
        if int(self.n_angle) < 2:
            raise ValueError("n_angle must be at least two")
        if not (
            0.0 <= float(self.delta_beta_min)
            < float(self.delta_beta_max)
            <= np.pi
        ):
            raise ValueError(
                "require 0 <= delta_beta_min < delta_beta_max <= pi"
            )
        if self.basis not in {"fourier-even", "fourier"}:
            raise ValueError("basis must be 'fourier-even' or 'fourier'")
        if self.decomposition_angle not in {"outer", "inner"}:
            raise ValueError("decomposition_angle must be 'outer' or 'inner'")
        if self.method not in {"gauss-legendre", "linear", "riemann"}:
            raise ValueError("unsupported decomposition method")
        object.__setattr__(self, "mode_max", int(self.mode_max))
        object.__setattr__(self, "n_angle", int(self.n_angle))
        object.__setattr__(
            self, "delta_beta_min", float(self.delta_beta_min)
        )
        object.__setattr__(
            self, "delta_beta_max", float(self.delta_beta_max)
        )

    @property
    def delta_beta(self) -> np.ndarray:
        return np.linspace(
            self.delta_beta_min,
            self.delta_beta_max,
            self.n_angle,
        )

    @property
    def modes(self) -> np.ndarray:
        if self.basis == "fourier":
            return np.arange(-self.mode_max, self.mode_max + 1)
        return np.arange(0, self.mode_max + 1)
