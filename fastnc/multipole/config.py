"""Configuration for numeric angular-bispectrum multipoles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class NumericMultipoleConfig:
    """Angular sampling, quadrature, and basis conventions."""

    n_angle: int = 257
    delta_beta_min: float = 5.0e-4
    delta_beta_max: float = np.pi - 5.0e-4
    basis: Literal["cosine", "sine", "fourier"] = "cosine"
    decomposition_angle: str = "outer"
    method: str = "gauss-legendre"

    def __post_init__(self):
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
        basis = "cosine" if self.basis == "fourier-even" else self.basis
        if basis not in {"cosine", "sine", "fourier"}:
            raise ValueError("basis must be 'cosine', 'sine', or 'fourier'")
        if self.decomposition_angle not in {"outer", "inner"}:
            raise ValueError("decomposition_angle must be 'outer' or 'inner'")
        if self.method not in {"gauss-legendre", "linear", "riemann"}:
            raise ValueError("unsupported decomposition method")
        object.__setattr__(self, "n_angle", int(self.n_angle))
        object.__setattr__(self, "delta_beta_min", float(self.delta_beta_min))
        object.__setattr__(self, "delta_beta_max", float(self.delta_beta_max))
        object.__setattr__(self, "basis", basis)

    @property
    def delta_beta(self) -> np.ndarray:
        return np.linspace(
            self.delta_beta_min,
            self.delta_beta_max,
            self.n_angle,
        )
