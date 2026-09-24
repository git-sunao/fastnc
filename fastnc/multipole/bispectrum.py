"""Route-independent lazy evaluation of angular bispectrum multipoles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BispectrumMultipole:
    """Lazily evaluated bispectrum multipoles with a route-independent API."""

    bispectrum: Any
    calculator: Any

    @classmethod
    def from_numeric(
        cls,
        config,
        b2d,
        *,
        basis: str = "cosine",
    ) -> "BispectrumMultipole":
        """Construct lazily evaluated numeric multipoles."""
        from .numeric import NumericBispectrumMultipoleCalculator

        calculator = NumericBispectrumMultipoleCalculator(config, basis=basis)
        return cls(bispectrum=b2d, calculator=calculator)

    @property
    def basis(self) -> str:
        """Basis convention used by the retained calculator."""
        return self.calculator.basis

    def evaluate(self, mode, ell2, ell3, **params) -> np.ndarray:
        """Evaluate one mode or a one-dimensional collection of modes."""
        return self.calculator.evaluate(
            self.bispectrum,
            mode,
            ell2=ell2,
            ell3=ell3,
            **params,
        )
