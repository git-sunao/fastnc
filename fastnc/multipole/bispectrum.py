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

    @classmethod
    def from_hybrid(
        cls, config, b2d, *, semi_analytic_calculator, basis="fourier"
    ):
        """Construct multipoles with semi-analytic-first term dispatch."""
        from .semi_analytic import HybridBispectrumMultipoleCalculator

        return cls(
            bispectrum=b2d,
            calculator=HybridBispectrumMultipoleCalculator(
                config,
                semi_analytic_calculator=semi_analytic_calculator,
                basis=basis,
            ),
        )

    @classmethod
    def from_semi_analytic(cls, b2d, *, calculator):
        """Construct lazily evaluated coefficient-level multipoles."""
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
