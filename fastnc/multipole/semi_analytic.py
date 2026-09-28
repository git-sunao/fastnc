"""Term-wise assembly of semi-analytic and numeric multipoles."""
from __future__ import annotations

from fastnc.bispectrum import NumericRepresentation2D
from fastnc.projection import ProjectedSemiAnalyticRepresentation2D

from .numeric import NumericBispectrumMultipoleCalculator


class HybridBispectrumMultipoleCalculator:
    """Dispatch each term to semi-analytic evaluation or numeric fallback."""

    route = "hybrid"

    def __init__(self, config, *, semi_analytic_calculator, basis="fourier"):
        self.basis = str(basis)
        if self.basis != "fourier":
            raise NotImplementedError(
                "semi-analytic multipoles currently require the Fourier basis"
            )
        self.semi_analytic = semi_analytic_calculator
        self.numeric = NumericBispectrumMultipoleCalculator(config, basis=self.basis)

    @staticmethod
    def _select(source, names):
        return source.select_terms(*names) if names else None

    def evaluate(self, source, mode, *, ell2, ell3, **params):
        semi_names = []
        numeric_names = []
        for term in source.terms:
            if any(
                isinstance(rep, ProjectedSemiAnalyticRepresentation2D)
                for rep in term.representations
            ):
                semi_names.append(term.name)
            elif any(
                isinstance(rep, NumericRepresentation2D)
                for rep in term.representations
            ):
                numeric_names.append(term.name)
            else:
                raise TypeError(f"term {term.name!r} has no multipole representation")

        contributions = []
        semi_source = self._select(source, semi_names)
        if semi_source is not None:
            contributions.append(
                self.semi_analytic.evaluate(
                    semi_source, mode, ell2=ell2, ell3=ell3, **params
                )
            )
        numeric_source = self._select(source, numeric_names)
        if numeric_source is not None:
            contributions.append(
                self.numeric.evaluate(
                    numeric_source, mode, ell2=ell2, ell3=ell3, **params
                )
            )
        if not contributions:
            raise ValueError("source has no terms")
        return sum(contributions)
