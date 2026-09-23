"""2D representations produced by projecting 3D bispectrum terms."""
from __future__ import annotations

from dataclasses import dataclass

from fastnc.bispectrum import (
    NumericRepresentation2D,
    NumericRepresentation3D,
    WeightedTerm3D,
)

from .strategies import NUMERIC_LOS_PROJECTION, NumericLOSProjectionRule


@dataclass(frozen=True)
class ProjectedNumericRepresentation2D(NumericRepresentation2D):
    """Deferred numeric LOS projection of one weighted 3D term.

    The object retains the source term, the exact source representation, and
    the projector configuration. Evaluation applies the projection rule at
    the requested angular triangle, so source-state updates remain visible.
    """

    source_term: WeightedTerm3D
    source_representation: NumericRepresentation3D
    projector: object
    sample_combination: tuple[str, ...] | None = None
    projection_rule: NumericLOSProjectionRule = NUMERIC_LOS_PROJECTION

    def __post_init__(self):
        if not isinstance(self.source_term, WeightedTerm3D):
            raise TypeError("source_term must be a WeightedTerm3D")
        if not isinstance(
            self.source_representation, NumericRepresentation3D
        ):
            raise TypeError(
                "source_representation must be a NumericRepresentation3D"
            )
        if self.sample_combination is not None:
            object.__setattr__(
                self,
                "sample_combination",
                tuple(self.sample_combination),
            )
        if not hasattr(self.projector, "_evaluate_numeric"):
            raise TypeError("projector must provide numeric projection")

    def evaluate(self, ell1, ell2, ell3, **params):
        return self.projection_rule.evaluate(
            self, ell1, ell2, ell3, **params
        )

    def evaluate_source(self, k1, k2, k3, z, **params):
        coefficient = self.source_term.coefficient
        if callable(coefficient):
            coefficient = coefficient(z)
        return coefficient * self.source_representation.evaluate(
            k1, k2, k3, z, **params
        )

    __call__ = evaluate
