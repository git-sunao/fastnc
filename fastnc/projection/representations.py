"""2D representations produced by projecting 3D bispectrum terms."""
from __future__ import annotations

from dataclasses import dataclass

from fastnc.bispectrum import (
    NumericRepresentation2D,
    NumericRepresentation3D,
    SlepianRepresentation2D,
    SlepianRepresentation3D,
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
        if not all(
            hasattr(self.projector, name)
            for name in ("z", "chi", "weight", "is_delta_like")
        ):
            raise TypeError("projector must provide LOS projection settings")

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


@dataclass(frozen=True)
class ProjectedSlepianRepresentation2D(SlepianRepresentation2D):
    """Deferred Slepian projection recipe for one weighted 3D term.

    LOS integration is deliberately not performed here. A Slepian route
    calculator consumes the retained 3D representation and projector and
    chooses coefficient-level or result-level projection.
    """

    source_term: WeightedTerm3D
    source_representation: SlepianRepresentation3D
    projector: object
    sample_combination: tuple[str, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.source_term, WeightedTerm3D):
            raise TypeError("source_term must be a WeightedTerm3D")
        if not isinstance(self.source_representation, SlepianRepresentation3D):
            raise TypeError(
                "source_representation must be a SlepianRepresentation3D"
            )
        if self.sample_combination is not None:
            object.__setattr__(
                self,
                "sample_combination",
                tuple(self.sample_combination),
            )
        if not all(
            hasattr(self.projector, name)
            for name in (
                "z",
                "chi",
                "weight",
                "integrate_coefficients",
                "is_delta_like",
            )
        ):
            raise TypeError("projector must provide LOS projection settings")
