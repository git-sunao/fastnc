"""Stateless strategies carried by projected 2D representations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericLOSProjectionRule:
    """Evaluate a weighted numeric 3D term through an LOS projector."""

    name: str = "numeric_los"

    def evaluate(self, representation, ell1, ell2, ell3, **params):
        return representation.projector._evaluate_numeric(
            representation.evaluate_source,
            ell1,
            ell2,
            ell3,
            sample_combination=representation.sample_combination,
            **params,
        )


NUMERIC_LOS_PROJECTION = NumericLOSProjectionRule()
