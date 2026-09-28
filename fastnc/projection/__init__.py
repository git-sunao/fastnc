"""Route-independent line-of-sight projection primitives."""

from .coefficient_los import integrate_coefficients
from .geometry import angular_to_comoving, validate_los_coordinates
from .kernels import Kernel1D, KernelSet
from .projector import LOSProjector
from .representations import (
    ProjectedNumericRepresentation2D,
    ProjectedSlepianRepresentation2D,
    ProjectedSemiAnalyticRepresentation2D,
)
from .strategies import NumericLOSProjectionRule

__all__ = [
    "Kernel1D",
    "KernelSet",
    "LOSProjector",
    "ProjectedNumericRepresentation2D",
    "ProjectedSlepianRepresentation2D",
    "ProjectedSemiAnalyticRepresentation2D",
    "NumericLOSProjectionRule",
    "angular_to_comoving",
    "integrate_coefficients",
    "validate_los_coordinates",
]
