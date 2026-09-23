"""Typed additive bispectrum terms and immutable term weights."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Callable, TypeVar

from .representations import (
    BispectrumRepresentation,
    BispectrumRepresentation2D,
    BispectrumRepresentation3D,
    NumericRepresentation2D,
    NumericRepresentation3D,
)


RepresentationT = TypeVar("RepresentationT", bound=BispectrumRepresentation)
Coefficient = Number | Callable


def _coefficient_value(coefficient: Coefficient, z=None):
    if not callable(coefficient):
        return coefficient
    if z is None:
        return coefficient()
    return coefficient(z)


def _combined_coefficient(left: Coefficient, right: Coefficient, *, is_3d: bool):
    if not callable(left) and not callable(right):
        return left * right
    if is_3d:
        def combined(z):
            return _coefficient_value(left, z) * _coefficient_value(right, z)
    else:
        def combined():
            return _coefficient_value(left) * _coefficient_value(right)
    return combined


@dataclass(frozen=True)
class _BispectrumTerm:
    name: str
    representations: tuple[BispectrumRepresentation, ...]

    def __post_init__(self):
        if not self.name:
            raise ValueError("term name must not be empty")
        representations = tuple(self.representations)
        if not representations:
            raise ValueError("a term must provide at least one representation")
        if not all(isinstance(rep, BispectrumRepresentation) for rep in representations):
            raise TypeError(
                "representations must contain BispectrumRepresentation objects"
            )
        representation_types = [type(rep) for rep in representations]
        duplicates = {
            rep_type.__name__
            for rep_type in representation_types
            if representation_types.count(rep_type) > 1
        }
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"term {self.name!r} has duplicate representations: {names}")
        object.__setattr__(self, "representations", representations)

    @property
    def available_representations(self) -> tuple[type[BispectrumRepresentation], ...]:
        return tuple(type(rep) for rep in self.representations)

    def get_representation(self, representation_type: type[RepresentationT]) -> RepresentationT:
        matches = [
            rep for rep in self.representations
            if isinstance(rep, representation_type)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"term {self.name!r} has multiple "
                f"{representation_type.__name__} representations"
            )
        available = ", ".join(rep.__name__ for rep in self.available_representations)
        raise LookupError(
            f"term {self.name!r} has no {representation_type.__name__}; "
            f"available representations: {available}"
        )


@dataclass(frozen=True)
class BispectrumTerm3D(_BispectrumTerm):
    """One indivisible additive contribution to a 3D bispectrum."""

    def __post_init__(self):
        super().__post_init__()
        if not all(
            isinstance(rep, BispectrumRepresentation3D)
            for rep in self.representations
        ):
            raise TypeError("a 3D term can contain only 3D representations")

    def evaluate_numeric(self, k1, k2, k3, z, **params):
        expression = self.get_representation(NumericRepresentation3D)
        return expression.evaluate(k1, k2, k3, z, **params)

    __call__ = evaluate_numeric

    def scaled_by(self, coefficient: Coefficient):
        return WeightedTerm3D(coefficient=coefficient, term=self)

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        from .bispectrum import Bispectrum3D
        return Bispectrum3D.from_components(self, other)


@dataclass(frozen=True)
class BispectrumTerm2D(_BispectrumTerm):
    """One indivisible additive contribution to a 2D bispectrum."""

    def __post_init__(self):
        super().__post_init__()
        if not all(
            isinstance(rep, BispectrumRepresentation2D)
            for rep in self.representations
        ):
            raise TypeError("a 2D term can contain only 2D representations")

    def evaluate_numeric(self, ell1, ell2, ell3, **params):
        expression = self.get_representation(NumericRepresentation2D)
        return expression.evaluate(ell1, ell2, ell3, **params)

    __call__ = evaluate_numeric

    def scaled_by(self, coefficient: Coefficient):
        return WeightedTerm2D(coefficient=coefficient, term=self)

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        from .bispectrum import Bispectrum2D
        return Bispectrum2D.from_components(self, other)


@dataclass(frozen=True)
class WeightedTerm3D:
    coefficient: Coefficient
    term: BispectrumTerm3D

    def __post_init__(self):
        if not isinstance(self.term, BispectrumTerm3D):
            raise TypeError("term must be a BispectrumTerm3D")
        if not callable(self.coefficient) and not isinstance(self.coefficient, Number):
            raise TypeError("coefficient must be numeric or callable as coefficient(z)")

    def evaluate_numeric(self, k1, k2, k3, z, **params):
        coefficient = _coefficient_value(self.coefficient, z)
        return coefficient * self.term.evaluate_numeric(k1, k2, k3, z, **params)

    __call__ = evaluate_numeric

    def scaled_by(self, coefficient: Coefficient):
        return WeightedTerm3D(
            _combined_coefficient(self.coefficient, coefficient, is_3d=True),
            self.term,
        )

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        from .bispectrum import Bispectrum3D
        return Bispectrum3D.from_components(self, other)


@dataclass(frozen=True)
class WeightedTerm2D:
    coefficient: Coefficient
    term: BispectrumTerm2D

    def __post_init__(self):
        if not isinstance(self.term, BispectrumTerm2D):
            raise TypeError("term must be a BispectrumTerm2D")
        if not callable(self.coefficient) and not isinstance(self.coefficient, Number):
            raise TypeError("coefficient must be numeric or callable with no arguments")

    def evaluate_numeric(self, ell1, ell2, ell3, **params):
        coefficient = _coefficient_value(self.coefficient)
        return coefficient * self.term.evaluate_numeric(ell1, ell2, ell3, **params)

    __call__ = evaluate_numeric

    def scaled_by(self, coefficient: Coefficient):
        return WeightedTerm2D(
            _combined_coefficient(self.coefficient, coefficient, is_3d=False),
            self.term,
        )

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        from .bispectrum import Bispectrum2D
        return Bispectrum2D.from_components(self, other)
