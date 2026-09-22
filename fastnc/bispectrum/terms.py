"""Additive bispectrum terms and immutable linear composition."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Callable, TypeVar

from .representations import (
    BispectrumRepresentation,
    BispectrumRepresentation2D,
    BispectrumRepresentation3D,
    NumericExpression2D,
    NumericExpression3D,
)


RepresentationT = TypeVar("RepresentationT", bound=BispectrumRepresentation)
Coefficient = Number | Callable


def _coefficient_value(coefficient: Coefficient, z=None):
    if not callable(coefficient):
        return coefficient
    if z is None:
        return coefficient()
    return coefficient(z)


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
        types = [type(rep) for rep in representations]
        duplicates = {rep_type.__name__ for rep_type in types if types.count(rep_type) > 1}
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
                f"term {self.name!r} has multiple {representation_type.__name__} representations"
            )
        available = ", ".join(rep.__name__ for rep in self.available_representations)
        raise LookupError(
            f"term {self.name!r} has no {representation_type.__name__}; "
            f"available representations: {available}"
        )

    def scaled_by(self, coefficient: Coefficient):
        return WeightedTerm(coefficient=coefficient, term=self)

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        return CompositeBispectrum.from_components(self, other)


@dataclass(frozen=True)
class BispectrumTerm3D(_BispectrumTerm):
    """One numerically stable additive contribution to a 3D bispectrum."""

    def __post_init__(self):
        super().__post_init__()
        if not all(
            isinstance(rep, BispectrumRepresentation3D)
            for rep in self.representations
        ):
            raise TypeError("a 3D term can contain only 3D representations")

    def evaluate(self, k1, k2, k3, z, **params):
        expression = self.get_representation(NumericExpression3D)
        return expression.evaluate(k1, k2, k3, z, **params)

    __call__ = evaluate


@dataclass(frozen=True)
class BispectrumTerm2D(_BispectrumTerm):
    """One numerically stable additive contribution to a 2D bispectrum."""

    def __post_init__(self):
        super().__post_init__()
        if not all(
            isinstance(rep, BispectrumRepresentation2D)
            for rep in self.representations
        ):
            raise TypeError("a 2D term can contain only 2D representations")

    def evaluate(self, ell1, ell2, ell3, **params):
        expression = self.get_representation(NumericExpression2D)
        return expression.evaluate(ell1, ell2, ell3, **params)

    __call__ = evaluate


@dataclass(frozen=True)
class WeightedTerm:
    """A coefficient times one leaf bispectrum term."""

    coefficient: Coefficient
    term: _BispectrumTerm

    def __post_init__(self):
        if not isinstance(self.term, _BispectrumTerm):
            raise TypeError("term must be a BispectrumTerm2D or BispectrumTerm3D")
        if not callable(self.coefficient) and not isinstance(self.coefficient, Number):
            raise TypeError("coefficient must be numeric or callable")

    def scaled_by(self, coefficient: Coefficient):
        if callable(self.coefficient) or callable(coefficient):
            if isinstance(self.term, BispectrumTerm3D):
                def combined(z):
                    left = _coefficient_value(self.coefficient, z)
                    right = _coefficient_value(coefficient, z)
                    return left * right
            else:
                def combined():
                    left = _coefficient_value(self.coefficient)
                    right = _coefficient_value(coefficient)
                    return left * right
        else:
            combined = self.coefficient * coefficient
        return WeightedTerm(combined, self.term)

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        return CompositeBispectrum.from_components(self, other)

    def evaluate(self, *args, **params):
        if isinstance(self.term, BispectrumTerm3D):
            if len(args) < 4:
                raise TypeError("3D weighted-term evaluation requires k1, k2, k3, z")
            z = args[3]
            coefficient = _coefficient_value(self.coefficient, z)
        else:
            coefficient = _coefficient_value(self.coefficient)
        return coefficient * self.term.evaluate(*args, **params)

    __call__ = evaluate


@dataclass(frozen=True)
class CompositeBispectrum:
    """An immutable, flat linear combination of weighted leaf terms."""

    weighted_terms: tuple[WeightedTerm, ...]

    def __post_init__(self):
        weighted_terms = tuple(self.weighted_terms)
        if not weighted_terms:
            raise ValueError("a composite bispectrum must contain at least one term")
        dimensions = {type(item.term) for item in weighted_terms}
        if len(dimensions) != 1:
            raise TypeError("2D and 3D bispectrum terms cannot be combined")
        object.__setattr__(self, "weighted_terms", weighted_terms)

    @classmethod
    def from_components(cls, *components):
        weighted_terms = []
        for component in components:
            if isinstance(component, cls):
                weighted_terms.extend(component.weighted_terms)
            elif isinstance(component, WeightedTerm):
                weighted_terms.append(component)
            elif isinstance(component, _BispectrumTerm):
                weighted_terms.append(WeightedTerm(1.0, component))
            else:
                raise TypeError(
                    "components must be bispectrum terms, weighted terms, or composites"
                )
        return cls(tuple(weighted_terms))

    def iter_terms(self):
        return iter(self.weighted_terms)

    @property
    def terms(self):
        return tuple(item.term for item in self.weighted_terms)

    def scaled_by(self, coefficient: Coefficient):
        return CompositeBispectrum(
            tuple(item.scaled_by(coefficient) for item in self.weighted_terms)
        )

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        return CompositeBispectrum.from_components(self, other)

    def evaluate(self, *args, **params):
        return sum(item.evaluate(*args, **params) for item in self.weighted_terms)

    __call__ = evaluate
