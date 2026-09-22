"""Term aggregates defining physical 3D and angular 2D bispectra."""
from __future__ import annotations

from .support import Support2D, Support3D
from .terms import (
    BispectrumTerm2D,
    BispectrumTerm3D,
    Coefficient,
    WeightedTerm2D,
    WeightedTerm3D,
)


class Bispectrum3D:
    """A typed aggregate of additive 3D bispectrum terms."""

    def __init__(
        self,
        terms,
        *,
        support: Support3D | None = None,
        _revision_sources=None,
    ):
        self._weighted_terms = self._normalize_components(terms)
        if not self._weighted_terms:
            raise ValueError("a 3D bispectrum must contain at least one term")
        self._terms = tuple(item.term for item in self._weighted_terms)
        self.support = support or Support3D(policy="ignore")
        self._state_revision = 0
        self._revision_sources = (
            (lambda: self._state_revision,)
            if _revision_sources is None
            else tuple(_revision_sources)
        )

    @staticmethod
    def _normalize_components(components):
        weighted_terms = []
        for component in tuple(components):
            if isinstance(component, WeightedTerm3D):
                weighted_terms.append(component)
            elif isinstance(component, BispectrumTerm3D):
                weighted_terms.append(WeightedTerm3D(1.0, component))
            else:
                raise TypeError("3D components must be 3D terms or weighted 3D terms")
        return tuple(weighted_terms)

    @classmethod
    def from_components(cls, *components, support: Support3D | None = None):
        weighted_terms = []
        revision_sources = []
        for component in components:
            if isinstance(component, Bispectrum3D):
                weighted_terms.extend(component.weighted_terms)
                revision_sources.extend(component._revision_sources)
            elif isinstance(component, (BispectrumTerm3D, WeightedTerm3D)):
                weighted_terms.append(component)
            elif isinstance(component, (Bispectrum2D, BispectrumTerm2D, WeightedTerm2D)):
                raise TypeError("2D and 3D bispectrum components cannot be combined")
            else:
                raise TypeError(
                    "components must be Bispectrum3D, BispectrumTerm3D, "
                    "or WeightedTerm3D objects"
                )
        return Bispectrum3D(
            weighted_terms,
            support=support,
            _revision_sources=revision_sources or None,
        )

    @property
    def weighted_terms(self) -> tuple[WeightedTerm3D, ...]:
        return self._weighted_terms

    @property
    def terms(self) -> tuple[BispectrumTerm3D, ...]:
        return self._terms

    def iter_terms(self):
        return iter(self._weighted_terms)

    @property
    def state_revision(self) -> int:
        return self._state_revision

    @property
    def state_token(self) -> tuple[int, ...]:
        """Revision token including every physical source of this aggregate."""
        return tuple(source() for source in self._revision_sources)

    def _state_updated(self):
        self._state_revision += 1

    def evaluate_numeric(self, k1, k2, k3, z, **params):
        return sum(
            item.evaluate_numeric(k1, k2, k3, z, **params)
            for item in self._weighted_terms
        )

    def evaluate(self, k1, k2, k3, z, **params):
        return self.evaluate_numeric(k1, k2, k3, z, **params)

    __call__ = evaluate

    def select_terms(self, *names: str):
        requested = set(names)
        selected = [item for item in self._weighted_terms if item.term.name in requested]
        found = {item.term.name for item in selected}
        missing = requested - found
        if missing:
            raise KeyError(f"unknown 3D bispectrum term(s): {sorted(missing)}")
        return Bispectrum3D(
            selected,
            support=self.support,
            _revision_sources=self._revision_sources,
        )

    def scaled_by(self, coefficient: Coefficient):
        return Bispectrum3D(
            [item.scaled_by(coefficient) for item in self._weighted_terms],
            support=self.support,
            _revision_sources=self._revision_sources,
        )

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        other_support = getattr(other, "support", self.support)
        if other_support != self.support:
            raise ValueError("3D bispectra with different support cannot be combined")
        return Bispectrum3D.from_components(self, other, support=self.support)


class Bispectrum2D:
    """A typed aggregate of additive angular bispectrum terms."""

    def __init__(
        self,
        terms,
        *,
        support: Support2D | None = None,
        _revision_sources=None,
    ):
        self._weighted_terms = self._normalize_components(terms)
        if not self._weighted_terms:
            raise ValueError("a 2D bispectrum must contain at least one term")
        self._terms = tuple(item.term for item in self._weighted_terms)
        self.support = support or Support2D(policy="ignore")
        self._state_revision = 0
        self._revision_sources = (
            (lambda: self._state_revision,)
            if _revision_sources is None
            else tuple(_revision_sources)
        )

    @staticmethod
    def _normalize_components(components):
        weighted_terms = []
        for component in tuple(components):
            if isinstance(component, WeightedTerm2D):
                weighted_terms.append(component)
            elif isinstance(component, BispectrumTerm2D):
                weighted_terms.append(WeightedTerm2D(1.0, component))
            else:
                raise TypeError("2D components must be 2D terms or weighted 2D terms")
        return tuple(weighted_terms)

    @classmethod
    def from_components(cls, *components, support: Support2D | None = None):
        weighted_terms = []
        revision_sources = []
        for component in components:
            if isinstance(component, Bispectrum2D):
                weighted_terms.extend(component.weighted_terms)
                revision_sources.extend(component._revision_sources)
            elif isinstance(component, (BispectrumTerm2D, WeightedTerm2D)):
                weighted_terms.append(component)
            elif isinstance(component, (Bispectrum3D, BispectrumTerm3D, WeightedTerm3D)):
                raise TypeError("2D and 3D bispectrum components cannot be combined")
            else:
                raise TypeError(
                    "components must be Bispectrum2D, BispectrumTerm2D, "
                    "or WeightedTerm2D objects"
                )
        return Bispectrum2D(
            weighted_terms,
            support=support,
            _revision_sources=revision_sources or None,
        )

    @property
    def weighted_terms(self) -> tuple[WeightedTerm2D, ...]:
        return self._weighted_terms

    @property
    def terms(self) -> tuple[BispectrumTerm2D, ...]:
        return self._terms

    def iter_terms(self):
        return iter(self._weighted_terms)

    @property
    def state_revision(self) -> int:
        return self._state_revision

    @property
    def state_token(self) -> tuple[int, ...]:
        """Revision token including every physical source of this aggregate."""
        return tuple(source() for source in self._revision_sources)

    def _state_updated(self):
        self._state_revision += 1

    def evaluate_numeric(self, ell1, ell2, ell3, **params):
        return sum(
            item.evaluate_numeric(ell1, ell2, ell3, **params)
            for item in self._weighted_terms
        )

    def evaluate(self, ell1, ell2, ell3, **params):
        return self.evaluate_numeric(ell1, ell2, ell3, **params)

    __call__ = evaluate

    def select_terms(self, *names: str):
        requested = set(names)
        selected = [item for item in self._weighted_terms if item.term.name in requested]
        found = {item.term.name for item in selected}
        missing = requested - found
        if missing:
            raise KeyError(f"unknown 2D bispectrum term(s): {sorted(missing)}")
        return Bispectrum2D(
            selected,
            support=self.support,
            _revision_sources=self._revision_sources,
        )

    def scaled_by(self, coefficient: Coefficient):
        return Bispectrum2D(
            [item.scaled_by(coefficient) for item in self._weighted_terms],
            support=self.support,
            _revision_sources=self._revision_sources,
        )

    def __mul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __rmul__(self, coefficient: Coefficient):
        return self.scaled_by(coefficient)

    def __add__(self, other):
        other_support = getattr(other, "support", self.support)
        if other_support != self.support:
            raise ValueError("2D bispectra with different support cannot be combined")
        return Bispectrum2D.from_components(self, other, support=self.support)
