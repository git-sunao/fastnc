"""Explicit composition of additive bispectrum terms."""
from __future__ import annotations

from dataclasses import dataclass

from .representations import NumericRepresentation2D
from .terms import BispectrumTerm2D, WeightedTerm2D


@dataclass(frozen=True)
class NumericSumRepresentation2D(NumericRepresentation2D):
    """Numeric sum of coefficient-bearing 2D terms."""

    components: tuple[WeightedTerm2D, ...]

    def __post_init__(self):
        components = tuple(self.components)
        if not components:
            raise ValueError("numeric sum requires at least one component")
        if not all(isinstance(item, WeightedTerm2D) for item in components):
            raise TypeError("components must be WeightedTerm2D objects")
        for item in components:
            item.term.get_representation(NumericRepresentation2D)
        object.__setattr__(self, "components", components)

    def evaluate(self, ell1, ell2, ell3, **params):
        return sum(
            component.evaluate_numeric(ell1, ell2, ell3, **params)
            for component in self.components
        )

    __call__ = evaluate


def combine_numeric_terms(
    bispectrum,
    *,
    name: str,
    terms,
    keep_unselected: bool = True,
):
    """Replace selected terms by one explicitly numeric composite term."""
    from .bispectrum import Bispectrum2D

    if not isinstance(bispectrum, Bispectrum2D):
        raise TypeError("bispectrum must be a Bispectrum2D")
    if not name:
        raise ValueError("name must not be empty")
    if isinstance(terms, str):
        raise TypeError("terms must be an iterable of term names, not a string")
    requested = tuple(terms)
    if len(requested) < 2:
        raise ValueError("terms must contain at least two term names")
    if len(set(requested)) != len(requested):
        raise ValueError("terms must not contain duplicate names")

    matches = {
        term_name: [
            item
            for item in bispectrum.weighted_terms
            if item.term.name == term_name
        ]
        for term_name in requested
    }
    missing = [term_name for term_name, items in matches.items() if not items]
    ambiguous = [
        term_name for term_name, items in matches.items() if len(items) > 1
    ]
    if missing:
        raise KeyError(f"unknown 2D bispectrum term(s): {missing}")
    if ambiguous:
        raise ValueError(f"term names are not unique: {ambiguous}")

    selected = tuple(matches[term_name][0] for term_name in requested)
    selected_ids = {id(item) for item in selected}
    unselected_names = {
        item.term.name
        for item in bispectrum.weighted_terms
        if id(item) not in selected_ids
    }
    if keep_unselected and name in unselected_names:
        raise ValueError(
            f"combined term name {name!r} conflicts with an unselected term"
        )
    combined = BispectrumTerm2D(
        name=name,
        representations=(NumericSumRepresentation2D(selected),),
    )
    if not keep_unselected:
        output = [combined]
    else:
        first_index = min(
            index
            for index, item in enumerate(bispectrum.weighted_terms)
            if id(item) in selected_ids
        )
        output = []
        for index, item in enumerate(bispectrum.weighted_terms):
            if index == first_index:
                output.append(combined)
            if id(item) not in selected_ids:
                output.append(item)

    return Bispectrum2D(
        output,
        support=bispectrum.support,
        _revision_sources=(lambda: bispectrum.state_token,),
    )
