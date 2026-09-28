"""Representation-level rules for projecting 3D terms into 2D terms."""
from __future__ import annotations

from functools import singledispatch

from fastnc.bispectrum import (
    BispectrumTerm2D,
    NumericRepresentation3D,
    SemiAnalyticRepresentation3D,
    SlepianRepresentation3D,
    WeightedTerm3D,
)

from .representations import (
    ProjectedNumericRepresentation2D,
    ProjectedSemiAnalyticRepresentation2D,
    ProjectedSlepianRepresentation2D,
)


@singledispatch
def project_representation(
    representation,
    *,
    source_term: WeightedTerm3D,
    projector,
    sample_combination=None,
):
    """Project one 3D representation or fail without dropping information."""
    raise NotImplementedError(
        "no LOS projection rule is registered for "
        f"{type(representation).__name__}"
    )


@project_representation.register
def _project_numeric(
    representation: NumericRepresentation3D,
    *,
    source_term: WeightedTerm3D,
    projector,
    sample_combination=None,
):
    return ProjectedNumericRepresentation2D(
        source_term=source_term,
        source_representation=representation,
        projector=projector,
        sample_combination=sample_combination,
    )


@project_representation.register
def _project_slepian(
    representation: SlepianRepresentation3D,
    *,
    source_term: WeightedTerm3D,
    projector,
    sample_combination=None,
):
    return ProjectedSlepianRepresentation2D(
        source_term=source_term,
        source_representation=representation,
        projector=projector,
        sample_combination=sample_combination,
    )


@project_representation.register
def _project_semi_analytic(
    representation: SemiAnalyticRepresentation3D,
    *,
    source_term: WeightedTerm3D,
    projector,
    sample_combination=None,
):
    return ProjectedSemiAnalyticRepresentation2D(
        source_term=source_term,
        source_representation=representation,
        projector=projector,
        sample_combination=sample_combination,
    )


def project_term(source_term, *, projector, sample_combination=None):
    """Project every representation of one weighted 3D term."""
    if not isinstance(source_term, WeightedTerm3D):
        raise TypeError("source_term must be a WeightedTerm3D")
    representations = tuple(
        project_representation(
            representation,
            source_term=source_term,
            projector=projector,
            sample_combination=sample_combination,
        )
        for representation in source_term.term.representations
    )
    return BispectrumTerm2D(
        name=source_term.term.name,
        representations=representations,
    )
