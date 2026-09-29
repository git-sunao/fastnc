"""Finite Fourier expansions of angular basis functions."""
from __future__ import annotations

from math import comb
from typing import Literal


CouplingBasis = Literal["cosine", "sine", "fourier", "legendre"]


def basis_fourier_terms(
    mode: int,
    basis: CouplingBasis,
) -> tuple[tuple[int, float | complex], ...]:
    r"""Expand one requested angular basis function into Fourier primitives.

    The returned pairs satisfy :math:`P_L(\phi)=\sum_m a_m e^{im\phi}` for
    ``fourier``, ``cosine``, ``sine``, or Legendre
    :math:`P_L(\cos\phi)`. Coupling caches therefore store only primitive
    Fourier kernels, whose analytic support contains exact zeros; other bases
    are finite resummations and require no independent zero threshold.
    """
    mode = int(mode)
    if basis == "fourier":
        return ((mode, 1.0),)
    if basis == "cosine":
        if mode < 0:
            raise ValueError("cosine modes must be non-negative")
        if mode == 0:
            return ((0, 1.0),)
        return ((mode, 0.5), (-mode, 0.5))
    if basis == "sine":
        if mode <= 0:
            raise ValueError("sine modes must be positive")
        return ((mode, 1.0 / (2.0j)), (-mode, -1.0 / (2.0j)))
    if basis == "legendre":
        if mode < 0:
            raise ValueError("Legendre modes must be non-negative")
        denominator = float(4**mode)
        return tuple(
            (
                mode - 2 * index,
                comb(2 * index, index)
                * comb(2 * mode - 2 * index, mode - index)
                / denominator,
            )
            for index in range(mode + 1)
        )
    raise ValueError(
        "basis must be 'cosine', 'sine', 'fourier', or 'legendre'"
    )
