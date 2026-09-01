"""Compiled/grouped Slepian term metadata for production assembly (Phase 10).

This module is numerical/calculator-side only.  Physical term definitions remain
in :mod:`fastnc.bispectrum.slepian` / :mod:`fastnc.bispectrum.models.spt`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .weber import canonical_bessel_order


@dataclass(frozen=True)
class RadialKernelKey:
    """Identity of one reusable radial object.

    ``kind`` is ``"single"`` or ``"double"``.  ``shift`` is the integer
    FFTLog exponent shift relative to the model's base P0 expansion.  Bessel
    orders are canonical non-negative orders; ``sign`` is deliberately *not*
    part of the key because it is applied during cheap assembly.
    """

    model_id: int
    leg: int
    radial_family: str
    shift: int
    kind: str
    order_x: int
    order_theta: int | None
    geometry_tag: str


@dataclass(frozen=True)
class CompiledSlepianTerm:
    term_index: int
    pair: str
    harmonic: int
    coefficient: complex
    power_shifts: tuple[int, int, int]
    other_leg: int
    angular_orders: tuple[int, int, int]


@dataclass(frozen=True)
class CompiledModePlan:
    sigma: tuple[int, int, int]
    k: float
    terms: tuple[CompiledSlepianTerm, ...]
    unique_radial_keys: tuple[RadialKernelKey, ...]

    @property
    def n_physical_terms(self):
        return len(self.terms)

    @property
    def n_unique_radial_kernels(self):
        return len(self.unique_radial_keys)


def compile_mode_plan(terms, *, sigma, k, effective_spin):
    """Compile immutable SPT term metadata and canonical radial identities.

    The plan does not evaluate physics.  It only records mathematical identity
    so the calculator can prepare each expensive radial kernel once.
    """
    m, n = effective_spin.bessel_orders(float(k))
    compiled = []
    keys = []
    for it, term in enumerate(terms):
        n1, n2, n3 = term.validated_angular_orders()
        p = n2 + effective_spin.sigma2 - m
        q = n3 + effective_spin.sigma3 - n
        shifts = tuple(int(v) for v in getattr(term, "power_shifts", (0, 0, 0)))
        model_id = id(getattr(term, "model", term))
        compiled.append(
            CompiledSlepianTerm(
                term_index=it,
                pair=str(getattr(term, "pair", "")),
                harmonic=int(getattr(term, "harmonic", 0)),
                coefficient=complex(getattr(term, "coefficient", 1.0)),
                power_shifts=shifts,
                other_leg=int(getattr(term, "other_leg", -1)),
                angular_orders=(int(n1), int(n2), int(n3)),
            )
        )
        o1, _ = canonical_bessel_order(n1 + effective_spin.sigma1)
        op, _ = canonical_bessel_order(p)
        om, _ = canonical_bessel_order(m)
        oq, _ = canonical_bessel_order(q)
        on, _ = canonical_bessel_order(n)
        keys.append(RadialKernelKey(model_id, 0, getattr(term.radial_metadata(0), "family", "generic"), shifts[0], "single", o1, None, "x"))
        # Keep the other leg in the compiled identity.  A constant leg may
        # still have a non-zero Jacobi regular part for general spin, so it is
        # not mathematically absent even when it also carries a contact term.
        keys.append(RadialKernelKey(
            model_id, 1, getattr(term.radial_metadata(1), "family", "generic"),
            shifts[1], "double", op, om, "theta-x"
        ))
        keys.append(RadialKernelKey(
            model_id, 2, getattr(term.radial_metadata(2), "family", "generic"),
            shifts[2], "double", oq, on, "theta-x"
        ))
    unique = tuple(dict.fromkeys(keys))
    return CompiledModePlan(
        sigma=tuple(int(v) for v in sigma), k=float(k), terms=tuple(compiled),
        unique_radial_keys=unique,
    )


__all__ = [
    "RadialKernelKey", "CompiledSlepianTerm", "CompiledModePlan", "compile_mode_plan",
]
