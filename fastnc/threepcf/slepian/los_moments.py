"""LOS moment rules for the Slepian route.

Phase 8 introduced the exact factorized-growth SPT rule for one sample
combination. Phase 9 promotes the sample combination to a leading batch
dimension so all requested LOS weights share the same exponent lattice,
growth array, coefficient factor, and quadrature work.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class PreparedLOSMoments:
    exponent_sums: np.ndarray
    values: np.ndarray
    unique_exponents: np.ndarray


@dataclass(frozen=True)
class PreparedBatchedLOSMoments:
    exponent_sums: np.ndarray
    values: np.ndarray
    unique_exponents: np.ndarray
    sample_combinations: tuple


class SlepianLOSMomentRule:
    """Abstract numerical LOS rule consumed by the Slepian calculator."""

    def prepare(self, *args, **kwargs):
        raise NotImplementedError


def _normalize_combo(combo):
    return None if combo is None else tuple(combo)


class FactorizedGrowthBatchMomentRule(SlepianLOSMomentRule):
    r"""Batched exact SPT moment rule for ``P(k,z)=D(z)^2 P0(k)``.

    The sample axis is vectorized. Only unique FFTLog exponent sums are
    integrated, so an ``N``-mode lattice needs ``2*N-1`` quadratures rather
    than ``N_sample*(2*N-1)``.
    """

    def __init__(self, model, projector, sample_combinations):
        if not getattr(model, "has_factorized_growth", False):
            raise RuntimeError(
                "FactorizedGrowthBatchMomentRule requires an explicit factorized-growth capability"
            )
        if float(getattr(projector, "l_shift", 0.0)) != 0.0:
            raise NotImplementedError("Phase-9 Slepian LOS moments require projector.l_shift == 0")
        combos = tuple(_normalize_combo(c) for c in sample_combinations)
        if not combos:
            raise ValueError("at least one sample combination is required")
        self.model = model
        self.projector = projector
        self.sample_combinations = combos
        self._cache = {}
        self.integral_evaluations = 0

    @staticmethod
    def _canonical_complex(z):
        z = complex(z)
        return (round(z.real, 13), round(z.imag, 13))

    def weight_matrix(self):
        return np.asarray(
            [self.projector.los_weight(c) for c in self.sample_combinations],
            dtype=complex,
        )

    def _state_signature(self, coefficient_factor):
        p = self.projector
        _, growth = self.model.factorized_linear_power()
        z = np.asarray(p.z, dtype=float)
        chi = np.asarray(p.chi, dtype=float)
        weights = self.weight_matrix()
        g = np.asarray(growth(z), dtype=complex)
        a = np.asarray(coefficient_factor(z), dtype=complex)
        h = hashlib.blake2b(digest_size=16)
        for arr in (z, chi, weights, g, a):
            aa = np.ascontiguousarray(arr)
            h.update(str(aa.dtype).encode())
            h.update(str(aa.shape).encode())
            h.update(aa.view(np.uint8))
        h.update(repr(self.sample_combinations).encode())
        return h.digest(), weights, g, a

    def prepare(self, exponents_a, exponents_b, *, coefficient_factor=lambda z: 1.0):
        ea = np.asarray(exponents_a, dtype=complex)
        eb = np.asarray(exponents_b, dtype=complex)
        sums = ea[:, None] + eb[None, :]
        state, weights, growth, amp = self._state_signature(coefficient_factor)
        flat_keys = [self._canonical_complex(v) for v in sums.ravel()]
        unique_keys = list(dict.fromkeys(flat_keys))
        cache_key = (state, tuple(unique_keys))
        vals = self._cache.get(cache_key)
        if vals is None:
            chi = np.asarray(self.projector.chi, dtype=float)
            common = weights * (growth**4 * amp)[None, :]
            vals = {}
            for key in unique_keys:
                lam = complex(*key)
                vals[key] = np.trapezoid(common * chi[None, :] ** (-lam), chi, axis=1)
                self.integral_evaluations += 1
            self._cache[cache_key] = vals
        matrix = np.stack([vals[k] for k in flat_keys], axis=1)
        matrix = matrix.reshape(len(self.sample_combinations), *sums.shape)
        unique = np.asarray([complex(*k) for k in unique_keys], dtype=complex)
        return PreparedBatchedLOSMoments(
            exponent_sums=sums, values=matrix, unique_exponents=unique,
            sample_combinations=self.sample_combinations,
        )


class FactorizedGrowthMomentRule(SlepianLOSMomentRule):
    """Backward-compatible single-sample wrapper around the batch rule."""

    def __init__(self, model, projector, sample_combination=None):
        self.sample_combination = _normalize_combo(sample_combination)
        self._batch = FactorizedGrowthBatchMomentRule(
            model, projector, (self.sample_combination,)
        )
        self.model = model
        self.projector = projector

    @property
    def integral_evaluations(self):
        return self._batch.integral_evaluations

    def prepare(self, exponents_a, exponents_b, *, coefficient_factor=lambda z: 1.0):
        out = self._batch.prepare(
            exponents_a, exponents_b, coefficient_factor=coefficient_factor
        )
        return PreparedLOSMoments(
            exponent_sums=out.exponent_sums, values=out.values[0],
            unique_exponents=out.unique_exponents,
        )


class GeneralCoefficientBatchMomentRule(SlepianLOSMomentRule):
    r"""Exact node-quadrature LOS rule for non-factorized radial coefficients.

    This is the Phase-12 reference implementation for models such as
    BiHalofit, whose FFTLog coefficients vary with redshift and therefore do
    not reduce to the SPT frequency-sum moment.  The rule deliberately makes
    no low-rank or soft-leg approximation: the calculator evaluates the
    already validated fixed-redshift Slepian kernel on the projector nodes and
    this object performs the batched LOS contraction with the authoritative
    ``projector.los_weight`` values.

    The interface is intentionally separate from the SPT Mellin-moment rule so
    a future compressed coefficient-matrix implementation can replace this
    reference path without changing the calculator or bispectrum terms.
    """

    def __init__(self, model, projector, sample_combinations):
        if float(getattr(projector, "l_shift", 0.0)) != 0.0:
            raise NotImplementedError("Phase-12 general Slepian LOS requires projector.l_shift == 0")
        combos = tuple(_normalize_combo(c) for c in sample_combinations)
        if not combos:
            raise ValueError("at least one sample combination is required")
        self.model = model
        self.projector = projector
        self.sample_combinations = combos
        self.integral_evaluations = 0

    def weight_matrix(self):
        return np.asarray(
            [self.projector.los_weight(c) for c in self.sample_combinations],
            dtype=complex,
        )

    @property
    def nodes(self):
        return (
            np.asarray(self.projector.z, dtype=float),
            np.asarray(self.projector.chi, dtype=float),
        )

    def integrate_node_values(self, values, *, weight_matrix=None):
        """Integrate ``values[z,...]`` for all sample combinations at once."""
        values = np.asarray(values, dtype=complex)
        z, chi = self.nodes
        if values.shape[0] != len(z):
            raise ValueError("leading values axis must match projector z/chi nodes")
        weights = self.weight_matrix() if weight_matrix is None else np.asarray(weight_matrix, dtype=complex)
        out = np.trapezoid(weights[(slice(None), slice(None)) + (None,) * (values.ndim - 1)]
                           * values[None, ...], chi, axis=1)
        self.integral_evaluations += 1
        return out


class GeneralCoefficientMomentRule(SlepianLOSMomentRule):
    """Single-sample wrapper around :class:`GeneralCoefficientBatchMomentRule`."""

    def __init__(self, model, projector, sample_combination=None):
        combo = _normalize_combo(sample_combination)
        self.sample_combination = combo
        self._batch = GeneralCoefficientBatchMomentRule(model, projector, (combo,))
        self.model = model
        self.projector = projector

    @property
    def integral_evaluations(self):
        return self._batch.integral_evaluations

    @property
    def nodes(self):
        return self._batch.nodes

    def weight_matrix(self):
        return self._batch.weight_matrix()

    def integrate_node_values(self, values, *, weight_matrix=None):
        return self._batch.integrate_node_values(values, weight_matrix=weight_matrix)[0]


__all__ = [
    "PreparedLOSMoments", "PreparedBatchedLOSMoments",
    "SlepianLOSMomentRule", "FactorizedGrowthMomentRule",
    "FactorizedGrowthBatchMomentRule",
    "GeneralCoefficientMomentRule", "GeneralCoefficientBatchMomentRule",
]
