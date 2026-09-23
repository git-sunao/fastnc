"""Term-wise interpolation of numeric angular bispectrum representations."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .bispectrum import Bispectrum2D
from .representations import (
    NumericRepresentation2D,
)
from .terms import BispectrumTerm2D


def _validated_axis(values, name, *, positive=False, lower=None, upper=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two points")
    if np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if lower is not None and np.any(values <= lower):
        raise ValueError(f"{name} must be greater than {lower}")
    if upper is not None and np.any(values > upper):
        raise ValueError(f"{name} must not exceed {upper}")
    values = np.array(values, copy=True)
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class TriangleInterpolationConfig:
    """Regular grid in ``(log ell2, log ell3, mu23)`` coordinates."""

    ell2: np.ndarray
    ell3: np.ndarray
    mu23: np.ndarray
    method: str = "linear"
    bounds_error: bool = True

    def __post_init__(self):
        if self.method not in {"linear", "nearest"}:
            raise ValueError("method must be 'linear' or 'nearest'")
        object.__setattr__(
            self, "ell2", _validated_axis(self.ell2, "ell2", positive=True)
        )
        object.__setattr__(
            self, "ell3", _validated_axis(self.ell3, "ell3", positive=True)
        )
        object.__setattr__(
            self,
            "mu23",
            _validated_axis(self.mu23, "mu23", lower=-1.0, upper=1.0),
        )


@dataclass(frozen=True)
class TriangleInterpolationCache:
    """Passive interpolation table and its source-state identity."""

    log_ell2: np.ndarray
    log_ell3: np.ndarray
    mu23: np.ndarray
    values: np.ndarray
    source_state_token: tuple
    interpolator: RegularGridInterpolator = field(repr=False, compare=False)

    @classmethod
    def build(cls, evaluator, config, source_state_token):
        log_ell2 = np.log(config.ell2)
        log_ell3 = np.log(config.ell3)
        ell2, ell3, mu23 = np.meshgrid(
            config.ell2,
            config.ell3,
            config.mu23,
            indexing="ij",
        )
        ell1 = np.sqrt(ell2**2 + ell3**2 + 2.0 * ell2 * ell3 * mu23)
        values = np.asarray(evaluator(ell1, ell2, ell3), dtype=float)
        try:
            values = np.array(np.broadcast_to(values, ell1.shape), copy=True)
        except ValueError as exc:
            raise ValueError(
                "source representation output must match the triangle grid"
            ) from exc
        if np.any(~np.isfinite(values)):
            raise ValueError("source representation returned non-finite values")
        for array in (log_ell2, log_ell3, values):
            array.setflags(write=False)
        interpolator = RegularGridInterpolator(
            (log_ell2, log_ell3, config.mu23),
            values,
            method=config.method,
            bounds_error=config.bounds_error,
            fill_value=None if not config.bounds_error else np.nan,
        )
        return cls(
            log_ell2=log_ell2,
            log_ell3=log_ell3,
            mu23=config.mu23,
            values=values,
            source_state_token=tuple(source_state_token),
            interpolator=interpolator,
        )

    def evaluate(self, ell1, ell2, ell3):
        scalar = all(np.ndim(value) == 0 for value in (ell1, ell2, ell3))
        ell1, ell2, ell3 = np.broadcast_arrays(
            np.asarray(ell1, dtype=float),
            np.asarray(ell2, dtype=float),
            np.asarray(ell3, dtype=float),
        )
        if np.any(~np.isfinite(ell1)) or np.any(ell1 < 0.0):
            raise ValueError("ell1 must be finite and non-negative")
        if np.any(~np.isfinite(ell2)) or np.any(ell2 <= 0.0):
            raise ValueError("ell2 must be finite and positive")
        if np.any(~np.isfinite(ell3)) or np.any(ell3 <= 0.0):
            raise ValueError("ell3 must be finite and positive")
        mu23 = (ell1**2 - ell2**2 - ell3**2) / (2.0 * ell2 * ell3)
        tolerance = 32.0 * np.finfo(float).eps
        if np.any(mu23 < -1.0 - tolerance) or np.any(mu23 > 1.0 + tolerance):
            raise ValueError("ell1, ell2, and ell3 do not form a closed triangle")
        mu23 = np.clip(mu23, -1.0, 1.0)
        points = np.column_stack(
            (np.log(ell2).ravel(), np.log(ell3).ravel(), mu23.ravel())
        )
        result = np.asarray(self.interpolator(points)).reshape(ell1.shape)
        return result.item() if scalar else result


@dataclass
class _InterpolationCacheSlot:
    table: TriangleInterpolationCache | None = None
    lock: RLock = field(default_factory=RLock, repr=False)


@dataclass(frozen=True)
class InterpolatedNumericRepresentation2D(NumericRepresentation2D):
    """Numeric representation evaluated from a self-owned triangle cache."""

    source_term: BispectrumTerm2D
    source_representation: NumericRepresentation2D
    config: TriangleInterpolationConfig
    source_state_token: Callable[[], tuple]
    _cache: _InterpolationCacheSlot = field(
        default_factory=_InterpolationCacheSlot,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if not isinstance(self.source_term, BispectrumTerm2D):
            raise TypeError("source_term must be a BispectrumTerm2D")
        if not isinstance(self.source_representation, NumericRepresentation2D):
            raise TypeError(
                "source_representation must be a NumericRepresentation2D"
            )
        if not isinstance(self.config, TriangleInterpolationConfig):
            raise TypeError("config must be a TriangleInterpolationConfig")
        if not callable(self.source_state_token):
            raise TypeError("source_state_token must be callable")

    @property
    def cache(self) -> TriangleInterpolationCache | None:
        return self._cache.table

    def prepare(self, *, force=False):
        """Build or refresh the interpolation table and return ``self``."""
        token = tuple(self.source_state_token())
        with self._cache.lock:
            if (
                not force
                and self._cache.table is not None
                and self._cache.table.source_state_token == token
            ):
                return self
            table = TriangleInterpolationCache.build(
                self.source_representation.evaluate,
                self.config,
                token,
            )
            if tuple(self.source_state_token()) != token:
                raise RuntimeError(
                    "source bispectrum state changed while building interpolation"
                )
            self._cache.table = table
        return self

    def invalidate(self):
        with self._cache.lock:
            self._cache.table = None

    def cache_info(self):
        table = self._cache.table
        return {
            "ready": table is not None,
            "shape": None if table is None else table.values.shape,
            "source_state_token": (
                None if table is None else table.source_state_token
            ),
        }

    def evaluate(self, ell1, ell2, ell3, **params):
        if params:
            names = ", ".join(sorted(params))
            raise TypeError(
                "interpolated evaluation does not accept runtime parameters: "
                f"{names}"
            )
        token = tuple(self.source_state_token())
        table = self._cache.table
        if table is None or table.source_state_token != token:
            self.prepare()
            table = self._cache.table
        return table.evaluate(ell1, ell2, ell3)

    __call__ = evaluate


def interpolate_numeric(bispectrum, config, *, prepare=True):
    """Return a new bispectrum with each numeric representation interpolated."""
    if not isinstance(bispectrum, Bispectrum2D):
        raise TypeError("bispectrum must be a Bispectrum2D")
    if not isinstance(config, TriangleInterpolationConfig):
        raise TypeError("config must be a TriangleInterpolationConfig")

    output_terms = []
    interpolated = []
    for weighted_term in bispectrum.weighted_terms:
        representations = []
        for representation in weighted_term.term.representations:
            if isinstance(representation, NumericRepresentation2D):
                replacement = InterpolatedNumericRepresentation2D(
                    source_term=weighted_term.term,
                    source_representation=representation,
                    config=config,
                    source_state_token=lambda source=bispectrum: source.state_token,
                )
                representations.append(replacement)
                interpolated.append(replacement)
            else:
                representations.append(representation)
        term = BispectrumTerm2D(
            name=weighted_term.term.name,
            representations=tuple(representations),
        )
        output_terms.append(term.scaled_by(weighted_term.coefficient))

    if not interpolated:
        raise LookupError("bispectrum has no numeric 2D representations")
    if prepare:
        for representation in interpolated:
            representation.prepare()
    return Bispectrum2D(
        output_terms,
        support=bispectrum.support,
        _revision_sources=(lambda: bispectrum.state_token,),
    )
