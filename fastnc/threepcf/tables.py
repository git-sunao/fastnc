"""Passive result tables shared by all 3PCF calculation routes."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from fastnc.hankel.grid import TunedFFTGrid


@dataclass(frozen=True)
class HKernelKey:
    """Canonical label for one deduplicated numeric H kernel."""

    sigma1: int
    two_nu: int


@dataclass(frozen=True)
class ZetaKKey:
    """Route-independent label for one opening-angle mode."""

    hkey: HKernelKey
    m: int
    n: int
    Sigma: int


def _readonly_array(values, *, dtype=None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _positive_axis(values, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(axis)) or np.any(axis <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return _readonly_array(axis)


def _labels(values, name: str) -> tuple[Hashable, ...]:
    labels = tuple(values)
    if not labels:
        raise ValueError(f"{name} must contain at least one label")
    if any(not isinstance(label, Hashable) for label in labels):
        raise TypeError(f"every {name} entry must be hashable")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} entries must be unique")
    return labels


@dataclass(frozen=True)
class HKernelTable:
    """Angularly coupled kernels sampled on a full FFTLog ell grid."""

    grid: TunedFFTGrid
    keys: tuple[Hashable, ...]
    values: np.ndarray

    def __post_init__(self):
        if not isinstance(self.grid, TunedFFTGrid):
            raise TypeError("grid must be a TunedFFTGrid")
        keys = _labels(self.keys, "keys")
        values = np.asarray(self.values)
        expected = (len(keys), self.grid.ell.size, self.grid.ell.size)
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}; got {values.shape}")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "values", _readonly_array(values))

    @property
    def ell(self) -> np.ndarray:
        return self.grid.ell

    def get(self, key: Hashable) -> np.ndarray:
        try:
            index = self.keys.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        return self.values[index]


@dataclass(frozen=True)
class ZetaKTable:
    """Opening-angle modes on the final user-facing theta grid."""

    theta: np.ndarray
    keys: tuple[Hashable, ...]
    values: np.ndarray

    def __post_init__(self):
        theta = _positive_axis(self.theta, "theta")
        keys = _labels(self.keys, "keys")
        values = np.asarray(self.values)
        expected = (len(keys), theta.size, theta.size)
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}; got {values.shape}")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "values", _readonly_array(values))

    def get(self, key: Hashable) -> np.ndarray:
        try:
            index = self.keys.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        return self.values[index]


@dataclass(frozen=True)
class ZetaTable:
    """Final 3PCF components on side-length and opening-angle grids."""

    theta: np.ndarray
    phi: np.ndarray
    components: tuple[Hashable, ...]
    values: np.ndarray

    def __post_init__(self):
        theta = _positive_axis(self.theta, "theta")
        phi = np.asarray(self.phi, dtype=float)
        if phi.ndim != 1 or phi.size < 1:
            raise ValueError("phi must be a non-empty one-dimensional array")
        if np.any(~np.isfinite(phi)) or np.any(np.diff(phi) <= 0.0):
            raise ValueError("phi must be finite and strictly increasing")
        components = _labels(self.components, "components")
        values = np.asarray(self.values)
        expected = (
            len(components),
            theta.size,
            theta.size,
            phi.size,
        )
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}; got {values.shape}")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "phi", _readonly_array(phi))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "values", _readonly_array(values))

    def get(self, component: Hashable) -> np.ndarray:
        try:
            index = self.components.index(component)
        except ValueError as exc:
            raise KeyError(component) from exc
        return self.values[index]
