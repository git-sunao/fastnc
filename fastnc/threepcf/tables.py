"""Passive result tables shared by all 3PCF calculation routes."""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from fastnc.hankel.grid import TunedFFTGrid

from .conventions.projection import _projection_name, convert_projection


@dataclass(frozen=True)
class ComponentModeKey:
    """Physical label for one epsilon component and opening-angle mode."""

    epsilon: tuple[int, int, int]
    two_k: int

    def __post_init__(self):
        epsilon = tuple(int(value) for value in self.epsilon)
        if len(epsilon) != 3:
            raise ValueError("epsilon must contain exactly three entries")
        if any(value not in (-1, 1) for value in epsilon):
            raise ValueError("epsilon entries must be +1 or -1")
        raw_two_k = float(self.two_k)
        if not np.isfinite(raw_two_k):
            raise ValueError("two_k must be an integer")
        two_k = int(round(raw_two_k))
        if not np.isclose(
            raw_two_k, two_k, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("two_k must be an integer")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "two_k", two_k)

    @classmethod
    def from_epsilon_k(cls, epsilon, k: float) -> "ComponentModeKey":
        two_k = int(round(2.0 * float(k)))
        if not np.isclose(2.0 * float(k), two_k, rtol=0.0, atol=1.0e-12):
            raise ValueError("k must be integer or half-integer")
        return cls(tuple(epsilon), two_k)

    @property
    def k(self) -> float:
        return 0.5 * self.two_k


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


def _aliases(values, keys) -> Mapping[ComponentModeKey, Hashable]:
    aliases = dict(values)
    for physical_key, storage_key in aliases.items():
        if not isinstance(physical_key, ComponentModeKey):
            raise TypeError("alias keys must be ComponentModeKey instances")
        if storage_key not in keys:
            raise ValueError(
                f"alias refers to an unknown storage key: {storage_key}"
            )
    return MappingProxyType(aliases)


@dataclass(frozen=True)
class HKernelTable:
    """Angularly coupled kernels sampled on a full FFTLog ell grid."""

    grid: TunedFFTGrid
    keys: tuple[Hashable, ...]
    values: np.ndarray
    aliases: Mapping[ComponentModeKey, Hashable] = field(default_factory=dict)

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
        object.__setattr__(self, "aliases", _aliases(self.aliases, keys))

    @property
    def ell(self) -> np.ndarray:
        return self.grid.ell

    def get(self, key: Hashable) -> np.ndarray:
        try:
            index = self.keys.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        return self.values[index]

    def key_for_mode(self, epsilon, k: float) -> Hashable:
        physical_key = ComponentModeKey.from_epsilon_k(epsilon, k)
        try:
            return self.aliases[physical_key]
        except KeyError as exc:
            raise KeyError(physical_key) from exc

    def get_for_mode(self, epsilon, k: float) -> np.ndarray:
        return self.get(self.key_for_mode(epsilon, k))


@dataclass(frozen=True)
class ZetaKTable:
    """Opening-angle modes on the final user-facing theta grid."""

    theta: np.ndarray
    keys: tuple[Hashable, ...]
    values: np.ndarray
    aliases: Mapping[ComponentModeKey, Hashable] = field(default_factory=dict)

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
        object.__setattr__(self, "aliases", _aliases(self.aliases, keys))

    def get(self, key: Hashable) -> np.ndarray:
        try:
            index = self.keys.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        return self.values[index]

    def key_for_mode(self, epsilon, k: float) -> Hashable:
        physical_key = ComponentModeKey.from_epsilon_k(epsilon, k)
        try:
            return self.aliases[physical_key]
        except KeyError as exc:
            raise KeyError(physical_key) from exc

    def get_for_mode(self, epsilon, k: float) -> np.ndarray:
        return self.get(self.key_for_mode(epsilon, k))


@dataclass(frozen=True)
class ZetaTable:
    """Final 3PCF components on side-length and opening-angle grids."""

    theta: np.ndarray
    phi: np.ndarray
    components: tuple[Hashable, ...]
    values: np.ndarray
    sigmas: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    projection: str = "x"

    def __post_init__(self):
        theta = _positive_axis(self.theta, "theta")
        phi = np.asarray(self.phi, dtype=float)
        if phi.ndim != 1 or phi.size < 1:
            raise ValueError("phi must be a non-empty one-dimensional array")
        if np.any(~np.isfinite(phi)) or np.any(np.diff(phi) <= 0.0):
            raise ValueError("phi must be finite and strictly increasing")
        components = _labels(self.components, "components")
        sigmas = tuple(
            tuple(int(value) for value in sigma) for sigma in self.sigmas
        )
        if sigmas and len(sigmas) != len(components):
            raise ValueError("sigmas must contain one triple per component")
        if any(len(sigma) != 3 for sigma in sigmas):
            raise ValueError("each sigma must contain exactly three entries")
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
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "projection", _projection_name(self.projection))

    def get(self, component: Hashable) -> np.ndarray:
        try:
            index = self.components.index(component)
        except ValueError as exc:
            raise KeyError(component) from exc
        return self.values[index]

    def to_projection(self, projection: str) -> "ZetaTable":
        """Return a new table in another shear-projection convention."""
        destination = _projection_name(projection)
        if destination == self.projection:
            return self
        if len(self.sigmas) != len(self.components):
            raise ValueError(
                "projection conversion requires one sigma triple per component"
            )
        converted = np.stack(
            [
                convert_projection(
                    self.values[index],
                    self.theta,
                    self.theta,
                    self.phi,
                    from_projection=self.projection,
                    to_projection=destination,
                    sigma=sigma,
                )
                for index, sigma in enumerate(self.sigmas)
            ]
        )
        return ZetaTable(
            theta=self.theta,
            phi=self.phi,
            components=self.components,
            values=converted,
            sigmas=self.sigmas,
            projection=destination,
        )
