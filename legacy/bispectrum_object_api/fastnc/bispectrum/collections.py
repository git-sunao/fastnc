"""Collections and views for projected 2D bispectra.

This module contains classes that manage several sample/tomographic
combinations sharing the same line-of-sight base cache.  Single-combination
projection machinery remains in :mod:`fastnc.bispectrum.los`.
"""
from __future__ import annotations

import numpy as np

from .base import Bispectrum2D, Bispectrum3D
from .support import Support2D
from .los import BaseLOSIntegrand, LineOfSightProjector, _normalize_combo


class ProjectedBispectrum2DCollection:
    """Collection of projected 2D bispectra sharing a LOS-base cache."""

    def __init__(
        self,
        bispectrum3d: Bispectrum3D,
        projector: LineOfSightProjector,
        sample_combinations,
        window=None,
    ):
        self.bispectrum3d = bispectrum3d
        self.projector = projector
        self.sample_combinations = [_normalize_combo(c) for c in sample_combinations]
        self.window = window
        chi_min = np.nanmin(projector.chi[projector.chi > 0])
        chi_max = np.nanmax(projector.chi)
        sup3 = bispectrum3d.support
        self.support = Support2D(sup3.k_min * chi_min, sup3.k_max * chi_max, policy=sup3.policy)
        self._last_key = None
        self._last_base: BaseLOSIntegrand | None = None

    @staticmethod
    def _array_signature(x):
        arr = np.asarray(x, dtype=float)
        # ascontiguousarray makes tobytes deterministic for views/transposes.
        arr = np.ascontiguousarray(arr)
        return (arr.shape, str(arr.dtype), arr.tobytes())

    @staticmethod
    def _params_signature(params):
        # Params are usually scalars.  repr is intentionally used as a broad,
        # conservative fallback for non-hashable objects.
        return tuple(sorted((k, repr(v)) for k, v in params.items()))

    def _cache_key(self, ell1, ell2, ell3, params):
        return (
            self._array_signature(ell1),
            self._array_signature(ell2),
            self._array_signature(ell3),
            self._params_signature(params),
        )

    def clear_cache(self):
        self._last_key = None
        self._last_base = None

    def get_base(self, ell1, ell2, ell3, **params):
        key = self._cache_key(ell1, ell2, ell3, params)
        if key != self._last_key:
            self._last_base = self.projector.evaluate_base_integrand(
                self.bispectrum3d,
                ell1,
                ell2,
                ell3,
                **params,
            )
            self._last_key = key
        return self._last_base

    def evaluate_one(self, sample_combination, ell1, ell2, ell3, **params):
        combo = _normalize_combo(sample_combination)
        base = self.get_base(ell1, ell2, ell3, **params)
        out = self.projector.integrate_base(base, sample_combination=combo)
        if self.window is not None:
            out = out * self.window(ell1, ell2, ell3)
        return out

    def evaluate(self, ell1, ell2, ell3, **params):
        base = self.get_base(ell1, ell2, ell3, **params)
        out = {
            combo: self.projector.integrate_base(base, sample_combination=combo)
            for combo in self.sample_combinations
        }
        if self.window is not None:
            window = self.window(ell1, ell2, ell3)
            out = {combo: val * window for combo, val in out.items()}
        return out

    def __call__(self, ell1, ell2, ell3, **params):
        return self.evaluate(ell1, ell2, ell3, **params)

    def __getitem__(self, sample_combination):
        combo = _normalize_combo(sample_combination)
        if combo not in self.sample_combinations:
            raise KeyError(f"sample combination {combo!r} is not in this collection")
        return ProjectedBispectrum2DView(self, combo)

    def keys(self):
        return tuple(self.sample_combinations)

    def values(self):
        return tuple(self[combo] for combo in self.sample_combinations)

    def items(self):
        return tuple((combo, self[combo]) for combo in self.sample_combinations)

    def as_list(self):
        return [self[combo] for combo in self.sample_combinations]


class ProjectedBispectrum2DView(Bispectrum2D):
    """Single-combination view backed by a shared collection/cache."""

    def __init__(self, collection: ProjectedBispectrum2DCollection, sample_combination):
        self.collection = collection
        self.sample_combination = _normalize_combo(sample_combination)
        self.support = collection.support
        self.window = collection.window

    def evaluate(self, ell1, ell2, ell3, **params):
        return self.collection.evaluate_one(
            self.sample_combination,
            ell1,
            ell2,
            ell3,
            **params,
        )


# Backward-compatible names used by previous revisions.
ProjectedBispectra2D = ProjectedBispectrum2DCollection
ProjectedBispectrum2DGroup = ProjectedBispectrum2DCollection

# Backward-compatible aliases.  New code should use names without ``Angular``.
ProjectedAngularBispectrum2DCollection = ProjectedBispectrum2DCollection
ProjectedAngularBispectrum2DGroup = ProjectedBispectrum2DCollection
ProjectedAngularBispectrum2DView = ProjectedBispectrum2DView
ProjectedAngularBispectra2D = ProjectedBispectra2D
