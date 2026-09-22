"""Configured, route-independent line-of-sight projection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .coefficient_los import integrate_coefficients
from .geometry import validate_los_coordinates
from .kernels import KernelSet
from .numeric_los import LOSValues, evaluate_numeric_los, integrate_numeric_los


LOSPrefactor = float | np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LOSProjector:
    """LOS coordinates and weights shared by independent calculation routes.

    The projector knows neither bispectrum representations nor 3PCF routes.
    Callers provide either a numeric evaluator or already sampled coefficient
    arrays. The prefactor is deliberately unity by default so geometrical and
    physical projection conventions remain explicit.
    """

    z: np.ndarray
    chi: np.ndarray
    kernels: KernelSet | None = None
    prefactor: LOSPrefactor = 1.0
    shift: float = 0.0

    def __post_init__(self):
        z, chi = validate_los_coordinates(self.z, self.chi)
        z = np.array(z, copy=True)
        chi = np.array(chi, copy=True)
        z.setflags(write=False)
        chi.setflags(write=False)
        if self.kernels is not None and not isinstance(self.kernels, KernelSet):
            raise TypeError("kernels must be a KernelSet or None")
        if not callable(self.prefactor):
            prefactor = np.asarray(self.prefactor, dtype=float)
            try:
                np.broadcast_to(prefactor, chi.shape)
            except ValueError as exc:
                raise ValueError(
                    "prefactor must be scalar or have the same shape as chi"
                ) from exc
            if np.any(~np.isfinite(prefactor)):
                raise ValueError("prefactor must be finite")
            if prefactor.ndim == 0:
                prefactor = prefactor.item()
            else:
                prefactor = np.array(prefactor, copy=True)
                prefactor.setflags(write=False)
            object.__setattr__(self, "prefactor", prefactor)
        shift = float(self.shift)
        if not np.isfinite(shift):
            raise ValueError("shift must be finite")
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "shift", shift)

    def weight(self, kernel_names=None) -> np.ndarray:
        """Return the configured prefactor times selected radial kernels."""
        if callable(self.prefactor):
            prefactor = np.asarray(self.prefactor(self.z, self.chi), dtype=float)
        else:
            prefactor = np.asarray(self.prefactor, dtype=float)
        try:
            weight = np.array(np.broadcast_to(prefactor, self.chi.shape), copy=True)
        except ValueError as exc:
            raise ValueError(
                "prefactor output must be scalar or have the same shape as chi"
            ) from exc
        if np.any(~np.isfinite(weight)):
            raise ValueError("prefactor output must be finite")

        if self.kernels is None:
            if kernel_names is not None and tuple(kernel_names):
                raise ValueError(
                    "kernel names were provided but no KernelSet is configured"
                )
            return weight

        names = self.kernels.names if kernel_names is None else tuple(kernel_names)
        return weight * self.kernels.product(names, self.chi)

    def sample_numeric(self, evaluator, ell1, ell2, ell3, **params) -> LOSValues:
        """Evaluate a numeric 3D callable at this projector's LOS nodes."""
        return evaluate_numeric_los(
            evaluator,
            ell1,
            ell2,
            ell3,
            z=self.z,
            chi=self.chi,
            shift=self.shift,
            **params,
        )

    def integrate_numeric(self, sampled: LOSValues, *, kernel_names=None):
        """Integrate values returned by sample_numeric."""
        return integrate_numeric_los(
            sampled,
            self.chi,
            weight=self.weight(kernel_names),
        )

    def project_numeric(
        self,
        evaluator,
        ell1,
        ell2,
        ell3,
        *,
        kernel_names=None,
        **params,
    ):
        """Sample and integrate a numeric 3D callable."""
        sampled = self.sample_numeric(evaluator, ell1, ell2, ell3, **params)
        return self.integrate_numeric(sampled, kernel_names=kernel_names)

    def integrate_coefficients(
        self,
        coefficients,
        *,
        axis: int = -1,
        kernel_names=None,
    ):
        """Integrate route-produced coefficients along their LOS axis."""
        return integrate_coefficients(
            coefficients,
            self.chi,
            weight=self.weight(kernel_names),
            axis=axis,
        )
