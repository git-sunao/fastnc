"""Configured, route-independent line-of-sight projection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fastnc.bispectrum import (
    Bispectrum2D,
    Bispectrum3D,
)

from .coefficient_los import integrate_coefficients
from .geometry import validate_los_coordinates
from .kernels import KernelSet
from .rules import project_term


LOSPrefactor = (
    float
    | np.ndarray
    | Callable[[np.ndarray, np.ndarray], np.ndarray]
    | None
)


@dataclass(frozen=True)
class LOSProjector:
    """LOS coordinates and weights shared by independent calculation routes.

    Numeric projection maps a ``Bispectrum3D`` to a ``Bispectrum2D`` while
    preserving its additive term names. The projector knows no 3PCF routes.
    By default the geometrical prefactor is chi**-4, matching the standard
    projected-bispectrum convention used by fastnc v2.
    """

    z: np.ndarray
    chi: np.ndarray
    kernels: KernelSet | None = None
    prefactor: LOSPrefactor = None
    shift: float = 0.0
    _evaluate_at_point: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        if self._evaluate_at_point:
            z = np.atleast_1d(np.asarray(self.z, dtype=float))
            chi = np.atleast_1d(np.asarray(self.chi, dtype=float))
            if z.shape != (1,) or chi.shape != (1,):
                raise ValueError("delta_like z and chi must be scalar")
            if not np.isfinite(z[0]):
                raise ValueError("z must be finite")
            if not np.isfinite(chi[0]) or chi[0] <= 0.0:
                raise ValueError("chi must be finite and positive")
            if self.kernels is not None:
                raise ValueError("delta_like projector does not accept kernels")
            if self.prefactor not in (None, 1.0):
                raise ValueError("delta_like projector does not accept a prefactor")
        else:
            z, chi = validate_los_coordinates(self.z, self.chi)
        z = np.array(z, copy=True)
        chi = np.array(chi, copy=True)
        z.setflags(write=False)
        chi.setflags(write=False)
        if self.kernels is not None and not isinstance(self.kernels, KernelSet):
            raise TypeError("kernels must be a KernelSet or None")
        if self.prefactor is not None and not callable(self.prefactor):
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

    @classmethod
    def delta_like(cls, *, z: float, chi: float, shift: float = 0.0):
        """Return a projector that evaluates exactly at one ``(z, chi)``.

        This is an exact fixed-redshift benchmark, not a finite-width radial
        kernel. It applies neither LOS quadrature nor the usual ``chi**-4``
        prefactor.
        """
        return cls(
            z=z,
            chi=chi,
            prefactor=1.0,
            shift=shift,
            _evaluate_at_point=True,
        )

    @property
    def is_delta_like(self) -> bool:
        return self._evaluate_at_point

    @staticmethod
    def _array_signature(values) -> tuple:
        values = np.ascontiguousarray(np.asarray(values))
        return values.dtype.str, values.shape, values.tobytes()

    @property
    def grid_signature(self) -> tuple:
        """Immutable signature of the LOS nodes and quadrature convention."""
        return (
            "los-grid-v1",
            self._array_signature(self.z),
            self._array_signature(self.chi),
            "point" if self.is_delta_like else "trapezoid-chi",
        )

    @property
    def physical_state_token(self) -> tuple:
        """Signature of projection values at fixed LOS nodes."""
        if callable(self.prefactor):
            prefactor = ("callable", id(self.prefactor))
        elif self.prefactor is None:
            prefactor = ("default-chi^-4",)
        else:
            prefactor = ("values", self._array_signature(self.prefactor))
        kernels = None
        if self.kernels is not None:
            kernels = tuple(
                (
                    name,
                    self._array_signature(self.kernels[name].z),
                    self._array_signature(self.kernels[name].chi),
                    self._array_signature(self.kernels[name].weight),
                )
                for name in self.kernels.names
            )
        return (
            "los-physics-v1",
            float(self.shift),
            prefactor,
            kernels,
            bool(self.is_delta_like),
        )

    def weight(self, sample_combination=None) -> np.ndarray:
        """Return the configured prefactor times selected radial kernels."""
        if self._evaluate_at_point:
            if sample_combination is not None and tuple(sample_combination):
                raise ValueError("delta_like projector does not accept kernels")
            return np.ones(1, dtype=float)
        if self.prefactor is None:
            prefactor = self.chi**-4
        elif callable(self.prefactor):
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
            if sample_combination is not None and tuple(sample_combination):
                raise ValueError(
                    "kernel names were provided but no KernelSet is configured"
                )
            return weight

        if sample_combination is None:
            return weight
        return weight * self.kernels.product(sample_combination, self.chi)

    def project(self, bispectrum, *, sample_combination=None) -> Bispectrum2D:
        """Assemble supported 3D representations into angular recipes.

        Each weighted 3D term becomes a same-named 2D term. Numeric evaluation
        performs the configured LOS integral immediately. Route-specific
        projected representations retain this projector for later integration.
        """
        if not isinstance(bispectrum, Bispectrum3D):
            raise TypeError("bispectrum must be a Bispectrum3D")
        self.weight(sample_combination)

        projected_terms = [
            project_term(
                weighted_term,
                projector=self,
                sample_combination=sample_combination,
            )
            for weighted_term in bispectrum.weighted_terms
        ]

        return Bispectrum2D(
            projected_terms,
            _revision_sources=(lambda: bispectrum.state_token,),
        )

    def integrate_coefficients(
        self,
        coefficients,
        *,
        axis: int = -1,
        sample_combination=None,
    ):
        r"""Apply this projector to route-produced coefficient samples.

        A finite-width projector evaluates
        :math:`\int d\chi\,W(\chi)c(\chi)`, with ``weight()`` supplying kernels
        and prefactors. A delta-like projector instead selects its sole LOS
        sample exactly and applies neither quadrature nor a
        :math:`\chi^{-4}` factor. Route calculators therefore share one API
        without approximating a Dirac delta by a narrow window.
        """
        if self._evaluate_at_point:
            self.weight(sample_combination)
            coefficients = np.asarray(coefficients)
            axis = np.lib.array_utils.normalize_axis_index(
                axis, coefficients.ndim
            )
            if coefficients.shape[axis] != 1:
                raise ValueError(
                    "delta_like coefficients must have one LOS value"
                )
            return np.take(coefficients, 0, axis=axis)
        return integrate_coefficients(
            coefficients,
            self.chi,
            weight=self.weight(sample_combination),
            axis=axis,
        )
