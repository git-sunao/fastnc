"""Declarative interfaces for bispectrum terms compatible with the Slepian route.

This module is deliberately model-facing only.  It contains no FFTLog,
Weber--Schafheitlin, LOS integration, or 3PCF execution code.  Those numerical
operations belong under :mod:`fastnc.threepcf.slepian`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .terms import BispectrumTerm


@dataclass(frozen=True)
class SlepianLOSMomentMetadata:
    """Declarative description of the redshift/LOS structure of a term.

    Parameters
    ----------
    kind
        Name of the LOS structure understood by a future numerical moment
        rule, e.g. ``"factorized-growth"``.  Phase 4 does not interpret it.
    signature
        Immutable, model-defined data that distinguishes terms requiring
        different LOS moments.  It is intentionally opaque to the bispectrum
        layer and can later participate in cache keys.

    Notes
    -----
    This object is metadata, not a numerical LOS rule.  In particular it must
    not hold a projector or perform integrations.  The executable
    ``SlepianLOSMomentRule`` hierarchy is introduced in the 3PCF layer when LOS
    acceleration is implemented.
    """

    kind: str
    signature: tuple[Any, ...] = ()

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        object.__setattr__(self, "signature", tuple(self.signature))


class SlepianTerm(BispectrumTerm):
    r"""Physical bispectrum term with a separable Slepian representation.

    A term represents

    .. math::

        B_a = C_a(z)\prod_{i=1}^3 f_{ai}(k_i;z)
              \exp\!\left[i\sum_{i=1}^3 n_{ai}\phi_i\right],
        \qquad \sum_i n_{ai}=0.

    Subclasses provide :meth:`c`, :meth:`f1`, :meth:`f2`, :meth:`f3`, and
    :attr:`angular_orders`.  The default :meth:`evaluate` reconstructs a
    canonical closed Fourier triangle from ``(k1,k2,k3)`` and evaluates this
    same representation directly.  This keeps every Slepian term usable by
    the generic route when Slepian acceleration is disabled.

    The class contains no numerical 3PCF machinery.  Expensive radial
    transforms, Weber tables, geometry caches, and LOS moments are calculator
    responsibilities.
    """

    @property
    def angular_orders(self):
        """Integer tuple ``(n1,n2,n3)`` with ``n1+n2+n3 == 0``."""
        raise NotImplementedError

    @property
    def los_moment_metadata(self) -> SlepianLOSMomentMetadata | None:
        """Optional immutable metadata consumed by a future LOS moment rule.

        ``None`` means only that no accelerated LOS rule has been declared;
        it does not prevent fixed-redshift Slepian evaluation.
        """
        return None

    def c(self, z, **params):
        """Return the scalar/redshift coefficient ``C_a(z)``."""
        raise NotImplementedError

    def f1(self, k, z, **params):
        """Return the separable radial factor on leg 1."""
        raise NotImplementedError

    def f2(self, k, z, **params):
        """Return the separable radial factor on leg 2."""
        raise NotImplementedError

    def f3(self, k, z, **params):
        """Return the separable radial factor on leg 3."""
        raise NotImplementedError

    def validated_angular_orders(self) -> tuple[int, int, int]:
        """Return normalized angular orders after checking the term contract."""
        orders = tuple(self.angular_orders)
        if len(orders) != 3:
            raise ValueError("angular_orders must contain exactly three integers")
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) for n in orders):
            raise TypeError("angular_orders must contain integers")
        orders = tuple(int(n) for n in orders)
        if sum(orders) != 0:
            raise ValueError("Slepian angular orders must satisfy n1 + n2 + n3 = 0")
        return orders

    @staticmethod
    def canonical_triangle_angles(k1, k2, k3, *, orientation: int = 1):
        r"""Return canonical vector angles for a closed triangle.

        ``phi1`` is fixed to zero.  ``phi2`` is chosen above the x-axis for
        ``orientation=+1`` and below it for ``orientation=-1``; ``phi3`` then
        follows from ``k1 + k2 + k3 = 0``.  Since valid Slepian terms obey
        ``sum(n_i)=0``, the angular phase is independent of the arbitrary
        global rotation used by this convention.
        """
        if orientation not in (-1, 1):
            raise ValueError("orientation must be +1 or -1")

        k1, k2, k3 = np.broadcast_arrays(
            np.asarray(k1, dtype=float),
            np.asarray(k2, dtype=float),
            np.asarray(k3, dtype=float),
        )
        if np.any(k1 <= 0) or np.any(k2 <= 0) or np.any(k3 <= 0):
            raise ValueError("triangle side lengths must be strictly positive")

        with np.errstate(divide="ignore", invalid="ignore"):
            mu12 = (k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2)

        # Tolerate only round-off excursions outside the triangle domain.
        tol = 64.0 * np.finfo(float).eps
        if np.any(mu12 < -1.0 - tol) or np.any(mu12 > 1.0 + tol):
            raise ValueError("(k1,k2,k3) do not form a closed triangle")
        mu12 = np.clip(mu12, -1.0, 1.0)

        phi1 = np.zeros_like(mu12)
        phi2 = orientation * np.arccos(mu12)
        v3 = -(k1 + k2 * np.exp(1j * phi2))
        phi3 = np.angle(v3)
        return phi1, phi2, phi3

    def angular_factor(self, phi1, phi2, phi3):
        """Evaluate the rotationally invariant exponential angular factor."""
        n1, n2, n3 = self.validated_angular_orders()
        return np.exp(1j * (n1 * phi1 + n2 * phi2 + n3 * phi3))

    def evaluate_separable(self, k1, k2, k3, z, *, orientation: int = 1, **params):
        """Evaluate ``c*f1*f2*f3*angular_factor`` on a closed triangle."""
        phi1, phi2, phi3 = self.canonical_triangle_angles(
            k1, k2, k3, orientation=orientation
        )
        return (
            self.c(z, **params)
            * self.f1(k1, z, **params)
            * self.f2(k2, z, **params)
            * self.f3(k3, z, **params)
            * self.angular_factor(phi1, phi2, phi3)
        )

    def evaluate(self, k1, k2, k3, z, **params):
        """Directly evaluate the same physical separable representation."""
        return self.evaluate_separable(k1, k2, k3, z, **params)


class ModelSlepianTerm(SlepianTerm):
    """Slepian term holding a live reference to its owning bispectrum model."""

    def __init__(self, model: Any):
        if model is None:
            raise ValueError("model must be the owning bispectrum object")
        self._model = model

    @property
    def model(self):
        return self._model

    def _merge_model_defaults(self, params):
        merge = getattr(self.model, "_merge_default_kwargs", None)
        return merge(params) if merge is not None else dict(params)


class BackendSlepianTerm(SlepianTerm):
    """Slepian term holding a reference to a shared stateful physics backend."""

    def __init__(self, backend: Any):
        if backend is None:
            raise ValueError("backend must be a shared model/backend object")
        self._backend = backend

    @property
    def backend(self):
        return self._backend
