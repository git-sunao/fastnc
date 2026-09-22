"""One-halo 3D bispectrum term aggregates."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .bispectrum import Bispectrum3D
from .representations import NumericExpression3D
from .support import Support3D
from .terms import BispectrumTerm3D, WeightedTerm3D


class OneHaloProductBispectrum3D(Bispectrum3D):
    """One-halo-only product bispectrum model.

    The model is

    ``B(k1,k2,k3,z) = amplitude(z) * u(k1,z) * u(k2,z) * u(k3,z)``.
    """

    def __init__(
        self,
        profile: Callable,
        amplitude: float | Callable = 1.0,
        support: Support3D | None = None,
    ):
        if not callable(profile):
            raise TypeError("profile must be callable as profile(k, z)")
        self.profile = profile
        self.amplitude = amplitude
        term = BispectrumTerm3D(
            name="one-halo:product",
            representations=(NumericExpression3D(self._evaluate_profile_product),),
        )
        super().__init__(
            (WeightedTerm3D(self._amplitude, term),),
            support=support,
        )

    def _amplitude(self, z):
        if callable(self.amplitude):
            return self.amplitude(z)
        return self.amplitude

    def _evaluate_profile_product(self, k1, k2, k3, z, **params):
        u1 = self.profile(k1, z, **params)
        u2 = self.profile(k2, z, **params)
        u3 = self.profile(k3, z, **params)
        return u1 * u2 * u3

    def update_physics(self, *, profile=None, amplitude=None):
        """Update product-model inputs and invalidate dependent calculations."""
        if profile is not None:
            if not callable(profile):
                raise TypeError("profile must be callable as profile(k, z)")
            self.profile = profile
        if amplitude is not None:
            self.amplitude = amplitude
        self._state_updated()
        return self


class NFWOneHaloBispectrum3D(OneHaloProductBispectrum3D):
    """Simple NFW-like one-halo product bispectrum.

    The default profile is not an exact truncated-NFW Fourier transform.  It is
    a smooth NFW-like debug profile,

    ``u(k,z) = [1 + (k/k_s(z))**slope]**(-amplitude_power/slope)``.
    """

    def __init__(
        self,
        k_s: float | Callable = 1.0,
        slope: float = 2.0,
        amplitude_power: float = 1.0,
        redshift_scaling: float = 0.0,
        amplitude: float | Callable = 1.0,
        support: Support3D | None = None,
    ):
        self.k_s = k_s
        self.slope = float(slope)
        self.amplitude_power = float(amplitude_power)
        self.redshift_scaling = float(redshift_scaling)
        super().__init__(profile=self.profile, amplitude=amplitude, support=support)

    @classmethod
    def default(cls, support: Support3D | None = None) -> "NFWOneHaloBispectrum3D":
        """Return the standard NFW-like debug preset."""
        return cls(
            k_s=1.0,
            slope=2.0,
            amplitude_power=2.0,
            redshift_scaling=0.0,
            amplitude=1.0,
            support=support,
        )

    @classmethod
    def shallow(cls, support: Support3D | None = None) -> "NFWOneHaloBispectrum3D":
        """Return a shallower high-k profile preset."""
        return cls(
            k_s=1.0,
            slope=1.0,
            amplitude_power=2.0,
            redshift_scaling=0.0,
            amplitude=1.0,
            support=support,
        )

    @classmethod
    def steep(cls, support: Support3D | None = None) -> "NFWOneHaloBispectrum3D":
        """Return a steeper high-k profile preset."""
        return cls(
            k_s=1.0,
            slope=3.0,
            amplitude_power=2.0,
            redshift_scaling=0.0,
            amplitude=1.0,
            support=support,
        )

    @classmethod
    def with_parameters(
        cls,
        *,
        k_s: float | Callable = 1.0,
        slope: float = 2.0,
        amplitude_power: float = 2.0,
        redshift_scaling: float = 0.0,
        amplitude: float | Callable = 1.0,
        support: Support3D | None = None,
    ) -> "NFWOneHaloBispectrum3D":
        """Explicit factory for named parameter presets in user code."""
        return cls(
            k_s=k_s,
            slope=slope,
            amplitude_power=amplitude_power,
            redshift_scaling=redshift_scaling,
            amplitude=amplitude,
            support=support,
        )

    def _ks(self, z):
        if callable(self.k_s):
            return self.k_s(z)
        return self.k_s * (1.0 + np.asarray(z)) ** self.redshift_scaling

    def profile(self, k, z, **params):
        k = np.asarray(k, dtype=float)
        ks = self._ks(z)
        x = np.maximum(k / ks, 0.0)
        return (1.0 + x ** self.slope) ** (-self.amplitude_power / self.slope)

    def update_physics(
        self,
        *,
        k_s=None,
        slope=None,
        amplitude_power=None,
        redshift_scaling=None,
        amplitude=None,
    ):
        """Update NFW-like profile parameters and invalidate dependent values."""
        if k_s is not None:
            self.k_s = k_s
        if slope is not None:
            self.slope = float(slope)
        if amplitude_power is not None:
            self.amplitude_power = float(amplitude_power)
        if redshift_scaling is not None:
            self.redshift_scaling = float(redshift_scaling)
        if amplitude is not None:
            self.amplitude = amplitude
        self._state_updated()
        return self
