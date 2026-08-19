"""Direct one-halo 3D bispectrum models.

The models in this module implement the :class:`Bispectrum3D` interface
directly. Named constructors such as ``NFWOneHaloBispectrum3D.default()``
provide convenience parameter choices, but the model classes themselves are
not presets.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import Bispectrum3D
from .support import Support3D


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
        self.profile = profile
        self.amplitude = amplitude
        self.support = support or Support3D(policy="ignore")

    def _amplitude(self, z):
        if callable(self.amplitude):
            return self.amplitude(z)
        return self.amplitude

    def evaluate(self, k1, k2, k3, z, **params):
        amp = self._amplitude(z)
        u1 = self.profile(k1, z, **params)
        u2 = self.profile(k2, z, **params)
        u3 = self.profile(k3, z, **params)
        return amp * u1 * u2 * u3


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
        return cls(k_s=1.0, slope=2.0, amplitude_power=2.0, redshift_scaling=0.0, amplitude=1.0, support=support)

    @classmethod
    def shallow(cls, support: Support3D | None = None) -> "NFWOneHaloBispectrum3D":
        """Return a shallower high-k profile preset."""
        return cls(k_s=1.0, slope=1.0, amplitude_power=2.0, redshift_scaling=0.0, amplitude=1.0, support=support)

    @classmethod
    def steep(cls, support: Support3D | None = None) -> "NFWOneHaloBispectrum3D":
        """Return a steeper high-k profile preset."""
        return cls(k_s=1.0, slope=3.0, amplitude_power=2.0, redshift_scaling=0.0, amplitude=1.0, support=support)

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
