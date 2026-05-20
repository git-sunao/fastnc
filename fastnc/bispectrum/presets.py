"""Preset and convenience 3D bispectrum models.

This module keeps the class hierarchy tied to *model behavior* rather than to
particular parameter presets.  Common/debug initializations are exposed as
classmethod factories, e.g. ``BiHalofitBispectrum3D.simple_debug()`` and
``NFWOneHaloBispectrum3D.default()``.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .base import Bispectrum3D
from .models import ExternalBispectrum3D
from .support import Support3D
from .halofit import Halofit


_DEFAULT_COSMO_WMAP_LIKE = {
    "Om0": 0.279,
    "Ode0": 0.721,
    "ns": 0.972,
    "w0": -1.0,
    "wa": 0.0,
    "fnu0": 0.0,
    "sigma8": 0.82,
    "h": 0.70,
    "Ob0": 0.046,
}


def default_wmap_like_cosmology() -> dict[str, float]:
    """Return a WMAP-like debug cosmology dictionary.

    This is intended for code-path tests, not precision calculations.
    """
    return dict(_DEFAULT_COSMO_WMAP_LIKE)


class BiHalofitBispectrum3D(ExternalBispectrum3D):
    """Wrapper for the bundled :class:`Halofit` / Bihalofit model.

    This class does not self-initialize by default.  A scientific run should
    configure it explicitly with cosmology, linear power spectrum, and growth
    factor.

    Use ``BiHalofitBispectrum3D.simple_debug()`` only for quick tests.
    """

    def __init__(
        self,
        halofit: Halofit | None = None,
        support: Support3D | None = None,
        support_policy: str = "zero",
        method_name: str = "get_bihalofit",
    ):
        self.halofit = halofit or Halofit()
        self._support_policy = support_policy
        self._user_support = support is not None
        super().__init__(
            self.halofit,
            support=support or Support3D(policy=support_policy),
            method_name=method_name,
        )

    @property
    def ready(self) -> bool:
        return (
            getattr(self.halofit, "cosmo", None) is not None
            and getattr(self.halofit, "k", None) is not None
            and getattr(self.halofit, "pklin", None) is not None
            and getattr(self.halofit, "z", None) is not None
            and getattr(self.halofit, "lgr", None) is not None
        )

    def _refresh_support(self) -> None:
        if self._user_support:
            return
        k = getattr(self.halofit, "k", None)
        z = getattr(self.halofit, "z", None)
        if k is None or z is None:
            return
        k = np.asarray(k, dtype=float)
        z = np.asarray(z, dtype=float)
        self.support = Support3D(
            k_min=float(np.nanmin(k)),
            k_max=float(np.nanmax(k)),
            z_min=float(np.nanmin(z)),
            z_max=float(np.nanmax(z)),
            policy=self._support_policy,
        )

    def set_cosmology(self, cosmo: Mapping[str, float]) -> "BiHalofitBispectrum3D":
        self.halofit.set_cosmology(dict(cosmo))
        self._refresh_support()
        return self

    def set_pklin(self, k, pklin) -> "BiHalofitBispectrum3D":
        self.halofit.set_pklin(np.asarray(k, dtype=float), np.asarray(pklin, dtype=float))
        self._refresh_support()
        return self

    def set_growth(self, z, growth) -> "BiHalofitBispectrum3D":
        """Set the linear growth factor/grid used by the bundled Halofit code."""
        self.halofit.set_lgr(np.asarray(z, dtype=float), np.asarray(growth, dtype=float))
        self._refresh_support()
        return self

    def set_lgr(self, z, lgr) -> "BiHalofitBispectrum3D":
        """Alias matching the naming in ``halofit.py``."""
        return self.set_growth(z, lgr)

    def configure(
        self,
        *,
        cosmo: Mapping[str, float] | None = None,
        k=None,
        pklin=None,
        z=None,
        growth=None,
        lgr=None,
    ) -> "BiHalofitBispectrum3D":
        """Configure this instance in-place and return ``self``."""
        if cosmo is not None:
            self.set_cosmology(cosmo)
        if k is not None or pklin is not None:
            if k is None or pklin is None:
                raise ValueError("Both k and pklin must be supplied together.")
            self.set_pklin(k, pklin)
        if z is not None or growth is not None or lgr is not None:
            g = growth if growth is not None else lgr
            if z is None or g is None:
                raise ValueError("Both z and growth/lgr must be supplied together.")
            self.set_growth(z, g)
        return self

    @classmethod
    def from_cosmology(
        cls,
        *,
        cosmo: Mapping[str, float],
        k,
        pklin,
        z,
        growth=None,
        lgr=None,
        support: Support3D | None = None,
        support_policy: str = "zero",
        halofit: Halofit | None = None,
    ) -> "BiHalofitBispectrum3D":
        """Create and configure a Bihalofit wrapper from user inputs."""
        obj = cls(halofit=halofit, support=support, support_policy=support_policy)
        return obj.configure(cosmo=cosmo, k=k, pklin=pklin, z=z, growth=growth, lgr=lgr)

    @classmethod
    def simple_debug(
        cls,
        *,
        cosmo: Mapping[str, float] | None = None,
        k=None,
        z=None,
        amplitude: float = 1.0e4,
        k_eq: float = 2.0e-2,
        transfer_power: float = 1.5,
        pklin_kind: str = "debug",
        support_policy: str = "zero",
    ) -> "BiHalofitBispectrum3D":
        """Return a self-initialized Bihalofit object for debugging.

        The linear spectrum and growth are deliberately simple and should not
        be used for scientific inference.  For production, use
        ``from_cosmology`` with CAMB/CLASS inputs.
        """
        cosmo_dict = default_wmap_like_cosmology() if cosmo is None else dict(cosmo)
        k_arr = np.logspace(-4, 2, 512) if k is None else np.asarray(k, dtype=float)
        z_arr = np.linspace(0.0, 3.0, 128) if z is None else np.asarray(z, dtype=float)

        if pklin_kind == "debug":
            pklin = simple_debug_pklin(
                k_arr,
                cosmo=cosmo_dict,
                amplitude=amplitude,
                k_eq=k_eq,
                transfer_power=transfer_power,
            )
        elif pklin_kind in {"eisenstein-hu", "eisenstein_hu", "eh"}:
            pklin = eisenstein_hu_like_pklin(k_arr, cosmo=cosmo_dict, amplitude=amplitude)
        else:
            raise ValueError("pklin_kind must be 'debug' or 'eisenstein-hu'.")

        growth = simple_linear_growth(z_arr, cosmo=cosmo_dict)
        return cls.from_cosmology(
            cosmo=cosmo_dict,
            k=k_arr,
            pklin=pklin,
            z=z_arr,
            growth=growth,
            support_policy=support_policy,
        )

    def evaluate(self, k1, k2, k3, z, **params):
        if not self.ready:
            raise RuntimeError(
                "BiHalofitBispectrum3D is not configured. "
                "Call set_cosmology(), set_pklin(), and set_growth(), or use "
                "BiHalofitBispectrum3D.from_cosmology(...)."
            )
        return super().evaluate(k1, k2, k3, z, **params)


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


def simple_linear_growth(z, cosmo: Mapping[str, float] | None = None):
    """Simple debug growth factor, normalized to D(0)=1."""
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + z)


def simple_debug_pklin(
    k,
    cosmo: Mapping[str, float] | None = None,
    amplitude: float = 1.0e4,
    k_eq: float = 2.0e-2,
    transfer_power: float = 1.5,
):
    """Smooth positive debug linear power spectrum.

    This is EH/BBKS-like in spirit but intentionally chosen with a stable
    high-k tail for the bundled Halofit implementation.
    """
    cosmo = _DEFAULT_COSMO_WMAP_LIKE if cosmo is None else cosmo
    ns = float(cosmo.get("ns", 0.97))
    k = np.asarray(k, dtype=float)
    return amplitude * k**ns / (1.0 + (k / k_eq) ** 2) ** transfer_power


def eisenstein_hu_like_pklin(
    k,
    cosmo: Mapping[str, float] | None = None,
    amplitude: float = 1.0e4,
):
    """Crude Eisenstein-Hu/BBKS-like no-wiggle spectrum for diagnostics.

    This is not a replacement for CAMB/CLASS.  It is provided only as a
    convenient preset for debugging code paths.
    """
    cosmo = _DEFAULT_COSMO_WMAP_LIKE if cosmo is None else cosmo
    ns = float(cosmo.get("ns", 0.97))
    Om0 = float(cosmo.get("Om0", 0.279))
    h = float(cosmo.get("h", 0.70))
    theta = 2.7255 / 2.7
    gamma_eff = Om0 * h / theta**2
    q = np.asarray(k, dtype=float) / gamma_eff
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T = L0 / (L0 + C0 * q**2)
    return amplitude * np.asarray(k, dtype=float) ** ns * T**2
