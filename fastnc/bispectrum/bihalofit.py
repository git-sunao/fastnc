"""Direct 3D BiHalofit bispectrum model.

This module adapts the bundled :class:`Halofit` implementation to the
:class:`Bispectrum3D` interface.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .base import Bispectrum3D
from .support import Support3D
from .halofit import Halofit
from .analytic import (
    BiHalofitBispectrumMultipole3D,
    PowerLawAngularKernelTableConfig,
)
from fastnc.hankel.wrapper import PowerLawFFTLogConfig
from fastnc.utils.cosmology import (
    default_wmap_like_cosmology,
    eisenstein_hu_like_pklin,
    simple_debug_pklin,
    simple_linear_growth,
)


def _normalize_bihalofit_terms(which):
    """Validate and canonicalize a BiHalofit term selection."""
    if isinstance(which, str):
        which = (which,)
    else:
        which = tuple(which)
    valid = {"Bh1", "Bh3"}
    invalid = set(which) - valid
    if invalid:
        raise ValueError(f"Unknown BiHalofit term(s): {sorted(invalid)}")
    if not which:
        raise ValueError("Select at least one of 'Bh1' or 'Bh3'.")
    return tuple(term for term in ("Bh1", "Bh3") if term in which)


class BiHalofitBispectrum3D(Bispectrum3D):
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
    ):
        self.halofit = halofit or Halofit()
        self._support_policy = support_policy
        self._user_support = support is not None
        self.support = support or Support3D(policy=support_policy)

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
        return self.halofit.get_bihalofit(k1, k2, k3, z, **params)

    def fourier_multipole(
        self,
        *,
        r1: float = 0.5,
        r2: float = 0.0,
        k_grid=None,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
    ):
        """Return semi-analytic multipoles of the complete BiHalofit model.

        The one-halo fitting shape parameters are fixed to ``r1`` and ``r2``,
        making the one-halo product separable.  The complete three-halo term
        is included in the same composite object.
        """
        if not self.ready:
            raise RuntimeError("Configure cosmology, pklin, and growth first.")
        return BiHalofitBispectrumMultipole3D(
            self.halofit,
            r1=r1,
            r2=r2,
            k_grid=k_grid,
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
        )

    def projected_fourier_multipole(
        self,
        projector,
        *,
        sample_combination=None,
        modes=None,
        mode_max=None,
        r1: float = 0.5,
        r2: float = 0.0,
        k_grid=None,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
    ):
        """Return the coefficient-level LOS projection of the multipoles."""
        return self.fourier_multipole(
            r1=r1,
            r2=r2,
            k_grid=k_grid,
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
        ).project_los(
            projector,
            sample_combination=sample_combination,
            modes=modes,
            mode_max=mode_max,
        )

