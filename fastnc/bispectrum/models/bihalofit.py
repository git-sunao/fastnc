"""Direct 3D BiHalofit bispectrum model.

This module adapts the bundled :class:`Halofit` implementation to the
:class:`Bispectrum3D` interface.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..base import Bispectrum3D
from ..terms import BackendBispectrumTerm
from ..slepian import BackendSlepianTerm, SlepianLOSMomentMetadata
from ..support import Support3D
from .halofit import Halofit
from ..analytic import (
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



_CYCLIC_LEGS = {
    "12": (0, 1, 2),
    "23": (1, 2, 0),
    "31": (2, 0, 1),
}

_BIHALOFIT_F2_SLEPIAN_SPECS = (
    (12.0 / 7.0,  0, ( 0,  0)),
    ( 1.0 / 2.0,  1, ( 1, -1)),
    ( 1.0 / 2.0, -1, ( 1, -1)),
    ( 1.0 / 2.0,  1, (-1,  1)),
    ( 1.0 / 2.0, -1, (-1,  1)),
    ( 1.0 / 7.0,  2, ( 0,  0)),
    ( 1.0 / 7.0, -2, ( 0,  0)),
)


def _pair_cosine(k1, k2, k3):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2)


class BiHalofitBh3PairTerm(BackendBispectrumTerm):
    """Exact regular BiHalofit 3-halo cyclic sector for the generic route."""

    def __init__(self, backend, pair: str, model=None):
        super().__init__(backend)
        if pair not in _CYCLIC_LEGS:
            raise ValueError("pair must be one of '12', '23', or '31'")
        self.pair = pair
        self._model = model

    @property
    def model(self):
        return self._model

    def evaluate(self, k1, k2, k3, z, **params):
        ks = [np.asarray(k1, dtype=float), np.asarray(k2, dtype=float), np.asarray(k3, dtype=float)]
        i, j, other = _CYCLIC_LEGS[self.pair]
        ri = self.backend.get_bihalofit_3h_radials(ks[i], z)
        rj = self.backend.get_bihalofit_3h_radials(ks[j], z)
        ro = self.backend.get_bihalofit_3h_radials(ks[other], z)
        mu = _pair_cosine(ks[i], ks[j], ks[other])
        F2 = 5.0 / 7.0 + 0.5 * mu * (ks[i] / ks[j] + ks[j] / ks[i]) + 2.0 / 7.0 * mu**2
        return (
            2.0 * F2 * ri["H"] * rj["H"] * ro["I"]
            + 2.0 * ro["dn"] * ro["r_sigma"] * ri["H"] * rj["H"] * ro["kI"]
        )


class BiHalofitSlepianF2Term(BackendSlepianTerm):
    """One of the seven separable exponential terms in a BiHalofit F2 pair."""

    def __init__(self, backend, pair: str, *, coefficient, harmonic, shifts, model=None):
        super().__init__(backend)
        if pair not in _CYCLIC_LEGS:
            raise ValueError("pair must be one of '12', '23', or '31'")
        self._model = model
        self.pair = pair
        i, j, other = _CYCLIC_LEGS[pair]
        self.paired_legs = (i, j)
        self.other_leg = other
        self.coefficient = float(coefficient)
        self.harmonic = int(harmonic)
        shifts = tuple(int(v) for v in shifts)
        self.pair_power_shifts = shifts
        global_shifts = [0, 0, 0]
        global_shifts[i], global_shifts[j] = shifts
        self.power_shifts = tuple(global_shifts)
        orders = [0, 0, 0]
        orders[i], orders[j] = self.harmonic, -self.harmonic
        self._angular_orders = tuple(orders)
        self._los_metadata = SlepianLOSMomentMetadata(
            kind="general-coefficient",
            signature=("bihalofit-F2", pair, self.power_shifts),
        )

    @property
    def model(self):
        return self._model

    @property
    def angular_orders(self):
        return self._angular_orders

    @property
    def los_moment_metadata(self):
        return self._los_metadata

    @property
    def radial_signature(self):
        return ("bihalofit-HHI", self.pair, self.power_shifts)

    def c(self, z, **params):
        return self.coefficient

    def _radial_leg(self, leg, k, z):
        r = self.backend.get_bihalofit_3h_radials(k, z)
        if leg == self.other_leg:
            return r["I"]
        return r["H"] * np.asarray(k, dtype=float) ** self.power_shifts[leg]

    def f1(self, k, z, **params): return self._radial_leg(0, k, z)
    def f2(self, k, z, **params): return self._radial_leg(1, k, z)
    def f3(self, k, z, **params): return self._radial_leg(2, k, z)


class BiHalofitSlepianDnTerm(BackendSlepianTerm):
    """Separable isotropic ``2 d_n q_other`` contribution for one cyclic pair."""

    def __init__(self, backend, pair: str, model=None):
        super().__init__(backend)
        if pair not in _CYCLIC_LEGS:
            raise ValueError("pair must be one of '12', '23', or '31'")
        self._model = model
        self.pair = pair
        i, j, other = _CYCLIC_LEGS[pair]
        self.paired_legs = (i, j)
        self.other_leg = other
        self.coefficient = 1.0
        self.power_shifts = (0, 0, 0)
        self._angular_orders = (0, 0, 0)
        self._los_metadata = SlepianLOSMomentMetadata(
            kind="general-coefficient",
            signature=("bihalofit-dn", pair),
        )

    @property
    def model(self):
        return self._model

    @property
    def angular_orders(self):
        return self._angular_orders

    @property
    def los_moment_metadata(self):
        return self._los_metadata

    @property
    def radial_signature(self):
        return ("bihalofit-dn-HHkI", self.pair)

    def c(self, z, **params):
        zz = np.asarray(z, dtype=float)
        # k is irrelevant for r_sigma and d_n; use a harmless in-support probe.
        probe = np.ones_like(zz) * max(float(np.nanmin(self.backend.k)), 1.0e-6)
        r = self.backend.get_bihalofit_3h_radials(probe, zz)
        return 2.0 * r["dn"] * r["r_sigma"]

    def _radial_leg(self, leg, k, z):
        r = self.backend.get_bihalofit_3h_radials(k, z)
        if leg == self.other_leg:
            return r["kI"]
        return r["H"]

    def f1(self, k, z, **params): return self._radial_leg(0, k, z)
    def f2(self, k, z, **params): return self._radial_leg(1, k, z)
    def f3(self, k, z, **params): return self._radial_leg(2, k, z)


def _build_bihalofit_slepian_pair_terms(model, pair):
    f2 = tuple(
        BiHalofitSlepianF2Term(
            model.backend, pair, coefficient=c, harmonic=h, shifts=sh, model=model
        )
        for c, h, sh in _BIHALOFIT_F2_SLEPIAN_SPECS
    )
    return f2 + (BiHalofitSlepianDnTerm(model.backend, pair, model=model),)

class BiHalofitTerm(BackendBispectrumTerm):
    """One BiHalofit physical sector sharing the model's Halofit backend."""

    def __init__(self, backend, which: str, model=None):
        super().__init__(backend)
        self._model = model
        if which not in {"Bh1", "Bh3"}:
            raise ValueError("which must be 'Bh1' or 'Bh3'")
        self.which = which

    @property
    def model(self):
        return self._model

    def evaluate(self, k1, k2, k3, z, **params):
        if self.model is not None:
            params = self.model._merge_default_kwargs(params)
        else:
            params = dict(params)
        params.pop("which", None)
        return self.backend.get_bihalofit(k1, k2, k3, z, which=self.which, **params)


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
        self._bh1_term = BiHalofitTerm(self.halofit, "Bh1", model=self)
        self._bh3_pair_terms = {
            pair: BiHalofitBh3PairTerm(self.halofit, pair, model=self)
            for pair in ("12", "23", "31")
        }
        self._bh3_slepian_terms = (
            _build_bihalofit_slepian_pair_terms(self, "12")
            + _build_bihalofit_slepian_pair_terms(self, "31")
        )
        self._generic_terms_by_name = {"Bh1": self._bh1_term}
        self._full_generic_terms_by_name = {
            "Bh1": self._bh1_term,
            "Bh3": BiHalofitTerm(self.halofit, "Bh3", model=self),
        }

    @property
    def backend(self):
        """Shared stateful physics backend used by future term views.

        The existing :class:`Halofit` instance remains the single source of
        truth.  Exposing it through this read-only property avoids creating a
        second state container during the term refactor.
        """
        return self.halofit

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

    supports_slepian = True
    slepian_los_kind = "general-coefficient"

    def generic_terms(self, **params):
        """Return the regular BiHalofit sectors for the active hybrid split.

        ``Bh1`` remains fully generic.  For ``Bh3`` only the regular B23 pair
        is generic; the high-L B12+B31 sectors are exposed by
        :meth:`slepian_terms`.  In Slepian ``mode='off'`` the hybrid
        calculator recombines both collections before generic projection.
        """
        merged = self._merge_default_kwargs(params)
        names = _normalize_bihalofit_terms(merged.get("which", ("Bh1", "Bh3")))
        out = []
        if "Bh1" in names:
            out.append(self._bh1_term)
        if "Bh3" in names:
            out.append(self._bh3_pair_terms["23"])
        return tuple(out)

    def slepian_terms(self, **params):
        """Return 7+1 terms for B12 and 7+1 terms for B31 of ``Bh3``."""
        merged = self._merge_default_kwargs(params)
        names = _normalize_bihalofit_terms(merged.get("which", ("Bh1", "Bh3")))
        return self._bh3_slepian_terms if "Bh3" in names else ()

    def generic_fallback_terms(self, **params):
        """Return unsplit direct sectors for exact ``SlepianConfig(mode='off')`` fallback.

        This preserves the historical squeezed-safe Bh3 evaluator in the debug
        route.  The hybrid partition itself uses the exact separable F2 algebra;
        the safe cyclic cancellation cannot be assigned to individual 12/31
        terms without destroying that decomposition.
        """
        merged = self._merge_default_kwargs(params)
        names = _normalize_bihalofit_terms(merged.get("which", ("Bh1", "Bh3")))
        return tuple(self._full_generic_terms_by_name[name] for name in names)

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

