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
from .multipole import BispectrumMultipole3D
from .los import LineOfSightProjector
from .support import Support3D
from .halofit import Halofit
from .analytic import (
    FactorizedBispectrum3D, FactorizedFourierMultipole3D,
    FourierPowerKernelTableConfig,
)


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
        which=("Bh1", "Bh3"),
        one_halo_kwargs: Mapping | None = None,
        three_halo_kwargs: Mapping | None = None,
    ):
        r"""Return accelerated full-Fourier BiHalofit multipoles.

        ``Bh1`` is evaluated by the shape-response approximation and ``Bh3``
        by the dressed-tree semi-analytic evaluator.  When both are selected,
        the returned object adds their multipoles.  The two child evaluators
        use the same kernel-table configuration, so the universal geometry
        cache is shared whenever their FFTLog exponent grids agree.
        """
        if not self.ready:
            raise RuntimeError("Configure cosmology, pklin, and growth first.")
        selected = _normalize_bihalofit_terms(which)
        one_halo_kwargs = dict(one_halo_kwargs or {})
        three_halo_kwargs = dict(three_halo_kwargs or {})
        # The geometry table is shared through a process-wide cache.  Passing
        # the same configuration object makes the intended common setup
        # explicit, while still allowing an expert caller to override either
        # term independently.
        shared_kernel_config = one_halo_kwargs.get(
            "kernel_table_config", three_halo_kwargs.get("kernel_table_config")
        )
        if shared_kernel_config is not None:
            one_halo_kwargs.setdefault("kernel_table_config", shared_kernel_config)
            three_halo_kwargs.setdefault("kernel_table_config", shared_kernel_config)
        shared_fftlog_config = one_halo_kwargs.get(
            "fftlog_config", three_halo_kwargs.get("fftlog_config")
        )
        if shared_fftlog_config is not None:
            one_halo_kwargs.setdefault("fftlog_config", shared_fftlog_config)
            three_halo_kwargs.setdefault("fftlog_config", shared_fftlog_config)
        return BiHalofitFourierMultipole3D(
            self,
            which=selected,
            one_halo_kwargs=one_halo_kwargs,
            three_halo_kwargs=three_halo_kwargs,
        )

    def projected_fourier_multipole(
        self,
        projector,
        *,
        sample_combination=None,
        which=("Bh1", "Bh3"),
        modes=None,
        mode_max=None,
        one_halo_kwargs: Mapping | None = None,
        three_halo_kwargs: Mapping | None = None,
    ):
        """Return the LOS projection of :meth:`fourier_multipole`."""
        m3d = self.fourier_multipole(
            which=which,
            one_halo_kwargs=one_halo_kwargs,
            three_halo_kwargs=three_halo_kwargs,
        )
        return m3d.project_los(
            projector,
            sample_combination=sample_combination,
            modes=modes,
            mode_max=mode_max,
        )

    def three_halo_fourier_multipole(
        self,
        *,
        fftlog_config=None,
        kernel_table_config=None,
    ):
        """Return full-Fourier semi-analytic multipoles of the BiHalofit 3h term.

        This is the dressed-tree implementation based on the universal
        FFTLog geometry table.  It performs no opening-angle quadrature in
        the public evaluation path.
        """
        if not self.ready:
            raise RuntimeError("Configure cosmology, pklin, and growth first.")
        return BiHalofitThreeHaloFourierMultipole3D(
            self.halofit,
            fftlog_config=fftlog_config,
            kernel_table_config=kernel_table_config,
        )

    def one_halo_response_multipole(
        self,
        *,
        reference_r1: float = 0.5,
        reference_r2: float = 0.0,
        response_step: float = 1.0e-4,
        shape_n_phi: int = 64,
        shape_mode_max: int = 8,
        fftlog_config=None,
        kernel_table_config=None,
    ):
        """Return the first-order shape-response approximation to ``Bh1``.

        The backbone is a factorized one-halo profile at fixed reference
        triangle shape.  The dependence of ``log10(an)``, ``log10(alphan)``,
        and ``log10(betan)`` on the actual BiHalofit shape variables
        ``(r1,r2)`` is restored to first order.  Shape-response Fourier
        coefficients are evaluated by a small FFT on inexpensive scalar
        coefficient functions; all radial ``k3`` factors use the universal
        FFTLog geometry table.
        """
        if not self.ready:
            raise RuntimeError(
                "BiHalofitBispectrum3D is not configured. "
                "Configure cosmology, pklin, and growth first."
            )
        return BiHalofitOneHaloResponseMultipole3D(
            self.halofit,
            reference_r1=reference_r1,
            reference_r2=reference_r2,
            response_step=response_step,
            shape_n_phi=shape_n_phi,
            shape_mode_max=shape_mode_max,
            fftlog_config=fftlog_config,
            kernel_table_config=kernel_table_config,
        )




class BiHalofitOneHaloResponseMultipole3D(BispectrumMultipole3D):
    r"""First-order response multipoles for the BiHalofit one-halo term.

    This is an experimental semi-analytic approximation.  It expands the
    profile around a fixed reference triangle shape ``(r1_ref,r2_ref)`` and
    retains the first-order response to the three shape-dependent logarithmic
    parameters ``log10(an)``, ``log10(alphan)``, and ``log10(betan)``.

    The costly radial dependence on ``k3`` is evaluated through
    :class:`FactorizedFourierMultipole3D`; only the inexpensive triangle-shape
    coefficient functions are sampled on a small Fourier grid.
    """

    basis = "fourier"
    _PARAMETERS = ("log10an", "log10aln", "log10ben")

    def __init__(
        self,
        halofit,
        *,
        reference_r1: float = 0.5,
        reference_r2: float = 0.0,
        response_step: float = 1.0e-4,
        shape_n_phi: int = 64,
        shape_mode_max: int = 8,
        fftlog_config=None,
        kernel_table_config=None,
    ):
        if not (0.0 <= reference_r1 <= 1.0 and 0.0 <= reference_r2 <= 1.0):
            raise ValueError("reference_r1 and reference_r2 must lie in [0,1]")
        if response_step <= 0.0:
            raise ValueError("response_step must be positive")
        if int(shape_n_phi) < 16 or int(shape_n_phi) % 2:
            raise ValueError("shape_n_phi must be an even integer >= 16")
        if int(shape_mode_max) < 0 or int(shape_mode_max) >= int(shape_n_phi)//2:
            raise ValueError("shape_mode_max must satisfy 0 <= M < shape_n_phi/2")
        self.halofit = halofit
        self.reference_r1 = float(reference_r1)
        self.reference_r2 = float(reference_r2)
        self.response_step = float(response_step)
        self.shape_n_phi = int(shape_n_phi)
        self.shape_mode_max = int(shape_mode_max)
        self.fftlog_config = fftlog_config
        self.kernel_table_config = (
            FourierPowerKernelTableConfig() if kernel_table_config is None else kernel_table_config
        )
        self.support = Support3D(
            k_min=float(np.min(halofit.k)),
            k_max=float(np.max(halofit.k)),
            z_min=float(np.min(halofit.z)),
            z_max=float(np.max(halofit.z)),
            policy="ignore",
        )
        self._models: dict[str, FactorizedBispectrum3D] = {}
        self._multipoles: dict[str, FactorizedFourierMultipole3D] = {}
        self._build_radial_models()

    @staticmethod
    def _clip_alpha(log10_alpha, ns):
        alpha = 10.0 ** log10_alpha
        return np.minimum(alpha, 1.0 - (2.0 / 3.0) * ns)

    def _theta(self, z, r1, r2):
        self.halofit.update()
        c = self.halofit.get_bihalofit_coeffs(np.asarray([z], dtype=float))[0]
        r1 = np.asarray(r1, dtype=float)
        r2 = np.asarray(r2, dtype=float)
        logan = c["log10an1"] + c["log10an2"] * r1 ** c["gan"]
        logaln = c["log10aln1"] + c["log10aln2"] * r2**2
        logben = c["log10ben1"] + c["log10ben2"] * r2
        return {
            "log10an": np.asarray(logan, float),
            "log10aln": np.asarray(logaln, float),
            "log10ben": np.asarray(logben, float),
            "bn": float(c["bn"]),
            "cn": float(c["cn"]),
            "r_sigma": float(c["r_sigma"]),
            "ns": float(self.halofit.cosmo["ns"]),
        }

    @staticmethod
    def _profile_q(q, theta):
        q = np.maximum(np.asarray(q, dtype=float), 1.0e-100)
        an = 10.0 ** theta["log10an"]
        alpha = np.minimum(10.0 ** theta["log10aln"], 1.0 - (2.0 / 3.0) * theta["ns"])
        beta = 10.0 ** theta["log10ben"]
        denom = an * q**alpha + theta["bn"] * q**beta
        return 1.0 / denom / (1.0 + 1.0 / (theta["cn"] * q))

    def _reference_theta(self, z):
        return self._theta(z, self.reference_r1, self.reference_r2)

    def _radial(self, kind, k, z):
        theta = self._reference_theta(float(z))
        q = np.asarray(k, float) * theta["r_sigma"]
        if kind == "base":
            return self._profile_q(q, theta)
        param = kind
        h = self.response_step
        tp = dict(theta)
        tm = dict(theta)
        tp[param] = theta[param] + h
        tm[param] = theta[param] - h
        return (self._profile_q(q, tp) - self._profile_q(q, tm)) / (2.0 * h)

    def _build_radial_models(self):
        kinds = ("base",) + self._PARAMETERS
        for kind in kinds:
            model = FactorizedBispectrum3D(
                self.halofit.k,
                lambda k, z, _kind=kind: self._radial(_kind, k, z),
                allow_signed=(kind != "base"),
            )
            self._models[kind] = model
            self._multipoles[kind] = model.analytic_multipole(
                fftlog_config=self.fftlog_config,
                kernel_table_config=self.kernel_table_config,
            )

    def build_kernel_table(self, mode_max=None):
        if mode_max is None:
            mode_max = self.shape_mode_max
        target = int(mode_max) + self.shape_mode_max
        return self._multipoles["base"].build_kernel_table(target)

    def _shape_delta_modes(self, k1, k2, z):
        """Return Fourier modes of parameter shifts around the reference."""
        k1, k2 = np.broadcast_arrays(np.asarray(k1, float), np.asarray(k2, float))
        nphi = self.shape_n_phi
        phi = 2.0*np.pi*(np.arange(nphi, dtype=float)+0.5)/float(nphi)
        cphi = np.cos(phi)
        a = k1[..., None]
        b = k2[..., None]
        k3 = np.sqrt(a*a + b*b + 2.0*a*b*cphi)
        sides = np.stack((np.broadcast_to(a, k3.shape), np.broadcast_to(b, k3.shape), k3), axis=0)
        ordered = np.sort(sides, axis=0)
        kmin, kmid, kmax = ordered
        r1 = kmin / np.maximum(kmax, np.finfo(float).tiny)
        r2 = np.maximum((kmid + kmin - kmax) / np.maximum(kmax, np.finfo(float).tiny), 0.0)
        actual = self._theta(float(z), r1, r2)
        ref = self._reference_theta(float(z))
        modes = np.arange(-self.shape_mode_max, self.shape_mode_max + 1)
        phase = np.exp(-1j * modes * np.pi / float(nphi))
        out = {}
        for name in self._PARAMETERS:
            delta = actual[name] - ref[name]
            fft = np.fft.fft(delta, axis=-1) / float(nphi)
            out[name] = phase.reshape((1,)*k1.ndim + (-1,)) * fft[..., np.mod(modes, nphi)]
        return modes, out

    @staticmethod
    def _convolve_shape(shape_modes, shape_coeff, radial_modes, wanted_modes):
        """Convolve finite shape Fourier modes with radial multipoles."""
        shape_modes = np.asarray(shape_modes, int)
        wanted_modes = np.asarray(wanted_modes, int)
        max_radial = (radial_modes.shape[0] - 1) // 2
        out = np.zeros((wanted_modes.size,) + radial_modes.shape[1:], complex)
        for ell, coeff in zip(shape_modes, np.moveaxis(shape_coeff, -1, 0)):
            index = wanted_modes - int(ell) + max_radial
            valid = (index >= 0) & (index < radial_modes.shape[0])
            if np.any(valid):
                out[valid] += coeff[None, ...] * radial_modes[index[valid]]
        return out

    def _evaluate_one_redshift(self, modes, k1, k2, z):
        ext = int(np.max(np.abs(modes))) + self.shape_mode_max
        radial_modes = np.arange(-ext, ext + 1)
        base = self._multipoles["base"].evaluate(radial_modes, k1, k2, z)
        u0_1 = self._models["base"].factor(k1, z)
        u0_2 = self._models["base"].factor(k2, z)
        # Extract the one-variable U(k3) Fourier multipoles directly.  Calling
        # a FactorizedFourierMultipole for a response factor would instead
        # construct dU(k1)dU(k2)dU(k3), which is not the derivative required
        # by the product rule.
        hi = np.maximum(k1, k2)
        ratio = np.minimum(k1, k2) / hi
        base_mp = self._multipoles["base"]
        u0_3 = base_mp._u3_stencil(
            base_mp._coefficients(float(z)), ext, hi, ratio
        )[np.abs(radial_modes)]
        shape_modes, delta = self._shape_delta_modes(k1, k2, z)
        result = base.copy()
        for param in self._PARAMETERS:
            up_1 = self._models[param].factor(k1, z)
            up_2 = self._models[param].factor(k2, z)
            response_mp = self._multipoles[param]
            up3 = response_mp._u3_stencil(
                response_mp._coefficients(float(z)), ext, hi, ratio
            )[np.abs(radial_modes)]
            derivative = (
                up_1[None, ...] * u0_2[None, ...] * u0_3
                + u0_1[None, ...] * up_2[None, ...] * u0_3
                + u0_1[None, ...] * u0_2[None, ...] * up3
            )
            result += self._convolve_shape(shape_modes, delta[param], derivative, radial_modes)
        select = modes + ext
        return result[select]

    def evaluate(self, mode, k1, k2, z, **params):
        if params:
            raise TypeError(f"Unexpected parameter(s): {', '.join(sorted(params))}")
        scalar = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, int))
        k1, k2, z = np.broadcast_arrays(np.asarray(k1, float), np.asarray(k2, float), np.asarray(z, float))
        out = np.empty((modes.size,) + k1.shape, complex)
        for z0 in np.unique(z.ravel()):
            mask = (z == z0)
            vals = self._evaluate_one_redshift(modes, k1[mask], k2[mask], float(z0))
            out.reshape(modes.size, -1)[:, mask.ravel()] = vals.reshape(modes.size, -1)
        out = np.real_if_close(out, tol=500)
        return out[0] if scalar else out


class BiHalofitThreeHaloFourierMultipole3D(BispectrumMultipole3D):
    r"""Full-Fourier semi-analytic BiHalofit three-halo multipoles.

    Write ``D(k)=I(k)`` and ``H(k)=I(k)P_E(k)``.  The BiHalofit three-halo
    contribution is a dressed tree bispectrum.  The angular dependence is
    reduced to Fourier stencils of ``D(k3) k3**s`` and ``H(k3) k3**s`` with
    shifts ``s in {-2,0,1,2,4}``, all contracted against the same universal
    geometry table used by the tree-level implementation.
    """

    basis = "fourier"

    def __init__(self, halofit, *, fftlog_config=None, kernel_table_config=None):
        self.halofit = halofit
        self.fftlog_config = fftlog_config
        self.kernel_table_config = (
            FourierPowerKernelTableConfig() if kernel_table_config is None else kernel_table_config
        )
        self.support = Support3D(
            k_min=float(np.min(halofit.k)), k_max=float(np.max(halofit.k)),
            z_min=float(np.min(halofit.z)), z_max=float(np.max(halofit.z)), policy="ignore",
        )
        self._models = {
            "D": FactorizedBispectrum3D(halofit.k, lambda k,z: self._radial("D", k, z)),
            "H": FactorizedBispectrum3D(halofit.k, lambda k,z: self._radial("H", k, z)),
        }
        self._multipoles = {
            name: model.analytic_multipole(
                fftlog_config=self.fftlog_config,
                kernel_table_config=self.kernel_table_config,
            )
            for name, model in self._models.items()
        }

    def build_kernel_table(self, mode_max=16):
        """Build the shared universal geometry table up to ``mode_max``."""
        return self._multipoles["D"].build_kernel_table(int(mode_max) + 2)

    def _coeff(self, z):
        self.halofit.update()
        return self.halofit.get_bihalofit_coeffs(np.asarray([float(z)]))[0]

    def _radial_values(self, k, z, c=None):
        if c is None:
            c = self._coeff(z)
        k = np.asarray(k, float)
        q = k * float(c["r_sigma"])
        D = 1.0 / (1.0 + float(c["en"]) * q)
        PL = self.halofit.get_interpolated_pklin(k, float(z))
        PE = (
            ((1.0 + float(c["fn"]) * q*q) / (1.0 + float(c["gn"]) * q + float(c["hn"]) * q*q)) * PL
            + 1.0 / (float(c["mn"]) * q**float(c["mun"]) + float(c["nn"]) * q**float(c["nun"]))
              / (1.0 + (float(c["pn"]) * q)**-3)
        )
        return D, PE, D * PE

    def _radial(self, which, k, z):
        D, _, H = self._radial_values(k, z)
        return D if which == "D" else H

    @staticmethod
    def _T12_coefficients(k1, k2):
        D = 2.0*k1*k2
        S = k1*k1 + k2*k2
        R = 0.5*(k1/k2 + k2/k1)
        return (
            5.0/7.0 - R*S/D + (2.0/7.0)*S*S/(D*D),
            R/D - (4.0/7.0)*S/(D*D),
            (2.0/7.0)/(D*D),
        )

    @staticmethod
    def _cyclic_coefficients(k1, k2, PE1, PE2, eta):
        def variable(kfixed, kopp):
            U=(kopp*kopp-kfixed*kfixed)/(2.0*kfixed)
            V=-1.0/(2.0*kfixed)
            return (
                0.5*kfixed*U+(2.0/7.0)*U*U,
                5.0/7.0+0.5*(kfixed*V+U/kfixed)+(4.0/7.0)*U*V,
                0.5*V/kfixed+(2.0/7.0)*V*V,
            )
        a=variable(k2,k1); b=variable(k1,k2)
        return (
            PE2*a[0]+PE1*b[0],
            PE2*a[1]+PE1*b[1]+eta*(k1*PE2+k2*PE1),
            PE2*a[2]+PE1*b[2],
        )

    @staticmethod
    def _select(stencil, modes):
        return stencil[np.abs(np.asarray(modes, int))]

    def _evaluate_at_z(self, modes, k1, k2, z):
        modes = np.asarray(modes, int)
        max_mode = int(np.max(np.abs(modes))) + 2
        c = self._coeff(z)
        D1, PE1, _ = self._radial_values(k1, z, c=c)
        D2, PE2, _ = self._radial_values(k2, z, c=c)
        hi = np.maximum(k1, k2)
        r = np.minimum(k1, k2) / hi
        coeffD = self._multipoles["D"]._coefficients(float(z))
        coeffH = self._multipoles["H"]._coefficients(float(z))
        mD = self._multipoles["D"]
        mH = self._multipoles["H"]

        # Direct (12) channel: D3 times a quartic polynomial in k3.
        C0, C2, C4 = self._T12_coefficients(k1, k2)
        eta = float(c["dn"]) * float(c["r_sigma"])
        D0 = self._select(mD._u3_stencil_shift(coeffD, max_mode, hi, r, 0), modes)
        D1s = self._select(mD._u3_stencil_shift(coeffD, max_mode, hi, r, 1), modes)
        D2s = self._select(mD._u3_stencil_shift(coeffD, max_mode, hi, r, 2), modes)
        D4s = self._select(mD._u3_stencil_shift(coeffD, max_mode, hi, r, 4), modes)
        T12 = 2.0 * (D1 * D2 * PE1 * PE2)[None, ...] * (
            C0[None, ...]*D0 + eta*D1s + C2[None, ...]*D2s + C4[None, ...]*D4s
        )

        # Pair-combined cyclic channels.  The k3^-2 coefficient is combined
        # before using the scaled universal kernel, retaining r -> 1 regularity.
        Em2, E0, E2 = self._cyclic_coefficients(k1, k2, PE1, PE2, eta)
        H0 = self._select(mH._u3_stencil_shift(coeffH, max_mode, hi, r, 0), modes)
        H2 = self._select(mH._u3_stencil_shift(coeffH, max_mode, hi, r, 2), modes)
        Hm2_scaled = self._select(mH._u3_stencil_shift(coeffH, max_mode, hi, r, -2), modes)
        scale = (1.0-r*r)**2
        Tcyc = 2.0 * (D1*D2)[None, ...] * (E0[None,...]*H0 + E2[None,...]*H2)
        diagonal = np.isclose(r, 1.0, rtol=0.0, atol=8*np.finfo(float).eps)
        if np.any(~diagonal):
            regular_m2 = np.where(diagonal, 0.0, Em2/scale)
            Tcyc += 2.0 * (D1*D2)[None,...] * regular_m2[None,...] * Hm2_scaled
        if np.any(diagonal):
            # Exact k1=k2 identity for the pair-combined cyclic sum.
            k = k1
            diag = (D1*D1*PE1)[None,...] * (
                (13.0/7.0)*H0 - (5.0/7.0)*H2 + 4.0*eta*k[None,...]*H0
            )
            Tcyc = np.where(diagonal[None,...], diag, Tcyc)
        return T12 + Tcyc

    def evaluate(self, mode, k1, k2, z, **params):
        if params:
            raise TypeError(f"Unexpected parameter(s): {', '.join(sorted(params))}")
        scalar = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, int))
        k1,k2,z = np.broadcast_arrays(np.asarray(k1,float), np.asarray(k2,float), np.asarray(z,float))
        if np.any(k1 <= 0.0) or np.any(k2 <= 0.0):
            raise ValueError("k1 and k2 must be positive")
        out=np.empty((modes.size,) + k1.shape, complex)
        for zi in np.unique(z):
            mask=(z==zi)
            out.reshape(modes.size,-1)[:,mask.ravel()] = self._evaluate_at_z(modes,k1[mask],k2[mask],float(zi)).reshape(modes.size,-1)
        out=np.real_if_close(out,tol=1000)
        return out[0] if scalar else out


class BiHalofitFourierMultipole3D(BispectrumMultipole3D):
    r"""Combined accelerated full-Fourier BiHalofit multipoles.

    This class is a thin composition of the retained one-halo response and
    three-halo dressed-tree evaluators.  It contains no legacy cosine-basis
    machinery and evaluates

    .. math:: B_m = B_{
m h1,m} + B_{
m h3,m}

    when both terms are requested.
    """

    basis = "fourier"

    def __init__(
        self,
        bihalofit: BiHalofitBispectrum3D,
        *,
        which=("Bh1", "Bh3"),
        one_halo_kwargs: Mapping | None = None,
        three_halo_kwargs: Mapping | None = None,
    ):
        self.bihalofit = bihalofit
        self.which = _normalize_bihalofit_terms(which)
        self.support = bihalofit.support
        self.one_halo = None
        self.three_halo = None
        if "Bh1" in self.which:
            self.one_halo = bihalofit.one_halo_response_multipole(
                **dict(one_halo_kwargs or {})
            )
        if "Bh3" in self.which:
            self.three_halo = bihalofit.three_halo_fourier_multipole(
                **dict(three_halo_kwargs or {})
            )

    def build_kernel_table(self, mode_max=16):
        """Build one common geometry-table extent for the selected terms.

        The underlying factorized evaluators are assigned the same requested
        extent so that their class-level universal-table cache resolves to a
        single immutable geometry object whenever their FFTLog exponent grids
        coincide.
        """
        mode_max = int(mode_max)
        extra = 2
        if self.one_halo is not None:
            extra = max(extra, self.one_halo.shape_mode_max)
        required = mode_max + extra
        built = {}
        if self.one_halo is not None:
            built["Bh1"] = {
                name: multipole.build_kernel_table(required)
                for name, multipole in self.one_halo._multipoles.items()
            }
        if self.three_halo is not None:
            built["Bh3"] = {
                name: multipole.build_kernel_table(required)
                for name, multipole in self.three_halo._multipoles.items()
            }
        return built

    def evaluate(self, mode, k1, k2, z, **params):
        if params:
            unknown = ", ".join(sorted(params))
            raise TypeError(f"Unexpected BiHalofit multipole parameter(s): {unknown}")
        total = None
        if self.one_halo is not None:
            total = self.one_halo.evaluate(mode, k1, k2, z)
        if self.three_halo is not None:
            value = self.three_halo.evaluate(mode, k1, k2, z)
            total = value if total is None else total + value
        return total



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
