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
from .halofit import Halofit, HalofitMultipole
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

    def analytic_multipole(
        self,
        *,
        which=("Bh3",),
        **params,
    ):
        """Return the semi-analytic 3D BiHalofit multipole object.

        The returned object evaluates ``B_L^3D(k1,k2,z)``.  LOS projection is
        applied later through ``LineOfSightProjector.as_multipole_projector()``
        or ``projected_analytic_multipole``.
        """
        return BiHalofitBispectrumMultipole3D(
            self.halofit,
            which=which,
            **params,
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

    def projected_analytic_multipole(
        self,
        projector,
        *,
        sample_combination=None,
        which=("Bh3",),
        modes=None,
        mode_max=None,
        **params,
    ):
        """Return the LOS-projected semi-analytic angular multipole object."""
        m3d = self.analytic_multipole(which=which, **params)
        return m3d.project_los(
            projector,
            sample_combination=sample_combination,
            modes=modes,
            mode_max=mode_max,
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


class BiHalofitBispectrumMultipole3D(BispectrumMultipole3D):
    """Semi-analytic 3D multipole for BiHalofit.

    The public name intentionally omits ``3h``.  Currently the implemented
    semi-analytic term is the BiHalofit 3-halo contribution, so ``which`` must
    be equivalent to ``["Bh3"]``.  The ``which`` interface is kept for future
    extension to ``["Bh1", "Bh3"]``.
    """

    basis = "fourier-even"

    def __init__(
        self,
        halofit,
        *,
        which=("Bh3",),
        n_fftlog=128,
        k_fft_min=None,
        k_fft_max=None,
        fftlog_pad=4.0,
        bias_D=0.0,
        bias_H=0.0,
        kernel_method="auto",
        n_kernel_phi=128,
        cyclic_r_quad=0.97,
        cyclic_quad_n_phi=256,
    ):
        self.which = tuple(which) if isinstance(which, (list, tuple)) else (which,)
        if set(self.which) != {"Bh3"}:
            raise NotImplementedError(
                "BiHalofitBispectrumMultipole3D currently implements only which=['Bh3']. "
                "The which argument is reserved for future Bh1+Bh3 support."
            )

        self.halofit = halofit.to_multipole(
            n_fftlog=n_fftlog,
            k_fft_min=k_fft_min,
            k_fft_max=k_fft_max,
            fftlog_pad=fftlog_pad,
            bias_D=bias_D,
            bias_H=bias_H,
            n_kernel_phi=n_kernel_phi,
        ) if not isinstance(halofit, HalofitMultipole) else halofit

        if isinstance(halofit, HalofitMultipole):
            self.halofit.n_fftlog = int(n_fftlog)
            self.halofit.k_fft_min = k_fft_min
            self.halofit.k_fft_max = k_fft_max
            self.halofit.fftlog_pad = float(fftlog_pad)
            self.halofit.bias_D = float(bias_D)
            self.halofit.bias_H = float(bias_H)
            self.halofit.n_kernel_phi = int(n_kernel_phi)
            self.halofit._clear_radial_fftlog_cache()

        self.kernel_method = kernel_method
        self.cyclic_r_quad = float(cyclic_r_quad)
        self.cyclic_quad_n_phi = int(cyclic_quad_n_phi)

    def evaluate(self, mode, k1, k2, z, **params):
        scalar_mode = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, dtype=int)).ravel()
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        z = np.asarray(z, dtype=float)
        k1, k2, z = np.broadcast_arrays(k1, k2, z)

        kernel_method = params.pop("kernel_method", self.kernel_method)
        cyclic_r_quad = params.pop("cyclic_r_quad", self.cyclic_r_quad)
        cyclic_quad_n_phi = params.pop("cyclic_quad_n_phi", self.cyclic_quad_n_phi)
        if params:
            unknown = ", ".join(sorted(params))
            raise TypeError(f"Unexpected BiHalofit multipole parameter(s): {unknown}")

        vals = []
        z_flat = z.ravel()
        unique_z = np.unique(z_flat)
        for L in modes:
            out_L = np.empty_like(k1, dtype=complex)
            for z0 in unique_z:
                mask = (z == z0)
                out_L[mask] = self.halofit.get_bihalofit_3h_multipole_semianalytic(
                    k1[mask],
                    k2[mask],
                    L=int(L),
                    z=float(z0),
                    kernel_method=kernel_method,
                    cyclic_r_quad=cyclic_r_quad,
                    cyclic_quad_n_phi=cyclic_quad_n_phi,
                    return_parts=False,
                )
            vals.append(out_L)
        out = np.asarray(vals)
        out = np.real_if_close(out, tol=1000)
        return out[0] if scalar_mode else out




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
