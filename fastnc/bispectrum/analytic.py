"""Analytic and semi-analytic bispectrum building blocks.

This module implements the tree-level matter bispectrum in the full Fourier
basis.  The linear spectrum is represented by a shared complex-power FFTLog
expansion.  Its third-side Fourier coefficients are computed in a single
batched angular FFT and then cached in a contracted ``(log k_>, r)`` table for
fast scalar and grid evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline

from .base import Bispectrum3D
from .multipole import BispectrumMultipole3D
from .support import Support3D
from fastnc.hankel.wrapper import PowerLawFFTLogConfig, power_law_fftlog_coefficients


def standard_linear_growth(z, cosmo: Mapping[str, float] | None = None):
    r"""Linear growth factor for a flat constant-``w`` background.

    The growing-mode solution is evaluated from

    .. math::
       D(a) \propto E(a)\int_0^a \frac{da'}{a'^3 E(a')^3},

    and normalized to ``D(z=0)=1``.  Radiation is intentionally neglected;
    this is the standard late-time growth prescription used for the notebook.
    """
    cosmo = {} if cosmo is None else dict(cosmo)
    Om0 = float(cosmo.get("Om0", 0.3))
    Ode0 = float(cosmo.get("Ode0", 1.0 - Om0))
    w0 = float(cosmo.get("w0", -1.0))
    if Om0 <= 0.0 or Ode0 < 0.0:
        raise ValueError("standard_linear_growth requires Om0>0 and Ode0>=0")

    z_arr = np.asarray(z, dtype=float)
    if np.any(z_arr < -0.999999):
        raise ValueError("z must satisfy z > -1")

    def E(a):
        return np.sqrt(Om0 * a ** -3.0 + Ode0 * a ** (-3.0 * (1.0 + w0)))

    def raw(a):
        value, _ = quad(lambda ap: 1.0 / (ap**3 * E(ap) ** 3), 0.0, float(a),
                        epsabs=1.0e-10, epsrel=1.0e-8, limit=200)
        return 2.5 * Om0 * E(float(a)) * value

    a_arr = 1.0 / (1.0 + z_arr)
    d0 = raw(1.0)
    out = np.array([raw(a) / d0 for a in np.ravel(a_arr)], dtype=float).reshape(a_arr.shape)
    return out.item() if np.isscalar(z) else out


def eisenstein_hu_no_wiggle_pklin(
    k,
    cosmo: Mapping[str, float] | None = None,
    *,
    amplitude: float = 1.0,
):
    """Eisenstein--Hu no-wiggle linear matter spectrum at ``z=0``.

    This is the common zero-baryon transfer-function form from
    Eisenstein & Hu (1998), multiplied by ``amplitude * k**ns``.  It is useful
    for validation and examples; scientific analyses should normally provide a
    CAMB/CLASS spectrum and its physical normalization.
    """
    cosmo = {} if cosmo is None else dict(cosmo)
    Om0 = float(cosmo.get("Om0", 0.3))
    h = float(cosmo.get("h", 0.7))
    ns = float(cosmo.get("ns", 0.965))
    theta = float(cosmo.get("Tcmb", 2.7255)) / 2.7
    k = np.asarray(k, dtype=float)
    if np.any(k <= 0.0):
        raise ValueError("k must be strictly positive")

    omhh = Om0 * h * h
    q = k * theta**2 / omhh
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    transfer = L0 / (L0 + C0 * q * q)
    return float(amplitude) * k**ns * transfer**2


def _validate_log_grid(k, pk):
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    if k.ndim != 1 or pk.ndim != 1 or k.shape != pk.shape:
        raise ValueError("k and pklin must be matching one-dimensional arrays")
    if k.size < 8 or np.any(k <= 0.0) or np.any(pk <= 0.0):
        raise ValueError("k and pklin must be positive; at least eight samples are required")
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("k must be strictly increasing")
    dln = np.diff(np.log(k))
    if not np.allclose(dln, dln[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("the FFTLog representation requires a logarithmically spaced k grid")
    return k, pk


def fourier_power_kernel(mode, nu, r, n_phi: int = 512):
    """Fourier coefficient of ``(1 + 2 r cos(phi) + r**2)**(nu/2)``.

    The kernel itself is universal.  It is evaluated by a periodic trapezoid
    rule (an FFT coefficient), not by evaluating a bispectrum at opening-angle
    samples.  For analytic functions at ``r<1`` this converges spectrally; the
    default resolution is deliberately conservative for the validation path.
    """
    mode = np.asarray(mode, dtype=int)
    r = np.asarray(r, dtype=float)
    if np.any(r < 0.0) or np.any(r > 1.0 + 1.0e-12):
        raise ValueError("r must lie in [0, 1]")
    r = np.minimum(r, np.nextafter(1.0, 0.0))
    phi = 2.0 * np.pi * np.arange(int(n_phi), dtype=float) / int(n_phi)
    # Broadcasting: r[...,None] is a batch of ratios, mode can be scalar or
    # match r.  Current callers use scalar mode, which keeps this compact.
    rr = r[..., None]
    shape = 1.0 + 2.0 * rr * np.cos(phi) + rr * rr
    values = np.exp(0.5 * complex(nu) * np.log(shape))
    phase = np.exp(-1j * np.asarray(mode)[..., None] * phi)
    return np.mean(values * phase, axis=-1)


_DEFAULT_KERNEL_TABLE_CONFIG = object()


class TreeLevelBispectrum3D(Bispectrum3D):
    """Matter tree-level bispectrum using a supplied ``P_L(k,z=0)``.

    ``P_L(k,z)=D(z)^2 P_L(k,0)`` and

    ``B_tree = 2 F2(1,2) P1 P2 + cyclic``.
    """

    def __init__(self, k, pklin, *, growth=standard_linear_growth,
                 cosmo: Mapping[str, float] | None = None,
                 support: Support3D | None = None,
                 support_policy: str = "ignore"):
        self.k, self.pklin = _validate_log_grid(k, pklin)
        self.cosmo = {} if cosmo is None else dict(cosmo)
        self.growth = growth
        self._logpk = InterpolatedUnivariateSpline(np.log(self.k), np.log(self.pklin), k=3, ext=0)
        self.support = support or Support3D(k_min=float(self.k[0]), k_max=float(self.k[-1]), policy=support_policy)

    def power0(self, k):
        k = np.asarray(k, dtype=float)
        if np.any(k <= 0.0):
            raise ValueError("k must be positive")
        lnk = np.log(k)
        val = self._logpk(lnk)
        lo = lnk < np.log(self.k[0])
        hi = lnk > np.log(self.k[-1])
        if np.any(lo):
            n = np.log(self.pklin[1] / self.pklin[0]) / np.log(self.k[1] / self.k[0])
            val = np.asarray(val)
            val[lo] = np.log(self.pklin[0]) + n * (lnk[lo] - np.log(self.k[0]))
        if np.any(hi):
            n = np.log(self.pklin[-1] / self.pklin[-2]) / np.log(self.k[-1] / self.k[-2])
            val = np.asarray(val)
            val[hi] = np.log(self.pklin[-1]) + n * (lnk[hi] - np.log(self.k[-1]))
        return np.exp(val)

    def growth_factor(self, z):
        try:
            return self.growth(z, self.cosmo)
        except TypeError:
            return self.growth(z)

    @staticmethod
    def f2(k1, k2, mu):
        return 5.0 / 7.0 + 0.5 * (k1 / k2 + k2 / k1) * mu + 2.0 / 7.0 * mu * mu

    def evaluate(self, k1, k2, k3, z, **params):
        if params:
            raise TypeError(f"Unexpected parameter(s): {', '.join(sorted(params))}")
        k1, k2, k3, z = np.broadcast_arrays(np.asarray(k1, float), np.asarray(k2, float), np.asarray(k3, float), np.asarray(z, float))
        if np.any(k1 <= 0.0) or np.any(k2 <= 0.0) or np.any(k3 <= 0.0):
            raise ValueError("triangle side lengths must be positive")
        mu12 = (k3*k3 - k1*k1 - k2*k2) / (2.0*k1*k2)
        mu23 = (k1*k1 - k2*k2 - k3*k3) / (2.0*k2*k3)
        mu31 = (k2*k2 - k3*k3 - k1*k1) / (2.0*k3*k1)
        p1, p2, p3 = self.power0(k1), self.power0(k2), self.power0(k3)
        d4 = np.asarray(self.growth_factor(z), dtype=float) ** 4
        return d4 * 2.0 * (self.f2(k1, k2, mu12)*p1*p2 + self.f2(k2, k3, mu23)*p2*p3 + self.f2(k3, k1, mu31)*p3*p1)

    def analytic_multipole(
        self,
        *,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        kernel_table_config=_DEFAULT_KERNEL_TABLE_CONFIG,
    ):
        """Construct Fourier multipoles with a universal geometry-kernel table by default.

        Pass ``kernel_table_config=None`` to force the batched direct FFTLog
        route, which is useful for validation or table-convergence tests.
        """
        if kernel_table_config is _DEFAULT_KERNEL_TABLE_CONFIG:
            kernel_table_config = FourierPowerKernelTableConfig()
        return TreeLevelFourierMultipole3D(
            self,
            fftlog_config=fftlog_config,
            kernel_table_config=kernel_table_config,
        )



@dataclass(frozen=True)
class FourierPowerKernelTableConfig:
    r"""Configuration for the universal Fourier-geometry table.

    The table stores only

    .. math::

        \mathcal K_m^{(\nu+s)}(r)
        = \frac{1}{2\pi}\int d\varphi\,
          (1+2r\cos\varphi+r^2)^{(\nu+s)/2}e^{-im\varphi},

    for the FFTLog exponents and shifts ``s=0,-2``.  It contains no linear
    power-spectrum coefficient and is therefore reusable across cosmologies
    whenever the FFTLog exponent grid and table configuration are unchanged.
    """

    n_r: int = 160
    n_phi: int = 256
    r_max: float = 1.0
    r_spacing_power: float = 4.0
    min_mode_max: int = 16
    interpolation: str = "cubic"

    def validate(self):
        if int(self.n_r) < 8:
            raise ValueError("n_r must be at least eight")
        if int(self.n_phi) < 16:
            raise ValueError("n_phi must be at least sixteen")
        if not (0.0 < float(self.r_max) <= 1.0):
            raise ValueError("r_max must lie in (0, 1]")
        if float(self.r_spacing_power) <= 0.0:
            raise ValueError("r_spacing_power must be positive")
        if int(self.min_mode_max) < 0:
            raise ValueError("min_mode_max must be non-negative")
        if self.interpolation not in {"linear", "cubic"}:
            raise ValueError("interpolation must be 'linear' or 'cubic'")


class _UniversalFourierGeometryKernelTable:
    """Interpolated universal table of Fourier geometry kernels."""

    def __init__(self, nu, mode_max, config):
        config.validate()
        self.nu = np.asarray(nu, complex)
        self.mode_max = int(mode_max)
        self.config = config
        if self.mode_max >= int(config.n_phi) // 2:
            raise ValueError("n_phi must exceed twice the maximum stored Fourier mode")
        t = np.linspace(0.0, 1.0, int(config.n_r))
        self.r = float(config.r_max) * (1.0 - (1.0 - t) ** float(config.r_spacing_power))
        self.kernel0, self.kernel_m2_scaled = self._build()

    def contains(self, r):
        r = np.asarray(r, float)
        return (r >= self.r[0]) & (r <= self.r[-1])

    def _build(self):
        nnu, nmode, nr = self.nu.size, self.mode_max + 1, self.r.size
        out0 = np.empty((nnu, nmode, nr), complex)
        outm2 = np.empty_like(out0)
        # Midpoint nodes avoid evaluating the integrable r=1 endpoint at
        # phi=pi.  The phase restores the Fourier convention at phi=2pi j/N.
        n_phi = int(self.config.n_phi)
        phi = 2.0*np.pi*(np.arange(n_phi) + 0.5)/float(n_phi)
        cphi = np.cos(phi)
        phase = np.exp(-1j * np.arange(nmode) * np.pi / float(n_phi))
        for ir, ratio in enumerate(self.r):
            arg = 1.0 + 2.0*ratio*cphi + ratio*ratio
            base = np.exp(0.5*self.nu[:, None]*np.log(arg)[None, :])
            out0[:, :, ir] = phase[None, :] * np.fft.fft(base, axis=1)[:, :nmode]/float(n_phi)
            # The isolated q^{-1} kernel is not regular at r=1.  Store the
            # cancellation-safe object (1-r^2)^2 K^{nu-2}; its r=1 value is
            # not used because the cyclic diagonal is imposed analytically.
            scale = (1.0-ratio*ratio)**2
            if scale == 0.0:
                outm2[:, :, ir] = 0.0
            else:
                outm2[:, :, ir] = scale * phase[None, :] * np.fft.fft(base/arg[None, :], axis=1)[:, :nmode]/float(n_phi)
        return out0, outm2

    @staticmethod
    def _linear(grid, x):
        ix = np.searchsorted(grid, x, side="right") - 1
        ix = np.clip(ix, 0, grid.size - 2)
        t = (x-grid[ix])/(grid[ix+1]-grid[ix])
        return ix, t

    @staticmethod
    def _cubic(grid, x):
        c = np.clip(np.searchsorted(grid, x, side="right")-1, 1, grid.size-3)
        ind = np.stack((c-1,c,c+1,c+2),axis=0)
        nodes=grid[ind]
        w=np.ones_like(nodes)
        for a in range(4):
            for b in range(4):
                if a != b: w[a] *= (x-nodes[b])/(nodes[a]-nodes[b])
        return ind,w

    def interpolate(self, shift, mode_max, r):
        if mode_max > self.mode_max: raise ValueError("requested mode exceeds table support")
        r=np.asarray(r,float); flat=r.ravel()
        if not np.all(self.contains(flat)): raise ValueError("r is outside geometry-table domain")
        values=self.kernel0 if shift == 0 else self.kernel_m2_scaled
        values=values[:, :mode_max+1]
        if self.config.interpolation == "linear":
            ix,t=self._linear(self.r,flat)
            out=(1-t)[None,None,:]*values[:,:,ix] + t[None,None,:]*values[:,:,ix+1]
        else:
            ix,w=self._cubic(self.r,flat)
            out=np.zeros((self.nu.size,mode_max+1,flat.size),complex)
            for a in range(4): out += w[a][None,None,:]*values[:,:,ix[a]]
        return out.reshape((self.nu.size,mode_max+1)+r.shape)


class TreeLevelFourierMultipole3D(BispectrumMultipole3D):
    """Semi-analytic full-Fourier multipoles of :class:`TreeLevelBispectrum3D`.

    The scalar hot path uses universal FFTLog geometry kernels.  The cyclic
    channels are combined analytically before the q^{-1}=k_>^2/k_3^2 kernel is
    interpolated, so the evaluation remains regular at and near ``k_1=k_2``.
    """
    basis = "fourier"

    def __init__(
        self,
        model: TreeLevelBispectrum3D,
        *,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        kernel_table_config: FourierPowerKernelTableConfig | None = None,
    ):
        self.model = model
        self.support = model.support
        self.fftlog_config = fftlog_config or PowerLawFFTLogConfig(
            bias=-1.5,
            c_window_width=0.25,
            N_pad=64,
        )
        self.coeff, self.nu = power_law_fftlog_coefficients(
            model.k,
            model.pklin,
            self.fftlog_config,
        )
        self.kernel_table_config = kernel_table_config
        self._geometry_table: _UniversalFourierGeometryKernelTable | None = None
        self._phi_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _positive_mode(mode):
        return abs(int(mode))

    @staticmethod
    def _cos_stencil(values, modes):
        modes = np.asarray(modes, dtype=int)
        return 0.5 * (values[np.abs(modes - 1)] + values[np.abs(modes + 1)])

    @staticmethod
    def _cos2_stencil(values, modes):
        modes = np.asarray(modes, dtype=int)
        return (
            0.5 * values[np.abs(modes)]
            + 0.25 * (values[np.abs(modes - 2)] + values[np.abs(modes + 2)])
        )

    def _phi_grid(self, n_phi):
        n_phi = int(n_phi)
        cached = self._phi_cache.get(n_phi)
        if cached is None:
            phi = 2.0 * np.pi * np.arange(n_phi, dtype=float) / float(n_phi)
            cached = phi, np.cos(phi)
            self._phi_cache[n_phi] = cached
        return cached

    _UNIVERSAL_TABLE_CACHE: dict[tuple, _UniversalFourierGeometryKernelTable] = {}

    def _geometry_cache_key(self, mode_max):
        cfg = self.kernel_table_config
        return (
            tuple(np.round(self.nu.real, 14)), tuple(np.round(self.nu.imag, 14)),
            int(mode_max), int(cfg.n_r), int(cfg.n_phi), float(cfg.r_max),
            float(cfg.r_spacing_power), cfg.interpolation,
        )

    def _build_or_extend_table(self, mode_max):
        cfg = self.kernel_table_config
        if cfg is None:
            return None
        target_mode = max(int(mode_max), int(cfg.min_mode_max))
        if self._geometry_table is not None and self._geometry_table.mode_max >= target_mode:
            return self._geometry_table
        key = self._geometry_cache_key(target_mode)
        table = self._UNIVERSAL_TABLE_CACHE.get(key)
        if table is None:
            table = _UniversalFourierGeometryKernelTable(self.nu, target_mode, cfg)
            self._UNIVERSAL_TABLE_CACHE[key] = table
        self._geometry_table = table
        return table

    def build_kernel_table(self, mode_max=None):
        """Build the universal Fourier-geometry table explicitly.

        This is optional because the table is otherwise created lazily at the
        first multipole evaluation.  The returned object is internal; the
        public operation is still :meth:`evaluate`.
        """
        if self.kernel_table_config is None:
            raise RuntimeError("kernel_table_config=None disables the kernel table")
        if mode_max is None:
            mode_max = self.kernel_table_config.min_mode_max
        return self._build_or_extend_table(int(mode_max))

    def _p3_stencil_direct(self, mode_max, k_hi, r, *, n_phi=256):
        """Batched direct FFTLog contraction for modes ``0,...,mode_max``.

        One pair of angular FFTs yields every mode required by the cyclic
        tree-level stencil.  This replaces the former repeated scalar
        ``_p3_kernel`` calls even when the interpolation table is disabled.
        """
        mode_max = int(mode_max)
        k_hi, r = np.broadcast_arrays(np.asarray(k_hi, float), np.asarray(r, float))
        flat_k = k_hi.ravel()
        flat_r = np.minimum(r.ravel(), 1.0 - 1.0e-8)
        if np.any(flat_k <= 0.0) or np.any(flat_r < 0.0):
            raise ValueError("k_hi must be positive and r must be non-negative")

        _, cos_phi = self._phi_grid(n_phi)
        out0 = np.empty((mode_max + 1, flat_k.size), dtype=complex)
        out_m2 = np.empty_like(out0)

        for i, (kh, ratio) in enumerate(zip(flat_k, flat_r)):
            argument = 1.0 + 2.0 * ratio * cos_phi + ratio * ratio
            base = np.exp(0.5 * self.nu[:, None] * np.log(argument)[None, :])
            kernel0 = np.fft.fft(base, axis=1)[:, : mode_max + 1] / float(n_phi)
            kernel_m2 = np.fft.fft(base / argument[None, :], axis=1)[:, : mode_max + 1] / float(n_phi)

            amp0 = self.coeff * kh ** self.nu
            amp_m2 = amp0 / (kh * kh)
            out0[:, i] = amp0 @ kernel0
            out_m2[:, i] = (1.0 - ratio * ratio) ** 2 * (amp_m2 @ kernel_m2)

        shape = (mode_max + 1,) + k_hi.shape
        return out0.reshape(shape), out_m2.reshape(shape)

    def _p3_stencil(self, mode_max, k_hi, r):
        """Contract a universal geometry table with this cosmology's FFTLog coefficients."""
        mode_max = int(mode_max)
        k_hi, r = np.broadcast_arrays(np.asarray(k_hi, float), np.asarray(r, float))
        table = self._build_or_extend_table(mode_max)
        if table is None:
            return self._p3_stencil_direct(mode_max, k_hi, r)
        valid = table.contains(r)
        result0 = np.empty((mode_max + 1,) + k_hi.shape, dtype=complex)
        result_m2 = np.empty_like(result0)
        flat_valid, flat_k, flat_r = valid.ravel(), k_hi.ravel(), r.ravel()
        if np.any(flat_valid):
            kk = flat_k[flat_valid]
            geom0 = table.interpolate(0, mode_max, flat_r[flat_valid])
            geomm2 = table.interpolate(-2, mode_max, flat_r[flat_valid])
            amp0 = self.coeff[:, None] * np.exp(self.nu[:, None] * np.log(kk)[None, :])
            ampm2 = amp0 / (kk[None, :]**2)
            # ``geomm2`` is the cancellation-safe scaled q^{-1} kernel.
            p0 = np.sum(amp0[:, None, :] * geom0, axis=0)
            pm2 = np.sum(ampm2[:, None, :] * geomm2, axis=0)
            result0.reshape(mode_max+1,-1)[:, flat_valid] = p0
            result_m2.reshape(mode_max+1,-1)[:, flat_valid] = pm2
        if np.any(~flat_valid):
            p0, pm2 = self._p3_stencil_direct(mode_max, flat_k[~flat_valid], flat_r[~flat_valid])
            result0.reshape(mode_max+1,-1)[:, ~flat_valid] = p0.reshape(mode_max+1,-1)
            result_m2.reshape(mode_max+1,-1)[:, ~flat_valid] = pm2.reshape(mode_max+1,-1)
        return result0, result_m2

    def _p3_kernel(self, mode, k_hi, r, shift=0):
        """Return one table-accelerated Fourier coefficient of ``P_L(k_3)k_3^s``.

        This compatibility method is intentionally thin: the actual
        implementation always obtains a batched stencil, then selects the
        requested mode.  ``shift`` is currently restricted to the two shifts
        required by the tree-level cyclic kernels, ``0`` and ``-2``.
        """
        mode = self._positive_mode(mode)
        p0, pm2 = self._p3_stencil(mode, k_hi, r)
        if shift == 0:
            return p0[mode]
        if shift == -2:
            # Compatibility API returns the unscaled coefficient.  The
            # regular cyclic path deliberately uses the scaled stencil.
            scale = (1.0 - np.asarray(r, float)**2)**2
            return pm2[mode] / scale
        raise ValueError("tree-level _p3_kernel supports only shift=0 or shift=-2")

    @staticmethod
    def _q_times_stencil(values, r, modes):
        """Fourier modes of q P(k3), q=1+r^2+2r cos(phi)."""
        modes = np.asarray(modes, dtype=int)
        rr = np.asarray(r, float)[None, ...]
        return (
            (1.0 + rr * rr) * values[np.abs(modes)]
            + rr * (values[np.abs(modes - 1)] + values[np.abs(modes + 1)])
        )

    def _regular_cyclic_sum(self, modes, k1, k2):
        """Return the analytically regular sum of the (23) and (31) channels.

        The q^{-1}=k_>^2/k_3^2 pieces are combined before interpolation:
        their coefficient is O((1-r^2)^2).  The geometry table stores the
        correspondingly scaled kernel, avoiding cancellation between large
        separately interpolated cyclic terms near k1=k2.
        """
        modes = np.asarray(modes, dtype=int)
        max_mode = int(np.max(np.abs(modes))) + 1
        k1v = np.asarray(k1, float)
        k2v = np.asarray(k2, float)
        hi = np.maximum(k1v, k2v)
        lo = np.minimum(k1v, k2v)
        r = lo / hi
        # The analytic cancellation formula uses dimensionless side lengths
        # x=k1/K and y=k2/K with K=max(k1,k2).
        x = k1v / hi
        y = k2v / hi
        p0, pm2_scaled = self._p3_stencil(max_mode, hi, r)
        p = p0[np.abs(modes)]
        qp = self._q_times_stencil(p0, r, modes)

        p1 = np.asarray(self.model.power0(k1v))[None, ...]
        p2 = np.asarray(self.model.power0(k2v))[None, ...]
        x2 = x[None, ...] ** 2
        y2 = y[None, ...] ** 2
        delta = x2 - y2

        # Non-singular terms from F23+F31 after exact side-length algebra.
        b0 = (13.0 / 28.0) * (p1 + p2)
        b0 += (3.0 / 28.0) * delta * (p2 / y2 - p1 / x2)
        bq = (-5.0 / 28.0) * (p2 / y2 + p1 / x2)

        # Stable form of A = P2 C23 + P1 C31.  Csum is O(delta^2),
        # Cdiff is O(delta), so A/(1-r^2)^2 remains finite as r->1.
        csum = delta * delta * (x2 + y2) / (14.0 * x2 * y2)
        cdiff = delta * (x2 * x2 + 5.0 * x2 * y2 + y2 * y2) / (14.0 * x2 * y2)
        acoef = 0.5 * (p1 + p2) * csum + 0.5 * (p2 - p1) * cdiff
        scale = (1.0 - r[None, ...] ** 2) ** 2

        regular = 2.0 * (b0 * p + bq * qp)
        out = regular.astype(complex, copy=False)
        diagonal = np.isclose(r, 1.0, rtol=0.0, atol=8.0*np.finfo(float).eps)
        if np.any(~diagonal):
            # pm2_scaled = (1-r^2)^2 [P(k3)/k3^2]_m.
            term = 2.0 * (hi[None, ...] ** 2) * (acoef / scale) * pm2_scaled[np.abs(modes)]
            out = out + np.where(diagonal[None, ...], 0.0, term)

        if np.any(diagonal):
            # Exact r=1 identity:
            # B23+B31 = P(k1)[13/7 P(k3) - 5/7 q P(k3)].
            diag_value = p1 * ((13.0 / 7.0) * p - (5.0 / 7.0) * qp)
            out = np.where(diagonal[None, ...], diag_value, out)
        return out

    def _direct_12(self, modes, k1, k2):
        modes = np.asarray(modes, dtype=int)
        p12 = np.asarray(self.model.power0(k1) * self.model.power0(k2))[None, ...]
        a0 = 5.0 / 7.0 + 1.0 / 7.0
        a1 = 0.5 * (np.asarray(k1) / np.asarray(k2) + np.asarray(k2) / np.asarray(k1))
        a2 = 1.0 / 14.0
        m = modes.reshape((-1,) + (1,) * np.asarray(k1).ndim)
        return 2.0 * p12 * (
            (m == 0) * a0
            + (np.abs(m) == 1) * a1[None, ...] / 2.0
            + (np.abs(m) == 2) * a2
        )

    def _direct_angle_modes(self, modes, k1, k2, z, *, n_phi=None):
        """Numerical angular fallback for the near-diagonal singular split.

        The public semi-analytic path does not sample opening angles.  This
        method is used only for ``r`` above the table boundary, where separate
        ``P_L(k_3)/k_3^2`` terms lose regularity while their sum remains finite.
        Sampling midpoint angles avoids the exactly degenerate ``k_3=0`` node.
        """
        modes = np.asarray(modes, dtype=int)
        k1, k2, z = np.broadcast_arrays(np.asarray(k1, float), np.asarray(k2, float), np.asarray(z, float))
        max_mode = int(np.max(np.abs(modes)))
        if n_phi is None:
            n_phi = max(512, 8 * (max_mode + 1))
        n_phi = 1 << int(np.ceil(np.log2(int(n_phi))))

        j = np.arange(n_phi, dtype=float)
        phi = 2.0 * np.pi * (j + 0.5) / float(n_phi)
        cos_phi = np.cos(phi)
        phase = np.exp(-1j * modes * np.pi / float(n_phi))
        out = np.empty((modes.size, k1.size), dtype=complex)

        for i, (a, b, zi) in enumerate(zip(k1.ravel(), k2.ravel(), z.ravel())):
            k3 = np.sqrt(a * a + b * b + 2.0 * a * b * cos_phi)
            values = self.model.evaluate(
                np.full_like(k3, a),
                np.full_like(k3, b),
                k3,
                zi,
            )
            fft = np.fft.fft(values) / float(n_phi)
            out[:, i] = phase * fft[np.mod(modes, n_phi)]
        return np.real_if_close(out.reshape((modes.size,) + k1.shape), tol=500)

    def evaluate(self, mode, k1, k2, z, **params):
        if params:
            raise TypeError(f"Unexpected parameter(s): {', '.join(sorted(params))}")
        scalar = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, dtype=int))
        k1, k2, z = np.broadcast_arrays(np.asarray(k1, float), np.asarray(k2, float), np.asarray(z, float))
        if np.any(k1 <= 0.0) or np.any(k2 <= 0.0):
            raise ValueError("k1 and k2 must be positive")

        direct = self._direct_12(modes, k1, k2)
        cyclic = self._regular_cyclic_sum(modes, k1, k2)
        d4 = np.asarray(self.model.growth_factor(z), float) ** 4
        result = d4[None, ...] * (direct + cyclic)

        result = np.real_if_close(result, tol=500)
        return result[0] if scalar else result



class FactorizedBispectrum3D(Bispectrum3D):
    r"""A bispectrum of the fully factorized form

    .. math::

       B(k_1,k_2,k_3;z)=U(k_1;z)U(k_2;z)U(k_3;z).

    Parameters
    ----------
    k
        Logarithmically spaced positive grid on which the radial factor is
        decomposed by FFTLog.
    radial_factor
        Callable ``radial_factor(k, z)`` returning :math:`U(k;z)`.  It must be
        positive on ``k`` for the complex-power FFTLog representation used by
        :meth:`analytic_multipole`.

    Notes
    -----
    The class is intended as the generic semi-analytic building block for
    one-halo-like terms.  In particular, the BiHalofit one-halo contribution
    is obtained by supplying its scalar one-halo radial factor as
    ``radial_factor``.
    """

    def __init__(
        self,
        k,
        radial_factor,
        *,
        support: Support3D | None = None,
        support_policy: str = "ignore",
        allow_signed: bool = False,
    ):
        k = np.asarray(k, dtype=float)
        if k.ndim != 1 or k.size < 8 or np.any(k <= 0.0):
            raise ValueError("k must be a positive one-dimensional grid with at least eight samples")
        if np.any(np.diff(k) <= 0.0):
            raise ValueError("k must be strictly increasing")
        dln = np.diff(np.log(k))
        if not np.allclose(dln, dln[0], rtol=1.0e-7, atol=1.0e-12):
            raise ValueError("the FFTLog representation requires a logarithmically spaced k grid")
        if not callable(radial_factor):
            raise TypeError("radial_factor must be callable as radial_factor(k, z)")
        self.k = k
        self.radial_factor = radial_factor
        self.allow_signed = bool(allow_signed)
        self.support = support or Support3D(
            k_min=float(k[0]), k_max=float(k[-1]), policy=support_policy
        )

    def factor(self, k, z):
        value = np.asarray(self.radial_factor(np.asarray(k, dtype=float), z), dtype=float)
        if np.any(~np.isfinite(value)):
            raise ValueError("radial_factor must return finite values")
        if self.allow_signed:
            return value
        if np.any(value <= 0.0):
            raise ValueError("radial_factor must return finite positive values")
        return value

    def evaluate(self, k1, k2, k3, z, **params):
        if params:
            raise TypeError(f"Unexpected parameter(s): {', '.join(sorted(params))}")
        k1, k2, k3, z = np.broadcast_arrays(
            np.asarray(k1, float), np.asarray(k2, float),
            np.asarray(k3, float), np.asarray(z, float),
        )
        if np.any(k1 <= 0.0) or np.any(k2 <= 0.0) or np.any(k3 <= 0.0):
            raise ValueError("triangle side lengths must be positive")
        return self.factor(k1, z) * self.factor(k2, z) * self.factor(k3, z)

    def analytic_multipole(
        self,
        *,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        kernel_table_config=_DEFAULT_KERNEL_TABLE_CONFIG,
    ):
        """Construct semi-analytic full-Fourier multipoles.

        Only the universal geometry table is retained across calls.  FFTLog
        coefficients are rebuilt from ``radial_factor(k, z)`` for each
        requested redshift, so no cosmology-dependent cache is introduced.
        """
        if kernel_table_config is _DEFAULT_KERNEL_TABLE_CONFIG:
            kernel_table_config = FourierPowerKernelTableConfig()
        return FactorizedFourierMultipole3D(
            self,
            fftlog_config=fftlog_config,
            kernel_table_config=kernel_table_config,
        )


class FactorizedFourierMultipole3D(BispectrumMultipole3D):
    r"""Semi-analytic Fourier multipoles of :class:`FactorizedBispectrum3D`.

    With :math:`U(k;z)\simeq\sum_n c_n(z)k^{\nu_n}`, this evaluator uses

    .. math::

       B_m(k_1,k_2;z)=U(k_1;z)U(k_2;z)
       \sum_n c_n(z)k_>^{\nu_n}\mathcal K_m^{(\nu_n)}(k_</k_>).

    The geometry table contains only :math:`\mathcal K`; it is independent of
    the radial factor, cosmology, and redshift.
    """

    basis = "fourier"
    _UNIVERSAL_TABLE_CACHE: dict[tuple, _UniversalFourierGeometryKernelTable] = {}

    def __init__(
        self,
        model: FactorizedBispectrum3D,
        *,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        kernel_table_config: FourierPowerKernelTableConfig | None = None,
    ):
        self.model = model
        self.support = model.support
        self.fftlog_config = fftlog_config or PowerLawFFTLogConfig(
            bias=-1.5,
            c_window_width=0.25,
            N_pad=64,
        )
        # The exponent grid follows the FFTLog configuration and k grid, not
        # the values of U.  A unit spectrum obtains it without introducing a
        # cosmology-dependent stored coefficient.
        _, self.nu = power_law_fftlog_coefficients(
            model.k, np.ones_like(model.k), self.fftlog_config
        )
        self.kernel_table_config = kernel_table_config
        self._geometry_table: _UniversalFourierGeometryKernelTable | None = None

    def _geometry_cache_key(self, mode_max):
        cfg = self.kernel_table_config
        return (
            tuple(np.round(self.nu.real, 14)), tuple(np.round(self.nu.imag, 14)),
            int(mode_max), int(cfg.n_r), int(cfg.n_phi), float(cfg.r_max),
            float(cfg.r_spacing_power), cfg.interpolation,
        )

    def _build_or_extend_table(self, mode_max):
        cfg = self.kernel_table_config
        if cfg is None:
            return None
        target_mode = max(int(mode_max), int(cfg.min_mode_max))
        if self._geometry_table is not None and self._geometry_table.mode_max >= target_mode:
            return self._geometry_table
        key = self._geometry_cache_key(target_mode)
        table = self._UNIVERSAL_TABLE_CACHE.get(key)
        if table is None:
            table = _UniversalFourierGeometryKernelTable(self.nu, target_mode, cfg)
            self._UNIVERSAL_TABLE_CACHE[key] = table
        self._geometry_table = table
        return table

    def build_kernel_table(self, mode_max=None):
        if self.kernel_table_config is None:
            raise RuntimeError("kernel_table_config=None disables the kernel table")
        if mode_max is None:
            mode_max = self.kernel_table_config.min_mode_max
        return self._build_or_extend_table(int(mode_max))

    def _coefficients(self, z):
        values = self.model.factor(self.model.k, z)
        coeff, nu = power_law_fftlog_coefficients(self.model.k, values, self.fftlog_config)
        if not np.allclose(nu, self.nu, rtol=0.0, atol=1.0e-13):
            raise RuntimeError("FFTLog exponent grid unexpectedly changed")
        return coeff

    def _u3_stencil_direct(self, coeff, mode_max, k_hi, r, *, n_phi=256):
        k_hi, r = np.broadcast_arrays(np.asarray(k_hi, float), np.asarray(r, float))
        flat_k, flat_r = k_hi.ravel(), r.ravel()
        n_phi = int(n_phi)
        phi = 2.0*np.pi*(np.arange(n_phi, dtype=float)+0.5)/float(n_phi)
        cphi = np.cos(phi)
        phase = np.exp(-1j*np.arange(int(mode_max)+1)*np.pi/float(n_phi))
        out = np.empty((int(mode_max)+1, flat_k.size), complex)
        for i, (kh, ratio) in enumerate(zip(flat_k, flat_r)):
            arg = 1.0 + 2.0*ratio*cphi + ratio*ratio
            base = np.exp(0.5*self.nu[:, None]*np.log(arg)[None, :])
            geom = phase[None, :] * np.fft.fft(base, axis=1)[:, :int(mode_max)+1]/float(n_phi)
            out[:, i] = (coeff * kh**self.nu) @ geom
        return out.reshape((int(mode_max)+1,) + k_hi.shape)

    def _u3_stencil(self, coeff, mode_max, k_hi, r):
        k_hi, r = np.broadcast_arrays(np.asarray(k_hi, float), np.asarray(r, float))
        table = self._build_or_extend_table(int(mode_max))
        if table is None:
            return self._u3_stencil_direct(coeff, mode_max, k_hi, r)
        valid = table.contains(r)
        out = np.empty((int(mode_max)+1,) + k_hi.shape, complex)
        flat_out = out.reshape(int(mode_max)+1, -1)
        flat_k, flat_r, flat_valid = k_hi.ravel(), r.ravel(), valid.ravel()
        if np.any(flat_valid):
            kk = flat_k[flat_valid]
            geom = table.interpolate(0, int(mode_max), flat_r[flat_valid])
            amp = coeff[:, None] * np.exp(self.nu[:, None]*np.log(kk)[None, :])
            flat_out[:, flat_valid] = np.sum(amp[:, None, :]*geom, axis=0)
        if np.any(~flat_valid):
            flat_out[:, ~flat_valid] = self._u3_stencil_direct(
                coeff, mode_max, flat_k[~flat_valid], flat_r[~flat_valid]
            ).reshape(int(mode_max)+1, -1)
        return out

    def _evaluate_one_redshift(self, modes, k1, k2, z):
        hi = np.maximum(k1, k2)
        r = np.minimum(k1, k2)/hi
        max_mode = int(np.max(np.abs(modes)))
        coeff = self._coefficients(float(z))
        u3 = self._u3_stencil(coeff, max_mode, hi, r)[np.abs(modes)]
        u1 = self.model.factor(k1, float(z))[None, ...]
        u2 = self.model.factor(k2, float(z))[None, ...]
        return u1*u2*u3

    def evaluate(self, mode, k1, k2, z, **params):
        if params:
            raise TypeError(f"Unexpected parameter(s): {', '.join(sorted(params))}")
        scalar = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, int))
        k1, k2, z = np.broadcast_arrays(np.asarray(k1, float), np.asarray(k2, float), np.asarray(z, float))
        if np.any(k1 <= 0.0) or np.any(k2 <= 0.0):
            raise ValueError("k1 and k2 must be positive")
        result = np.empty((modes.size,) + k1.shape, complex)
        # Coefficients are cosmology/redshift dependent and intentionally not
        # cached.  Grouping equal z values avoids redundant FFTLogs within one
        # vectorized call without retaining proposal-specific state.
        flat_z = z.ravel()
        for zi in np.unique(flat_z):
            mask = (z == zi)
            values = self._evaluate_one_redshift(modes, k1[mask], k2[mask], float(zi))
            result.reshape(modes.size, -1)[:, mask.ravel()] = values.reshape(modes.size, -1)
        result = np.real_if_close(result, tol=500)
        return result[0] if scalar else result
