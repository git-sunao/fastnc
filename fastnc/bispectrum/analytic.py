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

# -----------------------------------------------------------------------------
# Composable semi-analytic multipoles
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PowerLawAngularKernelTableConfig:
    """Numerical table configuration for the kernel in Eq. (app_kernel_def).

    The geometry convention is fixed to
    ``r=min(k1,k2)/sqrt(k1**2+k2**2)`` and
    ``s**2=1+2*r*sqrt(1-r**2)*cos(phi)``.
    """
    n_r: int = 256
    n_phi: int = 512
    r_max: float = 1.0 / np.sqrt(2.0)


class PowerLawAngularKernelTable:
    """Shared table of :math:`K_L^{(nu+p)}(r)` for one FFTLog exponent grid.

    Tables are constructed lazily for requested ``(mode, shift)`` pairs and
    retained in memory.  Several physical terms can therefore share one table
    whenever they use the same FFTLog component.
    """

    convention = "quadrature_ratio_v1"

    def __init__(self, nu, config: PowerLawAngularKernelTableConfig | None = None):
        self.nu = np.asarray(nu, dtype=complex)
        self.config = config or PowerLawAngularKernelTableConfig()
        if self.nu.ndim != 1:
            raise ValueError("nu must be one-dimensional")
        self.r_grid = np.linspace(0.0, np.nextafter(float(self.config.r_max), 0.0), int(self.config.n_r))
        self._tables: dict[tuple[int, int], np.ndarray] = {}

    def _build(self, mode: int, shift: int):
        key = (int(mode), int(shift))
        if key in self._tables:
            return self._tables[key]
        phi = 2.0 * np.pi * np.arange(int(self.config.n_phi), dtype=float) / int(self.config.n_phi)
        r = self.r_grid[:, None]
        s2 = 1.0 + 2.0 * r * np.sqrt(np.maximum(0.0, 1.0 - r * r)) * np.cos(phi)
        s2 = np.maximum(s2, np.finfo(float).tiny)
        # shape: (n_nu, n_r, n_phi)
        values = np.exp(0.5 * (self.nu[:, None, None] + int(shift)) * np.log(s2)[None, :, :])
        phase = np.exp(-1j * int(mode) * phi)[None, None, :]
        table = np.mean(values * phase, axis=-1)
        self._tables[key] = table
        return table

    def _interpolate(self, table, r):
        """Vectorized linear interpolation along the tabulated ``r`` axis."""
        r = np.asarray(r, dtype=float)
        flat = np.clip(r.ravel(), self.r_grid[0], self.r_grid[-1])
        upper = np.searchsorted(self.r_grid, flat, side="right")
        lower = np.clip(upper - 1, 0, self.r_grid.size - 2)
        upper = lower + 1
        r0 = self.r_grid[lower]
        r1 = self.r_grid[upper]
        weight = (flat - r0) / (r1 - r0)
        out = (
            table[:, lower] * (1.0 - weight)[None, :]
            + table[:, upper] * weight[None, :]
        )
        return out.reshape((self.nu.size,) + r.shape)

    def evaluate(self, mode, r, *, shift: int = 0):
        """Return ``K_mode^(nu+shift)(r)`` with leading FFTLog-index axis."""
        r = np.asarray(r, dtype=float)
        if np.any(r < 0.0) or np.any(r > self.config.r_max + 1.0e-12):
            raise ValueError("r is outside the table domain")
        return self._interpolate(self._build(int(mode), int(shift)), r)

    def evaluate_modes(
        self,
        modes,
        r,
        *,
        shift: int = 0,
        unique: bool = False,
        unique_r=None,
        inverse=None,
    ):
        """Return a stack with shape ``(n_mode, n_nu) + r.shape``.

        Parameters
        ----------
        unique
            When ``True``, interpolate the angular kernels only at the unique
            values of ``r`` and restore the original layout afterwards.
        unique_r, inverse
            Optional precomputed ``np.unique(r.ravel(), return_inverse=True)``
            result.  Supplying both avoids recomputing the ratio grouping when
            several component/shift groups share the same geometry.
        """
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        r = np.asarray(r, dtype=float)
        if not unique or r.ndim == 0:
            return np.stack(
                [self.evaluate(mode, r, shift=shift) for mode in modes],
                axis=0,
            )

        if (unique_r is None) != (inverse is None):
            raise ValueError("unique_r and inverse must be supplied together")
        if unique_r is None:
            unique_r, inverse = np.unique(r.ravel(), return_inverse=True)
        else:
            unique_r = np.asarray(unique_r, dtype=float)
            inverse = np.asarray(inverse, dtype=int)
            if inverse.shape != (r.size,):
                raise ValueError("inverse must have shape (r.size,)")

        values_unique = np.stack(
            [self.evaluate(mode, unique_r, shift=shift) for mode in modes],
            axis=0,
        )
        values = values_unique[:, :, inverse]
        return values.reshape((modes.size, self.nu.size) + r.shape)



@dataclass(frozen=True)
class TensorProductGeometryCache:
    r"""Geometry shared by evaluations on a tensor-product ``(k1, k2)`` grid.

    The cache is independent of redshift and of the physical bispectrum model.
    It stores the full geometric quantities that cannot generally be compressed,
    together with an exact compression of the ratio

    .. math::
       r = \min(k_1,k_2)/\sqrt{k_1^2+k_2^2}

    when the two axes are the same logarithmically spaced grid.  In that case
    ``r`` depends only on the index separation ``|i-j|`` and the angular
    kernels need be evaluated at only ``n_k`` ratios rather than at every
    pixel of the two-dimensional grid.

    Instances are intended to be retained by callers and passed to
    :meth:`CompositeSemiAnalyticBispectrumMultipole3D.evaluate_modes_grid`
    for repeated redshift or mode evaluations.
    """

    k1_axis: np.ndarray
    k2_axis: np.ndarray
    k: np.ndarray
    x1: np.ndarray
    x2: np.ndarray
    r: np.ndarray
    r_unique: np.ndarray | None
    r_inverse: np.ndarray | None
    r_groups: tuple[np.ndarray, ...] | None
    has_exact_ratio_groups: bool

    @classmethod
    def from_axes(cls, k1_axis, k2_axis):
        k1_axis = np.asarray(k1_axis, dtype=float)
        k2_axis = np.asarray(k2_axis, dtype=float)
        if k1_axis.ndim != 1 or k2_axis.ndim != 1:
            raise ValueError("k1_axis and k2_axis must be one-dimensional")
        if k1_axis.size == 0 or k2_axis.size == 0:
            raise ValueError("k1_axis and k2_axis must be non-empty")
        if np.any(k1_axis <= 0.0) or np.any(k2_axis <= 0.0):
            raise ValueError("tensor-product axes must be strictly positive")
        if np.any(np.diff(k1_axis) <= 0.0) or np.any(np.diff(k2_axis) <= 0.0):
            raise ValueError("tensor-product axes must be strictly increasing")

        k1 = k1_axis[:, None]
        k2 = k2_axis[None, :]
        k = np.hypot(k1, k2)
        x1 = k1 / k
        x2 = k2 / k

        same_axis = (
            k1_axis.shape == k2_axis.shape
            and np.array_equal(k1_axis, k2_axis)
        )
        is_log_uniform = False
        dln = None
        if same_axis and k1_axis.size > 1:
            log_axis = np.log(k1_axis)
            dln_values = np.diff(log_axis)
            is_log_uniform = np.allclose(
                dln_values,
                dln_values[0],
                rtol=1.0e-10,
                atol=1.0e-13,
            )
            if is_log_uniform:
                dln = float(dln_values[0])

        if same_axis and is_log_uniform:
            # For k_i=k_0 exp(i Delta),
            # r_ij=[1+exp(2 |i-j| Delta)]^{-1/2}.  This construction is
            # exact at the level of the grid definition, avoiding fragile
            # floating-point equality grouping of the full r array.
            n_axis = k1_axis.size
            separation = np.abs(
                np.arange(n_axis)[:, None] - np.arange(n_axis)[None, :]
            )
            r_unique = 1.0 / np.sqrt(
                1.0 + np.exp(2.0 * dln * np.arange(n_axis, dtype=float))
            )
            r = r_unique[separation]
            r_inverse = separation.ravel()
            r_groups = tuple(
                np.flatnonzero(r_inverse == index)
                for index in range(r_unique.size)
            )
            has_exact_ratio_groups = True
        else:
            r = np.minimum(k1, k2) / k
            r_unique = None
            r_inverse = None
            r_groups = None
            has_exact_ratio_groups = False

        return cls(
            k1_axis=k1_axis,
            k2_axis=k2_axis,
            k=k,
            x1=x1,
            x2=x2,
            r=r,
            r_unique=r_unique,
            r_inverse=r_inverse,
            r_groups=r_groups,
            has_exact_ratio_groups=has_exact_ratio_groups,
        )

    @property
    def shape(self):
        return self.k.shape



@dataclass(frozen=True)
class FFTLogComponent:
    """A named reusable FFTLog target :math:`W(k;z)`.

    ``name`` is descriptive only.  Cache sharing is keyed by object identity,
    the FFTLog configuration, and the supplied logarithmic grid.
    """
    name: str
    k_grid: np.ndarray
    evaluator: object
    fftlog_config: PowerLawFFTLogConfig = PowerLawFFTLogConfig()

    def __post_init__(self):
        k = np.asarray(self.k_grid, dtype=float)
        if k.ndim != 1 or k.size < 8 or np.any(k <= 0.0):
            raise ValueError("FFTLogComponent.k_grid must be a positive one-dimensional grid")
        if np.any(np.diff(k) <= 0.0):
            raise ValueError("FFTLogComponent.k_grid must be increasing")
        dln = np.diff(np.log(k))
        if not np.allclose(dln, dln[0], rtol=1.0e-7, atol=1.0e-12):
            raise ValueError("FFTLogComponent.k_grid must be logarithmically spaced")
        object.__setattr__(self, "k_grid", k)


class FFTLogCoefficientCache:
    """In-memory cache of FFTLog coefficients ``w_n(z)`` by component.

    Coefficients are stored independently for every scalar redshift.  The
    :meth:`get_many` and :meth:`warm` helpers provide an explicit batch API
    for line-of-sight grids while retaining :meth:`get` as the scalar fast
    path used by ordinary multipole evaluation.
    """

    def __init__(self):
        self._coefficients: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _scalar_redshift(z):
        value = np.asarray(z, dtype=float)
        if value.ndim != 0:
            raise ValueError(
                "FFTLogCoefficientCache.get requires a scalar redshift; "
                "use get_many or warm for a redshift array"
            )
        return float(value)

    def get(self, component: FFTLogComponent, z):
        z_value = self._scalar_redshift(z)
        key = (id(component), z_value)
        cached = self._coefficients.get(key)
        if cached is None:
            values = np.asarray(
                component.evaluator(component.k_grid, z_value),
                dtype=float,
            )
            try:
                values = np.broadcast_to(values, component.k_grid.shape)
            except ValueError as error:
                raise ValueError(
                    "FFTLog component evaluator must return values "
                    "broadcastable to component.k_grid"
                ) from error
            coeff, nu = power_law_fftlog_coefficients(
                component.k_grid,
                values,
                component.fftlog_config,
            )
            cached = (
                np.asarray(coeff, dtype=complex),
                np.asarray(nu, dtype=complex),
            )
            self._coefficients[key] = cached
        return cached

    def get_many(self, component: FFTLogComponent, z_values):
        """Return cached coefficients on a redshift grid.

        Parameters
        ----------
        component
            FFTLog target whose coefficients are requested.
        z_values
            Scalar or array-like redshifts.  The returned leading dimensions
            follow ``np.asarray(z_values).shape``.

        Returns
        -------
        coefficients, nu
            ``coefficients`` has shape ``z_values.shape + (n_nu,)`` and
            contains ``w_n(z)``.  ``nu`` is the common one-dimensional FFTLog
            exponent grid.
        """
        z_array = np.asarray(z_values, dtype=float)
        flat_z = z_array.reshape(-1)
        if flat_z.size == 0:
            raise ValueError("z_values must contain at least one redshift")

        coefficients = []
        nu_reference = None
        for z_value in flat_z:
            coeff, nu = self.get(component, float(z_value))
            if nu_reference is None:
                nu_reference = nu
            elif not np.array_equal(nu, nu_reference):
                raise RuntimeError(
                    "FFTLog exponent grid changed across redshift for one component"
                )
            coefficients.append(coeff)

        stacked = np.stack(coefficients, axis=0)
        stacked = stacked.reshape(z_array.shape + (stacked.shape[-1],))
        return stacked, nu_reference

    def warm(self, component: FFTLogComponent, z_values):
        """Populate coefficient entries for all supplied redshifts."""
        self.get_many(component, z_values)
        return self

    def clear(self):
        self._coefficients.clear()


class SemiAnalyticMultipoleTerm:
    """Base class for one additive contribution to a semi-analytic multipole."""

    def evaluate(self, mode, k1, k2, z, *, cache, kernel_tables):
        raise NotImplementedError


@dataclass(frozen=True)
class SeparableMultipoleTerm(SemiAnalyticMultipoleTerm):
    """One term ``U V (k3/k)^p W`` in the appendix convention."""

    name: str
    component: FFTLogComponent
    p: int
    u: object
    v: object
    amplitude: complex = 1.0

    @property
    def kernel_shifts(self):
        return (int(self.p),)

    def evaluate(self, mode, k1, k2, z, *, cache, kernel_tables):
        k1, k2 = np.broadcast_arrays(np.asarray(k1, dtype=float), np.asarray(k2, dtype=float))
        k = np.hypot(k1, k2)
        x1, x2 = k1 / k, k2 / k
        r = np.minimum(k1, k2) / k
        coeff, nu = cache.get(self.component, z)
        table = kernel_tables[id(self.component)]
        kernel = table.evaluate(int(mode), r, shift=int(self.p))
        powers = k.reshape((1,) + k.shape) ** nu.reshape((-1,) + (1,) * k.ndim)
        result = np.sum(coeff.reshape((-1,) + (1,) * k.ndim) * powers * kernel, axis=0)
        result = self.amplitude * self.u(x1, x2) * self.v(k1, k2, z) * result
        return result.item() if result.shape == () else result


@dataclass(frozen=True)
class DirectFourierTerm(SemiAnalyticMultipoleTerm):
    """Term with a finite, explicitly known Fourier multipole coefficient."""

    name: str
    coefficient: object
    amplitude: complex = 1.0

    def evaluate(self, mode, k1, k2, z, *, cache, kernel_tables):
        value = self.amplitude * self.coefficient(int(mode), k1, k2, z)
        return np.asarray(value).item() if np.asarray(value).shape == () else value


class CompositeSemiAnalyticBispectrumMultipole3D(BispectrumMultipole3D):
    """Composable 3D multipole model built from a fixed list of terms.

    Public physical models should subclass this class and construct their term
    list in ``__init__``.  Advanced users may define an analogous subclass for
    custom bias, EFT, or response contributions.
    """

    basis = "fourier"

    def __init__(self, terms, *, angular_kernel_config: PowerLawAngularKernelTableConfig | None = None):
        self.terms = tuple(terms)
        self.angular_kernel_config = angular_kernel_config or PowerLawAngularKernelTableConfig()
        self.coefficient_cache = FFTLogCoefficientCache()
        self._kernel_tables: dict[int, PowerLawAngularKernelTable] = {}
        for term in self.terms:
            component = getattr(term, "component", None)
            if component is None:
                continue
            key = id(component)
            if key not in self._kernel_tables:
                # Exponents are component-specific but independent of z.
                _, nu = self.coefficient_cache.get(component, 0.0)
                self._kernel_tables[key] = PowerLawAngularKernelTable(nu, self.angular_kernel_config)

    @property
    def components(self):
        out = []
        seen = set()
        for term in self.terms:
            component = getattr(term, "component", None)
            if component is not None and id(component) not in seen:
                out.append(component)
                seen.add(id(component))
        return tuple(out)

    def _ensure_kernel_table(self, component):
        """Return the shared angular-kernel table for ``component``."""
        key = id(component)
        table = self._kernel_tables.get(key)
        if table is None:
            _, nu = self.coefficient_cache.get(component, 0.0)
            table = PowerLawAngularKernelTable(nu, self.angular_kernel_config)
            self._kernel_tables[key] = table
        return table

    def clear_cache(self):
        self.coefficient_cache.clear()
        self._kernel_tables.clear()

    def prepare_grid(self, k1_axis, k2_axis):
        """Prepare reusable geometry for a tensor-product Fourier grid.

        The returned :class:`TensorProductGeometryCache` is redshift
        independent.  It should be reused when evaluating the same axes at
        multiple redshifts or for several physical models.
        """
        return TensorProductGeometryCache.from_axes(k1_axis, k2_axis)

    @staticmethod
    def _resolve_grid_geometry(k1_axis, k2_axis, geometry):
        if geometry is None:
            if k1_axis is None or k2_axis is None:
                raise ValueError(
                    "k1_axis and k2_axis are required when geometry is not supplied"
                )
            return TensorProductGeometryCache.from_axes(k1_axis, k2_axis)
        if not isinstance(geometry, TensorProductGeometryCache):
            raise TypeError("geometry must be a TensorProductGeometryCache")
        if k1_axis is not None and not np.array_equal(
            np.asarray(k1_axis, dtype=float), geometry.k1_axis
        ):
            raise ValueError("k1_axis does not match the supplied geometry")
        if k2_axis is not None and not np.array_equal(
            np.asarray(k2_axis, dtype=float), geometry.k2_axis
        ):
            raise ValueError("k2_axis does not match the supplied geometry")
        return geometry

    def evaluate_modes_grid(
        self,
        modes,
        k1_axis=None,
        k2_axis=None,
        z=None,
        *,
        geometry: TensorProductGeometryCache | None = None,
        chunk_size=4096,
        **params,
    ):
        """Evaluate modes on a tensor-product grid.

        This generic implementation preserves the arbitrary-term API.  Models
        with additional algebraic structure may override it with a specialised
        axis-aware implementation.  ``TreeBispectrumMultipole3D`` does so in
        order to reuse one-dimensional spectrum evaluations and exact
        index-separation ratio groups.
        """
        if z is None:
            raise ValueError("z is required")
        geometry = self._resolve_grid_geometry(k1_axis, k2_axis, geometry)
        return self.evaluate_modes(
            modes,
            geometry.k1_axis[:, None],
            geometry.k2_axis[None, :],
            z,
            chunk_size=chunk_size,
            **params,
        )

    def _component_cores_grid(
        self,
        modes,
        component,
        shifts,
        geometry: TensorProductGeometryCache,
        z,
    ):
        r"""Evaluate shared FFTLog contractions for a tensor-product grid.

        For each requested shift this returns

        .. math::
           S_{L,p}(k_1,k_2;z)
           = \sum_n w_n(z) k^{\nu_n}
             \mathcal K_L^{(\nu_n+p)}(r).

        The exact index-separation grouping is used whenever ``geometry`` was
        prepared from identical logarithmic axes.  It avoids both repeated
        interpolation at equal ratios and the large temporary tensor with
        shape ``(n_mode, n_nu, n_k1, n_k2)``.
        """
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        shifts = tuple(sorted({int(shift) for shift in shifts}))
        coeff, nu = self.coefficient_cache.get(component, z)
        table = self._ensure_kernel_table(component)
        n_mode = modes.size
        n_point = geometry.k.size

        # This factor is independent of the angular-kernel shift and is
        # deliberately formed once for all requested p values.
        powers = geometry.k.reshape(1, n_point) ** nu.reshape(-1, 1)
        cores = {}

        if geometry.has_exact_ratio_groups:
            assert geometry.r_unique is not None
            assert geometry.r_groups is not None
            for shift in shifts:
                kernels_unique = table.evaluate_modes(
                    modes,
                    geometry.r_unique,
                    shift=shift,
                )
                core_flat = np.empty((n_mode, n_point), dtype=complex)
                for ratio_index, point_index in enumerate(geometry.r_groups):
                    left = kernels_unique[:, :, ratio_index] * coeff[None, :]
                    core_flat[:, point_index] = left @ powers[:, point_index]
                cores[shift] = core_flat.reshape((n_mode,) + geometry.shape)
            return cores

        # Fallback for arbitrary axes: retain the fully vectorized route.
        # Exact r compression is intentionally not inferred from floating
        # values, because rounding-based grouping would mix interpolation and
        # geometry errors.
        for shift in shifts:
            kernels = table.evaluate_modes(modes, geometry.r, shift=shift)
            core = np.einsum(
                "n,nc,lnc->lc",
                coeff,
                powers,
                kernels.reshape((n_mode, nu.size, n_point)),
                optimize=True,
            )
            cores[shift] = core.reshape((n_mode,) + geometry.shape)
        return cores

    def warm_cache(self, *, z, modes, shifts=None):
        """Precompute FFTLog coefficients and requested angular-kernel tables.

        Parameters
        ----------
        z
            Scalar or array-like redshifts at which FFTLog coefficients are
            required.  Passing the LOS redshift grid warms all ``w_n(z)``
            entries before projection.
        modes
            Fourier modes to precompute.
        shifts
            Optional iterable of integer kernel shifts.  If omitted, the
            shifts required by this model's separable terms are used.

        Returns
        -------
        self
            Returned for convenient method chaining.
        """
        z_values = np.asarray(z, dtype=float)
        if z_values.size == 0:
            raise ValueError("z must contain at least one redshift")
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        required = (
            {int(term.p) for term in self.terms if isinstance(term, SeparableMultipoleTerm)}
            if shifts is None else {int(shift) for shift in shifts}
        )
        for component in self.components:
            self.coefficient_cache.warm(component, z_values)
            table = self._ensure_kernel_table(component)
            for shift in required:
                for mode in modes:
                    table._build(int(mode), int(shift))
        return self

    def evaluate(self, mode, k1, k2, z, **params):
        values = self.evaluate_modes(np.atleast_1d(np.asarray(mode, dtype=int)), k1, k2, z, **params)
        return values[0] if np.isscalar(mode) else values

    def evaluate_modes(self, modes, k1, k2, z, *, chunk_size=4096, **params):
        """Evaluate many Fourier modes while sharing FFTLog contractions.

        ``z`` may be scalar or array-like.  Array-valued redshifts are grouped
        by their distinct values and each group is delegated to the scalar
        implementation :meth:`_evaluate_modes_scalar`.

        Parameters
        ----------
        modes
            Integer Fourier modes.
        k1, k2
            Broadcast-compatible Fourier-mode arrays.
        z
            Scalar or broadcast-compatible array of redshifts.
        chunk_size
            Number of flattened ``(k1,k2)`` points processed at once.  Set to
            ``None`` to process the entire input in one contraction.
        """
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        z_array = np.asarray(z, dtype=float)

        if z_array.ndim == 0:
            return self._evaluate_modes_scalar(
                modes,
                k1,
                k2,
                float(z_array),
                chunk_size=chunk_size,
                **params,
            )

        # LOS projection evaluates k1, k2, and z on a common redshift grid.
        # Group equal redshifts so each FFTLog coefficient vector is obtained
        # once and evaluated through the explicit scalar fast path.
        k1_array, k2_array, z_array = np.broadcast_arrays(
            np.asarray(k1, dtype=float),
            np.asarray(k2, dtype=float),
            z_array,
        )
        output_shape = k1_array.shape
        flat_k1 = k1_array.ravel()
        flat_k2 = k2_array.ravel()
        flat_z = z_array.ravel()
        result = np.empty((modes.size, flat_z.size), dtype=complex)

        unique_z, inverse = np.unique(flat_z, return_inverse=True)
        for i_z, z_value in enumerate(unique_z):
            select = inverse == i_z
            values = self._evaluate_modes_scalar(
                modes,
                flat_k1[select],
                flat_k2[select],
                float(z_value),
                chunk_size=chunk_size,
                **params,
            )
            result[:, select] = np.asarray(values, dtype=complex).reshape(
                modes.size, -1
            )

        return result.reshape((modes.size,) + output_shape)

    def _evaluate_modes_scalar(
        self,
        modes,
        k1,
        k2,
        z,
        *,
        chunk_size=4096,
        **params,
    ):
        """Evaluate Fourier modes at one scalar redshift."""
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        z_array = np.asarray(z, dtype=float)
        if z_array.ndim != 0:
            raise ValueError("_evaluate_modes_scalar requires scalar z")
        z = float(z_array)

        k1, k2 = np.broadcast_arrays(np.asarray(k1, dtype=float), np.asarray(k2, dtype=float))
        if np.any(k1 <= 0.0) or np.any(k2 <= 0.0):
            raise ValueError("k1 and k2 must be strictly positive")

        shape = k1.shape
        flat_k1 = k1.ravel()
        flat_k2 = k2.ravel()
        n_point = flat_k1.size
        if chunk_size is None:
            chunk_size = n_point
        chunk_size = max(1, int(chunk_size))

        result = np.zeros((modes.size, n_point), dtype=complex)

        direct_terms = [term for term in self.terms if isinstance(term, DirectFourierTerm)]
        separable_groups = {}
        for term in self.terms:
            if isinstance(term, SeparableMultipoleTerm):
                separable_groups.setdefault((id(term.component), int(term.p)), []).append(term)

        # Direct Fourier terms are inexpensive and generally have only a few
        # non-zero modes.
        for i_mode, mode in enumerate(modes):
            for term in direct_terms:
                result[i_mode] += np.asarray(
                    term.evaluate(int(mode), flat_k1, flat_k2, z,
                                  cache=self.coefficient_cache,
                                  kernel_tables=self._kernel_tables),
                    dtype=complex,
                ).ravel()

        # Geometry is independent of the FFTLog component and of the shift
        # p.  Build it once per chunk.  In particular, retain the unique r
        # values so that K_L^(nu+p)(r) is interpolated only once per distinct
        # ratio, not once per pixel.
        geometry_chunks = []
        for start in range(0, n_point, chunk_size):
            stop = min(start + chunk_size, n_point)
            k1_chunk = flat_k1[start:stop]
            k2_chunk = flat_k2[start:stop]
            k = np.hypot(k1_chunk, k2_chunk)
            r = np.minimum(k1_chunk, k2_chunk) / k
            unique_r, r_inverse = np.unique(r, return_inverse=True)
            geometry_chunks.append(
                (start, stop, k1_chunk, k2_chunk, k, r, unique_r, r_inverse)
            )

        for (_, shift), terms in separable_groups.items():
            component = terms[0].component
            coeff, nu = self.coefficient_cache.get(component, z)
            table = self._ensure_kernel_table(component)

            for start, stop, k1_chunk, k2_chunk, k, r, unique_r, r_inverse in geometry_chunks:
                x1 = k1_chunk / k
                x2 = k2_chunk / k

                # ``unique=True`` evaluates the interpolation at unique r
                # values and restores the original chunk ordering.  The
                # inverse map is constructed above once and reused for every
                # component/shift group in this call.
                kernels = table.evaluate_modes(
                    modes,
                    r,
                    shift=shift,
                    unique=True,
                    unique_r=unique_r,
                    inverse=r_inverse,
                )
                powers = k[None, :] ** nu[:, None]
                core = np.einsum(
                    "n,nc,lnc->lc",
                    coeff,
                    powers,
                    kernels,
                    optimize=True,
                )

                for term in terms:
                    prefactor = (
                        term.amplitude
                        * np.asarray(term.u(x1, x2))
                        * np.asarray(term.v(k1_chunk, k2_chunk, z))
                    )
                    result[:, start:stop] += prefactor[None, :] * core

        result = result.reshape((modes.size,) + shape)
        if shape == ():
            return result
        return result


# -----------------------------------------------------------------------------
# Predefined physical model: tree-level matter bispectrum
# -----------------------------------------------------------------------------

def _a23(p, x1, x2):
    if p == 2:
        return -5.0 / (28.0 * x2**2)
    if p == 0:
        return (10.0 * x2**2 + 3.0 * x1**2) / (28.0 * x2**2)
    if p == -2:
        return (2.0 * x1**4 + 3.0 * x1**2 * x2**2 - 5.0 * x2**4) / (28.0 * x2**2)
    raise ValueError("tree F2 shifts are p=0,+/-2")


def _tree12_coefficient(mode, linear_power):
    abs_mode = abs(int(mode))

    def coefficient(requested_mode, k1, k2, z):
        if abs(int(requested_mode)) != abs_mode:
            return np.zeros(np.broadcast(k1, k2).shape, dtype=float)

        p1 = linear_power(k1, z)
        p2 = linear_power(k2, z)

        if abs_mode == 0:
            prefactor = 12.0 / 7.0
        elif abs_mode == 1:
            prefactor = 0.5 * (k1 / k2 + k2 / k1)
        elif abs_mode == 2:
            prefactor = 1.0 / 7.0
        else:
            raise ValueError(f"Unsupported tree 12 mode: {mode}")

        return prefactor * p1 * p2

    return coefficient


class TreeBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    """Predefined tree-level matter-bispectrum multipole model.

    The list of terms is intentionally hard-coded here.  Custom models should
    subclass :class:`CompositeSemiAnalyticBispectrumMultipole3D` and define a
    corresponding explicit list, rather than composing models after creation.
    """

    def __init__(self, linear_power, k_grid, *, fftlog_config: PowerLawFFTLogConfig | None = None,
                 angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
                 regularize_squeezed: bool = True):
        component = FFTLogComponent(
            name="linear_power",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm("tree-12-F2", _tree12_coefficient(0, linear_power)),
            DirectFourierTerm("tree-12-F2", _tree12_coefficient(1, linear_power)),
            DirectFourierTerm("tree-12-F2", _tree12_coefficient(2, linear_power)),
            SeparableMultipoleTerm("tree-23-F2-p0", component, 0,
                                   lambda x1, x2: 2.0 * _a23(0, x1, x2),
                                   lambda k1, k2, z: linear_power(k2, z)),
            SeparableMultipoleTerm("tree-23-F2-p2", component, 2,
                                   lambda x1, x2: 2.0 * _a23(2, x1, x2),
                                   lambda k1, k2, z: linear_power(k2, z)),
            SeparableMultipoleTerm("tree-31-F2-p0", component, 0,
                                   lambda x1, x2: 2.0 * _a23(0, x2, x1),
                                   lambda k1, k2, z: linear_power(k1, z)),
            SeparableMultipoleTerm("tree-31-F2-p2", component, 2,
                                   lambda x1, x2: 2.0 * _a23(2, x2, x1),
                                   lambda k1, k2, z: linear_power(k1, z)),
        ]
        if regularize_squeezed:
            def ureg(x1, x2):
                return (x1*x1 - x2*x2)**2 / (14.0*x1*x1*x2*x2)
            def vreg(k1, k2, z):
                p1, p2 = linear_power(k1, z), linear_power(k2, z)
                ksq = k1*k1 + k2*k2
                pbar = 0.5*(p1 + p2)
                denom = k1*k1 - k2*k2
                # Stable local derivative on the diagonal.
                with np.errstate(divide="ignore", invalid="ignore"):
                    dp = (p1 - p2) / denom
                diagonal = np.isclose(k1, k2, rtol=1.0e-8, atol=0.0)
                if np.any(diagonal):
                    eps = 1.0e-4
                    kp = k1 * np.exp(eps)
                    km = k1 * np.exp(-eps)
                    deriv = (linear_power(kp, z) - linear_power(km, z)) / (kp - km)
                    dp = np.where(diagonal, deriv / (2.0*k1), dp)
                return 2.0*pbar - (k1**4 + 5.0*k1*k1*k2*k2 + k2**4) / ksq * dp
            terms.append(SeparableMultipoleTerm("tree-23plus31-regularized", component, -2, ureg, vreg))
        else:
            terms.extend([
                SeparableMultipoleTerm("tree-23-F2-pminus2", component, -2,
                                       lambda x1, x2: 2.0 * _a23(-2, x1, x2),
                                       lambda k1, k2, z: linear_power(k2, z)),
                SeparableMultipoleTerm("tree-31-F2-pminus2", component, -2,
                                       lambda x1, x2: 2.0 * _a23(-2, x2, x1),
                                       lambda k1, k2, z: linear_power(k1, z)),
            ])
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.regularize_squeezed = bool(regularize_squeezed)

        self._linear_power_component = component

    def warm_cache(self, *, z, modes, shifts=None):
        """Warm only non-negative modes for the parity-even tree model."""
        modes = np.unique(np.abs(np.atleast_1d(np.asarray(modes, dtype=int))))
        return super().warm_cache(z=z, modes=modes, shifts=shifts)

    def evaluate_modes(self, modes, k1, k2, z, *, chunk_size=4096, **params):
        """Generic-array evaluation exploiting ``B_{-L}=B_L`` for tree level."""
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        work_modes = np.unique(np.abs(modes))
        work = super().evaluate_modes(
            work_modes,
            k1,
            k2,
            z,
            chunk_size=chunk_size,
            **params,
        )
        inverse = np.searchsorted(work_modes, np.abs(modes))
        return work[inverse]

    @staticmethod
    def _axis_values(evaluator, axis, z):
        values = np.asarray(evaluator(axis, z), dtype=float)
        try:
            values = np.broadcast_to(values, axis.shape)
        except ValueError as error:
            raise ValueError(
                "linear_power must return values broadcastable to its k input"
            ) from error
        return np.asarray(values)

    def _axis_divided_difference(self, geometry, p1_axis, p2_axis, z):
        """Return ``[P(k1)-P(k2)]/(k1^2-k2^2)`` on the prepared grid."""
        k1 = geometry.k1_axis[:, None]
        k2 = geometry.k2_axis[None, :]
        p1 = p1_axis[:, None]
        p2 = p2_axis[None, :]
        denominator = k1 * k1 - k2 * k2
        with np.errstate(divide="ignore", invalid="ignore"):
            divided = (p1 - p2) / denominator

        diagonal = np.isclose(k1, k2, rtol=1.0e-12, atol=0.0)
        if np.any(diagonal):
            # This is the continuous diagonal limit
            # D_P(k,k) = [2k]^{-1} dP/dk.  The derivative is evaluated only
            # once on the k1 axis, rather than once per diagonal grid point.
            eps = 1.0e-4
            kp = geometry.k1_axis * np.exp(eps)
            km = geometry.k1_axis * np.exp(-eps)
            derivative = (
                self._axis_values(self.linear_power, kp, z)
                - self._axis_values(self.linear_power, km, z)
            ) / (kp - km)
            diagonal_limit = derivative[:, None] / (2.0 * k1)
            divided = np.where(diagonal, diagonal_limit, divided)
        return divided

    def evaluate_modes_grid(
        self,
        modes,
        k1_axis=None,
        k2_axis=None,
        z=None,
        *,
        geometry: TensorProductGeometryCache | None = None,
        **params,
    ):
        """Fast tree-level evaluation on a tensor-product Fourier grid.

        The method evaluates ``P(k1;z)`` and ``P(k2;z)`` only on their
        one-dimensional axes, reuses the exact ratio grouping for identical
        logarithmic axes, and assembles the hard-coded tree terms in a fused
        expression.  It returns an array with shape
        ``(len(modes), len(k1_axis), len(k2_axis))``.
        """
        if z is None:
            raise ValueError("z is required")
        geometry = self._resolve_grid_geometry(k1_axis, k2_axis, geometry)
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        work_modes = np.unique(np.abs(modes))

        p1_axis = self._axis_values(self.linear_power, geometry.k1_axis, z)
        p2_axis = self._axis_values(self.linear_power, geometry.k2_axis, z)
        p1 = p1_axis[:, None]
        p2 = p2_axis[None, :]
        n_mode = work_modes.size
        result = np.zeros((n_mode,) + geometry.shape, dtype=complex)

        # Finite Fourier contribution from the 12 permutation.  These terms
        # require no FFTLog expansion and are assembled directly from the
        # axis-cached power spectra.
        p12 = p1 * p2
        k1 = geometry.k1_axis[:, None]
        k2 = geometry.k2_axis[None, :]
        ratio = k1 / k2
        abs_modes = np.abs(work_modes)
        result[abs_modes == 0] += (12.0 / 7.0) * p12
        result[abs_modes == 1] += 0.5 * (ratio + 1.0 / ratio) * p12
        result[abs_modes == 2] += (1.0 / 7.0) * p12

        shifts = (0, 2, -2)
        cores = self._component_cores_grid(
            work_modes,
            self._linear_power_component,
            shifts,
            geometry,
            z,
        )

        # The p=0,+2 23 and 31 pieces share the same core for a fixed p.
        for shift in (0, 2):
            prefactor = (
                2.0 * _a23(shift, geometry.x1, geometry.x2) * p2
                + 2.0 * _a23(shift, geometry.x2, geometry.x1) * p1
            )
            result += prefactor[None, :, :] * cores[shift]

        if self.regularize_squeezed:
            ureg = (
                (geometry.x1 * geometry.x1 - geometry.x2 * geometry.x2) ** 2
                / (14.0 * geometry.x1 * geometry.x1 * geometry.x2 * geometry.x2)
            )
            dp = self._axis_divided_difference(geometry, p1_axis, p2_axis, z)
            pbar = 0.5 * (p1 + p2)
            k1sq = k1 * k1
            k2sq = k2 * k2
            vreg = 2.0 * pbar - (
                k1sq * k1sq + 5.0 * k1sq * k2sq + k2sq * k2sq
            ) / (geometry.k * geometry.k) * dp
            result += (ureg * vreg)[None, :, :] * cores[-2]
        else:
            prefactor = (
                2.0 * _a23(-2, geometry.x1, geometry.x2) * p2
                + 2.0 * _a23(-2, geometry.x2, geometry.x1) * p1
            )
            result += prefactor[None, :, :] * cores[-2]

        inverse = np.searchsorted(work_modes, np.abs(modes))
        return result[inverse]

# -----------------------------------------------------------------------------
# Predefined physical models: quadratic and tidal galaxy-bias contributions
# -----------------------------------------------------------------------------

def _value_at_z(value, z):
    """Evaluate a scalar bias/coefficient or a callable ``value(z)``."""
    return value(z) if callable(value) else value


def _pair_coefficients(coefficients):
    """Normalize pair coefficients to the ordered tuple ``(12, 23, 31)``."""
    if isinstance(coefficients, Mapping):
        unknown = set(coefficients) - {"12", "23", "31"}
        if unknown:
            raise ValueError(f"Unknown pair coefficient(s): {sorted(unknown)}")
        return tuple(coefficients.get(pair, 0.0) for pair in ("12", "23", "31"))
    values = tuple(coefficients)
    if len(values) != 3:
        raise ValueError("pair_coefficients must contain (C12, C23, C31)")
    return values


def _bias12_direct_coefficient(mode, linear_power, coefficient, fourier_factor):
    target = abs(int(mode))

    def evaluate(requested_mode, k1, k2, z):
        if abs(int(requested_mode)) != target:
            return np.zeros(np.broadcast(k1, k2).shape, dtype=float)
        return (
            _value_at_z(coefficient, z)
            * fourier_factor
            * linear_power(k1, z)
            * linear_power(k2, z)
        )

    return evaluate


def _t23(p, x1, x2):
    r"""Coefficient in ``S23=sum_p T23^(p) s^p``.

    Here ``x_i=k_i/sqrt(k1**2+k2**2)`` and ``p in {+2,0,-2}``.
    """
    if p == 2:
        return 1.0 / (4.0 * x2**2)
    if p == 0:
        return (x2**2 - 3.0 * x1**2) / (6.0 * x2**2)
    if p == -2:
        return (x1**2 - x2**2) ** 2 / (4.0 * x2**2)
    raise ValueError("tidal shifts are p=0,+/-2")


class QuadraticBiasBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    r"""Semi-analytic multipoles of the local quadratic-bias contribution.

    The represented bispectrum is

    .. math::
       B_{b_2}=C_{12}P_1P_2+C_{23}P_2P_3+C_{31}P_3P_1.

    The coefficients may be scalars or callables ``C_ij(z)``.  They include
    every physical prefactor, such as ``b2`` and the linear-bias factors on the
    two first-order legs.  This class intentionally contains no matter-tree or
    tidal contribution, so it can be tested and combined independently.
    """

    def __init__(self, linear_power, k_grid, *, pair_coefficients=(0.0, 0.0, 0.0),
                 fftlog_config: PowerLawFFTLogConfig | None = None,
                 angular_kernel_config: PowerLawAngularKernelTableConfig | None = None):
        c12, c23, c31 = _pair_coefficients(pair_coefficients)
        component = FFTLogComponent(
            name="linear_power_quadratic_bias",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm(
                "quadratic-bias-12-L0",
                _bias12_direct_coefficient(0, linear_power, c12, 1.0),
            ),
            SeparableMultipoleTerm(
                "quadratic-bias-23",
                component,
                0,
                lambda x1, x2: np.ones(np.broadcast(x1, x2).shape),
                lambda k1, k2, z: _value_at_z(c23, z) * linear_power(k2, z),
            ),
            SeparableMultipoleTerm(
                "quadratic-bias-31",
                component,
                0,
                lambda x1, x2: np.ones(np.broadcast(x1, x2).shape),
                lambda k1, k2, z: _value_at_z(c31, z) * linear_power(k1, z),
            ),
        ]
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.pair_coefficients = (c12, c23, c31)


class TidalBiasBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    r"""Semi-analytic multipoles of the tidal-bias contribution.

    The represented bispectrum is

    .. math::
       B_{K^2}=C_{12}S_{12}P_1P_2
                +C_{23}S_{23}P_2P_3
                +C_{31}S_{31}P_3P_1.

    ``C_ij`` contains the complete physical prefactor, conventionally
    ``2 b_K2`` times the linear-bias factors on the first-order legs.
    """

    def __init__(self, linear_power, k_grid, *, pair_coefficients=(0.0, 0.0, 0.0),
                 fftlog_config: PowerLawFFTLogConfig | None = None,
                 angular_kernel_config: PowerLawAngularKernelTableConfig | None = None):
        c12, c23, c31 = _pair_coefficients(pair_coefficients)
        component = FFTLogComponent(
            name="linear_power_tidal_bias",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm(
                "tidal-bias-12-L0",
                _bias12_direct_coefficient(0, linear_power, c12, 1.0 / 6.0),
            ),
            DirectFourierTerm(
                "tidal-bias-12-L2",
                _bias12_direct_coefficient(2, linear_power, c12, 1.0 / 4.0),
            ),
        ]
        for pair, coefficient, swap, power_leg in (
            ("23", c23, False, 2),
            ("31", c31, True, 1),
        ):
            for shift in (0, 2, -2):
                if swap:
                    u = lambda x1, x2, p=shift: _t23(p, x2, x1)
                else:
                    u = lambda x1, x2, p=shift: _t23(p, x1, x2)
                if power_leg == 2:
                    v = lambda k1, k2, z, c=coefficient: _value_at_z(c, z) * linear_power(k2, z)
                else:
                    v = lambda k1, k2, z, c=coefficient: _value_at_z(c, z) * linear_power(k1, z)
                terms.append(SeparableMultipoleTerm(
                    f"tidal-bias-{pair}-p{shift:+d}", component, shift, u, v
                ))
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.pair_coefficients = (c12, c23, c31)


class LinearCombinationBispectrumMultipole3D(BispectrumMultipole3D):
    """Weighted sum of independently defined 3D multipole models."""

    basis = "fourier"

    def __init__(self, components):
        normalized = []
        for item in components:
            if len(item) != 2:
                raise ValueError("Each component must be a (weight, model) pair")
            weight, model = item
            normalized.append((weight, model))
        if not normalized:
            raise ValueError("At least one component is required")
        self.components = tuple(normalized)

    def evaluate(self, mode, k1, k2, z, **params):
        total = 0.0
        for weight, model in self.components:
            total = total + _value_at_z(weight, z) * model.evaluate(mode, k1, k2, z, **params)
        return total

    def evaluate_modes(self, modes, k1, k2, z, **params):
        total = None
        for weight, model in self.components:
            if hasattr(model, "evaluate_modes"):
                value = model.evaluate_modes(modes, k1, k2, z, **params)
            else:
                value = np.asarray([model.evaluate(mode, k1, k2, z, **params) for mode in modes])
            value = _value_at_z(weight, z) * value
            total = value if total is None else total + value
        return total

    def evaluate_modes_grid(self, modes, k1_axis=None, k2_axis=None, z=None, *, geometry=None, **params):
        if z is None:
            raise ValueError("z is required")
        total = None
        for weight, model in self.components:
            if hasattr(model, "evaluate_modes_grid"):
                value = model.evaluate_modes_grid(
                    modes, k1_axis, k2_axis, z, geometry=geometry, **params
                )
            else:
                if geometry is not None:
                    k1 = geometry.k1_axis[:, None]
                    k2 = geometry.k2_axis[None, :]
                else:
                    k1 = np.asarray(k1_axis)[:, None]
                    k2 = np.asarray(k2_axis)[None, :]
                value = model.evaluate_modes(modes, k1, k2, z, **params)
            value = _value_at_z(weight, z) * value
            total = value if total is None else total + value
        return total

    def prepare_grid(self, k1_axis, k2_axis):
        for _, model in self.components:
            if hasattr(model, "prepare_grid"):
                return model.prepare_grid(k1_axis, k2_axis)
        return TensorProductGeometryCache.from_axes(k1_axis, k2_axis)

    def warm_cache(self, *, z, modes, shifts=None):
        for _, model in self.components:
            if hasattr(model, "warm_cache"):
                model.warm_cache(z=z, modes=modes, shifts=shifts)
        return self

    def clear_cache(self):
        for _, model in self.components:
            if hasattr(model, "clear_cache"):
                model.clear_cache()


@dataclass(frozen=True)
class TracerBias:
    r"""Eulerian bias parameters for one deterministic LSS tracer.

    Each entry may be either a scalar or a callable ``value(z)``.  Different
    tracer labels may therefore carry independent, redshift-dependent bias
    functions while sharing the same underlying linear matter spectrum.
    """

    b1: object
    b2: object = 0.0
    bK2: object = 0.0


def _coerce_tracer_bias(name, value):
    """Return a :class:`TracerBias` from a dataclass or mapping input."""
    if isinstance(value, TracerBias):
        return value
    if isinstance(value, Mapping):
        unknown = set(value) - {"b1", "b2", "bK2", "bs2"}
        if unknown:
            raise ValueError(
                f"Unknown bias parameter(s) for tracer {name!r}: {sorted(unknown)}"
            )
        if "b1" not in value:
            raise ValueError(f"Tracer {name!r} requires a 'b1' entry")
        if "bK2" in value and "bs2" in value:
            raise ValueError(
                f"Tracer {name!r} specifies both 'bK2' and its alias 'bs2'"
            )
        return TracerBias(
            b1=value["b1"],
            b2=value.get("b2", 0.0),
            bK2=value.get("bK2", value.get("bs2", 0.0)),
        )
    raise TypeError(
        f"Bias definition for tracer {name!r} must be TracerBias or a mapping"
    )


def _is_matter_field(field):
    return str(field).lower() in {"m", "matter"}


class SPTMultiTracerBispectrumMultipole3D(LinearCombinationBispectrumMultipole3D):
    r"""Tree-level real-space SPT multipoles for an ordered multi-tracer triple.

    Parameters
    ----------
    field_order : sequence of str
        Tracer identity assigned to ``(k1, k2, k3)``.  Matter may be written as
        ``"m"`` or ``"matter"``.  Every other entry must be a key of
        ``tracer_biases``.  For example,
        ``("LOWZ", "CMASS", "matter")`` represents
        :math:`B_{g_{\rm LOWZ}g_{\rm CMASS}m}` with that vertex ordering.
    tracer_biases : mapping
        Mapping from tracer label to :class:`TracerBias`, or to a mapping with
        entries ``b1``, ``b2`` and ``bK2`` (``bs2`` is accepted as an alias).

    Notes
    -----
    The model is assembled from independently reusable components,

    .. math::
       B_{A_1A_2A_3}^{\rm tree}
       = C_F B_{mmm}^{\rm tree} + B_{b_2} + B_{K^2}.

    The pair coefficient for ``ij`` is the product of the two linear biases on
    legs ``i,j`` and the second-order bias of the remaining vertex.  Thus the
    ordering of ``field_order`` is physically meaningful.
    """

    def __init__(
        self,
        linear_power,
        k_grid,
        *,
        field_order,
        tracer_biases,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
        regularize_squeezed: bool = True,
    ):
        fields = tuple(str(field) for field in field_order)
        if len(fields) != 3:
            raise ValueError("field_order must contain exactly three vertex labels")

        raw_biases = dict(tracer_biases)
        for name in raw_biases:
            if _is_matter_field(name):
                raise ValueError(
                    f"{name!r} is reserved for the matter field and must not "
                    "appear in tracer_biases"
                )
        biases = {
            str(name): _coerce_tracer_bias(str(name), value)
            for name, value in raw_biases.items()
        }
        missing = sorted({field for field in fields if not _is_matter_field(field)} - set(biases))
        if missing:
            raise ValueError(
                "Missing tracer_biases entries for field_order label(s): "
                + ", ".join(repr(name) for name in missing)
            )

        def bias(index):
            field = fields[index]
            return None if _is_matter_field(field) else biases[field]

        def lam(index, z):
            tracer = bias(index)
            return 1.0 if tracer is None else _value_at_z(tracer.b1, z)

        def second_order(index, parameter, z):
            tracer = bias(index)
            if tracer is None:
                return 0.0
            return _value_at_z(getattr(tracer, parameter), z)

        cf = lambda z: lam(0, z) * lam(1, z) * lam(2, z)

        # Pair ij means that the remaining vertex is evaluated to second order.
        c12_b2 = lambda z: lam(0, z) * lam(1, z) * second_order(2, "b2", z)
        c23_b2 = lambda z: lam(1, z) * lam(2, z) * second_order(0, "b2", z)
        c31_b2 = lambda z: lam(2, z) * lam(0, z) * second_order(1, "b2", z)

        c12_k2 = lambda z: 2.0 * lam(0, z) * lam(1, z) * second_order(2, "bK2", z)
        c23_k2 = lambda z: 2.0 * lam(1, z) * lam(2, z) * second_order(0, "bK2", z)
        c31_k2 = lambda z: 2.0 * lam(2, z) * lam(0, z) * second_order(1, "bK2", z)

        tree = TreeBispectrumMultipole3D(
            linear_power,
            k_grid,
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
            regularize_squeezed=regularize_squeezed,
        )
        quadratic = QuadraticBiasBispectrumMultipole3D(
            linear_power,
            k_grid,
            pair_coefficients=(c12_b2, c23_b2, c31_b2),
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
        )
        tidal = TidalBiasBispectrumMultipole3D(
            linear_power,
            k_grid,
            pair_coefficients=(c12_k2, c23_k2, c31_k2),
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
        )
        super().__init__(((cf, tree), (1.0, quadratic), (1.0, tidal)))

        self.field_order = fields
        self.tracer_biases = biases
        self.tree_coefficient = cf
        self.tree_matter = tree
        self.quadratic_bias = quadratic
        self.tidal_bias = tidal
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)


class SPTGalaxyBispectrumMultipole3D(SPTMultiTracerBispectrumMultipole3D):
    r"""Backward-compatible single-galaxy-tracer SPT model.

    ``field_order`` uses the legacy labels ``"g"`` and ``"m"``.  Internally
    this is a thin wrapper around :class:`SPTMultiTracerBispectrumMultipole3D`
    with one tracer named ``"galaxy"``.
    """

    def __init__(self, linear_power, k_grid, *, field_order=("g", "g", "g"),
                 b1=1.0, b2=0.0, bK2=0.0,
                 fftlog_config: PowerLawFFTLogConfig | None = None,
                 angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
                 regularize_squeezed: bool = True):
        legacy_fields = tuple(str(field).lower() for field in field_order)
        if len(legacy_fields) != 3 or any(field not in {"g", "m"} for field in legacy_fields):
            raise ValueError(
                "field_order must be a length-three tuple containing only 'g' and 'm'"
            )
        fields = tuple("galaxy" if field == "g" else "matter" for field in legacy_fields)
        super().__init__(
            linear_power,
            k_grid,
            field_order=fields,
            tracer_biases={"galaxy": TracerBias(b1=b1, b2=b2, bK2=bK2)},
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
            regularize_squeezed=regularize_squeezed,
        )
        # Preserve the public attributes and legacy field labels.
        self.field_order = legacy_fields
        self.b1 = b1
        self.b2 = b2
        self.bK2 = bK2

