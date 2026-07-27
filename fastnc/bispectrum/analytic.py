"""Analytic and semi-analytic bispectrum building blocks.

This module implements the tree-level matter bispectrum in the full Fourier
basis.  The linear spectrum is represented by a shared complex-power FFTLog
expansion.  Its third-side Fourier coefficients are computed in a single
batched angular FFT and then cached in a contracted ``(log k_>, r)`` table for
fast scalar and grid evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.integrate import quad

from .base import Bispectrum3D
from .multipole import BispectrumMultipole2D, BispectrumMultipole3D
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
    ``r=min(k2,k3)/sqrt(k2**2+k3**2)`` and
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
    r"""Geometry shared by evaluations on a tensor-product ``(k2, k3)`` grid.

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

    k2_axis: np.ndarray
    k3_axis: np.ndarray
    k: np.ndarray
    x2: np.ndarray
    x3: np.ndarray
    r: np.ndarray
    r_unique: np.ndarray | None
    r_inverse: np.ndarray | None
    r_groups: tuple[np.ndarray, ...] | None
    has_exact_ratio_groups: bool

    @classmethod
    def from_axes(cls, k2_axis, k3_axis):
        k2_axis = np.asarray(k2_axis, dtype=float)
        k3_axis = np.asarray(k3_axis, dtype=float)
        if k2_axis.ndim != 1 or k3_axis.ndim != 1:
            raise ValueError("k2_axis and k3_axis must be one-dimensional")
        if k2_axis.size == 0 or k3_axis.size == 0:
            raise ValueError("k2_axis and k3_axis must be non-empty")
        if np.any(k2_axis <= 0.0) or np.any(k3_axis <= 0.0):
            raise ValueError("tensor-product axes must be strictly positive")
        if np.any(np.diff(k2_axis) <= 0.0) or np.any(np.diff(k3_axis) <= 0.0):
            raise ValueError("tensor-product axes must be strictly increasing")

        k2 = k2_axis[:, None]
        k3 = k3_axis[None, :]
        k = np.hypot(k2, k3)
        x2 = k2 / k
        x3 = k3 / k

        same_axis = (
            k2_axis.shape == k3_axis.shape
            and np.array_equal(k2_axis, k3_axis)
        )
        is_log_uniform = False
        dln = None
        if same_axis and k2_axis.size > 1:
            log_axis = np.log(k2_axis)
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
            n_axis = k2_axis.size
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
            r = np.minimum(k2, k3) / k
            r_unique = None
            r_inverse = None
            r_groups = None
            has_exact_ratio_groups = False

        return cls(
            k2_axis=k2_axis,
            k3_axis=k3_axis,
            k=k,
            x2=x2,
            x3=x3,
            r=r,
            r_unique=r_unique,
            r_inverse=r_inverse,
            r_groups=r_groups,
            has_exact_ratio_groups=has_exact_ratio_groups,
        )

    @property
    def shape(self):
        return self.k.shape



class _MutablePhysicalCallable:
    """Stable callable identity whose physical evaluator can be replaced.

    Semi-analytic terms and FFTLog components keep a reference to this proxy,
    so updating the underlying power spectrum does not rebuild the term graph
    or the redshift-independent angular-kernel tables.
    """

    def __init__(self, evaluator):
        if not callable(evaluator):
            raise TypeError("physical evaluator must be callable")
        self._evaluator = evaluator

    @property
    def evaluator(self):
        return self._evaluator

    def update(self, evaluator):
        if not callable(evaluator):
            raise TypeError("physical evaluator must be callable")
        self._evaluator = evaluator
        return self

    def __call__(self, *args, **kwargs):
        return self._evaluator(*args, **kwargs)


@dataclass(frozen=True, eq=False)
class FFTLogComponent:
    """A named reusable FFTLog target :math:`W(k;z)`.

    ``name`` is descriptive only.  Instances are hashable by identity, so
    coefficient and angular-kernel caches are shared only when terms reference
    the same :class:`FFTLogComponent` object.
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
        self._coefficients: dict[tuple[FFTLogComponent, float], tuple[np.ndarray, np.ndarray]] = {}

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
        key = (component, z_value)
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

    def discard(self, component: FFTLogComponent):
        """Discard coefficients for one component at every redshift."""
        keys = [key for key in self._coefficients if key[0] is component]
        for key in keys:
            del self._coefficients[key]
        return self

    def discard_many(self, components):
        """Discard coefficients for selected components only."""
        component_ids = {id(component) for component in components}
        keys = [
            key for key in self._coefficients
            if id(key[0]) in component_ids
        ]
        for key in keys:
            del self._coefficients[key]
        return self

    def clear(self):
        self._coefficients.clear()
        return self


@dataclass(frozen=True, kw_only=True)
class SemiAnalyticMultipoleTerm:
    """Base class for one additive contribution to a semi-analytic multipole.

    ``weight`` is a scalar or a callable ``weight(z)``.  It is deliberately
    kept separate from the FFTLog component so scaling a term does not create
    a new coefficient or angular-kernel cache entry.
    """

    weight: object = 1.0

    def scaled_by(self, weight):
        """Return an immutable copy multiplied by an additional weight."""
        return replace(self, weight=_multiply_term_weights(self.weight, weight))

    def evaluate(self, mode, k2, k3, z, *, cache, kernel_tables):
        raise NotImplementedError


def _term_weight_value(weight, z):
    return weight(z) if callable(weight) else weight


def _multiply_term_weights(left, right):
    if not callable(left) and not callable(right):
        return left * right

    def combined(z):
        return _term_weight_value(left, z) * _term_weight_value(right, z)

    return combined


@dataclass(frozen=True)
class SeparableMultipoleTerm(SemiAnalyticMultipoleTerm):
    """One term ``U V (k3/k)^p W`` in the appendix convention."""

    name: str
    component: FFTLogComponent
    p: int
    u: object
    v: object
    amplitude: complex = 1.0
    modes: tuple[int, ...] | None = None

    def supports_mode(self, mode: int) -> bool:
        return self.modes is None or int(mode) in self.modes

    @property
    def kernel_shifts(self):
        return (int(self.p),)

    def evaluate(self, mode, k2, k3, z, *, cache, kernel_tables):
        if not self.supports_mode(int(mode)):
            shape = np.broadcast(np.asarray(k2), np.asarray(k3)).shape
            value = np.zeros(shape, dtype=complex)
            return value.item() if value.shape == () else value
        k2, k3 = np.broadcast_arrays(np.asarray(k2, dtype=float), np.asarray(k3, dtype=float))
        k = np.hypot(k2, k3)
        x2, x3 = k2 / k, k3 / k
        r = np.minimum(k2, k3) / k
        coeff, nu = cache.get(self.component, z)
        table = kernel_tables[id(self.component)]
        kernel = table.evaluate(int(mode), r, shift=int(self.p))
        powers = k.reshape((1,) + k.shape) ** nu.reshape((-1,) + (1,) * k.ndim)
        result = np.sum(coeff.reshape((-1,) + (1,) * k.ndim) * powers * kernel, axis=0)
        result = (
            _term_weight_value(self.weight, z)
            * self.amplitude
            * self.u(x2, x3)
            * self.v(k2, k3, z)
            * result
        )
        return result.item() if result.shape == () else result


@dataclass(frozen=True)
class DirectFourierTerm(SemiAnalyticMultipoleTerm):
    """Term with a finite, explicitly known Fourier multipole coefficient."""

    name: str
    coefficient: object
    amplitude: complex = 1.0

    def evaluate(self, mode, k2, k3, z, *, cache, kernel_tables):
        value = (
            _term_weight_value(self.weight, z)
            * self.amplitude
            * self.coefficient(int(mode), k2, k3, z)
        )
        return np.asarray(value).item() if np.asarray(value).shape == () else value


_PROJECTED_STATE_UNSET = object()


class _SemiAnalyticMultipoleLineOfSightProjector:
    """Coefficient-level LOS projector for semi-analytic multipoles.

    For a separable term, this object evaluates the appendix-B coefficient

        d_n(ell2, ell3) = int dchi W(chi) V(ell2/chi, ell3/chi; z)
                         w_n(z) / chi**nu_n

    before contracting it with ``ell**nu_n K_L^(nu_n+p)(r)``.  It therefore
    never evaluates ``B_L(k2,k3,z)`` on a three-dimensional LOS grid.
    """

    def __init__(self, projector, sample_combination=None):
        self.z = np.asarray(projector.z, dtype=float)
        self.chi = np.asarray(projector.chi, dtype=float)
        self.weight = np.asarray(
            projector.los_weight(sample_combination), dtype=float
        )
        self.sample_combination = (
            tuple(sample_combination) if sample_combination is not None else None
        )
        if getattr(projector, "l_shift", 0.0) != 0.0:
            raise NotImplementedError(
                "Coefficient-level semi-analytic projection requires l_shift=0"
            )

    @staticmethod
    def _as_los_values(value, n_point, n_chi):
        value = np.asarray(value)
        try:
            return np.broadcast_to(value, (n_point, n_chi))
        except ValueError as error:
            raise ValueError(
                "A semi-analytic term prefactor must broadcast to "
                "(n_ell_point, n_chi) during LOS projection"
            ) from error

    def evaluate(self, multipole3d, mode, ell2, ell3):
        modes = np.atleast_1d(np.asarray(mode, dtype=int))
        scalar_mode = np.isscalar(mode)
        ell2, ell3 = np.broadcast_arrays(
            np.asarray(ell2, dtype=float), np.asarray(ell3, dtype=float)
        )
        if np.any(ell2 <= 0.0) or np.any(ell3 <= 0.0):
            raise ValueError("ell2 and ell3 must be strictly positive")
        shape = ell2.shape
        e1 = ell2.ravel()
        e2 = ell3.ravel()
        n_point = e1.size
        n_chi = self.chi.size
        ell = np.hypot(e1, e2)
        x2 = e1 / ell
        x3 = e2 / ell
        r = np.minimum(e1, e2) / ell
        k2 = e1[:, None] / self.chi[None, :]
        k3 = e2[:, None] / self.chi[None, :]
        z = self.z[None, :]
        los_weight = self.weight[None, :]
        result = np.zeros((modes.size, n_point), dtype=complex)

        for term in multipole3d.terms:
            if isinstance(term, DirectFourierTerm):
                for i_mode, requested_mode in enumerate(modes):
                    value = term.coefficient(
                        int(requested_mode), k2, k3, z
                    )
                    value = self._as_los_values(value, n_point, n_chi)
                    term_weight = self._as_los_values(
                        _term_weight_value(term.weight, z), n_point, n_chi
                    )
                    result[i_mode] += term.amplitude * np.trapezoid(
                        los_weight * term_weight * value, self.chi, axis=1
                    )
                continue

            if not isinstance(term, SeparableMultipoleTerm):
                raise TypeError(
                    f"Unsupported semi-analytic term type: {type(term).__name__}"
                )

            coeff, nu = multipole3d.coefficient_cache.get_many(
                term.component, self.z
            )
            table = multipole3d._ensure_kernel_table(term.component)
            kernels = table.evaluate_modes(
                modes, r, shift=int(term.p), unique=True
            )
            v = self._as_los_values(
                term.v(k2, k3, z), n_point, n_chi
            )
            chi_power = self.chi[:, None] ** (-nu[None, :])
            # Shape: (n_point, n_nu).  These are the projected d_n
            # coefficients of Eq. (B11), including the model-specific V.
            term_weight = self._as_los_values(
                _term_weight_value(term.weight, z), n_point, n_chi
            )
            d_n = np.trapezoid(
                (los_weight * term_weight * v)[:, :, None]
                * coeff[None, :, :]
                * chi_power[None, :, :],
                self.chi,
                axis=1,
            )
            ell_power = ell[:, None] ** nu[None, :]
            core = np.einsum(
                "pn,pn,lnp->lp", d_n, ell_power, kernels, optimize=True
            )
            prefactor = (
                term.amplitude
                * np.asarray(term.u(x2, x3))
            )
            if term.modes is None:
                result += prefactor[None, :] * core
            else:
                active = np.isin(modes, np.asarray(term.modes, dtype=int))
                result[active] += prefactor[None, :] * core[active]

        result = result.reshape((modes.size,) + shape)
        return result[0] if scalar_mode else result


class CompositeSemiAnalyticBispectrumMultipole2D(BispectrumMultipole2D):
    """LOS-projected counterpart of
    :class:`CompositeSemiAnalyticBispectrumMultipole3D`.

    Projection is performed on the FFTLog coefficients ``w_n(z)`` to form
    the angular coefficients ``d_n(ell2,ell3)`` before the universal angular
    kernels are contracted.
    """

    def __init__(self, multipole3d, projector, sample_combination=None, modes=None):
        super().__init__(evaluator=None, basis=multipole3d.basis, modes=modes)
        self.multipole3d = multipole3d
        self._source_projector = projector
        self.projector = _SemiAnalyticMultipoleLineOfSightProjector(
            projector, sample_combination=sample_combination
        )
        self.sample_combination = self.projector.sample_combination

    def evaluate(self, mode, ell2, ell3):
        return self.projector.evaluate(self.multipole3d, mode, ell2, ell3)

    def prepare_grid(self, ell2_axis, ell3_axis):
        """Validate tensor-product angular axes.

        The coefficient-level LOS projector already shares all geometry and
        projected FFTLog contractions across the requested modes, so no
        additional persistent geometry object is required here.
        """
        ell2_axis = np.asarray(ell2_axis, dtype=float)
        ell3_axis = np.asarray(ell3_axis, dtype=float)
        if ell2_axis.ndim != 1 or ell3_axis.ndim != 1:
            raise ValueError("ell2_axis and ell3_axis must be one-dimensional")
        if np.any(ell2_axis <= 0.0) or np.any(ell3_axis <= 0.0):
            raise ValueError("ell2_axis and ell3_axis must be strictly positive")
        return None

    def evaluate_modes_grid(
        self,
        modes,
        ell2_axis,
        ell3_axis,
        *,
        geometry=None,
    ):
        """Evaluate all modes in one coefficient-level LOS projection.

        In contrast to the base-class fallback, this method passes the full
        mode vector to the semi-analytic projector.  Mode-independent LOS
        geometry, FFTLog coefficients, projected ``d_n`` coefficients, and
        term prefactors are therefore constructed only once per grid.
        """
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        ell2_axis = np.asarray(ell2_axis, dtype=float)
        ell3_axis = np.asarray(ell3_axis, dtype=float)
        self.prepare_grid(ell2_axis, ell3_axis)
        if geometry is not None:
            raise TypeError(
                "CompositeSemiAnalyticBispectrumMultipole2D does not require "
                "an external geometry object"
            )
        ell2, ell3 = np.meshgrid(ell2_axis, ell3_axis, indexing="ij")
        return self.projector.evaluate(self.multipole3d, modes, ell2, ell3)

    def update_physics(self, **changes):
        """Forward physical-state updates to the underlying 3D model.

        Physical FFTLog coefficients are invalidated by the 3D model while
        universal angular-kernel tables and the LOS-projection state are kept.
        """
        self.multipole3d.update_physics(**changes)
        return self

    def update_projection(
        self,
        projector=None,
        *,
        sample_combination=_PROJECTED_STATE_UNSET,
        **projector_changes,
    ):
        """Refresh or replace the LOS-projector state.

        When ``projector`` is omitted, ``projector_changes`` are applied to
        the source projector through ``update_state()`` and the internal
        coefficient-level snapshot is rebuilt.  With no changes, this simply
        refreshes the snapshot after an external projector update.

        This operation does not invalidate 3D FFTLog coefficients or angular
        kernels.  A changed LOS redshift grid is warmed lazily unless
        :meth:`warm` is called explicitly afterwards.
        """
        if projector is not None and projector_changes:
            raise TypeError(
                "Pass either a replacement projector or projector state changes, not both"
            )
        if projector is not None:
            self._source_projector = projector
        elif projector_changes:
            update_state = getattr(self._source_projector, "update_state", None)
            if update_state is None:
                raise TypeError(
                    "The source projector does not provide update_state()"
                )
            update_state(**projector_changes)

        if sample_combination is _PROJECTED_STATE_UNSET:
            sample_combination = self.sample_combination
        self.projector = _SemiAnalyticMultipoleLineOfSightProjector(
            self._source_projector, sample_combination=sample_combination
        )
        self.sample_combination = self.projector.sample_combination
        return self

    def warm(self, modes=None):
        """Warm FFTLog coefficients and angular kernels used by projection."""
        if modes is None:
            modes = self.available_modes()
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        self.multipole3d.warm_cache(z=self.projector.z, modes=modes)
        return self


class CompositeSemiAnalyticBispectrumMultipole3D(BispectrumMultipole3D):
    """Composable 3D multipole model built from a fixed list of terms.

    Public physical models should subclass this class and construct their term
    list in ``__init__``.  Advanced users may define an analogous subclass for
    custom bias, EFT, or response contributions.
    """

    basis = "fourier"

    def project_los(self, projector, sample_combination=None, modes=None, mode_max=None):
        if modes is None and mode_max is not None:
            mode_max = int(mode_max)
            modes = np.arange(-mode_max, mode_max + 1)
        return CompositeSemiAnalyticBispectrumMultipole2D(
            self, projector, sample_combination=sample_combination, modes=modes
        )

    def __init__(self, terms, *, angular_kernel_config: PowerLawAngularKernelTableConfig | None = None):
        normalized = tuple(terms)
        if not normalized:
            raise ValueError("At least one semi-analytic term is required")
        if not all(isinstance(term, SemiAnalyticMultipoleTerm) for term in normalized):
            raise TypeError("terms must contain only SemiAnalyticMultipoleTerm instances")
        self._terms = normalized
        self.angular_kernel_config = angular_kernel_config or PowerLawAngularKernelTableConfig()
        self.coefficient_cache = FFTLogCoefficientCache()
        self._kernel_tables: dict[FFTLogComponent, PowerLawAngularKernelTable] = {}
        for term in self._terms:
            component = getattr(term, "component", None)
            if component is None:
                continue
            if component not in self._kernel_tables:
                # Exponents are component-specific but independent of z.
                _, nu = self.coefficient_cache.get(component, 0.0)
                self._kernel_tables[component] = PowerLawAngularKernelTable(
                    nu, self.angular_kernel_config
                )

    @classmethod
    def from_terms(cls, terms, *, angular_kernel_config=None):
        """Construct a composite from an unweighted sequence of terms."""
        return cls(terms, angular_kernel_config=angular_kernel_config)

    @classmethod
    def from_weighted_terms(cls, weighted_terms, *, angular_kernel_config=None):
        """Construct a composite from ``(weight, term)`` pairs."""
        terms = []
        for item in weighted_terms:
            if len(item) != 2:
                raise ValueError("Each weighted term must be a (weight, term) pair")
            weight, term = item
            if not isinstance(term, SemiAnalyticMultipoleTerm):
                raise TypeError("weighted_terms must contain semi-analytic terms")
            terms.append(term.scaled_by(weight))
        return cls.from_terms(terms, angular_kernel_config=angular_kernel_config)

    @classmethod
    def from_composites(cls, weighted_composites, *, angular_kernel_config=None):
        """Flatten weighted semi-analytic composites into one term list."""
        terms = []
        for item in weighted_composites:
            if len(item) != 2:
                raise ValueError(
                    "Each weighted composite must be a (weight, composite) pair"
                )
            weight, composite = item
            if not isinstance(composite, CompositeSemiAnalyticBispectrumMultipole3D):
                raise TypeError(
                    "from_composites accepts only "
                    "CompositeSemiAnalyticBispectrumMultipole3D instances"
                )
            terms.extend(term.scaled_by(weight) for term in composite.terms)
        return cls.from_terms(terms, angular_kernel_config=angular_kernel_config)

    @property
    def terms(self):
        """Immutable flattened term sequence defining this model."""
        return self._terms

    @property
    def components(self):
        out = []
        seen = set()
        for term in self.terms:
            component = getattr(term, "component", None)
            if component is not None and component not in seen:
                out.append(component)
                seen.add(component)
        return tuple(out)

    def _ensure_kernel_table(self, component):
        """Return the shared angular-kernel table for ``component``."""
        key = component
        table = self._kernel_tables.get(key)
        if table is None:
            _, nu = self.coefficient_cache.get(component, 0.0)
            table = PowerLawAngularKernelTable(nu, self.angular_kernel_config)
            self._kernel_tables[key] = table
        return table

    def invalidate_components(self, components):
        """Invalidate physical FFTLog coefficients, preserving kernels.

        The angular tables depend on the FFTLog exponent grid and geometry,
        not on the sampled values of the physical spectrum.
        """
        self.coefficient_cache.discard_many(components)
        return self

    def clear_coefficient_cache(self):
        self.coefficient_cache.clear()
        return self

    def clear_kernel_cache(self):
        self._kernel_tables.clear()
        return self

    def clear_cache(self):
        self.clear_coefficient_cache()
        self.clear_kernel_cache()
        return self

    def update_physics(self, **changes):
        raise NotImplementedError(
            f"{type(self).__name__} does not define physical-state updates"
        )

    def prepare_grid(self, k2_axis, k3_axis):
        """Prepare reusable geometry for a tensor-product Fourier grid.

        The returned :class:`TensorProductGeometryCache` is redshift
        independent.  It should be reused when evaluating the same axes at
        multiple redshifts or for several physical models.
        """
        return TensorProductGeometryCache.from_axes(k2_axis, k3_axis)

    @staticmethod
    def _resolve_grid_geometry(k2_axis, k3_axis, geometry):
        if geometry is None:
            if k2_axis is None or k3_axis is None:
                raise ValueError(
                    "k2_axis and k3_axis are required when geometry is not supplied"
                )
            return TensorProductGeometryCache.from_axes(k2_axis, k3_axis)
        if not isinstance(geometry, TensorProductGeometryCache):
            raise TypeError("geometry must be a TensorProductGeometryCache")
        if k2_axis is not None and not np.array_equal(
            np.asarray(k2_axis, dtype=float), geometry.k2_axis
        ):
            raise ValueError("k2_axis does not match the supplied geometry")
        if k3_axis is not None and not np.array_equal(
            np.asarray(k3_axis, dtype=float), geometry.k3_axis
        ):
            raise ValueError("k3_axis does not match the supplied geometry")
        return geometry

    def evaluate_modes_grid(
        self,
        modes,
        k2_axis=None,
        k3_axis=None,
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
        geometry = self._resolve_grid_geometry(k2_axis, k3_axis, geometry)
        return self.evaluate_modes(
            modes,
            geometry.k2_axis[:, None],
            geometry.k3_axis[None, :],
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

    def evaluate(self, mode, k2, k3, z, **params):
        values = self.evaluate_modes(np.atleast_1d(np.asarray(mode, dtype=int)), k2, k3, z, **params)
        return values[0] if np.isscalar(mode) else values

    def evaluate_modes(self, modes, k2, k3, z, *, chunk_size=4096, **params):
        """Evaluate many Fourier modes while sharing FFTLog contractions.

        ``z`` may be scalar or array-like.  Array-valued redshifts are grouped
        by their distinct values and each group is delegated to the scalar
        implementation :meth:`_evaluate_modes_scalar`.

        Parameters
        ----------
        modes
            Integer Fourier modes.
        k2, k3
            Broadcast-compatible Fourier-mode arrays.
        z
            Scalar or broadcast-compatible array of redshifts.
        chunk_size
            Number of flattened ``(k2,k3)`` points processed at once.  Set to
            ``None`` to process the entire input in one contraction.
        """
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        z_array = np.asarray(z, dtype=float)

        if z_array.ndim == 0:
            return self._evaluate_modes_scalar(
                modes,
                k2,
                k3,
                float(z_array),
                chunk_size=chunk_size,
                **params,
            )

        # LOS projection evaluates k2, k3, and z on a common redshift grid.
        # Group equal redshifts so each FFTLog coefficient vector is obtained
        # once and evaluated through the explicit scalar fast path.
        k1_array, k2_array, z_array = np.broadcast_arrays(
            np.asarray(k2, dtype=float),
            np.asarray(k3, dtype=float),
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
        k2,
        k3,
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

        k2, k3 = np.broadcast_arrays(np.asarray(k2, dtype=float), np.asarray(k3, dtype=float))
        if np.any(k2 <= 0.0) or np.any(k3 <= 0.0):
            raise ValueError("k2 and k3 must be strictly positive")

        shape = k2.shape
        flat_k1 = k2.ravel()
        flat_k2 = k3.ravel()
        n_point = flat_k1.size
        if chunk_size is None:
            chunk_size = n_point
        chunk_size = max(1, int(chunk_size))

        result = np.zeros((modes.size, n_point), dtype=complex)

        direct_terms = [term for term in self.terms if isinstance(term, DirectFourierTerm)]
        separable_groups = {}
        for term in self.terms:
            if isinstance(term, SeparableMultipoleTerm):
                separable_groups.setdefault((term.component, int(term.p)), []).append(term)

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
                x2 = k1_chunk / k
                x3 = k2_chunk / k

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
                        _term_weight_value(term.weight, z)
                        * term.amplitude
                        * np.asarray(term.u(x2, x3))
                        * np.asarray(term.v(k1_chunk, k2_chunk, z))
                    )
                    if term.modes is None:
                        result[:, start:stop] += prefactor[None, :] * core
                    else:
                        active = np.isin(modes, np.asarray(term.modes, dtype=int))
                        result[active, start:stop] += prefactor[None, :] * core[active]

        result = result.reshape((modes.size,) + shape)
        if shape == ():
            return result
        return result


# -----------------------------------------------------------------------------
# Predefined physical model: tree-level matter bispectrum
# -----------------------------------------------------------------------------

def _a31(p, x2, x3):
    if p == 2:
        return -5.0 / (28.0 * x3**2)
    if p == 0:
        return (10.0 * x3**2 + 3.0 * x2**2) / (28.0 * x3**2)
    if p == -2:
        return (2.0 * x2**4 + 3.0 * x2**2 * x3**2 - 5.0 * x3**4) / (28.0 * x3**2)
    raise ValueError("tree F2 shifts are p=0,+/-2")


def _tree23_coefficient(mode, linear_power):
    abs_mode = abs(int(mode))

    def coefficient(requested_mode, k2, k3, z):
        if abs(int(requested_mode)) != abs_mode:
            return np.zeros(np.broadcast(k2, k3).shape, dtype=float)

        p1 = linear_power(k2, z)
        p2 = linear_power(k3, z)

        if abs_mode == 0:
            prefactor = 12.0 / 7.0
        elif abs_mode == 1:
            prefactor = 0.5 * (k2 / k3 + k3 / k2)
        elif abs_mode == 2:
            prefactor = 1.0 / 7.0
        else:
            raise ValueError(f"Unsupported tree 23 mode: {mode}")

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
        linear_power = (linear_power if isinstance(linear_power, _MutablePhysicalCallable)
                        else _MutablePhysicalCallable(linear_power))
        component = FFTLogComponent(
            name="linear_power",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm("tree-23-F2", _tree23_coefficient(0, linear_power)),
            DirectFourierTerm("tree-23-F2", _tree23_coefficient(1, linear_power)),
            DirectFourierTerm("tree-23-F2", _tree23_coefficient(2, linear_power)),
            SeparableMultipoleTerm("tree-31-F2-p0", component, 0,
                                   lambda x2, x3: 2.0 * _a31(0, x2, x3),
                                   lambda k2, k3, z: linear_power(k3, z)),
            SeparableMultipoleTerm("tree-31-F2-p2", component, 2,
                                   lambda x2, x3: 2.0 * _a31(2, x2, x3),
                                   lambda k2, k3, z: linear_power(k3, z)),
            SeparableMultipoleTerm("tree-12-F2-p0", component, 0,
                                   lambda x2, x3: 2.0 * _a31(0, x3, x2),
                                   lambda k2, k3, z: linear_power(k2, z)),
            SeparableMultipoleTerm("tree-12-F2-p2", component, 2,
                                   lambda x2, x3: 2.0 * _a31(2, x3, x2),
                                   lambda k2, k3, z: linear_power(k2, z)),
        ]
        if regularize_squeezed:
            def ureg(x2, x3):
                return (x2*x2 - x3*x3)**2 / (14.0*x2*x2*x3*x3)
            def vreg(k2, k3, z):
                p1, p2 = linear_power(k2, z), linear_power(k3, z)
                ksq = k2*k2 + k3*k3
                pbar = 0.5*(p1 + p2)
                denom = k2*k2 - k3*k3
                # Stable local derivative on the diagonal.
                with np.errstate(divide="ignore", invalid="ignore"):
                    dp = (p1 - p2) / denom
                diagonal = np.isclose(k2, k3, rtol=1.0e-8, atol=0.0)
                if np.any(diagonal):
                    eps = 1.0e-4
                    kp = k2 * np.exp(eps)
                    km = k2 * np.exp(-eps)
                    deriv = (linear_power(kp, z) - linear_power(km, z)) / (kp - km)
                    dp = np.where(diagonal, deriv / (2.0*k2), dp)
                return 2.0*pbar - (k2**4 + 5.0*k2*k2*k3*k3 + k3**4) / ksq * dp
            terms.append(SeparableMultipoleTerm("tree-31plus12-regularized", component, -2, ureg, vreg))
        else:
            terms.extend([
                SeparableMultipoleTerm("tree-31-F2-pminus2", component, -2,
                                       lambda x2, x3: 2.0 * _a31(-2, x2, x3),
                                       lambda k2, k3, z: linear_power(k3, z)),
                SeparableMultipoleTerm("tree-12-F2-pminus2", component, -2,
                                       lambda x2, x3: 2.0 * _a31(-2, x3, x2),
                                       lambda k2, k3, z: linear_power(k2, z)),
            ])
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.regularize_squeezed = bool(regularize_squeezed)

        self._linear_power_component = component

    def update_physics(self, *, linear_power):
        """Replace the linear spectrum without rebuilding model structure."""
        self.linear_power.update(linear_power)
        self.invalidate_components((self._linear_power_component,))
        return self

    def warm_cache(self, *, z, modes, shifts=None):
        """Warm only non-negative modes for the parity-even tree model."""
        modes = np.unique(np.abs(np.atleast_1d(np.asarray(modes, dtype=int))))
        return super().warm_cache(z=z, modes=modes, shifts=shifts)

    def evaluate_modes(self, modes, k2, k3, z, *, chunk_size=4096, **params):
        """Generic-array evaluation exploiting ``B_{-L}=B_L`` for tree level."""
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        work_modes = np.unique(np.abs(modes))
        work = super().evaluate_modes(
            work_modes,
            k2,
            k3,
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
        """Return ``[P(k2)-P(k3)]/(k2^2-k3^2)`` on the prepared grid."""
        k2 = geometry.k2_axis[:, None]
        k3 = geometry.k3_axis[None, :]
        p1 = p1_axis[:, None]
        p2 = p2_axis[None, :]
        denominator = k2 * k2 - k3 * k3
        with np.errstate(divide="ignore", invalid="ignore"):
            divided = (p1 - p2) / denominator

        diagonal = np.isclose(k2, k3, rtol=1.0e-12, atol=0.0)
        if np.any(diagonal):
            # This is the continuous diagonal limit
            # D_P(k,k) = [2k]^{-1} dP/dk.  The derivative is evaluated only
            # once on the k2 axis, rather than once per diagonal grid point.
            eps = 1.0e-4
            kp = geometry.k2_axis * np.exp(eps)
            km = geometry.k2_axis * np.exp(-eps)
            derivative = (
                self._axis_values(self.linear_power, kp, z)
                - self._axis_values(self.linear_power, km, z)
            ) / (kp - km)
            diagonal_limit = derivative[:, None] / (2.0 * k2)
            divided = np.where(diagonal, diagonal_limit, divided)
        return divided

    def evaluate_modes_grid(
        self,
        modes,
        k2_axis=None,
        k3_axis=None,
        z=None,
        *,
        geometry: TensorProductGeometryCache | None = None,
        **params,
    ):
        """Fast tree-level evaluation on a tensor-product Fourier grid.

        The method evaluates ``P(k2;z)`` and ``P(k3;z)`` only on their
        one-dimensional axes, reuses the exact ratio grouping for identical
        logarithmic axes, and assembles the hard-coded tree terms in a fused
        expression.  It returns an array with shape
        ``(len(modes), len(k2_axis), len(k3_axis))``.
        """
        if z is None:
            raise ValueError("z is required")
        geometry = self._resolve_grid_geometry(k2_axis, k3_axis, geometry)
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        work_modes = np.unique(np.abs(modes))

        p1_axis = self._axis_values(self.linear_power, geometry.k2_axis, z)
        p2_axis = self._axis_values(self.linear_power, geometry.k3_axis, z)
        p1 = p1_axis[:, None]
        p2 = p2_axis[None, :]
        n_mode = work_modes.size
        result = np.zeros((n_mode,) + geometry.shape, dtype=complex)

        # Finite Fourier contribution from the 12 permutation.  These terms
        # require no FFTLog expansion and are assembled directly from the
        # axis-cached power spectra.
        p12 = p1 * p2
        k2 = geometry.k2_axis[:, None]
        k3 = geometry.k3_axis[None, :]
        ratio = k2 / k3
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
                2.0 * _a31(shift, geometry.x2, geometry.x3) * p2
                + 2.0 * _a31(shift, geometry.x3, geometry.x2) * p1
            )
            result += prefactor[None, :, :] * cores[shift]

        if self.regularize_squeezed:
            ureg = (
                (geometry.x2 * geometry.x2 - geometry.x3 * geometry.x3) ** 2
                / (14.0 * geometry.x2 * geometry.x2 * geometry.x3 * geometry.x3)
            )
            dp = self._axis_divided_difference(geometry, p1_axis, p2_axis, z)
            pbar = 0.5 * (p1 + p2)
            k1sq = k2 * k2
            k2sq = k3 * k3
            vreg = 2.0 * pbar - (
                k1sq * k1sq + 5.0 * k1sq * k2sq + k2sq * k2sq
            ) / (geometry.k * geometry.k) * dp
            result += (ureg * vreg)[None, :, :] * cores[-2]
        else:
            prefactor = (
                2.0 * _a31(-2, geometry.x2, geometry.x3) * p2
                + 2.0 * _a31(-2, geometry.x3, geometry.x2) * p1
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


def _bias23_direct_coefficient(mode, linear_power, coefficient, fourier_factor):
    target = abs(int(mode))

    def evaluate(requested_mode, k2, k3, z):
        if abs(int(requested_mode)) != target:
            return np.zeros(np.broadcast(k2, k3).shape, dtype=float)
        return (
            _value_at_z(coefficient, z)
            * fourier_factor
            * linear_power(k2, z)
            * linear_power(k3, z)
        )

    return evaluate


def _t31(p, x2, x3):
    r"""Coefficient in ``S31=sum_p T31^(p) s^p``.

    Here ``x_i=k_i/sqrt(k2**2+k3**2)`` and ``p in {+2,0,-2}``.
    """
    if p == 2:
        return 1.0 / (4.0 * x3**2)
    if p == 0:
        return (x3**2 - 3.0 * x2**2) / (6.0 * x3**2)
    if p == -2:
        return (x2**2 - x3**2) ** 2 / (4.0 * x3**2)
    raise ValueError("tidal shifts are p=0,+/-2")



class BiHalofitBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    r"""Semi-analytic full-Fourier multipoles of the complete BiHalofit model.

    The three-halo term is decomposed into separable FFTLog contributions.
    The one-halo shape variables are fixed to the constructor values
    ``r1`` and ``r2``, making that contribution exactly separable:

    .. math::
       B_{1h}^{\rm fixed}=H(k_1;z,r_1,r_2)H(k_2;z,r_1,r_2)
                           H(k_3;z,r_1,r_2).

    One- and three-halo contributions are combined in this single model.
    """

    def __init__(
        self,
        halofit,
        *,
        r1: float = 0.5,
        r2: float = 0.0,
        k_grid=None,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
    ):
        r1 = float(r1)
        r2 = float(r2)
        if not np.isfinite(r1) or not 0.0 <= r1 <= 1.0:
            raise ValueError("r1 must be finite and lie in [0, 1]")
        if not np.isfinite(r2) or not 0.0 <= r2 <= 1.0:
            raise ValueError("r2 must be finite and lie in [0, 1]")

        halofit.update()
        state = {"halofit": halofit}
        k_grid = np.asarray(halofit.k if k_grid is None else k_grid, dtype=float)
        if k_grid.ndim != 1 or k_grid.size < 2 or np.any(k_grid <= 0.0):
            raise ValueError("k_grid must be a one-dimensional positive grid")
        fftlog_config = fftlog_config or PowerLawFFTLogConfig()
        def coeffs(z):
            return state["halofit"].get_bihalofit_coeffs(np.asarray(z, dtype=float))

        def q_and_coeff(k, z):
            c = coeffs(z)
            q = np.maximum(np.asarray(k, dtype=float) * c["r_sigma"], 1.0e-100)
            return q, c

        def damping(k, z):
            q, c = q_and_coeff(k, z)
            return 1.0 / (1.0 + c["en"] * q)

        def effective_power(k, z):
            q, c = q_and_coeff(k, z)
            pl = state["halofit"].get_interpolated_pklin(np.asarray(k, dtype=float), z)
            return (
                (1.0 + c["fn"] * q**2)
                / (1.0 + c["gn"] * q + c["hn"] * q**2)
                * pl
                + 1.0
                / (c["mn"] * q**c["mun"] + c["nn"] * q**c["nun"])
                / (1.0 + (c["pn"] * q) ** -3)
            )

        def dressed_power(k, z):
            return damping(k, z) * effective_power(k, z)

        def one_halo_profile(k, z):
            q, c = q_and_coeff(k, z)
            an = 10.0 ** (c["log10an1"] + c["log10an2"] * r1 ** c["gan"])
            aln = 10.0 ** (c["log10aln1"] + c["log10aln2"] * r2**2)
            ns = float(state["halofit"].cosmo["ns"])
            aln = np.minimum(aln, 1.0 - (2.0 / 3.0) * ns)
            ben = 10.0 ** (c["log10ben1"] + c["log10ben2"] * r2)
            return (
                1.0 / (an * q**aln + c["bn"] * q**ben)
                / (1.0 + 1.0 / (c["cn"] * q))
            )

        component_i = FFTLogComponent("bihalofit-I", k_grid, damping, fftlog_config)
        component_h = FFTLogComponent("bihalofit-IPE", k_grid, dressed_power, fftlog_config)
        component_1h = FFTLogComponent(
            f"bihalofit-1h-r1={r1:g}-r2={r2:g}",
            k_grid,
            one_halo_profile,
            fftlog_config,
        )

        def one(x2, x3):
            return np.ones(np.broadcast(x2, x3).shape, dtype=float)

        def v_1h(k2, k3, z):
            return one_halo_profile(k2, z) * one_halo_profile(k3, z)

        def v_23(k2, k3, z):
            return dressed_power(k2, z) * dressed_power(k3, z)

        def v_31(k2, k3, z):
            return damping(k2, z) * dressed_power(k3, z)

        def v_12(k2, k3, z):
            return dressed_power(k2, z) * damping(k3, z)

        def f23_u(mode_abs):
            if mode_abs == 0:
                return lambda x2, x3: np.full(np.broadcast(x2, x3).shape, 12.0 / 7.0)
            if mode_abs == 1:
                return lambda x2, x3: 0.5 * (x2 / x3 + x3 / x2)
            if mode_abs == 2:
                return lambda x2, x3: np.full(np.broadcast(x2, x3).shape, 1.0 / 7.0)
            raise ValueError("F2 23 has only |L|=0,1,2")

        terms = [
            SeparableMultipoleTerm(
                "bihalofit-1h-fixed-shape", component_1h, 0, one, v_1h
            )
        ]

        for mode_abs in (0, 1, 2):
            active_modes = (0,) if mode_abs == 0 else (-mode_abs, mode_abs)
            terms.append(
                SeparableMultipoleTerm(
                    f"bihalofit-3h-23-F2-L{mode_abs}",
                    component_i,
                    0,
                    f23_u(mode_abs),
                    v_23,
                    modes=active_modes,
                )
            )

        for p in (-2, 0, 2):
            terms.extend(
                [
                    SeparableMultipoleTerm(
                        f"bihalofit-3h-31-F2-p{p}",
                        component_h,
                        p,
                        lambda x2, x3, p=p: 2.0 * _a31(p, x2, x3),
                        v_31,
                    ),
                    SeparableMultipoleTerm(
                        f"bihalofit-3h-12-F2-p{p}",
                        component_h,
                        p,
                        lambda x2, x3, p=p: 2.0 * _a31(p, x3, x2),
                        v_12,
                    ),
                ]
            )

        def v_dn23(k2, k3, z):
            c = coeffs(z)
            k = np.hypot(k2, k3)
            return 2.0 * c["dn"] * c["r_sigma"] * k * v_23(k2, k3, z)

        def v_dn31(k2, k3, z):
            c = coeffs(z)
            return (
                2.0
                * c["dn"]
                * c["r_sigma"]
                * k3
                * damping(k2, z)
                * dressed_power(k3, z)
            )

        def v_dn12(k2, k3, z):
            c = coeffs(z)
            return (
                2.0
                * c["dn"]
                * c["r_sigma"]
                * k2
                * dressed_power(k2, z)
                * damping(k3, z)
            )

        terms.extend(
            [
                SeparableMultipoleTerm(
                    "bihalofit-3h-23-dnq1", component_i, 1, one, v_dn23
                ),
                SeparableMultipoleTerm(
                    "bihalofit-3h-31-dnq3", component_h, 0, one, v_dn31
                ),
                SeparableMultipoleTerm(
                    "bihalofit-3h-12-dnq2", component_h, 0, one, v_dn12
                ),
            ]
        )

        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.halofit = halofit
        self.r1 = r1
        self.r2 = r2
        self.k_grid = k_grid
        self.fftlog_config = fftlog_config
        self._damping = damping
        self._effective_power = effective_power
        self._dressed_power = dressed_power
        self._one_halo_profile = one_halo_profile
        self._physical_state = state
        self._physical_components = (component_i, component_h, component_1h)

    def update_physics(self, *, halofit):
        """Replace the Halofit state while retaining angular kernels."""
        halofit.update()
        self._physical_state["halofit"] = halofit
        self.halofit = halofit
        self.invalidate_components(self._physical_components)
        return self


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
        linear_power = (linear_power if isinstance(linear_power, _MutablePhysicalCallable)
                        else _MutablePhysicalCallable(linear_power))
        c12, c23, c31 = _pair_coefficients(pair_coefficients)
        component = FFTLogComponent(
            name="linear_power_quadratic_bias",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm(
                "quadratic-bias-23-L0",
                _bias23_direct_coefficient(0, linear_power, c23, 1.0),
            ),
            SeparableMultipoleTerm(
                "quadratic-bias-31",
                component,
                0,
                lambda x2, x3: np.ones(np.broadcast(x2, x3).shape),
                lambda k2, k3, z: _value_at_z(c31, z) * linear_power(k3, z),
            ),
            SeparableMultipoleTerm(
                "quadratic-bias-12",
                component,
                0,
                lambda x2, x3: np.ones(np.broadcast(x2, x3).shape),
                lambda k2, k3, z: _value_at_z(c12, z) * linear_power(k2, z),
            ),
        ]
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.pair_coefficients = (c12, c23, c31)
        self._linear_power_component = component

    def update_physics(self, *, linear_power):
        self.linear_power.update(linear_power)
        self.invalidate_components((self._linear_power_component,))
        return self


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
        linear_power = (linear_power if isinstance(linear_power, _MutablePhysicalCallable)
                        else _MutablePhysicalCallable(linear_power))
        c12, c23, c31 = _pair_coefficients(pair_coefficients)
        component = FFTLogComponent(
            name="linear_power_tidal_bias",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm(
                "tidal-bias-23-L0",
                _bias23_direct_coefficient(0, linear_power, c23, 1.0 / 6.0),
            ),
            DirectFourierTerm(
                "tidal-bias-23-L2",
                _bias23_direct_coefficient(2, linear_power, c23, 1.0 / 4.0),
            ),
        ]
        for pair, coefficient, swap, power_leg in (
            ("31", c31, False, 3),
            ("12", c12, True, 2),
        ):
            for shift in (0, 2, -2):
                if swap:
                    u = lambda x2, x3, p=shift: _t31(p, x3, x2)
                else:
                    u = lambda x2, x3, p=shift: _t31(p, x2, x3)
                if power_leg == 3:
                    v = lambda k2, k3, z, c=coefficient: _value_at_z(c, z) * linear_power(k3, z)
                else:
                    v = lambda k2, k3, z, c=coefficient: _value_at_z(c, z) * linear_power(k2, z)
                terms.append(SeparableMultipoleTerm(
                    f"tidal-bias-{pair}-p{shift:+d}", component, shift, u, v
                ))
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.pair_coefficients = (c12, c23, c31)
        self._linear_power_component = component

    def update_physics(self, *, linear_power):
        self.linear_power.update(linear_power)
        self.invalidate_components((self._linear_power_component,))
        return self


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


class SPTMultiTracerBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    r"""Tree-level real-space SPT multipoles for an ordered multi-tracer triple.

    Parameters
    ----------
    field_order : sequence of str
        Tracer identity assigned to ``(k2, k3, k3)``.  Matter may be written as
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

        linear_power = (linear_power if isinstance(linear_power, _MutablePhysicalCallable)
                        else _MutablePhysicalCallable(linear_power))

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
        # All three pieces use the same linear-power FFTLog expansion.  Rebind
        # separable terms to the tree component so identity-keyed coefficient
        # and angular-kernel caches are shared by the flattened SPT model.
        shared_component = tree._linear_power_component

        def share_component(term):
            if isinstance(term, SeparableMultipoleTerm):
                return replace(term, component=shared_component)
            return term

        terms = [term.scaled_by(cf) for term in tree.terms]
        terms.extend(share_component(term) for term in quadratic.terms)
        terms.extend(share_component(term) for term in tidal.terms)
        super().__init__(terms, angular_kernel_config=angular_kernel_config)

        self.field_order = fields
        self._physical_field_order = fields
        self.tracer_biases = biases
        self.tree_coefficient = cf
        self.tree_matter = tree
        self.quadratic_bias = quadratic
        self.tidal_bias = tidal
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self._linear_power_component = shared_component

    def update_physics(self, *, linear_power=None, tracer_biases=None):
        """Atomically update spectrum and/or tracer-bias state.

        Bias-only updates retain every cache entry.  A spectrum update removes
        only FFTLog coefficients for the shared power component; the expensive
        angular-kernel table is preserved.
        """
        if linear_power is None and tracer_biases is None:
            raise ValueError("at least one physical-state change is required")

        if tracer_biases is not None:
            raw = dict(tracer_biases)
            for name in raw:
                if _is_matter_field(name):
                    raise ValueError(
                        f"{name!r} is reserved for the matter field and must not "
                        "appear in tracer_biases"
                    )
            updated = {
                str(name): _coerce_tracer_bias(str(name), value)
                for name, value in raw.items()
            }
            required = {
                field for field in self._physical_field_order if not _is_matter_field(field)
            }
            missing = sorted(required - set(updated))
            if missing:
                raise ValueError(
                    "Missing tracer_biases entries for field_order label(s): "
                    + ", ".join(repr(name) for name in missing)
                )
            self.tracer_biases.clear()
            self.tracer_biases.update(updated)

        if linear_power is not None:
            self.linear_power.update(linear_power)
            self.invalidate_components((self._linear_power_component,))
        return self


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

    def update_physics(
        self,
        *,
        linear_power=None,
        b1=None,
        b2=None,
        bK2=None,
    ):
        current = self.tracer_biases["galaxy"]
        bias_changed = any(value is not None for value in (b1, b2, bK2))
        biases = None
        if bias_changed:
            updated = TracerBias(
                b1=current.b1 if b1 is None else b1,
                b2=current.b2 if b2 is None else b2,
                bK2=current.bK2 if bK2 is None else bK2,
            )
            biases = {"galaxy": updated}
        super().update_physics(
            linear_power=linear_power,
            tracer_biases=biases,
        )
        updated = self.tracer_biases["galaxy"]
        self.b1, self.b2, self.bK2 = updated.b1, updated.b2, updated.bK2
        return self
