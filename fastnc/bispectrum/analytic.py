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
    """In-memory cache of FFTLog coefficients ``w_n(z)`` by component."""

    def __init__(self):
        self._coefficients: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}

    def get(self, component: FFTLogComponent, z):
        key = (id(component), float(np.asarray(z)))
        cached = self._coefficients.get(key)
        if cached is None:
            values = np.asarray(component.evaluator(component.k_grid, z), dtype=float)
            coeff, nu = power_law_fftlog_coefficients(component.k_grid, values, component.fftlog_config)
            cached = (np.asarray(coeff, dtype=complex), np.asarray(nu, dtype=complex))
            self._coefficients[key] = cached
        return cached

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

    def warm_cache(self, *, z, modes, shifts=None):
        """Precompute FFTLog coefficients and requested angular-kernel tables.

        Parameters
        ----------
        z
            Redshift at which FFTLog coefficients are required.
        modes
            Fourier modes to precompute.
        shifts
            Optional iterable of integer kernel shifts.  If omitted, the
            shifts required by this model's separable terms are used.
        """
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        required = (
            {int(term.p) for term in self.terms if isinstance(term, SeparableMultipoleTerm)}
            if shifts is None else {int(shift) for shift in shifts}
        )
        for component in self.components:
            self.coefficient_cache.get(component, z)
            table = self._ensure_kernel_table(component)
            for shift in required:
                for mode in modes:
                    table._build(int(mode), int(shift))

    def evaluate(self, mode, k1, k2, z, **params):
        values = self.evaluate_modes(np.atleast_1d(np.asarray(mode, dtype=int)), k1, k2, z, **params)
        return values[0] if np.isscalar(mode) else values

    def evaluate_modes(self, modes, k1, k2, z, *, chunk_size=4096, **params):
        """Evaluate many Fourier modes while sharing FFTLog contractions.

        The expensive contraction
        ``sum_n w_n k**nu_n K_L^(nu_n+p)`` is evaluated once for each
        unique ``(FFTLog component, p)`` pair.  All terms referring to that
        pair subsequently differ only by their inexpensive ``U*V`` prefactor.

        Parameters
        ----------
        modes
            Integer Fourier modes.
        k1, k2
            Broadcast-compatible Fourier-mode arrays.
        z
            Redshift.
        chunk_size
            Number of flattened ``(k1,k2)`` points processed at once.  Set to
            ``None`` to process the entire input in one contraction.
        """
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
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
            print(unique_r.shape)
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
