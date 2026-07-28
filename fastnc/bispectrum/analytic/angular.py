from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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





def _a31(p, x2, x3):
    if p == 2:
        return -5.0 / (28.0 * x3**2)
    if p == 0:
        return (10.0 * x3**2 + 3.0 * x2**2) / (28.0 * x3**2)
    if p == -2:
        return (2.0 * x2**4 + 3.0 * x2**2 * x3**2 - 5.0 * x3**4) / (28.0 * x3**2)
    raise ValueError("tree F2 shifts are p=0,+/-2")




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



