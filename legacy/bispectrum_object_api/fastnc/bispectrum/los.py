"""Line-of-sight projection from 3D to 2D bispectrum.

The 3D bispectrum part of the projection is independent of source/sample
combination. ``LineOfSightProjector`` therefore separates the computation into

    base = projector.evaluate_base_integrand(b3d, ell1, ell2, ell3)
    Babc = projector.integrate_base(base, sample_combination=("a", "b", "c"))

For multiple sample combinations, use the same ``project`` method with a list of
sample-combination tuples.  It returns a list of 2D-bispectrum views sharing
one internal group/cache, so evaluating those views on the same 2D grid can
reuse the same 3D bispectrum LOS-grid evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .base import Bispectrum2D, Bispectrum3D
from .support import Support2D


@dataclass
class Kernel1D:
    z: np.ndarray
    chi: np.ndarray
    weight: np.ndarray
    name: str | None = None

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)
        self.chi = np.asarray(self.chi, dtype=float)
        self.weight = np.asarray(self.weight, dtype=float)
        if self.z.shape != self.chi.shape or self.chi.shape != self.weight.shape:
            raise ValueError("z, chi, and weight must have the same shape")
        if self.chi.ndim != 1:
            raise ValueError("Kernel1D arrays must be one-dimensional")
        if self.chi.size < 2:
            raise ValueError("Kernel1D needs at least two chi samples for spline/trapezoid integration")
        if np.any(~np.isfinite(self.z)) or np.any(~np.isfinite(self.chi)) or np.any(~np.isfinite(self.weight)):
            raise ValueError("z, chi, and weight must be finite")
        if np.any(np.diff(self.chi) <= 0.0):
            order = np.argsort(self.chi)
            self.z = self.z[order]
            self.chi = self.chi[order]
            self.weight = self.weight[order]
            if np.any(np.diff(self.chi) <= 0.0):
                raise ValueError("chi samples must be strictly increasing")

    def copy(self, *, weight=None, name=None):
        return type(self)(
            z=self.z.copy(),
            chi=self.chi.copy(),
            weight=self.weight.copy() if weight is None else np.asarray(weight, dtype=float),
            name=self.name if name is None else name,
        )

    def interp_chi(self):
        return InterpolatedUnivariateSpline(self.chi, self.weight, k=min(3, self.chi.size - 1), ext=1)

    def evaluate(self, chi):
        return self.interp_chi()(chi)

    def resample(self, z, chi, *, name=None):
        chi = np.asarray(chi, dtype=float)
        z = np.asarray(z, dtype=float)
        if z.shape != chi.shape:
            raise ValueError("z and chi must have the same shape")
        return type(self)(z=z, chi=chi, weight=self.evaluate(chi), name=self.name if name is None else name)

    def resample_like(self, other: "Kernel1D", *, name=None):
        return self.resample(other.z, other.chi, name=name)

    def integral(self):
        return np.trapezoid(self.weight, self.chi)

    def normalized(self, *, integral: float = 1.0, name=None):
        current = self.integral()
        if current == 0.0:
            raise ValueError("cannot normalize a kernel with zero integral")
        return self.copy(weight=self.weight * (float(integral) / current), name=self.name if name is None else name)

    def _binary_kernel_op(self, other, op, symbol: str):
        if np.isscalar(other):
            return self.copy(weight=op(self.weight, float(other)), name=self.name)
        if not isinstance(other, Kernel1D):
            return NotImplemented
        other_weight = other.weight if np.array_equal(self.chi, other.chi) else other.evaluate(self.chi)
        name = None
        if self.name is not None or other.name is not None:
            name = f"({self.name or 'kernel'}{symbol}{other.name or 'kernel'})"
        return type(self)(z=self.z, chi=self.chi, weight=op(self.weight, other_weight), name=name)

    def __add__(self, other):
        return self._binary_kernel_op(other, np.add, "+")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_kernel_op(other, np.subtract, "-")

    def __rsub__(self, other):
        if np.isscalar(other):
            return self.copy(weight=float(other) - self.weight, name=self.name)
        if isinstance(other, Kernel1D):
            return other.__sub__(self)
        return NotImplemented

    def __mul__(self, other):
        return self._binary_kernel_op(other, np.multiply, "*")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_kernel_op(other, np.divide, "/")

    def __neg__(self):
        return self.copy(weight=-self.weight, name=None if self.name is None else f"(-{self.name})")

    @staticmethod
    def _validate_z_grid(z, name: str = "z"):
        z = np.asarray(z, dtype=float)
        if z.ndim != 1 or z.size < 2:
            raise ValueError(f"{name} must be a one-dimensional array with at least two samples")
        if np.any(~np.isfinite(z)):
            raise ValueError(f"{name} must be finite")
        if np.any(np.diff(z) <= 0.0):
            raise ValueError(f"{name} samples must be strictly increasing")
        return z

    @staticmethod
    def _normalized_nz(z_nz, nz):
        z_nz = Kernel1D._validate_z_grid(z_nz, "z_nz")
        nz = np.asarray(nz, dtype=float)
        if z_nz.shape != nz.shape:
            raise ValueError("z_nz and nz must have the same shape")
        if np.any(~np.isfinite(nz)):
            raise ValueError("nz must be finite")
        norm = np.trapezoid(nz, z_nz)
        if norm <= 0.0:
            raise ValueError("nz must have a positive integral over z_nz")
        return nz / norm

    @staticmethod
    def _parse_nz_args(z_kernel, args, z_nz, method_name: str):
        if len(args) == 1:
            nz = args[0]
            if z_nz is None:
                z_nz = z_kernel
        elif len(args) == 2:
            if z_nz is not None:
                raise TypeError(f"{method_name} received both positional z_nz and keyword z_nz")
            z_nz, nz = args
        else:
            raise TypeError(
                f"{method_name} expects either (z_kernel, chi_kernel, nz) "
                f"or (z_kernel, chi_kernel, z_nz, nz)"
            )
        return np.asarray(z_nz, dtype=float), np.asarray(nz, dtype=float)

    @staticmethod
    def _dz_dchi(z, chi):
        return np.gradient(np.asarray(z, dtype=float), np.asarray(chi, dtype=float), edge_order=1)

    @staticmethod
    def _interp_nz_to_kernel_z(z_kernel, z_nz, nz):
        z_kernel = np.asarray(z_kernel, dtype=float)
        z_nz = Kernel1D._validate_z_grid(z_nz, "z_nz")
        nz = np.asarray(nz, dtype=float)
        if z_nz.shape != nz.shape:
            raise ValueError("z_nz and nz must have the same shape")
        spline = InterpolatedUnivariateSpline(z_nz, nz, k=min(3, z_nz.size - 1), ext=1)
        return spline(z_kernel)

    @staticmethod
    def _chi_at_source_z(z_kernel, chi_kernel, z_nz):
        z_kernel = Kernel1D._validate_z_grid(z_kernel, "z_kernel")
        chi_kernel = np.asarray(chi_kernel, dtype=float)
        z_nz = Kernel1D._validate_z_grid(z_nz, "z_nz")
        if z_kernel.shape != chi_kernel.shape:
            raise ValueError("z_kernel and chi_kernel must have the same shape")
        spline = InterpolatedUnivariateSpline(z_kernel, chi_kernel, k=min(3, z_kernel.size - 1), ext=1)
        return spline(z_nz)

    @classmethod
    def from_nz(
        cls,
        z,
        chi,
        *args,
        z_nz=None,
        normalize: bool = True,
        name: str | None = None,
    ):
        """Return n(chi) on the kernel grid.

        Accepted forms are ``from_nz(z, chi, nz)`` and
        ``from_nz(z_kernel, chi_kernel, z_nz, nz)``.  The keyword form
        ``from_nz(z_kernel, chi_kernel, nz, z_nz=z_nz)`` is also accepted.
        """
        z = cls._validate_z_grid(z, "z_kernel")
        chi = np.asarray(chi, dtype=float)
        z_nz, nz = cls._parse_nz_args(z, args, z_nz, "from_nz")
        nz = cls._normalized_nz(z_nz, nz) if normalize else np.asarray(nz, dtype=float)
        nz_on_kernel = cls._interp_nz_to_kernel_z(z, z_nz, nz)
        n_chi = nz_on_kernel * cls._dz_dchi(z, chi)
        return cls(z=z, chi=chi, weight=n_chi, name=name)

    @classmethod
    def lensing_from_nz(
        cls,
        z,
        chi,
        *args,
        z_nz=None,
        omega_m: float = 0.3,
        h0_over_c: float = 100.0 / 299792.458,
        normalize_nz: bool = True,
        name: str | None = None,
    ):
        """Return the weak-lensing efficiency kernel on the kernel grid."""
        z = cls._validate_z_grid(z, "z_kernel")
        chi = np.asarray(chi, dtype=float)
        z_nz, nz = cls._parse_nz_args(z, args, z_nz, "lensing_from_nz")
        nz = cls._normalized_nz(z_nz, nz) if normalize_nz else np.asarray(nz, dtype=float)
        if z.shape != chi.shape:
            raise ValueError("z_kernel and chi_kernel must have the same shape")
        if np.any(chi <= 0.0):
            raise ValueError("chi_kernel must be positive")

        chi_s = cls._chi_at_source_z(z, chi, z_nz)
        valid = chi_s > 0.0
        if not np.all(valid):
            # ``ext=1`` gives zero outside the interpolation range.  Those
            # points should not contribute to the source integral.
            chi_s = chi_s[valid]
            z_src = z_nz[valid]
            nz_src = nz[valid]
        else:
            z_src = z_nz
            nz_src = nz

        if chi_s.size < 2:
            raise ValueError("source z_nz grid has too few points inside the kernel z range")

        geom = np.maximum(chi_s[None, :] - chi[:, None], 0.0) / chi_s[None, :]
        source_integral = np.trapezoid(nz_src[None, :] * geom, z_src, axis=1)
        pref = 1.5 * float(omega_m) * float(h0_over_c) ** 2
        weight = pref * chi * (1.0 + z) * source_integral
        return cls(z=z, chi=chi, weight=weight, name=name)

    @classmethod
    def nla_from_nz(
        cls,
        z,
        chi,
        *args,
        z_nz=None,
        amplitude: float = 1.0,
        omega_m: float = 0.3,
        c1rho_crit: float = 0.0134,
        growth=None,
        eta: float = 0.0,
        z0: float = 0.62,
        normalize_nz: bool = True,
        name: str | None = None,
    ):
        """Return an NLA intrinsic-alignment kernel on the kernel grid."""
        z = cls._validate_z_grid(z, "z_kernel")
        chi = np.asarray(chi, dtype=float)
        z_nz, nz = cls._parse_nz_args(z, args, z_nz, "nla_from_nz")
        nz = cls._normalized_nz(z_nz, nz) if normalize_nz else np.asarray(nz, dtype=float)
        if z.shape != chi.shape:
            raise ValueError("z_kernel and chi_kernel must have the same shape")

        nz_on_kernel = cls._interp_nz_to_kernel_z(z, z_nz, nz)
        n_chi = nz_on_kernel * cls._dz_dchi(z, chi)
        if growth is None:
            Dz = np.ones_like(z)
        elif callable(growth):
            Dz = np.asarray(growth(z), dtype=float)
        else:
            Dz = np.asarray(growth, dtype=float)
        if Dz.shape != z.shape:
            raise ValueError("growth must be callable on z_kernel or have the same shape as z_kernel")
        if np.any(Dz == 0.0):
            raise ValueError("growth must be non-zero")
        redshift_scaling = ((1.0 + z) / (1.0 + float(z0))) ** float(eta)
        weight = -float(amplitude) * float(c1rho_crit) * float(omega_m) * redshift_scaling * n_chi / Dz
        return cls(z=z, chi=chi, weight=weight, name=name)

    @classmethod
    def lensing_plus_nla(cls, z, chi, *args, z_nz=None, name: str | None = None, **kwargs):
        """Return W_G + W_IA on the kernel grid."""
        lensing_kwargs = dict(kwargs.pop("lensing", {}))
        nla_kwargs = dict(kwargs.pop("nla", {}))
        if kwargs:
            raise TypeError(f"unexpected keyword(s): {', '.join(kwargs)}")
        z_nz, nz = cls._parse_nz_args(np.asarray(z, dtype=float), args, z_nz, "lensing_plus_nla")
        wg = cls.lensing_from_nz(z, chi, z_nz, nz, **lensing_kwargs)
        wi = cls.nla_from_nz(z, chi, z_nz, nz, **nla_kwargs)
        return (wg + wi).copy(name=name)

    @classmethod
    def delta_like(cls, z: float, chi: float, width: float | None = None, power: int = 3, name: str | None = None):
        """Return a narrow top-hat kernel representing a fixed-chi delta."""
        z = float(z)
        chi = float(chi)
        if chi <= 0.0:
            raise ValueError("chi must be positive")
        if power <= 0:
            raise ValueError("power must be positive")
        if width is None:
            width = max(abs(chi) * 1.0e-6, 1.0e-8)
        width = float(width)
        if width <= 0.0:
            raise ValueError("width must be positive")
        chi_grid = np.array([chi - 0.5 * width, chi + 0.5 * width], dtype=float)
        if np.any(chi_grid <= 0.0):
            chi_grid = np.array([chi, chi + width], dtype=float)
        z_grid = np.array([z, z], dtype=float)
        weight = np.full(2, width ** (-1.0 / float(power)), dtype=float)
        return cls(z=z_grid, chi=chi_grid, weight=weight, name=name)


class KernelSet:
    """Container for named line-of-sight kernels."""

    def __init__(self, kernels: Mapping[str, Kernel1D]):
        self.kernels = dict(kernels)
        self._splines = {name: ker.interp_chi() for name, ker in self.kernels.items()}

    @classmethod
    def delta_like(cls, name: str = "delta", *, z: float, chi: float, width: float | None = None, power: int = 3):
        """Return a KernelSet containing one fixed-chi delta-like kernel."""
        return cls({name: Kernel1D.delta_like(z=z, chi=chi, width=width, power=power)})

    def names(self):
        return tuple(self.kernels.keys())

    def product(self, sample_combination, chi):
        out = np.ones_like(chi, dtype=float)
        for name in sample_combination:
            out *= self._splines[name](chi)
        return out


@dataclass(frozen=True)
class BaseLOSIntegrand:
    """Sample-independent LOS integrand for 2D bispectrum projection.

    Attributes
    ----------
    values:
        Array with shape ``(n_eval, n_chi)`` containing
        ``prefactor(z, chi) * B_3D(k1, k2, k3; z)``.
    shape:
        Original 2D-input shape before flattening.
    scalar:
        Whether the original angular input was scalar.
    """

    values: np.ndarray
    shape: tuple[int, ...]
    scalar: bool = False


def _is_sequence_of_sample_combinations(sample_combination) -> bool:
    """Return True for list/tuple-like containers of sample-combination tuples.

    A single sample combination is expected to look like ``("src0", "src1",
    "src1")``.  A multiple-combination input looks like ``[(...), (...)]``.
    Strings are deliberately not treated as sequences here.
    """
    if sample_combination is None:
        return False
    if not isinstance(sample_combination, (list, tuple)):
        return False
    if len(sample_combination) == 0:
        return True
    first = sample_combination[0]
    if first is None:
        return True
    return isinstance(first, (list, tuple)) and not isinstance(first, str)


def _normalize_combo(combo):
    if combo is None:
        return None
    return tuple(combo)


_PROJECTOR_STATE_UNSET = object()


def _normalize_prefactor(prefactor):
    """Return a callable LOS prefactor.

    ``None`` means the default cosmological/geometrical prefactor.  A scalar
    is accepted for debug projections, e.g. ``prefactor=1.0`` with a
    delta-like kernel.
    """
    if prefactor is None:
        return lambda z, chi: chi**-4
    if callable(prefactor):
        return prefactor
    value = float(prefactor)
    return lambda z, chi, value=value: np.full_like(chi, value, dtype=float)


class LOSProjectorBase:
    """Shared line-of-sight integration machinery.

    This base class manages the redshift/chi grid, kernel products, projection
    prefactor, and the final ``dchi`` integration.  Subclasses define how the
    3D object is evaluated on that LOS grid.
    """

    def __init__(
        self,
        z,
        chi,
        kernels: Optional[KernelSet] = None,
        prefactor: Optional[Callable] = None,
        l_shift: float = 0.0,
        support_policy: str = "zero",
    ):
        self.z = np.asarray(z, dtype=float)
        self.chi = np.asarray(chi, dtype=float)
        if self.z.shape != self.chi.shape:
            raise ValueError("z and chi must have the same shape")
        self.kernels = kernels
        self.prefactor = _normalize_prefactor(prefactor)
        self.l_shift = float(l_shift)
        self.support_policy = support_policy

    def update_state(
        self,
        *,
        z=_PROJECTOR_STATE_UNSET,
        chi=_PROJECTOR_STATE_UNSET,
        kernels=_PROJECTOR_STATE_UNSET,
        prefactor=_PROJECTOR_STATE_UNSET,
        l_shift=_PROJECTOR_STATE_UNSET,
        support_policy=_PROJECTOR_STATE_UNSET,
    ):
        """Update the LOS-projection state in place.

        Omitted arguments retain their current values.  Passing ``None`` for
        ``kernels`` removes sample-dependent kernels, while passing ``None``
        for ``prefactor`` restores the default geometrical prefactor.

        Existing projected objects that copied this projector state must call
        their ``update_projection()`` method to refresh that snapshot.
        """
        new_z = self.z if z is _PROJECTOR_STATE_UNSET else np.asarray(z, dtype=float)
        new_chi = self.chi if chi is _PROJECTOR_STATE_UNSET else np.asarray(chi, dtype=float)
        if new_z.shape != new_chi.shape:
            raise ValueError("z and chi must have the same shape")

        new_l_shift = self.l_shift if l_shift is _PROJECTOR_STATE_UNSET else float(l_shift)
        if isinstance(self, MultipoleLineOfSightProjector) and new_l_shift != 0.0:
            raise NotImplementedError(
                "Multipole LOS projection currently supports only l_shift=0"
            )

        self.z = new_z
        self.chi = new_chi
        if kernels is not _PROJECTOR_STATE_UNSET:
            self.kernels = kernels
        if prefactor is not _PROJECTOR_STATE_UNSET:
            self.prefactor = _normalize_prefactor(prefactor)
        self.l_shift = new_l_shift
        if support_policy is not _PROJECTOR_STATE_UNSET:
            self.support_policy = support_policy
        return self

    def kernel_product(self, sample_combination=None):
        if self.kernels is not None and sample_combination is not None:
            return self.kernels.product(sample_combination, self.chi)
        return np.ones_like(self.chi, dtype=float)

    def los_weight(self, sample_combination=None):
        return self.prefactor(self.z, self.chi) * self.kernel_product(sample_combination)

    def integrate_base(self, base: BaseLOSIntegrand | np.ndarray, sample_combination=None, shape=None, scalar=False):
        if isinstance(base, BaseLOSIntegrand):
            values = base.values
            shape = base.shape
            scalar = base.scalar
        else:
            values = np.asarray(base)
            if shape is None:
                raise ValueError("shape is required when base is passed as an array")
        kernel = self.kernel_product(sample_combination)
        integrand = values * kernel[None, :]
        out = np.trapezoid(integrand, self.chi, axis=1).reshape(shape)
        return out.item() if scalar else out


class LineOfSightProjector(LOSProjectorBase):
    """Project a 3D bispectrum to an 2D bispectrum.

    The default projection is

        B_2D = int dchi W1(chi) W2(chi) W3(chi) (1+z)^3 / chi
               B_3D((ell1+l_shift)/chi, ...; z).

    The factor ``(1+z)^3 / chi`` is sample-independent and is included in the
    base integrand.  Source/sample kernels are applied only in the final LOS
    integration step, so a single 3D bispectrum evaluation can be reused across
    multiple sample combinations.
    """

    def __init__(
        self,
        z,
        chi,
        kernels: Optional[KernelSet] = None,
        prefactor: Optional[Callable] = None,
        l_shift: float = 0.0,
        support_policy: str = "zero",
    ):
        super().__init__(
            z=z,
            chi=chi,
            kernels=kernels,
            prefactor=prefactor,
            l_shift=l_shift,
            support_policy=support_policy,
        )

    def as_multipole_projector(self):
        """Return a projector with identical LOS settings for 3D multipoles."""
        return MultipoleLineOfSightProjector(
            self.z,
            self.chi,
            kernels=self.kernels,
            prefactor=self.prefactor,
            l_shift=self.l_shift,
            support_policy=self.support_policy,
        )

    def project(self, bispectrum3d: Bispectrum3D, sample_combination=None, window=None):
        """Create a single projected 2D bispectrum.

        ``project`` is intentionally single-combination only.  Use
        :meth:`project_many` when several tomographic/sample combinations should
        share the same LOS-grid cache.
        """
        if _is_sequence_of_sample_combinations(sample_combination):
            raise TypeError(
                "LineOfSightProjector.project expects one sample combination. "
                "Use project_many(bispectrum3d, sample_combinations, ...) for "
                "multiple combinations."
            )

        evaluator = _LoSBispectrum2DEvaluator(
            bispectrum3d=bispectrum3d,
            projector=self,
            sample_combination=_normalize_combo(sample_combination),
            window=window,
        )
        return Bispectrum2D(evaluator=evaluator, support=evaluator.support)

    def project_many(self, bispectrum3d: Bispectrum3D, sample_combinations, window=None):
        """Create a collection of projected 2D bispectra.

        The collection evaluates the 3D bispectrum on the LOS grid only once for
        a given ``(ell1, ell2, ell3, params)`` input and reuses that cache for
        all requested sample combinations.
        """
        from .collections import ProjectedBispectrum2DCollection

        return ProjectedBispectrum2DCollection(
            bispectrum3d,
            self,
            sample_combinations=sample_combinations,
            window=window,
        )

    def kernel_product(self, sample_combination=None):
        """Return only the sample-dependent kernel product."""
        if self.kernels is not None and sample_combination is not None:
            return self.kernels.product(sample_combination, self.chi)
        return np.ones_like(self.chi, dtype=float)

    def los_weight(self, sample_combination=None):
        """Return the full LOS weight, including the sample-independent prefactor."""
        return self.prefactor(self.z, self.chi) * self.kernel_product(sample_combination)

    def _prepare_ell_inputs(self, ell1, ell2, ell3):
        scalar = np.isscalar(ell1)
        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell3 = np.asarray(ell3, dtype=float)
        if ell1.shape != ell2.shape or ell1.shape != ell3.shape:
            raise ValueError("ell1, ell2, and ell3 must have the same shape")

        shape = ell1.shape
        e1 = ell1.ravel()[:, None]
        e2 = ell2.ravel()[:, None]
        e3 = ell3.ravel()[:, None]
        return scalar, shape, e1, e2, e3

    def evaluate_3d_grid(self, bispectrum3d: Bispectrum3D, ell1, ell2, ell3, **params):
        """Evaluate ``B_3D(k1, k2, k3; z)`` on the LOS grid.

        This method performs no LOS integration and applies no source/sample
        kernels.  The returned array has shape ``(n_eval, n_chi)``.  Values
        outside the 3D support are handled according to ``bispectrum3d.support``.
        """
        scalar, shape, e1, e2, e3 = self._prepare_ell_inputs(ell1, ell2, ell3)

        chi = self.chi[None, :]
        z = self.z[None, :]

        k1 = (e1 + self.l_shift) / chi
        k2 = (e2 + self.l_shift) / chi
        k3 = (e3 + self.l_shift) / chi

        # z must be broadcast to the same shape as k1/k2/k3 before boolean
        # indexing.  Otherwise a mask of shape (n_eval, n_chi) cannot be used
        # to index z of shape (1, n_chi).
        z_eval = np.broadcast_to(z, k1.shape)

        mask = bispectrum3d.support.contains(k1, k2, k3, z_eval)
        if bispectrum3d.support.policy == "raise" and not np.all(mask):
            raise ValueError("requested points exceed the 3D bispectrum support")

        if bispectrum3d.support.policy == "ignore":
            b3d = bispectrum3d(k1, k2, k3, z_eval, **params)
        elif bispectrum3d.support.policy == "clip":
            sup = bispectrum3d.support
            b3d = bispectrum3d(
                np.clip(k1, sup.k_min, sup.k_max),
                np.clip(k2, sup.k_min, sup.k_max),
                np.clip(k3, sup.k_min, sup.k_max),
                np.clip(z_eval, sup.z_min, sup.z_max),
                **params,
            )
        else:
            b3d = np.zeros_like(k1, dtype=float)
            if np.any(mask):
                b3d[mask] = bispectrum3d(
                    k1[mask],
                    k2[mask],
                    k3[mask],
                    z_eval[mask],
                    **params,
                )

        return b3d, shape, scalar

    def evaluate_base_integrand(self, bispectrum3d: Bispectrum3D, ell1, ell2, ell3, **params):
        """Evaluate the sample-independent LOS integrand.

        Returns a :class:`BaseLOSIntegrand` whose ``values`` field is

            prefactor(z, chi) * B_3D((ell1+l_shift)/chi, ...; z).

        This object can be passed to :meth:`integrate_base` for any sample
        combination without re-evaluating the 3D bispectrum.
        """
        b3d, shape, scalar = self.evaluate_3d_grid(
            bispectrum3d,
            ell1,
            ell2,
            ell3,
            **params,
        )
        base = b3d * self.prefactor(self.z, self.chi)[None, :]
        return BaseLOSIntegrand(values=base, shape=shape, scalar=scalar)

    def integrate_base(self, base: BaseLOSIntegrand | np.ndarray, sample_combination=None, shape=None, scalar=False):
        """Integrate a sample-independent base integrand for one combination.

        Parameters
        ----------
        base:
            Either a :class:`BaseLOSIntegrand` or an array with shape
            ``(n_eval, n_chi)``.  When an array is passed, ``shape`` must be
            supplied.
        sample_combination:
            Tuple/list of kernel names, e.g. ``("src0", "src1", "src1")``.
        shape:
            Original 2D-input shape.  Required if ``base`` is an array.
        scalar:
            Whether to return a scalar.  Used only when ``base`` is an array.
        """
        if isinstance(base, BaseLOSIntegrand):
            values = base.values
            shape  = base.shape
            scalar = base.scalar
        else:
            values = np.asarray(base)
            if shape is None:
                raise ValueError("shape is required when base is passed as an array")

        kernel = self.kernel_product(sample_combination)
        integrand = values * kernel[None, :]
        out = np.trapezoid(integrand, self.chi, axis=1).reshape(shape)
        return out.item() if scalar else out

    def evaluate(self, bispectrum3d: Bispectrum3D, ell1, ell2, ell3, sample_combination=None, **params):
        base = self.evaluate_base_integrand(bispectrum3d, ell1, ell2, ell3, **params)
        return self.integrate_base(base, sample_combination=sample_combination)

    def evaluate_many(
        self,
        bispectrum3d: Bispectrum3D,
        ell1,
        ell2,
        ell3,
        sample_combinations: Sequence,
        **params,
    ):
        """Evaluate many sample combinations with one 3D bispectrum call.

        Parameters
        ----------
        bispectrum3d:
            3D bispectrum model.
        ell1, ell2, ell3:
            Angular triangle sides.  All must have the same shape.
        sample_combinations:
            Iterable of sample-combination tuples.  Each tuple is passed to
            :meth:`KernelSet.product`.

        Returns
        -------
        dict
            Mapping ``sample_combination -> B_2D``.  The 3D bispectrum is
            evaluated once and reused for all entries.
        """
        combos = [_normalize_combo(c) for c in sample_combinations]
        base = self.evaluate_base_integrand(bispectrum3d, ell1, ell2, ell3, **params)
        return {combo: self.integrate_base(base, sample_combination=combo) for combo in combos}


class MultipoleLineOfSightProjector(LOSProjectorBase):
    """Project ``B_L^3D(k1,k2,z)`` to ``B_L^2D(ell1,ell2)``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.l_shift != 0.0:
            raise NotImplementedError(
                "Multipole LOS projection currently supports only l_shift=0. "
                "Use the ordinary B3D -> B2D -> multipole route for l_shift != 0."
            )

    def project(self, multipole3d, sample_combination=None, modes=None, mode_max=None):
        # Semi-analytic models own their coefficient-level projector.  This
        # dispatch avoids the generic route that evaluates B_L(k1,k2,z) first.
        from .analytic import CompositeSemiAnalyticBispectrumMultipole3D
        if isinstance(multipole3d, CompositeSemiAnalyticBispectrumMultipole3D):
            return multipole3d.project_los(
                self, sample_combination=sample_combination,
                modes=modes, mode_max=mode_max,
            )

        from .multipole import BispectrumMultipole2D
        if modes is None and mode_max is not None:
            mode_max = int(mode_max)
            if getattr(multipole3d, "basis", "fourier-even") == "fourier":
                modes = np.arange(-mode_max, mode_max + 1)
            else:
                modes = np.arange(0, mode_max + 1)
        return BispectrumMultipole2D.from_multipole3d(
            multipole3d,
            self,
            sample_combination=_normalize_combo(sample_combination),
            modes=modes,
        )

    def _prepare_ell_inputs(self, ell1, ell2):
        scalar = np.isscalar(ell1)
        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell1, ell2 = np.broadcast_arrays(ell1, ell2)
        shape = ell1.shape
        e1 = ell1.ravel()[:, None]
        e2 = ell2.ravel()[:, None]
        return scalar, shape, e1, e2

    def evaluate_3d_grid(self, multipole3d, mode, ell1, ell2, **params):
        scalar, shape, e1, e2 = self._prepare_ell_inputs(ell1, ell2)
        chi = self.chi[None, :]
        z = self.z[None, :]
        k1 = e1 / chi
        k2 = e2 / chi
        z_eval = np.broadcast_to(z, k1.shape)
        values = multipole3d(mode, k1, k2, z_eval, **params)
        return values, shape, scalar

    def evaluate_base_integrand(self, multipole3d, mode, ell1, ell2, **params):
        values, shape, scalar = self.evaluate_3d_grid(multipole3d, mode, ell1, ell2, **params)
        base = values * self.prefactor(self.z, self.chi)[None, :]
        return BaseLOSIntegrand(values=base, shape=shape, scalar=scalar)

    def evaluate(self, multipole3d, mode, ell1, ell2, sample_combination=None, **params):
        base = self.evaluate_base_integrand(multipole3d, mode, ell1, ell2, **params)
        return self.integrate_base(base, sample_combination=sample_combination)


class _LoSBispectrum2DEvaluator:
    """Internal evaluator for the 3D-bispectrum -> 2D-bispectrum route."""

    def __init__(
        self,
        bispectrum3d: Bispectrum3D,
        projector: LineOfSightProjector,
        sample_combination=None,
        window=None,
    ):
        self.bispectrum3d = bispectrum3d
        self.projector = projector
        self.sample_combination = _normalize_combo(sample_combination)
        self.window = window
        chi_min = np.nanmin(projector.chi[projector.chi > 0])
        chi_max = np.nanmax(projector.chi)
        sup3 = bispectrum3d.support
        self.support = Support2D(sup3.k_min * chi_min, sup3.k_max * chi_max, policy=sup3.policy)

    def __call__(self, ell1, ell2, ell3, **params):
        out = self.projector.evaluate(
            self.bispectrum3d,
            ell1,
            ell2,
            ell3,
            sample_combination=self.sample_combination,
            **params,
        )
        if self.window is not None:
            out = out * self.window(ell1, ell2, ell3)
        return out
