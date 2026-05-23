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

    def interp_chi(self):
        return InterpolatedUnivariateSpline(self.chi, self.weight, k=min(3, self.chi.size - 1), ext=1)

    @classmethod
    def delta_like(cls, z: float, chi: float, width: float | None = None, power: int = 3):
        """Return a narrow top-hat kernel representing a fixed-chi delta.

        The returned kernel is normalized so that, if the same kernel is used
        ``power`` times in a :class:`KernelSet.product`, then

            int dchi W(chi)**power = 1

        under trapezoidal integration.  For bispectra, ``power=3`` is the
        natural choice because the projected integrand contains a product of
        three line-of-sight kernels.
        """
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
        return cls(z=z_grid, chi=chi_grid, weight=weight)


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


def _normalize_prefactor(prefactor):
    """Return a callable LOS prefactor.

    ``None`` means the default cosmological/geometrical prefactor.  A scalar
    is accepted for debug projections, e.g. ``prefactor=1.0`` with a
    delta-like kernel.
    """
    if prefactor is None:
        return lambda z, chi: (1.0 + z) ** 3 / chi
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

        return ProjectedBispectrum2D(
            bispectrum3d,
            self,
            sample_combination=_normalize_combo(sample_combination),
            window=window,
        )

    def project_many(self, bispectrum3d: Bispectrum3D, sample_combinations, window=None):
        """Create a collection of projected 2D bispectra.

        The collection evaluates the 3D bispectrum on the LOS grid only once for
        a given ``(ell1, ell2, ell3, params)`` input and reuses that cache for
        all requested sample combinations.
        """
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
        from .multipole import ProjectedBispectrumMultipole2D
        if modes is None and mode_max is not None:
            modes = np.arange(0, int(mode_max) + 1)
        return ProjectedBispectrumMultipole2D(
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


class ProjectedBispectrum2D(Bispectrum2D):
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

    def evaluate(self, ell1, ell2, ell3, **params):
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


class ProjectedBispectrum2DCollection:
    """Collection of projected 2D bispectra sharing a LOS-base cache."""

    def __init__(
        self,
        bispectrum3d: Bispectrum3D,
        projector: LineOfSightProjector,
        sample_combinations,
        window=None,
    ):
        self.bispectrum3d = bispectrum3d
        self.projector = projector
        self.sample_combinations = [_normalize_combo(c) for c in sample_combinations]
        self.window = window
        chi_min = np.nanmin(projector.chi[projector.chi > 0])
        chi_max = np.nanmax(projector.chi)
        sup3 = bispectrum3d.support
        self.support = Support2D(sup3.k_min * chi_min, sup3.k_max * chi_max, policy=sup3.policy)
        self._last_key = None
        self._last_base: BaseLOSIntegrand | None = None

    @staticmethod
    def _array_signature(x):
        arr = np.asarray(x, dtype=float)
        # ascontiguousarray makes tobytes deterministic for views/transposes.
        arr = np.ascontiguousarray(arr)
        return (arr.shape, str(arr.dtype), arr.tobytes())

    @staticmethod
    def _params_signature(params):
        # Params are usually scalars.  repr is intentionally used as a broad,
        # conservative fallback for non-hashable objects.
        return tuple(sorted((k, repr(v)) for k, v in params.items()))

    def _cache_key(self, ell1, ell2, ell3, params):
        return (
            self._array_signature(ell1),
            self._array_signature(ell2),
            self._array_signature(ell3),
            self._params_signature(params),
        )

    def clear_cache(self):
        self._last_key = None
        self._last_base = None

    def get_base(self, ell1, ell2, ell3, **params):
        key = self._cache_key(ell1, ell2, ell3, params)
        if key != self._last_key:
            self._last_base = self.projector.evaluate_base_integrand(
                self.bispectrum3d,
                ell1,
                ell2,
                ell3,
                **params,
            )
            self._last_key = key
        return self._last_base

    def evaluate_one(self, sample_combination, ell1, ell2, ell3, **params):
        combo = _normalize_combo(sample_combination)
        base = self.get_base(ell1, ell2, ell3, **params)
        out = self.projector.integrate_base(base, sample_combination=combo)
        if self.window is not None:
            out = out * self.window(ell1, ell2, ell3)
        return out

    def evaluate(self, ell1, ell2, ell3, **params):
        base = self.get_base(ell1, ell2, ell3, **params)
        out = {
            combo: self.projector.integrate_base(base, sample_combination=combo)
            for combo in self.sample_combinations
        }
        if self.window is not None:
            window = self.window(ell1, ell2, ell3)
            out = {combo: val * window for combo, val in out.items()}
        return out

    def __call__(self, ell1, ell2, ell3, **params):
        return self.evaluate(ell1, ell2, ell3, **params)

    def __getitem__(self, sample_combination):
        combo = _normalize_combo(sample_combination)
        if combo not in self.sample_combinations:
            raise KeyError(f"sample combination {combo!r} is not in this collection")
        return ProjectedBispectrum2DView(self, combo)

    def keys(self):
        return tuple(self.sample_combinations)

    def values(self):
        return tuple(self[combo] for combo in self.sample_combinations)

    def items(self):
        return tuple((combo, self[combo]) for combo in self.sample_combinations)

    def as_list(self):
        return [self[combo] for combo in self.sample_combinations]


class ProjectedBispectrum2DView(Bispectrum2D):
    """Single-combination view backed by a shared collection/cache."""

    def __init__(self, collection: ProjectedBispectrum2DCollection, sample_combination):
        self.collection = collection
        self.sample_combination = _normalize_combo(sample_combination)
        self.support = collection.support
        self.window = collection.window

    def evaluate(self, ell1, ell2, ell3, **params):
        return self.collection.evaluate_one(
            self.sample_combination,
            ell1,
            ell2,
            ell3,
            **params,
        )


# Backward-compatible names used by previous revisions.
ProjectedBispectra2D = ProjectedBispectrum2DCollection
ProjectedBispectrum2DGroup = ProjectedBispectrum2DCollection

# Backward-compatible aliases.  New code should use names without ``Angular``.
ProjectedAngularBispectrum2D = ProjectedBispectrum2D
ProjectedAngularBispectrum2DCollection = ProjectedBispectrum2DCollection
ProjectedAngularBispectrum2DGroup = ProjectedBispectrum2DCollection
ProjectedAngularBispectrum2DView = ProjectedBispectrum2DView
ProjectedAngularBispectra2D = ProjectedBispectra2D
