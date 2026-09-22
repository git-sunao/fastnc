from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..multipole import BispectrumMultipole3D
from .angular import PowerLawAngularKernelTable, PowerLawAngularKernelTableConfig, TensorProductGeometryCache
from .fftlog import FFTLogCoefficientCache, FFTLogComponent
from .projection import CompositeSemiAnalyticBispectrumMultipole2D
from .terms import DirectFourierTerm, SemiAnalyticMultipoleTerm, SeparableMultipoleTerm, _term_weight_value



def _safe_prefactor_core_product(prefactor, core):
    """Multiply a scalar-grid prefactor by a mode-grid core safely.

    Negative FFTLog shifts can make the angular core non-finite exactly at
    ``k2 == k3`` even when the analytic prefactor vanishes there.  The full
    term has a removable zero, but ordinary floating-point multiplication
    evaluates it as ``0 * inf -> NaN``.
    """
    prefactor = np.asarray(prefactor)
    core = np.asarray(core)
    contribution = np.zeros_like(core)
    nonzero = prefactor != 0
    np.multiply(
        core,
        prefactor[None, ...],
        out=contribution,
        where=nonzero[None, ...],
    )
    return contribution

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
                        result[:, start:stop] += _safe_prefactor_core_product(
                            prefactor, core
                        )
                    else:
                        active = np.isin(modes, np.asarray(term.modes, dtype=int))
                        result[active, start:stop] += _safe_prefactor_core_product(
                            prefactor, core[active]
                        )

        result = result.reshape((modes.size,) + shape)
        if shape == ():
            return result
        return result


# -----------------------------------------------------------------------------
# Predefined physical model: tree-level matter bispectrum
# -----------------------------------------------------------------------------

