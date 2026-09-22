from __future__ import annotations

import numpy as np

from ..multipole import BispectrumMultipole2D
from ..support import Support3D
from .terms import DirectFourierTerm, LowRankVFunction, SeparableMultipoleTerm, _PROJECTED_STATE_UNSET, _term_weight_value

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
        self.integration_weights = self._trapezoid_weights(self.chi)
        # Persistent tensor-product geometry cache.  This survives repeated
        # evaluations at fixed angular axes and is independent of the physical
        # state of the underlying 3D model.
        self._grid_cache = {}
        self.sample_combination = (
            tuple(sample_combination) if sample_combination is not None else None
        )
        if getattr(projector, "l_shift", 0.0) != 0.0:
            raise NotImplementedError(
                "Coefficient-level semi-analytic projection requires l_shift=0"
            )

    @staticmethod
    def _trapezoid_weights(x):
        """Return weights for composite trapezoidal integration on ``x``."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError(
                "LOS chi grid must be one-dimensional with at least two points"
            )
        dx = np.diff(x)
        if np.any(dx <= 0.0):
            raise ValueError("LOS chi grid must be strictly increasing")
        weights = np.empty_like(x)
        weights[0] = 0.5 * dx[0]
        weights[-1] = 0.5 * dx[-1]
        if x.size > 2:
            weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        return weights

    @staticmethod
    def _axis_key(axis):
        axis = np.ascontiguousarray(np.asarray(axis, dtype=float))
        return axis.shape, axis.dtype.str, axis.tobytes()

    def prepare_grid(self, ell2_axis, ell3_axis, modes=None):
        """Build or retrieve cached tensor-product angular geometry."""
        ell2_axis = np.asarray(ell2_axis, dtype=float)
        ell3_axis = np.asarray(ell3_axis, dtype=float)
        if ell2_axis.ndim != 1 or ell3_axis.ndim != 1:
            raise ValueError("ell2_axis and ell3_axis must be one-dimensional")
        if np.any(ell2_axis <= 0.0) or np.any(ell3_axis <= 0.0):
            raise ValueError("ell2_axis and ell3_axis must be strictly positive")

        key = (self._axis_key(ell2_axis), self._axis_key(ell3_axis))
        geometry = self._grid_cache.get(key)
        if geometry is None:
            ell2, ell3 = np.meshgrid(ell2_axis, ell3_axis, indexing="ij")
            ell = np.hypot(ell2, ell3)
            ratio = np.minimum(ell2, ell3) / ell
            unique_ratio, inverse = np.unique(ratio.ravel(), return_inverse=True)
            geometry = {
                "ell2_axis": ell2_axis,
                "ell3_axis": ell3_axis,
                "ell2": ell2,
                "ell3": ell3,
                "ell": ell,
                "x2": ell2 / ell,
                "x3": ell3 / ell,
                "ratio": ratio,
                "unique_ratio": unique_ratio,
                "ratio_inverse": inverse,
                "ell_power": {},
                "kernels": {},
            }
            self._grid_cache[key] = geometry
        return geometry

    def clear_grid_cache(self):
        """Discard cached tensor-product angular geometry and contractions."""
        self._grid_cache.clear()
        return self

    @staticmethod
    def _component_key(component):
        return id(component)

    def _cached_ell_power(self, geometry, component, nu):
        nu = np.asarray(nu)
        key = (self._component_key(component), nu.dtype.str, nu.tobytes())
        value = geometry["ell_power"].get(key)
        if value is None:
            value = geometry["ell"].ravel()[:, None] ** nu[None, :]
            geometry["ell_power"][key] = value
        return value

    def _cached_kernels(self, geometry, table, component, shift, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        key = (self._component_key(component), int(shift), tuple(map(int, modes)))
        value = geometry["kernels"].get(key)
        if value is None:
            value = table.evaluate_modes(
                modes,
                geometry["ratio"].ravel(),
                shift=int(shift),
                unique=True,
            )
            geometry["kernels"][key] = value
        return value

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
                    weighted_value = (
                        los_weight * term_weight * value
                    )
                    result[i_mode] += term.amplitude * (
                        weighted_value @ self.integration_weights
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
            # Apply the composite-trapezoid weights before contraction and
            # evaluate all FFTLog coefficients with one matrix product:
            #
            #   d[p, n] = sum_j q[j] W[p, j] V[p, j]
            #                     w[j, n] chi[j]**(-nu[n]).
            #
            # This is algebraically identical to ``np.trapezoid`` but avoids
            # materializing an (n_point, n_chi, n_nu) temporary array.
            projected_prefactor = (
                los_weight * term_weight * v
            ) * self.integration_weights[None, :]
            coefficient_matrix = coeff * chi_power
            d_n = projected_prefactor @ coefficient_matrix
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

    @staticmethod
    def _as_factor_values(value, rank, n_axis, n_chi, side):
        value = np.asarray(value)
        try:
            return np.broadcast_to(value, (rank, n_axis, n_chi))
        except ValueError as error:
            raise ValueError(
                f"A low-rank V {side} factor must broadcast to "
                "(rank, n_ell_axis, n_chi)"
            ) from error

    def _project_low_rank_v(self, v, ell2_axis, ell3_axis, coefficient_matrix):
        """Project an explicitly low-rank V without constructing V(k2,k3,z)."""
        n2 = ell2_axis.size
        n3 = ell3_axis.size
        n_chi, n_nu = coefficient_matrix.shape
        z = self.z[None, :]
        k2 = ell2_axis[:, None] / self.chi[None, :]
        k3 = ell3_axis[:, None] / self.chi[None, :]
        left = self._as_factor_values(
            v.evaluate_left(k2, z), v.rank, n2, n_chi, "left"
        )
        right = self._as_factor_values(
            v.evaluate_right(k3, z), v.rank, n3, n_chi, "right"
        )

        # For each FFTLog coefficient n,
        #
        #   d_n[i,j] = sum_a (left[a] * C[:,n]) @ right[a].T.
        #
        # This uses small dense matrix products and never materializes the
        # full (n_ell2, n_ell3, n_chi) V array.
        dtype = np.result_type(left, right, coefficient_matrix, complex)
        d_n = np.zeros((n2, n3, n_nu), dtype=dtype)
        for i_n in range(n_nu):
            weighted_left = left * coefficient_matrix[None, None, :, i_n]
            for i_rank in range(v.rank):
                d_n[:, :, i_n] += weighted_left[i_rank] @ right[i_rank].T
        return d_n

    def evaluate_grid(
        self,
        multipole3d,
        mode,
        ell2_axis,
        ell3_axis,
        *,
        geometry=None,
    ):
        """Evaluate on a tensor-product grid, using low-rank V when declared.

        Parameters
        ----------
        geometry : dict, optional
            Geometry returned by :meth:`prepare_grid` for the same angular
            axes.  Supplying it avoids a redundant cache lookup and preserves
            the common ``prepare_grid``/``evaluate_modes_grid`` contract.
        """
        modes = np.atleast_1d(np.asarray(mode, dtype=int))
        scalar_mode = np.isscalar(mode)
        ell2_axis = np.asarray(ell2_axis, dtype=float)
        ell3_axis = np.asarray(ell3_axis, dtype=float)
        if ell2_axis.ndim != 1 or ell3_axis.ndim != 1:
            raise ValueError("ell2_axis and ell3_axis must be one-dimensional")
        if np.any(ell2_axis <= 0.0) or np.any(ell3_axis <= 0.0):
            raise ValueError("ell2_axis and ell3_axis must be strictly positive")

        if geometry is None:
            geometry = self.prepare_grid(ell2_axis, ell3_axis, modes=modes)
        else:
            if not isinstance(geometry, dict):
                raise TypeError("geometry must be returned by prepare_grid()")
            try:
                prepared_ell2_axis = np.asarray(
                    geometry["ell2_axis"], dtype=float
                )
                prepared_ell3_axis = np.asarray(
                    geometry["ell3_axis"], dtype=float
                )
            except KeyError as error:
                raise ValueError(
                    "geometry is missing tensor-product axis metadata"
                ) from error
            if (
                prepared_ell2_axis.shape != ell2_axis.shape
                or prepared_ell3_axis.shape != ell3_axis.shape
                or not np.array_equal(prepared_ell2_axis, ell2_axis)
                or not np.array_equal(prepared_ell3_axis, ell3_axis)
            ):
                raise ValueError(
                    "geometry was prepared for different ell2/ell3 axes"
                )

        ell2 = geometry["ell2"]
        ell3 = geometry["ell3"]
        shape = ell2.shape
        e1 = ell2.ravel()
        e2 = ell3.ravel()
        n_point = e1.size
        n_chi = self.chi.size
        ell = geometry["ell"].ravel()
        x2 = geometry["x2"].ravel()
        x3 = geometry["x3"].ravel()
        result = np.zeros((modes.size, n_point), dtype=complex)

        # Generic flattened LOS arrays are constructed lazily, only when a
        # direct or non-low-rank term requires them.
        generic_arrays = None

        def get_generic_arrays():
            nonlocal generic_arrays
            if generic_arrays is None:
                generic_arrays = (
                    e1[:, None] / self.chi[None, :],
                    e2[:, None] / self.chi[None, :],
                    self.z[None, :],
                )
            return generic_arrays

        los_weight = self.weight[None, :]
        for term in multipole3d.terms:
            if isinstance(term, DirectFourierTerm):
                k2, k3, z = get_generic_arrays()
                for i_mode, requested_mode in enumerate(modes):
                    value = self._as_los_values(
                        term.coefficient(int(requested_mode), k2, k3, z),
                        n_point, n_chi,
                    )
                    term_weight = self._as_los_values(
                        _term_weight_value(term.weight, z), n_point, n_chi
                    )
                    result[i_mode] += term.amplitude * (
                        (los_weight * term_weight * value) @ self.integration_weights
                    )
                continue

            if not isinstance(term, SeparableMultipoleTerm):
                raise TypeError(
                    f"Unsupported semi-analytic term type: {type(term).__name__}"
                )

            coeff, nu = multipole3d.coefficient_cache.get_many(term.component, self.z)
            table = multipole3d._ensure_kernel_table(term.component)
            kernels = self._cached_kernels(
                geometry, table, term.component, int(term.p), modes
            )
            chi_power = self.chi[:, None] ** (-nu[None, :])

            # A term weight is defined as a scalar or weight(z), so on the
            # tensor-product path it is a one-dimensional LOS quantity.
            term_weight = np.asarray(_term_weight_value(term.weight, self.z))
            try:
                term_weight = np.broadcast_to(term_weight, (n_chi,))
            except ValueError as error:
                raise ValueError(
                    "A term weight used by evaluate_modes_grid must broadcast "
                    "to the one-dimensional LOS grid"
                ) from error
            coefficient_matrix = (
                coeff * chi_power
                * (self.weight * term_weight * self.integration_weights)[:, None]
            )

            if isinstance(term.v, LowRankVFunction):
                d_grid = self._project_low_rank_v(
                    term.v, ell2_axis, ell3_axis, coefficient_matrix
                )
                d_n = d_grid.reshape((n_point, nu.size))
            else:
                k2, k3, z = get_generic_arrays()
                v = self._as_los_values(term.v(k2, k3, z), n_point, n_chi)
                d_n = v @ coefficient_matrix

            ell_power = self._cached_ell_power(geometry, term.component, nu)
            core = np.einsum(
                "pn,pn,lnp->lp", d_n, ell_power, kernels, optimize=True
            )
            prefactor = term.amplitude * np.asarray(term.u(x2, x3))
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
        """Return reusable tensor-product geometry for the angular axes."""
        return self.projector.prepare_grid(ell2_axis, ell3_axis, modes=self.modes)

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
        if geometry is None:
            geometry = self.prepare_grid(ell2_axis, ell3_axis)
        return self.projector.evaluate_grid(
            self.multipole3d,
            modes,
            ell2_axis,
            ell3_axis,
            geometry=geometry,
        )

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


