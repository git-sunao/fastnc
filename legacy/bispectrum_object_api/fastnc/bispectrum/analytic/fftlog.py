from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fastnc.hankel.wrapper import PowerLawFFTLogConfig, power_law_fftlog_coefficients

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


