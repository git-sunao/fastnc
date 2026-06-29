"""Appendix-B-style brute-force reference solver for projected spin 3PCFs.

This module evaluates the X-projection three-point correlation function directly
from a scalar-source angular bispectrum ``B(ell1, ell2, ell3)``.  It is an
independent validation path for the fastnc multipole pipeline::

    B(ell1, ell2, ell3)
      -> one-dimensional FFTLog in the common Fourier scale ell
      -> numerical quadrature in (psi, Delta beta).

No bispectrum angular multipoles, spin-coupling matrices, or double-FFTLog
transforms are used here.

The angular quadrature can run in two modes:

``fixed``
    Evaluate once on the configured ``(n_psi, n_delta_beta)`` grid.

``adaptive``
    Compute on successively refined global Fourier-angle grids until the
    real-space result changes by less than the requested absolute/relative
    tolerance.  This is an angular-convergence controller; it deliberately
    makes no parity or exchange-symmetry assumptions about the bispectrum.

The expensive radial FFTLog tasks and, optionally, the real-space
``(theta1, theta2)`` tasks can be distributed with Python's standard
:mod:`multiprocessing` module.
"""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from typing import Any, Callable, Mapping
import warnings

import numpy as np

from ..hankel.fftlog import hankel as _FastncHankel
from .spin import SpinSpec


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class BruteForce3PCFConfig:
    """Numerical settings for :class:`BruteForceX3PCF`.

    Parameters
    ----------
    ell_min, ell_max, n_ell
        Common-scale FFTLog grid.  The radial transform is tabulated on the
        corresponding ``A`` grid, approximately ``[1 / ell_max, 1 / ell_min]``.
    n_psi
        Gauss--Legendre nodes for ``psi in (0, pi/2)``.
    n_delta_beta
        Midpoint trapezoidal nodes for the periodic integral
        ``Delta beta in [0, 2 pi)``.  This deliberately performs no parity or
        exchange reduction.
    angular_mode
        ``"fixed"`` evaluates the configured angular grid once.  ``"adaptive"``
        repeatedly refines the global grid in the selected angular directions,
        comparing the full requested real-space 3PCF after every refinement.
    angular_rtol, angular_atol
        Convergence is accepted when every target configuration satisfies

        ``abs(zeta_new-zeta_old) <= atol + rtol * max(abs(zeta_new), abs(zeta_old))``.

        The result reports the largest normalized difference.
    angular_max_refinements
        Maximum number of grid refinements after the initial grid.
    refine_psi, refine_delta_beta
        Select which Fourier-angle grid dimensions are refined in adaptive
        mode.  At least one must be true when ``angular_mode="adaptive"``.
    adaptive_strict
        If true, failure to reach the requested tolerance raises
        :class:`RuntimeError`.  If false, a result with
        ``angular_converged=False`` is returned after a warning.
    n_processes
        Number of standard-library multiprocessing workers.  ``1`` disables
        process parallelism.  Radial FFTLog tasks are parallelized when
        ``parallel_prepare=True``; real-space theta-pair tasks are parallelized
        when ``parallel_compute=True``.
    mp_start_method
        Optional multiprocessing start method (``"fork"``, ``"spawn"`` or
        ``"forkserver"`` where supported).  With ``"spawn"``, the bispectrum
        model and ``model_kwargs`` must be pickleable.  For a large cached
        radial table, ``"fork"`` is normally most memory-efficient on POSIX.
    """

    ell_min: float = 1.0e-1
    ell_max: float = 1.0e5
    n_ell: int = 256

    n_psi: int = 32
    n_delta_beta: int = 96

    fftlog_nu: float = 1.01
    c_window_width: float = 0.25
    N_extrap_low: int = 0
    N_extrap_high: int = 0
    N_pad: int = 0

    interpolation_bounds: str = "raise"

    angular_mode: str = "fixed"
    angular_rtol: float = 1.0e-3
    angular_atol: float = 0.0
    angular_max_refinements: int = 4
    angular_refinement_factor: int = 2
    refine_psi: bool = True
    refine_delta_beta: bool = True
    adaptive_strict: bool = True

    n_processes: int = 1
    mp_start_method: str | None = None
    mp_chunksize: int = 1
    parallel_prepare: bool = True
    parallel_compute: bool = True

    verbose: bool = False

    def validate(self) -> "BruteForce3PCFConfig":
        if not (0.0 < self.ell_min < self.ell_max):
            raise ValueError("Require 0 < ell_min < ell_max.")
        if int(self.n_ell) < 8 or int(self.n_ell) % 2:
            raise ValueError("n_ell must be an even integer >= 8 for fastnc FFTLog.")
        if int(self.n_psi) < 2:
            raise ValueError("n_psi must be >= 2.")
        if int(self.n_delta_beta) < 4:
            raise ValueError("n_delta_beta must be >= 4.")
        if self.interpolation_bounds not in {"raise", "clip"}:
            raise ValueError("interpolation_bounds must be 'raise' or 'clip'.")
        if self.angular_mode not in {"fixed", "adaptive"}:
            raise ValueError("angular_mode must be 'fixed' or 'adaptive'.")
        if self.angular_rtol < 0.0 or self.angular_atol < 0.0:
            raise ValueError("angular_rtol and angular_atol must be non-negative.")
        if self.angular_rtol == 0.0 and self.angular_atol == 0.0:
            raise ValueError("At least one of angular_rtol or angular_atol must be positive.")
        if int(self.angular_max_refinements) < 0:
            raise ValueError("angular_max_refinements must be >= 0.")
        if int(self.angular_refinement_factor) < 2:
            raise ValueError("angular_refinement_factor must be an integer >= 2.")
        if self.angular_mode == "adaptive" and not (self.refine_psi or self.refine_delta_beta):
            raise ValueError("Adaptive mode requires refine_psi and/or refine_delta_beta.")
        if int(self.n_processes) < 1:
            raise ValueError("n_processes must be >= 1.")
        if int(self.mp_chunksize) < 1:
            raise ValueError("mp_chunksize must be >= 1.")
        if self.mp_start_method is not None:
            allowed = set(mp.get_all_start_methods())
            if self.mp_start_method not in allowed:
                raise ValueError(
                    f"mp_start_method={self.mp_start_method!r} is not available; "
                    f"available methods are {sorted(allowed)}."
                )
        return self


@dataclass(frozen=True)
class BruteForce3PCFResult:
    """Computed X-projection 3PCF and angular-convergence diagnostics."""

    theta1: np.ndarray
    theta2: np.ndarray
    delta_phi: np.ndarray
    value: np.ndarray
    spin: tuple[int, int, int]
    component: int
    epsilon: tuple[int, int, int]
    sigma: tuple[int, int, int]
    q_epsilon: complex
    angular_mode: str
    angular_converged: bool
    angular_refinements: int
    n_psi_used: int
    n_delta_beta_used: int
    angular_error_norm: float | None
    angular_max_abs_change: float | None


@dataclass(frozen=True)
class _RadialWorkerState:
    """Pickleable state required by a radial FFTLog worker."""

    bispectrum: Callable[..., np.ndarray]
    model_kwargs: Mapping[str, Any]
    ell: np.ndarray
    bessel_order: int
    fftlog_nu: float
    c_window_width: float
    N_extrap_low: int
    N_extrap_high: int
    N_pad: int


@dataclass(frozen=True)
class _ComputeWorkerState:
    """Read-only state required by a real-space theta-pair worker."""

    psi: np.ndarray
    psi_weight: np.ndarray
    delta_beta: np.ndarray
    delta_beta_weight: float
    radial_transform: np.ndarray
    log_a_grid: np.ndarray
    a_min: float
    a_max: float
    phase_beta_bar: np.ndarray
    sigma: tuple[int, int, int]
    Sigma: int
    q_epsilon: complex
    interpolation_bounds: str


_RADIAL_WORKER_STATE: _RadialWorkerState | None = None
_COMPUTE_WORKER_STATE: _ComputeWorkerState | None = None


def _phase_beta_bar(psi: float, delta_beta: float) -> complex:
    """Return ``exp(i beta_bar)`` without selecting an angle branch."""
    c = np.cos(psi)
    s = np.sin(psi)
    denom_sq = 1.0 + np.sin(2.0 * psi) * np.cos(delta_beta)
    denom = np.sqrt(max(denom_sq, np.finfo(float).tiny))
    return -(c * np.exp(0.5j * delta_beta) + s * np.exp(-0.5j * delta_beta)) / denom


def _one_dimensional_hankel(
    ell: np.ndarray,
    f_ell: np.ndarray,
    order: int,
    *,
    fftlog_nu: float,
    c_window_width: float,
    N_extrap_low: int,
    N_extrap_high: int,
    N_pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply fastnc's Hankel FFTLog to a possibly complex radial integrand."""
    order = int(order)
    abs_order = abs(order)
    sign = -1.0 if (order < 0 and abs_order % 2) else 1.0

    def transform_real(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fft = _FastncHankel(
            ell,
            np.asarray(value, dtype=float),
            nu=fftlog_nu,
            N_extrap_low=N_extrap_low,
            N_extrap_high=N_extrap_high,
            c_window_width=c_window_width,
            N_pad=N_pad,
        )
        return fft.hankel(abs_order)

    a, real = transform_real(np.real(f_ell))
    if np.any(np.imag(f_ell)):
        a_imag, imag = transform_real(np.imag(f_ell))
        if not np.array_equal(a, a_imag):
            raise RuntimeError("fastnc FFTLog returned inconsistent A grids for real/imaginary parts.")
        transformed = real + 1j * imag
    else:
        transformed = np.asarray(real, dtype=np.complex128)

    return np.asarray(a, dtype=float), sign * np.asarray(transformed, dtype=np.complex128)


def _evaluate_radial_task(
    task: tuple[int, int, float, float],
    state: _RadialWorkerState,
) -> tuple[int, int, complex, np.ndarray, np.ndarray]:
    """Evaluate one angular node's source bispectrum and radial FFTLog."""
    ipsi, idb, psi, delta_beta = task
    ell = state.ell

    ell2 = ell * np.cos(psi)
    ell3 = ell * np.sin(psi)
    ell1_sq = ell2**2 + ell3**2 + 2.0 * ell2 * ell3 * np.cos(delta_beta)
    ell1 = np.sqrt(np.maximum(ell1_sq, 0.0))

    source_b = np.asarray(
        state.bispectrum(ell1, ell2, ell3, **dict(state.model_kwargs)),
        dtype=np.complex128,
    )
    try:
        source_b = np.broadcast_to(source_b, ell.shape).astype(np.complex128, copy=False)
    except ValueError as exc:
        raise ValueError(
            "bispectrum must return a scalar or an array broadcastable to the FFTLog ell-grid shape."
        ) from exc

    if not np.all(np.isfinite(source_b)):
        raise ValueError(
            "bispectrum returned non-finite values on the FFTLog ell grid "
            f"at psi={psi:.6e}, Delta beta={delta_beta:.6e}."
        )

    a_grid, transformed = _one_dimensional_hankel(
        ell,
        ell**4 * source_b,
        state.bessel_order,
        fftlog_nu=state.fftlog_nu,
        c_window_width=state.c_window_width,
        N_extrap_low=state.N_extrap_low,
        N_extrap_high=state.N_extrap_high,
        N_pad=state.N_pad,
    )
    return ipsi, idb, _phase_beta_bar(psi, delta_beta), a_grid, transformed


def _radial_worker_initializer(state: _RadialWorkerState) -> None:
    global _RADIAL_WORKER_STATE
    _RADIAL_WORKER_STATE = state


def _radial_worker(task: tuple[int, int, float, float]) -> tuple[int, int, complex, np.ndarray, np.ndarray]:
    if _RADIAL_WORKER_STATE is None:  # pragma: no cover - defensive
        raise RuntimeError("Radial worker state was not initialized.")
    return _evaluate_radial_task(task, _RADIAL_WORKER_STATE)


def _interpolate_radial_from_state(a: np.ndarray, state: _ComputeWorkerState) -> np.ndarray:
    """Interpolate one cached radial-transform table linearly in ``ln A``."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 3 or a.shape[1:] != (state.psi.size, state.delta_beta.size):
        raise ValueError(
            "A must have shape (n_delta_phi, n_psi, n_delta_beta) matching the cached table."
        )
    if np.any(~np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(
            "Encountered A <= 0. Increase angular resolution or treat the isolated "
            "degenerate geometry separately."
        )

    outside = (a < state.a_min) | (a > state.a_max)
    if np.any(outside):
        found_min = float(np.min(a))
        found_max = float(np.max(a))
        if state.interpolation_bounds == "raise":
            raise ValueError(
                "Required A values fall outside the cached FFTLog grid: "
                f"requested [{found_min:.6e}, {found_max:.6e}], "
                f"available [{state.a_min:.6e}, {state.a_max:.6e}]. "
                "Increase ell_max and/or decrease ell_min."
            )
        a = np.clip(a, state.a_min, state.a_max)

    loga = np.log(a)
    index = np.searchsorted(state.log_a_grid, loga, side="right") - 1
    index = np.clip(index, 0, state.log_a_grid.size - 2)

    x0 = state.log_a_grid[index]
    x1 = state.log_a_grid[index + 1]
    weight = (loga - x0) / (x1 - x0)

    ipsi = np.arange(state.psi.size)[None, :, None]
    idb = np.arange(state.delta_beta.size)[None, None, :]
    y0 = state.radial_transform[ipsi, idb, index]
    y1 = state.radial_transform[ipsi, idb, index + 1]
    return (1.0 - weight) * y0 + weight * y1


def _evaluate_theta_pair_from_state(
    theta1: float,
    theta2: float,
    delta_phi: np.ndarray,
    *,
    delta_phi_chunk: int,
    state: _ComputeWorkerState,
) -> np.ndarray:
    """Evaluate all requested opening angles for one real-space theta pair."""
    psi = state.psi[None, :, None]
    dbeta = state.delta_beta[None, None, :]
    c = np.cos(psi)
    s = np.sin(psi)
    sigma1, sigma2, sigma3 = state.sigma
    out = np.empty(np.asarray(delta_phi).size, dtype=np.complex128)

    angular_weight = state.psi_weight[None, :, None] * np.sin(2.0 * psi)
    prefactor = state.q_epsilon * ((-1j) ** state.Sigma) / (2.0 * (2.0 * np.pi) ** 3)

    for start in range(0, out.size, int(delta_phi_chunk)):
        stop = min(start + int(delta_phi_chunk), out.size)
        dphi = np.asarray(delta_phi[start:stop], dtype=float)[:, None, None]
        chi = dbeta - dphi

        z = theta1 * c * np.exp(0.5j * chi) + theta2 * s * np.exp(-0.5j * chi)
        a = np.abs(z)
        alpha = np.angle(z)
        radial = _interpolate_radial_from_state(a, state)

        phase = (
            np.power(state.phase_beta_bar, sigma1)[None, :, :]
            * np.exp(0.5j * (sigma2 - sigma3) * chi)
            * np.exp(-1j * state.Sigma * alpha)
        )
        integral = state.delta_beta_weight * np.sum(angular_weight * phase * radial, axis=(1, 2))
        out[start:stop] = prefactor * integral

    return out


def _compute_worker_initializer(state: _ComputeWorkerState) -> None:
    global _COMPUTE_WORKER_STATE
    _COMPUTE_WORKER_STATE = state


def _compute_worker(
    task: tuple[int, int, float, float, np.ndarray, int],
) -> tuple[int, int, np.ndarray]:
    if _COMPUTE_WORKER_STATE is None:  # pragma: no cover - defensive
        raise RuntimeError("Compute worker state was not initialized.")
    i, j, theta1, theta2, delta_phi, delta_phi_chunk = task
    value = _evaluate_theta_pair_from_state(
        theta1,
        theta2,
        delta_phi,
        delta_phi_chunk=delta_phi_chunk,
        state=_COMPUTE_WORKER_STATE,
    )
    return i, j, value


class BruteForceX3PCF:
    r"""Direct general-spin X-projection 3PCF reference calculation.

    ``bispectrum`` must be a fastnc ``Bispectrum2D`` instance, or a compatible
    callable with signature ``B(ell1, ell2, ell3, **model_kwargs)``.  It is the
    scalar-source bispectrum associated with the ordered fields at vertices
    ``(1, 2, 3)``.  For a shear leg, supply its convergence-like scalar source;
    the spin phases are controlled by ``spin`` and ``component``.

    The returned X-projection quantity is

    .. math::

        \zeta^\times(\theta_1,\theta_2,\Delta\phi)
        = \frac{Q_\epsilon(-i)^\Sigma}{2(2\pi)^3}
          \int d\psi\,\sin(2\psi)\int d\Delta\beta\,
          e^{i\sigma_1\bar\beta}
          e^{i(\sigma_2-\sigma_3)\chi/2}
          e^{-i\Sigma\alpha}
          \mathcal T_\Sigma(A;\psi,\Delta\beta),

    where ``chi = Delta beta - Delta phi`` and

    .. math::

        \mathcal T_\Sigma(A;\psi,\Delta\beta)
        = \int d\ln\ell\,\ell^4 J_\Sigma(\ell A)
          B(\ell_1,\ell_2,\ell_3).

    The radial transforms are tabulated in :meth:`prepare`.  The same table is
    then reused for arbitrary real-space grids.  In adaptive mode, the whole
    Fourier-angle grid is globally refined and the final table corresponds to
    the converged angular resolution.
    """

    def __init__(
        self,
        bispectrum: Callable[..., np.ndarray],
        *,
        spin: tuple[int, int, int],
        component: int = 0,
        config: BruteForce3PCFConfig | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        q_epsilon: complex = 1.0 + 0.0j,
    ) -> None:
        if not callable(bispectrum):
            raise TypeError(
                "bispectrum must be a fastnc Bispectrum2D object or a callable "
                "B(ell1, ell2, ell3, **model_kwargs)."
            )

        self.bispectrum = bispectrum
        self.config = (config or BruteForce3PCFConfig()).validate()
        self.model_kwargs = dict(model_kwargs or {})
        self.q_epsilon = complex(q_epsilon)

        self.spin_spec = SpinSpec(tuple(int(s) for s in spin))
        self.component = int(component)
        component_spec = self.spin_spec.component(self.component)
        self.spin = component_spec.spin
        self.epsilon = component_spec.epsilon
        self.sigma = component_spec.sigma
        self.Sigma = int(sum(self.sigma))

        self.ell: np.ndarray | None = None
        self.psi: np.ndarray | None = None
        self.psi_weight: np.ndarray | None = None
        self.delta_beta: np.ndarray | None = None
        self.delta_beta_weight: float | None = None
        self.a_grid: np.ndarray | None = None
        self.radial_transform: np.ndarray | None = None
        self._log_a_grid: np.ndarray | None = None
        self._phase_beta_bar_table: np.ndarray | None = None
        self._n_psi_current: int | None = None
        self._n_delta_beta_current: int | None = None

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_positive_grid(x: ArrayLike, name: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
            raise ValueError(f"{name} must contain positive finite values.")
        return x

    @staticmethod
    def _as_angle_grid(x: ArrayLike, name: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        if np.any(~np.isfinite(x)):
            raise ValueError(f"{name} must contain finite values.")
        return x

    @property
    def prepared(self) -> bool:
        return self.radial_transform is not None

    @property
    def n_psi_current(self) -> int | None:
        """Number of Fourier-ratio angular nodes in the cached radial table."""
        return self._n_psi_current

    @property
    def n_delta_beta_current(self) -> int | None:
        """Number of relative-Fourier-angle nodes in the cached radial table."""
        return self._n_delta_beta_current

    def _mp_context(self):
        return mp.get_context(self.config.mp_start_method)

    def _worker_count(self, n_tasks: int) -> int:
        return min(int(self.config.n_processes), max(1, int(n_tasks)))

    # ------------------------------------------------------------------
    # Angular-grid and radial-table construction
    # ------------------------------------------------------------------
    @staticmethod
    def _make_angular_grid(n_psi: int, n_delta_beta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return full-domain angular quadrature nodes and weights.

        The psi nodes remain strictly inside ``(0, pi/2)`` to avoid requiring
        an arbitrary general bispectrum to be finite at a zero Fourier side.
        Delta-beta uses a periodic midpoint rule and retains the full domain.
        """
        xpsi, wpsi = np.polynomial.legendre.leggauss(int(n_psi))
        psi = 0.25 * np.pi * (xpsi + 1.0)
        psi_weight = 0.25 * np.pi * wpsi

        ndb = int(n_delta_beta)
        delta_beta = 2.0 * np.pi * (np.arange(ndb, dtype=float) + 0.5) / ndb
        delta_beta_weight = 2.0 * np.pi / ndb
        return psi, psi_weight, delta_beta, delta_beta_weight

    def _radial_worker_state(self, ell: np.ndarray) -> _RadialWorkerState:
        cfg = self.config
        return _RadialWorkerState(
            bispectrum=self.bispectrum,
            model_kwargs=self.model_kwargs,
            ell=ell,
            bessel_order=self.Sigma,
            fftlog_nu=cfg.fftlog_nu,
            c_window_width=cfg.c_window_width,
            N_extrap_low=cfg.N_extrap_low,
            N_extrap_high=cfg.N_extrap_high,
            N_pad=cfg.N_pad,
        )

    def _prepare_angular_grid(self, n_psi: int, n_delta_beta: int, *, force: bool = False) -> "BruteForceX3PCF":
        """Build or replace the radial table for one full Fourier-angle grid."""
        n_psi = int(n_psi)
        n_delta_beta = int(n_delta_beta)
        if n_psi < 2 or n_delta_beta < 4:
            raise ValueError("Require n_psi >= 2 and n_delta_beta >= 4.")

        same_grid = (
            self.prepared
            and self._n_psi_current == n_psi
            and self._n_delta_beta_current == n_delta_beta
        )
        if same_grid and not force:
            return self

        cfg = self.config
        ell = np.geomspace(cfg.ell_min, cfg.ell_max, int(cfg.n_ell))
        psi, psi_weight, delta_beta, delta_beta_weight = self._make_angular_grid(n_psi, n_delta_beta)
        state = self._radial_worker_state(ell)

        tasks = [
            (ipsi, idb, float(psi_value), float(dbeta_value))
            for ipsi, psi_value in enumerate(psi)
            for idb, dbeta_value in enumerate(delta_beta)
        ]
        total = len(tasks)
        table: np.ndarray | None = None
        a_grid: np.ndarray | None = None
        phase_beta_bar = np.empty((n_psi, n_delta_beta), dtype=np.complex128)

        def consume(entry: tuple[int, int, complex, np.ndarray, np.ndarray], count: int) -> None:
            nonlocal table, a_grid
            ipsi, idb, phase, a, transformed = entry
            if a_grid is None:
                if np.any(a <= 0.0) or np.any(np.diff(a) <= 0.0):
                    raise RuntimeError("fastnc FFTLog did not return a strictly increasing positive A grid.")
                a_grid = a
                table = np.empty((n_psi, n_delta_beta, a.size), dtype=np.complex128)
            elif not np.array_equal(a, a_grid):
                raise RuntimeError("The FFTLog A grid changed across Fourier-angle nodes.")

            assert table is not None
            phase_beta_bar[ipsi, idb] = phase
            table[ipsi, idb] = transformed
            if cfg.verbose and (count == 1 or count % max(1, total // 20) == 0 or count == total):
                print(
                    "[BruteForceX3PCF] radial FFTLog "
                    f"{count}/{total} (n_psi={n_psi}, n_delta_beta={n_delta_beta})"
                )

        use_parallel = bool(cfg.parallel_prepare) and int(cfg.n_processes) > 1 and total > 1
        if use_parallel:
            n_workers = self._worker_count(total)
            context = self._mp_context()
            # On spawn, ``bispectrum`` and model_kwargs must be pickleable.
            with context.Pool(
                processes=n_workers,
                initializer=_radial_worker_initializer,
                initargs=(state,),
            ) as pool:
                for count, entry in enumerate(
                    pool.imap_unordered(_radial_worker, tasks, chunksize=int(cfg.mp_chunksize)),
                    start=1,
                ):
                    consume(entry, count)
        else:
            for count, task in enumerate(tasks, start=1):
                consume(_evaluate_radial_task(task, state), count)

        assert a_grid is not None and table is not None
        self.ell = ell
        self.psi = psi
        self.psi_weight = psi_weight
        self.delta_beta = delta_beta
        self.delta_beta_weight = delta_beta_weight
        self.a_grid = a_grid
        self._log_a_grid = np.log(a_grid)
        self.radial_transform = table
        self._phase_beta_bar_table = phase_beta_bar
        self._n_psi_current = n_psi
        self._n_delta_beta_current = n_delta_beta
        return self

    def prepare(self, *, force: bool = False) -> "BruteForceX3PCF":
        """Tabulate the configured initial Fourier-angle grid.

        This method always prepares the base grid from ``config.n_psi`` and
        ``config.n_delta_beta``.  Adaptive refinement occurs in :meth:`compute`,
        because convergence must be assessed on a requested real-space grid.
        """
        return self._prepare_angular_grid(
            int(self.config.n_psi),
            int(self.config.n_delta_beta),
            force=force,
        )

    # ------------------------------------------------------------------
    # Real-space interpolation and angular quadrature
    # ------------------------------------------------------------------
    def _compute_worker_state(self) -> _ComputeWorkerState:
        if not self.prepared:
            raise RuntimeError("Call prepare() before evaluating the real-space 3PCF.")
        assert self.psi is not None
        assert self.psi_weight is not None
        assert self.delta_beta is not None
        assert self.delta_beta_weight is not None
        assert self.radial_transform is not None
        assert self._log_a_grid is not None
        assert self.a_grid is not None
        assert self._phase_beta_bar_table is not None
        return _ComputeWorkerState(
            psi=self.psi,
            psi_weight=self.psi_weight,
            delta_beta=self.delta_beta,
            delta_beta_weight=self.delta_beta_weight,
            radial_transform=self.radial_transform,
            log_a_grid=self._log_a_grid,
            a_min=float(self.a_grid[0]),
            a_max=float(self.a_grid[-1]),
            phase_beta_bar=self._phase_beta_bar_table,
            sigma=self.sigma,
            Sigma=self.Sigma,
            q_epsilon=self.q_epsilon,
            interpolation_bounds=self.config.interpolation_bounds,
        )

    def _evaluate_realspace_grid(
        self,
        theta1: np.ndarray,
        theta2: np.ndarray,
        delta_phi: np.ndarray,
        *,
        delta_phi_chunk: int,
    ) -> np.ndarray:
        """Compute a 3PCF grid using the presently cached angular table."""
        state = self._compute_worker_state()
        out = np.empty((theta1.size, theta2.size, delta_phi.size), dtype=np.complex128)
        tasks = [
            (i, j, float(t1), float(t2), delta_phi, int(delta_phi_chunk))
            for i, t1 in enumerate(theta1)
            for j, t2 in enumerate(theta2)
        ]
        use_parallel = bool(self.config.parallel_compute) and int(self.config.n_processes) > 1 and len(tasks) > 1

        if use_parallel:
            n_workers = self._worker_count(len(tasks))
            context = self._mp_context()
            with context.Pool(
                processes=n_workers,
                initializer=_compute_worker_initializer,
                initargs=(state,),
            ) as pool:
                for i, j, value in pool.imap_unordered(
                    _compute_worker,
                    tasks,
                    chunksize=int(self.config.mp_chunksize),
                ):
                    out[i, j] = value
        else:
            for i, j, t1, t2, dphi, chunk in tasks:
                out[i, j] = _evaluate_theta_pair_from_state(
                    t1,
                    t2,
                    dphi,
                    delta_phi_chunk=chunk,
                    state=state,
                )
        return out

    @staticmethod
    def _convergence_error(
        current: np.ndarray,
        previous: np.ndarray,
        *,
        rtol: float,
        atol: float,
    ) -> tuple[float, float]:
        """Return largest normalized and absolute inter-grid changes."""
        delta = np.abs(current - previous)
        scale = atol + rtol * np.maximum(np.abs(current), np.abs(previous))
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                delta,
                scale,
                out=np.where(delta == 0.0, 0.0, np.inf),
                where=scale > 0.0,
            )
        return float(np.max(normalized)), float(np.max(delta))

    def _compute_adaptive(
        self,
        theta1: np.ndarray,
        theta2: np.ndarray,
        delta_phi: np.ndarray,
        *,
        delta_phi_chunk: int,
    ) -> tuple[np.ndarray, bool, int, float | None, float | None]:
        """Globally refine Fourier-angle grids until the output converges."""
        cfg = self.config
        n_psi = int(cfg.n_psi)
        n_delta_beta = int(cfg.n_delta_beta)
        self._prepare_angular_grid(n_psi, n_delta_beta)
        previous = self._evaluate_realspace_grid(
            theta1,
            theta2,
            delta_phi,
            delta_phi_chunk=delta_phi_chunk,
        )

        last_norm: float | None = None
        last_abs: float | None = None
        for refinement in range(1, int(cfg.angular_max_refinements) + 1):
            next_n_psi = n_psi * int(cfg.angular_refinement_factor) if cfg.refine_psi else n_psi
            next_n_delta_beta = (
                n_delta_beta * int(cfg.angular_refinement_factor)
                if cfg.refine_delta_beta
                else n_delta_beta
            )
            self._prepare_angular_grid(next_n_psi, next_n_delta_beta)
            current = self._evaluate_realspace_grid(
                theta1,
                theta2,
                delta_phi,
                delta_phi_chunk=delta_phi_chunk,
            )
            last_norm, last_abs = self._convergence_error(
                current,
                previous,
                rtol=cfg.angular_rtol,
                atol=cfg.angular_atol,
            )
            if cfg.verbose:
                print(
                    "[BruteForceX3PCF] angular refinement "
                    f"{refinement}: n_psi={next_n_psi}, n_delta_beta={next_n_delta_beta}, "
                    f"max normalized change={last_norm:.3e}, max abs change={last_abs:.3e}"
                )
            if last_norm <= 1.0:
                return current, True, refinement, last_norm, last_abs

            previous = current
            n_psi = next_n_psi
            n_delta_beta = next_n_delta_beta

        message = (
            "Fourier-angle adaptive quadrature did not reach the requested tolerance "
            f"after {cfg.angular_max_refinements} refinements. "
            f"Last grid: n_psi={self._n_psi_current}, "
            f"n_delta_beta={self._n_delta_beta_current}; "
            f"max normalized change={last_norm!r}, max abs change={last_abs!r}."
        )
        if cfg.adaptive_strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return previous, False, int(cfg.angular_max_refinements), last_norm, last_abs

    def compute(
        self,
        theta1: ArrayLike,
        theta2: ArrayLike,
        delta_phi: ArrayLike,
        *,
        delta_phi_chunk: int = 32,
        angular_mode: str | None = None,
    ) -> BruteForce3PCFResult:
        """Evaluate the X-projection 3PCF on a requested real-space grid.

        Parameters
        ----------
        theta1, theta2, delta_phi
            One-dimensional target grids.  The output has shape
            ``(len(theta1), len(theta2), len(delta_phi))``.
        delta_phi_chunk
            Number of opening-angle values evaluated together for one theta
            pair.  Lower values reduce peak temporary memory.
        angular_mode
            Optional per-call override of ``config.angular_mode``.
            ``"adaptive"`` uses full-grid dyadic refinement in the selected
            Fourier-angle dimensions and tests convergence on this exact target
            real-space grid.
        """
        theta1 = self._as_positive_grid(theta1, "theta1")
        theta2 = self._as_positive_grid(theta2, "theta2")
        delta_phi = self._as_angle_grid(delta_phi, "delta_phi")
        if int(delta_phi_chunk) < 1:
            raise ValueError("delta_phi_chunk must be >= 1.")

        mode = self.config.angular_mode if angular_mode is None else str(angular_mode)
        if mode not in {"fixed", "adaptive"}:
            raise ValueError("angular_mode must be 'fixed' or 'adaptive'.")
        if mode == "adaptive" and not (self.config.refine_psi or self.config.refine_delta_beta):
            raise ValueError("Adaptive mode requires refine_psi and/or refine_delta_beta.")

        if mode == "fixed":
            self.prepare()
            value = self._evaluate_realspace_grid(
                theta1,
                theta2,
                delta_phi,
                delta_phi_chunk=int(delta_phi_chunk),
            )
            converged = True
            refinements = 0
            error_norm = None
            error_abs = None
        else:
            value, converged, refinements, error_norm, error_abs = self._compute_adaptive(
                theta1,
                theta2,
                delta_phi,
                delta_phi_chunk=int(delta_phi_chunk),
            )

        assert self._n_psi_current is not None
        assert self._n_delta_beta_current is not None
        return BruteForce3PCFResult(
            theta1=theta1,
            theta2=theta2,
            delta_phi=delta_phi,
            value=value,
            spin=self.spin,
            component=self.component,
            epsilon=self.epsilon,
            sigma=self.sigma,
            q_epsilon=self.q_epsilon,
            angular_mode=mode,
            angular_converged=converged,
            angular_refinements=refinements,
            n_psi_used=self._n_psi_current,
            n_delta_beta_used=self._n_delta_beta_current,
            angular_error_norm=error_norm,
            angular_max_abs_change=error_abs,
        )
