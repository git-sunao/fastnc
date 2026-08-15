"""Direct brute-force reference solver for projected spin 3PCFs.

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
    tolerance.

By default the solver integrates the full Fourier-angle domain
``psi in (0, pi/2)``, ``Delta beta in [0, 2 pi)``.  An opt-in
symmetry-reduced integration mode is also available.  It is valid only when
the scalar-source bispectrum is mirror symmetric (automatic for this
length-only ``Bispectrum2D`` interface) *and* invariant under exchange of
legs 2 and 3.  The latter is checked numerically by default before the
reduced quadrature is used.

The expensive radial FFTLog tasks and, optionally, the real-space
``(theta1, theta2)`` tasks can be distributed with Python's standard
:mod:`multiprocessing` module.
"""
from __future__ import annotations

import logging

from dataclasses import dataclass
import multiprocessing as mp
from typing import Any, Callable, Mapping, Literal
import warnings

import numpy as np

from ..hankel.fftlog import hankel as _FastncHankel
from .spin import SpinSpec


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class BruteForce3PCFConfig:
    """Numerical settings for :class:`BruteForceX3PCF`.

    Parameters
    ----------
    ell_min, ell_max, n_ell
        Physical common-scale integration grid.
    radial_cache_padding_factor
        Multiplicative zero-padding factor used only when constructing the
        radial FFTLog cache.  Values larger than one extend the logarithmic
        ``ell`` grid below ``ell_min`` and above ``ell_max`` at the same
        ``dln(ell)`` spacing, while setting the radial source identically zero
        outside the physical interval.  This leaves the physical integration
        range unchanged but expands the tabulated ``A`` range approximately to
        ``[1 / (factor * ell_max), factor / ell_min]``.
    n_psi
        Gauss--Legendre nodes per active ``psi`` interval.  This is
        ``psi in (0, pi/2)`` by default and ``psi in (0, pi/4)`` when
        ``reduce_domain=True``.
    n_delta_beta
        Angular nodes per active ``Delta beta`` integration interval.  In the
        default full-domain calculation this is ``[0, 2 pi)``; with
        ``reduce_domain=True`` it is the reduced integration interval ``[0, pi)``.
    reduce_domain
        Enable the four-image symmetry-reduced quadrature using four symmetry-related images.  The
        radial FFTLog table is then constructed only for
        ``psi in (0, pi/4)`` and ``Delta beta in [0, pi)``.  The other three
        sectors are reconstructed by the mirror and 2<->3 exchange symmetries.
        This option is off by default because the required exchange symmetry
        does not hold for a generic ordered cross bispectrum.
    reduce_domain_validate_exchange
        Numerically test ``B(ell1, ell2, ell3) = B(ell1, ell3, ell2)`` before
        the reduced domain is used.  Leave this enabled for validation and for
        general cross-field work.
    reduce_domain_n_symmetry_tests, reduce_domain_rtol,
    reduce_domain_atol
        Size and tolerance of that exchange-symmetry check.
    angular_mode
        ``"fixed"`` evaluates the configured angular grid once.  ``"adaptive"`` repeatedly refines the global grid in the selected angular directions;
        ``"local_adaptive"`` refines only configurable neighborhoods of the
        squeezed hotspot,
        comparing the full requested real-space 3PCF after every refinement.
    angular_rtol, angular_atol
        Convergence is accepted when every target configuration satisfies

        ``abs(zeta_new-zeta_old) <= atol + rtol * max(abs(zeta_new), abs(zeta_old))``.

        The result reports the largest normalized difference.
    refine_psi, refine_delta_beta
        Select which Fourier-angle grid dimensions are refined in adaptive
        mode.  At least one must be true when ``angular_mode="adaptive"``.
    angular_max_refinements
        Maximum number of grid refinements after the initial grid.  Set to
        ``None`` to continue refinement until convergence; this is intentionally
        unbounded and should only be used with an external resource limit.
    adaptive_strict
        Legacy failure switch.  If true, failure to reach the requested
        tolerance raises :class:`RuntimeError`; if false, the final trial is
        returned after a warning.  ``adaptive_failure`` takes precedence.
    adaptive_failure
        Explicit failure policy.  ``"raise"`` raises after a bounded adaptive
        run fails.  ``"return_last"`` returns the finest trial with
        ``angular_converged=False``.  ``None`` retains the legacy
        ``adaptive_strict`` behavior.
    adaptive_store_trials
        Controls retained adaptive trial fields in the result: ``"none"`` keeps
        none, ``"last"`` keeps the finest trial, and ``"all"`` keeps the
        initial grid and every refinement.  Trial values are full 3PCF arrays,
        so ``"all"`` can use substantial memory.
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
    radial_cache_padding_factor: float = 1.0

    n_psi: int = 32
    n_delta_beta: int = 96

    # Optional four-image symmetry reduction.  This is only valid for a
    # mirror-even source bispectrum with 2<->3 exchange symmetry.
    reduce_domain: bool = False
    reduce_domain_validate_exchange: bool = True
    reduce_domain_n_symmetry_tests: int = 4
    reduce_domain_rtol: float = 1.0e-10
    reduce_domain_atol: float = 0.0

    fftlog_nu: float = 1.01
    c_window_width: float = 0.25
    N_extrap_low: int = 0
    N_extrap_high: int = 0
    N_pad: int = 0

    interpolation_bounds: str = "raise"
    small_a_fallback: bool = True
    small_a_terms: int = 4

    angular_mode: str = "fixed"
    local_psi_center: float = 0.25 * np.pi
    local_psi_half_width: float = 0.12
    local_delta_beta_center: float = np.pi
    local_delta_beta_half_width: float = 0.30
    angular_rtol: float = 1.0e-3
    angular_atol: float = 0.0
    angular_max_refinements: int | None = 4
    angular_refinement_factor: int = 2
    refine_psi: bool = True
    refine_delta_beta: bool = True
    adaptive_strict: bool = True
    adaptive_failure: Literal["raise", "return_last"] | None = None
    adaptive_store_trials: Literal["none", "last", "all"] = "none"

    n_processes: int = 1
    mp_start_method: str | None = None
    mp_chunksize: int = 1
    parallel_prepare: bool = True
    parallel_compute: bool = True


    def validate(self) -> "BruteForce3PCFConfig":
        if not (0.0 < self.ell_min < self.ell_max):
            raise ValueError("Require 0 < ell_min < ell_max.")
        if int(self.n_ell) < 8 or int(self.n_ell) % 2:
            raise ValueError("n_ell must be an even integer >= 8 for fastnc FFTLog.")
        if (
            not np.isfinite(self.radial_cache_padding_factor)
            or self.radial_cache_padding_factor < 1.0
        ):
            raise ValueError("radial_cache_padding_factor must be finite and >= 1.")
        if int(self.n_psi) < 2:
            raise ValueError("n_psi must be >= 2.")
        if int(self.n_delta_beta) < 4:
            raise ValueError("n_delta_beta must be >= 4.")
        if int(self.reduce_domain_n_symmetry_tests) < 1:
            raise ValueError("reduce_domain_n_symmetry_tests must be >= 1.")
        if self.reduce_domain_rtol < 0.0 or self.reduce_domain_atol < 0.0:
            raise ValueError(
                "reduce_domain_rtol and reduce_domain_atol must be non-negative."
            )
        if (
            self.reduce_domain
            and self.reduce_domain_rtol == 0.0
            and self.reduce_domain_atol == 0.0
        ):
            raise ValueError(
                "At least one of reduce_domain_rtol or reduce_domain_atol "
                "must be positive when reduce_domain=True."
            )
        if self.interpolation_bounds not in {"raise", "clip"}:
            raise ValueError("interpolation_bounds must be 'raise' or 'clip'.")
        if int(self.small_a_terms) < 1:
            raise ValueError("small_a_terms must be >= 1.")
        if self.angular_mode not in {"fixed", "adaptive", "local_adaptive"}:
            raise ValueError("angular_mode must be 'fixed', 'adaptive', or 'local_adaptive'.")
        if not (0.0 < self.local_psi_half_width < 0.25 * np.pi):
            raise ValueError("local_psi_half_width must lie in (0, pi/4).")
        if not (0.0 < self.local_delta_beta_half_width < np.pi):
            raise ValueError("local_delta_beta_half_width must lie in (0, pi).")
        if self.angular_rtol < 0.0 or self.angular_atol < 0.0:
            raise ValueError("angular_rtol and angular_atol must be non-negative.")
        if self.angular_rtol == 0.0 and self.angular_atol == 0.0:
            raise ValueError("At least one of angular_rtol or angular_atol must be positive.")
        if self.angular_max_refinements is not None and int(self.angular_max_refinements) < 0:
            raise ValueError("angular_max_refinements must be >= 0 or None.")
        if self.adaptive_failure not in {None, "raise", "return_last"}:
            raise ValueError(
                "adaptive_failure must be None, 'raise', or 'return_last'."
            )
        if self.adaptive_store_trials not in {"none", "last", "all"}:
            raise ValueError(
                "adaptive_store_trials must be 'none', 'last', or 'all'."
            )
        if int(self.angular_refinement_factor) < 2:
            raise ValueError("angular_refinement_factor must be an integer >= 2.")
        if self.angular_mode in {"adaptive", "local_adaptive"} and not (self.refine_psi or self.refine_delta_beta):
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
class BruteForce3PCFAdaptiveTrial:
    """One real-space evaluation in an adaptive Fourier-angle sequence."""

    refinement: int
    n_psi: int
    n_delta_beta: int
    value: np.ndarray
    error_norm: float | None
    max_abs_change: float | None


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
    used_reduced_domain: bool
    reduced_domain_exchange_validated: bool
    reduced_domain_exchange_error_norm: float | None
    angular_converged: bool
    angular_refinements: int
    n_psi_used: int
    n_delta_beta_used: int
    angular_error_norm: float | None
    angular_max_abs_change: float | None
    adaptive_trials: tuple[BruteForce3PCFAdaptiveTrial, ...] = ()


@dataclass(frozen=True)
class _RadialWorkerState:
    """Pickleable state required by a radial FFTLog worker."""

    bispectrum: Callable[..., np.ndarray]
    model_kwargs: Mapping[str, Any]
    ell: np.ndarray
    physical_start: int
    physical_stop: int
    bessel_order: int
    fftlog_nu: float
    c_window_width: float
    N_extrap_low: int
    N_extrap_high: int
    N_pad: int
    small_a_terms: int


@dataclass(frozen=True)
class _ComputeWorkerState:
    """Read-only state required by a real-space theta-pair worker."""

    psi: np.ndarray
    psi_weight: np.ndarray
    delta_beta: np.ndarray
    delta_beta_weight: np.ndarray
    radial_transform: np.ndarray
    small_a_moments: np.ndarray | None
    log_a_grid: np.ndarray
    a_min: float
    a_max: float
    phase_beta_bar: np.ndarray
    reduce_domain: bool
    reduced_domain_psi_images: np.ndarray | None
    reduced_domain_delta_beta_images: np.ndarray | None
    reduced_domain_phase_beta_bar_images: np.ndarray | None
    sigma: tuple[int, int, int]
    Sigma: int
    q_epsilon: complex
    interpolation_bounds: str
    small_a_fallback: bool


_RADIAL_WORKER_STATE: _RadialWorkerState | None = None
_COMPUTE_WORKER_STATE: _ComputeWorkerState | None = None


def _phase_beta_bar_array(psi: np.ndarray | float, delta_beta: np.ndarray | float) -> np.ndarray:
    """Return ``exp(i beta_bar)`` without selecting an angle branch.

    ``psi`` and ``delta_beta`` follow NumPy broadcasting.  The formula is
    branch-free and is therefore stable both for the full-domain quadrature
    and for the four images used by the symmetry reduction.
    """
    psi = np.asarray(psi, dtype=float)
    delta_beta = np.asarray(delta_beta, dtype=float)
    c = np.cos(psi)
    s = np.sin(psi)
    numerator = -(c * np.exp(0.5j * delta_beta) + s * np.exp(-0.5j * delta_beta))
    # Compute the norm from the complex numerator itself rather than from
    # 1 + sin(2 psi) cos(Delta beta), which suffers catastrophic cancellation
    # near (psi, Delta beta)=(pi/4, pi).  The phase is undefined at the exact
    # measure-zero degenerate point; assign unit phase there.
    denom = np.abs(numerator)
    return np.divide(
        numerator, denom,
        out=np.ones(np.broadcast_shapes(numerator.shape, denom.shape), dtype=np.complex128),
        where=denom > 8.0 * np.finfo(float).eps,
    )


def _phase_beta_bar(psi: float, delta_beta: float) -> complex:
    """Scalar convenience wrapper for :func:`_phase_beta_bar_array`."""
    return complex(_phase_beta_bar_array(psi, delta_beta))


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
) -> tuple[int, int, complex, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one angular node's source bispectrum, radial FFTLog, and small-A moments."""
    ipsi, idb, psi, delta_beta = task
    ell = state.ell

    # The FFTLog grid may be wider than the physical integration interval.
    # Evaluate the bispectrum only on the physical samples and set the source
    # to zero on the cache-only extension.  This expands the reciprocal A grid
    # without changing the Fourier integral being approximated.
    sl = slice(int(state.physical_start), int(state.physical_stop))
    ell_phys = ell[sl]
    ell2 = ell_phys * np.cos(psi)
    ell3 = ell_phys * np.sin(psi)
    ell1_sq = ell2**2 + ell3**2 + 2.0 * ell2 * ell3 * np.cos(delta_beta)
    ell1 = np.sqrt(np.maximum(ell1_sq, 0.0))

    source_phys = np.asarray(
        state.bispectrum(ell1, ell2, ell3, **dict(state.model_kwargs)),
        dtype=np.complex128,
    )
    try:
        source_phys = np.broadcast_to(source_phys, ell_phys.shape).astype(
            np.complex128, copy=False
        )
    except ValueError as exc:
        raise ValueError(
            "bispectrum must return a scalar or an array broadcastable to the physical FFTLog ell-grid shape."
        ) from exc

    if not np.all(np.isfinite(source_phys)):
        raise ValueError(
            "bispectrum returned non-finite values on the physical FFTLog ell grid "
            f"at psi={psi:.6e}, Delta beta={delta_beta:.6e}."
        )

    source_b = np.zeros(ell.shape, dtype=np.complex128)
    source_b[sl] = source_phys

    # Small-A Bessel-series moments.  For n=|Sigma|,
    # J_n(x)=sum_m (-1)^m (x/2)^(n+2m)/(m! Gamma(n+m+1)).
    # Store the corresponding physical dln(ell) moments once per angular node.
    n = abs(int(state.bessel_order))
    sign = -1.0 if (state.bessel_order < 0 and n % 2) else 1.0
    lnell = np.log(ell_phys)
    moments = np.empty(int(state.small_a_terms), dtype=np.complex128)
    import math
    for m in range(int(state.small_a_terms)):
        power = 4 + n + 2 * m
        coeff = sign * ((-1.0) ** m) / (2.0 ** (n + 2*m) * math.factorial(m) * math.gamma(n + m + 1))
        moments[m] = coeff * np.trapezoid((ell_phys ** power) * source_phys, x=lnell)

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
    return ipsi, idb, _phase_beta_bar(psi, delta_beta), a_grid, transformed, moments


def _radial_worker_initializer(state: _RadialWorkerState) -> None:
    global _RADIAL_WORKER_STATE
    _RADIAL_WORKER_STATE = state


def _radial_worker(task: tuple[int, int, float, float]) -> tuple[int, int, complex, np.ndarray, np.ndarray, np.ndarray]:
    if _RADIAL_WORKER_STATE is None:  # pragma: no cover - defensive
        raise RuntimeError("Radial worker state was not initialized.")
    return _evaluate_radial_task(task, _RADIAL_WORKER_STATE)


def _interpolate_radial_from_state(a: np.ndarray, state: _ComputeWorkerState) -> np.ndarray:
    """Interpolate the cached radial transform in ``ln A``.

    Values below the FFTLog cache are evaluated with the small-A Bessel
    expansion when ``small_a_fallback`` is enabled.  This removes the
    structural lower-bound failure near degenerate real-space geometry
    (A -> 0) without clipping the physical transform.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 3 or a.shape[1:] != (state.psi.size, state.delta_beta.size):
        raise ValueError(
            "A must have shape (n_delta_phi, n_psi, n_delta_beta) matching the cached table."
        )
    if np.any(~np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("Encountered non-finite or negative A.")

    high = a > state.a_max
    low = a < state.a_min
    if np.any(high) or (np.any(low) and not state.small_a_fallback):
        found_min = float(np.min(a))
        found_max = float(np.max(a))
        if state.interpolation_bounds == "raise":
            raise ValueError(
                "Required A values fall outside the cached FFTLog grid: "
                f"requested [{found_min:.6e}, {found_max:.6e}], "
                f"available [{state.a_min:.6e}, {state.a_max:.6e}]. "
                "For A below the cache enable small_a_fallback; for A above "
                "the cache increase radial_cache_padding_factor."
            )

    # Standard interpolation, clipping only for constructing safe indices.
    a_safe = np.clip(a, state.a_min, state.a_max)
    loga = np.log(a_safe)
    index = np.searchsorted(state.log_a_grid, loga, side="right") - 1
    index = np.clip(index, 0, state.log_a_grid.size - 2)
    x0 = state.log_a_grid[index]
    x1 = state.log_a_grid[index + 1]
    weight = (loga - x0) / (x1 - x0)

    ipsi = np.arange(state.psi.size)[None, :, None]
    idb = np.arange(state.delta_beta.size)[None, None, :]
    y0 = state.radial_transform[ipsi, idb, index]
    y1 = state.radial_transform[ipsi, idb, index + 1]
    out = (1.0 - weight) * y0 + weight * y1

    if np.any(low) and state.small_a_fallback:
        if state.small_a_moments is None:
            raise RuntimeError("small_a_fallback requested but no small-A moments were cached.")
        series = np.zeros_like(out)
        a2 = a * a
        n = abs(int(state.Sigma))
        # moments[m] already contains the Bessel-series coefficient.
        for m in range(state.small_a_moments.shape[-1]):
            series += state.small_a_moments[None, :, :, m] * a ** (n + 2*m)
        out = np.where(low, series, out)

    # Upper clipping remains an explicit diagnostic mode only.
    if np.any(high) and state.interpolation_bounds == "clip":
        pass
    return out


def _evaluate_theta_pair_image_from_state(
    theta1: float,
    theta2: float,
    delta_phi: np.ndarray,
    *,
    psi_image: np.ndarray,
    delta_beta_image: np.ndarray,
    phase_beta_bar_image: np.ndarray,
    delta_phi_chunk: int,
    state: _ComputeWorkerState,
) -> np.ndarray:
    """Evaluate one Fourier-angle image with the cached radial table.

    In the full-domain calculation the image is simply the cached grid.  For
    the symmetry reduction, the cached table belongs to the fundamental
    rectangle ``psi in (0, pi/4)``, ``Delta beta in [0, pi)``; this function
    evaluates the physical phase and real-space geometry of one of its four
    symmetry images while reusing that same radial table.
    """
    psi = np.asarray(psi_image, dtype=float)[None, :, None]
    dbeta = np.asarray(delta_beta_image, dtype=float)[None, None, :]
    phase_beta_bar_image = np.asarray(phase_beta_bar_image, dtype=np.complex128)
    sigma1, sigma2, sigma3 = state.sigma
    out = np.empty(np.asarray(delta_phi).size, dtype=np.complex128)

    # ``psi_weight`` and ``delta_beta_weight`` belong to the fundamental grid.
    # For the reduced-domain construction, sin[2(pi/2-psi)] = sin(2 psi), so these
    # are also the correct weights for every image.
    angular_weight = state.psi_weight[None, :, None] * np.sin(2.0 * psi)
    prefactor = state.q_epsilon * ((-1j) ** state.Sigma) / (2.0 * (2.0 * np.pi) ** 3)

    for start in range(0, out.size, int(delta_phi_chunk)):
        stop = min(start + int(delta_phi_chunk), out.size)
        dphi = np.asarray(delta_phi[start:stop], dtype=float)[:, None, None]
        chi = dbeta - dphi

        z = theta1 * np.cos(psi) * np.exp(0.5j * chi) + theta2 * np.sin(psi) * np.exp(-0.5j * chi)
        a = np.abs(z)
        alpha = np.angle(z)
        radial = _interpolate_radial_from_state(a, state)

        phase = (
            np.power(phase_beta_bar_image, sigma1)[None, :, :]
            * np.exp(0.5j * (sigma2 - sigma3) * chi)
            * np.exp(-1j * state.Sigma * alpha)
        )
        integral = np.sum(angular_weight * state.delta_beta_weight[None, None, :] * phase * radial, axis=(1, 2))
        out[start:stop] = prefactor * integral

    return out


def _evaluate_theta_pair_from_state(
    theta1: float,
    theta2: float,
    delta_phi: np.ndarray,
    *,
    delta_phi_chunk: int,
    state: _ComputeWorkerState,
) -> np.ndarray:
    """Evaluate all requested opening angles for one real-space theta pair."""
    if not state.reduce_domain:
        return _evaluate_theta_pair_image_from_state(
            theta1,
            theta2,
            delta_phi,
            psi_image=state.psi,
            delta_beta_image=state.delta_beta,
            phase_beta_bar_image=state.phase_beta_bar,
            delta_phi_chunk=delta_phi_chunk,
            state=state,
        )

    if (
        state.reduced_domain_psi_images is None
        or state.reduced_domain_delta_beta_images is None
        or state.reduced_domain_phase_beta_bar_images is None
    ):
        raise RuntimeError("symmetry-reduced-domain state was not initialized.")

    result = np.zeros(np.asarray(delta_phi).size, dtype=np.complex128)
    for image in range(4):
        result += _evaluate_theta_pair_image_from_state(
            theta1,
            theta2,
            delta_phi,
            psi_image=state.reduced_domain_psi_images[image],
            delta_beta_image=state.reduced_domain_delta_beta_images[image],
            phase_beta_bar_image=state.reduced_domain_phase_beta_bar_images[image],
            delta_phi_chunk=delta_phi_chunk,
            state=state,
        )
    return result

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
        self.delta_beta_weight: np.ndarray | None = None
        self.a_grid: np.ndarray | None = None
        self.radial_transform: np.ndarray | None = None
        self._small_a_moments: np.ndarray | None = None
        self._log_a_grid: np.ndarray | None = None
        self._phase_beta_bar_table: np.ndarray | None = None
        self._n_psi_current: int | None = None
        self._n_delta_beta_current: int | None = None
        self._n_psi_base_current: int | None = None
        self._n_delta_beta_base_current: int | None = None
        self._local_factor_current: int = 1
        self._reduced_domain_exchange_validated = False
        self._reduced_domain_exchange_error_norm: float | None = None

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
        """Number of Fourier-ratio nodes in the cached radial table.

        With ``reduce_domain=True`` this is the number of nodes on the
        reduced interval ``(0, pi/4)`` rather than on the full interval.
        """
        return self._n_psi_current

    @property
    def n_delta_beta_current(self) -> int | None:
        """Number of relative-angle nodes in the cached radial table.

        With ``reduce_domain=True`` this is the number of nodes on
        ``[0, pi)`` rather than on the full ``[0, 2 pi)`` domain.
        """
        return self._n_delta_beta_current

    def _mp_context(self):
        return mp.get_context(self.config.mp_start_method)

    def _worker_count(self, n_tasks: int) -> int:
        return min(int(self.config.n_processes), max(1, int(n_tasks)))

    # ------------------------------------------------------------------
    # Angular-grid and radial-table construction
    # ------------------------------------------------------------------
    @staticmethod
    def _make_angular_grid(
        n_psi: int,
        n_delta_beta: int,
        *,
        reduce_domain: bool,
        local_factor: int = 1,
        refine_psi: bool = True,
        refine_delta_beta: bool = True,
        psi_center: float = 0.25 * np.pi,
        psi_half_width: float = 0.12,
        delta_beta_center: float = np.pi,
        delta_beta_half_width: float = 0.30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return Fourier-angle quadrature nodes and weights.

        ``local_factor=1`` gives the legacy global grid.  Values >1 add
        quadrature nodes only inside configurable neighborhoods of the
        squeezed hotspot, leaving the outer domain unchanged.
        """
        if reduce_domain and local_factor > 1:
            raise ValueError("local angular refinement is currently supported only for reduce_domain=False.")

        psi_extent = 0.25 * np.pi if reduce_domain else 0.5 * np.pi
        db_extent = np.pi if reduce_domain else 2.0 * np.pi

        def gl_segment(a, b, n):
            if b <= a or n <= 0:
                return np.empty(0), np.empty(0)
            x, w = np.polynomial.legendre.leggauss(int(n))
            return 0.5*(b-a)*x + 0.5*(a+b), 0.5*(b-a)*w

        # Legacy grid exactly when no local refinement is requested.
        if int(local_factor) <= 1:
            psi, psi_weight = gl_segment(0.0, psi_extent, int(n_psi))
            ndb = int(n_delta_beta)
            dbeta = db_extent * (np.arange(ndb, dtype=float) + 0.5) / ndb
            db_weight = np.full(ndb, db_extent / ndb, dtype=float)
            return psi, psi_weight, dbeta, db_weight

        # Composite psi quadrature: preserve baseline outer resolution and
        # multiply only the central segment's node density.
        pc = float(np.clip(psi_center, 0.0, psi_extent))
        ph = min(float(psi_half_width), pc, psi_extent-pc)
        p0, p1 = pc-ph, pc+ph
        density_p = float(n_psi) / psi_extent
        segs_p = []
        for a,b,mul in ((0.0,p0,1),(p0,p1,int(local_factor) if refine_psi else 1),(p1,psi_extent,1)):
            nn = max(1, int(np.ceil(density_p*(b-a)*mul))) if b>a else 0
            if mul > 1 and nn % 2:
                nn += 1  # avoid sampling the exact degenerate center
            segs_p.append(gl_segment(a,b,nn))
        psi = np.concatenate([x for x,_ in segs_p])
        psi_weight = np.concatenate([w for _,w in segs_p])

        # Composite midpoint rule in Delta beta, likewise locally refined.
        bc = float(delta_beta_center) % db_extent
        bh = min(float(delta_beta_half_width), bc, db_extent-bc)
        b0, b1 = bc-bh, bc+bh
        density_b = float(n_delta_beta) / db_extent
        dbs=[]; dws=[]
        for a,b,mul in ((0.0,b0,1),(b0,b1,int(local_factor) if refine_delta_beta else 1),(b1,db_extent,1)):
            if b<=a: continue
            nn=max(1,int(np.ceil(density_b*(b-a)*mul)))
            if mul > 1 and nn % 2:
                nn += 1  # midpoint rule then avoids Delta beta=center
            h=(b-a)/nn
            dbs.append(a + h*(np.arange(nn,dtype=float)+0.5))
            dws.append(np.full(nn,h,dtype=float))
        dbeta=np.concatenate(dbs)
        db_weight=np.concatenate(dws)
        return psi, psi_weight, dbeta, db_weight

    def _validate_reduce_domain_exchange_symmetry(self, ell: np.ndarray, *, force: bool = False) -> None:
        """Verify the nontrivial symmetry required by the symmetry-reduction shortcut.

        This solver accepts a length-only source bispectrum
        ``B(ell1, ell2, ell3)``.  Consequently its mirror symmetry under
        ``Delta beta -> -Delta beta`` is structural: the three side lengths
        are unchanged.  The additional reduction

        ``(psi, Delta beta) -> (pi/2 - psi, Delta beta)``

        requires the ordered source bispectrum to be invariant under the
        exchange of legs 2 and 3,

        ``B(ell1, ell2, ell3) = B(ell1, ell3, ell2)``.

        That property is automatic for the usual single-field shear
        bispectrum, but is not automatic for cross fields with distinct
        samples at vertices 2 and 3.  The check is deliberately performed on
        interior, nondegenerate triangles.
        """
        cfg = self.config
        if not cfg.reduce_domain:
            return
        if self._reduced_domain_exchange_validated and not force:
            return
        if not cfg.reduce_domain_validate_exchange:
            self._reduced_domain_exchange_validated = False
            self._reduced_domain_exchange_error_norm = None
            warnings.warn(
                "symmetry-reduced-domain acceleration is being used without "
                "checking the required 2<->3 source-bispectrum exchange symmetry. "
                "Use this only for a model whose symmetry has been independently "
                "established.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        ell = np.asarray(ell, dtype=float)
        ntest = int(cfg.reduce_domain_n_symmetry_tests)
        # Avoid the two FFTLog endpoints, where support cutoffs can make a
        # symmetry check uninformative.  The phase-space points are all well
        # inside the nondegenerate Fourier-triangle domain.
        sample_index = np.linspace(1, ell.size - 2, ntest).round().astype(int)
        ell_scale = ell[sample_index]
        psi = np.linspace(0.13 * np.pi, 0.37 * np.pi, ntest)
        delta_beta = np.linspace(0.17 * np.pi, 0.83 * np.pi, ntest)

        ell2 = ell_scale * np.cos(psi)
        ell3 = ell_scale * np.sin(psi)
        ell1 = np.sqrt(np.maximum(ell2**2 + ell3**2 + 2.0 * ell2 * ell3 * np.cos(delta_beta), 0.0))

        value_23 = np.asarray(
            self.bispectrum(ell1, ell2, ell3, **dict(self.model_kwargs)),
            dtype=np.complex128,
        )
        value_32 = np.asarray(
            self.bispectrum(ell1, ell3, ell2, **dict(self.model_kwargs)),
            dtype=np.complex128,
        )
        try:
            value_23 = np.broadcast_to(value_23, ell1.shape)
            value_32 = np.broadcast_to(value_32, ell1.shape)
        except ValueError as exc:
            raise ValueError(
                "bispectrum must return a scalar or an array broadcastable to the "
                "symmetry-validation test shape."
            ) from exc
        if not np.all(np.isfinite(value_23)) or not np.all(np.isfinite(value_32)):
            raise ValueError(
                "The symmetry-reduced-domain exchange-symmetry check encountered "
                "non-finite bispectrum values. Use the full angular domain, enlarge "
                "the model support, or disable the check only after an independent validation."
            )

        difference = np.abs(value_23 - value_32)
        tolerance = cfg.reduce_domain_atol + cfg.reduce_domain_rtol * np.maximum(
            np.abs(value_23), np.abs(value_32)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                difference,
                tolerance,
                out=np.where(difference == 0.0, 0.0, np.inf),
                where=tolerance > 0.0,
            )
        max_norm = float(np.max(normalized))
        self._reduced_domain_exchange_error_norm = max_norm
        if max_norm > 1.0:
            raise ValueError(
                "reduce_domain=True requires B(ell1, ell2, ell3) = "
                "B(ell1, ell3, ell2) for the ordered source bispectrum, but the "
                "configured numerical check failed: "
                f"max normalized difference={max_norm:.3e}.  This is expected for "
                "a generic cross bispectrum with inequivalent fields or samples at "
                "vertices 2 and 3. Use the full angular domain instead."
            )
        self._reduced_domain_exchange_validated = True
        logger.info(
            "[BruteForceX3PCF._validate_reduce_domain_exchange_symmetry] "
            "2<->3 exchange symmetry verified: max normalized difference=%.3e",
            max_norm,
        )

    @staticmethod
    def _radial_fftlog_grid(
        ell_physical: np.ndarray, padding_factor: float
    ) -> tuple[np.ndarray, int, int]:
        """Return a zero-padding FFTLog grid preserving the physical dln(ell)."""
        ell_physical = np.asarray(ell_physical, dtype=float)
        factor = float(padding_factor)
        if factor <= 1.0:
            return ell_physical, 0, ell_physical.size

        dln = float(np.log(ell_physical[1] / ell_physical[0]))
        n_extra = int(np.ceil(np.log(factor) / dln))
        index = np.arange(-n_extra, ell_physical.size + n_extra, dtype=float)
        ell_fftlog = ell_physical[0] * np.exp(dln * index)
        return ell_fftlog, n_extra, n_extra + ell_physical.size

    def _radial_worker_state(
        self, ell: np.ndarray, physical_start: int, physical_stop: int
    ) -> _RadialWorkerState:
        cfg = self.config
        return _RadialWorkerState(
            bispectrum=self.bispectrum,
            model_kwargs=self.model_kwargs,
            ell=ell,
            physical_start=int(physical_start),
            physical_stop=int(physical_stop),
            bessel_order=self.Sigma,
            fftlog_nu=cfg.fftlog_nu,
            c_window_width=cfg.c_window_width,
            N_extrap_low=cfg.N_extrap_low,
            N_extrap_high=cfg.N_extrap_high,
            N_pad=cfg.N_pad,
            small_a_terms=int(cfg.small_a_terms),
        )

    def _prepare_angular_grid(self, n_psi: int, n_delta_beta: int, *, force: bool = False, local_factor: int = 1) -> "BruteForceX3PCF":
        """Build or replace the radial table for one active Fourier-angle grid.

        In symmetry-reduced-domain mode this table belongs only to the fundamental
        Fourier-angle rectangle; the remaining sectors are reconstructed during
        the real-space angular integral.
        """
        n_psi = int(n_psi)
        n_delta_beta = int(n_delta_beta)
        if n_psi < 2 or n_delta_beta < 4:
            raise ValueError("Require n_psi >= 2 and n_delta_beta >= 4.")

        same_grid = (
            self.prepared
            and self._n_psi_base_current == n_psi
            and self._n_delta_beta_base_current == n_delta_beta
            and self._local_factor_current == int(local_factor)
        )
        if same_grid and not force:
            return self

        cfg = self.config
        ell_physical = np.geomspace(cfg.ell_min, cfg.ell_max, int(cfg.n_ell))
        self._validate_reduce_domain_exchange_symmetry(ell_physical, force=force)
        ell_fftlog, physical_start, physical_stop = self._radial_fftlog_grid(
            ell_physical, cfg.radial_cache_padding_factor
        )
        psi, psi_weight, delta_beta, delta_beta_weight = self._make_angular_grid(
            n_psi, n_delta_beta,
            reduce_domain=cfg.reduce_domain,
            local_factor=int(local_factor),
            refine_psi=bool(cfg.refine_psi),
            refine_delta_beta=bool(cfg.refine_delta_beta),
            psi_center=float(cfg.local_psi_center),
            psi_half_width=float(cfg.local_psi_half_width),
            delta_beta_center=float(cfg.local_delta_beta_center),
            delta_beta_half_width=float(cfg.local_delta_beta_half_width),
        )
        npsi_actual, ndb_actual = psi.size, delta_beta.size
        state = self._radial_worker_state(ell_fftlog, physical_start, physical_stop)

        tasks = [
            (ipsi, idb, float(psi_value), float(dbeta_value))
            for ipsi, psi_value in enumerate(psi)
            for idb, dbeta_value in enumerate(delta_beta)
        ]
        total = len(tasks)
        table: np.ndarray | None = None
        a_grid: np.ndarray | None = None
        phase_beta_bar = np.empty((npsi_actual, ndb_actual), dtype=np.complex128)
        small_a_moments = np.empty((npsi_actual, ndb_actual, int(cfg.small_a_terms)), dtype=np.complex128)

        def consume(entry: tuple[int, int, complex, np.ndarray, np.ndarray, np.ndarray], count: int) -> None:
            nonlocal table, a_grid
            ipsi, idb, phase, a, transformed, moments = entry
            if a_grid is None:
                if np.any(a <= 0.0) or np.any(np.diff(a) <= 0.0):
                    raise RuntimeError("fastnc FFTLog did not return a strictly increasing positive A grid.")
                a_grid = a
                table = np.empty((npsi_actual, ndb_actual, a.size), dtype=np.complex128)
            elif not np.array_equal(a, a_grid):
                raise RuntimeError("The FFTLog A grid changed across Fourier-angle nodes.")

            assert table is not None
            phase_beta_bar[ipsi, idb] = phase
            table[ipsi, idb] = transformed
            small_a_moments[ipsi, idb] = moments
            if logger.isEnabledFor(logging.DEBUG) and (
                count == 1 or count % max(1, total // 20) == 0 or count == total
            ):
                logger.debug(
                    "[BruteForceX3PCF._prepare_angular_grid] radial FFTLog "
                    "%d/%d (n_psi=%d, n_delta_beta=%d)",
                    count,
                    total,
                    npsi_actual,
                    ndb_actual,
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
        self.ell = ell_physical
        self.psi = psi
        self.psi_weight = psi_weight
        self.delta_beta = delta_beta
        self.delta_beta_weight = delta_beta_weight
        self.a_grid = a_grid
        self._log_a_grid = np.log(a_grid)
        self.radial_transform = table
        self._small_a_moments = small_a_moments
        self._phase_beta_bar_table = phase_beta_bar
        self._n_psi_current = npsi_actual
        self._n_delta_beta_current = ndb_actual
        self._n_psi_base_current = n_psi
        self._n_delta_beta_base_current = n_delta_beta
        self._local_factor_current = int(local_factor)
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
        assert self._small_a_moments is not None
        assert self._log_a_grid is not None
        assert self.a_grid is not None
        assert self._phase_beta_bar_table is not None
        use_reduced_domain = bool(self.config.reduce_domain)
        reduced_domain_psi_images: np.ndarray | None = None
        reduced_domain_delta_beta_images: np.ndarray | None = None
        reduced_domain_phase_beta_bar_images: np.ndarray | None = None
        if use_reduced_domain:
            # The four images tile the full Fourier-angle domain.  They are
            # ordered as (psi, dbeta), (pi/2-psi, dbeta),
            # (psi, 2pi-dbeta), (pi/2-psi, 2pi-dbeta).  The cached radial
            # table is reused for all images; their real-space geometry and
            # spin phases are nevertheless evaluated explicitly.
            reduced_domain_psi_images = np.stack(
                (self.psi, 0.5 * np.pi - self.psi, self.psi, 0.5 * np.pi - self.psi),
                axis=0,
            )
            reduced_domain_delta_beta_images = np.stack(
                (
                    self.delta_beta,
                    self.delta_beta,
                    2.0 * np.pi - self.delta_beta,
                    2.0 * np.pi - self.delta_beta,
                ),
                axis=0,
            )
            reduced_domain_phase_beta_bar_images = _phase_beta_bar_array(
                reduced_domain_psi_images[:, :, None],
                reduced_domain_delta_beta_images[:, None, :],
            )

        return _ComputeWorkerState(
            psi=self.psi,
            psi_weight=self.psi_weight,
            delta_beta=self.delta_beta,
            delta_beta_weight=self.delta_beta_weight,
            radial_transform=self.radial_transform,
            small_a_moments=self._small_a_moments,
            log_a_grid=self._log_a_grid,
            a_min=float(self.a_grid[0]),
            a_max=float(self.a_grid[-1]),
            phase_beta_bar=self._phase_beta_bar_table,
            reduce_domain=use_reduced_domain,
            reduced_domain_psi_images=reduced_domain_psi_images,
            reduced_domain_delta_beta_images=reduced_domain_delta_beta_images,
            reduced_domain_phase_beta_bar_images=reduced_domain_phase_beta_bar_images,
            sigma=self.sigma,
            Sigma=self.Sigma,
            q_epsilon=self.q_epsilon,
            interpolation_bounds=self.config.interpolation_bounds,
            small_a_fallback=bool(self.config.small_a_fallback),
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
        local_mode: bool = False,
    ) -> tuple[
        np.ndarray,
        bool,
        int,
        float | None,
        float | None,
        tuple[BruteForce3PCFAdaptiveTrial, ...],
    ]:
        """Refine Fourier-angle quadrature until the output converges.

        ``adaptive`` globally increases the base grid. ``local_adaptive`` keeps
        the outer grid fixed and increases only the configured hotspot density.
        """
        cfg = self.config
        n_psi = int(cfg.n_psi)
        n_delta_beta = int(cfg.n_delta_beta)
        local_factor = 1
        self._prepare_angular_grid(n_psi, n_delta_beta, local_factor=local_factor)
        previous = self._evaluate_realspace_grid(
            theta1,
            theta2,
            delta_phi,
            delta_phi_chunk=delta_phi_chunk,
        )

        trials: list[BruteForce3PCFAdaptiveTrial] = []
        if cfg.adaptive_store_trials == "all":
            trials.append(
                BruteForce3PCFAdaptiveTrial(
                    refinement=0,
                    n_psi=n_psi,
                    n_delta_beta=n_delta_beta,
                    value=previous,
                    error_norm=None,
                    max_abs_change=None,
                )
            )

        last_norm: float | None = None
        last_abs: float | None = None
        refinement = 0
        while cfg.angular_max_refinements is None or refinement < int(cfg.angular_max_refinements):
            refinement += 1
            if local_mode:
                next_n_psi = n_psi
                next_n_delta_beta = n_delta_beta
                local_factor *= int(cfg.angular_refinement_factor)
            else:
                next_n_psi = n_psi * int(cfg.angular_refinement_factor) if cfg.refine_psi else n_psi
                next_n_delta_beta = (
                    n_delta_beta * int(cfg.angular_refinement_factor)
                    if cfg.refine_delta_beta else n_delta_beta
                )
            self._prepare_angular_grid(next_n_psi, next_n_delta_beta, local_factor=local_factor)
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
            trial = BruteForce3PCFAdaptiveTrial(
                refinement=refinement,
                n_psi=next_n_psi,
                n_delta_beta=next_n_delta_beta,
                value=current,
                error_norm=last_norm,
                max_abs_change=last_abs,
            )
            if cfg.adaptive_store_trials == "all":
                trials.append(trial)
            logger.info(
                "[BruteForceX3PCF._compute_adaptive] angular refinement %d: "
                "n_psi=%d, n_delta_beta=%d, max normalized change=%.3e, "
                "max abs change=%.3e",
                refinement,
                next_n_psi,
                next_n_delta_beta,
                last_norm,
                last_abs,
            )
            if last_norm <= 1.0:
                if cfg.adaptive_store_trials == "last":
                    trials = [trial]
                return current, True, refinement, last_norm, last_abs, tuple(trials)

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
        failure = cfg.adaptive_failure
        if failure is None:
            failure = "raise" if cfg.adaptive_strict else "return_last"
            warn_on_return = not cfg.adaptive_strict
        else:
            warn_on_return = False
        if failure == "raise":
            raise RuntimeError(message)
        if warn_on_return:
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        if cfg.adaptive_store_trials == "last":
            trials = [
                BruteForce3PCFAdaptiveTrial(
                    refinement=refinement,
                    n_psi=n_psi,
                    n_delta_beta=n_delta_beta,
                    value=previous,
                    error_norm=last_norm,
                    max_abs_change=last_abs,
                )
            ]
        return previous, False, refinement, last_norm, last_abs, tuple(trials)

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
            ``"adaptive"`` uses dyadic refinement of the active Fourier-angle
            rectangle and tests convergence on this exact target real-space
            grid.  With ``reduce_domain=True``, each active rectangle
            is the symmetry-reduced fundamental domain and the four physical images
            are included before convergence is assessed.
        """
        theta1 = self._as_positive_grid(theta1, "theta1")
        theta2 = self._as_positive_grid(theta2, "theta2")
        delta_phi = self._as_angle_grid(delta_phi, "delta_phi")
        if int(delta_phi_chunk) < 1:
            raise ValueError("delta_phi_chunk must be >= 1.")

        mode = self.config.angular_mode if angular_mode is None else str(angular_mode)
        if mode not in {"fixed", "adaptive", "local_adaptive"}:
            raise ValueError("angular_mode must be 'fixed', 'adaptive', or 'local_adaptive'.")
        if mode in {"adaptive", "local_adaptive"} and not (self.config.refine_psi or self.config.refine_delta_beta):
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
            adaptive_trials: tuple[BruteForce3PCFAdaptiveTrial, ...] = ()
        else:
            value, converged, refinements, error_norm, error_abs, adaptive_trials = self._compute_adaptive(
                theta1,
                theta2,
                delta_phi,
                delta_phi_chunk=int(delta_phi_chunk),
                local_mode=(mode == "local_adaptive"),
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
            used_reduced_domain=bool(self.config.reduce_domain),
            reduced_domain_exchange_validated=bool(self._reduced_domain_exchange_validated),
            reduced_domain_exchange_error_norm=self._reduced_domain_exchange_error_norm,
            angular_converged=converged,
            angular_refinements=refinements,
            n_psi_used=self._n_psi_current,
            n_delta_beta_used=self._n_delta_beta_current,
            angular_error_norm=error_norm,
            angular_max_abs_change=error_abs,
            adaptive_trials=adaptive_trials,
        )
