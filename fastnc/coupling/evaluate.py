"""Evaluation layer for the Fourier-basis spin coupling kernel."""
from __future__ import annotations

from math import pi
from pathlib import Path
from typing import Iterable

import numpy as np

from .cache import CachePolicy, CouplingCache
from .compute import (
    _as_two_x,
    coupling_delta,
    coupling_delta_float,
    coupling_G,
    exact_zero_delta,
    two_delta_from_L_k,
)


def coupling_delta_from_cache(
    delta: float,
    sigma3: int,
    psi: float | np.ndarray,
    cache_file: str | Path,
    *,
    npsi: int = 1025,
    cache_policy: CachePolicy = "lazy",
    lazy: bool | None = None,
    fallback_direct: bool = True,
    atol: float = 1e-14,
) -> float | np.ndarray:
    """Evaluate G_delta(sigma3;psi) using cached b_{-delta}^{(sigma3/2)}.

    The cache is keyed by ``two_q=sigma3`` and ``two_p=-2*delta``.
    """
    if lazy is not None:
        cache_policy = "lazy" if lazy else "read_only"

    two_delta = _as_two_x(delta, name="delta", atol=atol)
    two_q = int(sigma3)
    two_p = -two_delta

    x = np.asarray(psi, dtype=float)
    scalar_input = x.ndim == 0
    xflat = x.reshape(-1)
    out = np.zeros_like(xflat, dtype=float)

    nonzero_mask = np.array([not exact_zero_delta(two_delta, two_q, float(xx), atol=atol) for xx in xflat])
    if not np.any(nonzero_mask):
        return float(0.0) if scalar_input else out.reshape(x.shape)

    cache = CouplingCache(cache_file)
    try:
        key = cache.get_b(two_q, two_p, two_p, npsi=npsi, policy=cache_policy)
        psi_grid, two_p_grid, b_grid = cache.read_b(key)
        if two_p not in set(int(x) for x in two_p_grid):
            raise KeyError(f"two_p={two_p} not present in cache block {key.group}")
        col = int(np.where(two_p_grid == two_p)[0][0])
        vals = 2 * pi * np.interp(xflat[nonzero_mask], psi_grid, b_grid[:, col])
    except Exception:
        if not fallback_direct:
            raise
        vals = np.array([coupling_delta(two_delta, two_q, float(xx), atol=atol) for xx in xflat[nonzero_mask]])

    vals[np.abs(vals) < 10 * np.finfo(float).eps] = 0.0
    out[nonzero_mask] = vals
    return float(out[0]) if scalar_input else out.reshape(x.shape)


def coupling_from_cache(
    L: int,
    k: float,
    sigma: Iterable[int],
    psi: float | np.ndarray,
    cache_file: str | Path,
    *,
    npsi: int = 1025,
    cache_policy: CachePolicy = "lazy",
    lazy: bool | None = None,
    fallback_direct: bool = True,
    atol: float = 1e-14,
) -> float | np.ndarray:
    """Backward-compatible wrapper for G_{Lk}(sigma;psi)."""
    sigma_tuple = tuple(int(x) for x in sigma)
    two_delta = two_delta_from_L_k(int(L), k, sigma_tuple, atol=atol)
    x = np.asarray(psi, dtype=float)
    if two_delta is None:
        return float(0.0) if x.ndim == 0 else np.zeros_like(x, dtype=float)
    return coupling_delta_from_cache(
        0.5 * two_delta,
        sigma_tuple[2],
        psi,
        cache_file,
        npsi=npsi,
        cache_policy=cache_policy,
        lazy=lazy,
        fallback_direct=fallback_direct,
        atol=atol,
    )


def coupling(
    L: int,
    k: float,
    sigma: Iterable[int],
    psi: float | np.ndarray,
    *,
    cache_file: str | Path | None = None,
    npsi: int = 1025,
    cache_policy: CachePolicy = "lazy",
) -> float | np.ndarray:
    x = np.asarray(psi, dtype=float)
    scalar_input = x.ndim == 0
    if cache_file is not None:
        return coupling_from_cache(L, k, sigma, x, cache_file, npsi=npsi, cache_policy=cache_policy)
    vals = np.array([coupling_G(int(L), k, sigma, float(xx)) for xx in x.reshape(-1)])
    return float(vals[0]) if scalar_input else vals.reshape(x.shape)


def coupling_by_delta(
    delta: float,
    sigma3: int,
    psi: float | np.ndarray,
    *,
    cache_file: str | Path | None = None,
    npsi: int = 1025,
    cache_policy: CachePolicy = "lazy",
) -> float | np.ndarray:
    """Evaluate G_delta(sigma3;psi), where delta=L-nu_k."""
    x = np.asarray(psi, dtype=float)
    scalar_input = x.ndim == 0
    if cache_file is not None:
        return coupling_delta_from_cache(delta, sigma3, x, cache_file, npsi=npsi, cache_policy=cache_policy)
    vals = np.array([coupling_delta_float(delta, sigma3, float(xx)) for xx in x.reshape(-1)])
    return float(vals[0]) if scalar_input else vals.reshape(x.shape)
