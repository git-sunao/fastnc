"""Callable objects for the Fourier-basis spin coupling kernel.

This module provides two callable objects:

- CouplingKernel(sigma3): natural API for the Fourier-basis kernel
  G_delta(sigma3; psi), where delta = L - nu_k.

- CouplingMatrix(sigma1, sigma2, sigma3): backward-compatible API that
  accepts (L, k) at call time, converts them to delta = L - nu_k, and then
  delegates to CouplingKernel.

The important implementation detail is that CouplingKernel.strength() uses
_call_delta() rather than self(...).  This avoids accidental dispatch to
CouplingMatrix.__call__() when CouplingMatrix subclasses CouplingKernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from .cache import CachePolicy, CouplingCache
from .compute import _as_two_x, coupling_delta, two_delta_from_L_k
from .evaluate import coupling_delta_from_cache

Method = Literal["auto", "cache", "direct"]


@dataclass(frozen=True)
class CouplingKernelConfig:
    sigma3: int
    use_cache: bool = True
    cache_file: str | Path = "coupling_b_cache.h5"
    npsi: int = 1025
    cache_policy: CachePolicy = "lazy"
    fallback_direct: bool = True
    atol: float = 1e-14


class CouplingKernel:
    """Callable object for G_delta(sigma3; psi) with fixed sigma3.

    The natural label is

        delta = L - nu_k,

    so this object is independent of sigma1 and sigma2.
    """

    def __init__(
        self,
        sigma3: int,
        *,
        use_cache: bool = True,
        cache_file: str | Path = "coupling_b_cache.h5",
        npsi: int = 1025,
        cache_policy: CachePolicy = "lazy",
        lazy: bool | None = None,
        fallback_direct: bool = True,
        atol: float = 1e-14,
    ) -> None:
        if lazy is not None:
            cache_policy = "lazy" if lazy else "read_only"

        self.config = CouplingKernelConfig(
            sigma3=int(sigma3),
            use_cache=bool(use_cache),
            cache_file=Path(cache_file),
            npsi=int(npsi),
            cache_policy=cache_policy,
            fallback_direct=bool(fallback_direct),
            atol=float(atol),
        )
        if self.config.npsi < 3:
            raise ValueError("npsi must be >= 3.")

        self._cache = CouplingCache(self.config.cache_file) if self.config.use_cache else None

    @property
    def sigma3(self) -> int:
        return self.config.sigma3

    @property
    def two_q(self) -> int:
        return self.sigma3

    @property
    def q(self) -> float:
        return 0.5 * self.sigma3

    @property
    def cache(self) -> CouplingCache:
        if self._cache is None:
            self._cache = CouplingCache(self.config.cache_file)
        return self._cache

    def __repr__(self) -> str:
        mode = "cache" if self.config.use_cache else "direct"
        return (
            f"CouplingKernel(sigma3={self.sigma3}, q={self.q}, default={mode}, "
            f"cache_file='{self.config.cache_file}', npsi={self.config.npsi}, "
            f"cache_policy='{self.config.cache_policy}')"
        )

    def _resolve_method(
        self,
        *,
        method: Method | None,
        use_cache: bool | None,
    ) -> Literal["cache", "direct"]:
        if method is not None and use_cache is not None:
            raise ValueError("Specify either method or use_cache, not both.")

        if method is None or method == "auto":
            default_use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
            return "cache" if default_use_cache else "direct"

        if method == "cache":
            return "cache"
        if method == "direct":
            return "direct"

        raise ValueError("method must be one of None, 'auto', 'cache', or 'direct'.")

    def _call_delta(
        self,
        delta: int | float,
        psi: float | np.ndarray,
        *,
        method: Method | None = None,
        use_cache: bool | None = None,
        cache_file: str | Path | None = None,
        npsi: int | None = None,
        cache_policy: CachePolicy | None = None,
        lazy: bool | None = None,
        fallback_direct: bool | None = None,
    ) -> float | np.ndarray:
        """Evaluate the delta-labelled kernel without using dynamic __call__ dispatch.

        This method is intentionally separate from __call__ so that subclasses
        can define their own public __call__ signatures without breaking
        CouplingKernel.strength().
        """
        selected = self._resolve_method(method=method, use_cache=use_cache)

        if selected == "direct":
            x = np.asarray(psi, dtype=float)
            scalar_input = x.ndim == 0
            two_delta = _as_two_x(delta, name="delta", atol=self.config.atol)
            vals = np.array(
                [
                    coupling_delta(two_delta, self.sigma3, float(xx), atol=self.config.atol)
                    for xx in x.reshape(-1)
                ]
            )
            return float(vals[0]) if scalar_input else vals.reshape(x.shape)

        cfile = Path(cache_file) if cache_file is not None else self.config.cache_file
        return coupling_delta_from_cache(
            float(delta),
            self.sigma3,
            psi,
            cfile,
            npsi=self.config.npsi if npsi is None else int(npsi),
            cache_policy=self.config.cache_policy if cache_policy is None else cache_policy,
            lazy=lazy,
            fallback_direct=(
                self.config.fallback_direct if fallback_direct is None else bool(fallback_direct)
            ),
            atol=self.config.atol,
        )

    def __call__(
        self,
        delta: int | float,
        psi: float | np.ndarray,
        *,
        method: Method | None = None,
        use_cache: bool | None = None,
        cache_file: str | Path | None = None,
        npsi: int | None = None,
        cache_policy: CachePolicy | None = None,
        lazy: bool | None = None,
        fallback_direct: bool | None = None,
    ) -> float | np.ndarray:
        return self._call_delta(
            delta,
            psi,
            method=method,
            use_cache=use_cache,
            cache_file=cache_file,
            npsi=npsi,
            cache_policy=cache_policy,
            lazy=lazy,
            fallback_direct=fallback_direct,
        )

    def ensure_cache_for_delta(self, delta: int | float, *, npsi: int | None = None) -> None:
        two_delta = _as_two_x(delta, name="delta", atol=self.config.atol)
        two_p = -two_delta
        self.cache.ensure_b(
            self.two_q,
            two_p,
            two_p,
            npsi=self.config.npsi if npsi is None else int(npsi),
        )

    def strength(
        self,
        delta: int | float,
        psi: np.ndarray | None = None,
        *,
        npsi_strength: int | None = None,
        average: Literal["integral", "mean"] = "integral",
        rms: bool = False,
        method: Method | None = None,
        use_cache: bool | None = None,
        cache_file: str | Path | None = None,
        npsi: int | None = None,
        cache_policy: CachePolicy | None = None,
        lazy: bool | None = None,
        fallback_direct: bool | None = None,
    ) -> float:
        """Return <|G_delta(psi)|^2>_psi, or its square root if rms=True."""
        if psi is None:
            ngrid = self.config.npsi if npsi_strength is None else int(npsi_strength)
            if ngrid < 2:
                raise ValueError("npsi_strength must be >= 2.")
            psi_arr = np.linspace(0.0, 0.5 * np.pi, ngrid)
        else:
            psi_arr = np.asarray(psi, dtype=float)
            if psi_arr.ndim != 1:
                raise ValueError("psi must be a one-dimensional array for strength().")
            if psi_arr.size < 2:
                raise ValueError("psi must contain at least two points for strength().")

        vals = np.asarray(
            self._call_delta(
                delta,
                psi=psi_arr,
                method=method,
                use_cache=use_cache,
                cache_file=cache_file,
                npsi=npsi,
                cache_policy=cache_policy,
                lazy=lazy,
                fallback_direct=fallback_direct,
            ),
            dtype=float,
        )
        sq = np.real(vals * np.conjugate(vals))

        if average == "integral":
            denom = float(psi_arr[-1] - psi_arr[0])
            if denom <= 0.0:
                raise ValueError("psi must be increasing with nonzero range for integral averaging.")
            strength = float(np.trapezoid(sq, psi_arr) / denom)
        elif average == "mean":
            strength = float(np.mean(sq))
        else:
            raise ValueError("average must be either 'integral' or 'mean'.")

        return float(np.sqrt(strength)) if rms else strength

    def strength_grid(
        self,
        delta_values: Iterable[int | float],
        psi: np.ndarray | None = None,
        *,
        npsi_strength: int | None = None,
        average: Literal["integral", "mean"] = "integral",
        rms: bool = False,
        method: Method | None = None,
        use_cache: bool | None = None,
        cache_file: str | Path | None = None,
        npsi: int | None = None,
        cache_policy: CachePolicy | None = None,
        lazy: bool | None = None,
        fallback_direct: bool | None = None,
    ) -> np.ndarray:
        deltas = list(delta_values)
        out = np.empty(len(deltas), dtype=float)
        for i, delta in enumerate(deltas):
            out[i] = self.strength(
                delta,
                psi,
                npsi_strength=npsi_strength,
                average=average,
                rms=rms,
                method=method,
                use_cache=use_cache,
                cache_file=cache_file,
                npsi=npsi,
                cache_policy=cache_policy,
                lazy=lazy,
                fallback_direct=fallback_direct,
            )
        return out

    def list_cache_blocks(self) -> list[str]:
        return self.cache.list_blocks()

    def describe_cache(self) -> None:
        self.cache.describe()


@dataclass(frozen=True)
class CouplingMatrixConfig(CouplingKernelConfig):
    sigma: tuple[int, int, int] = (0, 0, 0)


class CouplingMatrix(CouplingKernel):
    """Backward-compatible callable object for G_{Lk}(sigma; psi).

    The object accepts sigma1, sigma2, sigma3 at initialization and L, k at
    call time, but internally computes delta = L - nu_k and delegates to the
    CouplingKernel delta implementation.  Thus the actual cached/evaluated
    object depends only on delta and sigma3.
    """

    def __init__(
        self,
        sigma1: int,
        sigma2: int,
        sigma3: int,
        *,
        use_cache: bool = True,
        cache_file: str | Path = "coupling_b_cache.h5",
        npsi: int = 1025,
        cache_policy: CachePolicy = "lazy",
        lazy: bool | None = None,
        fallback_direct: bool = True,
        atol: float = 1e-14,
    ) -> None:
        self.sigma1 = int(sigma1)
        self.sigma2 = int(sigma2)
        super().__init__(
            int(sigma3),
            use_cache=use_cache,
            cache_file=cache_file,
            npsi=npsi,
            cache_policy=cache_policy,
            lazy=lazy,
            fallback_direct=fallback_direct,
            atol=atol,
        )

    @property
    def sigma(self) -> tuple[int, int, int]:
        return (self.sigma1, self.sigma2, self.sigma3)

    def __repr__(self) -> str:
        mode = "cache" if self.config.use_cache else "direct"
        return (
            f"CouplingMatrix(sigma={self.sigma}, q={self.q}, default={mode}, "
            f"cache_file='{self.config.cache_file}', npsi={self.config.npsi}, "
            f"cache_policy='{self.config.cache_policy}')"
        )

    def delta(self, L: int, k: int | float) -> float | None:
        two_delta = two_delta_from_L_k(int(L), k, self.sigma, atol=self.config.atol)
        if two_delta is None:
            return None
        return 0.5 * two_delta

    def __call__(
        self,
        L: int,
        k: int | float,
        psi: float | np.ndarray,
        *,
        method: Method | None = None,
        use_cache: bool | None = None,
        cache_file: str | Path | None = None,
        npsi: int | None = None,
        cache_policy: CachePolicy | None = None,
        lazy: bool | None = None,
        fallback_direct: bool | None = None,
    ) -> float | np.ndarray:
        delta = self.delta(int(L), k)
        x = np.asarray(psi, dtype=float)
        if delta is None:
            return float(0.0) if x.ndim == 0 else np.zeros_like(x, dtype=float)

        return self._call_delta(
            delta,
            psi,
            method=method,
            use_cache=use_cache,
            cache_file=cache_file,
            npsi=npsi,
            cache_policy=cache_policy,
            lazy=lazy,
            fallback_direct=fallback_direct,
        )

    def ensure_cache_for(self, L: int, k: int | float, *, npsi: int | None = None) -> None:
        delta = self.delta(int(L), k)
        if delta is None:
            return
        self.ensure_cache_for_delta(delta, npsi=npsi)

    def strength(self, L: int, k: int | float, psi: np.ndarray | None = None, **kwargs) -> float:
        delta = self.delta(int(L), k)
        if delta is None:
            return 0.0
        return super().strength(delta, psi=psi, **kwargs)

    def strength_grid(
        self,
        L_values: Iterable[int],
        k_values: Iterable[int | float],
        psi: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        L_arr = [int(L) for L in L_values]
        k_arr = list(k_values)
        out = np.empty((len(L_arr), len(k_arr)), dtype=float)
        for i, L in enumerate(L_arr):
            for j, k in enumerate(k_arr):
                out[i, j] = self.strength(L, k, psi, **kwargs)
        return out
