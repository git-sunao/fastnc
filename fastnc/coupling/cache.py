"""Appendable HDF5 cache for spin-phase Fourier coefficients b_p^{(q)}(psi).

The cache stores b_p^{(q)}(psi) rather than G_{Lk}.  The HDF5 key uses
``two_q = 2 q = sigma_3`` instead of a floating-point representation of q,
so integer and half-integer q are both represented exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from .compute import b_array_from_two_q

CachePolicy = Literal["read_only", "lazy", "refresh"]


@dataclass(frozen=True)
class BCacheKey:
    two_q: int
    two_p_min: int
    two_p_max: int
    npsi: int

    @property
    def group(self) -> str:
        return f"b/two_q_{self.two_q:+d}/two_p_{self.two_p_min}_{self.two_p_max}/npsi_{self.npsi}"

    @property
    def q(self) -> float:
        return 0.5 * self.two_q


class CouplingCache:
    """Lazy, appendable HDF5 cache for b_p^{(q)}(psi)."""

    def __init__(self, filename: str | Path):
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default_psi_grid(npsi: int, *, include_endpoints: bool = True) -> np.ndarray:
        if npsi < 3:
            raise ValueError("npsi must be >= 3.")
        if include_endpoints:
            return np.linspace(0.0, 0.5 * np.pi, npsi)
        return np.linspace(0.0, 0.5 * np.pi, npsi + 2)[1:-1]

    @staticmethod
    def key(two_q: int, two_p_min: int, two_p_max: int, npsi: int) -> BCacheKey:
        return BCacheKey(int(two_q), int(two_p_min), int(two_p_max), int(npsi))

    def has_b(self, two_q: int, two_p_min: int, two_p_max: int, npsi: int) -> bool:
        key = self.key(two_q, two_p_min, two_p_max, npsi)
        if not self.filename.exists():
            return False
        with h5py.File(self.filename, "r") as h5:
            return key.group in h5

    def build_b(self, two_q: int, two_p_min: int, two_p_max: int, *, npsi: int = 1025, overwrite: bool = False) -> BCacheKey:
        """Append one b-cache block if missing, and return its key."""
        if two_p_min > two_p_max:
            raise ValueError("two_p_min must be <= two_p_max.")
        key = self.key(two_q, two_p_min, two_p_max, npsi)
        psi = self.default_psi_grid(npsi)
        two_p_values = np.arange(two_p_min, two_p_max + 1, 2, dtype=int)

        with h5py.File(self.filename, "a") as h5:
            h5.attrs["format"] = "multipole_coupling_b_cache_v2"
            h5.attrs["description"] = "b_p^(q)(psi) cache keyed by two_q=2q=sigma3 and two_p=2p"
            if key.group in h5:
                if not overwrite:
                    return key
                del h5[key.group]
            values = b_array_from_two_q(two_q, two_p_values, psi)
            g = h5.create_group(key.group)
            g.attrs["two_q"] = int(two_q)
            g.attrs["q"] = 0.5 * int(two_q)
            g.attrs["two_p_min"] = int(two_p_min)
            g.attrs["two_p_max"] = int(two_p_max)
            g.attrs["npsi"] = int(npsi)
            g.create_dataset("psi", data=psi, compression="gzip", shuffle=True)
            g.create_dataset("two_p", data=two_p_values, compression="gzip", shuffle=True)
            g.create_dataset("b", data=values, compression="gzip", shuffle=True)
        return key

    def get_b(self, two_q: int, two_p_min: int, two_p_max: int, *, npsi: int = 1025, policy: CachePolicy = "lazy") -> BCacheKey:
        """Return a cache key according to ``read_only``, ``lazy``, or ``refresh``."""
        key = self.key(two_q, two_p_min, two_p_max, npsi)
        if policy == "read_only":
            if not self.has_b(two_q, two_p_min, two_p_max, npsi):
                raise KeyError(f"Cache block not found: {key.group}")
            return key
        if policy == "lazy":
            if self.has_b(two_q, two_p_min, two_p_max, npsi):
                return key
            return self.build_b(two_q, two_p_min, two_p_max, npsi=npsi, overwrite=False)
        if policy == "refresh":
            return self.build_b(two_q, two_p_min, two_p_max, npsi=npsi, overwrite=True)
        raise ValueError("policy must be one of 'read_only', 'lazy', or 'refresh'.")

    def ensure_b(self, two_q: int, two_p_min: int, two_p_max: int, *, npsi: int = 1025) -> BCacheKey:
        return self.get_b(two_q, two_p_min, two_p_max, npsi=npsi, policy="lazy")

    def read_b(self, key: BCacheKey) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(self.filename, "r") as h5:
            if key.group not in h5:
                raise KeyError(f"Cache block not found: {key.group}")
            g = h5[key.group]
            return g["psi"][:], g["two_p"][:], g["b"][:]

    def list_blocks(self) -> list[str]:
        if not self.filename.exists():
            return []
        out: list[str] = []
        with h5py.File(self.filename, "r") as h5:
            def visit(name: str, obj):
                if isinstance(obj, h5py.Group) and {"psi", "two_p", "b"}.issubset(set(obj.keys())):
                    out.append(name)
            h5.visititems(visit)
        return out

    def describe(self) -> None:
        if not self.filename.exists():
            print(f"Cache file does not exist: {self.filename}")
            return
        with h5py.File(self.filename, "r") as h5:
            for k, v in h5.attrs.items():
                print(f"@{k} = {v}")
            def visit(name: str, obj):
                indent = "  " * name.count("/")
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}{name}: dataset shape={obj.shape}, dtype={obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}{name}/")
                    for k, v in obj.attrs.items():
                        print(f"{indent}  @{k} = {v}")
            h5.visititems(visit)
