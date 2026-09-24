"""Configuration for the unified ThreePCF route manager."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastnc.coupling import CachePolicy
from fastnc.hankel import DoubleHankelConfig


@dataclass(frozen=True)
class NumericRouteConfig:
    """Numerical multipole coupling and double-Hankel settings."""

    Lmax: int = 30
    kmax: float = 30.0
    use_coupling_cache: bool = True
    coupling_cache_file: str | Path | None = None
    coupling_npsi: int = 1025
    coupling_cache_policy: CachePolicy = "lazy"
    coupling_fallback_direct: bool = True
    coupling_atol: float = 1.0e-14
    hankel: DoubleHankelConfig = field(default_factory=DoubleHankelConfig)
    bin_width_logtheta: float | None = None

    def __post_init__(self):
        if int(self.Lmax) < 0:
            raise ValueError("Lmax must be non-negative")
        if float(self.kmax) < 0.0:
            raise ValueError("kmax must be non-negative")
        if int(self.coupling_npsi) < 3:
            raise ValueError("coupling_npsi must be at least three")
        if self.coupling_cache_policy not in {"read_only", "lazy", "refresh"}:
            raise ValueError("unsupported coupling_cache_policy")
        if (
            self.bin_width_logtheta is not None
            and float(self.bin_width_logtheta) <= 0.0
        ):
            raise ValueError("bin_width_logtheta must be positive")
        object.__setattr__(self, "Lmax", int(self.Lmax))
        object.__setattr__(self, "kmax", float(self.kmax))
        object.__setattr__(self, "coupling_npsi", int(self.coupling_npsi))

    def coupling_kwargs(self) -> dict[str, object]:
        return {
            "use_cache": bool(self.use_coupling_cache),
            "cache_file": self.coupling_cache_file,
            "npsi": self.coupling_npsi,
            "cache_policy": self.coupling_cache_policy,
            "fallback_direct": bool(self.coupling_fallback_direct),
            "atol": float(self.coupling_atol),
        }
