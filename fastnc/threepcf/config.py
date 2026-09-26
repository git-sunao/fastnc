"""Configuration for the unified ThreePCF calculation object."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastnc.coupling import CachePolicy
from fastnc.hankel import DoubleHankelConfig
from fastnc.multipole import NumericMultipoleConfig


@dataclass(frozen=True)
class SlepianConfig:
    """Numerical controls for FFTLog and Weber-Schafheitlin transforms."""

    bias: float = 0.0
    taper_fraction: float = 0.0
    window_fraction: float = 0.0
    weber_rtol: float = 1.0e-12
    weber_method: Literal["direct", "interpolated"] = "direct"
    weber_interpolation_nodes: int = 64
    weber_interpolation_max_ratio: float = 0.8
    diagonal_correction: Literal["brute", "none"] = "brute"
    regular_method: Literal[
        "quadrature", "full_matrix", "low_rank"
    ] = "quadrature"
    regular_n_x: int = 256
    regular_x_padding: float = 20.0
    regular_low_rank_rank: int | None = None
    regular_low_rank_rtol: float = 1.0e-6

    def __post_init__(self):
        for name in ("taper_fraction", "window_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 0.5:
                raise ValueError(f"{name} must lie in [0, 0.5)")
            object.__setattr__(self, name, value)
        if float(self.weber_rtol) <= 0.0:
            raise ValueError("weber_rtol must be positive")
        if self.weber_method not in {"direct", "interpolated"}:
            raise ValueError("weber_method must be 'direct' or 'interpolated'")
        if int(self.weber_interpolation_nodes) < 4:
            raise ValueError("weber_interpolation_nodes must be at least four")
        if not 0.0 < float(self.weber_interpolation_max_ratio) < 1.0:
            raise ValueError("weber_interpolation_max_ratio must lie in (0, 1)")
        if self.diagonal_correction not in {"brute", "none"}:
            raise ValueError("diagonal_correction must be 'brute' or 'none'")
        if self.regular_method not in {"quadrature", "full_matrix", "low_rank"}:
            raise ValueError(
                "regular_method must be 'quadrature', 'full_matrix', or 'low_rank'"
            )
        if (
            self.regular_low_rank_rank is not None
            and int(self.regular_low_rank_rank) < 1
        ):
            raise ValueError("regular_low_rank_rank must be positive")
        if not 0.0 < float(self.regular_low_rank_rtol) < 1.0:
            raise ValueError("regular_low_rank_rtol must lie in (0, 1)")
        if int(self.regular_n_x) < 16:
            raise ValueError("regular_n_x must be at least 16")
        if float(self.regular_x_padding) <= 1.0:
            raise ValueError("regular_x_padding must be greater than one")
        object.__setattr__(self, "bias", float(self.bias))
        object.__setattr__(self, "weber_rtol", float(self.weber_rtol))
        object.__setattr__(
            self, "weber_interpolation_nodes", int(self.weber_interpolation_nodes)
        )
        object.__setattr__(
            self,
            "weber_interpolation_max_ratio",
            float(self.weber_interpolation_max_ratio),
        )
        object.__setattr__(self, "regular_n_x", int(self.regular_n_x))
        object.__setattr__(
            self, "regular_x_padding", float(self.regular_x_padding)
        )
        if self.regular_low_rank_rank is not None:
            object.__setattr__(
                self, "regular_low_rank_rank", int(self.regular_low_rank_rank)
            )
        object.__setattr__(
            self, "regular_low_rank_rtol", float(self.regular_low_rank_rtol)
        )


@dataclass(frozen=True)
class ThreePCFConfig:
    """Physical conventions, Fourier grid, and route settings."""

    spin: tuple[int, int, int] = (0, 0, 0)
    basis: Literal["cosine", "sine", "fourier", "legendre"] = "fourier"
    Lmax: int = 30
    kmax: float = 30.0
    ell_min: float = 1.0e-1
    ell_max: float = 1.0e5
    n_ell: int = 200
    multipole: NumericMultipoleConfig = field(
        default_factory=NumericMultipoleConfig
    )
    use_coupling_cache: bool = True
    coupling_cache_file: str | Path | None = None
    coupling_npsi: int = 1025
    coupling_cache_policy: CachePolicy = "lazy"
    coupling_fallback_direct: bool = True
    coupling_atol: float = 1.0e-14
    hankel: DoubleHankelConfig = field(default_factory=DoubleHankelConfig)
    slepian: SlepianConfig = field(default_factory=SlepianConfig)
    bin_width_logtheta: float | None = None

    def __post_init__(self):
        spin = tuple(int(value) for value in self.spin)
        if len(spin) != 3:
            raise ValueError("spin must contain exactly three entries")
        if self.basis not in {"cosine", "sine", "fourier", "legendre"}:
            raise ValueError(
                "basis must be 'cosine', 'sine', 'fourier', or 'legendre'"
            )
        if int(self.Lmax) < 0:
            raise ValueError("Lmax must be non-negative")
        if self.basis == "sine" and int(self.Lmax) < 1:
            raise ValueError("Lmax must be at least one for the sine basis")
        if float(self.kmax) < 0.0:
            raise ValueError("kmax must be non-negative")
        if float(self.ell_min) <= 0.0:
            raise ValueError("ell_min must be positive")
        if float(self.ell_max) <= float(self.ell_min):
            raise ValueError("ell_max must be greater than ell_min")
        if int(self.n_ell) < 2:
            raise ValueError("n_ell must be at least two")
        if not isinstance(self.multipole, NumericMultipoleConfig):
            raise TypeError("multipole must be a NumericMultipoleConfig")
        if not isinstance(self.slepian, SlepianConfig):
            raise TypeError("slepian must be a SlepianConfig")
        if int(self.coupling_npsi) < 3:
            raise ValueError("coupling_npsi must be at least three")
        if self.coupling_cache_policy not in {"read_only", "lazy", "refresh"}:
            raise ValueError("unsupported coupling_cache_policy")
        if (
            self.bin_width_logtheta is not None
            and float(self.bin_width_logtheta) <= 0.0
        ):
            raise ValueError("bin_width_logtheta must be positive")
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "Lmax", int(self.Lmax))
        object.__setattr__(self, "kmax", float(self.kmax))
        object.__setattr__(self, "ell_min", float(self.ell_min))
        object.__setattr__(self, "ell_max", float(self.ell_max))
        object.__setattr__(self, "n_ell", int(self.n_ell))
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
