"""Public high-level 3PCF API."""
from __future__ import annotations

from ..bispectrum import BispectrumMultipole2DConfig
from .bmultipole_grid import BMultipoleGrid
from .calculator import HybridThreePCFCalculator
from .config import ThreePCFConfig
from .grid import FFTGrid
from .hkernel_grid import HKernel, HKernelGrid, HKernelKey
from .zeta_grid import ZetaGrid
from .zetak_grid import ZetaKGrid, ZetaKKey, ZetaKMode


class ThreePCF:
    """High-level facade for generic and future specialized 3PCF routes.

    Prefer the explicit named constructors :meth:`from_3d`, :meth:`from_2d`,
    and :meth:`from_multipole`.  The historical direct constructor remains as
    a compatibility dispatcher and retains generic-only behavior.
    """

    def __init__(
        self,
        bispectrum,
        config: ThreePCFConfig | None = None,
        *,
        coupling_kwargs: dict | None = None,
        multipole_config: BispectrumMultipole2DConfig | None = None,
        multipole_basis: str = "fourier-even",
        regulator=None,
        multipole_kwargs: dict | None = None,
    ):
        self.bispectrum = bispectrum
        self.config = config or ThreePCFConfig()
        self.coupling_kwargs = coupling_kwargs
        self.multipole_config = multipole_config
        self.multipole_basis = multipole_basis
        self.regulator = regulator
        self.multipole_kwargs = dict(multipole_kwargs or {})

        self._input_kind = "compat"
        self.bispectrum3d = None
        self.bispectrum2d = bispectrum
        self.projector = None
        self.sample_combinations = None
        self.bmultipole = None
        self.calculator: HybridThreePCFCalculator | None = None

    @classmethod
    def from_3d(
        cls,
        bispectrum,
        projector,
        *,
        config: ThreePCFConfig | None = None,
        sample_combinations=None,
        coupling_kwargs: dict | None = None,
        multipole_config: BispectrumMultipole2DConfig | None = None,
        multipole_basis: str = "fourier-even",
        regulator=None,
        multipole_kwargs: dict | None = None,
    ):
        """Construct a hybrid-capable 3PCF from an unprojected 3D model."""
        obj = cls.__new__(cls)
        obj.bispectrum = bispectrum
        obj.bispectrum3d = bispectrum
        obj.bispectrum2d = None
        obj.projector = projector
        obj.sample_combinations = sample_combinations
        obj.config = config or ThreePCFConfig()
        obj.coupling_kwargs = coupling_kwargs
        obj.multipole_config = multipole_config
        obj.multipole_basis = multipole_basis
        obj.regulator = regulator
        obj.multipole_kwargs = dict(multipole_kwargs or {})
        obj._input_kind = "3d"
        obj.bmultipole = None
        obj.calculator = None
        return obj

    @classmethod
    def from_2d(
        cls,
        bispectrum,
        *,
        config: ThreePCFConfig | None = None,
        coupling_kwargs: dict | None = None,
        multipole_config: BispectrumMultipole2DConfig | None = None,
        multipole_basis: str = "fourier-even",
        regulator=None,
        multipole_kwargs: dict | None = None,
    ):
        """Construct a generic-route 3PCF from an already projected 2D bispectrum."""
        obj = cls(
            bispectrum,
            config=config,
            coupling_kwargs=coupling_kwargs,
            multipole_config=multipole_config,
            multipole_basis=multipole_basis,
            regulator=regulator,
            multipole_kwargs=multipole_kwargs,
        )
        obj._input_kind = "2d"
        return obj

    @classmethod
    def from_multipole(
        cls,
        bmultipole,
        *,
        config: ThreePCFConfig | None = None,
        coupling_kwargs: dict | None = None,
    ):
        """Construct a generic-route 3PCF from a precomputed/callable multipole object."""
        obj = cls.__new__(cls)
        obj.bispectrum = bmultipole
        obj.bispectrum3d = None
        obj.bispectrum2d = None
        obj.projector = None
        obj.sample_combinations = None
        obj.config = config or ThreePCFConfig()
        obj.coupling_kwargs = coupling_kwargs
        obj.multipole_config = None
        obj.multipole_basis = getattr(bmultipole, "basis", "fourier-even")
        obj.regulator = None
        obj.multipole_kwargs = {}
        obj._input_kind = "multipole"
        obj.bmultipole = bmultipole
        obj.calculator = None
        return obj

    @property
    def grid(self) -> FFTGrid:
        return self._require_calculator().grid

    @property
    def Bgrid(self) -> BMultipoleGrid:
        return self._require_calculator().Bgrid

    @property
    def Hgrid(self) -> HKernelGrid:
        return self._require_calculator().Hgrid

    @property
    def ZKgrid(self) -> ZetaKGrid:
        return self._require_calculator().ZKgrid

    @property
    def Zgrid(self) -> ZetaGrid | None:
        return self._require_calculator().Zgrid

    @property
    def timings(self):
        """Per-route elapsed wall times for the latest calculation stages."""
        return self._require_calculator().timings

    def _require_calculator(self) -> HybridThreePCFCalculator:
        if self.calculator is None:
            raise RuntimeError("A compute method must be called before accessing stage grids.")
        return self.calculator

    def _ensure_calculator(self) -> HybridThreePCFCalculator:
        if self.calculator is None:
            common = dict(
                config=self.config,
                coupling_kwargs=self.coupling_kwargs,
                multipole_config=self.multipole_config,
                multipole_basis=self.multipole_basis,
                regulator=self.regulator,
                multipole_kwargs=self.multipole_kwargs,
            )
            if self._input_kind == "3d":
                self.calculator = HybridThreePCFCalculator(
                    bispectrum3d=self.bispectrum3d,
                    projector=self.projector,
                    sample_combinations=self.sample_combinations,
                    **common,
                )
            elif self._input_kind == "multipole":
                self.calculator = HybridThreePCFCalculator(
                    bmultipole=self.bmultipole,
                    **common,
                )
            else:
                # Compatibility/direct and explicit 2D paths are generic-only.
                if hasattr(self.bispectrum2d, "basis") and callable(self.bispectrum2d):
                    self.calculator = HybridThreePCFCalculator(
                        bmultipole=self.bispectrum2d,
                        **common,
                    )
                else:
                    self.calculator = HybridThreePCFCalculator(
                        bispectrum2d=self.bispectrum2d,
                        **common,
                    )
            self.bmultipole = self.calculator.bmultipole
        return self.calculator

    def make_multipole(self):
        """Compatibility helper returning the generic multipole representation."""
        return self._ensure_calculator().bmultipole

    def compute_bmultipoles(self, *args, **kwargs) -> BMultipoleGrid:
        return self._ensure_calculator().compute_bmultipoles(*args, **kwargs)

    def compute_hkernels(self, *args, **kwargs) -> HKernelGrid:
        return self._ensure_calculator().compute_hkernels(*args, **kwargs)

    def compute_zetak(self, *args, **kwargs) -> ZetaKGrid:
        return self._ensure_calculator().compute_zetak(*args, **kwargs)

    compute_zeta_k = compute_zetak

    def compute_zeta(self, delta_phi, *args, **kwargs) -> ZetaGrid:
        return self._ensure_calculator().compute_zeta(delta_phi, *args, **kwargs)

    def compute(self, delta_phi, *args, **kwargs) -> ZetaGrid:
        return self.compute_zeta(delta_phi, *args, **kwargs)

    def get_H_kernel(self, key: HKernelKey) -> HKernel:
        return self.Hgrid.get(key)

    def get_zeta_k(self, key: ZetaKKey) -> ZetaKMode:
        return self.ZKgrid.get(key)
