"""Public high-level 3PCF API."""
from __future__ import annotations

from ..bispectrum import BispectrumMultipole2DConfig
from .bmultipole_grid import BMultipoleGrid
from .calculator import ThreePCFCalculator
from .config import ThreePCFConfig
from .grid import FFTGrid
from .hkernel_grid import HKernel, HKernelGrid, HKernelKey
from .zeta_grid import ZetaGrid
from .zetak_grid import ZetaKGrid, ZetaKKey, ZetaKMode


class ThreePCF:
    """High-level object for 3PCF calculations.

    The preferred user-facing method is :meth:`compute_zeta`, which returns a
    :class:`ZetaGrid`.  Stage methods are also exposed for debugging:

    ``compute_bmultipoles() -> BMultipoleGrid``
    ``compute_hkernels() -> HKernelGrid``
    ``compute_zetak() -> ZetaKGrid``
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

        self.bmultipole = None
        self.calculator: ThreePCFCalculator | None = None

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
    def timings(self) -> dict[str, float]:
        """Elapsed wall times (seconds) for the latest calculation stages."""
        return dict(self._require_calculator().timings)

    def _require_calculator(self) -> ThreePCFCalculator:
        if self.calculator is None:
            raise RuntimeError("A compute method must be called before accessing stage grids.")
        return self.calculator

    def _default_multipole_config(self) -> BispectrumMultipole2DConfig:
        return BispectrumMultipole2DConfig(
            mode_max=int(self.config.Lmax),
            ell_min=float(self.config.ell_min),
            ell_max=float(self.config.ell_max),
            n_ell=int(self.config.n_ell),
        )

    def make_multipole(self):
        """Return a bispectrum-multipole object compatible with the 3PCF engine."""
        if hasattr(self.bispectrum, "basis") and callable(self.bispectrum):
            return self.bispectrum
        if not hasattr(self.bispectrum, "multipole"):
            raise TypeError(
                "bispectrum must be either a callable multipole object with a basis attribute "
                "or an object exposing multipole(config=..., basis=...)."
            )
        mp_config = self.multipole_config or self._default_multipole_config()
        return self.bispectrum.multipole(
            config=mp_config,
            basis=self.multipole_basis,
            regulator=self.regulator,
            **self.multipole_kwargs,
        )

    def _ensure_calculator(self) -> ThreePCFCalculator:
        if self.calculator is None:
            self.bmultipole = self.make_multipole()
            self.calculator = ThreePCFCalculator(
                self.bmultipole,
                config=self.config,
                coupling_kwargs=self.coupling_kwargs,
            )
        return self.calculator

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
        """Alias for :meth:`compute_zeta`."""
        return self.compute_zeta(delta_phi, *args, **kwargs)

    def get_H_kernel(self, key: HKernelKey) -> HKernel:
        return self.Hgrid.get(key)

    def get_zeta_k(self, key: ZetaKKey) -> ZetaKMode:
        return self.ZKgrid.get(key)
