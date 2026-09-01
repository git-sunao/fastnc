"""Direct Slepian-route numerical machinery."""
from .calculator import SlepianThreePCFCalculator
from .fftlog import FFTLogExpansion, decompose_log_powerlaw, reconstruct
from .geometry import SlepianRadialGrid
from .los_moments import (PreparedLOSMoments, PreparedBatchedLOSMoments, SlepianLOSMomentRule, FactorizedGrowthMomentRule, FactorizedGrowthBatchMomentRule, GeneralCoefficientMomentRule, GeneralCoefficientBatchMomentRule)
from .weber import (WeberTableCache, ContactTerm, ConstantWeberKernel, canonical_bessel_order, single_bessel_factor)
from .rational_weber import SameOrderInfo, same_canonical_order, RationalIWeberSameOrder
from .integrated_kernel import IntegratedKernelGeometry, IntegratedKernelMatrix, IntegratedKernelCache

__all__ = [
    'SlepianThreePCFCalculator', 'FFTLogExpansion', 'decompose_log_powerlaw',
    'reconstruct', 'SlepianRadialGrid', 'WeberTableCache',
    'canonical_bessel_order', 'single_bessel_factor', 'ContactTerm', 'ConstantWeberKernel',
    'SameOrderInfo', 'same_canonical_order', 'RationalIWeberSameOrder',
    'IntegratedKernelGeometry', 'IntegratedKernelMatrix', 'IntegratedKernelCache',
    'PreparedLOSMoments', 'PreparedBatchedLOSMoments', 'SlepianLOSMomentRule',
    'FactorizedGrowthMomentRule', 'FactorizedGrowthBatchMomentRule',
    'GeneralCoefficientMomentRule', 'GeneralCoefficientBatchMomentRule',
]

from .terms import RadialKernelKey, CompiledSlepianTerm, CompiledModePlan, compile_mode_plan
