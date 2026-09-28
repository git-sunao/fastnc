import unittest

import fastnc.threepcf as threepcf
from fastnc.threepcf.conventions import (
    EffectiveSpinTriple,
    SpinSpec,
    projection_factor,
)


class ThreePCFStructureTests(unittest.TestCase):
    def test_top_level_exposes_conventions_and_flat_route_modules(self):
        self.assertEqual(
            threepcf.__all__,
            [
                "ComponentModeKey",
                "BruteForce3PCFAdaptiveTrial",
                "BruteForce3PCFConfig",
                "BruteForce3PCFResult",
                "BruteForceX3PCF",
                "HKernelKey",
                "HKernelTable",
                "ThreePCF",
                "ThreePCFConfig",
                "SlepianConfig",
                "SemiAnalyticCalculator",
                "SemiAnalyticConfig",
                "ZetaKKey",
                "ZetaKTable",
                "ZetaTable",
                "bruteforce",
                "config",
                "conventions",
                "numeric",
                "semi_analytic",
                "slepian",
                "tables",
            ],
        )

    def test_spin_conventions_remain_active_without_grid_pipeline(self):
        effective = EffectiveSpinTriple((2, -2, 2))
        self.assertEqual(effective.bessel_orders(1.0), (2, 0))
        self.assertEqual(SpinSpec((2, 2, 2)).n_components, 4)
        self.assertEqual(
            SpinSpec((0, 2, 0)).normalize_epsilon((-1, 1, -1)),
            (1, 1, 1),
        )
        self.assertTrue(callable(projection_factor))


if __name__ == "__main__":
    unittest.main()
