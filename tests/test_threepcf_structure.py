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
                "HKernelKey",
                "HKernelTable",
                "NumericRouteConfig",
                "ThreePCF",
                "ZetaKKey",
                "ZetaKTable",
                "ZetaTable",
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
        self.assertTrue(callable(projection_factor))


if __name__ == "__main__":
    unittest.main()
