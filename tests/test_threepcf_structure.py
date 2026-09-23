import unittest

import fastnc.threepcf as threepcf
from fastnc.threepcf.conventions import (
    EffectiveSpinTriple,
    SpinSpec,
    projection_factor,
)


class ThreePCFStructureTests(unittest.TestCase):
    def test_top_level_exposes_only_conventions_and_routes(self):
        self.assertEqual(threepcf.__all__, ["conventions", "routes"])

    def test_spin_conventions_remain_active_without_grid_pipeline(self):
        effective = EffectiveSpinTriple((2, -2, 2))
        self.assertEqual(effective.bessel_orders(1.0), (2, 0))
        self.assertEqual(SpinSpec((2, 2, 2)).n_components, 4)
        self.assertTrue(callable(projection_factor))


if __name__ == "__main__":
    unittest.main()
