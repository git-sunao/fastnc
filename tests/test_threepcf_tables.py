import unittest

import numpy as np

from fastnc.hankel import make_fftlog_grid
from fastnc.threepcf import HKernelTable, ZetaKTable, ZetaTable


class TunedFFTGridTests(unittest.TestCase):
    def setUp(self):
        self.target_theta = np.geomspace(1.0e-3, 1.0e-1, 9)
        self.grid = make_fftlog_grid(
            10.0,
            1.0e4,
            64,
            theta=self.target_theta,
        )

    def test_target_bins_are_exact_subsequence_of_high_resolution_grid(self):
        self.assertGreater(self.grid.theta.size, self.target_theta.size)
        np.testing.assert_allclose(
            self.grid.target_theta,
            self.target_theta,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertFalse(self.grid.ell.flags.writeable)
        self.assertFalse(self.grid.theta.flags.writeable)
        self.assertFalse(self.grid.down_sampler.flags.writeable)

    def test_downsample_2d_selects_both_theta_axes_without_interpolation(self):
        n = self.grid.theta.size
        full = np.arange(2 * n * n).reshape(2, n, n)
        selected = self.grid.downsample_2d(full)
        expected = full[:, self.grid.down_sampler, :][
            :, :, self.grid.down_sampler
        ]
        np.testing.assert_array_equal(selected, expected)
        self.assertEqual(selected.shape, (2, 9, 9))


class ThreePCFTableTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.geomspace(1.0e-3, 1.0e-1, 5)
        self.grid = make_fftlog_grid(10.0, 1.0e4, 32, theta=self.theta)

    def test_hkernel_table_uses_full_tuned_ell_grid(self):
        keys = ((0, 0), (2, 1))
        values = np.ones((2, self.grid.ell.size, self.grid.ell.size))
        table = HKernelTable(self.grid, keys, values)
        self.assertIs(table.grid, self.grid)
        np.testing.assert_array_equal(table.ell, self.grid.ell)
        np.testing.assert_array_equal(table.get((2, 1)), 1.0)
        self.assertFalse(table.values.flags.writeable)

    def test_numeric_and_slepian_results_fit_same_zetak_table_contract(self):
        keys = ("k=0", "k=2")
        numeric_full = np.arange(
            2 * self.grid.theta.size**2,
            dtype=float,
        ).reshape(2, self.grid.theta.size, self.grid.theta.size)
        numeric = ZetaKTable(
            self.grid.target_theta,
            keys,
            self.grid.downsample_2d(numeric_full),
        )
        slepian = ZetaKTable(
            self.theta,
            keys,
            np.zeros((2, self.theta.size, self.theta.size)),
        )
        self.assertEqual(numeric.values.shape, slepian.values.shape)
        np.testing.assert_allclose(
            numeric.theta,
            slepian.theta,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertFalse(numeric.values.flags.writeable)

    def test_zeta_table_validates_component_and_coordinate_axes(self):
        phi = np.linspace(0.0, np.pi, 7)
        values = np.zeros((2, self.theta.size, self.theta.size, phi.size))
        table = ZetaTable(
            self.theta,
            phi,
            components=("plus", "minus"),
            values=values,
        )
        self.assertEqual(table.get("plus").shape, (5, 5, 7))
        self.assertFalse(table.theta.flags.writeable)
        self.assertFalse(table.phi.flags.writeable)
        self.assertFalse(table.values.flags.writeable)

    def test_tables_reject_inconsistent_shapes(self):
        with self.assertRaises(ValueError):
            ZetaKTable(self.theta, ("k=0",), np.zeros((1, 4, 5)))
        with self.assertRaises(ValueError):
            HKernelTable(self.grid, ("h0",), np.zeros((1, 2, 2)))


if __name__ == "__main__":
    unittest.main()
