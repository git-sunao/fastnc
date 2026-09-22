import unittest
from unittest.mock import patch

import numpy as np

from fastnc.bispectrum.spt import f2_kernel
from fastnc.hankel.wrapper import PowerLawFFTLogConfig
from fastnc.threepcf.grid import FFTGrid
from fastnc.threepcf.assembly import (
    ZetaKAccumulator,
    ZetaKContribution,
    resum_multipoles,
)
from fastnc.threepcf.hkernel_grid import HKernel, HKernelGrid
from fastnc.threepcf.routes.numeric import ZetaKTransformCalculator
from fastnc.threepcf.slepian import MellinExpansion
from fastnc.threepcf.spin import EffectiveSpinTriple
from fastnc.threepcf.zetak_grid import ZetaKGrid


class SlepianFoundationTests(unittest.TestCase):
    @staticmethod
    def make_grid():
        ell = np.geomspace(1.0e-2, 1.0e2, 8)
        theta = np.geomspace(1.0e-1, 1.0e1, 8)
        down = np.array([1, 3, 5])
        tuned = type(
            "Tuned",
            (),
            {"ell": ell, "theta": theta, "down_sampler": down, "xy": 1.0},
        )()
        return FFTGrid.from_tuned_grid(tuned)

    def test_spin_orders_follow_x1_convention(self):
        spin = EffectiveSpinTriple((0, 2, 2))
        self.assertEqual(spin.bessel_orders(2), (4, 0))
        np.testing.assert_array_equal(spin.k_values(2), [-2, -1, 0, 1, 2])

    def test_mellin_shift_reuses_coefficients(self):
        k = np.geomspace(1.0e-3, 1.0e2, 64, endpoint=False)
        values = k ** -0.75
        expansion = MellinExpansion.from_samples(
            k,
            values,
            PowerLawFFTLogConfig(bias=-0.75, c_window_width=0.0),
        )
        shifted = expansion.shifted(1.5)
        np.testing.assert_array_equal(shifted.coefficients, expansion.coefficients)
        np.testing.assert_allclose(shifted.exponents, expansion.exponents + 1.5)
        np.testing.assert_allclose(expansion.evaluate(k), values, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(shifted.evaluate(k), k**1.5 * values, rtol=1.0e-11, atol=1.0e-11)

    def test_precomputed_mode_can_be_stored_without_hkernel(self):
        grid = self.make_grid()
        down = grid.down_sampler
        zgrid = ZetaKGrid(spin=(0, 0, 0), kmax=2, grid=grid)
        values = np.arange(64).reshape(8, 8).astype(complex)
        mode = zgrid.add_mode(values, sigma=(0, 0, 0), epsilon=(1, 1, 1), k=1)
        np.testing.assert_array_equal(mode.get_value_fft(), values)
        np.testing.assert_array_equal(mode.get_value(), values[np.ix_(down, down)])
        self.assertIs(zgrid.get_for_epsilon((1, 1, 1), 1), mode)

    def test_existing_hkernel_route_still_builds_same_mode_type(self):
        grid = self.make_grid()
        sigma = (0, 0, 0)
        kval = 1.0
        hkernel = HKernel(
            grid=grid,
            key=HKernelGrid.key_from_sigma_k(sigma, kval),
            value=np.ones(grid.shape_ell),
            source_k=kval,
            source_sigma=sigma,
        )
        transformed = np.full(grid.shape_theta_fft, 3.0 + 2.0j)
        zgrid = ZetaKGrid(spin=(0, 0, 0), kmax=2, grid=grid)
        with patch(
            "fastnc.threepcf.routes.numeric.double_hankel_transform",
            return_value=(grid.theta_fft, grid.theta_fft, transformed),
        ):
            values = ZetaKTransformCalculator.transform_mode(
                hkernel,
                sigma=sigma,
                k=kval,
                hankel_config=None,
            )
            mode = zgrid.add_mode(
                values, sigma=sigma, k=kval, source_route="numeric"
            )
        self.assertEqual(mode.key, zgrid.key_from_sigma_k(sigma, kval))
        np.testing.assert_array_equal(mode.get_value_fft(), transformed)

    def test_contributions_from_multiple_routes_are_added(self):
        grid = self.make_grid()
        target = ZetaKGrid(spin=(0, 0, 0), kmax=1, grid=grid)
        accumulator = ZetaKAccumulator(target)
        common = dict(sigma=(0, 0, 0), epsilon=(1, 1, 1), k=0.0)
        accumulator.add(
            ZetaKContribution(np.ones(grid.shape_theta_fft), route="numeric", **common)
        )
        accumulator.add(
            ZetaKContribution(2.0 * np.ones(grid.shape_theta_fft), route="slepian", **common)
        )
        mode = target.get_for_epsilon((1, 1, 1), 0.0)
        np.testing.assert_array_equal(mode.get_value_fft(), 3.0)
        self.assertEqual(mode.source_route, "numeric+slepian")

    def test_recomputing_one_contributor_replaces_only_that_contribution(self):
        grid = self.make_grid()
        target = ZetaKGrid(spin=(0, 0, 0), kmax=1, grid=grid)
        accumulator = ZetaKAccumulator(target)
        common = dict(sigma=(0, 0, 0), epsilon=(1, 1, 1), k=0.0)
        accumulator.add(
            ZetaKContribution(np.ones(grid.shape_theta_fft), route="numeric:tree", **common)
        )
        accumulator.add(
            ZetaKContribution(2.0 * np.ones(grid.shape_theta_fft), route="slepian:loop", **common)
        )
        accumulator.add(
            ZetaKContribution(4.0 * np.ones(grid.shape_theta_fft), route="numeric:tree", **common)
        )
        mode = target.get_for_epsilon((1, 1, 1), 0.0)
        np.testing.assert_array_equal(mode.get_value_fft(), 6.0)
        self.assertEqual(mode.source_route, "numeric:tree+slepian:loop")

    def test_opening_angle_resummation(self):
        zeta_k = np.array([np.ones((2, 2)), 2.0 * np.ones((2, 2))], dtype=complex)
        modes = np.array([0.0, 1.0])
        phi = np.array([0.0, np.pi])
        result = resum_multipoles(zeta_k, modes, phi, (0, 0, 0), phase="k")
        np.testing.assert_allclose(result[..., 0], 3.0)
        np.testing.assert_allclose(result[..., 1], -1.0, atol=1.0e-15)

    def test_spt_f2_reference_values(self):
        self.assertAlmostEqual(float(f2_kernel(1.0, 1.0, 1.0)), 2.0)
        self.assertAlmostEqual(float(f2_kernel(1.0, 1.0, 0.0)), 5.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
