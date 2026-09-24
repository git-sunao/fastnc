import unittest
from unittest.mock import patch

import numpy as np

from fastnc.hankel import make_fftlog_grid
from fastnc.multipole import BispectrumMultipole, NumericMultipoleConfig
from fastnc.threepcf import (
    HKernelKey,
    NumericRouteConfig,
    ThreePCF,
    ZetaKKey,
)


class FakeCoupling:
    def __init__(self, sigma):
        self.sigma = sigma
        self.calls = 0

    def __call__(self, mode, k, psi):
        self.calls += 1
        return np.full(np.shape(psi), 2.0 if mode == 0 else 0.0)


class ThreePCFNumericTests(unittest.TestCase):
    def setUp(self):
        self.created = []

        def factory(sigma):
            coupling = FakeCoupling(sigma)
            self.created.append(coupling)
            return coupling

        self.manager = ThreePCF(
            NumericRouteConfig(Lmax=1, kmax=0.0),
            coupling_factory=factory,
        )
        self.grid = make_fftlog_grid(
            10.0,
            1.0e3,
            12,
            theta=np.geomspace(1.0e-3, 1.0e-2, 4),
        )
        self.multipole = BispectrumMultipole.from_numeric(
            NumericMultipoleConfig(
                n_angle=65,
                delta_beta_min=0.0,
                delta_beta_max=np.pi,
            ),
            lambda ell1, ell2, ell3: np.full(np.shape(ell1), 3.0),
        )

    def test_manager_retains_coupling_and_builds_passive_hkernel(self):
        first = self.manager.coupling((0, 0, 0))
        second = self.manager.coupling((0, 0, 0))
        self.assertIs(first, second)
        self.assertEqual(len(self.created), 1)

        table = self.manager.hkernel(
            self.multipole,
            self.grid,
            spin=(0, 0, 0),
        )
        key = HKernelKey(sigma1=0, two_nu=0)
        self.assertIs(self.manager.hkernel_table, table)
        self.assertIsNone(self.manager.zetak_table)
        np.testing.assert_allclose(table.get(key), 6.0, atol=1.0e-12)

    def test_numeric_route_uses_grid_xy_and_downsamples_to_target_theta(self):
        captured = {}

        def fake_transform(ell1, ell2, kernel, m, n, *, config, **kwargs):
            captured["xy"] = config.xy
            captured["orders"] = (m, n)
            return (
                self.grid.theta,
                self.grid.theta,
                np.asarray(kernel),
            )

        with patch(
            "fastnc.threepcf.threepcf.double_hankel_transform",
            side_effect=fake_transform,
        ):
            table = self.manager.zetak_numeric(
                self.multipole,
                self.grid,
                spin=(0, 0, 0),
            )

        hkey = HKernelKey(sigma1=0, two_nu=0)
        key = ZetaKKey(hkey=hkey, m=0, n=0, Sigma=0)
        ell2, ell3 = np.meshgrid(
            self.grid.ell,
            self.grid.ell,
            indexing="ij",
        )
        expected_full = 6.0 * ell2**2 * ell3**2 / (2.0 * np.pi) ** 3
        np.testing.assert_allclose(
            table.get(key),
            self.grid.downsample_2d(expected_full),
        )
        self.assertEqual(captured["xy"], self.grid.xy)
        self.assertEqual(captured["orders"], (0, 0))
        self.assertIs(self.manager.zetak_table, table)


if __name__ == "__main__":
    unittest.main()
