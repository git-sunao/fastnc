import unittest
from unittest.mock import patch

import numpy as np

from fastnc.multipole import NumericMultipoleConfig
from fastnc.threepcf import (
    ComponentModeKey,
    HKernelKey,
    ThreePCF,
    ThreePCFConfig,
    ZetaKKey,
)


class FakeCoupling:
    def __init__(self, sigma):
        self.sigma = sigma
        self.calls = 0

    def __call__(self, mode, k, psi):
        self.calls += 1
        return np.full(np.shape(psi), 2.0 if mode == 0 else 0.0)


class StatefulSource:
    def __init__(self, value):
        self.value = float(value)
        self.state_token = (0,)

    def __call__(self, ell1, ell2, ell3):
        return np.full(np.shape(ell1), self.value)

    def update(self, value):
        self.value = float(value)
        self.state_token = (self.state_token[0] + 1,)


class ThreePCFNumericTests(unittest.TestCase):
    def setUp(self):
        self.created = []

        def factory(sigma):
            coupling = FakeCoupling(sigma)
            self.created.append(coupling)
            return coupling

        self.source = lambda ell1, ell2, ell3: np.full(np.shape(ell1), 3.0)
        self.theta = np.geomspace(1.0e-3, 1.0e-2, 4)
        self.phi = np.linspace(0.0, np.pi, 5)
        self.manager = ThreePCF(
            ThreePCFConfig(
                Lmax=1,
                kmax=0.0,
                ell_min=10.0,
                ell_max=1.0e3,
                n_ell=12,
                multipole=NumericMultipoleConfig(
                    n_angle=65,
                    delta_beta_min=0.0,
                    delta_beta_max=np.pi,
                ),
            ),
            self.source,
            self.theta,
            self.phi,
            route="numeric",
            coupling_factory=factory,
        )
        self.grid = self.manager.grid

    def test_manager_retains_coupling_and_builds_passive_hkernel(self):
        first = self.manager.coupling((0, 0, 0))
        second = self.manager.coupling((0, 0, 0))
        self.assertIs(first, second)
        self.assertEqual(len(self.created), 1)

        table = self.manager.hkernel()
        key = HKernelKey(sigma1=0, two_nu=0)
        self.assertIs(self.manager.hkernel(), table)
        np.testing.assert_allclose(table.get(key), 6.0, atol=1.0e-12)
        physical_key = ComponentModeKey(epsilon=(1, 1, 1), two_k=0)
        self.assertEqual(table.aliases[physical_key], key)
        np.testing.assert_allclose(
            table.get_for_mode((1, 1, 1), 0.0),
            6.0,
            atol=1.0e-12,
        )

    def test_spin_components_keep_all_three_epsilon_entries(self):
        manager = ThreePCF(
            ThreePCFConfig(
                spin=(2, 2, 2),
                Lmax=1,
                kmax=0.0,
                ell_min=10.0,
                ell_max=1.0e3,
                n_ell=12,
                multipole=self.manager.config.multipole,
            ),
            self.source,
            self.theta,
            self.phi,
            coupling_factory=self.manager._coupling_factory,
        )
        table = manager.hkernel()
        expected = {
            ComponentModeKey((1, 1, 1), 0),
            ComponentModeKey((-1, 1, 1), 0),
            ComponentModeKey((1, -1, 1), 0),
            ComponentModeKey((1, 1, -1), 0),
        }
        self.assertEqual(set(table.aliases), expected)
        for physical_key in expected:
            np.testing.assert_allclose(
                table.get_for_mode(physical_key.epsilon, physical_key.k),
                6.0,
                atol=1.0e-12,
            )

    def test_scalar_epsilon_minus_is_rejected_and_duplicates_are_deduplicated(self):
        with self.assertRaisesRegex(ValueError, "spin-zero vertices"):
            self.manager.hkernel(epsilons=[(-1, 1, -1)])

        table = self.manager.hkernel(
            epsilons=[(1, 1, 1), (1, 1, 1)]
        )
        expected = {ComponentModeKey((1, 1, 1), 0)}
        self.assertEqual(set(table.aliases), expected)

    def test_nonrepresentative_conjugate_epsilon_is_rejected(self):
        manager = ThreePCF(
            ThreePCFConfig(
                spin=(0, 2, 0),
                Lmax=1,
                kmax=0.0,
                ell_min=10.0,
                ell_max=1.0e3,
                n_ell=12,
                multipole=self.manager.config.multipole,
            ),
            self.source,
            self.theta,
            self.phi,
            route="numeric",
            coupling_factory=self.manager._coupling_factory,
        )
        with self.assertRaisesRegex(
            ValueError,
            "complex-conjugate representative.*\\(1, 1, 1\\)",
        ):
            manager.hkernel(epsilons=[(1, -1, 1)])

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
            table = self.manager.zetak()

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
        self.assertEqual(table.key_for_mode((1, 1, 1), 0.0), key)
        np.testing.assert_allclose(
            table.get_for_mode((1, 1, 1), 0.0),
            self.grid.downsample_2d(expected_full),
        )
        self.assertEqual(captured["xy"], self.grid.xy)
        self.assertEqual(captured["orders"], (0, 0))
        self.assertIs(self.manager.zetak(), table)

    def test_zeta_resums_physical_modes_on_the_retained_phi_grid(self):
        def fake_transform(ell1, ell2, kernel, m, n, *, config, **kwargs):
            return self.grid.theta, self.grid.theta, np.asarray(kernel)

        with patch(
            "fastnc.threepcf.threepcf.double_hankel_transform",
            side_effect=fake_transform,
        ):
            zetak = self.manager.zetak()
            zeta = self.manager.zeta()

        epsilon = (1, 1, 1)
        expected = np.repeat(
            zetak.get_for_mode(epsilon, 0.0)[..., None],
            self.phi.size,
            axis=-1,
        )
        self.assertEqual(zeta.components, (epsilon,))
        self.assertEqual(zeta.sigmas, ((0, 0, 0),))
        self.assertEqual(zeta.projection, "x")
        np.testing.assert_allclose(zeta.get(epsilon), expected)
        self.assertIs(self.manager.zeta(), zeta)
        epsilon_set = (epsilon,)
        self.assertEqual(set(self.manager._zetak_tables), {epsilon_set})
        self.assertEqual(set(self.manager._zeta_tables), {epsilon_set})
        self.assertIs(self.manager._zeta_tables[epsilon_set], zeta)

        centroid = self.manager.zeta(projection="centroid")
        self.assertEqual(centroid.projection, "cent")
        np.testing.assert_allclose(centroid.values, zeta.values)
        centroid_again = self.manager.zeta(projection="centroid")
        self.assertIsNot(centroid_again, centroid)
        np.testing.assert_allclose(
            centroid_again.values,
            centroid.values,
        )
        self.assertIs(self.manager.zeta(projection="x"), zeta)
        with self.assertRaises(ValueError):
            self.manager.zeta(projection="unknown")

    def test_zeta_uses_nu_k_as_the_opening_angle_phase(self):
        manager = ThreePCF(
            ThreePCFConfig(
                spin=(0, 2, 0),
                Lmax=1,
                kmax=0.0,
                ell_min=10.0,
                ell_max=1.0e3,
                n_ell=12,
                multipole=self.manager.config.multipole,
            ),
            self.source,
            self.theta,
            self.phi,
            route="numeric",
            coupling_factory=self.manager._coupling_factory,
        )

        def fake_transform(ell1, ell2, kernel, m, n, *, config, **kwargs):
            return manager.grid.theta, manager.grid.theta, np.asarray(kernel)

        with patch(
            "fastnc.threepcf.threepcf.double_hankel_transform",
            side_effect=fake_transform,
        ):
            zetak = manager.zetak()
            zeta = manager.zeta()

        epsilon = (1, 1, 1)
        radial = zetak.get_for_mode(epsilon, 0.0)
        expected = radial[..., None] * np.exp(-1j * self.phi)
        np.testing.assert_allclose(zeta.get(epsilon), expected)

    def test_route_is_retained_and_setter_invalidates_route_results(self):
        grid = self.manager.grid
        coupling = self.manager.coupling((0, 0, 0))
        multipole = self.manager.multipoles()
        hkernel = self.manager.hkernel()

        def fake_transform(ell1, ell2, kernel, m, n, *, config, **kwargs):
            return self.grid.theta, self.grid.theta, np.asarray(kernel)

        with patch(
            "fastnc.threepcf.threepcf.double_hankel_transform",
            side_effect=fake_transform,
        ):
            self.manager.zeta()
        self.assertTrue(self.manager._zetak_tables)
        self.assertTrue(self.manager._zeta_tables)

        self.manager.set_route("numeric")
        self.assertIs(self.manager.hkernel(), hkernel)
        self.assertTrue(self.manager._zetak_tables)
        self.assertTrue(self.manager._zeta_tables)

        self.manager.set_route("slepian")

        self.assertEqual(self.manager.route, "slepian")
        self.assertIs(self.manager.grid, grid)
        self.assertIs(self.manager.coupling((0, 0, 0)), coupling)
        self.assertIsNone(self.manager._multipole)
        self.assertEqual(self.manager._hkernel_tables, {})
        self.assertEqual(self.manager._zetak_tables, {})
        self.assertEqual(self.manager._zeta_tables, {})
        with self.assertRaises(NotImplementedError):
            self.manager.zetak()

        self.manager.set_route("numeric")
        self.assertIsNot(self.manager.multipoles(), multipole)
        with self.assertRaises(ValueError):
            self.manager.set_route("unknown")

    def test_coordinate_setters_apply_dependency_aware_invalidation(self):
        hkernel = self.manager.hkernel()
        self.manager._zeta_tables["demo"] = object()
        original_grid = self.manager.grid

        self.manager.set_phi(np.linspace(0.0, 2.0 * np.pi, 7))
        self.assertIs(self.manager.grid, original_grid)
        self.assertIs(self.manager.hkernel(), hkernel)
        self.assertEqual(self.manager._zeta_tables, {})

        self.manager.set_theta(np.geomspace(2.0e-3, 2.0e-2, 4))
        self.assertIsNot(self.manager.grid, original_grid)
        self.assertEqual(self.manager._hkernel_tables, {})
        self.assertEqual(self.manager._zetak_tables, {})
        np.testing.assert_allclose(
            self.manager.grid.target_theta,
            self.manager.theta,
        )

    def test_bispectrum_setter_preserves_grid_and_coupling_resources(self):
        coupling = self.manager.coupling((0, 0, 0))
        grid = self.manager.grid
        old_multipole = self.manager.multipoles()
        self.manager.hkernel()

        replacement = lambda ell1, ell2, ell3: np.full(np.shape(ell1), 4.0)
        self.manager.set_bispectrum(replacement)

        self.assertIs(self.manager.grid, grid)
        self.assertIs(self.manager.coupling((0, 0, 0)), coupling)
        self.assertIsNot(self.manager.multipoles(), old_multipole)
        self.assertEqual(self.manager._hkernel_tables, {})

    def test_coordinates_are_copied_read_only_and_theta_must_be_log_spaced(self):
        self.assertFalse(self.manager.theta.flags.writeable)
        self.assertFalse(self.manager.phi.flags.writeable)
        with self.assertRaises(ValueError):
            self.manager.set_theta([1.0e-3, 2.0e-3, 5.0e-3])

    def test_source_state_change_invalidates_cached_physical_results(self):
        source = StatefulSource(3.0)
        self.manager.set_bispectrum(source)
        first = self.manager.hkernel()
        source.update(4.0)
        second = self.manager.hkernel()
        self.assertIsNot(first, second)
        np.testing.assert_allclose(
            second.values,
            first.values * (4.0 / 3.0),
            rtol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
