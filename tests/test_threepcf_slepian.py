import unittest
from unittest.mock import patch

import numpy as np

from fastnc.bispectrum import (
    Bispectrum2D,
    BispectrumTerm2D,
    NumericExpression2D,
    SlepianExpression2D,
    SlepianRadialFactor2D,
)
from fastnc.threepcf import (
    ComponentModeKey,
    SlepianConfig,
    ThreePCF,
    ThreePCFConfig,
)
from fastnc.threepcf.slepian import (
    SlepianCalculator,
    WeberGeometry,
    _interpolated_weber_unit_power,
    _powerlaw_double_kernel,
    _weber_unit_power,
)
from fastnc.multipole import NumericMultipoleConfig


def toy_bispectrum():
    def radial(ell, center):
        ell = np.asarray(ell, dtype=float)
        safe = np.maximum(ell, np.finfo(float).tiny)
        return np.where(ell > 0.0, np.exp(-0.5 * np.log(safe / center) ** 2), 0.0)

    f1 = lambda ell: radial(ell, 30.0)
    f2 = lambda ell: radial(ell, 50.0)
    slepian = SlepianExpression2D(
        coefficient=2.0,
        radial_factors=(
            SlepianRadialFactor2D(f1),
            SlepianRadialFactor2D(f2),
            SlepianRadialFactor2D.constant(),
        ),
    )
    numeric = NumericExpression2D(
        lambda ell1, ell2, ell3: 2.0 * f1(ell1) * f2(ell2)
    )
    return Bispectrum2D((BispectrumTerm2D("toy", (numeric, slepian)),))


class SlepianRouteTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.geomspace(1.0e-3, 1.0e-2, 3)
        self.manager = ThreePCF(
            ThreePCFConfig(
                kmax=1.0,
                ell_min=1.0,
                ell_max=1.0e3,
                n_ell=12,
            ),
            toy_bispectrum(),
            self.theta,
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        )

    def test_route_builds_shared_zetak_contract_without_hkernel(self):
        size = self.theta.size
        modes = {
            -1: np.full((size, size), -1.0),
            0: np.full((size, size), 2.0),
            1: np.full((size, size), 3.0),
        }
        with patch.object(
            SlepianCalculator, "evaluate_modes", return_value=modes
        ) as evaluate:
            table = self.manager.zetak()

        evaluate.assert_called_once()
        np.testing.assert_allclose(evaluate.call_args.args[2], self.theta)
        self.assertEqual(self.manager._hkernel_tables, {})
        self.assertEqual(
            set(table.aliases),
            {
                ComponentModeKey((1, 1, 1), -2),
                ComponentModeKey((1, 1, 1), 0),
                ComponentModeKey((1, 1, 1), 2),
            },
        )
        np.testing.assert_allclose(
            table.get_for_mode((1, 1, 1), 1.0), 3.0
        )

    def test_weber_method_configuration_is_explicit(self):
        self.assertEqual(SlepianConfig().weber_method, "direct")
        self.assertEqual(
            SlepianConfig(weber_method="interpolated").weber_method,
            "interpolated",
        )
        with self.assertRaisesRegex(ValueError, "weber_method"):
            SlepianConfig(weber_method="unknown")
        with self.assertRaisesRegex(ValueError, "at least four"):
            SlepianConfig(weber_interpolation_nodes=3)

    def test_route_rejects_non_scalar_spin(self):
        manager = ThreePCF(
            ThreePCFConfig(
                spin=(2, 0, 0),
                ell_min=1.0,
                ell_max=1.0e3,
                n_ell=12,
            ),
            toy_bispectrum(),
            self.theta,
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        )
        with self.assertRaisesRegex(NotImplementedError, r"spin=\(0, 0, 0\)"):
            manager.zetak()

    def test_calculator_combines_expression_and_term_coefficients(self):
        bispectrum = 3.0 * toy_bispectrum()
        calculator = SlepianCalculator(self.manager.config.slepian)
        ell = self.manager.grid.ell
        theta = self.manager.grid.theta[:3]
        with (
            patch(
                "fastnc.threepcf.slepian.single_radial_transform",
                return_value=np.array([1.0, 2.0, 3.0]),
            ),
            patch(
                "fastnc.threepcf.slepian.double_radial_transform",
                return_value=np.ones((3, 3)),
            ),
        ):
            result = calculator.evaluate_modes(bispectrum, ell, theta, [0])[0]
        expected = 6.0 * np.array([[1.0, 2.0, 3.0]]) / (2.0 * np.pi) ** 2
        np.testing.assert_allclose(result, np.repeat(expected, 3, axis=0))

    def test_theta_change_discards_slepian_calculator(self):
        size = self.theta.size
        with patch.object(
            SlepianCalculator,
            "evaluate_modes",
            return_value={
                -1: np.zeros((size, size)),
                0: np.zeros((size, size)),
                1: np.zeros((size, size)),
            },
        ):
            self.manager.zetak()
        self.assertIsNotNone(self.manager._slepian_calculator)
        self.manager.set_theta(np.geomspace(2.0e-3, 2.0e-2, 3))
        self.assertIsNone(self.manager._slepian_calculator)

    def test_toy_zetak_agrees_with_numeric_route(self):
        config = ThreePCFConfig(
            Lmax=12,
            kmax=0.0,
            ell_min=0.3,
            ell_max=3.0e3,
            n_ell=32,
            use_coupling_cache=False,
            multipole=NumericMultipoleConfig(
                n_angle=513,
                delta_beta_min=0.0,
                delta_beta_max=np.pi,
            ),
        )
        numeric = ThreePCF(
            config,
            toy_bispectrum(),
            self.theta,
            np.linspace(0.0, np.pi, 4),
            route="numeric",
        ).zetak().get_for_mode((1, 1, 1), 0.0)
        slepian = ThreePCF(
            config,
            toy_bispectrum(),
            self.theta,
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        ).zetak().get_for_mode((1, 1, 1), 0.0)
        relative = np.max(np.abs(slepian - numeric)) / np.max(np.abs(numeric))
        self.assertLess(relative, 3.0e-2)

    def test_weber_geometry_evaluates_only_unique_log_grid_ratios(self):
        theta = np.geomspace(1.0e-2, 1.0e-1, 4)
        geometry = WeberGeometry.from_coordinates(theta, theta)
        with patch(
            "fastnc.threepcf.slepian._weber_unit_power",
            return_value=1.0,
        ) as unit_power:
            result = _powerlaw_double_kernel(
                -1.0 + 0.5j,
                0,
                0,
                theta,
                theta,
                rtol=1.0e-12,
                omit_diagonal=True,
                geometry=geometry,
            )
        self.assertEqual(unit_power.call_count, theta.size - 1)
        np.testing.assert_allclose(np.diag(result), 0.0)

    def test_unique_ratio_kernel_matches_pointwise_evaluation(self):
        x = np.geomspace(8.0e-3, 7.0e-2, 4)
        theta = np.geomspace(1.0e-2, 1.0e-1, 5)
        exponent = -0.8 + 0.35j
        order_x, order_theta = 1, 2
        optimized = _powerlaw_double_kernel(
            exponent,
            order_x,
            order_theta,
            x,
            theta,
            rtol=1.0e-12,
            omit_diagonal=False,
        )

        expected = np.empty((x.size, theta.size), dtype=complex)
        for i, x_value in enumerate(x):
            for j, theta_value in enumerate(theta):
                scale = max(x_value, theta_value)
                ratio = min(x_value, theta_value) / scale
                if x_value <= theta_value:
                    small_order, big_order = order_x, order_theta
                else:
                    small_order, big_order = order_theta, order_x
                expected[i, j] = scale ** (-exponent - 2.0) * _weber_unit_power(
                    exponent,
                    small_order,
                    big_order,
                    ratio,
                    1.0e-12,
                )

        np.testing.assert_allclose(optimized, expected, rtol=2.0e-13)

    def test_interpolated_weber_matches_direct_with_fewer_unit_evaluations(self):
        ratios = np.exp(-np.linspace(2.0e-3, 5.0, 301))
        exponent = -0.8 + 0.35j
        orders = (-2, 1)
        direct = np.array(
            [
                _weber_unit_power(exponent, *orders, float(ratio), 1.0e-12)
                for ratio in ratios
            ]
        )
        with patch(
            "fastnc.threepcf.slepian._weber_unit_power",
            wraps=_weber_unit_power,
        ) as unit_power:
            interpolated = _interpolated_weber_unit_power(
                exponent,
                *orders,
                ratios,
                nodes=64,
                max_ratio=0.8,
                rtol=1.0e-12,
            )
        scaled_error = np.max(np.abs(interpolated - direct)) / np.max(
            np.abs(direct)
        )
        self.assertLess(scaled_error, 1.0e-6)
        expected_direct = np.count_nonzero(ratios >= 0.8)
        self.assertEqual(unit_power.call_count, 64 + expected_direct)

    def test_interpolated_weber_uses_direct_evaluation_for_small_requests(self):
        ratios = np.array([0.2, 0.5, 0.8])
        with patch(
            "fastnc.threepcf.slepian._weber_unit_power",
            wraps=_weber_unit_power,
        ) as unit_power:
            _interpolated_weber_unit_power(
                -0.8 + 0.35j,
                0,
                1,
                ratios,
                nodes=4,
                max_ratio=0.9,
                rtol=1.0e-12,
            )
        self.assertEqual(unit_power.call_count, ratios.size)

    def test_interpolated_and_direct_routes_agree_for_toy_term(self):
        theta = np.geomspace(1.0e-3, 1.0e-1, 36)
        common = dict(
            kmax=0.0,
            ell_min=0.3,
            ell_max=3.0e3,
            n_ell=24,
        )
        direct = ThreePCF(
            ThreePCFConfig(**common, slepian=SlepianConfig()),
            toy_bispectrum(),
            theta,
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        ).zetak().get_for_mode((1, 1, 1), 0.0)
        interpolated = ThreePCF(
            ThreePCFConfig(
                **common,
                slepian=SlepianConfig(
                    weber_method="interpolated",
                    weber_interpolation_nodes=24,
                ),
            ),
            toy_bispectrum(),
            theta,
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        ).zetak().get_for_mode((1, 1, 1), 0.0)
        scaled_error = np.max(np.abs(interpolated - direct)) / np.max(
            np.abs(direct)
        )
        self.assertLess(scaled_error, 2.0e-5)

    def test_calculator_reuses_weber_geometry_and_primitive_kernels(self):
        calculator = SlepianCalculator(self.manager.config.slepian)
        ell = self.manager.grid.ell
        with patch(
            "fastnc.threepcf.slepian._powerlaw_double_kernel",
            wraps=_powerlaw_double_kernel,
        ) as kernel:
            calculator.evaluate_modes(
                toy_bispectrum(), ell, self.theta, [0]
            )
            first_calls = kernel.call_count
            calculator.evaluate_modes(
                toy_bispectrum(), ell, self.theta, [0]
            )
        self.assertGreater(first_calls, 0)
        self.assertEqual(kernel.call_count, first_calls)


if __name__ == "__main__":
    unittest.main()
