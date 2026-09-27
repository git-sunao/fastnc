import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import iv, jv

from fastnc.bispectrum import (
    Bispectrum2D,
    Bispectrum3D,
    BispectrumTerm2D,
    BispectrumTerm3D,
    NumericExpression2D,
    NumericExpression3D,
    SlepianExpression2D,
    SlepianExpression3D,
    SlepianRadialFactor2D,
    SlepianRadialFactor3D,
    SlepianRepresentation2D,
    SlepianRepresentation3D,
)
from fastnc.projection import LOSProjector
from fastnc.threepcf import (
    ComponentModeKey,
    SlepianConfig,
    ThreePCF,
    ThreePCFConfig,
)
from fastnc.threepcf.slepian import (
    ConstantLegKernel,
    LOSMellinFactors,
    LowRankRegularMellinMatrix,
    RegularMellinMatrix,
    SlepianCalculator,
    WeberGeometry,
    _contact_coefficient,
    _interpolated_weber_unit_power,
    _powerlaw_double_kernel,
    _regular_quadrature_supported,
    _weber_unit_power,
    constant_leg_kernel,
    contract_low_rank_regular_mellin_matrix,
    contract_low_rank_regular_mellin_matrix_los,
    contract_regular_mellin_matrix,
    compress_regular_mellin_matrix,
    regular_mellin_matrix,
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

    def test_scalar_gaussian_contact_toy_matches_analytic_zeta(self):
        amplitude = 1.7
        a = 1.0e-3
        b = 5.0e-4

        def factor(ell, width):
            ell = np.asarray(ell, dtype=float)
            return ell**2 * np.exp(-width * ell**2)

        f1 = lambda ell: factor(ell, a)
        f2 = lambda ell: factor(ell, b)
        expression = SlepianExpression2D(
            coefficient=amplitude,
            radial_factors=(
                SlepianRadialFactor2D(f1),
                SlepianRadialFactor2D(f2),
                SlepianRadialFactor2D.constant(),
            ),
        )
        bispectrum = Bispectrum2D(
            (BispectrumTerm2D("analytic-gaussian", (expression,)),)
        )
        theta = np.geomspace(3.0e-3, 2.0e-2, 6)
        phi = np.linspace(0.2, 2.9, 5)
        manager = ThreePCF(
            ThreePCFConfig(
                kmax=0.0,
                ell_min=1.0e-3,
                ell_max=500.0,
                n_ell=128,
                slepian=SlepianConfig(taper_fraction=0.1),
            ),
            bispectrum,
            theta,
            phi,
            route="slepian",
        )

        theta1, theta2 = np.meshgrid(theta, theta, indexing="ij")
        single = (
            (1.0 - theta2**2 / (4.0 * a))
            * np.exp(-theta2**2 / (4.0 * a))
            / (2.0 * a**2)
        )
        radius_sum = theta1**2 + theta2**2
        radius_product = theta1 * theta2
        argument = radius_product / (2.0 * b)
        double_zero = (
            np.exp(-radius_sum / (4.0 * b))
            * iv(0, argument)
            / (2.0 * b)
        )
        double = double_zero * (
            1.0 / b
            - radius_sum / (4.0 * b**2)
            + radius_product
            * iv(1, argument)
            / (2.0 * b**2 * iv(0, argument))
        )
        expected = amplitude * single * double / (2.0 * np.pi) ** 2
        actual = manager.zeta().values[0]

        np.testing.assert_allclose(
            actual,
            np.broadcast_to(expected[:, :, None], actual.shape),
            rtol=1.0e-5,
            atol=1.0e-8 * np.max(np.abs(expected)),
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
        with self.assertRaisesRegex(ValueError, "weber_brute_min_ratio"):
            SlepianConfig(weber_brute_min_ratio=0.0)
        self.assertEqual(
            SlepianConfig(regular_method="full_matrix").regular_method,
            "full_matrix",
        )
        self.assertEqual(
            SlepianConfig(regular_method="low_rank").regular_method,
            "low_rank",
        )
        self.assertEqual(SlepianConfig().regular_quadrature, "ratio_gauss")
        self.assertEqual(
            SlepianConfig(regular_quadrature="legacy_log").regular_quadrature,
            "legacy_log",
        )
        with self.assertRaisesRegex(ValueError, "regular_quadrature"):
            SlepianConfig(regular_quadrature="unknown")
        with self.assertRaisesRegex(ValueError, "regular_n_ratio"):
            SlepianConfig(regular_n_ratio=4)
        with self.assertRaisesRegex(ValueError, "regular_ratio_min"):
            SlepianConfig(regular_ratio_min=1.0)
        with self.assertRaisesRegex(ValueError, "regular_low_rank_rank"):
            SlepianConfig(regular_low_rank_rank=0)
        with self.assertRaisesRegex(ValueError, "regular_low_rank_rtol"):
            SlepianConfig(regular_low_rank_rtol=1.0)
        with self.assertRaisesRegex(ValueError, "regular_n_x"):
            SlepianConfig(regular_n_x=8)

    def test_constant_leg_contact_coefficient_uses_exact_order_parity(self):
        expected = {
            (0, 0): 1,
            (-1, 1): -1,
            (0, 2): -1,
            (1, -3): 1,
            (0, 1): 0,
            (-2, 1): 0,
        }
        for orders, coefficient in expected.items():
            with self.subTest(orders=orders):
                self.assertEqual(_contact_coefficient(*orders), coefficient)

    def test_equal_order_constant_leg_is_contact_only(self):
        coordinates = np.geomspace(0.5, 2.0, 4)
        kernel = constant_leg_kernel(
            2, 2, coordinates, coordinates, SlepianConfig()
        )
        self.assertIsInstance(kernel, ConstantLegKernel)
        self.assertEqual(kernel.contact_coefficient, 1)
        self.assertFalse(kernel.has_regular)
        np.testing.assert_array_equal(kernel.regular_less, 0.0)
        np.testing.assert_array_equal(kernel.regular_greater, 0.0)

    def test_unequal_order_constant_leg_separates_heaviside_support(self):
        x = np.array([0.5, 1.0, 2.0])
        theta = np.array([0.75, 1.5])
        geometry = WeberGeometry.from_coordinates(x, theta)
        full_regular = _powerlaw_double_kernel(
            0.0,
            0,
            2,
            x,
            theta,
            rtol=1.0e-12,
            omit_diagonal=True,
            geometry=geometry,
        )
        kernel = constant_leg_kernel(
            0, 2, x, theta, SlepianConfig(), geometry=geometry
        )
        less = x[:, None] < theta[None, :]
        greater = x[:, None] > theta[None, :]

        self.assertEqual(kernel.contact_coefficient, -1)
        self.assertTrue(kernel.has_regular)
        np.testing.assert_allclose(kernel.regular_less[less], full_regular[less])
        expected_less = np.broadcast_to(
            2.0 / theta[None, :] ** 2, kernel.regular_less.shape
        )
        np.testing.assert_allclose(
            kernel.regular_less[less], expected_less[less]
        )
        np.testing.assert_array_equal(kernel.regular_less[~less], 0.0)
        np.testing.assert_allclose(
            kernel.regular_greater[greater], full_regular[greater]
        )
        np.testing.assert_array_equal(kernel.regular_greater, 0.0)

    def test_odd_order_difference_has_no_contact_but_both_regular_sides(self):
        x = np.array([0.5, 2.0])
        theta = np.array([0.75, 1.5])
        kernel = constant_leg_kernel(0, 1, x, theta, SlepianConfig())

        self.assertEqual(kernel.contact_coefficient, 0)
        self.assertTrue(np.any(kernel.regular_less != 0.0))
        self.assertTrue(np.any(kernel.regular_greater != 0.0))

    def test_regular_classification_does_not_depend_on_sampled_off_diagonal(self):
        kernel = constant_leg_kernel(
            0, 1, np.array([1.0]), np.array([1.0]), SlepianConfig()
        )
        self.assertTrue(kernel.has_regular)
        np.testing.assert_array_equal(kernel.regular_less, 0.0)
        np.testing.assert_array_equal(kernel.regular_greater, 0.0)

    def test_regular_quadrature_support_is_explicit(self):
        self.assertTrue(_regular_quadrature_supported(0, 2))
        self.assertTrue(_regular_quadrature_supported(4, -2))
        self.assertFalse(_regular_quadrature_supported(0, 0))
        self.assertFalse(_regular_quadrature_supported(-1, 1))
        self.assertFalse(_regular_quadrature_supported(0, 1))

    def test_regular_x_grid_contains_targets_and_padded_range(self):
        config = SlepianConfig(regular_n_x=16, regular_x_padding=5.0)
        calculator = SlepianCalculator(config)
        theta = np.array([0.5, 1.0, 2.0])
        x = calculator._regular_x_grid(theta)

        self.assertEqual(x[0], theta[0] / 5.0)
        self.assertEqual(x[-1], theta[-1] * 5.0)
        for target in theta:
            self.assertTrue(np.any(x == target))
        self.assertTrue(np.all(np.diff(x) > 0.0))

    def test_regular_ratio_rule_is_open_and_integrates_constant(self):
        calculator = SlepianCalculator(
            SlepianConfig(regular_n_ratio=24, regular_ratio_min=0.2)
        )
        ratio, weights = calculator._regular_ratio_rule()

        self.assertTrue(np.all(ratio > 0.2))
        self.assertTrue(np.all(ratio < 1.0))
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(np.sum(weights), 0.8, places=14)

    def test_regular_quadrature_contracts_x_and_both_theta_axes(self):
        radial = SlepianRadialFactor2D(lambda ell: np.exp(-np.asarray(ell)))
        expression = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(radial, radial, SlepianRadialFactor2D.constant()),
            angular_orders=(-2, 0, 2),
        )
        bispectrum = Bispectrum2D(
            (BispectrumTerm2D("regular", (expression,)),)
        )
        calculator = SlepianCalculator(
            SlepianConfig(regular_n_x=16, regular_quadrature="legacy_log")
        )
        ell = np.geomspace(1.0, 10.0, 12)
        theta = np.array([0.5, 1.0])
        x = np.array([0.25, 0.75, 1.5])
        radial1 = np.array([2.0, 3.0, 5.0])
        radial2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        regular = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        sampled = ConstantLegKernel(0, regular, np.zeros_like(regular), True)
        diagonal = ConstantLegKernel(
            0, np.zeros((2, 2)), np.zeros((2, 2)), True
        )

        with (
            patch.object(calculator, "_regular_x_grid", return_value=x),
            patch.object(
                calculator,
                "_constant_leg_kernel",
                side_effect=(diagonal, sampled),
            ),
            patch(
                "fastnc.threepcf.slepian.single_radial_transform",
                return_value=radial1,
            ),
            patch(
                "fastnc.threepcf.slepian.double_radial_transform",
                return_value=radial2,
            ),
        ):
            result = calculator.evaluate_modes(bispectrum, ell, theta, [0])[0]

        integrand = (
            x[:, None, None]
            * radial1[:, None, None]
            * radial2[:, :, None]
            * regular[:, None, :]
        )
        expected = np.trapezoid(integrand, x, axis=0) / (2.0 * np.pi) ** 2
        np.testing.assert_allclose(result, expected)

    def test_regular_quadrature_rejects_two_sided_odd_order_kernel(self):
        radial = SlepianRadialFactor2D(lambda ell: np.exp(-np.asarray(ell)))
        expression = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(radial, radial, SlepianRadialFactor2D.constant()),
            angular_orders=(-1, 0, 1),
        )
        bispectrum = Bispectrum2D(
            (BispectrumTerm2D("two-sided", (expression,)),)
        )
        with self.assertRaisesRegex(NotImplementedError, "positive even"):
            SlepianCalculator(SlepianConfig()).evaluate_modes(
                bispectrum, np.geomspace(1.0, 10.0, 12), np.array([0.5]), [0]
            )

    def test_full_matrix_contraction_validates_coefficient_shapes(self):
        matrix = RegularMellinMatrix(
            np.ones((2, 3, 1, 1)),
            np.array([0.0, 1.0j]),
            np.array([0.0, 1.0j, -1.0j]),
            np.array([0.5, 1.0]),
            np.array([1.0]),
            0,
            (0, 0),
            (0, 2),
        )
        result = contract_regular_mellin_matrix(
            matrix,
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0]),
        )
        np.testing.assert_allclose(result, 54.0)
        with self.assertRaisesRegex(ValueError, "single_coefficients"):
            contract_regular_mellin_matrix(matrix, np.ones(2), np.ones(2))

    def test_low_rank_matrix_reconstructs_and_contracts_full_rank(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(3, 4, 2, 2)) + 1j * rng.normal(
            size=(3, 4, 2, 2)
        )
        matrix = RegularMellinMatrix(
            values,
            1j * np.arange(3),
            1j * np.arange(4),
            np.array([0.5, 1.0]),
            np.array([1.0, 2.0]),
            0,
            (0, 0),
            (0, 2),
        )
        compressed = compress_regular_mellin_matrix(matrix, rank=3)
        self.assertIsInstance(compressed, LowRankRegularMellinMatrix)
        self.assertEqual(compressed.retained_rank, 3)
        self.assertLess(compressed.relative_reconstruction_error, 1.0e-14)
        single = rng.normal(size=4) + 1j * rng.normal(size=4)
        double = rng.normal(size=3) + 1j * rng.normal(size=3)
        expected = contract_regular_mellin_matrix(matrix, single, double)
        actual = contract_low_rank_regular_mellin_matrix(
            compressed, single, double
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)
        automatic = compress_regular_mellin_matrix(matrix, rtol=1.0e-12)
        self.assertLessEqual(automatic.relative_reconstruction_error, 1.0e-12)

    def test_low_rank_los_contraction_avoids_dense_coefficient_matrix(self):
        rng = np.random.default_rng(123)
        values = rng.normal(size=(3, 4, 2, 2)) + 1j * rng.normal(
            size=(3, 4, 2, 2)
        )
        double_exponents = 1j * np.arange(3)
        single_exponents = 1j * np.arange(4)
        matrix = RegularMellinMatrix(
            values,
            double_exponents,
            single_exponents,
            np.array([0.5, 1.0]),
            np.array([1.0, 2.0]),
            0,
            (0, 0),
            (0, 2),
        )
        compressed = compress_regular_mellin_matrix(matrix, rank=3)
        z = np.linspace(0.2, 1.0, 6)
        chi = np.geomspace(500.0, 2500.0, z.size)
        weight = (1.0 + z) / chi**2
        single = rng.normal(size=(z.size, 4)) + 1j * rng.normal(
            size=(z.size, 4)
        )
        double = rng.normal(size=(z.size, 3)) + 1j * rng.normal(
            size=(z.size, 3)
        )
        factors = LOSMellinFactors(
            z=z,
            chi=chi,
            weight=weight,
            single_coefficients=single,
            double_coefficients=double,
            single_exponents=single_exponents,
            double_exponents=double_exponents,
        )

        per_node = np.stack(
            [
                contract_regular_mellin_matrix(matrix, single[i], double[i])
                for i in range(z.size)
            ]
        )
        expected = np.trapezoid(
            weight[:, None, None] * per_node,
            chi,
            axis=0,
        )
        dense_coefficients = np.trapezoid(
            weight[:, None, None]
            * double[:, :, None]
            * single[:, None, :],
            chi,
            axis=0,
        )
        dense_result = np.einsum(
            "ab,abij->ij",
            dense_coefficients,
            matrix.values,
        )
        actual = contract_low_rank_regular_mellin_matrix_los(
            compressed,
            factors,
        )

        np.testing.assert_allclose(dense_result, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        self.assertFalse(factors.single_coefficients.flags.writeable)
        self.assertFalse(factors.double_coefficients.flags.writeable)
        mismatched = LOSMellinFactors(
            z=z,
            chi=chi,
            weight=weight,
            single_coefficients=single,
            double_coefficients=double,
            single_exponents=single_exponents + 1.0,
            double_exponents=double_exponents,
        )
        with self.assertRaisesRegex(ValueError, "single_exponents"):
            contract_low_rank_regular_mellin_matrix_los(
                compressed,
                mismatched,
            )

    def test_full_matrix_matches_quadrature_and_is_cached(self):
        radial1 = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 8.0) ** 2)
        )
        radial2 = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 15.0) ** 2)
        )
        expression = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(
                radial1,
                radial2,
                SlepianRadialFactor2D.constant(),
            ),
            angular_orders=(-2, 0, 2),
        )
        bispectrum = Bispectrum2D(
            (BispectrumTerm2D("regular-matrix", (expression,)),)
        )
        ell = np.geomspace(1.0e-2, 1.0e3, 16)
        theta = np.geomspace(2.0e-2, 2.0e-1, 4)
        common = dict(
            regular_n_x=32,
            regular_x_padding=10.0,
            regular_n_ratio=24,
        )
        quadrature = SlepianCalculator(
            SlepianConfig(**common, regular_method="quadrature")
        ).evaluate_modes(bispectrum, ell, theta, [0])[0]
        calculator = SlepianCalculator(
            SlepianConfig(**common, regular_method="full_matrix")
        )
        with patch(
            "fastnc.threepcf.slepian.regular_mellin_matrix",
            wraps=regular_mellin_matrix,
        ) as build:
            full_matrix = calculator.evaluate_modes(
                bispectrum, ell, theta, [0]
            )[0]
            repeated = calculator.evaluate_modes(bispectrum, ell, theta, [0])[0]

        np.testing.assert_allclose(full_matrix, quadrature, rtol=2.0e-12)
        np.testing.assert_allclose(repeated, full_matrix)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(len(calculator._regular_mellin_matrices), 1)

    def test_low_rank_route_matches_full_matrix_and_is_cached(self):
        radial1 = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 8.0) ** 2)
        )
        radial2 = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 15.0) ** 2)
        )
        expression = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(
                radial1,
                radial2,
                SlepianRadialFactor2D.constant(),
            ),
            angular_orders=(-2, 0, 2),
        )
        bispectrum = Bispectrum2D(
            (BispectrumTerm2D("low-rank-matrix", (expression,)),)
        )
        ell = np.geomspace(1.0e-2, 1.0e3, 16)
        theta = np.geomspace(2.0e-2, 2.0e-1, 4)
        common = dict(regular_n_x=32, regular_x_padding=10.0)
        full = SlepianCalculator(
            SlepianConfig(**common, regular_method="full_matrix")
        ).evaluate_modes(bispectrum, ell, theta, [0])[0]
        calculator = SlepianCalculator(
            SlepianConfig(
                **common,
                regular_method="low_rank",
                regular_low_rank_rank=16,
            )
        )
        low_rank = calculator.evaluate_modes(bispectrum, ell, theta, [0])[0]
        repeated = calculator.evaluate_modes(bispectrum, ell, theta, [0])[0]
        np.testing.assert_allclose(low_rank, full, rtol=2.0e-12)
        np.testing.assert_allclose(repeated, low_rank)
        self.assertEqual(len(calculator._low_rank_regular_mellin_matrices), 1)
        self.assertEqual(calculator._regular_mellin_matrices, {})

    def test_route_supports_spin_on_nonreference_leg(self):
        manager = ThreePCF(
            ThreePCFConfig(
                spin=(0, 2, 0),
                kmax=0.0,
                ell_min=1.0,
                ell_max=1.0e3,
                n_ell=12,
            ),
            toy_bispectrum(),
            self.theta,
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        )
        table = manager.zetak()
        self.assertEqual(
            set(table.aliases),
            {ComponentModeKey.from_epsilon_k((1, 1, 1), 0.0)},
        )

    def test_route_supports_all_components_of_even_spin_triples(self):
        for spin in ((2, 0, 0), (2, 2, 2), (4, 2, 6)):
            with self.subTest(spin=spin):
                manager = ThreePCF(
                    ThreePCFConfig(
                        spin=spin,
                        kmax=2.0,
                        ell_min=0.1,
                        ell_max=1.0e3,
                        n_ell=16,
                        slepian=SlepianConfig(
                            regular_n_x=16,
                            regular_x_padding=5.0,
                        ),
                    ),
                    toy_bispectrum(),
                    np.geomspace(0.02, 0.08, 3),
                    np.linspace(0.2, 2.8, 5),
                    route="slepian",
                )
                table = manager.zeta()
                self.assertEqual(table.values.shape[0], 2 ** (sum(s != 0 for s in spin) - 1))
                self.assertTrue(np.all(np.isfinite(table.values)))

    def test_reference_spin_mode_matches_direct_radial_quadrature(self):
        bispectrum = toy_bispectrum()
        expression = next(bispectrum.iter_terms()).term.get_representation(
            SlepianRepresentation2D
        )
        ell = np.geomspace(1.0e-3, 1.0e4, 256)
        theta = np.array([0.002, 0.004])
        actual = SlepianCalculator(SlepianConfig()).evaluate_modes(
            bispectrum,
            ell,
            theta,
            [0.0],
            sigma=(2, 0, 0),
        )[0.0]

        integration_ell = np.geomspace(1.0e-4, 1.0e5, 50_000)
        factor1 = expression.radial_factors[0].evaluate(integration_ell)
        factor2 = expression.radial_factors[1].evaluate(integration_ell)
        radial1 = np.array(
            [
                np.trapezoid(
                    integration_ell * factor1 * jv(2, integration_ell * radius),
                    integration_ell,
                )
                for radius in theta
            ]
        )
        radial2 = np.array(
            [
                [
                    np.trapezoid(
                        integration_ell
                        * factor2
                        * jv(-1, integration_ell * x)
                        * jv(1, integration_ell * target),
                        integration_ell,
                    )
                    for target in theta
                ]
                for x in theta
            ]
        )
        expected = 2.0 * radial2.T * radial1[None, :] / (2.0 * np.pi) ** 2
        np.testing.assert_allclose(actual.real, expected, rtol=4.0e-4, atol=1.0e-5)
        self.assertLess(
            np.max(np.abs(actual.imag)) / np.max(np.abs(expected)),
            1.0e-6,
        )

    def test_route_supports_half_integer_modes(self):
        manager = ThreePCF(
            ThreePCFConfig(
                spin=(0, 1, 0),
                kmax=0.5,
                ell_min=1.0,
                ell_max=1.0e3,
                n_ell=12,
            ),
            toy_bispectrum(),
            self.theta,
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        )
        epsilon = (1, 1, 1)
        table = manager.zetak()
        self.assertEqual(
            set(table.aliases),
            {
                ComponentModeKey.from_epsilon_k(epsilon, -0.5),
                ComponentModeKey.from_epsilon_k(epsilon, 0.5),
            },
        )
        self.assertEqual(table.get_for_mode(epsilon, -0.5).shape, (3, 3))
        self.assertEqual(table.get_for_mode(epsilon, 0.5).shape, (3, 3))

    def test_spin_zetak_agrees_with_numeric_route(self):
        config = ThreePCFConfig(
            spin=(0, 2, 0),
            Lmax=12,
            kmax=0.0,
            ell_min=0.3,
            ell_max=3.0e3,
            n_ell=64,
            use_coupling_cache=False,
            multipole=NumericMultipoleConfig(
                n_angle=513,
                delta_beta_min=0.0,
                delta_beta_max=np.pi,
            ),
        )
        epsilon = (1, 1, 1)
        phi = np.linspace(0.0, np.pi, 4)
        numeric = ThreePCF(
            config, toy_bispectrum(), self.theta, phi, route="numeric"
        ).zetak().get_for_mode(epsilon, 0.0)
        slepian = ThreePCF(
            config, toy_bispectrum(), self.theta, phi, route="slepian"
        ).zetak().get_for_mode(epsilon, 0.0)
        relative = np.max(np.abs(slepian - numeric)) / np.max(np.abs(numeric))
        self.assertLess(relative, 3.0e-2)

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

    def test_constant_leg_two_and_three_contact_terms_are_symmetric(self):
        radial1 = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 8.0) ** 2)
        )
        radial2 = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 15.0) ** 2)
        )
        leg3_constant = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(
                radial1,
                radial2,
                SlepianRadialFactor2D.constant(),
            ),
        )
        leg2_constant = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(
                radial1,
                SlepianRadialFactor2D.constant(),
                radial2,
            ),
        )
        b_leg3 = Bispectrum2D(
            (BispectrumTerm2D("leg3-constant", (leg3_constant,)),)
        )
        b_leg2 = Bispectrum2D(
            (BispectrumTerm2D("leg2-constant", (leg2_constant,)),)
        )
        ell = np.geomspace(1.0e-2, 1.0e3, 16)
        theta = np.geomspace(2.0e-2, 2.0e-1, 4)
        calculator = SlepianCalculator(SlepianConfig())
        modes = (-1, 0, 1)

        from_leg3 = calculator.evaluate_modes(b_leg3, ell, theta, modes)
        from_leg2 = calculator.evaluate_modes(b_leg2, ell, theta, modes)

        for mode in modes:
            np.testing.assert_allclose(
                from_leg2[mode],
                from_leg3[-mode].T,
                rtol=2.0e-13,
                atol=2.0e-13,
            )

    def test_constant_leg_two_and_three_regular_terms_are_symmetric(self):
        radial1 = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 8.0) ** 2)
        )
        radial2 = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 15.0) ** 2)
        )
        leg3_constant = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(
                radial1,
                radial2,
                SlepianRadialFactor2D.constant(),
            ),
            angular_orders=(-2, 0, 2),
        )
        leg2_constant = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(
                radial1,
                SlepianRadialFactor2D.constant(),
                radial2,
            ),
            angular_orders=(-2, 2, 0),
        )
        b_leg3 = Bispectrum2D(
            (BispectrumTerm2D("leg3-regular", (leg3_constant,)),)
        )
        b_leg2 = Bispectrum2D(
            (BispectrumTerm2D("leg2-regular", (leg2_constant,)),)
        )
        ell = np.geomspace(1.0e-2, 1.0e3, 16)
        theta = np.geomspace(2.0e-2, 2.0e-1, 4)
        common = dict(regular_n_x=32, regular_x_padding=10.0)

        for method in ("quadrature", "full_matrix", "low_rank"):
            config = SlepianConfig(
                **common,
                regular_method=method,
                regular_low_rank_rank=16,
            )
            calculator = SlepianCalculator(config)
            from_leg3 = calculator.evaluate_modes(
                b_leg3, ell, theta, [0]
            )[0]
            from_leg2 = calculator.evaluate_modes(
                b_leg2, ell, theta, [0]
            )[0]
            np.testing.assert_allclose(
                from_leg2,
                from_leg3.T,
                rtol=2.0e-12,
                atol=2.0e-12,
                err_msg=f"regular_method={method}",
            )

    def test_projected_constant_leg_two_and_three_are_symmetric(self):
        radial1 = SlepianRadialFactor3D(
            lambda k, z: np.exp(-0.5 * np.log(np.asarray(k) / 8.0e-3) ** 2)
        )
        radial2 = SlepianRadialFactor3D(
            lambda k, z: np.exp(-0.5 * np.log(np.asarray(k) / 1.5e-2) ** 2)
        )
        leg3_constant = SlepianExpression3D(
            coefficient=1.0,
            radial_factors=(
                radial1,
                radial2,
                SlepianRadialFactor3D.constant(),
            ),
            angular_orders=(-2, 0, 2),
        )
        leg2_constant = SlepianExpression3D(
            coefficient=1.0,
            radial_factors=(
                radial1,
                SlepianRadialFactor3D.constant(),
                radial2,
            ),
            angular_orders=(-2, 2, 0),
        )
        z = np.array([0.3, 0.5, 0.8])
        chi = np.array([800.0, 1100.0, 1500.0])
        projector = LOSProjector(z=z, chi=chi, prefactor=np.ones(3))
        b_leg3 = projector.project(
            Bispectrum3D(
                (BispectrumTerm3D("leg3-projected", (leg3_constant,)),)
            )
        )
        b_leg2 = projector.project(
            Bispectrum3D(
                (BispectrumTerm3D("leg2-projected", (leg2_constant,)),)
            )
        )
        ell = np.geomspace(1.0e-2, 1.0e3, 16)
        theta = np.geomspace(2.0e-2, 2.0e-1, 3)
        calculator = SlepianCalculator(
            SlepianConfig(
                regular_method="full_matrix",
                regular_n_x=16,
                regular_x_padding=10.0,
            )
        )

        from_leg3 = calculator.evaluate_modes(b_leg3, ell, theta, [0])[0]
        from_leg2 = calculator.evaluate_modes(b_leg2, ell, theta, [0])[0]
        np.testing.assert_allclose(
            from_leg2,
            from_leg3.T,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def test_constant_leg_one_is_explicitly_unsupported(self):
        radial = SlepianRadialFactor2D(
            lambda ell: np.exp(-0.5 * np.log(np.asarray(ell) / 8.0) ** 2)
        )
        expression = SlepianExpression2D(
            coefficient=1.0,
            radial_factors=(
                SlepianRadialFactor2D.constant(),
                radial,
                radial,
            ),
        )
        bispectrum = Bispectrum2D(
            (BispectrumTerm2D("leg1-constant", (expression,)),)
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "cannot eliminate physical leg 1",
        ):
            SlepianCalculator(SlepianConfig()).evaluate_modes(
                bispectrum,
                np.geomspace(1.0e-2, 1.0e3, 16),
                np.geomspace(2.0e-2, 2.0e-1, 3),
                [0],
            )

    def test_source_updates_preserve_structural_slepian_cache(self):
        state = {"scale": 1.0, "revision": 0}

        def make_bispectrum(scale, revision_source):
            radial1 = SlepianRadialFactor2D(
                lambda ell: scale()
                * np.exp(-0.5 * np.log(np.asarray(ell) / 8.0) ** 2)
            )
            radial2 = SlepianRadialFactor2D(
                lambda ell: np.exp(
                    -0.5 * np.log(np.asarray(ell) / 15.0) ** 2
                )
            )
            expression = SlepianExpression2D(
                coefficient=1.0,
                radial_factors=(
                    radial1,
                    radial2,
                    SlepianRadialFactor2D.constant(),
                ),
                angular_orders=(-2, 0, 2),
            )
            return Bispectrum2D(
                (BispectrumTerm2D("stateful-regular", (expression,)),),
                _revision_sources=(revision_source,),
            )

        bispectrum = make_bispectrum(
            lambda: state["scale"],
            lambda: state["revision"],
        )
        config = ThreePCFConfig(
            kmax=0.0,
            ell_min=1.0e-2,
            ell_max=1.0e3,
            n_ell=16,
            slepian=SlepianConfig(
                regular_method="full_matrix",
                regular_n_x=16,
                regular_x_padding=10.0,
            ),
        )
        manager = ThreePCF(
            config,
            bispectrum,
            np.geomspace(2.0e-2, 2.0e-1, 3),
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        )

        with patch(
            "fastnc.threepcf.slepian.regular_mellin_matrix",
            wraps=regular_mellin_matrix,
        ) as build:
            first = manager.zetak().get_for_mode((1, 1, 1), 0.0)
            calculator = manager._slepian_calculator

            state["scale"] = 2.0
            state["revision"] += 1
            updated = manager.zetak().get_for_mode((1, 1, 1), 0.0)

            replacement = make_bispectrum(lambda: 3.0, lambda: 0)
            manager.set_bispectrum(replacement)
            replaced = manager.zetak().get_for_mode((1, 1, 1), 0.0)

        np.testing.assert_allclose(updated, 2.0 * first, rtol=2.0e-13)
        np.testing.assert_allclose(replaced, 3.0 * first, rtol=2.0e-13)
        self.assertIs(manager._slepian_calculator, calculator)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(len(calculator._regular_mellin_matrices), 1)

    def test_projector_change_preserves_structural_slepian_cache(self):
        def radial(k, center):
            return np.exp(-0.5 * np.log(np.asarray(k) / center) ** 2)

        expression = SlepianExpression3D(
            coefficient=1.0,
            radial_factors=(
                SlepianRadialFactor3D(
                    lambda k, z: radial(k, 8.0e-3)
                ),
                SlepianRadialFactor3D(
                    lambda k, z: radial(k, 1.5e-2)
                ),
                SlepianRadialFactor3D.constant(),
            ),
            angular_orders=(-2, 0, 2),
        )
        b3d = Bispectrum3D(
            (BispectrumTerm3D("projected-regular", (expression,)),)
        )
        z = np.array([0.3, 0.5, 0.8])
        chi = np.array([800.0, 1100.0, 1500.0])
        first_b2d = LOSProjector(
            z=z, chi=chi, prefactor=np.ones(3)
        ).project(b3d)
        second_b2d = LOSProjector(
            z=z, chi=chi, prefactor=np.full(3, 2.0)
        ).project(b3d)
        config = ThreePCFConfig(
            kmax=0.0,
            ell_min=1.0e-2,
            ell_max=1.0e3,
            n_ell=16,
            slepian=SlepianConfig(
                regular_method="full_matrix",
                regular_n_x=16,
                regular_x_padding=10.0,
            ),
        )
        manager = ThreePCF(
            config,
            first_b2d,
            np.geomspace(2.0e-2, 2.0e-1, 3),
            np.linspace(0.0, np.pi, 4),
            route="slepian",
        )

        with patch(
            "fastnc.threepcf.slepian.regular_mellin_matrix",
            wraps=regular_mellin_matrix,
        ) as build:
            first = manager.zetak().get_for_mode((1, 1, 1), 0.0)
            calculator = manager._slepian_calculator
            manager.set_bispectrum(second_b2d)
            second = manager.zetak().get_for_mode((1, 1, 1), 0.0)

        np.testing.assert_allclose(second, 2.0 * first, rtol=2.0e-13)
        self.assertIs(manager._slepian_calculator, calculator)
        self.assertEqual(build.call_count, 1)

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

    def test_finite_width_projected_zetak_agrees_with_numeric_route(self):
        z = np.linspace(0.3, 0.8, 9)
        chi = 500.0 + 1200.0 * z
        projector = LOSProjector(
            z=z,
            chi=chi,
            prefactor=np.exp(-0.5 * ((z - 0.55) / 0.16) ** 2),
        )

        def radial(k, z_value, center):
            center_at_z = center * (1.0 + 0.1 * np.asarray(z_value))
            k = np.asarray(k)
            safe_k = np.maximum(k, np.finfo(float).tiny)
            return np.where(
                k > 0.0,
                np.exp(-0.5 * np.log(safe_k / center_at_z) ** 2),
                0.0,
            )

        def coefficient(z_value):
            return 1.2 * (1.0 + np.asarray(z_value))

        slepian = SlepianExpression3D(
            coefficient=coefficient,
            radial_factors=(
                SlepianRadialFactor3D(
                    lambda k, z_value: radial(k, z_value, 0.035)
                ),
                SlepianRadialFactor3D(
                    lambda k, z_value: radial(k, z_value, 0.055)
                ),
                SlepianRadialFactor3D.constant(),
            ),
        )
        numeric = NumericExpression3D(
            lambda k1, k2, k3, z_value: coefficient(z_value)
            * radial(k1, z_value, 0.035)
            * radial(k2, z_value, 0.055)
        )
        b3d = Bispectrum3D(
            (BispectrumTerm3D("finite-los-benchmark", (numeric, slepian)),)
        )
        b2d = projector.project(b3d)
        theta = np.geomspace(0.02, 0.08, 3)
        config = ThreePCFConfig(
            Lmax=12,
            kmax=0.0,
            ell_min=0.3,
            ell_max=3.0e3,
            n_ell=64,
            use_coupling_cache=False,
            multipole=NumericMultipoleConfig(
                n_angle=513,
                delta_beta_min=0.0,
                delta_beta_max=np.pi,
            ),
        )
        phi = np.linspace(0.0, np.pi, 4)
        numeric_zetak = ThreePCF(
            config, b2d, theta, phi, route="numeric"
        ).zetak().get_for_mode((1, 1, 1), 0.0)
        slepian_zetak = ThreePCF(
            config, b2d, theta, phi, route="slepian"
        ).zetak().get_for_mode((1, 1, 1), 0.0)

        relative = np.max(np.abs(slepian_zetak - numeric_zetak)) / np.max(
            np.abs(numeric_zetak)
        )
        self.assertLess(relative, 6.0e-2)

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

    def test_projected_slepian_rejects_unknown_3d_implementation(self):
        class ToySlepianRepresentation3D(SlepianRepresentation3D):
            pass

        b3d = Bispectrum3D(
            [
                BispectrumTerm3D(
                    "projected-slepian",
                    (ToySlepianRepresentation3D(),),
                )
            ]
        )
        b2d = LOSProjector(
            z=np.array([0.4, 0.6]),
            chi=np.array([900.0, 1100.0]),
        ).project(b3d)
        calculator = SlepianCalculator(self.manager.config.slepian)

        with self.assertRaisesRegex(TypeError, "unsupported"):
            calculator.evaluate_modes(
                b2d,
                self.manager.grid.ell,
                self.theta,
                [0],
            )

    def test_finite_width_projected_routes_match_nodewise_native_benchmark(self):
        z = np.array([0.3, 0.5, 0.8])
        chi = np.array([800.0, 1100.0, 1500.0])
        projector = LOSProjector(z=z, chi=chi, prefactor=np.array([1.0, 1.3, 0.7]))

        def radial(k, z_value, center):
            center_at_z = center * (1.0 + 0.15 * np.asarray(z_value))
            return np.exp(-0.5 * np.log(np.asarray(k) / center_at_z) ** 2)

        source = SlepianExpression3D(
            coefficient=lambda z_value: 1.5 * (1.0 + z_value),
            radial_factors=(
                SlepianRadialFactor3D(
                    lambda k, z_value: radial(k, z_value, 0.03)
                ),
                SlepianRadialFactor3D(
                    lambda k, z_value: radial(k, z_value, 0.05)
                ),
                SlepianRadialFactor3D.constant(),
            ),
            angular_orders=(-2, 0, 2),
        )
        b3d = Bispectrum3D((BispectrumTerm3D("finite-los", (source,)),))
        b2d = projector.project(b3d)
        ell = np.geomspace(1.0, 300.0, 12)
        theta = np.geomspace(0.02, 0.08, 2)
        common = dict(
            regular_n_x=16,
            regular_x_padding=6.0,
            regular_n_ratio=16,
        )

        node_values = []
        for z_value, chi_value in zip(z, chi):
            factors = tuple(
                SlepianRadialFactor2D.constant()
                if factor.is_constant
                else SlepianRadialFactor2D(
                    lambda ell_value, factor=factor, z_value=z_value,
                    chi_value=chi_value: factor.evaluate(
                        np.asarray(ell_value) / chi_value, z_value
                    )
                )
                for factor in source.radial_factors
            )
            native = SlepianExpression2D(
                coefficient=source.coefficient_at(z_value),
                radial_factors=factors,
                angular_orders=source.angular_orders,
            )
            native_bispectrum = Bispectrum2D(
                (BispectrumTerm2D("node", (native,)),)
            )
            node_values.append(
                SlepianCalculator(
                    SlepianConfig(**common, regular_method="quadrature")
                ).evaluate_modes(native_bispectrum, ell, theta, [0])[0]
            )
        benchmark = projector.integrate_coefficients(
            np.stack(node_values), axis=0
        )

        quadrature = SlepianCalculator(
            SlepianConfig(**common, regular_method="quadrature")
        ).evaluate_modes(b2d, ell, theta, [0])[0]
        full = SlepianCalculator(
            SlepianConfig(**common, regular_method="full_matrix")
        ).evaluate_modes(b2d, ell, theta, [0])[0]
        low_rank = SlepianCalculator(
            SlepianConfig(
                **common,
                regular_method="low_rank",
                regular_low_rank_rank=12,
            )
        ).evaluate_modes(b2d, ell, theta, [0])[0]

        np.testing.assert_allclose(quadrature, benchmark, rtol=2.0e-12)
        np.testing.assert_allclose(full, benchmark, rtol=2.0e-12)
        np.testing.assert_allclose(low_rank, benchmark, rtol=2.0e-12)

    def test_delta_projected_slepian_matches_native_2d_expression(self):
        z0 = 0.5
        chi0 = 1000.0
        shift = 0.5

        def radial(value, center):
            value = np.asarray(value, dtype=float)
            return np.exp(-0.5 * np.log(value / center) ** 2)

        expression3d = SlepianExpression3D(
            coefficient=lambda z: 2.0 * (1.0 + z),
            radial_factors=(
                SlepianRadialFactor3D(
                    lambda k, z: radial(k, 30.0 / chi0)
                ),
                SlepianRadialFactor3D(
                    lambda k, z: radial(k, 50.0 / chi0)
                ),
                SlepianRadialFactor3D.constant(),
            ),
        )
        term3d = BispectrumTerm3D("toy-3d", (expression3d,)).scaled_by(
            lambda z: 3.0 - z
        )
        projected = LOSProjector.delta_like(
            z=z0,
            chi=chi0,
            shift=shift,
        ).project(Bispectrum3D((term3d,)))

        native = Bispectrum2D(
            (
                BispectrumTerm2D(
                    "toy-2d",
                    (
                        SlepianExpression2D(
                            coefficient=(3.0 - z0) * 2.0 * (1.0 + z0),
                            radial_factors=(
                                SlepianRadialFactor2D(
                                    lambda ell: radial(ell + shift, 30.0)
                                ),
                                SlepianRadialFactor2D(
                                    lambda ell: radial(ell + shift, 50.0)
                                ),
                                SlepianRadialFactor2D.constant(),
                            ),
                        ),
                    ),
                ),
            )
        )
        calculator = SlepianCalculator(self.manager.config.slepian)
        ell = self.manager.grid.ell
        expected = calculator.evaluate_modes(native, ell, self.theta, [0])[0]
        actual = calculator.evaluate_modes(projected, ell, self.theta, [0])[0]
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
