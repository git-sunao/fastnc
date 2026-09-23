import unittest

import numpy as np

from fastnc.bispectrum import (
    Bispectrum3D,
    BispectrumTerm3D,
    NumericExpression2D,
    NumericExpression3D,
)
from fastnc.projection import LOSProjector
from fastnc.threepcf.routes.numeric import (
    NumericMultipoleCalculator,
    NumericMultipoleConfig,
    decompose_angular_multipoles,
    triangle_closing_side,
)


class AngularMultipoleKernelTests(unittest.TestCase):
    def setUp(self):
        self.delta_beta = np.linspace(0.0, np.pi, 2049)

    def test_constant_and_single_cosine_have_expected_even_coefficients(self):
        values = 3.0 + 2.0 * np.cos(4 * self.delta_beta)
        coefficients = decompose_angular_multipoles(
            values,
            self.delta_beta,
            modes=np.arange(7),
        )
        expected = np.zeros(7)
        expected[0] = 3.0
        expected[4] = 2.0
        np.testing.assert_allclose(coefficients, expected, atol=1.0e-5)

    def test_inner_and_outer_decomposition_store_same_outer_coefficients(self):
        values = (
            1.2
            - 0.4 * np.cos(self.delta_beta)
            + 0.7 * np.cos(3 * self.delta_beta)
        )
        modes = np.arange(6)
        outer = decompose_angular_multipoles(
            values,
            self.delta_beta,
            modes,
            decomposition_angle="outer",
        )
        inner = decompose_angular_multipoles(
            values,
            self.delta_beta,
            modes,
            decomposition_angle="inner",
        )
        np.testing.assert_allclose(inner, outer, rtol=1.0e-13, atol=1.0e-13)

    def test_full_fourier_duplicates_positive_and_negative_side_modes(self):
        values = 1.0 + 0.6 * np.cos(2 * self.delta_beta)
        modes = np.arange(-3, 4)
        coefficients = decompose_angular_multipoles(
            values,
            self.delta_beta,
            modes,
            basis="fourier",
        )
        expected = np.zeros(7)
        expected[modes == 0] = 1.0
        expected[np.abs(modes) == 2] = 0.3
        np.testing.assert_allclose(coefficients, expected, atol=1.0e-6)

    def test_triangle_closing_side_obeys_endpoint_geometry(self):
        ell2 = np.array([2.0, 5.0])
        ell3 = np.array([3.0, 1.0])
        np.testing.assert_allclose(
            triangle_closing_side(ell2, ell3, 0.0),
            ell2 + ell3,
        )
        np.testing.assert_allclose(
            triangle_closing_side(ell2, ell3, np.pi),
            np.abs(ell2 - ell3),
        )


class NumericMultipoleCalculatorTests(unittest.TestCase):
    def setUp(self):
        self.config = NumericMultipoleConfig(
            mode_max=4,
            n_angle=513,
            delta_beta_min=0.0,
            delta_beta_max=np.pi,
        )
        self.calculator = NumericMultipoleCalculator(self.config)

    def test_calculator_samples_broadcast_triangles_and_returns_mode_first(self):
        evaluator = lambda ell1, ell2, ell3: ell1**2 - ell2**2 - ell3**2
        ell2 = np.array([[2.0], [4.0]])
        ell3 = np.array([[3.0, 5.0, 7.0]])
        sampled = self.calculator.sample(evaluator, ell2, ell3)
        self.assertEqual(sampled.values.shape, (2, 3, self.config.n_angle))
        self.assertEqual(sampled.output_shape, (2, 3))
        np.testing.assert_allclose(
            sampled.ell1**2,
            sampled.ell2**2
            + sampled.ell3**2
            + 2.0
            * sampled.ell2
            * sampled.ell3
            * np.cos(sampled.delta_beta),
            atol=1.0e-13,
        )
        coefficients = self.calculator.decompose(sampled)
        self.assertEqual(coefficients.shape, (5, 2, 3))
        np.testing.assert_allclose(coefficients[0], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            coefficients[1],
            2.0 * ell2 * ell3,
            rtol=2.0e-5,
        )
        np.testing.assert_allclose(coefficients[2:], 0.0, atol=2.0e-12)

    def test_native_2d_and_fixed_redshift_3d_use_same_calculator(self):
        z = 0.7
        chi = 5.0
        calls = []

        def evaluator3d(k1, k2, k3, z_value):
            calls.append((k1.copy(), k2.copy(), k3.copy(), z_value))
            return (1.0 + z_value) * (k1**2 + 2.0 * k2 + 3.0 * k3)

        projector = LOSProjector.delta_like(z=z, chi=chi)
        b3d = Bispectrum3D(
            [BispectrumTerm3D("test", (NumericExpression3D(evaluator3d),))]
        )
        angularized = projector.project(b3d)
        native = NumericExpression2D(
            lambda ell1, ell2, ell3: (1.0 + z)
            * ((ell1 / chi) ** 2 + 2.0 * ell2 / chi + 3.0 * ell3 / chi)
        )
        ell2 = np.array([2.0, 4.0])
        ell3 = np.array([3.0, 6.0])
        from_3d = self.calculator.evaluate(angularized, ell2, ell3)
        from_2d = self.calculator.evaluate(native, ell2, ell3)
        np.testing.assert_allclose(from_3d, from_2d, atol=3.0e-14)

        sampled = self.calculator.sample(native, ell2, ell3)
        k1, k2, k3, z_seen = calls[0]
        np.testing.assert_allclose(k1, sampled.ell1 / chi)
        np.testing.assert_allclose(k2, sampled.ell2 / chi)
        np.testing.assert_allclose(k3, sampled.ell3 / chi)
        self.assertEqual(z_seen, z)


if __name__ == "__main__":
    unittest.main()
