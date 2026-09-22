import unittest

import numpy as np

from fastnc.projection import (
    KernelSet,
    RadialKernel,
    angular_to_comoving,
    evaluate_numeric_los,
    integrate_coefficients,
    integrate_numeric_los,
)


class ProjectionPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.z = np.array([0.0, 0.5, 1.0])
        self.chi = np.array([1.0, 2.0, 4.0])

    def test_angular_to_comoving_uses_ell_over_chi(self):
        np.testing.assert_allclose(
            angular_to_comoving(np.array([2.0, 6.0]), self.chi),
            np.array([[2.0, 1.0, 0.5], [6.0, 3.0, 1.5]]),
        )

    def test_numeric_los_evaluates_all_three_legs_and_integrates(self):
        def evaluator(k1, k2, k3, z):
            return (k1 + 2.0 * k2 + 3.0 * k3) * (1.0 + z)

        sampled = evaluate_numeric_los(
            evaluator,
            ell1=np.array([2.0, 4.0]),
            ell2=3.0,
            ell3=5.0,
            z=self.z,
            chi=self.chi,
        )
        expected = (
            angular_to_comoving(np.array([2.0, 4.0]), self.chi)
            + 2.0 * angular_to_comoving(np.array([3.0, 3.0]), self.chi)
            + 3.0 * angular_to_comoving(np.array([5.0, 5.0]), self.chi)
        ) * (1.0 + self.z[None, :])
        np.testing.assert_allclose(sampled.values, expected)
        np.testing.assert_allclose(
            integrate_numeric_los(sampled, self.chi, weight=self.chi**-4),
            np.trapezoid(expected * self.chi[None, :] ** -4, self.chi, axis=-1),
        )

    def test_radial_kernel_set_is_route_independent(self):
        first = RadialKernel(self.z, self.chi, np.array([1.0, 2.0, 3.0]))
        second = RadialKernel(self.z, self.chi, np.array([2.0, 2.0, 2.0]))
        kernels = KernelSet({"a": first, "b": second})
        np.testing.assert_allclose(
            kernels.product(("a", "b"), self.chi),
            first.weight * second.weight,
        )
        self.assertAlmostEqual(first.normalized().integral(), 1.0)

    def test_coefficient_integration_accepts_any_los_axis(self):
        coefficients = np.arange(12.0).reshape(2, 3, 2)
        expected = np.trapezoid(coefficients, self.chi, axis=1)
        np.testing.assert_allclose(
            integrate_coefficients(coefficients, self.chi, axis=1),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
