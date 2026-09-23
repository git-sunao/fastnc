import unittest

import numpy as np

from fastnc.projection import (
    KernelSet,
    LOSProjector,
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

    def test_projector_composes_numeric_projection_primitives(self):
        kernels = KernelSet(
            {
                "source": RadialKernel(
                    self.z,
                    self.chi,
                    np.array([1.0, 2.0, 3.0]),
                )
            }
        )
        projector = LOSProjector(
            self.z,
            self.chi,
            kernels=kernels,
            prefactor=lambda z, chi: (1.0 + z) / chi,
            shift=0.5,
        )

        def evaluator(k1, k2, k3, z, amplitude):
            return amplitude * (k1 + k2 + k3) * (1.0 + z)

        sampled = projector.sample_numeric(
            evaluator,
            2.0,
            3.0,
            4.0,
            amplitude=2.0,
        )
        expected_weight = (1.0 + self.z) / self.chi * kernels.product(
            ("source",), self.chi
        )
        np.testing.assert_allclose(
            projector.weight(),
            (1.0 + self.z) / self.chi,
        )
        np.testing.assert_allclose(
            projector.weight(("source",)),
            expected_weight,
        )
        np.testing.assert_allclose(
            projector.project_numeric(
                evaluator,
                2.0,
                3.0,
                4.0,
                sample_combination=("source",),
                amplitude=2.0,
            ),
            integrate_numeric_los(sampled, self.chi, weight=expected_weight),
        )

    def test_projector_integrates_route_coefficients_without_route_types(self):
        projector = LOSProjector(self.z, self.chi, prefactor=self.chi**-2)
        coefficients = np.arange(12.0).reshape(2, 3, 2)
        np.testing.assert_allclose(
            projector.integrate_coefficients(coefficients, axis=1),
            integrate_coefficients(
                coefficients,
                self.chi,
                weight=self.chi**-2,
                axis=1,
            ),
        )

    def test_projector_uses_chi_minus_four_by_default(self):
        projector = LOSProjector(self.z, self.chi)
        np.testing.assert_allclose(projector.weight(), self.chi**-4)

    def test_projector_accepts_explicit_unity_prefactor(self):
        projector = LOSProjector(self.z, self.chi, prefactor=1.0)
        np.testing.assert_allclose(projector.weight(), np.ones_like(self.chi))

    def test_delta_like_kernel_product_is_normalized_at_requested_power(self):
        kernels = KernelSet.delta_like(
            z=0.5,
            chi=1000.0,
            width=20.0,
            power=3,
        )
        kernel = kernels["delta"]
        product = kernels.product(("delta", "delta", "delta"), kernel.chi)
        self.assertAlmostEqual(np.trapezoid(product, kernel.chi), 1.0)
        projector = LOSProjector(
            kernel.z,
            kernel.chi,
            kernels=kernels,
            prefactor=1.0,
        )
        np.testing.assert_allclose(
            projector.weight(("delta", "delta", "delta")),
            product,
        )

    def test_kernel_factories_preserve_v2_nz_and_lensing_conventions(self):
        nz = np.array([1.0, 2.0, 1.0])
        normalized_nz = nz / np.trapezoid(nz, self.z)
        n_chi = RadialKernel.from_nz(
            self.z,
            self.chi,
            nz,
            name="source",
        )
        np.testing.assert_allclose(
            n_chi.weight,
            normalized_nz * np.gradient(self.z, self.chi, edge_order=1),
        )

        lensing = RadialKernel.lensing_from_nz(
            self.z,
            self.chi,
            nz,
            omega_m=0.3,
        )
        self.assertGreater(lensing.weight[0], 0.0)
        self.assertAlmostEqual(lensing.weight[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
