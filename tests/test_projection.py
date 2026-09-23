import unittest

import numpy as np

from fastnc.bispectrum import (
    BispectrumRepresentation3D,
    Bispectrum2D,
    Bispectrum3D,
    BispectrumTerm3D,
    NumericExpression3D,
    NumericRepresentation2D,
)
from fastnc.projection import (
    Kernel1D,
    KernelSet,
    LOSProjector,
    ProjectedNumericRepresentation2D,
    angular_to_comoving,
    integrate_coefficients,
)
from fastnc.projection.numeric_los import evaluate_numeric_los, integrate_numeric_los


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
        first = Kernel1D(self.z, self.chi, np.array([1.0, 2.0, 3.0]))
        second = Kernel1D(self.z, self.chi, np.array([2.0, 2.0, 2.0]))
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
                "source": Kernel1D(
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

        b3d = Bispectrum3D(
            [
                BispectrumTerm3D(
                    "test",
                    (NumericExpression3D(evaluator),),
                )
            ]
        )
        b2d = projector.project(b3d, sample_combination=("source",))
        sampled = evaluate_numeric_los(
            evaluator,
            2.0,
            3.0,
            4.0,
            z=self.z,
            chi=self.chi,
            shift=0.5,
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
            b2d(2.0, 3.0, 4.0, amplitude=2.0),
            integrate_numeric_los(sampled, self.chi, weight=expected_weight),
        )
        self.assertIsInstance(b2d, Bispectrum2D)
        representation = b2d.terms[0].get_representation(
            NumericRepresentation2D
        )
        self.assertIsInstance(
            representation, ProjectedNumericRepresentation2D
        )
        self.assertIs(representation.source_term, b3d.weighted_terms[0])
        self.assertIs(
            representation.source_representation,
            b3d.terms[0].representations[0],
        )
        self.assertIs(representation.projector, projector)
        self.assertEqual(representation.sample_combination, ("source",))
        self.assertEqual(representation.projection_rule.name, "numeric_los")

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

    def test_delta_projector_evaluates_exactly_at_requested_z_and_chi(self):
        calls = []

        def evaluator(k1, k2, k3, z, amplitude):
            calls.append((k1, k2, k3, z))
            return amplitude * (k1 + 2.0 * k2 + 3.0 * k3) * (1.0 + z)

        projector = LOSProjector.delta_like(z=0.5, chi=1000.0)
        b3d = Bispectrum3D(
            [BispectrumTerm3D("test", (NumericExpression3D(evaluator),))]
        )
        b2d = projector.project(b3d)
        actual = b2d(
            np.array([100.0, 200.0]), 300.0, 400.0, amplitude=2.0
        )
        expected = 2.0 * (
            np.array([0.1, 0.2]) + 2.0 * 0.3 + 3.0 * 0.4
        ) * 1.5
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][3], 0.5)

    def test_delta_projector_returns_a_bispectrum_2d(self):
        def evaluator(k1, k2, k3, z):
            return k1 * k2 * k3 + z

        projector = LOSProjector.delta_like(z=0.7, chi=1200.0, shift=0.5)
        expected = evaluator(
            (20.0 + 0.5) / 1200.0,
            (30.0 + 0.5) / 1200.0,
            (40.0 + 0.5) / 1200.0,
            0.7,
        )
        b3d = Bispectrum3D(
            [BispectrumTerm3D("test", (NumericExpression3D(evaluator),))]
        )
        b2d = projector.project(b3d)
        np.testing.assert_allclose(b2d(20.0, 30.0, 40.0), expected)
        self.assertIsInstance(b2d, Bispectrum2D)
        self.assertEqual(b2d.terms[0].name, "test")
        self.assertTrue(projector.is_delta_like)

    def test_projection_does_not_silently_drop_unknown_representations(self):
        class UnsupportedRepresentation3D(BispectrumRepresentation3D):
            pass

        b3d = Bispectrum3D(
            [BispectrumTerm3D("unknown", (UnsupportedRepresentation3D(),))]
        )
        projector = LOSProjector.delta_like(z=0.5, chi=1000.0)
        with self.assertRaisesRegex(
            NotImplementedError,
            "UnsupportedRepresentation3D",
        ):
            projector.project(b3d)

    def test_projected_representation_keeps_redshift_dependent_weight(self):
        expression = NumericExpression3D(
            lambda k1, k2, k3, z: k1 + k2 + k3
        )
        term = BispectrumTerm3D("weighted", (expression,))
        b3d = Bispectrum3D([term.scaled_by(lambda z: 1.0 + z)])
        projector = LOSProjector.delta_like(z=0.5, chi=10.0)
        b2d = projector.project(b3d)
        self.assertAlmostEqual(b2d(10.0, 20.0, 30.0), 9.0)

    def test_delta_projector_selects_one_coefficient_value(self):
        projector = LOSProjector.delta_like(z=0.7, chi=1200.0)
        coefficients = np.arange(6.0).reshape(2, 1, 3)
        np.testing.assert_array_equal(
            projector.integrate_coefficients(coefficients, axis=1),
            coefficients[:, 0, :],
        )

    def test_kernel_factories_preserve_v2_nz_and_lensing_conventions(self):
        nz = np.array([1.0, 2.0, 1.0])
        normalized_nz = nz / np.trapezoid(nz, self.z)
        n_chi = Kernel1D.from_nz(
            self.z,
            self.chi,
            nz,
            name="source",
        )
        np.testing.assert_allclose(
            n_chi.weight,
            normalized_nz * np.gradient(self.z, self.chi, edge_order=1),
        )

        lensing = Kernel1D.lensing_from_nz(
            self.z,
            self.chi,
            nz,
            omega_m=0.3,
        )
        self.assertGreater(lensing.weight[0], 0.0)
        self.assertAlmostEqual(lensing.weight[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
