import unittest

import numpy as np

from fastnc.bispectrum import (
    BispectrumTerm2D,
    BispectrumTerm3D,
    CompositeBispectrum,
    NumericExpression2D,
    NumericExpression3D,
    SPTMatterBispectrum3D,
)
from fastnc.bispectrum.spt import _pair_cosine, f2_kernel


class BispectrumRepresentationTests(unittest.TestCase):
    def test_numeric_expressions_evaluate_directly(self):
        expression3d = NumericExpression3D(
            lambda k1, k2, k3, z: (k1 + k2 + k3) * (1.0 + z)
        )
        expression2d = NumericExpression2D(
            lambda ell1, ell2, ell3: ell1 + ell2 + ell3
        )
        self.assertEqual(expression3d(1.0, 2.0, 3.0, 1.0), 12.0)
        self.assertEqual(expression2d(1.0, 2.0, 3.0), 6.0)

    def test_missing_representation_reports_available_types(self):
        term = BispectrumTerm3D(
            "numeric-only",
            (NumericExpression3D(lambda k1, k2, k3, z: 1.0),),
        )
        with self.assertRaisesRegex(
            LookupError,
            "available representations: NumericExpression3D",
        ):
            term.get_representation(NumericExpression2D)

    def test_term_rejects_wrong_dimensional_representation(self):
        with self.assertRaisesRegex(TypeError, "3D representations"):
            BispectrumTerm3D(
                "wrong-dimension",
                (NumericExpression2D(lambda ell1, ell2, ell3: 1.0),),
            )

    def test_term_rejects_duplicate_representation_types(self):
        with self.assertRaisesRegex(ValueError, "duplicate representations"):
            BispectrumTerm3D(
                "duplicate",
                (
                    NumericExpression3D(lambda k1, k2, k3, z: 1.0),
                    NumericExpression3D(lambda k1, k2, k3, z: 2.0),
                ),
            )

    def test_scaling_and_composition_are_immutable(self):
        first = BispectrumTerm3D(
            "first",
            (NumericExpression3D(lambda k1, k2, k3, z: np.asarray(k1)),),
        )
        second = BispectrumTerm3D(
            "second",
            (NumericExpression3D(lambda k1, k2, k3, z: np.asarray(k2)),),
        )
        composite = 2.0 * first + (lambda z: 1.0 + z) * second
        self.assertIsInstance(composite, CompositeBispectrum)
        self.assertEqual(len(first.representations), 1)
        np.testing.assert_allclose(
            composite.evaluate(np.array([1.0, 2.0]), 3.0, 4.0, 0.5),
            np.array([6.5, 8.5]),
        )
        self.assertEqual(
            [item.term.name for item in composite.iter_terms()],
            ["first", "second"],
        )

    def test_two_and_three_dimensional_terms_cannot_be_combined(self):
        term3d = BispectrumTerm3D(
            "3d",
            (NumericExpression3D(lambda k1, k2, k3, z: 1.0),),
        )
        term2d = BispectrumTerm2D(
            "2d",
            (NumericExpression2D(lambda ell1, ell2, ell3: 1.0),),
        )
        with self.assertRaisesRegex(TypeError, "2D and 3D"):
            _ = term3d + term2d


class SPTMatterTermTests(unittest.TestCase):
    @staticmethod
    def linear_power(k, z):
        return np.asarray(k) ** -0.75 / (1.0 + np.asarray(z)) ** 2

    @classmethod
    def direct_reference(cls, k1, k2, k3, z):
        p1 = cls.linear_power(k1, z)
        p2 = cls.linear_power(k2, z)
        p3 = cls.linear_power(k3, z)
        return (
            2.0 * f2_kernel(k1, k2, _pair_cosine(k1, k2, k3)) * p1 * p2
            + 2.0 * f2_kernel(k2, k3, _pair_cosine(k2, k3, k1)) * p2 * p3
            + 2.0 * f2_kernel(k3, k1, _pair_cosine(k3, k1, k2)) * p3 * p1
        )

    def test_spt_terms_match_direct_formula_for_scalar_inputs(self):
        model = SPTMatterBispectrum3D(self.linear_power)
        expected = self.direct_reference(0.7, 1.1, 1.3, 0.4)
        self.assertAlmostEqual(model.evaluate(0.7, 1.1, 1.3, 0.4), expected)
        self.assertEqual(
            [term.name for term in model.terms],
            ["tree:F2:12", "tree:F2:23", "tree:F2:31"],
        )

    def test_spt_terms_match_direct_formula_for_los_shaped_inputs(self):
        model = SPTMatterBispectrum3D(self.linear_power)
        k1 = np.array([[0.5, 0.7, 0.9], [0.8, 1.0, 1.2]])
        k2 = np.array([[0.8, 1.0, 1.2], [0.9, 1.1, 1.3]])
        k3 = np.array([[1.0, 1.2, 1.4], [1.1, 1.3, 1.5]])
        z = np.array([[0.1, 0.3, 0.5], [0.1, 0.3, 0.5]])
        np.testing.assert_allclose(
            model.evaluate(k1, k2, k3, z),
            self.direct_reference(k1, k2, k3, z),
            rtol=1.0e-14,
            atol=0.0,
        )

    def test_state_update_is_seen_by_existing_terms(self):
        model = SPTMatterBispectrum3D(self.linear_power)
        terms_before = model.terms
        value_before = model.evaluate(0.7, 1.1, 1.3, 0.4)
        revision_before = model.state_revision

        model.update_physics(
            linear_power=lambda k, z: 2.0 * self.linear_power(k, z)
        )

        self.assertIs(model.terms, terms_before)
        self.assertEqual(model.state_revision, revision_before + 1)
        np.testing.assert_allclose(
            model.evaluate(0.7, 1.1, 1.3, 0.4),
            4.0 * value_before,
        )


if __name__ == "__main__":
    unittest.main()
