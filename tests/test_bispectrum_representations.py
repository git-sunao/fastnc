import unittest

import numpy as np

from fastnc.bispectrum import (
    BispectrumTerm2D,
    BispectrumTerm3D,
    Bispectrum2D,
    Bispectrum3D,
    BiHalofitBispectrum3D,
    NFWOneHaloBispectrum3D,
    NumericExpression2D,
    NumericExpression3D,
    SemiAnalyticExpression3D,
    SlepianExpression2D,
    SlepianExpression3D,
    SlepianRadialFactor2D,
    SPTGalaxyBispectrum3D,
    SPTMatterBispectrum3D,
)
from fastnc.bispectrum.models.spt import _pair_cosine, f2_kernel, tidal_kernel


class BispectrumRepresentationTests(unittest.TestCase):
    def test_slepian_expression_keeps_mathematical_factorization(self):
        factor = SlepianRadialFactor2D(lambda ell: np.exp(-np.asarray(ell)))
        constant = SlepianRadialFactor2D.constant()
        expression = SlepianExpression2D(
            coefficient=2.0,
            radial_factors=(factor, factor, constant),
            angular_orders=(1, -1, 0),
        )
        self.assertEqual(expression.constant_legs, (2,))
        np.testing.assert_allclose(constant([1.0, 2.0]), 1.0)
        with self.assertRaisesRegex(ValueError, "sum to zero"):
            SlepianExpression2D(
                1.0,
                (factor, factor, constant),
                (1, 0, 0),
            )

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
        self.assertIsInstance(composite, Bispectrum3D)
        self.assertEqual(len(first.representations), 1)
        np.testing.assert_allclose(
            composite.evaluate_numeric(np.array([1.0, 2.0]), 3.0, 4.0, 0.5),
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

    def test_native_2d_terms_form_a_typed_aggregate(self):
        first = BispectrumTerm2D(
            "first-2d",
            (NumericExpression2D(lambda ell1, ell2, ell3: np.asarray(ell1)),),
        )
        second = BispectrumTerm2D(
            "second-2d",
            (NumericExpression2D(lambda ell1, ell2, ell3: np.asarray(ell2)),),
        )
        b2d = first + 2.0 * second
        self.assertIsInstance(b2d, Bispectrum2D)
        np.testing.assert_allclose(b2d([1.0, 2.0], 3.0, 4.0), [7.0, 8.0])


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
        self.assertEqual(len(model.terms), 17)
        self.assertEqual(
            sum(name.startswith("tree:F2:12:") for name in (
                term.name for term in model.terms
            )),
            7,
        )
        self.assertEqual(
            sum(name.startswith("tree:F2:31:") for name in (
                term.name for term in model.terms
            )),
            7,
        )
        pair23 = tuple(
            term for term in model.terms if term.name.startswith("tree:F2:23:")
        )
        self.assertEqual(len(pair23), 3)
        for term in pair23:
            term.get_representation(SemiAnalyticExpression3D)
            with self.assertRaises(LookupError):
                term.get_representation(SlepianExpression3D)
        for term in model.terms:
            if not term.name.startswith("tree:F2:23:"):
                term.get_representation(SlepianExpression3D)

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
        selected = model.select_terms("tree:F2:12:m+0")
        value_before = model.evaluate(0.7, 1.1, 1.3, 0.4)
        revision_before = model.state_revision
        token_before = selected.state_token

        model.update_physics(
            linear_power=lambda k, z: 2.0 * self.linear_power(k, z)
        )

        self.assertIs(model.terms, terms_before)
        self.assertEqual(model.state_revision, revision_before + 1)
        self.assertNotEqual(selected.state_token, token_before)
        np.testing.assert_allclose(
            model.evaluate(0.7, 1.1, 1.3, 0.4),
            4.0 * value_before,
        )



class MigratedPhysicalModelTests(unittest.TestCase):
    @staticmethod
    def linear_power(k, z):
        return np.asarray(k) ** -0.75 / (1.0 + np.asarray(z)) ** 2

    def test_spt_galaxy_terms_reproduce_direct_formula(self):
        b1, b2, bK2 = 1.7, 0.4, -0.2
        model = SPTGalaxyBispectrum3D(
            self.linear_power,
            b1=b1,
            b2=b2,
            bK2=bK2,
        )
        k1, k2, k3, z = 0.7, 1.1, 1.3, 0.4
        p1 = self.linear_power(k1, z)
        p2 = self.linear_power(k2, z)
        p3 = self.linear_power(k3, z)
        mu12 = _pair_cosine(k1, k2, k3)
        mu23 = _pair_cosine(k2, k3, k1)
        mu31 = _pair_cosine(k3, k1, k2)
        tree = (
            2.0 * f2_kernel(k1, k2, mu12) * p1 * p2
            + 2.0 * f2_kernel(k2, k3, mu23) * p2 * p3
            + 2.0 * f2_kernel(k3, k1, mu31) * p3 * p1
        )
        quadratic = p1 * p2 + p2 * p3 + p3 * p1
        tidal = (
            tidal_kernel(mu12) * p1 * p2
            + tidal_kernel(mu23) * p2 * p3
            + tidal_kernel(mu31) * p3 * p1
        )
        expected = (
            b1**3 * tree
            + b1**2 * b2 * quadratic
            + 2.0 * b1**2 * bK2 * tidal
        )
        np.testing.assert_allclose(model(k1, k2, k3, z), expected)
        self.assertEqual(len(model.terms), 25)
        numeric_only = {
            "tree:F2:23",
            "bias:quadratic:23",
            "bias:tidal:23",
        }
        for term in model.terms:
            if term.name in numeric_only:
                with self.assertRaises(LookupError):
                    term.get_representation(SlepianExpression3D)
            else:
                term.get_representation(SlepianExpression3D)

    def test_one_halo_is_one_weighted_product_term(self):
        model = NFWOneHaloBispectrum3D(
            k_s=0.8,
            slope=2.0,
            amplitude_power=1.5,
            redshift_scaling=0.3,
            amplitude=lambda z: 2.0 + z,
        )
        triangle = (0.4, 0.8, 1.2, 0.5)
        expected = (2.0 + triangle[3]) * np.prod(
            [model.profile(k, triangle[3]) for k in triangle[:3]]
        )
        np.testing.assert_allclose(model(*triangle), expected)
        self.assertEqual([term.name for term in model.terms], ["one-halo:product"])
        token = model.state_token
        model.update_physics(amplitude=3.0)
        self.assertNotEqual(model.state_token, token)
        np.testing.assert_allclose(
            model(*triangle),
            3.0 * np.prod([model.profile(k, triangle[3]) for k in triangle[:3]]),
        )

    def test_bihalofit_terms_reproduce_direct_halofit_evaluation(self):
        model = BiHalofitBispectrum3D.simple_debug(
            k=np.logspace(-3, 1, 128),
            z=np.linspace(0.0, 1.0, 32),
        )
        k1 = np.array([0.1, 0.2])
        k2 = np.array([0.15, 0.25])
        k3 = np.array([0.2, 0.3])
        z = np.array([0.3, 0.5])
        expected = model.halofit.get_bihalofit(k1, k2, k3, z)
        np.testing.assert_allclose(model(k1, k2, k3, z), expected, rtol=1.0e-13)
        np.testing.assert_allclose(
            model.select_terms("bihalofit:Bh1")(k1, k2, k3, z),
            model.halofit.get_bihalofit(k1, k2, k3, z, which="Bh1"),
            rtol=1.0e-13,
        )

    def test_bihalofit_bh3_primitives_preserve_grouped_numeric_result(self):
        model = BiHalofitBispectrum3D.simple_debug(
            k=np.logspace(-5, 2, 256),
            z=np.linspace(0.0, 1.0, 32),
        )
        triangles = np.array(
            [
                [0.2, 0.2, 0.2],
                [0.1, 0.15, 0.2],
                [0.1, 0.1, 0.199999],
                [1.0e-3, 0.5, 0.5005],
                [1.0e-5, 0.5, 0.500005],
            ]
        )
        redshift = np.full(triangles.shape[0], 0.5)
        primitive_names = [
            term.name
            for term in model.terms
            if term.name.startswith("bihalofit:Bh3:")
            and term.name != "bihalofit:Bh3:squeezed-correction"
        ]
        self.assertEqual(len(primitive_names), 24)

        actual = model.select_terms("bihalofit:Bh3")(
            triangles[:, 0], triangles[:, 1], triangles[:, 2], redshift
        )
        expected = model.halofit.get_bihalofit(
            triangles[:, 0],
            triangles[:, 1],
            triangles[:, 2],
            redshift,
            which="Bh3",
        )
        np.testing.assert_allclose(actual.real, expected, rtol=5.0e-11)
        np.testing.assert_allclose(actual.imag, 0.0, atol=2.0e-8)

        correction = model.select_terms(
            "bihalofit:Bh3:squeezed-correction"
        )(
            triangles[:, 0], triangles[:, 1], triangles[:, 2], redshift
        )
        np.testing.assert_allclose(correction[:-1], 0.0, atol=0.0)
        self.assertNotEqual(correction[-1], 0.0)

        direct = model.select_terms("bihalofit:Bh3")(
            triangles[:, 0],
            triangles[:, 1],
            triangles[:, 2],
            redshift,
            squeezed_safe=False,
        )
        expected_direct = model.halofit.get_bihalofit(
            triangles[:, 0],
            triangles[:, 1],
            triangles[:, 2],
            redshift,
            which="Bh3",
            squeezed_safe=False,
        )
        np.testing.assert_allclose(direct.real, expected_direct, rtol=5.0e-11)


if __name__ == "__main__":
    unittest.main()
