import unittest

import numpy as np

from fastnc.bispectrum import (
    Bispectrum2D,
    BispectrumRepresentation2D,
    BispectrumTerm2D,
    InterpolatedNumericRepresentation2D,
    NumericExpression2D,
    NumericRepresentation2D,
    NumericSumRepresentation2D,
    TriangleInterpolationConfig,
)


def triangle_side(ell2, ell3, mu23):
    return np.sqrt(ell2**2 + ell3**2 + 2.0 * ell2 * ell3 * mu23)


class BispectrumInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.config = TriangleInterpolationConfig(
            ell2=np.geomspace(10.0, 1000.0, 9),
            ell3=np.geomspace(20.0, 800.0, 8),
            mu23=np.linspace(-0.9, 1.0, 11),
        )

    @staticmethod
    def expression(ell1, ell2, ell3):
        mu23 = (ell1**2 - ell2**2 - ell3**2) / (2.0 * ell2 * ell3)
        return 2.0 * np.log(ell2) + 3.0 * np.log(ell3) + 4.0 * mu23

    def make_bispectrum(self, evaluator=None, coefficient=1.0):
        representation = NumericExpression2D(evaluator or self.expression)
        term = BispectrumTerm2D("test", (representation,))
        return Bispectrum2D([term.scaled_by(coefficient)])

    def test_termwise_interpolation_is_a_numeric_representation(self):
        source = self.make_bispectrum(coefficient=2.0)
        interpolated = source.interpolate(self.config)
        representation = interpolated.terms[0].get_representation(
            NumericRepresentation2D
        )
        self.assertIsInstance(
            representation, InterpolatedNumericRepresentation2D
        )
        self.assertIs(
            representation.source_representation,
            source.terms[0].representations[0],
        )
        self.assertTrue(representation.cache_info()["ready"])
        self.assertEqual(
            representation.cache_info()["shape"],
            (9, 8, 11),
        )
        self.assertFalse(representation.cache.values.flags.writeable)

        ell2 = np.array([17.0, 230.0, 710.0])
        ell3 = np.array([31.0, 190.0, 620.0])
        mu23 = np.array([-0.3, 0.2, 0.75])
        ell1 = triangle_side(ell2, ell3, mu23)
        np.testing.assert_allclose(
            interpolated(ell1, ell2, ell3),
            2.0 * self.expression(ell1, ell2, ell3),
            atol=2.0e-12,
        )

    def test_cache_rebuilds_after_source_state_update(self):
        state = {"amplitude": 1.0}
        source = self.make_bispectrum(
            lambda ell1, ell2, ell3: state["amplitude"]
            * self.expression(ell1, ell2, ell3)
        )
        interpolated = source.interpolate(self.config)
        representation = interpolated.terms[0].get_representation(
            NumericRepresentation2D
        )
        old_cache = representation.cache

        ell2, ell3, mu23 = 100.0, 120.0, 0.25
        ell1 = triangle_side(ell2, ell3, mu23)
        old_value = interpolated(ell1, ell2, ell3)
        state["amplitude"] = 3.0
        source._state_updated()
        new_value = interpolated(ell1, ell2, ell3)

        self.assertIsNot(representation.cache, old_cache)
        self.assertAlmostEqual(new_value, 3.0 * old_value)

    def test_evaluation_lazily_builds_an_unprepared_cache(self):
        interpolated = self.make_bispectrum().interpolate(
            self.config, prepare=False
        )
        representation = interpolated.terms[0].get_representation(
            NumericRepresentation2D
        )
        self.assertFalse(representation.cache_info()["ready"])
        ell2, ell3, mu23 = 80.0, 90.0, -0.2
        ell1 = triangle_side(ell2, ell3, mu23)
        interpolated(ell1, ell2, ell3)
        self.assertTrue(representation.cache_info()["ready"])

    def test_non_numeric_representations_are_preserved(self):
        class OtherRepresentation2D(BispectrumRepresentation2D):
            pass

        term = BispectrumTerm2D(
            "mixed",
            (NumericExpression2D(self.expression), OtherRepresentation2D()),
        )
        source = Bispectrum2D([term])
        interpolated = source.interpolate(self.config)
        self.assertIsInstance(
            interpolated.terms[0].representations[1],
            OtherRepresentation2D,
        )

    def test_out_of_bounds_evaluation_raises(self):
        interpolated = self.make_bispectrum().interpolate(self.config)
        with self.assertRaises(ValueError):
            interpolated(5.0, 5.0, 5.0)

    def test_interpolation_returns_a_new_bispectrum_and_replaces_numeric(self):
        source = self.make_bispectrum()
        interpolated = source.interpolate(self.config)
        self.assertIsInstance(
            source.terms[0].representations[0], NumericExpression2D
        )
        self.assertIsInstance(
            interpolated.terms[0].representations[0],
            InterpolatedNumericRepresentation2D,
        )
        self.assertEqual(len(interpolated.terms[0].representations), 1)

    def test_numeric_terms_can_be_combined_before_interpolation(self):
        first = BispectrumTerm2D(
            "first",
            (NumericExpression2D(lambda e1, e2, e3: np.log(e2)),),
        ).scaled_by(2.0)
        second = BispectrumTerm2D(
            "second",
            (NumericExpression2D(lambda e1, e2, e3: np.log(e3)),),
        ).scaled_by(3.0)
        untouched = BispectrumTerm2D(
            "untouched",
            (NumericExpression2D(lambda e1, e2, e3: e1),),
        )
        source = Bispectrum2D([first, second, untouched])
        grouped = source.combine_numeric_terms(
            name="first+second",
            terms=("first", "second"),
        )
        self.assertEqual(
            [term.name for term in grouped.terms],
            ["first+second", "untouched"],
        )
        summed = grouped.terms[0].get_representation(
            NumericRepresentation2D
        )
        self.assertIsInstance(summed, NumericSumRepresentation2D)
        self.assertEqual(
            [item.term.name for item in summed.components],
            ["first", "second"],
        )

        ell2, ell3, mu23 = 100.0, 120.0, 0.25
        ell1 = triangle_side(ell2, ell3, mu23)
        self.assertAlmostEqual(
            grouped.select_terms("first+second")(ell1, ell2, ell3),
            2.0 * np.log(ell2) + 3.0 * np.log(ell3),
        )

        interpolated = grouped.select_terms("first+second").interpolate(
            self.config
        )
        representation = interpolated.terms[0].get_representation(
            NumericRepresentation2D
        )
        self.assertIsInstance(
            representation, InterpolatedNumericRepresentation2D
        )
        self.assertIsInstance(
            representation.source_representation,
            NumericSumRepresentation2D,
        )
        self.assertAlmostEqual(
            interpolated(ell1, ell2, ell3),
            2.0 * np.log(ell2) + 3.0 * np.log(ell3),
        )


if __name__ == "__main__":
    unittest.main()
