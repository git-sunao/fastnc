import unittest

import numpy as np

from fastnc.bispectrum import (
    BiHalofitBh1SemiAnalyticConfig,
    BiHalofitBispectrum3D,
    NumericExpression3D,
    SemiAnalyticLowRankProductExpression3D,
)
from fastnc.projection import LOSProjector
from fastnc.threepcf import SemiAnalyticCalculator, SemiAnalyticConfig


class BiHalofitBh1SemiAnalyticTests(unittest.TestCase):
    def make_model(self, **kwargs):
        return BiHalofitBispectrum3D.simple_debug(
            k=np.logspace(-4.0, 2.0, 192),
            z=np.linspace(0.0, 1.0, 24),
            **kwargs,
        )

    def test_bh1_always_owns_numeric_and_semi_analytic_representations(self):
        model = self.make_model()
        term = model.select_terms("bihalofit:Bh1").terms[0]
        term.get_representation(NumericExpression3D)
        expression = term.get_representation(
            SemiAnalyticLowRankProductExpression3D
        )
        self.assertEqual(expression.rank, 8)
        self.assertEqual(expression.trained_basis, "broad-debug-v1")

    def test_config_selects_validated_trained_rank(self):
        config = BiHalofitBh1SemiAnalyticConfig(
            rank=5, trained_basis="broad-debug-v1"
        )
        model = self.make_model(config_semi_analytic=config)
        expression = model.select_terms("bihalofit:Bh1").terms[0].get_representation(
            SemiAnalyticLowRankProductExpression3D
        )
        self.assertIs(model.config_semi_analytic, config)
        self.assertEqual(expression.rank, 5)
        with self.assertRaisesRegex(ValueError, "rank must be one of"):
            BiHalofitBh1SemiAnalyticConfig(rank=7)

    def test_low_rank_expression_approximates_exact_bh1(self):
        model = self.make_model()
        source = model.select_terms("bihalofit:Bh1")
        expression = source.terms[0].get_representation(
            SemiAnalyticLowRankProductExpression3D
        )
        k1 = np.array([0.4, 0.8, 1.2])
        k2 = np.array([0.7, 0.9, 1.1])
        k3 = np.array([0.9, 1.0, 1.4])
        exact = source(k1, k2, k3, 0.4)
        approximate = expression(k1, k2, k3, 0.4)
        np.testing.assert_allclose(approximate, exact, rtol=2.0e-4)

    def test_generic_calculator_evaluates_projected_low_rank_expression(self):
        model = self.make_model()
        source = model.select_terms("bihalofit:Bh1")
        projected = LOSProjector.delta_like(z=0.4, chi=700.0).project(source)
        values = SemiAnalyticCalculator(
            SemiAnalyticConfig(angular_nodes=64, mellin_nodes=32)
        ).evaluate(
            projected,
            np.array([0, 1]),
            ell2=np.array([80.0, 140.0]),
            ell3=np.array([120.0, 210.0]),
        )
        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
