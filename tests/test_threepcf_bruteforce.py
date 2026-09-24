import unittest

import numpy as np

from fastnc.bispectrum import (
    Bispectrum2D,
    BispectrumTerm2D,
    NumericExpression2D,
)
from fastnc.threepcf import BruteForce3PCFConfig, BruteForceX3PCF


def _small_config(**overrides):
    values = {
        "ell_min": 1.0,
        "ell_max": 100.0,
        "n_ell": 32,
        "radial_cache_padding_factor": 2.0,
        "n_psi": 4,
        "n_delta_beta": 8,
        "interpolation_bounds": "clip",
        "angular_mode": "fixed",
        "n_processes": 1,
        "parallel_prepare": False,
        "parallel_compute": False,
    }
    values.update(overrides)
    return BruteForce3PCFConfig(**values)


class BruteForce3PCFTests(unittest.TestCase):
    def test_current_bispectrum2d_is_a_direct_source(self):
        first = BispectrumTerm2D(
            "first",
            (NumericExpression2D(
                lambda ell1, ell2, ell3: np.exp(
                    -(ell1**2 + ell2**2 + ell3**2) / 4000.0
                )
            ),),
        )
        second = BispectrumTerm2D(
            "second",
            (NumericExpression2D(
                lambda ell1, ell2, ell3: 0.1 * np.ones_like(ell1)
            ),),
        )
        bispectrum = Bispectrum2D.from_components(first, 2.0 * second)

        solver = BruteForceX3PCF(
            bispectrum,
            spin=(0, 0, 0),
            config=_small_config(),
        )
        result = solver.compute([0.03], [0.05], [0.4, 1.2])

        self.assertTrue(solver.prepared)
        self.assertEqual(result.value.shape, (1, 1, 2))
        self.assertTrue(np.all(np.isfinite(result.value)))
        self.assertEqual(result.epsilon, (1, 1, 1))
        self.assertEqual(result.sigma, (0, 0, 0))
        self.assertTrue(result.angular_converged)
        self.assertEqual(result.n_psi_used, 4)
        self.assertEqual(result.n_delta_beta_used, 8)

    def test_spin_component_uses_active_conventions(self):
        term = BispectrumTerm2D(
            "constant",
            (NumericExpression2D(
                lambda ell1, ell2, ell3: np.ones_like(ell1)
            ),),
        )
        solver = BruteForceX3PCF(
            Bispectrum2D.from_components(term),
            spin=(2, 2, 2),
            component=1,
            config=_small_config(),
        )

        self.assertEqual(solver.epsilon, (-1, 1, 1))
        self.assertEqual(solver.sigma, (-2, 2, 2))

    def test_source_state_update_rebuilds_the_radial_cache(self):
        term = BispectrumTerm2D(
            "source",
            (NumericExpression2D(
                lambda ell1, ell2, ell3: np.exp(-ell1 / 100.0)
            ),),
        )
        bispectrum = Bispectrum2D.from_components(term)
        solver = BruteForceX3PCF(
            bispectrum,
            spin=(0, 0, 0),
            config=_small_config(),
        ).prepare()
        first_cache = solver.radial_transform

        bispectrum._state_updated()
        solver.prepare()

        self.assertIsNot(solver.radial_transform, first_cache)
        self.assertEqual(solver._prepared_source_token, bispectrum.state_token)

    def test_invalid_fftlog_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "even integer"):
            _small_config(n_ell=31).validate()


if __name__ == "__main__":
    unittest.main()
