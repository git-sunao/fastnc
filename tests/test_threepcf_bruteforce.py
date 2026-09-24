import unittest

import numpy as np

from fastnc.bispectrum import (
    Bispectrum2D,
    BispectrumTerm2D,
    NumericExpression2D,
)
from fastnc.multipole import NumericMultipoleConfig
from fastnc.threepcf import (
    BruteForce3PCFConfig,
    BruteForceX3PCF,
    ThreePCF,
    ThreePCFConfig,
)


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


def _validation_brute_config(**overrides):
    values = {
        "ell_min": 0.01,
        "ell_max": 3000.0,
        "n_ell": 128,
        "radial_cache_padding_factor": 16.0,
        "n_psi": 16,
        "n_delta_beta": 32,
        "interpolation_bounds": "raise",
        "angular_mode": "fixed",
        "n_processes": 1,
        "parallel_prepare": False,
        "parallel_compute": False,
    }
    values.update(overrides)
    return BruteForce3PCFConfig(**values)


def _validation_threepcf_config(*, Lmax, kmax):
    return ThreePCFConfig(
        basis="fourier",
        Lmax=Lmax,
        kmax=kmax,
        ell_min=0.01,
        ell_max=3000.0,
        n_ell=128,
        multipole=NumericMultipoleConfig(
            n_angle=129,
            delta_beta_min=0.0,
            delta_beta_max=np.pi,
        ),
        use_coupling_cache=False,
    )


class BruteForce3PCFTests(unittest.TestCase):
    def test_scalar_separable_model_matches_analytic_hankel_product(self):
        scale = 300.0

        def radial(ell):
            return np.exp(-0.5 * (ell / scale) ** 2)

        def source(ell1, ell2, ell3):
            return radial(ell2) * radial(ell3)

        theta = np.geomspace(0.001, 0.006, 4)
        phi = np.array([0.3, 1.0])
        radial_transform = (
            scale**2
            / (2.0 * np.pi)
            * np.exp(-0.5 * (scale * theta) ** 2)
        )
        expected = radial_transform[:, None] * radial_transform[None, :]

        numeric = ThreePCF(
            _validation_threepcf_config(Lmax=0, kmax=0.0),
            source,
            theta,
            phi,
        ).zeta().values[0, :, :, 0].real
        brute = BruteForceX3PCF(
            source,
            spin=(0, 0, 0),
            config=_validation_brute_config(),
        ).compute(theta, theta, phi).value.real

        np.testing.assert_allclose(numeric, expected, rtol=0.02, atol=1.0)
        np.testing.assert_allclose(brute[:, :, 0], expected, rtol=0.03, atol=1.0)
        np.testing.assert_allclose(
            brute[:, :, 0], numeric, rtol=0.01, atol=1.0
        )
        self.assertLess(
            np.max(np.ptp(brute, axis=2)) / np.max(expected),
            2.0e-4,
        )

    def test_finite_angular_modes_match_numeric_route(self):
        scale = 300.0

        def source(ell1, ell2, ell3):
            mu = (ell1**2 - ell2**2 - ell3**2) / (2.0 * ell2 * ell3)
            radial = np.exp(
                -0.5 * (ell2 / scale) ** 2
                -0.5 * (ell3 / scale) ** 2
            )
            return radial * (1.0 + 0.2 * (2.0 * mu**2 - 1.0))

        theta = np.geomspace(0.001, 0.006, 4)
        phi = np.array([0.3, 1.0, 2.0])
        numeric = ThreePCF(
            _validation_threepcf_config(Lmax=2, kmax=2.0),
            source,
            theta,
            phi,
        ).zeta().values[0]
        brute = BruteForceX3PCF(
            source,
            spin=(0, 0, 0),
            config=_validation_brute_config(n_psi=24, n_delta_beta=64),
        ).compute(theta, theta, phi).value

        scale_value = max(np.max(np.abs(numeric)), np.max(np.abs(brute)))
        self.assertLess(np.max(np.abs(numeric - brute)) / scale_value, 0.01)

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
