import unittest

import numpy as np

from fastnc.coupling import CouplingMatrix, basis_fourier_terms


class CouplingBasisTests(unittest.TestCase):
    def test_scalar_limit_has_expected_basis_couplings(self):
        psi = np.array([0.2, 0.6, 1.0])
        kwargs = {"use_cache": False}

        cosine = CouplingMatrix(0, 0, 0, basis="cosine", **kwargs)
        sine = CouplingMatrix(0, 0, 0, basis="sine", **kwargs)
        legendre = CouplingMatrix(0, 0, 0, basis="legendre", **kwargs)

        np.testing.assert_allclose(cosine(1, 1.0, psi), np.pi)
        np.testing.assert_allclose(sine(1, 1.0, psi), -1j * np.pi)
        np.testing.assert_allclose(legendre(1, 1.0, psi), np.pi)
        np.testing.assert_allclose(legendre(2, 0.0, psi), 0.5 * np.pi)

    def test_each_basis_is_a_finite_sum_of_fourier_primitives(self):
        psi = np.array([0.19, 0.47, 0.91, 1.31])
        sigma = (2, -2, 2)
        k = 1.0
        fourier = CouplingMatrix(*sigma, basis="fourier", use_cache=False)

        for basis, mode in (("cosine", 3), ("sine", 3), ("legendre", 4)):
            coupling = CouplingMatrix(
                *sigma,
                basis=basis,
                use_cache=False,
            )
            expected = sum(
                coefficient * fourier(fourier_mode, k, psi)
                for fourier_mode, coefficient in basis_fourier_terms(
                    mode, basis
                )
            )
            np.testing.assert_allclose(
                coupling(mode, k, psi),
                expected,
                rtol=1.0e-14,
                atol=1.0e-14,
            )

    def test_legendre_fourier_coefficients_reconstruct_polynomials(self):
        angle = np.linspace(0.0, 2.0 * np.pi, 101)
        for mode in range(7):
            reconstructed = sum(
                coefficient * np.exp(1j * fourier_mode * angle)
                for fourier_mode, coefficient in basis_fourier_terms(
                    mode, "legendre"
                )
            )
            expected = np.polynomial.legendre.Legendre.basis(mode)(
                np.cos(angle)
            )
            np.testing.assert_allclose(
                reconstructed.imag,
                0.0,
                atol=1.0e-14,
            )
            np.testing.assert_allclose(
                reconstructed.real,
                expected,
                rtol=1.0e-13,
                atol=1.0e-13,
            )


if __name__ == "__main__":
    unittest.main()
