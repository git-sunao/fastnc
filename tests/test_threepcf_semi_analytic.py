import unittest

import numpy as np

from fastnc.bispectrum import (
    Bispectrum3D,
    BispectrumTerm3D,
    SemiAnalyticExpression3D,
)
from fastnc.projection import LOSProjector, ProjectedSemiAnalyticRepresentation2D
from fastnc.threepcf import (
    SemiAnalyticCalculator,
    SemiAnalyticConfig,
    ThreePCF,
    ThreePCFConfig,
)


def make_toy_expression():
    return SemiAnalyticExpression3D(
        exponents=np.array([0.0]),
        coefficient_evaluator=lambda z: np.asarray(z)[..., None] + 1.0,
        u_evaluator=lambda ratio2, ratio3: ratio2,
        v_evaluator=lambda k2, k3, z: k2 * k3,
        power=2.0,
    )


def make_complex_mellin_expression(amplitude=1.0):
    exponents = np.array([-0.4 + 5.0j, -0.4 - 5.0j])

    def coefficients(z):
        envelope = amplitude * (1.0 + 0.25 * np.asarray(z))
        positive = envelope * (0.7 + 0.2j)
        return np.stack((positive, np.conjugate(positive)), axis=-1)

    return SemiAnalyticExpression3D(
        exponents=exponents,
        coefficient_evaluator=coefficients,
        u_evaluator=lambda ratio2, ratio3: 1.0 + 0.3 * ratio2 * ratio3,
        v_evaluator=lambda k2, k3, z: (
            (1.0 + np.asarray(z)) * np.exp(-0.02 * (k2 + k3))
        ),
        power=1.5,
    )


def nodewise_multipole_los(projected, modes, ell2, ell3, *, angular_nodes):
    """Transparent benchmark: angular multipoles first, LOS integral second."""
    representation = projected.terms[0].representations[0]
    expression = representation.source_representation
    projector = representation.projector
    weighted = representation.source_term
    ell2, ell3 = np.broadcast_arrays(ell2, ell3)
    scale = np.sqrt(ell2**2 + ell3**2)
    phi = (np.arange(angular_nodes) + 0.5) * (2.0 * np.pi / angular_nodes)
    cosine = np.cos(phi).reshape((-1,) + (1,) * ell2.ndim)
    phase = np.exp(-1j * np.asarray(modes)[:, None] * phi[None, :])
    phase = phase.reshape((len(modes), angular_nodes) + (1,) * ell2.ndim)

    los_multipoles = []
    for z, chi in zip(projector.z, projector.chi):
        k2 = ell2 / chi
        k3 = ell3 / chi
        k = scale / chi
        k1 = np.sqrt(k2**2 + k3**2 + 2.0 * k2 * k3 * cosine)
        w = np.sum(
            expression.coefficients(float(z)).reshape(
                (-1,) + (1,) * k1.ndim
            ) * np.power(k1[None, ...].astype(complex),
                         expression.exponents.reshape((-1,) + (1,) * k1.ndim)),
            axis=0,
        )
        bispectrum = (
            expression.evaluate_u(k2 / k, k3 / k)[None, ...]
            * expression.evaluate_v(k2, k3, float(z))[None, ...]
            * np.power(k1 / k, expression.power)
            * w
        )
        coefficient = weighted.coefficient
        coefficient = coefficient(float(z)) if callable(coefficient) else coefficient
        los_multipoles.append(
            coefficient * np.mean(phase * bispectrum[None, ...], axis=1)
        )
    return projector.integrate_coefficients(
        np.stack(los_multipoles),
        axis=0,
        sample_combination=representation.sample_combination,
    )


class SemiAnalyticCalculatorTests(unittest.TestCase):
    def make_projected(self, projector):
        term = BispectrumTerm3D(
            "toy:semi-analytic", representations=(make_toy_expression(),)
        )
        return projector.project(Bispectrum3D((2.0 * term,)))

    def test_projection_is_deferred_to_calculator(self):
        projected = self.make_projected(
            LOSProjector.delta_like(z=0.5, chi=10.0)
        )
        representation = projected.terms[0].representations[0]
        self.assertIsInstance(
            representation, ProjectedSemiAnalyticRepresentation2D
        )
        self.assertFalse(hasattr(representation, "evaluate"))

    def test_finite_width_los_matches_analytic_fourier_modes(self):
        z = np.array([0.0, 0.5, 1.0])
        chi = np.array([10.0, 20.0, 40.0])
        projected = self.make_projected(
            LOSProjector(z=z, chi=chi, prefactor=1.0)
        )
        calculator = SemiAnalyticCalculator(
            SemiAnalyticConfig(angular_nodes=256)
        )
        ell2 = np.array([2.0, 3.0])
        ell3 = np.array([4.0, 1.5])
        modes = np.array([0, 1, 2, -1])
        actual = calculator.evaluate(
            projected, modes, ell2=ell2, ell3=ell3
        )

        scale2 = ell2**2 + ell3**2
        ratio = 2.0 * ell2 * ell3 / scale2
        u = ell2 / np.sqrt(scale2)
        los = np.trapezoid(
            2.0 * (1.0 + z[:, None])
            * ell2[None, :] * ell3[None, :] / chi[:, None] ** 2,
            x=chi,
            axis=0,
        )
        expected = np.stack(
            [u * los, u * los * ratio / 2.0, np.zeros_like(los),
             u * los * ratio / 2.0]
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_threepcf_owns_semi_analytic_calculator(self):
        projected = self.make_projected(
            LOSProjector.delta_like(z=0.25, chi=10.0)
        )
        config = ThreePCFConfig(
            basis="fourier",
            Lmax=2,
            kmax=0,
            ell_min=1.0,
            ell_max=100.0,
            n_ell=16,
            use_coupling_cache=False,
            semi_analytic=SemiAnalyticConfig(angular_nodes=128),
        )
        prediction = ThreePCF(
            config,
            projected,
            theta=np.geomspace(0.01, 0.1, 4),
            phi=np.linspace(0.0, np.pi, 5),
            route="semi_analytic",
        )
        multipoles = prediction.multipoles()
        self.assertIsInstance(multipoles.calculator, SemiAnalyticCalculator)
        values = multipoles.evaluate(0, np.array([2.0]), np.array([3.0]))
        self.assertEqual(values.shape, (1,))
        self.assertTrue(np.all(np.isfinite(values)))

        zetak = prediction.zetak()
        self.assertTrue(zetak.keys)
        self.assertTrue(np.all(np.isfinite(zetak.values)))

    def test_complex_mellin_los_matches_nodewise_multipole_projection(self):
        term = BispectrumTerm3D(
            "toy:complex-mellin",
            representations=(make_complex_mellin_expression(),),
        )
        source = Bispectrum3D(((lambda z: 1.0 + 0.1 * z) * term,))
        projector = LOSProjector(
            z=np.array([0.1, 0.4, 0.8, 1.2]),
            chi=np.array([300.0, 700.0, 1200.0, 1800.0]),
            prefactor=lambda z, chi: (1.0 + z) / chi**4,
        )
        projected = projector.project(source)
        angular_nodes = 4096
        calculator = SemiAnalyticCalculator(
            SemiAnalyticConfig(angular_nodes=angular_nodes)
        )
        ell2, ell3 = np.meshgrid(
            np.array([30.0, 90.0]), np.array([45.0, 120.0, 240.0]),
            indexing="ij",
        )
        modes = np.array([-3, -1, 0, 2])
        actual = calculator.evaluate(
            projected, modes, ell2=ell2, ell3=ell3
        )
        expected = nodewise_multipole_los(
            projected, modes, ell2, ell3, angular_nodes=angular_nodes
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=1.0e-25)

    def test_universal_kernel_cache_survives_coefficient_change(self):
        projector = LOSProjector.delta_like(z=0.5, chi=900.0)
        ell2 = np.array([40.0, 80.0])
        ell3 = np.array([70.0, 130.0])
        modes = np.array([-1, 0, 2])
        calculator = SemiAnalyticCalculator(
            SemiAnalyticConfig(angular_nodes=256)
        )

        def projected(amplitude):
            term = BispectrumTerm3D(
                f"toy:amplitude-{amplitude}",
                representations=(make_complex_mellin_expression(amplitude),),
            )
            return projector.project(Bispectrum3D((term,)))

        first = calculator.evaluate(
            projected(1.0), modes, ell2=ell2, ell3=ell3
        )
        cache_size = len(calculator._kernel_cache)
        second = calculator.evaluate(
            projected(2.0), modes, ell2=ell2, ell3=ell3
        )
        self.assertEqual(len(calculator._kernel_cache), cache_size)
        np.testing.assert_allclose(second, 2.0 * first)


if __name__ == "__main__":
    unittest.main()
