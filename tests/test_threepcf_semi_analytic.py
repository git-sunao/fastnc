import unittest

import numpy as np

from fastnc.bispectrum import (
    BiHalofitBispectrum3D,
    BiHalofitFixedShapeOneHaloBispectrum3D,
    Bispectrum3D,
    BispectrumTerm3D,
    NumericExpression3D,
    SemiAnalyticExpression3D,
    SemiAnalyticRadialExpression3D,
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


def direct_numeric_multipole_los(source, projector, modes, ell2, ell3, *, angular_nodes):
    """Independent benchmark using the source's numeric 3D representation."""
    ell2, ell3 = np.broadcast_arrays(
        np.asarray(ell2, dtype=float), np.asarray(ell3, dtype=float)
    )
    phi = (np.arange(angular_nodes) + 0.5) * (2.0 * np.pi / angular_nodes)
    angular_shape = (angular_nodes,) + (1,) * ell2.ndim
    cosine = np.cos(phi).reshape(angular_shape)
    phase = np.exp(-1j * np.asarray(modes)[:, None] * phi[None, :])
    phase = phase.reshape((len(modes), angular_nodes) + (1,) * ell2.ndim)

    los_samples = []
    for z, chi in zip(projector.z, projector.chi):
        k2 = ell2 / chi
        k3 = ell3 / chi
        k1 = np.sqrt(k2**2 + k3**2 + 2.0 * k2 * k3 * cosine)
        values = source.evaluate_numeric(k1, k2, k3, float(z))
        los_samples.append(np.mean(phase * values[None, ...], axis=1))
    return projector.integrate_coefficients(
        np.stack(los_samples), axis=0
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

    def test_radial_expression_matches_direct_numeric_angular_integral(self):
        def make_expression(order):
            return SemiAnalyticRadialExpression3D(
                u_evaluator=lambda ratio2, ratio3: 1.0 + 0.2 * ratio2,
                v_evaluator=lambda k2, k3, z: (1.0 + z) * k2 * k3,
                w_evaluator=lambda k1, z: (
                    np.exp(-0.3 * k1) * (1.0 + 0.1 * z)
                ),
                power=1.25,
                angular_order=order,
            )

        expressions = (make_expression(-2), make_expression(2))
        terms = tuple(
            BispectrumTerm3D(
                f"toy:radial:{expression.angular_order:+d}",
                representations=(
                    expression,
                    NumericExpression3D(
                        lambda k1, k2, k3, z, expression=expression: expression(
                            k1, k2, k3, z
                        )
                    ),
                ),
            )
            for expression in expressions
        )
        source = Bispectrum3D(tuple(1.7 * term for term in terms))
        projector = LOSProjector(
            z=np.array([0.1, 0.4, 0.9]),
            chi=np.array([200.0, 600.0, 1300.0]),
            prefactor=1.0,
        )
        projected = projector.project(source)
        nodes = 2048
        modes = np.array([-3, -2, 0, 2])
        ell2 = np.array([30.0, 80.0])
        ell3 = np.array([55.0, 140.0])
        actual = SemiAnalyticCalculator(
            SemiAnalyticConfig(angular_nodes=nodes)
        ).evaluate(projected, modes, ell2=ell2, ell3=ell3)
        expected = direct_numeric_multipole_los(
            source, projector, modes, ell2, ell3, angular_nodes=nodes
        )
        scale = np.max(np.abs(expected), axis=1, keepdims=True)
        self.assertLess(np.max(np.abs(actual - expected) / scale), 3.0e-6)

    def test_bihalofit_pair23_projected_multipoles_match_numeric_benchmark(self):
        model = BiHalofitBispectrum3D.simple_debug(
            k=np.logspace(-4, 2, 256),
            z=np.linspace(0.0, 1.0, 32),
        )
        source = model.select_terms(*(
            term.name for term in model.terms
            if term.name.startswith("bihalofit:Bh3:23:")
        ))
        projector = LOSProjector.delta_like(z=0.45, chi=700.0)
        projected = projector.project(source)
        nodes = 2048
        modes = np.array([-2, -1, 0, 1, 2, 4])
        ell2 = np.array([35.0, 90.0, 220.0])
        ell3 = np.array([60.0, 150.0, 310.0])
        actual = SemiAnalyticCalculator(
            SemiAnalyticConfig(angular_nodes=nodes)
        ).evaluate(projected, modes, ell2=ell2, ell3=ell3)
        expected = direct_numeric_multipole_los(
            source, projector, modes, ell2, ell3, angular_nodes=nodes
        )
        scale = np.max(np.abs(expected), axis=1, keepdims=True)
        self.assertLess(np.max(np.abs(actual - expected) / scale), 3.0e-6)

    def test_bihalofit_hybrid_plan_assigns_pair23_to_semi_analytic(self):
        model = BiHalofitBispectrum3D.simple_debug(
            k=np.logspace(-4, 2, 128),
            z=np.linspace(0.0, 1.0, 16),
        )
        projected = LOSProjector.delta_like(z=0.4, chi=700.0).project(model)
        prediction = ThreePCF(
            ThreePCFConfig(
                Lmax=2,
                kmax=0,
                ell_min=20.0,
                ell_max=200.0,
                n_ell=16,
                use_coupling_cache=False,
                semi_analytic=SemiAnalyticConfig(angular_nodes=64),
            ),
            projected,
            theta=np.geomspace(0.01, 0.03, 3),
            phi=np.array([0.0]),
            route="hybrid",
        )
        slepian, semi_analytic, numeric = prediction._plan_hybrid_sources()
        self.assertIsNone(slepian)
        self.assertEqual(len(semi_analytic.terms), 8)
        self.assertTrue(all(
            term.name.startswith("bihalofit:Bh3:23:")
            for term in semi_analytic.terms
        ))
        self.assertEqual(len(numeric.terms), 17)
        self.assertEqual(
            {term.name for term in projected.terms},
            {term.name for term in semi_analytic.terms}
            | {term.name for term in numeric.terms},
        )

    def test_radial_mellin_evaluation_reuses_angular_kernel_cache(self):
        expression = SemiAnalyticRadialExpression3D(
            u_evaluator=lambda ratio2, ratio3: 1.0,
            v_evaluator=lambda k2, k3, z: 1.0,
            w_evaluator=lambda k1, z: np.exp(-np.asarray(k1)),
        )
        source = LOSProjector.delta_like(z=0.3, chi=500.0).project(
            Bispectrum3D((BispectrumTerm3D("cache:radial", (expression,)),))
        )
        calculator = SemiAnalyticCalculator(
            SemiAnalyticConfig(
                angular_nodes=64,
                mellin_nodes=32,
                mellin_window_width=0.25,
            )
        )
        arguments = dict(
            mode=np.array([0, 1]),
            ell2=np.array([30.0, 70.0]),
            ell3=np.array([50.0, 110.0]),
        )
        calculator.evaluate(source, **arguments)
        first = calculator.cache_info()
        calculator.evaluate(source, **arguments)
        self.assertGreater(first["angular_kernels"], 0)
        self.assertEqual(calculator.cache_info(), first)

    def test_fixed_shape_bh1_finite_width_los_matches_numeric_multipoles(self):
        source = BiHalofitFixedShapeOneHaloBispectrum3D.simple_debug(
            fiducial_r1=0.8,
            fiducial_r2=0.5,
            k=np.logspace(-4, 2, 256),
            z=np.linspace(0.0, 1.0, 32),
        )
        projector = LOSProjector(
            z=np.linspace(0.1, 0.9, 8),
            chi=np.linspace(300.0, 1500.0, 8),
            prefactor=lambda z, chi: (1.0 + z) / chi**4,
        )
        projected = projector.project(source)
        modes = np.array([0, 1, 2, 4])
        ell2 = np.array([40.0, 90.0, 180.0])
        ell3 = np.array([65.0, 140.0, 260.0])
        calculator = SemiAnalyticCalculator(
            SemiAnalyticConfig(angular_nodes=512, mellin_nodes=1024)
        )
        actual = calculator.evaluate(
            projected, modes, ell2=ell2, ell3=ell3
        )
        expected = direct_numeric_multipole_los(
            source,
            projector,
            modes,
            ell2,
            ell3,
            angular_nodes=4096,
        )
        scale = np.max(np.abs(expected), axis=1, keepdims=True)
        self.assertLess(np.max(np.abs(actual - expected) / scale), 2.0e-5)


if __name__ == "__main__":
    unittest.main()
