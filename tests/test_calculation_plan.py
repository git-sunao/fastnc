import io
import os
from pathlib import Path
import shutil
import sys
import unittest

import numpy as np

from fastnc.bispectrum import (
    Bispectrum2D,
    BispectrumTerm2D,
    NumericRepresentation2D,
    SemiAnalyticRepresentation2D,
    SlepianRepresentation2D,
)
from fastnc.threepcf import CalculationPlan, ThreePCF, ThreePCFConfig


def make_prediction(route="hybrid"):
    source = Bispectrum2D(
        (
            BispectrumTerm2D(
                "slepian-term",
                (NumericRepresentation2D(), SlepianRepresentation2D()),
            ),
            BispectrumTerm2D(
                "semi-term",
                (NumericRepresentation2D(), SemiAnalyticRepresentation2D()),
            ),
            BispectrumTerm2D(
                "numeric-term",
                (NumericRepresentation2D(),),
            ),
        )
    )
    return ThreePCF(
        ThreePCFConfig(
            ell_min=10.0,
            ell_max=100.0,
            n_ell=8,
            use_coupling_cache=False,
        ),
        source,
        theta=np.geomspace(0.01, 0.02, 2),
        phi=np.array([0.0]),
        route=route,
    )


class CalculationPlanTests(unittest.TestCase):
    def test_hybrid_plan_assigns_each_term_once_by_priority(self):
        plan = make_prediction().calculation_plan()

        self.assertIsInstance(plan, CalculationPlan)
        self.assertEqual(plan.term_names("slepian"), ("slepian-term",))
        self.assertEqual(plan.term_names("semi_analytic"), ("semi-term",))
        self.assertEqual(plan.term_names("numeric"), ("numeric-term",))
        self.assertEqual(len(plan.assignments), 3)

    def test_hybrid_execution_consumes_the_public_plan(self):
        prediction = make_prediction()
        plan = prediction.calculation_plan()
        slepian, semi_analytic, numeric = prediction._plan_hybrid_sources()

        self.assertIs(plan, prediction.calculation_plan())
        self.assertEqual(
            tuple(term.name for term in slepian.terms),
            plan.term_names("slepian"),
        )
        self.assertEqual(
            tuple(term.name for term in semi_analytic.terms),
            plan.term_names("semi_analytic"),
        )
        self.assertEqual(
            tuple(term.name for term in numeric.terms),
            plan.term_names("numeric"),
        )

    def test_text_and_dot_show_route_merging(self):
        plan = make_prediction().calculation_plan()

        text = plan.to_text()
        self.assertIn("slepian-term", text)
        self.assertIn("BispectrumMultipole -> HKernel -> ZetaK", text)
        self.assertIn("Slepian -> ZetaK", text)

        dot = plan.to_dot(expand_terms=True)
        self.assertIn("slepian-term", dot)
        self.assertIn("route_slepian -> zetak", dot)
        self.assertIn("route_semi_analytic -> multipole", dot)
        self.assertIn("route_numeric -> multipole", dot)

        compact = plan.to_dot(expand_terms=False)
        self.assertIn("slepian-term", compact)
        self.assertNotIn("term_0 -> route_slepian", compact)

    def test_inspect_prints_and_returns_plan(self):
        prediction = make_prediction()
        stream = io.StringIO()

        returned = prediction.inspect(file=stream)

        self.assertIs(returned, prediction.calculation_plan())
        self.assertIn("ThreePCF calculation plan", stream.getvalue())

    def test_explicit_route_rejects_missing_representation(self):
        with self.assertRaisesRegex(TypeError, "no supported representation"):
            make_prediction(route="slepian").calculation_plan()

    def test_route_change_invalidates_plan(self):
        prediction = make_prediction()
        original = prediction.calculation_plan()

        prediction.set_route("numeric")
        updated = prediction.calculation_plan()

        self.assertIsNot(original, updated)
        self.assertEqual(updated.term_names("numeric"), (
            "slepian-term",
            "semi-term",
            "numeric-term",
        ))

    def test_graph_finds_dot_beside_active_conda_python(self):
        environment_dot = Path(sys.executable).resolve().parent / "dot"
        try:
            import graphviz  # noqa: F401
        except ImportError:
            self.skipTest("optional graphviz package is not installed")
        if not environment_dot.is_file():
            self.skipTest("dot is not installed beside the active Python")

        previous_path = os.environ.get("PATH")
        os.environ["PATH"] = "/usr/bin:/bin"
        try:
            graph = make_prediction().calculation_plan().graph()
            self.assertTrue(graph.pipe(format="svg").startswith(b"<?xml"))
            self.assertEqual(
                Path(shutil.which("dot")).resolve(), environment_dot
            )
        finally:
            if previous_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous_path


if __name__ == "__main__":
    unittest.main()
