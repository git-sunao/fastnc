"""Passive inspection objects for ThreePCF calculation plans."""
from __future__ import annotations

from dataclasses import dataclass
import html
import os
from pathlib import Path
import re
import shutil
import sys

from fastnc.bispectrum import (
    Bispectrum2D,
    NumericRepresentation2D,
    SemiAnalyticRepresentation2D,
    SlepianRepresentation2D,
)


_REPRESENTATIONS = {
    "numeric": NumericRepresentation2D,
    "semi_analytic": SemiAnalyticRepresentation2D,
    "slepian": SlepianRepresentation2D,
}

_ROUTE_LABELS = {
    "numeric": "Numeric",
    "semi_analytic": "Semi-analytic",
    "slepian": "Slepian",
}


@dataclass(frozen=True)
class TermRouteAssignment:
    """Route selected for one additive bispectrum term."""

    term_index: int
    term_name: str
    route: str
    representation: str
    available_representations: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CalculationPlan:
    """Immutable, calculation-free description of a ThreePCF execution DAG.

    The plan is the route planner's public inspection product. Rendering it
    never evaluates a bispectrum, multipole, coupling matrix, Hankel transform,
    or correlation function. ``graph()`` is the only method requiring the
    optional ``graphviz`` Python package and a working Graphviz ``dot`` binary.
    """

    requested_route: str
    assignments: tuple[TermRouteAssignment, ...]

    def assignments_for(self, route: str) -> tuple[TermRouteAssignment, ...]:
        """Return assignments belonging to one concrete route."""
        return tuple(item for item in self.assignments if item.route == route)

    @property
    def routes(self) -> tuple[str, ...]:
        """Return concrete routes in pipeline order."""
        return tuple(
            route
            for route in ("slepian", "semi_analytic", "numeric")
            if self.assignments_for(route)
        )

    def term_names(self, route: str) -> tuple[str, ...]:
        """Return term names assigned to one concrete route."""
        return tuple(item.term_name for item in self.assignments_for(route))

    def to_text(self, *, expand_terms: bool = True) -> str:
        """Render a dependency-free terminal representation of the plan."""
        lines = [f"ThreePCF calculation plan (route={self.requested_route})"]
        for route in self.routes:
            assignments = self.assignments_for(route)
            label = _ROUTE_LABELS[route]
            lines.append(f"+- {label} [{len(assignments)} term(s)]")
            if expand_terms:
                for index, assignment in enumerate(assignments):
                    branch = "`-" if index == len(assignments) - 1 else "+-"
                    lines.append(
                        f"|  {branch} {assignment.term_name} "
                        f"[{assignment.representation}]"
                    )
        if self.assignments_for("slepian"):
            lines.append("+- Slepian -> ZetaK")
        if self.assignments_for("semi_analytic") or self.assignments_for("numeric"):
            lines.append("+- Numeric/semi-analytic -> BispectrumMultipole -> HKernel -> ZetaK")
        lines.append("`- ZetaK -> Zeta")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    @staticmethod
    def _dot_quote(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'

    @staticmethod
    def _dot_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", value)

    def to_dot(self, *, expand_terms: bool = False) -> str:
        """Return Graphviz DOT source without importing optional packages."""
        colors = {
            "slepian": ("#DFF3E4", "#39804A"),
            "semi_analytic": ("#FFF0CC", "#A96800"),
            "numeric": ("#DCEBFA", "#3573A8"),
        }
        lines = [
            "digraph fastnc_plan {",
            '  graph [rankdir=LR, bgcolor="transparent", pad=0.25, nodesep=0.35, ranksep=0.65];',
            '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.07"];',
            '  edge [color="#6B7280", arrowsize=0.7];',
        ]
        for route in self.routes:
            fill, border = colors[route]
            route_id = f"route_{self._dot_id(route)}"
            assignments = self.assignments_for(route)
            label = f"{_ROUTE_LABELS[route]}\n{len(assignments)} term(s)"
            if not expand_terms:
                label += "\n\n" + "\n".join(
                    assignment.term_name for assignment in assignments
                )
            lines.append(
                f"  {route_id} [label={self._dot_quote(label)}, "
                f'fillcolor="{fill}", color="{border}", penwidth=1.4];'
            )
            if expand_terms:
                lines.append(f"  subgraph cluster_{self._dot_id(route)} {{")
                lines.append('    style="rounded,dashed"; color="#CBD5E1"; label="";')
                for assignment in assignments:
                    term_id = f"term_{assignment.term_index}"
                    lines.append(
                        f"    {term_id} [label={self._dot_quote(assignment.term_name)}, "
                        'fillcolor="#FFFFFF", color="#94A3B8"];'
                    )
                    lines.append(f"    {term_id} -> {route_id};")
                lines.append("  }")

        has_radial_pipeline = bool(
            self.assignments_for("semi_analytic") or self.assignments_for("numeric")
        )
        if has_radial_pipeline:
            lines.extend(
                [
                    '  multipole [label="BispectrumMultipole", fillcolor="#F3F4F6", color="#64748B"];',
                    '  hkernel [label="HKernel", fillcolor="#F3F4F6", color="#64748B"];',
                    "  multipole -> hkernel;",
                    "  hkernel -> zetak;",
                ]
            )
            for route in ("semi_analytic", "numeric"):
                if self.assignments_for(route):
                    lines.append(f"  route_{self._dot_id(route)} -> multipole;")
        if self.assignments_for("slepian"):
            lines.append("  route_slepian -> zetak;")
        lines.extend(
            [
                '  zetak [label="ZetaK", fillcolor="#F3F4F6", color="#475569", penwidth=1.4];',
                '  zeta [label="Zeta", fillcolor="#F3F4F6", color="#475569", penwidth=1.4];',
                "  zetak -> zeta;",
                "}",
            ]
        )
        return "\n".join(lines)

    def graph(self, *, expand_terms: bool = False):
        """Return a Graphviz object that renders as SVG in Jupyter.

        Raises:
            ImportError: If the optional ``graphviz`` Python package is absent.
        """
        try:
            from graphviz import Source
        except ImportError as error:
            raise ImportError(
                "calculation-graph rendering requires the optional graph extra; "
                "install fastnc[graph] and the Graphviz 'dot' executable"
            ) from error
        if shutil.which("dot") is None:
            environment_dot = Path(sys.executable).resolve().parent / "dot"
            if environment_dot.is_file():
                os.environ["PATH"] = os.pathsep.join(
                    (str(environment_dot.parent), os.environ.get("PATH", ""))
                )
            else:
                raise RuntimeError(
                    "calculation-graph rendering requires the Graphviz 'dot' "
                    "executable on PATH or beside the active Python executable"
                )
        return Source(self.to_dot(expand_terms=expand_terms), format="svg")

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(self.to_text())}</pre>"


def build_calculation_plan(source, requested_route: str) -> CalculationPlan:
    """Build the canonical term-wise route assignment without evaluating data."""
    if not isinstance(source, Bispectrum2D):
        raise TypeError("calculation planning requires a Bispectrum2D source")

    assignments = []
    unsupported = []
    for index, term in enumerate(source.terms):
        available = tuple(
            route
            for route, representation_type in _REPRESENTATIONS.items()
            if any(isinstance(rep, representation_type) for rep in term.representations)
        )
        if requested_route == "hybrid":
            selected = next(
                (route for route in ("slepian", "semi_analytic", "numeric") if route in available),
                None,
            )
            reason = "hybrid priority: Slepian -> semi-analytic -> numeric"
        else:
            selected = requested_route if requested_route in available else None
            reason = f"explicit {requested_route} route"
        if selected is None:
            unsupported.append((term.name, available))
            continue
        assignments.append(
            TermRouteAssignment(
                term_index=index,
                term_name=term.name,
                route=selected,
                representation=type(
                    term.get_representation(_REPRESENTATIONS[selected])
                ).__name__,
                available_representations=available,
                reason=reason,
            )
        )
    if unsupported:
        details = ", ".join(
            f"{name} (available={available or ('none',)})"
            for name, available in unsupported
        )
        raise TypeError(
            f"{requested_route} route found terms with no supported representation: {details}"
        )
    return CalculationPlan(requested_route, tuple(assignments))
