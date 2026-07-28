from __future__ import annotations

import numpy as np
from typing import Mapping
from dataclasses import dataclass

from .bias import QuadraticBiasBispectrumMultipole3D, TidalBiasBispectrumMultipole3D
from ..composite import CompositeSemiAnalyticBispectrumMultipole3D
from ..fftlog import _MutablePhysicalCallable
from .tree import TreeBispectrumMultipole3D

@dataclass(frozen=True)
class TracerBias:
    r"""Eulerian bias parameters for one deterministic LSS tracer.

    Each entry may be either a scalar or a callable ``value(z)``.  Different
    tracer labels may therefore carry independent, redshift-dependent bias
    functions while sharing the same underlying linear matter spectrum.
    """

    b1: object
    b2: object = 0.0
    bK2: object = 0.0


def _coerce_tracer_bias(name, value):
    """Return a :class:`TracerBias` from a dataclass or mapping input."""
    if isinstance(value, TracerBias):
        return value
    if isinstance(value, Mapping):
        unknown = set(value) - {"b1", "b2", "bK2", "bs2"}
        if unknown:
            raise ValueError(
                f"Unknown bias parameter(s) for tracer {name!r}: {sorted(unknown)}"
            )
        if "b1" not in value:
            raise ValueError(f"Tracer {name!r} requires a 'b1' entry")
        if "bK2" in value and "bs2" in value:
            raise ValueError(
                f"Tracer {name!r} specifies both 'bK2' and its alias 'bs2'"
            )
        return TracerBias(
            b1=value["b1"],
            b2=value.get("b2", 0.0),
            bK2=value.get("bK2", value.get("bs2", 0.0)),
        )
    raise TypeError(
        f"Bias definition for tracer {name!r} must be TracerBias or a mapping"
    )


def _is_matter_field(field):
    return str(field).lower() in {"m", "matter"}


class SPTMultiTracerBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    r"""Tree-level real-space SPT multipoles for an ordered multi-tracer triple.

    Parameters
    ----------
    field_order : sequence of str
        Tracer identity assigned to ``(k2, k3, k3)``.  Matter may be written as
        ``"m"`` or ``"matter"``.  Every other entry must be a key of
        ``tracer_biases``.  For example,
        ``("LOWZ", "CMASS", "matter")`` represents
        :math:`B_{g_{\rm LOWZ}g_{\rm CMASS}m}` with that vertex ordering.
    tracer_biases : mapping
        Mapping from tracer label to :class:`TracerBias`, or to a mapping with
        entries ``b1``, ``b2`` and ``bK2`` (``bs2`` is accepted as an alias).

    Notes
    -----
    The model is assembled from independently reusable components,

    .. math::
       B_{A_1A_2A_3}^{\rm tree}
       = C_F B_{mmm}^{\rm tree} + B_{b_2} + B_{K^2}.

    The pair coefficient for ``ij`` is the product of the two linear biases on
    legs ``i,j`` and the second-order bias of the remaining vertex.  Thus the
    ordering of ``field_order`` is physically meaningful.
    """

    def __init__(
        self,
        linear_power,
        k_grid,
        *,
        field_order,
        tracer_biases,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
        regularize_squeezed: bool = True,
    ):
        fields = tuple(str(field) for field in field_order)
        if len(fields) != 3:
            raise ValueError("field_order must contain exactly three vertex labels")

        raw_biases = dict(tracer_biases)
        for name in raw_biases:
            if _is_matter_field(name):
                raise ValueError(
                    f"{name!r} is reserved for the matter field and must not "
                    "appear in tracer_biases"
                )
        biases = {
            str(name): _coerce_tracer_bias(str(name), value)
            for name, value in raw_biases.items()
        }
        missing = sorted({field for field in fields if not _is_matter_field(field)} - set(biases))
        if missing:
            raise ValueError(
                "Missing tracer_biases entries for field_order label(s): "
                + ", ".join(repr(name) for name in missing)
            )

        def bias(index):
            field = fields[index]
            return None if _is_matter_field(field) else biases[field]

        def lam(index, z):
            tracer = bias(index)
            return 1.0 if tracer is None else _value_at_z(tracer.b1, z)

        def second_order(index, parameter, z):
            tracer = bias(index)
            if tracer is None:
                return 0.0
            return _value_at_z(getattr(tracer, parameter), z)

        cf = lambda z: lam(0, z) * lam(1, z) * lam(2, z)

        # Pair ij means that the remaining vertex is evaluated to second order.
        c12_b2 = lambda z: lam(0, z) * lam(1, z) * second_order(2, "b2", z)
        c23_b2 = lambda z: lam(1, z) * lam(2, z) * second_order(0, "b2", z)
        c31_b2 = lambda z: lam(2, z) * lam(0, z) * second_order(1, "b2", z)

        c12_k2 = lambda z: 2.0 * lam(0, z) * lam(1, z) * second_order(2, "bK2", z)
        c23_k2 = lambda z: 2.0 * lam(1, z) * lam(2, z) * second_order(0, "bK2", z)
        c31_k2 = lambda z: 2.0 * lam(2, z) * lam(0, z) * second_order(1, "bK2", z)

        linear_power = (linear_power if isinstance(linear_power, _MutablePhysicalCallable)
                        else _MutablePhysicalCallable(linear_power))

        tree = TreeBispectrumMultipole3D(
            linear_power,
            k_grid,
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
            regularize_squeezed=regularize_squeezed,
        )
        quadratic = QuadraticBiasBispectrumMultipole3D(
            linear_power,
            k_grid,
            pair_coefficients=(c12_b2, c23_b2, c31_b2),
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
        )
        tidal = TidalBiasBispectrumMultipole3D(
            linear_power,
            k_grid,
            pair_coefficients=(c12_k2, c23_k2, c31_k2),
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
        )
        # All three pieces use the same linear-power FFTLog expansion.  Rebind
        # separable terms to the tree component so identity-keyed coefficient
        # and angular-kernel caches are shared by the flattened SPT model.
        shared_component = tree._linear_power_component

        def share_component(term):
            if isinstance(term, SeparableMultipoleTerm):
                return replace(term, component=shared_component)
            return term

        terms = [term.scaled_by(cf) for term in tree.terms]
        terms.extend(share_component(term) for term in quadratic.terms)
        terms.extend(share_component(term) for term in tidal.terms)
        super().__init__(terms, angular_kernel_config=angular_kernel_config)

        self.field_order = fields
        self._physical_field_order = fields
        self.tracer_biases = biases
        self.tree_coefficient = cf
        self.tree_matter = tree
        self.quadratic_bias = quadratic
        self.tidal_bias = tidal
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self._linear_power_component = shared_component

    def update_physics(self, *, linear_power=None, tracer_biases=None):
        """Atomically update spectrum and/or tracer-bias state.

        Bias-only updates retain every cache entry.  A spectrum update removes
        only FFTLog coefficients for the shared power component; the expensive
        angular-kernel table is preserved.
        """
        if linear_power is None and tracer_biases is None:
            raise ValueError("at least one physical-state change is required")

        if tracer_biases is not None:
            raw = dict(tracer_biases)
            for name in raw:
                if _is_matter_field(name):
                    raise ValueError(
                        f"{name!r} is reserved for the matter field and must not "
                        "appear in tracer_biases"
                    )
            updated = {
                str(name): _coerce_tracer_bias(str(name), value)
                for name, value in raw.items()
            }
            required = {
                field for field in self._physical_field_order if not _is_matter_field(field)
            }
            missing = sorted(required - set(updated))
            if missing:
                raise ValueError(
                    "Missing tracer_biases entries for field_order label(s): "
                    + ", ".join(repr(name) for name in missing)
                )
            self.tracer_biases.clear()
            self.tracer_biases.update(updated)

        if linear_power is not None:
            self.linear_power.update(linear_power)
            self.invalidate_components((self._linear_power_component,))
        return self


class SPTGalaxyBispectrumMultipole3D(SPTMultiTracerBispectrumMultipole3D):
    r"""Backward-compatible single-galaxy-tracer SPT model.

    ``field_order`` uses the legacy labels ``"g"`` and ``"m"``.  Internally
    this is a thin wrapper around :class:`SPTMultiTracerBispectrumMultipole3D`
    with one tracer named ``"galaxy"``.
    """

    def __init__(self, linear_power, k_grid, *, field_order=("g", "g", "g"),
                 b1=1.0, b2=0.0, bK2=0.0,
                 fftlog_config: PowerLawFFTLogConfig | None = None,
                 angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
                 regularize_squeezed: bool = True):
        legacy_fields = tuple(str(field).lower() for field in field_order)
        if len(legacy_fields) != 3 or any(field not in {"g", "m"} for field in legacy_fields):
            raise ValueError(
                "field_order must be a length-three tuple containing only 'g' and 'm'"
            )
        fields = tuple("galaxy" if field == "g" else "matter" for field in legacy_fields)
        super().__init__(
            linear_power,
            k_grid,
            field_order=fields,
            tracer_biases={"galaxy": TracerBias(b1=b1, b2=b2, bK2=bK2)},
            fftlog_config=fftlog_config,
            angular_kernel_config=angular_kernel_config,
            regularize_squeezed=regularize_squeezed,
        )
        # Preserve the public attributes and legacy field labels.
        self.field_order = legacy_fields
        self.b1 = b1
        self.b2 = b2
        self.bK2 = bK2

    def update_physics(
        self,
        *,
        linear_power=None,
        b1=None,
        b2=None,
        bK2=None,
    ):
        current = self.tracer_biases["galaxy"]
        bias_changed = any(value is not None for value in (b1, b2, bK2))
        biases = None
        if bias_changed:
            updated = TracerBias(
                b1=current.b1 if b1 is None else b1,
                b2=current.b2 if b2 is None else b2,
                bK2=current.bK2 if bK2 is None else bK2,
            )
            biases = {"galaxy": updated}
        super().update_physics(
            linear_power=linear_power,
            tracer_biases=biases,
        )
        updated = self.tracer_biases["galaxy"]
        self.b1, self.b2, self.bK2 = updated.b1, updated.b2, updated.bK2
        return self
