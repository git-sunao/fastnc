"""Stateless strategies carried by projected 2D representations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numeric_los import evaluate_numeric_los, integrate_numeric_los


@dataclass(frozen=True)
class NumericLOSProjectionRule:
    """Evaluate a weighted numeric 3D term through an LOS projector."""

    name: str = "numeric_los"

    def evaluate(self, representation, ell1, ell2, ell3, **params):
        projector = representation.projector
        if projector.is_delta_like:
            return self._evaluate_at_point(
                representation.evaluate_source,
                projector,
                ell1,
                ell2,
                ell3,
                **params,
            )
        sampled = evaluate_numeric_los(
            representation.evaluate_source,
            ell1,
            ell2,
            ell3,
            z=projector.z,
            chi=projector.chi,
            shift=projector.shift,
            **params,
        )
        return integrate_numeric_los(
            sampled,
            projector.chi,
            weight=projector.weight(representation.sample_combination),
        )

    @staticmethod
    def _evaluate_at_point(evaluator, projector, ell1, ell2, ell3, **params):
        scalar = all(np.ndim(value) == 0 for value in (ell1, ell2, ell3))
        ell1, ell2, ell3 = np.broadcast_arrays(
            np.asarray(ell1, dtype=float),
            np.asarray(ell2, dtype=float),
            np.asarray(ell3, dtype=float),
        )
        output_shape = ell1.shape
        evaluator_shape = (1,) if scalar else output_shape
        values = np.asarray(
            evaluator(
                np.reshape(
                    (ell1 + projector.shift) / projector.chi[0],
                    evaluator_shape,
                ),
                np.reshape(
                    (ell2 + projector.shift) / projector.chi[0],
                    evaluator_shape,
                ),
                np.reshape(
                    (ell3 + projector.shift) / projector.chi[0],
                    evaluator_shape,
                ),
                projector.z[0],
                **params,
            )
        )
        try:
            values = np.broadcast_to(values, evaluator_shape)
        except ValueError as exc:
            raise ValueError(
                "evaluator output must broadcast to the angular shape"
            ) from exc
        result = values.reshape(output_shape)
        return result.item() if scalar else result


NUMERIC_LOS_PROJECTION = NumericLOSProjectionRule()
