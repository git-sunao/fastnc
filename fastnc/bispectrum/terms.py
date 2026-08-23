"""Declarative physical terms for 3D bispectrum models.

This module intentionally contains no LOS projection, multipole construction,
or 3PCF machinery.  A term defines a physical contribution and may keep a
reference to shared model physics, while numerical route selection belongs to
the calculator layer.
"""
from __future__ import annotations

from typing import Any


class BispectrumTerm:
    """Base class for one physical contribution to a 3D bispectrum.

    Subclasses must implement :meth:`evaluate`.  Inputs follow the same
    convention as :class:`fastnc.bispectrum.base.Bispectrum3D`.

    Notes
    -----
    The term object does *not* select a numerical route and does not perform
    LOS projection or 3PCF transforms.  Specialized route capabilities are
    added by more specific term interfaces in later implementation phases.
    """

    def __call__(self, k1, k2, k3, z, **params):
        return self.evaluate(k1, k2, k3, z, **params)

    def evaluate(self, k1, k2, k3, z, **params):
        """Evaluate this physical contribution."""
        raise NotImplementedError


class BackendBispectrumTerm(BispectrumTerm):
    """Bispectrum term holding a reference to shared model physics.

    ``backend`` is stored by reference.  This class deliberately performs no
    copy and exposes no backend-mutating API: the owning bispectrum/backend is
    responsible for coherent state updates, while terms remain lightweight
    views of that state.
    """

    def __init__(self, backend: Any):
        if backend is None:
            raise ValueError("backend must be a shared model/backend object")
        self._backend = backend

    @property
    def backend(self):
        """Shared backend object referenced by this term."""
        return self._backend


class ModelBispectrumTerm(BispectrumTerm):
    """Bispectrum term holding a live reference to its owning model.

    This is appropriate for lightweight models whose physical state already
    lives directly on the model rather than in a separate backend object.
    The model reference is never copied, so model-level updates are observed
    immediately by all terms.
    """

    def __init__(self, model: Any):
        if model is None:
            raise ValueError("model must be the owning bispectrum object")
        self._model = model

    @property
    def model(self):
        """Owning bispectrum object referenced by this term."""
        return self._model

    def _merge_model_defaults(self, params):
        """Apply the owning model's persistent evaluation defaults."""
        merge = getattr(self.model, "_merge_default_kwargs", None)
        return merge(params) if merge is not None else dict(params)
