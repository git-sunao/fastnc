"""Route orchestration for hybrid three-point correlation calculations.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..bispectrum.base import Bispectrum3D
from .config import ThreePCFConfig
from .generic_calculator import GenericThreePCFCalculator
from .slepian.calculator import SlepianThreePCFCalculator
from .zetak_grid import ZetaKGrid, ZetaKMode


class _TermCollectionBispectrum3D(Bispectrum3D):
    """Read-only 3D bispectrum view equal to the sum of a term collection.

    This adapter belongs to the calculator layer: it changes only the
    representation consumed by the generic numerical route and does not own or
    copy any model physics.  Individual terms retain their live references to
    the owning model/shared backend.
    """

    def __init__(self, model: Bispectrum3D, terms):
        self.model = model
        self.terms = tuple(terms)
        self.support = model.support

    def evaluate(self, k1, k2, k3, z, **params):
        if not self.terms:
            raise RuntimeError("term collection must not be empty")
        out = self.terms[0].evaluate(k1, k2, k3, z, **params)
        for term in self.terms[1:]:
            out = out + term.evaluate(k1, k2, k3, z, **params)
        return out


def _normalize_sample_combinations(sample_combinations):
    if sample_combinations is None:
        return (None,)
    # A single combination such as ("G1", "G1", "G1") is also accepted.
    if isinstance(sample_combinations, tuple) and len(sample_combinations) == 3:
        if not any(isinstance(x, (tuple, list)) for x in sample_combinations):
            return (tuple(sample_combinations),)
    combos = tuple(sample_combinations)
    if len(combos) == 0:
        return (None,)
    return tuple(None if c is None else tuple(c) for c in combos)




def _supports_slepian_los(model):
    kind = getattr(model, "slepian_los_kind", None)
    if kind == "factorized-growth":
        return bool(getattr(model, "has_factorized_growth", False))
    return kind == "general-coefficient"


class HybridThreePCFCalculator:
    """Top-level route orchestrator for a 3PCF calculation.
    """

    def __init__(
        self,
        *,
        config: ThreePCFConfig | None = None,
        bmultipole=None,
        bispectrum2d=None,
        bispectrum3d=None,
        projector=None,
        sample_combinations=None,
        coupling_kwargs: dict | None = None,
        multipole_config=None,
        multipole_basis: str = "fourier-even",
        regulator=None,
        multipole_kwargs: dict | None = None,
    ):
        provided = sum(x is not None for x in (bmultipole, bispectrum2d, bispectrum3d))
        if provided != 1:
            raise ValueError("provide exactly one of bmultipole, bispectrum2d, or bispectrum3d")
        if bispectrum3d is not None and projector is None:
            raise ValueError("projector is required for a 3D bispectrum input")
        if bispectrum3d is None and projector is not None:
            raise ValueError("projector is only valid with a 3D bispectrum input")

        self.config = config or ThreePCFConfig()
        self.bmultipole = bmultipole
        self.bispectrum2d = bispectrum2d
        self.bispectrum3d = bispectrum3d
        self.projector = projector
        self.sample_combinations = _normalize_sample_combinations(sample_combinations)
        self.coupling_kwargs = coupling_kwargs
        self.multipole_config = multipole_config
        self.multipole_basis = multipole_basis
        self.regulator = regulator
        self.multipole_kwargs = dict(multipole_kwargs or {})

        if bispectrum3d is None and self.sample_combinations != (None,):
            raise ValueError("sample combinations are only valid with a 3D bispectrum input")
        self.generic_terms = ()
        self.slepian_terms = ()
        self.analytic_terms = ()
        self.generic: GenericThreePCFCalculator | None = None
        self.generic_by_sample = {}
        self.slepian = None
        self.analytic = None
        self._combined_ZKgrid = None
        self._combined_ZKgrids = None
        self._combined_Zgrid = None
        self._combined_Zgrids = None
        self._prepare_children()

    def _default_multipole_config(self):
        from ..bispectrum import BispectrumMultipole2DConfig

        return BispectrumMultipole2DConfig(
            mode_max=int(self.config.Lmax),
            ell_min=float(self.config.ell_min),
            ell_max=float(self.config.ell_max),
            n_ell=int(self.config.n_ell),
        )

    def _make_multipole(self, bispectrum2d):
        if hasattr(bispectrum2d, "basis") and callable(bispectrum2d):
            return bispectrum2d
        if not hasattr(bispectrum2d, "multipole"):
            raise TypeError(
                "generic input must be either a callable multipole object with a basis "
                "attribute or an object exposing multipole(config=..., basis=...)."
            )
        mp_config = self.multipole_config or self._default_multipole_config()
        return bispectrum2d.multipole(
            config=mp_config,
            basis=self.multipole_basis,
            regulator=self.regulator,
            **self.multipole_kwargs,
        )

    def _prepare_3d_generic_model(self):
        model = self.bispectrum3d
        generic = tuple(model.generic_terms())
        slepian = tuple(model.slepian_terms())
        analytic = tuple(model.analytic_terms())
        self.generic_terms = generic
        self.slepian_terms = slepian
        self.analytic_terms = analytic

        mode = self.config.slepian.mode
        if analytic:
            raise RuntimeError("analytic_terms are not operational")
        if mode == "required" and not slepian:
            raise RuntimeError("SlepianConfig(mode='required') requested but the model exposes no Slepian terms")
        if mode in {"auto", "required"} and slepian:
            if not _supports_slepian_los(model):
                if mode == "required":
                    raise RuntimeError(
                        "SlepianConfig(mode='required') needs a supported model LOS rule"
                    )
                terms = generic + slepian
            else:
                terms = generic
        else:
            if mode == "off" and slepian:
                fallback = getattr(model, "generic_fallback_terms", None)
                fallback_terms = tuple(fallback()) if callable(fallback) else ()
                terms = fallback_terms if fallback_terms else (generic + slepian)
            else:
                terms = generic
        if terms:
            return _TermCollectionBispectrum3D(model, terms)
        return model

    def _prepare_children(self):
        if self.bispectrum3d is not None:
            model_for_generic = self._prepare_3d_generic_model()
            for combo in self.sample_combinations:
                b2d = self.projector.project(model_for_generic, sample_combination=combo)
                bm = self._make_multipole(b2d)
                self.generic_by_sample[combo] = GenericThreePCFCalculator(
                    bm, config=self.config, coupling_kwargs=self.coupling_kwargs
                )
            self.generic = self.generic_by_sample[self.sample_combinations[0]]
            self.bmultipole = self.generic.bmultipole
        else:
            if self.bmultipole is not None:
                bm = self.bmultipole
            else:
                bm = self._make_multipole(self.bispectrum2d)
                self.bmultipole = bm
            self.generic = GenericThreePCFCalculator(
                bm, config=self.config, coupling_kwargs=self.coupling_kwargs
            )
            self.generic_by_sample = {None: self.generic}

        if (
            self.bispectrum3d is not None
            and self.config.slepian.mode in {"auto", "required"}
            and self.slepian_terms
            and _supports_slepian_los(self.bispectrum3d)
        ):
            self.slepian = SlepianThreePCFCalculator(
                self.slepian_terms, config=self.config, projector=self.projector,
                sample_combinations=self.sample_combinations,
            )

    @property
    def is_batched(self):
        return len(self.sample_combinations) > 1

    @property
    def grid(self):
        return self.generic.grid

    @property
    def Bgrid(self):
        if self.is_batched:
            return {c: g.Bgrid for c, g in self.generic_by_sample.items()}
        return self.generic.Bgrid

    @property
    def Hgrid(self):
        if self.is_batched:
            return {c: g.Hgrid for c, g in self.generic_by_sample.items()}
        return self.generic.Hgrid

    @property
    def ZKgrid(self):
        if self.is_batched:
            if self._combined_ZKgrids is not None:
                return self._combined_ZKgrids
            return {c: g.ZKgrid for c, g in self.generic_by_sample.items()}
        return self._combined_ZKgrid if self._combined_ZKgrid is not None else self.generic.ZKgrid

    @property
    def Zgrid(self):
        if self.is_batched:
            if self._combined_Zgrids is not None:
                return self._combined_Zgrids
            return {c: g.Zgrid for c, g in self.generic_by_sample.items()}
        return self._combined_Zgrid if self._combined_Zgrid is not None else self.generic.Zgrid

    @property
    def timings(self):
        if self.is_batched:
            out = {"generic": {c: dict(g.timings) for c, g in self.generic_by_sample.items()}}
        else:
            out = {"generic": dict(self.generic.timings)}
        if self.slepian is not None:
            out["slepian"] = dict(self.slepian.timings)
        return out

    @property
    def n_components(self):
        return self.generic.n_components

    @property
    def components(self):
        return self.generic.components

    def epsilon_from_component(self, component):
        return self.generic.epsilon_from_component(component)

    def sigma_from_epsilon(self, epsilon):
        return self.generic.sigma_from_epsilon(epsilon)

    def k_values(self, **kwargs):
        return self.generic.k_values(**kwargs)

    def compute_bmultipoles(self, *args, **kwargs):
        if self.is_batched:
            return {c: g.compute_bmultipoles(*args, **kwargs) for c, g in self.generic_by_sample.items()}
        return self.generic.compute_bmultipoles(*args, **kwargs)

    def compute_hkernels(self, *args, **kwargs):
        if self.is_batched:
            return {c: g.compute_hkernels(*args, **kwargs) for c, g in self.generic_by_sample.items()}
        return self.generic.compute_hkernels(*args, **kwargs)

    @staticmethod
    def _merge_zetak(generic, slep):
        combined = ZetaKGrid(spin=generic.spin, kmax=generic.kmax, grid=generic.grid)
        combined.active_epsilons = generic.active_epsilons
        combined.aliases = dict(generic.aliases)
        keys = set(generic.modes) | set(slep.modes)
        for key in keys:
            gm = generic.modes.get(key)
            sm = slep.modes.get(key)
            if gm is not None:
                val = np.array(gm.value, copy=True)
                source_k, source_sigma = gm.source_k, gm.source_sigma
            else:
                val = np.zeros(generic.grid.shape_theta_fft, dtype=complex)
                source_k, source_sigma = sm.source_k, sm.source_sigma
            if sm is not None:
                val += sm.value
            combined.modes[key] = ZetaKMode(
                grid=generic.grid, key=key, value=val,
                source_k=source_k, source_sigma=source_sigma,
            )
        return combined

    def compute_zetak(self, *args, **kwargs):
        if not self.is_batched:
            generic = self.generic.compute_zetak(*args, **kwargs)
            if self.slepian is None:
                self._combined_ZKgrid = generic
                return generic
            slep = self.slepian.compute_zetak_los(
                epsilons=kwargs.get("epsilons"), epsilon=kwargs.get("epsilon"),
                component=kwargs.get("component"),
                all_components=kwargs.get("all_components", False),
                force=kwargs.get("force", False),
            )
            self._combined_ZKgrid = self._merge_zetak(generic, slep)
            return self._combined_ZKgrid

        generic = {c: g.compute_zetak(*args, **kwargs) for c, g in self.generic_by_sample.items()}
        if self.slepian is None:
            self._combined_ZKgrids = generic
            return generic
        slep = self.slepian.compute_zetak_los_many(
            sample_combinations=self.sample_combinations,
            epsilons=kwargs.get("epsilons"), epsilon=kwargs.get("epsilon"),
            component=kwargs.get("component"),
            all_components=kwargs.get("all_components", False),
            force=kwargs.get("force", False),
        )
        self._combined_ZKgrids = {
            c: self._merge_zetak(generic[c], slep[c]) for c in self.sample_combinations
        }
        return self._combined_ZKgrids

    compute_zeta_k = compute_zetak

    def compute_zeta(self, delta_phi, *args, **kwargs):
        phase = kwargs.pop("phase", "nu")
        normalization = kwargs.pop("normalization", 1.0)
        bin_width = kwargs.pop("bin_width", None)
        force = kwargs.pop("force", False)
        if kwargs:
            raise TypeError(f"unexpected compute_zeta keyword(s): {tuple(kwargs)}")
        zk = self.compute_zetak(all_components=True, force=force)
        if self.is_batched:
            self._combined_Zgrids = {
                c: grid.resum(
                    delta_phi, phase=phase, normalization=normalization,
                    bin_width=bin_width, config=self.config,
                ) for c, grid in zk.items()
            }
            return self._combined_Zgrids
        self._combined_Zgrid = zk.resum(
            delta_phi, phase=phase, normalization=normalization,
            bin_width=bin_width, config=self.config,
        )
        return self._combined_Zgrid

    def compute(self, delta_phi, *args, **kwargs):
        return self.compute_zeta(delta_phi, *args, **kwargs)


# Backward compatibility: the historical low-level name remains the generic
# stage calculator.  New production orchestration uses HybridThreePCFCalculator.
ThreePCFCalculator = GenericThreePCFCalculator


__all__ = [
    "HybridThreePCFCalculator",
    "GenericThreePCFCalculator",
    "ThreePCFCalculator",
]
