"""Direct 3D BiHalofit term aggregate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import eval_jacobi

from ..bispectrum import Bispectrum3D
from ..halofit import Halofit
from ..representations import (
    NumericExpression3D,
    SemiAnalyticLowRankProductExpression3D,
    SemiAnalyticRadialExpression3D,
)
from ..support import Support3D
from ..terms import BispectrumTerm3D
from fastnc.utils.cosmology import (
    default_wmap_like_cosmology,
    eisenstein_hu_like_pklin,
    simple_debug_pklin,
    simple_linear_growth,
)

_BH3_PAIR_LAYOUTS = {
    "12": (0, 1, 2),
    "23": (1, 2, 0),
    "31": (2, 0, 1),
}

_BH1_TRAINED_BASIS_FILES = {
    "broad-debug-v1": "bihalofit_bh1_broad_debug_v1.npz",
}


@dataclass(frozen=True)
class BiHalofitBh1SemiAnalyticConfig:
    """Construction settings for the trained Bh1 semi-analytic expression.

    ``rank`` selects one of the validated empirical-interpolation bases.  The
    trained basis is immutable model-family data: changing cosmology updates
    only the recovered radial profiles and never repeats the offline SVD.
    """

    rank: int = 8
    trained_basis: str = "broad-debug-v1"

    def __post_init__(self):
        rank = int(self.rank)
        if rank not in {5, 6, 8, 10}:
            raise ValueError("rank must be one of 5, 6, 8, or 10")
        if self.trained_basis not in _BH1_TRAINED_BASIS_FILES:
            available = ", ".join(sorted(_BH1_TRAINED_BASIS_FILES))
            raise ValueError(
                f"unknown trained_basis {self.trained_basis!r}; "
                f"available bases: {available}"
            )
        object.__setattr__(self, "rank", rank)


def _load_bh1_trained_basis(config):
    filename = _BH1_TRAINED_BASIS_FILES[config.trained_basis]
    path = Path(__file__).with_name("data") / filename
    with np.load(path) as data:
        rank = config.rank
        return {
            "degree": int(data["degree"]),
            "basis_vectors": np.array(data[f"basis_vectors_{rank}"], copy=True),
            "selected_shapes": np.array(data[f"selected_shapes_{rank}"], copy=True),
            "interpolation_matrix": np.array(
                data[f"interpolation_matrix_{rank}"], copy=True
            ),
        }


def _dubiner_basis(r1, r2, degree):
    """Evaluate the total-degree Dubiner basis on the Bh1 shape triangle."""
    r1, r2 = np.broadcast_arrays(
        np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    )
    xi = 2.0 * (r1 - r2)
    eta = r2
    denominator = 1.0 - eta
    collapsed = np.divide(
        2.0 * xi,
        denominator,
        out=np.zeros_like(xi),
        where=denominator > 1.0e-14,
    ) - 1.0
    vertical = 2.0 * eta - 1.0
    return np.stack([
        eval_jacobi(p, 0.0, 0.0, collapsed)
        * denominator**p
        * eval_jacobi(q, 2.0 * p + 1.0, 0.0, vertical)
        for p in range(degree + 1)
        for q in range(degree + 1 - p)
    ])


def _bh3_primitive_specs():
    """Return labels and Fourier/radial powers for exact Bh3 primitives."""
    specs = []
    for pair in _BH3_PAIR_LAYOUTS:
        specs.append((pair, "m+0", 0, 0, 12.0 / 7.0, False))
        for mode in (-1, 1):
            specs.append((pair, f"m{mode:+d}:r+1", mode, 1, 0.5, False))
            specs.append((pair, f"m{mode:+d}:r-1", mode, -1, 0.5, False))
        for mode in (-2, 2):
            specs.append((pair, f"m{mode:+d}", mode, 0, 1.0 / 7.0, False))
        specs.append((pair, "extra:m+0", 0, 0, 2.0, True))
    return tuple(specs)


class BiHalofitBispectrum3D(Bispectrum3D):
    """Wrapper for the bundled :class:`Halofit` / Bihalofit model.

    This class does not self-initialize by default.  A scientific run should
    configure it explicitly with cosmology, linear power spectrum, and growth
    factor.

    Use ``BiHalofitBispectrum3D.simple_debug()`` only for quick tests.
    """

    def __init__(
        self,
        halofit: Halofit | None = None,
        support: Support3D | None = None,
        support_policy: str = "zero",
        config_semi_analytic: BiHalofitBh1SemiAnalyticConfig | None = None,
    ):
        self.halofit = halofit or Halofit()
        self.config_semi_analytic = (
            BiHalofitBh1SemiAnalyticConfig()
            if config_semi_analytic is None
            else config_semi_analytic
        )
        if not isinstance(
            self.config_semi_analytic, BiHalofitBh1SemiAnalyticConfig
        ):
            raise TypeError(
                "config_semi_analytic must be a "
                "BiHalofitBh1SemiAnalyticConfig"
            )
        self._bh1_trained_basis = _load_bh1_trained_basis(
            self.config_semi_analytic
        )
        self._support_policy = support_policy
        self._user_support = support is not None
        terms = [
            BispectrumTerm3D(
                name="bihalofit:Bh1",
                representations=(
                    NumericExpression3D(self._evaluate_bh1),
                    SemiAnalyticLowRankProductExpression3D(
                        rank=self.config_semi_analytic.rank,
                        amplitude_evaluator=self._bh1_shape_amplitudes,
                        radial_evaluator=self._bh1_radial_profiles,
                        trained_basis=self.config_semi_analytic.trained_basis,
                    ),
                ),
            )
        ]
        for pair, label, mode, ratio_power, coefficient, extra in (
            _bh3_primitive_specs()
        ):
            representations = [
                NumericExpression3D(
                    self._make_bh3_primitive_evaluator(
                        pair,
                        mode=mode,
                        ratio_power=ratio_power,
                        coefficient=coefficient,
                        extra=extra,
                    )
                )
            ]
            if pair == "23":
                representations.append(
                    self._make_bh3_pair23_semi_expression(
                        mode=mode,
                        ratio_power=ratio_power,
                        coefficient=coefficient,
                        extra=extra,
                    )
                )
            terms.append(
                BispectrumTerm3D(
                    name=f"bihalofit:Bh3:{pair}:{label}",
                    representations=tuple(representations),
                )
            )
        super().__init__(tuple(terms), support=support or Support3D(policy=support_policy))

    @property
    def ready(self) -> bool:
        return (
            getattr(self.halofit, "cosmo", None) is not None
            and getattr(self.halofit, "k", None) is not None
            and getattr(self.halofit, "pklin", None) is not None
            and getattr(self.halofit, "z", None) is not None
            and getattr(self.halofit, "lgr", None) is not None
        )

    def _refresh_support(self) -> None:
        if self._user_support:
            return
        k = getattr(self.halofit, "k", None)
        z = getattr(self.halofit, "z", None)
        if k is None or z is None:
            return
        k = np.asarray(k, dtype=float)
        z = np.asarray(z, dtype=float)
        self.support = Support3D(
            k_min=float(np.nanmin(k)),
            k_max=float(np.nanmax(k)),
            z_min=float(np.nanmin(z)),
            z_max=float(np.nanmax(z)),
            policy=self._support_policy,
        )

    def set_cosmology(self, cosmo: Mapping[str, float]) -> "BiHalofitBispectrum3D":
        self.halofit.set_cosmology(dict(cosmo))
        self._refresh_support()
        self._state_updated()
        return self

    def set_pklin(self, k, pklin) -> "BiHalofitBispectrum3D":
        self.halofit.set_pklin(np.asarray(k, dtype=float), np.asarray(pklin, dtype=float))
        self._refresh_support()
        self._state_updated()
        return self

    def set_growth(self, z, growth) -> "BiHalofitBispectrum3D":
        """Set the linear growth factor/grid used by the bundled Halofit code."""
        self.halofit.set_lgr(np.asarray(z, dtype=float), np.asarray(growth, dtype=float))
        self._refresh_support()
        self._state_updated()
        return self

    def set_lgr(self, z, lgr) -> "BiHalofitBispectrum3D":
        """Alias matching the naming in ``halofit.py``."""
        return self.set_growth(z, lgr)

    def configure(
        self,
        *,
        cosmo: Mapping[str, float] | None = None,
        k=None,
        pklin=None,
        z=None,
        growth=None,
        lgr=None,
    ) -> "BiHalofitBispectrum3D":
        """Configure this instance in-place and return ``self``."""
        if cosmo is not None:
            self.set_cosmology(cosmo)
        if k is not None or pklin is not None:
            if k is None or pklin is None:
                raise ValueError("Both k and pklin must be supplied together.")
            self.set_pklin(k, pklin)
        if z is not None or growth is not None or lgr is not None:
            g = growth if growth is not None else lgr
            if z is None or g is None:
                raise ValueError("Both z and growth/lgr must be supplied together.")
            self.set_growth(z, g)
        return self

    @classmethod
    def from_cosmology(
        cls,
        *,
        cosmo: Mapping[str, float],
        k,
        pklin,
        z,
        growth=None,
        lgr=None,
        support: Support3D | None = None,
        support_policy: str = "zero",
        halofit: Halofit | None = None,
        config_semi_analytic: BiHalofitBh1SemiAnalyticConfig | None = None,
    ) -> "BiHalofitBispectrum3D":
        """Create and configure a Bihalofit wrapper from user inputs."""
        obj = cls(
            halofit=halofit,
            support=support,
            support_policy=support_policy,
            config_semi_analytic=config_semi_analytic,
        )
        return obj.configure(cosmo=cosmo, k=k, pklin=pklin, z=z, growth=growth, lgr=lgr)

    @classmethod
    def simple_debug(
        cls,
        *,
        cosmo: Mapping[str, float] | None = None,
        k=None,
        z=None,
        amplitude: float = 1.0e4,
        k_eq: float = 2.0e-2,
        transfer_power: float = 1.5,
        pklin_kind: str = "debug",
        support_policy: str = "zero",
        config_semi_analytic: BiHalofitBh1SemiAnalyticConfig | None = None,
    ) -> "BiHalofitBispectrum3D":
        """Return a self-initialized Bihalofit object for debugging.

        The linear spectrum and growth are deliberately simple and should not
        be used for scientific inference.  For production, use
        ``from_cosmology`` with CAMB/CLASS inputs.
        """
        cosmo_dict = default_wmap_like_cosmology() if cosmo is None else dict(cosmo)
        k_arr = np.logspace(-4, 2, 512) if k is None else np.asarray(k, dtype=float)
        z_arr = np.linspace(0.0, 3.0, 128) if z is None else np.asarray(z, dtype=float)

        if pklin_kind == "debug":
            pklin = simple_debug_pklin(
                k_arr,
                cosmo=cosmo_dict,
                amplitude=amplitude,
                k_eq=k_eq,
                transfer_power=transfer_power,
            )
        elif pklin_kind in {"eisenstein-hu", "eisenstein_hu", "eh"}:
            pklin = eisenstein_hu_like_pklin(k_arr, cosmo=cosmo_dict, amplitude=amplitude)
        else:
            raise ValueError("pklin_kind must be 'debug' or 'eisenstein-hu'.")

        growth = simple_linear_growth(z_arr, cosmo=cosmo_dict)
        return cls.from_cosmology(
            cosmo=cosmo_dict,
            k=k_arr,
            pklin=pklin,
            z=z_arr,
            growth=growth,
            support_policy=support_policy,
            config_semi_analytic=config_semi_analytic,
        )

    def _validate_evaluation(self, params):
        if not self.ready:
            raise RuntimeError(
                "BiHalofitBispectrum3D is not configured. "
                "Call set_cosmology(), set_pklin(), and set_growth(), or use "
                "BiHalofitBispectrum3D.from_cosmology(...)."
            )
        if "which" in params:
            raise TypeError(
                "select BiHalofit components with select_terms() instead of "
                "the legacy which parameter"
            )

    def _evaluate_bh1(self, k1, k2, k3, z, **params):
        self._validate_evaluation(params)
        return self.halofit.get_bihalofit(
            k1, k2, k3, z, which="Bh1", **params
        )

    def _bh1_shape_amplitudes(self, k1, k2, k3):
        sides = np.sort(np.stack(np.broadcast_arrays(k1, k2, k3)), axis=0)
        largest = sides[2]
        r1 = np.divide(sides[0], largest)
        r2 = np.divide(sides[1] + sides[0] - largest, largest)
        trained = self._bh1_trained_basis
        basis = _dubiner_basis(r1, r2, trained["degree"])
        return np.tensordot(trained["basis_vectors"].T, basis, axes=(1, 0))

    def _bh1_profile_at_shapes(self, k, z, shapes):
        """Evaluate all representative one-leg profiles in one broadcast."""
        self.halofit.update()
        k, z = np.broadcast_arrays(
            np.asarray(k, dtype=float), np.asarray(z, dtype=float)
        )
        shape_axis = (shapes.shape[0],) + (1,) * k.ndim
        r1 = shapes[:, 0].reshape(shape_axis)
        r2 = shapes[:, 1].reshape(shape_axis)
        fit = self.halofit.get_bihalofit_coeffs(z[None, ...])
        q = np.maximum(k[None, ...] * fit["r_sigma"], 1.0e-100)
        an = 10.0 ** (
            fit["log10an1"] + fit["log10an2"] * r1 ** fit["gan"]
        )
        alpha = 10.0 ** (
            fit["log10aln1"] + fit["log10aln2"] * r2**2
        )
        alpha = np.minimum(
            alpha, 1.0 - (2.0 / 3.0) * self.halofit.cosmo["ns"]
        )
        beta = 10.0 ** (
            fit["log10ben1"] + fit["log10ben2"] * r2
        )
        return (
            1.0 / (an * q**alpha + fit["bn"] * q**beta)
            / (1.0 + 1.0 / (fit["cn"] * q))
        )

    def _bh1_radial_profiles(self, k, z):
        trained = self._bh1_trained_basis
        values = self._bh1_profile_at_shapes(
            k, z, trained["selected_shapes"]
        )
        return np.linalg.solve(
            trained["interpolation_matrix"],
            values.reshape(self.config_semi_analytic.rank, -1),
        ).reshape(values.shape)

    def _bh3_radial_values(self, k1, k2, k3, z):
        self.halofit.update()
        k1, k2, k3, z = np.broadcast_arrays(
            np.asarray(k1, dtype=float),
            np.asarray(k2, dtype=float),
            np.asarray(k3, dtype=float),
            np.asarray(z, dtype=float),
        )
        kmin, kmid, kmax = np.sort(np.stack((k1, k2, k3)), axis=0)
        physical, _ = self.halofit._clip_triangle_boundary(kmin, kmid, kmax)
        coefficients = self.halofit.get_bihalofit_coeffs(z)
        q_floor = 1.0e-100
        q = tuple(
            np.maximum(k * coefficients["r_sigma"], q_floor)
            for k in (k1, k2, k3)
        )
        linear_power = tuple(
            self.halofit.get_interpolated_pklin(k, z) for k in (k1, k2, k3)
        )
        effective_power = tuple(
            (
                (1.0 + coefficients["fn"] * qi**2)
                / (
                    1.0
                    + coefficients["gn"] * qi
                    + coefficients["hn"] * qi**2
                )
                * power
                + 1.0
                / (
                    coefficients["mn"] * qi ** coefficients["mun"]
                    + coefficients["nn"] * qi ** coefficients["nun"]
                )
                / (1.0 + (coefficients["pn"] * qi) ** -3)
            )
            for qi, power in zip(q, linear_power)
        )
        damping = tuple(1.0 / (1.0 + coefficients["en"] * qi) for qi in q)
        radial = tuple(di * ei for di, ei in zip(damping, effective_power))
        return (k1, k2, k3), q, damping, radial, coefficients, physical

    def _bh3_leg_components(self, k, z):
        """Return ``q``, damping, and effective radial power for one leg."""
        self.halofit.update()
        k, z = np.broadcast_arrays(
            np.asarray(k, dtype=float), np.asarray(z, dtype=float)
        )
        coefficients = self.halofit.get_bihalofit_coeffs(z)
        q = np.maximum(k * coefficients["r_sigma"], 1.0e-100)
        linear_power = self.halofit.get_interpolated_pklin(k, z)
        effective_power = (
            (1.0 + coefficients["fn"] * q**2)
            / (1.0 + coefficients["gn"] * q + coefficients["hn"] * q**2)
            * linear_power
            + 1.0
            / (
                coefficients["mn"] * q ** coefficients["mun"]
                + coefficients["nn"] * q ** coefficients["nun"]
            )
            / (1.0 + (coefficients["pn"] * q) ** -3)
        )
        damping = 1.0 / (1.0 + coefficients["en"] * q)
        return q, damping, damping * effective_power, coefficients

    def _make_bh3_pair23_semi_expression(
        self,
        *,
        mode,
        ratio_power,
        coefficient,
        extra,
    ):
        """Return the grid-free semi-analytic expression for pair ``23|1``."""

        def u(ratio2, ratio3):
            if ratio_power == 1:
                radial_ratio = ratio2 / ratio3
            elif ratio_power == -1:
                radial_ratio = ratio3 / ratio2
            else:
                radial_ratio = 1.0
            return coefficient * radial_ratio

        def v(k2, k3, z):
            return (
                self._bh3_leg_components(k2, z)[2]
                * self._bh3_leg_components(k3, z)[2]
            )

        def w(k1, z):
            q1, damping1, _, fit = self._bh3_leg_components(k1, z)
            if extra:
                return fit["dn"] * q1 * damping1
            return damping1

        return SemiAnalyticRadialExpression3D(
            u_evaluator=u,
            v_evaluator=v,
            w_evaluator=w,
            angular_order=mode,
        )

    def _make_bh3_primitive_evaluator(
        self,
        pair,
        *,
        mode,
        ratio_power,
        coefficient,
        extra,
    ):
        left, right, opposite = _BH3_PAIR_LAYOUTS[pair]

        def evaluate(k1, k2, k3, z, **params):
            self._validate_evaluation(params)
            k, q, damping, radial, fit, physical = self._bh3_radial_values(
                k1, k2, k3, z
            )
            value = radial[left] * radial[right] * damping[opposite]
            if extra:
                value = value * fit["dn"] * q[opposite]
            elif ratio_power == 1:
                value = value * k[left] / k[right]
            elif ratio_power == -1:
                value = value * k[right] / k[left]
            if mode:
                cosine = (
                    k[opposite] ** 2 - k[left] ** 2 - k[right] ** 2
                ) / (2.0 * k[left] * k[right])
                angle = np.arccos(np.clip(cosine, -1.0, 1.0))
                value = value * np.exp(1j * mode * angle)
            value = coefficient * value
            return np.where(physical, value, np.nan)

        return evaluate

    def select_terms(self, *names: str):
        expanded = []
        for name in names:
            if name == "bihalofit:Bh3":
                expanded.extend(
                    term.name
                    for term in self.terms
                    if term.name.startswith("bihalofit:Bh3:")
                )
            else:
                expanded.append(name)
        return super().select_terms(*expanded)


class BiHalofitFixedShapeOneHaloBispectrum3D(Bispectrum3D):
    r"""Fixed-shape separable approximation to the BiHalofit one-halo term.

    The exact BiHalofit fitting parameters depend on the triangle variables
    ``r1`` and ``r2``. This model fixes them explicitly, making
    ``B_1h = H(k1,z) H(k2,z) H(k3,z)`` exactly separable within the
    approximation. It remains distinct from the exact Bh1 term.
    """

    def __init__(
        self,
        halofit: Halofit,
        *,
        fiducial_r1: float,
        fiducial_r2: float,
        support: Support3D | None = None,
        _revision_sources=None,
    ):
        if not isinstance(halofit, Halofit):
            raise TypeError("halofit must be a Halofit instance")
        self.halofit = halofit
        self._fiducial_r1, self._fiducial_r2 = self._validate_shape(
            fiducial_r1, fiducial_r2
        )
        term = BispectrumTerm3D(
            name="bihalofit:Bh1:fixed-shape",
            representations=(
                NumericExpression3D(self._evaluate_numeric),
                SemiAnalyticRadialExpression3D(
                    u_evaluator=lambda ratio2, ratio3: np.ones(
                        np.broadcast_shapes(np.shape(ratio2), np.shape(ratio3))
                    ),
                    v_evaluator=lambda k2, k3, z: (
                        self.profile(k2, z) * self.profile(k3, z)
                    ),
                    w_evaluator=self.profile,
                ),
            ),
        )
        if support is None and self.ready:
            support = Support3D(
                k_min=float(np.min(self.halofit.k)),
                k_max=float(np.max(self.halofit.k)),
                z_min=float(np.min(self.halofit.z)),
                z_max=float(np.max(self.halofit.z)),
                policy="zero",
            )
        revision_sources = None
        if _revision_sources is not None:
            revision_sources = (
                lambda: self._state_revision,
                *tuple(_revision_sources),
            )
        super().__init__(
            (term,),
            support=support or Support3D(policy="ignore"),
            _revision_sources=revision_sources,
        )

    @staticmethod
    def _validate_shape(r1, r2):
        r1 = float(r1)
        r2 = float(r2)
        if not 0.0 < r1 <= 1.0:
            raise ValueError("fiducial_r1 must lie in (0, 1]")
        if not 0.0 <= r2 <= 1.0:
            raise ValueError("fiducial_r2 must lie in [0, 1]")
        return r1, r2

    @property
    def ready(self) -> bool:
        return (
            getattr(self.halofit, "cosmo", None) is not None
            and getattr(self.halofit, "k", None) is not None
            and getattr(self.halofit, "pklin", None) is not None
            and getattr(self.halofit, "z", None) is not None
            and getattr(self.halofit, "lgr", None) is not None
        )

    @property
    def fiducial_shape(self) -> tuple[float, float]:
        return self._fiducial_r1, self._fiducial_r2

    def set_fiducial_shape(self, r1, r2):
        """Update the approximation point and source-state revision."""
        self._fiducial_r1, self._fiducial_r2 = self._validate_shape(r1, r2)
        self._state_updated()
        return self

    @classmethod
    def from_bihalofit(cls, source, *, fiducial_r1, fiducial_r2):
        """Construct the approximation from a configured exact model."""
        if not isinstance(source, BiHalofitBispectrum3D):
            raise TypeError("source must be a BiHalofitBispectrum3D")
        return cls(
            source.halofit,
            fiducial_r1=fiducial_r1,
            fiducial_r2=fiducial_r2,
            support=source.support,
            _revision_sources=source._revision_sources,
        )

    @classmethod
    def simple_debug(cls, *, fiducial_r1, fiducial_r2, **kwargs):
        """Construct a debug-only approximation from bundled toy inputs."""
        source = BiHalofitBispectrum3D.simple_debug(**kwargs)
        return cls.from_bihalofit(
            source,
            fiducial_r1=fiducial_r1,
            fiducial_r2=fiducial_r2,
        )

    def profile(self, k, z):
        """Evaluate the fixed-shape one-leg factor ``H(k,z)``."""
        if not self.ready:
            raise RuntimeError("the underlying Halofit instance is not configured")
        self.halofit.update()
        k, z = np.broadcast_arrays(
            np.asarray(k, dtype=float), np.asarray(z, dtype=float)
        )
        coefficients = self.halofit.get_bihalofit_coeffs(z)
        q = np.maximum(k * coefficients["r_sigma"], 1.0e-100)
        r1, r2 = self.fiducial_shape
        an = 10.0 ** (
            coefficients["log10an1"]
            + coefficients["log10an2"] * r1 ** coefficients["gan"]
        )
        alpha = 10.0 ** (
            coefficients["log10aln1"]
            + coefficients["log10aln2"] * r2**2
        )
        alpha = np.minimum(
            alpha, 1.0 - (2.0 / 3.0) * self.halofit.cosmo["ns"]
        )
        beta = 10.0 ** (
            coefficients["log10ben1"]
            + coefficients["log10ben2"] * r2
        )
        return (
            1.0 / (an * q**alpha + coefficients["bn"] * q**beta)
            / (1.0 + 1.0 / (coefficients["cn"] * q))
        )

    def _evaluate_numeric(self, k1, k2, k3, z, **params):
        if params:
            names = ", ".join(sorted(params))
            raise TypeError(f"unused fixed-shape Bh1 parameters: {names}")
        k1, k2, k3, z = np.broadcast_arrays(
            np.asarray(k1, dtype=float),
            np.asarray(k2, dtype=float),
            np.asarray(k3, dtype=float),
            np.asarray(z, dtype=float),
        )
        kmin, kmid, kmax = np.sort(np.stack((k1, k2, k3)), axis=0)
        physical = self.halofit._triangle_physical_mask(kmin, kmid, kmax)
        values = self.profile(k1, z) * self.profile(k2, z) * self.profile(k3, z)
        return np.where(physical, values, np.nan)
