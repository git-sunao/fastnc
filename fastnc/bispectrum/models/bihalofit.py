"""Direct 3D BiHalofit term aggregate."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..bispectrum import Bispectrum3D
from ..halofit import Halofit
from ..representations import NumericExpression3D
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
    ):
        self.halofit = halofit or Halofit()
        self._support_policy = support_policy
        self._user_support = support is not None
        terms = [
            BispectrumTerm3D(
                name="bihalofit:Bh1",
                representations=(NumericExpression3D(self._evaluate_bh1),),
            )
        ]
        terms.extend(
            BispectrumTerm3D(
                name=f"bihalofit:Bh3:{pair}:{label}",
                representations=(
                    NumericExpression3D(
                        self._make_bh3_primitive_evaluator(
                            pair,
                            mode=mode,
                            ratio_power=ratio_power,
                            coefficient=coefficient,
                            extra=extra,
                        )
                    ),
                ),
            )
            for pair, label, mode, ratio_power, coefficient, extra
            in _bh3_primitive_specs()
        )
        terms.append(
            BispectrumTerm3D(
                name="bihalofit:Bh3:squeezed-correction",
                representations=(
                    NumericExpression3D(self._evaluate_bh3_squeezed_correction),
                ),
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
    ) -> "BiHalofitBispectrum3D":
        """Create and configure a Bihalofit wrapper from user inputs."""
        obj = cls(halofit=halofit, support=support, support_policy=support_policy)
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

    def _evaluate_bh3_squeezed_correction(self, k1, k2, k3, z, **params):
        """Return the grouped squeezed-limit correction to direct primitives."""
        self._validate_evaluation(params)
        params = dict(params)
        squeezed_safe = bool(params.pop("squeezed_safe", True))
        if not squeezed_safe:
            shape = np.broadcast_shapes(
                np.shape(k1), np.shape(k2), np.shape(k3), np.shape(z)
            )
            return np.zeros(shape, dtype=float)
        safe = self.halofit.get_bihalofit(
            k1, k2, k3, z, which="Bh3", squeezed_safe=True, **params
        )
        direct = self.halofit.get_bihalofit(
            k1, k2, k3, z, which="Bh3", squeezed_safe=False, **params
        )
        return safe - direct

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
