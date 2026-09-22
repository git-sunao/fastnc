from __future__ import annotations

import numpy as np

from fastnc.hankel.wrapper import PowerLawFFTLogConfig
from ..angular import PowerLawAngularKernelTableConfig, _a31, _t31
from ..composite import CompositeSemiAnalyticBispectrumMultipole3D
from ..fftlog import FFTLogComponent
from ..terms import (DirectFourierTerm, LowRankVFunction, ProductVFunction, LeftVFunction, RightVFunction, SeparableMultipoleTerm)

class BiHalofitBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    r"""Semi-analytic full-Fourier multipoles of the complete BiHalofit model.

    The three-halo term is decomposed into separable FFTLog contributions.
    The one-halo shape variables are fixed to the constructor values
    ``r1`` and ``r2``, making that contribution exactly separable:

    .. math::
       B_{1h}^{\rm fixed}=H(k_1;z,r_1,r_2)H(k_2;z,r_1,r_2)
                           H(k_3;z,r_1,r_2).

    One- and three-halo contributions are combined in this single model.
    """

    def __init__(
        self,
        halofit,
        *,
        r1: float = 0.5,
        r2: float = 0.0,
        k_grid=None,
        fftlog_config: PowerLawFFTLogConfig | None = None,
        angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
    ):
        r1 = float(r1)
        r2 = float(r2)
        if not np.isfinite(r1) or not 0.0 <= r1 <= 1.0:
            raise ValueError("r1 must be finite and lie in [0, 1]")
        if not np.isfinite(r2) or not 0.0 <= r2 <= 1.0:
            raise ValueError("r2 must be finite and lie in [0, 1]")

        halofit.update()
        state = {"halofit": halofit}
        k_grid = np.asarray(halofit.k if k_grid is None else k_grid, dtype=float)
        if k_grid.ndim != 1 or k_grid.size < 2 or np.any(k_grid <= 0.0):
            raise ValueError("k_grid must be a one-dimensional positive grid")
        fftlog_config = fftlog_config or PowerLawFFTLogConfig()
        def coeffs(z):
            return state["halofit"].get_bihalofit_coeffs(np.asarray(z, dtype=float))

        def q_and_coeff(k, z):
            c = coeffs(z)
            q = np.maximum(np.asarray(k, dtype=float) * c["r_sigma"], 1.0e-100)
            return q, c

        def damping(k, z):
            q, c = q_and_coeff(k, z)
            return 1.0 / (1.0 + c["en"] * q)

        def effective_power(k, z):
            q, c = q_and_coeff(k, z)
            pl = state["halofit"].get_interpolated_pklin(np.asarray(k, dtype=float), z)
            return (
                (1.0 + c["fn"] * q**2)
                / (1.0 + c["gn"] * q + c["hn"] * q**2)
                * pl
                + 1.0
                / (c["mn"] * q**c["mun"] + c["nn"] * q**c["nun"])
                / (1.0 + (c["pn"] * q) ** -3)
            )

        def dressed_power(k, z):
            return damping(k, z) * effective_power(k, z)

        def one_halo_profile(k, z):
            q, c = q_and_coeff(k, z)
            an = 10.0 ** (c["log10an1"] + c["log10an2"] * r1 ** c["gan"])
            aln = 10.0 ** (c["log10aln1"] + c["log10aln2"] * r2**2)
            ns = float(state["halofit"].cosmo["ns"])
            aln = np.minimum(aln, 1.0 - (2.0 / 3.0) * ns)
            ben = 10.0 ** (c["log10ben1"] + c["log10ben2"] * r2)
            return (
                1.0 / (an * q**aln + c["bn"] * q**ben)
                / (1.0 + 1.0 / (c["cn"] * q))
            )

        component_i = FFTLogComponent("bihalofit-I", k_grid, damping, fftlog_config)
        component_h = FFTLogComponent("bihalofit-IPE", k_grid, dressed_power, fftlog_config)
        component_1h = FFTLogComponent(
            f"bihalofit-1h-r1={r1:g}-r2={r2:g}",
            k_grid,
            one_halo_profile,
            fftlog_config,
        )

        def one(x2, x3):
            return np.ones(np.broadcast(x2, x3).shape, dtype=float)

        v_1h = ProductVFunction(
            one_halo_profile, one_halo_profile, name="H(k2) H(k3)"
        )
        v_23 = ProductVFunction(
            dressed_power, dressed_power, name="IPE(k2) IPE(k3)"
        )
        v_31 = ProductVFunction(
            damping, dressed_power, name="I(k2) IPE(k3)"
        )
        v_12 = ProductVFunction(
            dressed_power, damping, name="IPE(k2) I(k3)"
        )

        def f23_u(mode_abs):
            if mode_abs == 0:
                return lambda x2, x3: np.full(np.broadcast(x2, x3).shape, 12.0 / 7.0)
            if mode_abs == 1:
                return lambda x2, x3: 0.5 * (x2 / x3 + x3 / x2)
            if mode_abs == 2:
                return lambda x2, x3: np.full(np.broadcast(x2, x3).shape, 1.0 / 7.0)
            raise ValueError("F2 23 has only |L|=0,1,2")

        terms = [
            SeparableMultipoleTerm(
                "bihalofit-1h-fixed-shape", component_1h, 0, one, v_1h
            )
        ]

        for mode_abs in (0, 1, 2):
            active_modes = (0,) if mode_abs == 0 else (-mode_abs, mode_abs)
            terms.append(
                SeparableMultipoleTerm(
                    f"bihalofit-3h-23-F2-L{mode_abs}",
                    component_i,
                    0,
                    f23_u(mode_abs),
                    v_23,
                    modes=active_modes,
                )
            )

        for p in (-2, 0, 2):
            terms.extend(
                [
                    SeparableMultipoleTerm(
                        f"bihalofit-3h-31-F2-p{p}",
                        component_h,
                        p,
                        lambda x2, x3, p=p: 2.0 * _a31(p, x2, x3),
                        v_31,
                    ),
                    SeparableMultipoleTerm(
                        f"bihalofit-3h-12-F2-p{p}",
                        component_h,
                        p,
                        lambda x2, x3, p=p: 2.0 * _a31(p, x3, x2),
                        v_12,
                    ),
                ]
            )

        def v_dn23(k2, k3, z):
            c = coeffs(z)
            k = np.hypot(k2, k3)
            return 2.0 * c["dn"] * c["r_sigma"] * k * v_23(k2, k3, z)

        def dn_prefactor(z):
            c = coeffs(z)
            return 2.0 * c["dn"] * c["r_sigma"]

        v_dn31 = ProductVFunction(
            damping,
            lambda k, z: dn_prefactor(z) * k * dressed_power(k, z),
            name="I(k2) [2 dn r_sigma k3 IPE(k3)]",
        )
        v_dn12 = ProductVFunction(
            lambda k, z: dn_prefactor(z) * k * dressed_power(k, z),
            damping,
            name="[2 dn r_sigma k2 IPE(k2)] I(k3)",
        )

        terms.extend(
            [
                SeparableMultipoleTerm(
                    "bihalofit-3h-23-dnq1", component_i, 1, one, v_dn23
                ),
                SeparableMultipoleTerm(
                    "bihalofit-3h-31-dnq3", component_h, 0, one, v_dn31
                ),
                SeparableMultipoleTerm(
                    "bihalofit-3h-12-dnq2", component_h, 0, one, v_dn12
                ),
            ]
        )

        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.halofit = halofit
        self.r1 = r1
        self.r2 = r2
        self.k_grid = k_grid
        self.fftlog_config = fftlog_config
        self._damping = damping
        self._effective_power = effective_power
        self._dressed_power = dressed_power
        self._one_halo_profile = one_halo_profile
        self._physical_state = state
        self._physical_components = (component_i, component_h, component_1h)

    def update_physics(self, *, halofit):
        """Replace the Halofit state while retaining angular kernels."""
        halofit.update()
        self._physical_state["halofit"] = halofit
        self.halofit = halofit
        self.invalidate_components(self._physical_components)
        return self


