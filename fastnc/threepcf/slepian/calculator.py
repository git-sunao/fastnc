"""Fixed-redshift, no-LOS Slepian 3PCF calculator (Phase 7)."""
from __future__ import annotations
import time
import hashlib
import numpy as np
from scipy.special import jv

from ..grid import FFTGrid
from ..spin import SpinSpec, as_effective_spin_triple
from ..zetak_grid import ZetaKGrid, ZetaKMode
from .fftlog import FFTLogExpansion, decompose_log_powerlaw
from .geometry import SlepianRadialGrid
from .weber import WeberTableCache, single_bessel_factor, canonical_bessel_order
from .terms import compile_mode_plan
from .los_moments import (
    FactorizedGrowthMomentRule, FactorizedGrowthBatchMomentRule,
    GeneralCoefficientMomentRule, GeneralCoefficientBatchMomentRule,
)


class SlepianThreePCFCalculator:
    """Direct separable ``B -> zeta_k`` transform at one fixed redshift.

    This Phase-7 calculator deliberately has no LOS integration.  ``z`` and
    ``chi`` must be supplied to :meth:`compute_zetak`; Phase 8 will provide the
    projector/moment wrapper.  No bispectrum multipoles ``B_L`` are formed.
    """
    is_operational = True

    def __init__(self, terms, *, config, projector=None, sample_combinations=None):
        self.terms = tuple(terms)
        if not self.terms:
            raise ValueError("SlepianThreePCFCalculator requires at least one term")
        self.config = config
        self.projector = projector
        self.sample_combinations = sample_combinations
        self.grid = FFTGrid.from_config(config)
        self.radial_grid = SlepianRadialGrid.from_config(config)
        self.weber = WeberTableCache(config.slepian.weber_r_points)
        self.spin_spec = SpinSpec(config.spin)
        self.ZKgrid = ZetaKGrid(spin=config.spin, kmax=config.kmax, grid=self.grid)
        self.timings = {}
        self._fftlog_cache = {}
        self._constant_cache = {}
        self._prepared_two_bessel = {}
        self._prepared_single_basis = {}
        self._prepared_double_basis = {}
        self._prepared_double_contact = {}
        self._compiled_mode_plans = {}
        self._factorized_shift_cache = {}
        self._contact_diag_cache = {}
        self._general_node_mode_cache = {}
        self._prepared_los_moment_cache = {}
        self._state_token_memo = {}
        self._phase10_counters = {
            "single_basis_builds": 0,
            "double_basis_builds": 0,
            "double_contact_builds": 0,
            "shifted_expansion_builds": 0,
            "compiled_plan_builds": 0,
            "contact_diag_builds": 0,
        }
        self._los_rule_cache = {}
        self._batch_ZKgrids = None
        self.constructed_bmultipoles = False

    def _constant_value(self, term, leg, z):
        key = (id(term), int(leg), float(z))
        if key in self._constant_cache:
            return self._constant_cache[key]
        f = (term.f1, term.f2, term.f3)[leg]
        probe = np.geomspace(self.config.slepian.k_min, self.config.slepian.k_max, 7)
        vals = np.asarray(f(probe, z), dtype=complex)
        vals = np.broadcast_to(vals, probe.shape)
        scale = max(1.0, float(np.max(np.abs(vals))))
        value = complex(vals[0]) if np.max(np.abs(vals - vals[0])) <= 1e-12 * scale else None
        self._constant_cache[key] = value
        return value

    def _expansion(self, term, leg, z, chi):
        key = (id(term), int(leg), float(z), float(chi))
        if key in self._fftlog_cache:
            return self._fftlog_cache[key]
        f = (term.f1, term.f2, term.f3)[leg]
        const = self._constant_value(term, leg, z)
        if const is not None:
            ex = FFTLogExpansion(
                k=np.asarray([self.config.slepian.k_min]),
                coefficients=np.asarray([const], dtype=complex),
                exponents=np.asarray([0.0 + 0.0j]),
                bias=0.0,
            )
        elif (
            hasattr(term, "power_shifts")
            and hasattr(term, "model")
            and getattr(term.model, "has_factorized_growth", False)
        ):
            # With an explicit P(k,z)=D(z)^2 P0(k) capability, preserve the
            # exact integer exponent-shift relation between P, kP and P/k.
            # This is required by the Phase-8 Mellin LOS moments and avoids
            # giving algebraically identical shifted radial factors unrelated
            # finite-grid FFTLog fits.
            base = self._factorized_expansion(term, leg)
            _, growth = term.model.factorized_linear_power()
            ex = FFTLogExpansion(
                k=base.k,
                coefficients=base.coefficients * complex(np.asarray(growth(z)))**2,
                exponents=base.exponents,
                bias=base.bias,
            )
        else:
            # Decompose the actual 3D radial factor f(k,z).  chi enters later as chi^-nu.
            ex = decompose_log_powerlaw(
                lambda k, zz: f(k, zz),
                k_min=self.config.slepian.k_min,
                k_max=self.config.slepian.k_max,
                n=self.config.slepian.n_fftlog,
                bias=self.config.slepian.fftlog_bias,
                z=z,
            )
        self._fftlog_cache[key] = ex
        return ex

    def _single_at(self, ex, order, chi, x):
        x = np.asarray(x, dtype=float)
        coeff = ex.coefficients * float(chi) ** (-ex.exponents)
        basis = single_bessel_factor(ex.exponents, order)[:, None] * x[None, :] ** (-ex.exponents[:, None] - 2.0)
        return np.sum(coeff[:, None] * basis, axis=0)

    def _single(self, ex, order, chi):
        return self._single_at(ex, order, chi, self.radial_grid.x)

    def _double_at(self, ex, order_x, order_theta, chi, x, theta):
        theta = np.asarray(theta, dtype=float)
        x = np.asarray(x, dtype=float)
        ox, sx = canonical_bessel_order(order_x)
        ot, st = canonical_bessel_order(order_theta)
        pkey = (ex.exponents.tobytes(), int(ox), int(ot), theta.tobytes(), x.tobytes())
        basis = self._prepared_two_bessel.get(pkey)
        if basis is None:
            basis = self.weber.prepared_basis(ex.exponents, ox, ot, x, theta)
            self._prepared_two_bessel[pkey] = basis
        coeff = ex.coefficients * float(chi) ** (-ex.exponents)
        return (sx * st) * np.einsum('j,jtx->tx', coeff, basis, optimize=True)

    def _double(self, ex, order_x, order_theta, chi):
        return self._double_at(ex, order_x, order_theta, chi, self.radial_grid.x, self.grid.theta_fft)


    def _double_leg_at(self, term, leg, ex, order_x, order_theta, z, chi, x, theta):
        """Two-Bessel transform with canonicalized coincident correction.

        Phase 10 caches the *physical* coincident transform by radial identity
        and canonical Bessel orders.  Hence conjugate / +/-k terms sharing the
        same radial function no longer repeat the 2048-point direct quadrature.
        """
        x = np.asarray(x, dtype=float)
        theta = np.asarray(theta, dtype=float)
        ox, sx = canonical_bessel_order(order_x)
        ot, st = canonical_bessel_order(order_theta)
        sign = sx * st
        if hasattr(term, "model") and getattr(term.model, "has_factorized_growth", False):
            radial_id = (id(term.model), self._factorized_state_token(term.model), int(term.power_shifts[leg]), int(leg))
        else:
            radial_id = (id(term), int(leg), float(z))
        ckey = (radial_id, self._expansion_cache_token(ex), int(ox), int(ot), float(chi),
                theta.tobytes(), x.tobytes())
        cached = self._prepared_double_contact.get(ckey)
        if cached is not None:
            return sign * cached
        # Construct the canonical positive-order analytic result first.
        out = self._double_at(ex, ox, ot, chi, x, theta)
        matches = np.argwhere(np.isclose(theta[:, None], x[None, :], rtol=1e-13, atol=0.0))
        if matches.size:
            f = (term.f1, term.f2, term.f3)[leg]
            nq = max(2048, 64 * int(self.config.slepian.n_fftlog))
            kval = np.geomspace(self.config.slepian.k_min, self.config.slepian.k_max, nq)
            radial = np.asarray(f(kval, z), dtype=complex)
            ell = float(chi) * kval
            base = float(chi) ** 2 * kval * radial
            for it, ix in matches:
                integrand = base * jv(int(ox), ell * x[ix]) * jv(int(ot), ell * theta[it])
                out[it, ix] = np.trapezoid(integrand, kval)
        self._prepared_double_contact[ckey] = out
        self._phase10_counters["double_contact_builds"] += 1
        return sign * out

    @staticmethod
    def _delta_sign(order_x, order_theta):
        from .weber import canonical_bessel_order
        ax, sx = canonical_bessel_order(order_x)
        at, st = canonical_bessel_order(order_theta)
        return sx * st if ax == at else None

    def _term_mode(self, term, sigma, k, z, chi):
        eff = as_effective_spin_triple(sigma)
        m, n = eff.bessel_orders(float(k))
        n1, n2, n3 = term.validated_angular_orders()
        p = n2 + eff.sigma2 - m
        q = n3 + eff.sigma3 - n
        e1 = self._expansion(term, 0, z, chi)
        e2 = self._expansion(term, 1, z, chi)
        e3 = self._expansion(term, 2, z, chi)

        # A k-independent radial leg with equal (up to J_-n=(-1)^n J_n)
        # Bessel orders is distributional: integral dl l J_a(lx)J_a(ltheta)
        # = delta(x-theta)/x.  Collapse that leg analytically instead of
        # asking an ordinary Weber table to represent a Dirac delta.
        c2 = self._constant_value(term, 1, z)
        sign2 = self._delta_sign(p, m) if c2 is not None else None
        c3 = self._constant_value(term, 2, z)
        sign3 = self._delta_sign(q, n) if c3 is not None else None
        theta = self.grid.theta_fft
        if sign2 is not None:
            R1t = self._single_at(e1, n1 + eff.sigma1, chi, theta)
            R3 = self._double_leg_at(term, 2, e3, q, n, z, chi, theta, theta)  # [theta2, theta1]
            value = c2 * sign2 * R1t[:, None] * R3.T
        elif sign3 is not None:
            R1t = self._single_at(e1, n1 + eff.sigma1, chi, theta)
            R2 = self._double_leg_at(term, 1, e2, p, m, z, chi, theta, theta)  # [theta1, theta2]
            value = c3 * sign3 * R2 * R1t[None, :]
        else:
            R1 = self._single(e1, n1 + eff.sigma1, chi)
            R2 = self._double(e2, p, m, chi)
            R3 = self._double(e3, q, n, chi)
            diag = self.radial_grid.weights * self.radial_grid.x * R1
            value = np.einsum('ix,x,jx->ij', R2, diag, R3, optimize=True)
        pref = ((-1j) ** eff.Sigma) / (2.0 * np.pi) ** 2
        return pref * term.c(z) * value


    def _factorized_state_token(self, model):
        """Small content fingerprint for model-dependent radial caches.

        The fingerprint is recomputed once per public calculation, not once per
        physical term.  This preserves setter-driven correctness without a
        version-number protocol and removes a significant warm-path overhead.
        """
        mid = id(model)
        if mid in self._state_token_memo:
            return self._state_token_memo[mid]
        p0, growth = model.factorized_linear_power()
        kp = np.geomspace(self.config.slepian.k_min, self.config.slepian.k_max, 9)
        if self.projector is not None:
            zp = np.asarray(self.projector.z, dtype=float)
        else:
            zp = np.asarray([0.0, 0.5, 1.0], dtype=float)
        h = hashlib.blake2b(digest_size=16)
        for arr in (np.asarray(p0(kp), dtype=complex), np.asarray(growth(zp), dtype=complex)):
            aa = np.ascontiguousarray(arr)
            h.update(str(aa.shape).encode()); h.update(aa.view(np.uint8))
        token = h.digest()
        self._state_token_memo[mid] = token
        return token

    def _factorized_expansion(self, term, leg):
        """Return the explicit P0 FFTLog expansion for one SPT radial leg."""
        if leg == getattr(term, "other_leg", -1):
            return FFTLogExpansion(
                k=np.asarray([self.config.slepian.k_min]),
                coefficients=np.asarray([1.0 + 0.0j]),
                exponents=np.asarray([0.0 + 0.0j]),
                bias=0.0,
            )
        model = term.model
        p0, _ = model.factorized_linear_power()
        state = self._factorized_state_token(model)
        key = ("factorized-p0", id(model), state)
        base = self._fftlog_cache.get(key)
        if base is None:
            base = decompose_log_powerlaw(
                p0,
                k_min=self.config.slepian.k_min,
                k_max=self.config.slepian.k_max,
                n=self.config.slepian.n_fftlog,
                bias=self.config.slepian.fftlog_bias,
            )
            self._fftlog_cache[key] = base
        shift = int(term.power_shifts[leg])
        skey = ("factorized-shift", id(model), state, shift)
        shifted = self._factorized_shift_cache.get(skey)
        if shifted is None:
            shifted = FFTLogExpansion(
                k=base.k,
                coefficients=base.coefficients,
                exponents=base.exponents + shift,
                bias=base.bias + shift,
            )
            self._factorized_shift_cache[skey] = shifted
            self._phase10_counters["shifted_expansion_builds"] += 1
        return shifted

    @staticmethod
    def _expansion_cache_token(ex):
        return (ex.exponents.tobytes(), ex.coefficients.tobytes())

    def _single_basis(self, ex, order, x):
        x = np.asarray(x, dtype=float)
        canon, sign = canonical_bessel_order(order)
        key = (self._expansion_cache_token(ex), int(canon), x.tobytes())
        basis = self._prepared_single_basis.get(key)
        if basis is None:
            basis = (
                ex.coefficients[:, None]
                * single_bessel_factor(ex.exponents, canon)[:, None]
                * x[None, :] ** (-ex.exponents[:, None] - 2.0)
            )
            self._prepared_single_basis[key] = basis
            self._phase10_counters["single_basis_builds"] += 1
        return sign * basis

    def _double_basis(self, ex, order_x, order_theta, x, theta):
        theta = np.asarray(theta, dtype=float)
        x = np.asarray(x, dtype=float)
        ox, sx = canonical_bessel_order(order_x)
        ot, st = canonical_bessel_order(order_theta)
        key = (self._expansion_cache_token(ex), int(ox), int(ot), theta.tobytes(), x.tobytes())
        weighted = self._prepared_double_basis.get(key)
        if weighted is None:
            # Store only the canonical positive-order object.  J_-n=(-1)^n J_n
            # signs are cheap assembly factors, so +k/-k can share this array.
            pkey = ("los-canonical", ex.exponents.tobytes(), int(ox), int(ot), theta.tobytes(), x.tobytes())
            basis = self._prepared_two_bessel.get(pkey)
            if basis is None:
                basis = self.weber.prepared_basis(ex.exponents, ox, ot, x, theta)
                self._prepared_two_bessel[pkey] = basis
            weighted = ex.coefficients[:, None, None] * basis
            self._prepared_double_basis[key] = weighted
            self._phase10_counters["double_basis_builds"] += 1
        return (sx * st) * weighted

    def _compiled_plan(self, sigma, k):
        key = (tuple(int(v) for v in sigma), float(k))
        plan = self._compiled_mode_plans.get(key)
        if plan is None:
            eff = as_effective_spin_triple(sigma)
            plan = compile_mode_plan(self.terms, sigma=sigma, k=float(k), effective_spin=eff)
            self._compiled_mode_plans[key] = plan
            self._phase10_counters["compiled_plan_builds"] += 1
        return plan

    def phase10_cache_stats(self):
        """Return production grouping/cache diagnostics for validation/benchmarks."""
        out = dict(self._phase10_counters)
        out.update({
            "compiled_mode_plans": len(self._compiled_mode_plans),
            "single_basis_cache": len(self._prepared_single_basis),
            "double_basis_cache": len(self._prepared_double_basis),
            "double_contact_cache": len(self._prepared_double_contact),
            "shifted_expansion_cache": len(self._factorized_shift_cache),
            "contact_diag_cache": len(self._contact_diag_cache),
            "general_node_mode_cache": len(self._general_node_mode_cache),
            "prepared_los_moment_cache": len(self._prepared_los_moment_cache),
            "weber_tables": self.weber.n_tables,
            "weber_hypergeom_evaluations": self.weber.hypergeom_evaluations,
        })
        return out

    def _double_basis_with_contact(self, term, leg, ex, order_x, order_theta, x, theta):
        """Mode-resolved basis, using direct quadrature only at x=theta contacts.

        The contact replacement is applied to the *summed* physical radial
        transform in the fixed-z path.  A mode-resolved contact split is not
        unique.  Phase 8 therefore keeps the analytic Weber modes here and
        relies on the exact delta collapse for constant legs; convergence of
        coincident non-constant kernels remains covered by the Phase-7 tests.
        """
        return self._double_basis(ex, order_x, order_theta, x, theta)

    @staticmethod
    def _coefficient_factor(term):
        base = complex(getattr(term, "coefficient", 1.0))
        if base == 0:
            return lambda z: np.ones_like(np.asarray(z, dtype=float), dtype=complex)
        return lambda z: np.asarray(term.c(z), dtype=complex) / base

    def _prepared_los_moments(self, rule, term, ea, eb):
        """Cache the assembled LOS moment matrix by mathematical identity."""
        coeff = self._coefficient_factor(term)
        z = np.asarray(rule.projector.z, dtype=float)
        amp = np.ascontiguousarray(np.asarray(coeff(z), dtype=complex))
        h = hashlib.blake2b(amp.view(np.uint8), digest_size=12).digest()
        key = (id(rule), ea.tobytes(), eb.tobytes(), h)
        out = self._prepared_los_moment_cache.get(key)
        if out is None:
            out = rule.prepare(ea, eb, coefficient_factor=coeff)
            self._prepared_los_moment_cache[key] = out
        return out

    def _term_mode_los_factorized(self, term, sigma, k, rule):
        """LOS-integrated one-term mode using Mellin exponent-sum moments."""
        eff = as_effective_spin_triple(sigma)
        m, n = eff.bessel_orders(float(k))
        n1, n2, n3 = term.validated_angular_orders()
        p = n2 + eff.sigma2 - m
        q = n3 + eff.sigma3 - n
        e1 = self._factorized_expansion(term, 0)
        e2 = self._factorized_expansion(term, 1)
        e3 = self._factorized_expansion(term, 2)
        ex = (e1, e2, e3)
        paired = tuple(int(i) for i in term.paired_legs)
        moments = self._prepared_los_moments(
            rule, term, ex[paired[0]].exponents, ex[paired[1]].exponents
        )
        M = moments.values
        theta = self.grid.theta_fft
        pref = ((-1j) ** eff.Sigma) / (2.0 * np.pi) ** 2
        coeff0 = complex(term.coefficient)

        c2 = 1.0 if term.other_leg == 1 else None
        sign2 = self._delta_sign(p, m) if c2 is not None else None
        c3 = 1.0 if term.other_leg == 2 else None
        sign3 = self._delta_sign(q, n) if c3 is not None else None

        if sign2 is not None:
            # B31: paired legs are (2,0).  Reorder M to (leg0, leg2).
            A1 = self._single_basis(e1, n1 + eff.sigma1, theta)       # [a,i]
            A3 = self._double_basis_with_contact(term, 2, e3, q, n, theta, theta)  # [b,j,i]
            M03 = M.T if paired == (2, 0) else M
            value = sign2 * np.einsum('ai,bji,ab->ij', A1, A3, M03, optimize=True)
        elif sign3 is not None:
            # B12: paired legs are (0,1).
            A1 = self._single_basis(e1, n1 + eff.sigma1, theta)       # [a,j]
            A2 = self._double_basis_with_contact(term, 1, e2, p, m, theta, theta)  # [b,i,j]
            M01 = M if paired == (0, 1) else M.T
            value = sign3 * np.einsum('aj,bij,ab->ij', A1, A2, M01, optimize=True)
        else:
            # General path, including non-delta constant-leg order combinations.
            x = self.radial_grid.x
            A1 = self._single_basis(e1, n1 + eff.sigma1, x)
            A2 = self._double_basis(e2, p, m, x, theta)
            A3 = self._double_basis(e3, q, n, x, theta)
            wx = self.radial_grid.weights * x
            if term.other_leg == 2:  # paired 0,1; e3 has one mode
                value = np.einsum('ax,bix,cjx,ab,x->ij', A1, A2, A3, M, wx, optimize=True)
            elif term.other_leg == 1:  # paired 2,0; e2 has one mode
                M02 = M.T if paired == (2, 0) else M
                value = np.einsum('ax,bix,cjx,ac,x->ij', A1, A2, A3, M02, wx, optimize=True)
            else:
                raise NotImplementedError("Phase-8 SPT LOS rule expects exactly two linear-power legs")
        result = pref * coeff0 * value
        # The constant-leg delta collapse evaluates the remaining Weber kernel
        # at x=theta on the theta1=theta2 diagonal.  Phase 7 established that
        # term-by-term analytic continuation misses a contact contribution at
        # this coincident geometry.  Preserve the Mellin-moment route on the
        # full off-diagonal plane, but replace the diagonal by direct LOS
        # integration of the already validated fixed-z term transform.
        if sign2 is not None or sign3 is not None:
            pjt = rule.projector
            weight = np.asarray(pjt.los_weight(rule.sample_combination), dtype=complex)
            diag_nodes = self._contact_diag_nodes(term, sigma, float(k))
            diag_los = np.trapezoid(diag_nodes * weight[:, None], pjt.chi, axis=0)
            ii = np.diag_indices_from(result)
            result[ii] = diag_los
        return result, moments

    def _contact_diag_nodes(self, term, sigma, k):
        """Return fixed-z contact diagonals, cached independently of LOS sample.

        These radial data are the expensive part of the Phase-8/9 diagonal
        correction but do not depend on sample weights.  The key includes a
        content fingerprint of factorized model physics and the term coefficient
        values on projector nodes, so model updates cannot reuse stale data.
        """
        pjt = self.projector
        model = term.model
        state = self._factorized_state_token(model)
        cvals = np.ascontiguousarray(np.asarray(term.c(np.asarray(pjt.z, dtype=float)), dtype=complex))
        ch = hashlib.blake2b(cvals.view(np.uint8), digest_size=12).digest()
        key = (id(term), tuple(int(v) for v in sigma), float(k), state, ch,
               np.asarray(pjt.z, dtype=float).tobytes(), np.asarray(pjt.chi, dtype=float).tobytes())
        nodes = self._contact_diag_cache.get(key)
        if nodes is None:
            nodes = []
            for zz, cc in zip(pjt.z, pjt.chi):
                fixed = self._term_mode(term, sigma, float(k), float(zz), float(cc))
                nodes.append(np.diag(fixed))
            nodes = np.asarray(nodes)
            self._contact_diag_cache[key] = nodes
            self._phase10_counters["contact_diag_builds"] += 1
        return nodes

    def _term_mode_los_factorized_batch(self, term, sigma, k, rule, weight_matrix=None):
        """Batched LOS-integrated one-term mode.

        Returns an array with shape ``(n_sample, n_theta_fft, n_theta_fft)``.
        All radial/Weber objects are sample independent and are built only once.
        """
        eff = as_effective_spin_triple(sigma)
        m, n = eff.bessel_orders(float(k))
        n1, n2, n3 = term.validated_angular_orders()
        p = n2 + eff.sigma2 - m
        q = n3 + eff.sigma3 - n
        e1 = self._factorized_expansion(term, 0)
        e2 = self._factorized_expansion(term, 1)
        e3 = self._factorized_expansion(term, 2)
        ex = (e1, e2, e3)
        paired = tuple(int(i) for i in term.paired_legs)
        moments = self._prepared_los_moments(
            rule, term, ex[paired[0]].exponents, ex[paired[1]].exponents
        )
        M = moments.values  # [sample, a, b]
        theta = self.grid.theta_fft
        pref = ((-1j) ** eff.Sigma) / (2.0 * np.pi) ** 2
        coeff0 = complex(term.coefficient)

        sign2 = self._delta_sign(p, m) if term.other_leg == 1 else None
        sign3 = self._delta_sign(q, n) if term.other_leg == 2 else None

        if sign2 is not None:
            A1 = self._single_basis(e1, n1 + eff.sigma1, theta)
            A3 = self._double_basis_with_contact(term, 2, e3, q, n, theta, theta)
            M03 = np.swapaxes(M, 1, 2) if paired == (2, 0) else M
            value = sign2 * np.einsum('ai,bji,sab->sij', A1, A3, M03, optimize=True)
        elif sign3 is not None:
            A1 = self._single_basis(e1, n1 + eff.sigma1, theta)
            A2 = self._double_basis_with_contact(term, 1, e2, p, m, theta, theta)
            M01 = M if paired == (0, 1) else np.swapaxes(M, 1, 2)
            value = sign3 * np.einsum('aj,bij,sab->sij', A1, A2, M01, optimize=True)
        else:
            x = self.radial_grid.x
            A1 = self._single_basis(e1, n1 + eff.sigma1, x)
            A2 = self._double_basis(e2, p, m, x, theta)
            A3 = self._double_basis(e3, q, n, x, theta)
            wx = self.radial_grid.weights * x
            if term.other_leg == 2:
                value = np.einsum('ax,bix,cjx,sab,x->sij', A1, A2, A3, M, wx, optimize=True)
            elif term.other_leg == 1:
                M02 = np.swapaxes(M, 1, 2) if paired == (2, 0) else M
                value = np.einsum('ax,bix,cjx,sac,x->sij', A1, A2, A3, M02, wx, optimize=True)
            else:
                raise NotImplementedError("Phase-9 SPT LOS rule expects exactly two linear-power legs")
        result = pref * coeff0 * value

        # Contact diagonal: fixed-z radial data are sample independent.  Build
        # them once, then contract all LOS weights as one leading sample axis.
        if sign2 is not None or sign3 is not None:
            pjt = rule.projector
            weights = rule.weight_matrix() if weight_matrix is None else weight_matrix
            diag_nodes = self._contact_diag_nodes(term, sigma, float(k))
            diag_los = np.trapezoid(
                weights[:, :, None] * diag_nodes[None, :, :],
                pjt.chi, axis=1,
            )
            ii = np.diag_indices(result.shape[-1])
            result[:, ii[0], ii[1]] = diag_los
        return result, moments

    def _general_node_mode(self, term, sigma, k, z, chi):
        """Cached fixed-z term mode used by the general-coefficient LOS rule."""
        key = (id(term), tuple(int(v) for v in sigma), float(k), float(z), float(chi))
        out = self._general_node_mode_cache.get(key)
        if out is None:
            out = self._term_mode(term, sigma, float(k), float(z), float(chi))
            self._general_node_mode_cache[key] = out
        return out

    def _mode_los_general_batch(self, sigma, k, rule, *, weight_matrix=None):
        """Exact batched LOS contraction for redshift-dependent radial shapes.

        All physical terms are summed on each projector node before the LOS
        quadrature.  This avoids one quadrature per term and makes sample
        combinations a cheap leading matrix dimension while retaining the
        exact fixed-z Slepian kernel as the reference representation.
        """
        z_nodes, chi_nodes = rule.nodes
        node_values = []
        for z, chi in zip(z_nodes, chi_nodes):
            value = np.zeros(self.grid.shape_theta_fft, dtype=complex)
            for term in self.terms:
                value += self._general_node_mode(term, sigma, float(k), float(z), float(chi))
            node_values.append(value)
        return rule.integrate_node_values(
            np.asarray(node_values), weight_matrix=weight_matrix
        )

    def compute_zetak_los_many(self, *, sample_combinations=None, epsilons=None, epsilon=None, component=None, all_components=False, force=False):
        """Compute a batch of LOS-integrated Slepian ``ZetaKGrid`` objects.

        The expensive radial preparation is shared across all samples.  Only
        LOS weights/moments and the final sample-leading contractions differ.
        """
        if self.projector is None:
            raise RuntimeError("compute_zetak_los_many requires a LineOfSightProjector")
        models = {id(term.model): term.model for term in self.terms if hasattr(term, "model")}
        if len(models) != 1:
            raise RuntimeError("Slepian LOS execution requires terms from one owning model")
        model = next(iter(models.values()))
        los_kind = getattr(model, "slepian_los_kind", None)
        if los_kind == "factorized-growth" and not getattr(model, "has_factorized_growth", False):
            raise RuntimeError("factorized-growth Slepian LOS requires explicit P(k,z)=D(z)^2 P0(k) metadata")
        if los_kind not in {"factorized-growth", "general-coefficient"}:
            raise RuntimeError(f"unsupported or missing Slepian LOS rule kind: {los_kind!r}")
        combos = sample_combinations if sample_combinations is not None else self.sample_combinations
        combos = tuple(None if c is None else tuple(c) for c in combos)
        if not combos:
            raise ValueError("at least one sample combination is required")
        rule_key = (los_kind, combos)
        rule = self._los_rule_cache.get(rule_key)
        if rule is None:
            if los_kind == "factorized-growth":
                rule = FactorizedGrowthBatchMomentRule(model, self.projector, combos)
            else:
                rule = GeneralCoefficientBatchMomentRule(model, self.projector, combos)
            self._los_rule_cache[rule_key] = rule
        grids = {c: ZetaKGrid(spin=self.config.spin, kmax=self.config.kmax, grid=self.grid) for c in combos}
        self._state_token_memo = {}
        weights_batch = rule.weight_matrix()
        t0 = time.perf_counter()
        epss = self._epsilons(epsilons, epsilon, component, all_components)
        unique_counts = []
        for g in grids.values():
            g.active_epsilons = epss
        for eps in epss:
            sigma = self.spin_spec.sigma_from_epsilon(eps)
            eff = as_effective_spin_triple(sigma)
            for kval in eff.k_values(self.config.kmax):
                self._compiled_plan(sigma, float(kval))
                key = self.ZKgrid.key_from_sigma_k(sigma, float(kval))
                if los_kind == "factorized-growth":
                    val = np.zeros((len(combos), *self.grid.shape_theta_fft), dtype=complex)
                    for term in self.terms:
                        termval, prepared = self._term_mode_los_factorized_batch(
                            term, sigma, float(kval), rule, weight_matrix=weights_batch
                        )
                        val += termval
                        unique_counts.append(len(prepared.unique_exponents))
                else:
                    val = self._mode_los_general_batch(
                        sigma, float(kval), rule, weight_matrix=weights_batch
                    )
                alias = (tuple(eps), self.ZKgrid.two_k(float(kval)))
                for isamp, combo in enumerate(combos):
                    grids[combo].modes[key] = ZetaKMode(
                        grid=self.grid, key=key, value=val[isamp],
                        source_k=float(kval), source_sigma=tuple(sigma),
                    )
                    grids[combo].aliases[alias] = key
        self.timings['los_batch_total'] = time.perf_counter() - t0
        self.timings['los_batch_size'] = len(combos)
        self.timings['los_rule_kind'] = los_kind
        self.timings['los_moment_integral_evaluations'] = rule.integral_evaluations
        self.timings['los_unique_exponents_max'] = max(unique_counts, default=0)
        self.timings['weber_hypergeom_evaluations'] = self.weber.hypergeom_evaluations
        self._batch_ZKgrids = grids
        return grids

    def compute_zetak_los(self, *, sample_combination=None, epsilons=None, epsilon=None, component=None, all_components=False, force=False):
        """Backward-compatible single-sample LOS wrapper."""
        combo = sample_combination
        if combo is None and self.sample_combinations:
            combo = self.sample_combinations[0]
        combo = None if combo is None else tuple(combo)
        return self.compute_zetak_los_many(
            sample_combinations=(combo,), epsilons=epsilons, epsilon=epsilon,
            component=component, all_components=all_components, force=force,
        )[combo]

    def _epsilons(self, epsilons=None, epsilon=None, component=None, all_components=False):
        specified = sum(x is not None for x in (epsilons, epsilon, component)) + int(bool(all_components))
        if specified > 1:
            raise ValueError("specify only one epsilon/components selector")
        if epsilons is not None:
            return tuple(tuple(int(e) for e in eps) for eps in epsilons)
        if epsilon is not None:
            return (tuple(int(e) for e in epsilon),)
        if component is not None:
            return (self.spin_spec.component(int(component)).epsilon,)
        if all_components:
            return self.spin_spec.representative_epsilons()
        reps = self.spin_spec.representative_epsilons()
        return (reps[0],)

    def compute_zetak(self, *, z=None, chi=None, epsilons=None, epsilon=None, component=None, all_components=False, force=False):
        if self.projector is not None:
            if z is not None or chi is not None:
                raise ValueError("projector-backed Slepian execution uses LOS execution, not fixed z/chi")
            combos = tuple(self.sample_combinations or (None,))
            if len(combos) > 1:
                return self.compute_zetak_los_many(
                    sample_combinations=combos, epsilons=epsilons, epsilon=epsilon,
                    component=component, all_components=all_components, force=force,
                )
            return self.compute_zetak_los(
                sample_combination=combos[0], epsilons=epsilons, epsilon=epsilon,
                component=component, all_components=all_components, force=force,
            )
        if z is None or chi is None:
            raise ValueError("fixed-z Slepian execution requires both z and chi")
        self._state_token_memo = {}
        t0 = time.perf_counter()
        epss = self._epsilons(epsilons, epsilon, component, all_components)
        self.ZKgrid.active_epsilons = epss
        for eps in epss:
            sigma = self.spin_spec.sigma_from_epsilon(eps)
            eff = as_effective_spin_triple(sigma)
            for k in eff.k_values(self.config.kmax):
                self._compiled_plan(sigma, float(k))
                key = self.ZKgrid.key_from_sigma_k(sigma, float(k))
                if key not in self.ZKgrid.modes or force:
                    val = np.zeros(self.grid.shape_theta_fft, dtype=complex)
                    # Pair conjugate-sensitive terms in their declared order;
                    # np.add.reduce is avoided here so assembly order stays explicit.
                    for term in self.terms:
                        val += self._term_mode(term, sigma, float(k), float(z), float(chi))
                    self.ZKgrid.modes[key] = ZetaKMode(
                        grid=self.grid, key=key, value=val,
                        source_k=float(k), source_sigma=tuple(sigma),
                    )
                self.ZKgrid.aliases[(tuple(eps), self.ZKgrid.two_k(float(k)))] = key
        self.timings['fixed_z_total'] = time.perf_counter() - t0
        self.timings['weber_hypergeom_evaluations'] = self.weber.hypergeom_evaluations
        return self.ZKgrid

    compute_zeta_k = compute_zetak
