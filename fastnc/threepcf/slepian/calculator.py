"""Fixed-redshift, no-LOS Slepian 3PCF calculator"""
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
from .weber import (
    WeberTableCache, ConstantWeberKernel, single_bessel_factor,
    canonical_bessel_order,
)
from .terms import compile_mode_plan
from .los_moments import (
    FactorizedGrowthMomentRule, FactorizedGrowthBatchMomentRule,
    GeneralCoefficientMomentRule, GeneralCoefficientBatchMomentRule,
)


class SlepianThreePCFCalculator:
    """Direct separable ``B -> zeta_k`` transform at one fixed redshift.
    """
    is_operational = True

    def __init__(self, terms, *, config, projector=None, sample_combinations=None):
        self.terms = tuple(terms)
        if not self.terms:
            raise ValueError("SlepianThreePCFCalculator requires at least one term")
        self.config = config
        if self.config.slepian.radial_backend == "integrated":
            raise NotImplementedError(
                "The integrated Slepian kernel API is present, but production "
                "dispatch is not enabled until diagonal and general-spin "
                "rational kernels are validated. Use radial_backend='reference'."
            )
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
        self._los_rule_cache = {}
        self._batch_ZKgrids = None
        self.constructed_bmultipoles = False

    def _constant_value(self, term, leg, z):
        key = (id(term), int(leg), float(z))
        if key in self._constant_cache:
            return self._constant_cache[key]
        f = (term.f1, term.f2, term.f3)[leg]

        # Built-in terms declare exact radial structure.  Prefer that contract
        # to numerical probing so constant/contact behavior is deterministic.
        metadata = None
        radial_metadata = getattr(term, "radial_metadata", None)
        if radial_metadata is not None:
            metadata = radial_metadata(int(leg))
        if metadata is not None and getattr(metadata, "family", None) == "constant":
            value = complex(np.asarray(f(1.0, z), dtype=complex))
            self._constant_cache[key] = value
            return value

        # Compatibility fallback for third-party/legacy Slepian terms that do
        # not yet advertise radial metadata.
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
        return sign * out


    def _constant_support_grid(self, kernel, theta_value):
        """Reference x grid restricted to the analytic constant-Weber support.

        The Heaviside branch point is inserted explicitly.  This avoids the
        saw-tooth error obtained when a moving x=theta discontinuity is sampled
        on a fixed logarithmic radial grid.
        """
        theta_value = float(theta_value)
        x0 = np.asarray(self.radial_grid.x, dtype=float)
        ax = kernel.canonical_order_x
        at = kernel.canonical_order_theta
        if ax == at:
            return np.asarray([theta_value], dtype=float)
        if ax > at:
            tail = x0[x0 > theta_value]
            return np.concatenate(([theta_value], tail))
        head = x0[x0 < theta_value]
        return np.concatenate((head, [theta_value]))

    def _constant_regular_fixed_z(self, term, constant_leg, kernel, e1, e2, e3,
                                  n1, p, m, q, n, eff, z, chi):
        """Branch-aware regular contribution for one constant double-Bessel leg."""
        theta = np.asarray(self.grid.theta_fft, dtype=float)
        out = np.zeros(self.grid.shape_theta_fft, dtype=complex)
        if kernel.canonical_order_x == kernel.canonical_order_theta:
            return out

        if constant_leg == 2:  # B12: leg 3 is constant; output [theta1, theta2]
            for j, th in enumerate(theta):
                x = self._constant_support_grid(kernel, th)
                if x.size < 2:
                    continue
                R1 = self._single_at(e1, n1 + eff.sigma1, chi, x)
                R2 = self._double_leg_at(term, 1, e2, p, m, z, chi, x, theta)
                Rc = kernel.regular(x, np.asarray([th]))[0]
                out[:, j] = np.trapezoid(
                    R2 * (x * R1 * Rc)[None, :], x, axis=-1
                )
            return out

        if constant_leg == 1:  # B31: leg 2 is constant
            for i, th in enumerate(theta):
                x = self._constant_support_grid(kernel, th)
                if x.size < 2:
                    continue
                R1 = self._single_at(e1, n1 + eff.sigma1, chi, x)
                R3 = self._double_leg_at(term, 2, e3, q, n, z, chi, x, theta)
                Rc = kernel.regular(x, np.asarray([th]))[0]
                out[i, :] = np.trapezoid(
                    R3 * (x * R1 * Rc)[None, :], x, axis=-1
                )
            return out

        raise ValueError("constant_leg must be 1 or 2")

    def _constant_regular_los_factorized_batch(
        self, term, constant_leg, kernel, e1, e2, e3, n1, p, m, q, n, eff,
        paired, moment_values,
    ):
        """LOS-moment contraction of the analytic constant-Weber regular part.

        ``moment_values`` already contains the LOS integral over the two
        non-constant Mellin exponents, including growth, coefficient factor,
        LOS weight and chi powers.  The remaining x integral is therefore
        cosmology independent and is performed on a branch-aware grid.
        """
        theta = np.asarray(self.grid.theta_fft, dtype=float)
        nsamp = int(moment_values.shape[0])
        out = np.zeros((nsamp, *self.grid.shape_theta_fft), dtype=complex)
        if kernel.canonical_order_x == kernel.canonical_order_theta:
            return out

        if constant_leg == 2:  # B12, paired=(0,1)
            M01 = moment_values if paired == (0, 1) else np.swapaxes(moment_values, 1, 2)
            for j, th in enumerate(theta):
                x = self._constant_support_grid(kernel, th)
                if x.size < 2:
                    continue
                A1 = self._single_basis(e1, n1 + eff.sigma1, x)
                A2 = self._double_basis(e2, p, m, x, theta)
                Rc = kernel.regular(x, np.asarray([th]))[0]
                gx = np.einsum('ax,bix,sab->six', A1, A2, M01, optimize=True)
                out[:, :, j] = np.trapezoid(
                    gx * (x * Rc)[None, None, :], x, axis=-1
                )
            return out

        if constant_leg == 1:  # B31, paired=(2,0)
            M03 = np.swapaxes(moment_values, 1, 2) if paired == (2, 0) else moment_values
            for i, th in enumerate(theta):
                x = self._constant_support_grid(kernel, th)
                if x.size < 2:
                    continue
                A1 = self._single_basis(e1, n1 + eff.sigma1, x)
                A3 = self._double_basis(e3, q, n, x, theta)
                Rc = kernel.regular(x, np.asarray([th]))[0]
                gx = np.einsum('ax,cjx,sac->sjx', A1, A3, M03, optimize=True)
                out[:, i, :] = np.trapezoid(
                    gx * (x * Rc)[None, None, :], x, axis=-1
                )
            return out

        raise ValueError("constant_leg must be 1 or 2")

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

        c2 = self._constant_value(term, 1, z)
        c3 = self._constant_value(term, 2, z)
        if c2 is not None and c3 is not None:
            raise NotImplementedError(
                "Slepian fixed-z reference path supports at most one constant "
                "double-Bessel leg"
            )

        theta = self.grid.theta_fft
        x = self.radial_grid.x
        wx = self.radial_grid.weights * x

        # A constant leg is a distribution-aware Weber kernel.  For an even
        # canonical order difference the recurrence chain yields an exact
        # Jacobi-polynomial regular part plus a completeness contact term.
        # The regular part is integrated on the reference x grid; the contact
        # part is applied analytically at x=theta.
        if c2 is not None:
            ck = ConstantWeberKernel(p, m)
            if ck.analytic_even_difference:
                value = c2 * self._constant_regular_fixed_z(
                    term, 1, ck, e1, e2, e3, n1, p, m, q, n, eff, z, chi
                )
                if ck.has_contact:
                    R1t = self._single_at(e1, n1 + eff.sigma1, chi, theta)
                    R3t = self._double_leg_at(
                        term, 2, e3, q, n, z, chi, theta, theta
                    )  # [theta2, theta1]
                    value += (
                        c2 * ck.contact_coefficient
                        * R1t[:, None] * R3t.T
                    )
            else:
                # Odd canonical difference has no completeness contact of the
                # constant-leg type.  Retain the ordinary Weber reference path.
                R1 = self._single(e1, n1 + eff.sigma1, chi)
                R2 = self._double(e2, p, m, chi)
                R3 = self._double_leg_at(term, 2, e3, q, n, z, chi, x, theta)
                value = np.einsum('ix,x,jx->ij', R2, wx * R1, R3, optimize=True)
        elif c3 is not None:
            ck = ConstantWeberKernel(q, n)
            if ck.analytic_even_difference:
                value = c3 * self._constant_regular_fixed_z(
                    term, 2, ck, e1, e2, e3, n1, p, m, q, n, eff, z, chi
                )
                if ck.has_contact:
                    R1t = self._single_at(e1, n1 + eff.sigma1, chi, theta)
                    R2t = self._double_leg_at(
                        term, 1, e2, p, m, z, chi, theta, theta
                    )  # [theta1, theta2]
                    value += (
                        c3 * ck.contact_coefficient
                        * R2t * R1t[None, :]
                    )
            else:
                R1 = self._single(e1, n1 + eff.sigma1, chi)
                R2 = self._double_leg_at(term, 1, e2, p, m, z, chi, x, theta)
                R3 = self._double(e3, q, n, chi)
                value = np.einsum('ix,x,jx->ij', R2, wx * R1, R3, optimize=True)
        else:
            R1 = self._single(e1, n1 + eff.sigma1, chi)
            R2 = self._double_leg_at(term, 1, e2, p, m, z, chi, x, theta)
            R3 = self._double_leg_at(term, 2, e3, q, n, z, chi, x, theta)
            value = np.einsum('ix,x,jx->ij', R2, wx * R1, R3, optimize=True)

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
        return (sx * st) * weighted

    def _compiled_plan(self, sigma, k):
        key = (tuple(int(v) for v in sigma), float(k))
        plan = self._compiled_mode_plans.get(key)
        if plan is None:
            eff = as_effective_spin_triple(sigma)
            plan = compile_mode_plan(self.terms, sigma=sigma, k=float(k), effective_spin=eff)
            self._compiled_mode_plans[key] = plan
        return plan

    def _double_basis_with_contact(self, term, leg, ex, order_x, order_theta, x, theta):
        """Mode-resolved basis, using direct quadrature only at x=theta contacts.

        The contact replacement is applied to the *summed* physical radial
        transform in the fixed-z path.  A mode-resolved contact split is not
        unique. 
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
        """LOS-integrated one-term mode using corrected constant-Weber algebra.

        The two non-constant legs still use the exact Mellin exponent-sum LOS
        moments.  If the remaining leg is constant, its Weber transform is
        represented as analytic regular + contact pieces.  The regular
        Heaviside branch is integrated on a branch-aware x grid whose endpoint
        includes x=theta exactly; the contact is collapsed analytically.
        """
        batch_rule = FactorizedGrowthBatchMomentRule(
            rule.model, rule.projector, (rule.sample_combination,)
        )
        value, prepared = self._term_mode_los_factorized_batch(
            term, sigma, k, batch_rule, weight_matrix=batch_rule.weight_matrix()
        )
        return value[0], PreparedLOSMoments(
            exponent_sums=prepared.exponent_sums,
            values=prepared.values[0],
            unique_exponents=prepared.unique_exponents,
        )

    def _contact_diag_nodes(self, term, sigma, k):
        """Return fixed-z contact diagonals, cached independently of LOS sample.
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
        return nodes

    def _term_mode_los_factorized_batch(self, term, sigma, k, rule, weight_matrix=None):
        """Batched factorized-growth LOS mode with distribution-aware constants.

        This is the production correctness path for SPT.  It retains the
        exponent-sum LOS moment acceleration but replaces the old
        equal-order-only ``_delta_sign`` shortcut by ``ConstantWeberKernel``.
        Hence spin-shifted families such as J_(1+k) J_(1-k) contribute both
        their Jacobi-polynomial regular part and their recurrence-generated
        contact term.
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
        theta = np.asarray(self.grid.theta_fft, dtype=float)
        pref = ((-1j) ** eff.Sigma) / (2.0 * np.pi) ** 2
        coeff0 = complex(term.coefficient)

        other = int(term.other_leg)
        if other == 1:
            ck = ConstantWeberKernel(p, m)
        elif other == 2:
            ck = ConstantWeberKernel(q, n)
        else:
            raise NotImplementedError(
                "factorized-growth SPT LOS expects exactly one constant double-Bessel leg"
            )

        if ck.analytic_even_difference:
            value = self._constant_regular_los_factorized_batch(
                term, other, ck, e1, e2, e3, n1, p, m, q, n, eff,
                paired, M,
            )

            if ck.has_contact:
                if other == 1:  # B31, collapse x=theta1
                    A1 = self._single_basis(e1, n1 + eff.sigma1, theta)
                    A3 = self._double_basis_with_contact(
                        term, 2, e3, q, n, theta, theta
                    )
                    M03 = np.swapaxes(M, 1, 2) if paired == (2, 0) else M
                    value += ck.contact_coefficient * np.einsum(
                        'ai,cji,sac->sij', A1, A3, M03, optimize=True
                    )
                else:  # B12, collapse x=theta2
                    A1 = self._single_basis(e1, n1 + eff.sigma1, theta)
                    A2 = self._double_basis_with_contact(
                        term, 1, e2, p, m, theta, theta
                    )
                    M01 = M if paired == (0, 1) else np.swapaxes(M, 1, 2)
                    value += ck.contact_coefficient * np.einsum(
                        'aj,bij,sab->sij', A1, A2, M01, optimize=True
                    )
        else:
            # Odd canonical order difference has no completeness contact.
            # The constant leg is an ordinary Weber kernel, so the historical
            # Mellin-basis x integral is valid.
            x = np.asarray(self.radial_grid.x, dtype=float)
            A1 = self._single_basis(e1, n1 + eff.sigma1, x)
            A2 = self._double_basis(e2, p, m, x, theta)
            A3 = self._double_basis(e3, q, n, x, theta)
            wx = self.radial_grid.weights * x
            if other == 2:
                M01 = M if paired == (0, 1) else np.swapaxes(M, 1, 2)
                value = np.einsum(
                    'ax,bix,cjx,sab,x->sij', A1, A2, A3, M01, wx,
                    optimize=True,
                )
            else:
                M03 = np.swapaxes(M, 1, 2) if paired == (2, 0) else M
                value = np.einsum(
                    'ax,bix,cjx,sac,x->sij', A1, A2, A3, M03, wx,
                    optimize=True,
                )

        result = pref * coeff0 * value

        # On the exact theta1=theta2 diagonal the non-constant companion Weber
        # is evaluated at its own branch point.  Keep the already validated
        # fixed-z coincident correction there and integrate only that diagonal
        # over LOS.  This is cheap compared with replacing the whole plane by
        # node-by-node fixed-z transforms.
        if ck.has_contact:
            pjt = rule.projector
            weights = rule.weight_matrix() if weight_matrix is None else np.asarray(weight_matrix, dtype=complex)
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
