import numpy as np

from .fftlog import fftlog


class RadialFFTLogExpansion(fftlog):
    """FFTLog radial expansion with reusable FFT grid and angular kernels."""

    def __init__(
        self,
        x_fft,
        fx=None,
        nu=1.1,
        N_extrap_low=0,
        N_extrap_high=0,
        c_window_width=0.25,
        N_pad=0,
        *,
        r_grid=None,
        n_r=512,
        r_max=1.0 - 1.0e-6,
        n_phi=256,
        mode_block=16,
        r_decimals=None,
        bounds_error=True,
    ):
        self.x_fft_input = np.asarray(x_fft, dtype=float)
        if self.x_fft_input.ndim != 1:
            raise ValueError("x_fft must be one-dimensional.")
        if np.any(self.x_fft_input <= 0.0):
            raise ValueError("x_fft must be positive.")
        if np.any(np.diff(self.x_fft_input) <= 0.0):
            raise ValueError("x_fft must be strictly increasing.")

        dummy_fx = np.ones_like(self.x_fft_input, dtype=float)

        super().__init__(
            x=self.x_fft_input,
            fx=dummy_fx,
            nu=nu,
            N_extrap_low=N_extrap_low,
            N_extrap_high=N_extrap_high,
            c_window_width=c_window_width,
            N_pad=N_pad,
        )

        self.bounds_error = bool(bounds_error)

        if r_grid is None:
            u = np.linspace(0.0, 1.0, int(n_r))
            r_grid = r_max * (1.0 - (1.0 - u) ** 2)

        self.r_grid = np.asarray(r_grid, dtype=float)
        if self.r_grid.ndim != 1:
            raise ValueError("r_grid must be one-dimensional.")
        if np.any(self.r_grid < 0.0) or np.any(self.r_grid >= 1.0):
            raise ValueError("r_grid must satisfy 0 <= r < 1.")
        if np.any(np.diff(self.r_grid) <= 0.0):
            raise ValueError("r_grid must be strictly increasing.")

        self.n_phi = int(n_phi)
        self.mode_block = int(mode_block)
        self.r_decimals = r_decimals

        self._kernel_bar_table = {}
        self._quad_cache = {}

        self._current_input_x = None
        self._current_input_fx = None

        if fx is None:
            self.fx = np.zeros_like(self.x, dtype=float)
            self.c_m = np.zeros_like(self.c_m, dtype=complex)
        else:
            self.set_function(self.x_fft_input, fx, bounds_error=False)

    # ------------------------------------------------------------------
    # FFTLog coefficients
    # ------------------------------------------------------------------
    @property
    def q(self):
        return self.nu

    @property
    def z_m(self):
        return self.nu + 1j * self.eta_m

    @property
    def n_mode(self):
        return self.z_m.size

    def _interp_to_fft_grid(self, x, fx, *, bounds_error=None):
        x = np.asarray(x, dtype=float)
        fx = np.asarray(fx)

        if bounds_error is None:
            bounds_error = self.bounds_error

        if x.ndim != 1:
            raise ValueError("x must be one-dimensional.")
        if fx.shape[0] != x.size:
            raise ValueError("fx.shape[0] must match x.size.")
        if np.any(x <= 0.0):
            raise ValueError("x must be positive.")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("x must be strictly increasing.")

        x_target = np.asarray(self.x, dtype=float)

        if bounds_error:
            if x_target[0] < x[0] or x_target[-1] > x[-1]:
                raise ValueError(
                    "x does not cover the internal FFTLog grid. "
                    "Set bounds_error=False to use edge extrapolation."
                )

        logx = np.log(x)
        logx_target = np.log(x_target)

        fx2 = np.reshape(fx, (x.size, -1))
        out = np.empty((x_target.size, fx2.shape[1]), dtype=np.result_type(fx, complex))

        for j in range(fx2.shape[1]):
            y = fx2[:, j]
            if np.iscomplexobj(y):
                out[:, j] = (
                    np.interp(logx_target, logx, np.real(y))
                    + 1j * np.interp(logx_target, logx, np.imag(y))
                )
            else:
                out[:, j] = np.interp(logx_target, logx, y)

        out = out.reshape((x_target.size,) + fx.shape[1:])

        if np.isrealobj(fx):
            out = np.real(out)

        return out

    def _compute_c_m_from_current_fx(self):
        if hasattr(self, "get_c_m"):
            return self.get_c_m()[1]

        fb = self.fx / self.x**self.nu
        return np.fft.rfft(fb)

    def set_function(self, x, fx, *, bounds_error=None):
        """Interpolate f(x) to the fixed FFTLog grid and update coefficients."""
        self._current_input_x = np.asarray(x, dtype=float)
        self._current_input_fx = np.asarray(fx)

        self.fx = self._interp_to_fft_grid(
            self._current_input_x,
            self._current_input_fx,
            bounds_error=bounds_error,
        )

        if self.fx.ndim != 1:
            raise ValueError(
                "set_function currently expects one radial function at a time. "
                "Loop over redshift/functions outside this object."
            )

        self.c_m = self._compute_c_m_from_current_fx()
        return self.c_m

    def coefficients_from_function(self, x, fx, *, bounds_error=None):
        """Return coefficients for f(x) without keeping input arrays."""
        fx_fft = self._interp_to_fft_grid(x, fx, bounds_error=bounds_error)

        old_fx = self.fx
        old_c_m = self.c_m

        self.fx = fx_fft
        c_m = self._compute_c_m_from_current_fx()

        self.fx = old_fx
        self.c_m = old_c_m

        return c_m

    def coefficients_table_from_functions(self, x, fx_table, *, axis=0, bounds_error=None):
        """
        Return coefficients for many functions.

        fx_table is moved so that `axis` is the x-axis.
        Output shape is fx_table.shape without x-axis plus (n_mode,).
        """
        x = np.asarray(x, dtype=float)
        fx_table = np.asarray(fx_table)

        fx_move = np.moveaxis(fx_table, axis, 0)
        if fx_move.shape[0] != x.size:
            raise ValueError("The selected x-axis length must match x.size.")

        rest_shape = fx_move.shape[1:]
        fx_flat = fx_move.reshape((x.size, -1))

        coeffs = []
        for j in range(fx_flat.shape[1]):
            coeffs.append(
                self.coefficients_from_function(
                    x,
                    fx_flat[:, j],
                    bounds_error=bounds_error,
                )
            )

        coeffs = np.asarray(coeffs)
        return coeffs.reshape(rest_shape + (self.n_mode,))

    def _positive_mode_weights(self, c_m=None):
        if c_m is None:
            c_m = self.c_m

        c_m = np.asarray(c_m, dtype=complex)
        if c_m.shape[-1] != self.n_mode:
            raise ValueError(f"Last axis of c_m must have length {self.n_mode}.")

        phase = self.x[0] ** (-1j * self.eta_m)
        return (c_m / float(self.N)) * phase

    def effective_coeff(self, chi, c_m=None, *, already_weighted=False):
        """Return ceff_m = w_m chi^{-z_m}; no shift axis is stored."""
        chi = np.asarray(chi, dtype=float)
        if np.any(chi <= 0.0):
            raise ValueError("chi must be positive.")

        if c_m is None:
            weights = self._positive_mode_weights()
        else:
            c_m = np.asarray(c_m, dtype=complex)
            if c_m.shape[-1] != self.n_mode:
                raise ValueError(f"Last axis of c_m must have length {self.n_mode}.")
            weights = c_m if already_weighted else self._positive_mode_weights(c_m)

        return weights * chi[..., None] ** (-self.z_m)

    def effective_coeff_table(self, chi_grid, c_m_table, *, already_weighted=False):
        """Return ceff table for chi_grid and c_m_table."""
        chi_grid = np.asarray(chi_grid, dtype=float)
        c_m_table = np.asarray(c_m_table, dtype=complex)

        if c_m_table.shape[-1] != self.n_mode:
            raise ValueError(f"Last axis of c_m_table must have length {self.n_mode}.")

        if c_m_table.shape[:-1] != chi_grid.shape:
            raise ValueError("c_m_table.shape[:-1] must match chi_grid.shape.")

        return self.effective_coeff(
            chi_grid,
            c_m=c_m_table,
            already_weighted=already_weighted,
        )

    # ------------------------------------------------------------------
    # Kbar table
    # ------------------------------------------------------------------
    def _leggauss_0_pi(self, n_phi=None):
        if n_phi is None:
            n_phi = self.n_phi
        n_phi = int(n_phi)

        if n_phi not in self._quad_cache:
            x, w = np.polynomial.legendre.leggauss(n_phi)
            phi = 0.5 * np.pi * (x + 1.0)
            weight = 0.5 * np.pi * w
            cosphi = np.cos(phi)
            self._quad_cache[n_phi] = (phi, weight, cosphi)

        return self._quad_cache[n_phi]

    def _compute_kernel_bar_table(self, L, s):
        L = int(L)
        s = int(s)

        phi, weight, cosphi = self._leggauss_0_pi(self.n_phi)
        cosLphi = np.cos(L * phi)

        r = self.r_grid
        z = self.z_m + s

        pref = (2.0 - (1.0 if L == 0 else 0.0)) / np.pi
        table = np.empty((self.n_mode, r.size), dtype=complex)

        base = 1.0 + 2.0 * r[:, None] * cosphi[None, :] + r[:, None] ** 2
        base = np.maximum(base, np.finfo(float).tiny)
        log_base = np.log(base)

        quad_weight = weight * cosLphi

        for start in range(0, self.n_mode, self.mode_block):
            stop = min(start + self.mode_block, self.n_mode)
            z_block = z[start:stop]

            powers = np.exp(0.5 * log_base[:, :, None] * z_block[None, None, :])
            vals = pref * np.sum(powers * quad_weight[None, :, None], axis=1)

            table[start:stop, :] = vals.T

        return table

    def _get_or_build_kernel_bar_table(self, L, s):
        key = (int(L), int(s))
        if key not in self._kernel_bar_table:
            self._kernel_bar_table[key] = self._compute_kernel_bar_table(L, s)
        return self._kernel_bar_table[key]

    @staticmethod
    def _interp_complex(x, y, xp):
        return (
            np.interp(xp, x, np.real(y))
            + 1j * np.interp(xp, x, np.imag(y))
        )

    def _unique_values(self, values, decimals=None):
        flat = np.asarray(values, dtype=float).ravel()
        key = flat if decimals is None else np.round(flat, int(decimals))
        _, index, inverse = np.unique(key, return_index=True, return_inverse=True)
        return flat[index], inverse

    def _kernel_bar_all_modes_no_unique(self, L, s, r):
        r = np.asarray(r, dtype=float)
        r_eval = np.clip(r, self.r_grid[0], self.r_grid[-1])

        table = self._get_or_build_kernel_bar_table(L, s)

        out = np.empty((r_eval.size, self.n_mode), dtype=complex)
        for imode in range(self.n_mode):
            out[:, imode] = self._interp_complex(
                self.r_grid,
                table[imode],
                r_eval,
            )
        return out

    def kernel_bar(self, L, imode, s, r):
        """Return Kbar_L^{z_imode+s}(r)."""
        imode = int(imode)
        if imode < 0 or imode >= self.n_mode:
            raise IndexError(f"imode={imode} is outside [0, {self.n_mode}).")

        r = np.asarray(r, dtype=float)
        if np.any(r < 0.0):
            raise ValueError("r must satisfy r >= 0.")

        r_eval = np.clip(r, self.r_grid[0], self.r_grid[-1])
        table = self._get_or_build_kernel_bar_table(L, s)
        out = self._interp_complex(self.r_grid, table[imode], r_eval)

        if out.shape == ():
            return out.item()
        return out

    def kernel_bar_all_modes(
        self,
        L,
        s,
        r,
        *,
        use_unique_r=True,
        r_decimals=None,
    ):
        """Return Kbar_L^{z_m+s}(r), shape r.shape + (n_mode,)."""
        r = np.asarray(r, dtype=float)
        if np.any(r < 0.0):
            raise ValueError("r must satisfy r >= 0.")

        if r_decimals is None:
            r_decimals = self.r_decimals

        shape = r.shape
        r_flat = r.ravel()

        if use_unique_r:
            r_unique, inverse = self._unique_values(r_flat, decimals=r_decimals)
            K_unique = self._kernel_bar_all_modes_no_unique(L, s, r_unique)
            K_flat = K_unique[inverse]
        else:
            K_flat = self._kernel_bar_all_modes_no_unique(L, s, r_flat)

        return K_flat.reshape(shape + (self.n_mode,))

    # ------------------------------------------------------------------
    # contractions
    # ------------------------------------------------------------------
    def _contract_positive_modes(self, terms):
        terms = np.asarray(terms, dtype=complex)
        if terms.shape[-1] != self.n_mode:
            raise ValueError(f"Last axis of terms must have length {self.n_mode}.")

        if self.n_mode == 1:
            return terms[..., 0]

        out = terms[..., 0]

        if self.N % 2 == 0 and self.n_mode >= 2:
            if self.n_mode > 2:
                out = out + 2.0 * np.real(np.sum(terms[..., 1:-1], axis=-1))
            out = out + terms[..., -1]
        else:
            out = out + 2.0 * np.real(np.sum(terms[..., 1:], axis=-1))

        return out

    # ------------------------------------------------------------------
    # k-space evaluation
    # ------------------------------------------------------------------
    def kernel_sum(
        self,
        L,
        s,
        k1,
        k2,
        *,
        c_m=None,
        already_weighted=False,
        use_unique_r=True,
        r_decimals=None,
    ):
        """Return sum_m c_m K_L^{z_m+s}(k1,k2)."""
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k1, k2 = np.broadcast_arrays(k1, k2)

        if np.any(k1 <= 0.0) or np.any(k2 <= 0.0):
            raise ValueError("k1 and k2 must be positive.")

        kmax = np.maximum(k1, k2)
        kmin = np.minimum(k1, k2)
        r = kmin / kmax

        Kbar = self.kernel_bar_all_modes(
            L=L,
            s=s,
            r=r,
            use_unique_r=use_unique_r,
            r_decimals=r_decimals,
        )

        z = self.z_m + int(s)
        phase = kmax[..., None] ** z[None, :]

        if c_m is None:
            weights = self._positive_mode_weights()
        else:
            weights = c_m if already_weighted else self._positive_mode_weights(c_m)

        terms = weights[None, :] * phase * Kbar
        return self._contract_positive_modes(terms)

    def kernel_sum_many_shifts(
        self,
        L,
        shifts,
        k1,
        k2,
        *,
        c_m=None,
        already_weighted=False,
        use_unique_r=True,
        r_decimals=None,
    ):
        """Return k-space kernel sums for several shifts."""
        shifts = tuple(int(s) for s in shifts)

        vals = [
            self.kernel_sum(
                L=L,
                s=s,
                k1=k1,
                k2=k2,
                c_m=c_m,
                already_weighted=already_weighted,
                use_unique_r=use_unique_r,
                r_decimals=r_decimals,
            )
            for s in shifts
        ]

        return np.stack(vals, axis=-1)

    # ------------------------------------------------------------------
    # ell-space evaluation
    # ------------------------------------------------------------------
    def kernel_sum_ell_from_ceff(
        self,
        L,
        s,
        ell1,
        ell2,
        chi,
        ceff,
        *,
        use_unique_r=True,
        r_decimals=None,
    ):
        """Evaluate ell-space shifted kernel sum using ceff without s-axis."""
        s = int(s)
        chi = float(chi)

        if chi <= 0.0:
            raise ValueError("chi must be positive.")

        ceff = np.asarray(ceff, dtype=complex)
        if ceff.shape != (self.n_mode,):
            raise ValueError(f"ceff must have shape ({self.n_mode},).")

        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell1, ell2 = np.broadcast_arrays(ell1, ell2)

        if np.any(ell1 <= 0.0) or np.any(ell2 <= 0.0):
            raise ValueError("ell1 and ell2 must be positive.")

        ellmax = np.maximum(ell1, ell2)
        ellmin = np.minimum(ell1, ell2)
        r = ellmin / ellmax

        Kbar = self.kernel_bar_all_modes(
            L=L,
            s=s,
            r=r,
            use_unique_r=use_unique_r,
            r_decimals=r_decimals,
        )

        ellpow = ellmax[..., None] ** self.z_m[None, :]
        terms = ceff[None, :] * ellpow * Kbar
        summed = self._contract_positive_modes(terms)

        return (ellmax**s) * (chi**(-s)) * summed

    def kernel_sum_ell(
        self,
        L,
        s,
        ell1,
        ell2,
        chi,
        *,
        c_m=None,
        ceff=None,
        already_weighted=False,
        use_unique_r=True,
        r_decimals=None,
    ):
        """Evaluate ell-space shifted kernel sum."""
        if ceff is None:
            ceff = self.effective_coeff(
                chi,
                c_m=c_m,
                already_weighted=already_weighted,
            )

        ceff = np.asarray(ceff, dtype=complex)
        if ceff.ndim != 1:
            raise ValueError("ceff must have shape (n_mode,) for one chi.")

        return self.kernel_sum_ell_from_ceff(
            L=L,
            s=s,
            ell1=ell1,
            ell2=ell2,
            chi=chi,
            ceff=ceff,
            use_unique_r=use_unique_r,
            r_decimals=r_decimals,
        )

    def kernel_sum_many_shifts_ell(
        self,
        L,
        shifts,
        ell1,
        ell2,
        chi,
        *,
        c_m=None,
        ceff=None,
        already_weighted=False,
        use_unique_r=True,
        r_decimals=None,
    ):
        """Evaluate ell-space shifted kernel sums for several shifts."""
        shifts = tuple(int(s) for s in shifts)

        if ceff is None:
            ceff = self.effective_coeff(
                chi,
                c_m=c_m,
                already_weighted=already_weighted,
            )

        vals = [
            self.kernel_sum_ell_from_ceff(
                L=L,
                s=s,
                ell1=ell1,
                ell2=ell2,
                chi=chi,
                ceff=ceff,
                use_unique_r=use_unique_r,
                r_decimals=r_decimals,
            )
            for s in shifts
        ]

        return np.stack(vals, axis=-1)

    # ------------------------------------------------------------------
    # fixed ell-grid helpers
    # ------------------------------------------------------------------
    def prepare_ell_grid(self, ell1, ell2, *, r_decimals=None):
        """Precompute geometry for a fixed ell1-ell2 grid."""
        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell1, ell2 = np.broadcast_arrays(ell1, ell2)

        if np.any(ell1 <= 0.0) or np.any(ell2 <= 0.0):
            raise ValueError("ell1 and ell2 must be positive.")

        if r_decimals is None:
            r_decimals = self.r_decimals

        ellmax = np.maximum(ell1, ell2)
        ellmin = np.minimum(ell1, ell2)
        r = ellmin / ellmax

        r_unique, inverse_r = self._unique_values(r.ravel(), decimals=r_decimals)

        return {
            "ellmax": ellmax,
            "r": r,
            "r_unique": r_unique,
            "inverse_r": inverse_r,
            "shape": r.shape,
            "r_decimals": r_decimals,
        }

    def kernel_sum_ell_prepared_from_ceff(self, L, s, geom, chi, ceff):
        """Evaluate ell-space shifted kernel sum on prepared geometry."""
        s = int(s)
        chi = float(chi)

        if chi <= 0.0:
            raise ValueError("chi must be positive.")

        ceff = np.asarray(ceff, dtype=complex)
        if ceff.shape != (self.n_mode,):
            raise ValueError(f"ceff must have shape ({self.n_mode},).")

        ellmax = geom["ellmax"]
        r_unique = geom["r_unique"]
        inverse_r = geom["inverse_r"]
        shape = geom["shape"]

        K_unique = self._kernel_bar_all_modes_no_unique(L, s, r_unique)
        Kbar = K_unique[inverse_r].reshape(shape + (self.n_mode,))

        ellpow = ellmax[..., None] ** self.z_m[None, :]
        terms = ceff[None, :] * ellpow * Kbar
        summed = self._contract_positive_modes(terms)

        return (ellmax**s) * (chi**(-s)) * summed

    def kernel_sum_many_shifts_ell_prepared(
        self,
        L,
        shifts,
        geom,
        chi,
        *,
        c_m=None,
        ceff=None,
        already_weighted=False,
    ):
        """Evaluate several ell-space shifted sums on prepared geometry."""
        shifts = tuple(int(s) for s in shifts)

        if ceff is None:
            ceff = self.effective_coeff(
                chi,
                c_m=c_m,
                already_weighted=already_weighted,
            )

        vals = [
            self.kernel_sum_ell_prepared_from_ceff(
                L=L,
                s=s,
                geom=geom,
                chi=chi,
                ceff=ceff,
            )
            for s in shifts
        ]

        return np.stack(vals, axis=-1)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def reconstruct_powerlaw(self, k, *, c_m=None, already_weighted=False):
        """Reconstruct f(k) from positive-mode power-law expansion."""
        k = np.asarray(k, dtype=float)
        if np.any(k <= 0.0):
            raise ValueError("k must be positive.")

        if c_m is None:
            weights = self._positive_mode_weights()
        else:
            weights = c_m if already_weighted else self._positive_mode_weights(c_m)

        terms = weights[None, :] * k.reshape(-1, 1) ** self.z_m[None, :]
        out = self._contract_positive_modes(terms)
        out = out.reshape(k.shape)

        return np.real_if_close(out, tol=1000)