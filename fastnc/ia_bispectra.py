import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simpson
from scipy.optimize import bisect
from scipy.special import gamma, gammaincc


# from .trigutils import is_cyclic_permutation

# Reusing window functions from halofit.py
def window_tophat(x):
    return 3.0 / x ** 3 * (np.sin(x) - x * np.cos(x))


def window_gaussian(x):
    return np.exp(-0.5 * x ** 2)


def window_gaussian_1deriv(x):
    return x * np.exp(-0.5 * x ** 2)


def window_gaussian_2deriv(x):
    return x ** 2 * np.exp(-0.5 * x ** 2)


class BispectraIA:
    def __init__(self, k=None, pklin=None, z=None, lgr=None, cosmo=None):
        self.set_cosmology(cosmo)
        self.set_lgr(z, lgr)
        self.set_pklin(k, pklin)

    def set_lgr(self, z, lgr):
        self.z = z
        self.lgr = lgr
        self.has_changed = True

    def set_pklin(self, k, pklin):
        self.k = k
        self.pklin = pklin
        self._normalize_pklin()
        self.has_changed = True

    def set_pknl(self, k, pknl):
        self.k = k
        self.pknl = pknl
        self.has_changed = True

    def set_cosmology(self, cosmo):
        self.cosmo = cosmo
        self.has_changed = True

    # def set_IA_param(self, ia_params):
    #     """
    #     dict of z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta
    #     """
    #     assert "z_piv"     in ia_params
    #     assert "A1"        in ia_params
    #     assert "alphaIA"   in ia_params
    #     assert "A2"        in ia_params
    #     assert "alphaIA_2" in ia_params
    #     assert "bias_ta"   in ia_params
    #     self.ia_params = ia_params
    #     self.has_changed = True

    def update(self):
        if self.has_changed:
            self._normalize_pklin()
            self._init_spline()
            self.has_changed = False

    def _normalize_pklin(self):
        if (self.k is not None) and (self.pklin is not None) and (self.cosmo is not None):
            k_interp = np.logspace(-3, 2, 1000)
            Delta = self.get_interpolated_pklin(k_interp) * k_interp ** 3 / 2 / np.pi ** 2
            sigma8temp = self._sigmam(k_interp, Delta, 8.0, window_tophat)
            self.pklin *= (self.cosmo["sigma8"] / sigma8temp) ** 2

    def get_interpolated_pklin(self, k, z=None, ext=True):
        k = np.atleast_1d(k)  # Ensure k is always at least 1D array
        pk = np.zeros_like(k, dtype=float)
        where = (self.k.min() <= k) & (k <= self.k.max())

        if np.any(where):
            logpk = ius(np.log(self.k), np.log(self.pklin))(np.log(k[where]))
            pk[where] = np.exp(logpk)

        if ext:
            where_low = k < self.k.min()
            if np.any(where_low):
                n_low = np.log(self.pklin[1] / self.pklin[0]) / np.log(self.k[1] / self.k[0])
                a_low = self.pklin[0]
                pk[where_low] = a_low * (k[where_low] / self.k[0]) ** n_low

            where_high = k > self.k.max()
            if np.any(where_high):
                n_high = np.log(self.pklin[-1] / self.pklin[-2]) / np.log(self.k[-1] / self.k[-2])
                a_high = self.pklin[-1]
                pk[where_high] = a_high * (k[where_high] / self.k[-1]) ** n_high

        if z is not None:
            pk *= ius(self.z, self.lgr, ext=2)(z) ** 2

        return pk

    def get_interpolated_pknl(self, k, z=None, ext=True):
        k = np.atleast_1d(k)  # Ensure k is always at least 1D array
        pk = np.zeros_like(k, dtype=float)
        where = (self.k.min() <= k) & (k <= self.k.max())

        if np.any(where):
            logpk = ius(np.log(self.k), np.log(self.pknl))(np.log(k[where]))
            pk[where] = np.exp(logpk)

        if ext:
            where_low = k < self.k.min()
            if np.any(where_low):
                n_low = np.log(self.pknl[1] / self.pknl[0]) / np.log(self.k[1] / self.k[0])
                a_low = self.pknl[0]
                pk[where_low] = a_low * (k[where_low] / self.k[0]) ** n_low

            where_high = k > self.k.max()
            if np.any(where_high):
                n_high = np.log(self.pknl[-1] / self.pknl[-2]) / np.log(self.k[-1] / self.k[-2])
                a_high = self.pknl[-1]
                pk[where_high] = a_high * (k[where_high] / self.k[-1]) ** n_high

        if z is not None:
            pk *= ius(self.z, self.lgr, ext=2)(z) ** 2

        return pk

    def _sigmam(self, k, Delta, r, window, extrap=False):
        I2 = simpson(Delta * window(k * r) ** 2, x=np.log(k))
        if extrap:
            n = np.diff(np.log(Delta))[-1] / np.diff(np.log(k))[-1]
            A = Delta[-1] * k[-1] ** (-n)
            tmin = k[-1] * r
            I2 += A * r ** -n * 0.5 * gamma(n / 2) * gammaincc(n / 2, tmin ** 2)
        return I2 ** 0.5

    def F2_tree(self, k1, k2, k3):
        costheta12 = 0.5 * (k3 * k3 - k1 * k1 - k2 * k2) / (k1 * k2)
        return (5. / 7.) + 0.5 * costheta12 * (k1 / k2 + k2 / k1) + (2. / 7.) * costheta12 * costheta12

    def _construct_k_vectors(self, k1_mag, k2_mag, k3_mag):

        original_shape = k1_mag.shape
        k1_mag_flat = k1_mag.ravel()
        k2_mag_flat = k2_mag.ravel()
        k3_mag_flat = k3_mag.ravel()

        k3_vec_flat = np.array([k3_mag_flat, np.zeros_like(k3_mag_flat)])
        cos_alpha = (k3_mag_flat ** 2 + k1_mag_flat ** 2 - k2_mag_flat ** 2) / (2 * k3_mag_flat * k1_mag_flat)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        sin_alpha = np.sqrt(1 - cos_alpha ** 2)

        k1_vec_flat = np.array([-k1_mag_flat * cos_alpha, -k1_mag_flat * sin_alpha])
        k2_vec_flat = - k3_vec_flat - k1_vec_flat

        k1_vec = k1_vec_flat.reshape((2,) + original_shape)
        k2_vec = k2_vec_flat.reshape((2,) + original_shape)
        k3_vec = k3_vec_flat.reshape((2,) + original_shape)

        return k1_vec, k2_vec, k3_vec

    def get_F_00(self, k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ssg'):

        k1_vec, k2_vec, k3_vec = self._construct_k_vectors(k1_mag, k2_mag, k3_mag)

        if mode == 'sgs' or mode == 'gss':
            kernel = cg1 * 2 * self.F2_tree(k1_mag, k2_mag, k3_mag) * PL1 * PL2

        elif mode == 'ggs':
            kernel = cg1 ** 2 * 2 * self.F2_tree(k1_mag, k2_mag, k3_mag) * PL1 * PL2

        elif mode == 'sgg' or mode == 'gsg':
            k1x, k1y = k1_vec[0] / k1_mag, k1_vec[1] / k1_mag
            k2x, k2y = k2_vec[0] / k2_mag, k2_vec[1] / k2_mag
            k1_dot_k2 = (k1x * k2x + k1y * k2y)
            kernel = cg1 * 2 * PL1 * PL2 * (
                        cg1 * self.F2_tree(k1_mag, k2_mag, k3_mag) + (cg2_2 + cg2_3) * k1_dot_k2 ** 2 + cg2_3)

        elif mode == 'sss':
            kernel = 2 * self.F2_tree(k1_mag, k2_mag, k3_mag) * PL1 * PL2

        return kernel

    def get_F_20(self, k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ssg'):

        k1_vec, k2_vec, k3_vec = self._construct_k_vectors(k1_mag, k2_mag, k3_mag)

        k1x, k1y = k1_vec[0] / k1_mag, k1_vec[1] / k1_mag
        k2x, k2y = k2_vec[0] / k2_mag, k2_vec[1] / k2_mag
        k1_dot_k2 = k1x * k2x + k1y * k2y

        term1 = cg1 * self.F2_tree(k1_mag, k2_mag, k3_mag)
        term2 = cg2_2 * (0.25 * (k1x ** 2 - k1y ** 2 + k2x ** 2 - k2y ** 2) + 0.5 * k1_dot_k2 * (k1x * k2x))
        term3 = cg2_3 * (0.25 * (2 * k1x ** 2 - k1y ** 2 + 2 * k2x ** 2 - k2y ** 2))
        #term4 = cg2_1 * (5/7* (1-k1_dot_k2**2) + 0.25 * (k1x ** 2 - k1y ** 2 + k2x ** 2 - k2y ** 2) + 0.5 * k1_dot_k2 * (k1x * k2x))
        term4 = cg2_1 * (5 / 7 * (1 - k1_dot_k2**2))

        if mode == 'ssg':
            result = 2 * np.sqrt(2 / 3) * PL1 * PL2 * (term1 + term2 + term3 + term4)

        elif mode == 'gsg':
            result = cg1 * 2 * np.sqrt(2 / 3) * PL1 * PL2 * (term1 + term2 + term3 + term4)

        elif mode == 'ggs':
            result = 2 * np.sqrt(2 / 3) * cg1 * PL1 * PL2 * term1

        elif mode == 'sgg':
            result = cg1 * 2 * np.sqrt(2 / 3) * PL1 * PL2 * (term1 + term2 + term3 + term4)

        elif mode == 'ggg':
            result = cg1 ** 2 * 2 * np.sqrt(2 / 3) * PL1 * PL2 * (term1 + term2 + term3 + term4)

        return result

    # def clockwise_sign(self, x1, x2, x3):
    #     # check the (d1 > d2 > d3) triangle is clockwise or not
    #     idx = np.argsort([x1, x2, x3], axis=0).T
    #     clk = [is_cyclic_permutation(_idx) for _idx in idx]
    #     sign = np.ones_like(clk, dtype=int)
    #     sign[np.logical_not(clk)] = -1
    #     return sign

    def get_F_21(self, k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ssg'):

        k1_vec, k2_vec, k3_vec = self._construct_k_vectors(k1_mag, k2_mag, k3_mag)

        k1x, k1y = k1_vec[0] / k1_mag, k1_vec[1] / k1_mag
        k2x, k2y = k2_vec[0] / k2_mag, k2_vec[1] / k2_mag

        # sign = self.clockwise_sign(k1_mag, k2_mag, k3_mag)

        partial = -1 * PL1 * PL2 * (cg2_2 + cg2_3) * (k1x * k1y + k2x * k2y)
        #partial = -1 * PL1 * PL2 * (cg2_1 + cg2_2 + cg2_3) * (k1x * k1y + k2x * k2y)
        # partial = -1 * PL1 * PL2 * (cg2_2 + cg2_3) * np.abs(k1x*k1y + k2x*k2y) * sign

        # print('F_21 called, mode is', mode)

        if mode == 'ssg':
            return partial

        elif mode == 'ggs':
            return 0

        elif mode == 'sgg':
            return cg1 * partial

        elif mode == 'gsg':
            return cg1 * partial

        elif mode == 'ggg':
            return cg1 ** 2 * partial

    def get_F_22(self, k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ssg'):

        k1_vec, k2_vec, k3_vec = self._construct_k_vectors(k1_mag, k2_mag, k3_mag)

        if mode == 'ggs':
            return 0

        k1x, k1y = k1_vec[0] / k1_mag, k1_vec[1] / k1_mag
        k2x, k2y = k2_vec[0] / k2_mag, k2_vec[1] / k2_mag
        k1_dot_k2 = (k1x * k2x + k1y * k2y)
        #partial = 2 * PL1 * PL2 * ((cg2_1 + cg2_2) * (0.5 * k1_dot_k2 * (k1y * k2y)) + cg2_3 * (0.25 * (k1y ** 2 + k2y ** 2)))
        partial = 2 * PL1 * PL2 * (cg2_2 * (0.5 * k1_dot_k2 * (k1y * k2y)) + cg2_3 * (0.25 * (k1y ** 2 + k2y ** 2)))

        if mode == 'ssg':
            result = partial

        elif mode == 'gsg':
            result = cg1 * partial

        elif mode == 'sgg':
            result = cg1 * partial

        elif mode == 'ggg':
            result = cg1 ** 2 * partial

        return result

    def get_delta_K(self, m_val):
        # Kronecker delta definition: zero if m is not zero, and N0^(-1) if m is zero
        if m_val == 0.0:
            return np.sqrt(2 / 3)
        else:
            return 0.0

    def get_B000(self, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3):
        # This is not used, here for completeness
        F0_12 = self.get_F_00(k1_mag, k2_mag, k3_mag, PL1, PL2, 0, 0, 0, mode='sss')
        F0_23 = self.get_F_00(k2_mag, k3_mag, k1_mag, PL2, PL3, 0, 0, 0, mode='sss')
        F0_31 = self.get_F_00(k3_mag, k1_mag, k2_mag, PL3, PL1, 0, 0, 0, mode='sss')
        return F0_12 + F0_23 + F0_31

    def get_B002(self, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2,
                 cg2_3, m_val):
        # Here our convention is to always use alpha, beta, gamma = ssg.

        if m_val == 0:
            F_12_m = self.get_F_20(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ssg')
            delta_K_0m = np.sqrt(2 / 3)
            F0_23 = self.get_F_00(k2_mag, k3_mag, k1_mag, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, mode='sgs')
            F0_31 = self.get_F_00(k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, mode='gss')
            return F_12_m + delta_K_0m * F0_23 + delta_K_0m * F0_31

        if m_val == 1:

            F_12_m = self.get_F_21(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ssg')
            return F_12_m

        elif m_val == 2:
            F_12_m = self.get_F_22(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ssg')
            return F_12_m

        else:
            raise ValueError("m_val must be 0, 1, or 2 for get_B002")

    def get_B022(self, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, m2_val, m3_val):

        delta_K_0m2 = self.get_delta_K(m2_val)
        delta_K_0m3 = self.get_delta_K(m3_val)

        if m3_val == 0:
            F_12_m3 = self.get_F_20(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='sgg')
        elif m3_val == 1:
            F_12_m3 = self.get_F_21(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='sgg')
        elif m3_val == 2:
            F_12_m3 = self.get_F_22(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='sgg')
        else:
            raise ValueError("m3_val must be 0, 1, or 2 for get_B022")

        F0_23 = self.get_F_00(k2_mag, k3_mag, k1_mag, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, mode='ggs')

        if m2_val == 0:
            F_31_m2 = self.get_F_20(k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, mode='gsg')
        elif m2_val == 1:
            F_31_m2 = self.get_F_21(k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, mode='gsg')
        elif m2_val == 2:
            F_31_m2 = self.get_F_22(k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, mode='gsg')
        else:
            raise ValueError("m2_val must be 0, 1, or 2 for get_B022")

        term1 = delta_K_0m2 * F_12_m3
        term2 = delta_K_0m2 * delta_K_0m3 * F0_23
        term3 = delta_K_0m3 * F_31_m2

        return term1 + term2 + term3

    def get_B222(self, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3,
                 m1_val, m2_val, m3_val):

        delta_K_0m1 = self.get_delta_K(m1_val)
        delta_K_0m2 = self.get_delta_K(m2_val)
        delta_K_0m3 = self.get_delta_K(m3_val)

        if m3_val == 0:
            F_12_m3 = self.get_F_20(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')
        elif m3_val == 1:
            F_12_m3 = self.get_F_21(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')
        elif m3_val == 2:
            F_12_m3 = self.get_F_22(k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')
        else:
            raise ValueError("m3_val must be 0, 1, or 2 for get_B222")

        if m1_val == 0:
            F_23_m1 = self.get_F_20(k2_mag, k3_mag, k1_mag, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')
        elif m1_val == 1:

            F_23_m1 = self.get_F_21(k2_mag, k3_mag, k1_mag, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')

        elif m1_val == 2:
            F_23_m1 = self.get_F_22(k2_mag, k3_mag, k1_mag, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')
        else:
            raise ValueError("m1_val must be 0, 1, or 2 for get_B222")

        if m2_val == 0:
            F_31_m2 = self.get_F_20(k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')
        elif m2_val == 1:

            F_31_m2 = self.get_F_21(k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')

        elif m2_val == 2:
            F_31_m2 = self.get_F_22(k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, mode='ggg')
        else:
            raise ValueError("m2_val must be 0, 1, or 2 for get_B222")

        term1 = delta_K_0m1 * delta_K_0m2 * F_12_m3
        term2 = delta_K_0m2 * delta_K_0m3 * F_23_m1
        term3 = delta_K_0m3 * delta_K_0m1 * F_31_m2
        return term1 + term2 + term3

        # ------------------------------------------------------------------
        # BδδE  —  Eq. 51
        # ------------------------------------------------------------------

    def get_B_ddE_gomes(self, k1_mag, k2_mag, k3_mag,
                        PL1, PL2, PL3, C1, C1delta, C2, Ct=0):
        """
        Eq. 51 of Gomes et al. (2026).
        BδδE(k1,k2,k3) = C1 * B^Tree_ddd
            + C2 * PL1 PL2 [ 1/2 (k1h.k2h)(k1hx k2hx - k1hy k2hy)
                             - 1/12 (k1hx^2 - k1hy^2 + k2hx^2 - k2hy^2) ]
            + C1delta * PL1 PL2 [ 1/2 (k1hx^2 - k1hy^2 + k2hx^2 - k2hy^2) ]

        With optional velocity-shear addition from Eq. 62.
        """
        B_tree = self._B_ddd(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3)

        k1hx, k1hy, k2hx, k2hy = self._khat_xy(k1_mag, k2_mag, k3_mag)
        k1dk2 = k1hx * k2hx + k1hy * k2hy
        diff1 = k1hx ** 2 - k1hy ** 2
        diff2 = k2hx ** 2 - k2hy ** 2

        result = (
                C1 * B_tree
                + C2 * PL1 * PL2 * (
                        0.5 * k1dk2 * (k1hx * k2hx - k1hy * k2hy)
                        - (1.0 / 12.0) * (diff1 + diff2)
                )
                + C1delta * PL1 * PL2 * (
                        0.5 * (diff1 + diff2)
                )
        )

        # Velocity-shear extension (Eq. 62)
        if Ct != 0:
            result += -Ct * PL1 * PL2 * (
                    (2.0 / 7.0) * (1.0 - k1dk2 ** 2)
                    + 0.1 * (diff1 + diff2)
                    + 0.2 * k1dk2 * (k1hx * k2hx - k1hy * k2hy)
            )

        return result

'''Alternative functions for debugging that use the direct eqs from Gomes et. al.'''
        # ------------------------------------------------------------------
        # BδEE  —  Eq. 52
        # ------------------------------------------------------------------

    def get_B_dEE_gomes(self, k1_mag, k2_mag, k3_mag,
                        PL1, PL2, PL3, C1, C1delta, C2, Ct=0):
        """
        Eq. 52 of Gomes et al. (2026).
        BδEE(k1,k2,k3) =
            C1^2 * B^Tree_ddd
            + (1/3) C1 C2 * U * PL1 PL2
            + C1 C1delta * U * PL1 PL2
            + C1 C2 * [V + (2/3) W] * PL1 PL3
            - C1 C1delta * W * PL1 PL3

        With optional velocity-shear addition from Eq. 63.
        """
        B_tree = self._B_ddd(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3)
        k1dk2, k1dk3, k2dk3 = self._dot_products(k1_mag, k2_mag, k3_mag)

        U = self._U_aux(k1_mag, k2_mag, k3_mag, k1dk2)
        V = self._V_aux(k1_mag, k2_mag, k3_mag, k1dk2, k1dk3)
        W = self._W_aux(k1_mag, k2_mag, k3_mag, k1dk2)

        result = (
                C1 ** 2 * B_tree
                + (1.0 / 3.0) * C1 * C2 * U * PL1 * PL2
                + C1 * C1delta * U * PL1 * PL2
                + C1 * C2 * (V + (2.0 / 3.0) * W) * PL1 * PL3
                - C1 * C1delta * W * PL1 * PL3
        )

        # Velocity-shear extension (Eq. 63)
        if Ct != 0:
            U2 = self._U2_aux(k1_mag, k2_mag, k3_mag, k1dk2)
            V2 = self._V2_aux(k1_mag, k2_mag, k3_mag, k1dk2, k1dk3)
            result += C1 * Ct * U2 * PL1 * PL2
            result += C1 * Ct * V2 * PL1 * PL3

        return result

        # ------------------------------------------------------------------
        # BEEE  —  Eq. 56
        # ------------------------------------------------------------------

    def get_B_EEE_gomes(self, k1_mag, k2_mag, k3_mag,
                        PL1, PL2, PL3, C1, C1delta, C2, Ct=0):
        """
        Eq. 56 of Gomes et al. (2026).
        BEEE(k1,k2,k3) =
            C1^3 * B^Tree_ddd
            + (1/6) C1^2 C2 * U(k1,k2,k3) * PL1 PL2
            + (1/2) C1^2 C1delta * U(k1,k2,k3) * PL1 PL2
            + C1^2 C2 * [T(k1,k2,k3) + (1/3) W(k1,k2,k3)] * PL1 PL3
            - (1/2) C1^2 C1delta * W(k1,k2,k3) * PL1 PL3
            + C1^2 C2 * [T(k2,k1,k3) + (1/3) W(k1,k2,k3)] * PL2 PL3
            - (1/2) C1^2 C1delta * W(k2,k1,k3) * PL2 PL3

        With optional velocity-shear addition from Eq. 66.
        """
        B_tree = self._B_ddd(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3)
        k1dk2, k1dk3, k2dk3 = self._dot_products(k1_mag, k2_mag, k3_mag)

        U_12 = self._U_aux(k1_mag, k2_mag, k3_mag, k1dk2)

        # W(k1,k2,k3) uses k1_mag, k2_mag, k3_mag, k1dk2
        W_12 = self._W_aux(k1_mag, k2_mag, k3_mag, k1dk2)

        # W(k2,k1,k3) — swap k1<->k2 in W definition
        # W(ka,kb,kc) = -1/(2 kc^2) * [ (ka_hat . kb_hat)^2 (ka^2 + kc^2) - ka^2 ]
        # So W(k2,k1,k3) uses k2_mag as "ka", k1_mag as "kb", k3_mag as "kc",
        # and the dot product k2hat.k1hat = k1dk2
        W_21 = self._W_aux(k2_mag, k1_mag, k3_mag, k1dk2)

        # T(k1,k2,k3) uses k1dk2 and k1dk3
        T_12 = self._T_aux(k1_mag, k2_mag, k3_mag, k1dk2, k1dk3)

        # T(k2,k1,k3) — swap k1<->k2 in T definition
        # T(ka,kb,kc) uses ka_hat.kb_hat and ka_hat.kc_hat
        # So T(k2,k1,k3) needs k2hat.k1hat = k1dk2 and k2hat.k3hat = k2dk3
        T_21 = self._T_aux(k2_mag, k1_mag, k3_mag, k1dk2, k2dk3)

        C1sq = C1 ** 2

        result = (
                C1 ** 3 * B_tree
                + (1.0 / 6.0) * C1sq * C2 * U_12 * PL1 * PL2
                + 0.5 * C1sq * C1delta * U_12 * PL1 * PL2
                + C1sq * C2 * (T_12 + (1.0 / 3.0) * W_12) * PL1 * PL3
                - 0.5 * C1sq * C1delta * W_12 * PL1 * PL3
                + C1sq * C2 * (T_21 + (1.0 / 3.0) * W_12) * PL2 * PL3
                - 0.5 * C1sq * C1delta * W_21 * PL2 * PL3
        )

        # Velocity-shear extension (Eq. 66)
        if Ct != 0:
            U2_12 = self._U2_aux(k1_mag, k2_mag, k3_mag, k1dk2)
            V2_13 = self._V2_aux(k1_mag, k2_mag, k3_mag, k1dk2, k1dk3)
            V2_23 = self._V2_aux(k2_mag, k1_mag, k3_mag, k1dk2, k2dk3)
            result += 0.5 * C1sq * Ct * U2_12 * PL1 * PL2
            result += 0.5 * C1sq * Ct * V2_13 * PL1 * PL3
            result += 0.5 * C1sq * Ct * V2_23 * PL2 * PL3

        return result

        # ------------------------------------------------------------------
        # Velocity-shear auxiliary functions (Eqs. 64-65)
        # ------------------------------------------------------------------

    def _U2_aux(self, k1_mag, k2_mag, k3_mag, k1dk2):
        """
        Eq. 64: U2(k1, k2, k3)
        """
        k1sq = k1_mag ** 2
        k2sq = k2_mag ** 2
        k3sq = k3_mag ** 2
        k1k2 = k1_mag * k2_mag
        k1dk2_sq = k1dk2 ** 2

        numer = (
                7.0 * k1dk2 * (
                k1_mag ** 6 + k2_mag ** 6
                - k3sq * (k1_mag ** 4 + k2_mag ** 4)
                + 9.0 * (k1sq * k2_mag ** 4 + k1_mag ** 4 * k2sq)
                + 2.0 * k1sq * k2sq * k3sq
        )
                + (10.0 + 32.0 * k1dk2_sq) * (
                        k1_mag ** 5 * k2_mag
                        + k2_mag ** 5 * k1_mag
                        + 2.0 * k1_mag ** 3 * k2_mag ** 3
                )
                + 2.0 * k1dk2_sq * (
                        20.0 * k1_mag ** 3 * k2_mag ** 3
                        - 14.0 * k1k2 * k3sq * (k1sq + k2sq)
                )
                + 2.0 * k1sq * k2sq * k1dk2 ** 3 * (
                        22.0 * k1sq + 22.0 * k2sq - k3sq
                )
                + 16.0 * k1dk2 ** 4 * k1_mag ** 3 * k2_mag ** 3
        )
        denom = 70.0 * k1k2 * k3_mag ** 4
        return -numer / denom

    def _V2_aux(self, k1_mag, k2_mag, k3_mag, k1dk2, k1dk3):
        """
        Eq. 65: V2(k1, k2, k3)
        V2 = (2/5) * (k1/k3) * (k1hat.k3hat) * ((k1hat.k2hat)^2 - 1)
             - (1/35) * (5 + 2*(k1hat.k3hat)^2)
        """
        return (
                (2.0 / 5.0) * (k1_mag / k3_mag) * k1dk3 * (k1dk2 ** 2 - 1.0)
                - (1.0 / 35.0) * (5.0 + 2.0 * k1dk3 ** 2)
        )

    '''Here the direct expressions end'''

    '''Main function to get the IA bispectra'''

    def get_ia_bispectra(self, k1_mag, k2_mag, k3_mag, z, z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta, Ct=0, remove_alignment=False,
                         do_non_linear=True):

        # Get C1, C1delta and C2 params
        c1rhocrit = 0.0134
        C1 = -A1 * ((1 + z) / (1 + z_piv)) ** alphaIA * c1rhocrit * self.cosmo['Om0'] / self.z2lgr(z)
        C1delta = bias_ta * C1
        C2 = A2 * ((1 + z) / (1 + z_piv)) ** alphaIA_2 * 5 * c1rhocrit * self.cosmo['Om0'] / self.z2lgr(z) ** 2

        Ct = C1*Ct #only for paper plots, after this we will have to adjust it better

        # Get cg1, cg2_2 and cg2_3 params
        cg1 = 2*C1 # normalization to match the A1 of TATT, according to Eq. 66 of Bakx et al (1) (2025) and Eq. 30 of Bakx et. al. (2) (2025)
        cg2_1 = -2*Ct/5
        cg2_2 = C2 #this is actually cg2_2+cg2_1, which we group together to avoid redundant calculations
        #cg2_3 = 0.5 * (3 * C1delta - C2)
        cg2_3 = C1delta - 2*C2/3

        # Get linear power spectra
        if do_non_linear:
            PL1 = self.get_interpolated_pknl(k1_mag, z)
            PL2 = self.get_interpolated_pknl(k2_mag, z)
            PL3 = self.get_interpolated_pknl(k3_mag, z)
        else:
            PL1 = self.get_interpolated_pklin(k1_mag, z)
            PL2 = self.get_interpolated_pklin(k2_mag, z)
            PL3 = self.get_interpolated_pklin(k3_mag, z)

        # B_delta_delta_E (Eq. 12)
        B_002_0 = self.get_B002(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 0)
        B_002_2 = self.get_B002(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 2)
        B_ddE = 0.5 * (np.sqrt(3 / 2) * B_002_0 - B_002_2)

        # B_E_delta_delta (permutation)
        B_002_0 = self.get_B002(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 0)
        B_002_2 = self.get_B002(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 2)
        B_Edd = 0.5 * (np.sqrt(3 / 2) * B_002_0 - B_002_2)

        # B_delta_E_delta (permutation)
        B_002_0 = self.get_B002(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 0)
        B_002_2 = self.get_B002(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 2)
        B_dEd = 0.5 * (np.sqrt(3 / 2) * B_002_0 - B_002_2)

        # B_delta_delta_B
        B_002_1 = self.get_B002(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 1)
        B_ddB = -B_002_1

        # B_B_delta_delta (permutation)
        B_002_1 = self.get_B002(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 1)
        B_Bdd = -B_002_1

        # B_delta_B_delta (permutation)
        B_002_1 = self.get_B002(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 1)
        B_dBd = -B_002_1

        # B_delta_EE (Eq. 13)
        B_022_00 = self.get_B022(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 0, 0)
        B_022_02_perm1 = self.get_B022(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 0, 2)
        B_022_02_perm2 = self.get_B022(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 2, 0)
        #B_dEE = np.sqrt(3 / 2) / 4 * (np.sqrt(3 / 2) * B_022_00 + B_022_02_perm1 + B_022_02_perm2)
        B_dEE = np.sqrt(3 / 2) / 4 * (np.sqrt(3 / 2) * B_022_00 - B_022_02_perm1 - B_022_02_perm2)

        # B_E_delta_E (permutation)
        B_022_00 = self.get_B022(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 0, 0)
        B_022_02_perm1 = self.get_B022(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 0, 2)
        B_022_02_perm2 = self.get_B022(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 2, 0)
        #B_EdE = np.sqrt(3 / 2) / 4 * (np.sqrt(3 / 2) * B_022_00 + B_022_02_perm1 + B_022_02_perm2)
        B_EdE = np.sqrt(3 / 2) / 4 * (np.sqrt(3 / 2) * B_022_00 - B_022_02_perm1 - B_022_02_perm2)

        # B_EE_delta (permutation)
        B_022_00 = self.get_B022(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 0, 0)
        B_022_02_perm1 = self.get_B022(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 0, 2)
        B_022_02_perm2 = self.get_B022(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 2, 0)
        #B_EEd = np.sqrt(3 / 2) / 4 * (np.sqrt(3 / 2) * B_022_00 + B_022_02_perm1 + B_022_02_perm2)
        B_EEd = np.sqrt(3 / 2) / 4 * (np.sqrt(3 / 2) * B_022_00 - B_022_02_perm1 - B_022_02_perm2)

        # B_delta_BB + permutations
        B_dBB = 0
        B_BdB = 0
        B_BBd = 0

        # B_delta_EB
        B_022_01 = self.get_B022(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 0, 1)
        B_dEB = -np.sqrt(3 / 8) * B_022_01

        # B_delta_BE (permutation)
        B_022_01 = self.get_B022(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 1, 0)
        B_dBE = -np.sqrt(3 / 8) * B_022_01

        # B_B_delta_E (permutation)
        B_022_01 = self.get_B022(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 0, 1)
        B_BdE = -np.sqrt(3 / 8) * B_022_01

        # B_E_delta_B (permutation)
        B_022_01 = self.get_B022(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 1, 0)
        B_EdB = -np.sqrt(3 / 8) * B_022_01

        # B_EB_delta (permutation)
        B_022_01 = self.get_B022(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 0, 1)
        B_EBd = -np.sqrt(3 / 8) * B_022_01

        # B_BE_delta (permutation)
        B_022_01 = self.get_B022(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 1, 0)
        B_BEd = -np.sqrt(3 / 8) * B_022_01

        # B_EEE (Eq. 14)
        B_222_000 = self.get_B222(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 0, 0, 0)
        B_222_002_perm1 = self.get_B222(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 0, 0, 2)
        B_222_002_perm2 = self.get_B222(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 2, 0, 0)
        B_222_002_perm3 = self.get_B222(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 0, 2, 0)
        B_EEE = 3 / 16 * (np.sqrt(3 / 2) * B_222_000 - B_222_002_perm1 - B_222_002_perm2 - B_222_002_perm3)

        # B_EEB (Eq. 19)
        B_222_001 = self.get_B222(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_1, cg2_2, cg2_3, 0, 0, 1)
        B_EEB = -3 / 8 * B_222_001

        # B_EBE
        B_222_010 = self.get_B222(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_1, cg2_2, cg2_3, 0, 0, 1)
        B_EBE = -3 / 8 * B_222_010

        # B_BEE
        B_222_100 = self.get_B222(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_1, cg2_2, cg2_3, 0, 0, 1)
        B_BEE = -3 / 8 * B_222_100

        # B_EBB + permutations
        B_EBB = 0
        B_BBE = 0
        B_BEB = 0

        # B_BBB
        B_BBB = 0

        direct_methods = False

        if direct_methods:
            B_ddE = self.get_B_ddE_gomes(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, C1, C1delta, C2)

            #try change
            B_dEd = self.get_B_ddE_gomes(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, C1, C1delta, C2)
            B_Edd = self.get_B_ddE_gomes(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, C1, C1delta, C2)
            B_dEE = self.get_B_dEE_gomes(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, C1, C1delta, C2)
            B_EdE = self.get_B_dEE_gomes(k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, C1, C1delta, C2)
            B_EEd = self.get_B_dEE_gomes(k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, C1, C1delta, C2)
            B_EEE = self.get_B_EEE_gomes(k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, C1, C1delta, C2)
            print("did direct method gomes")

        tl_mat = 2 * (self.F2_tree(k1_mag, k2_mag, k3_mag) * PL1 * PL2 + self.F2_tree(k2_mag, k3_mag, k1_mag) * PL2 * PL3 +
                                 self.F2_tree(k3_mag, k1_mag, k2_mag) * PL3 * PL1)

        if remove_alignment:
            return B_ddE - 1/2*cg1*tl_mat, B_dEd - 1/2*cg1*tl_mat, B_Edd - 1/2*cg1*tl_mat, B_dEE - 1/4*cg1**2*tl_mat, \
                   B_EEd - 1/4*cg1**2*tl_mat, B_EdE - 1/4*cg1**2*tl_mat, B_EEE - 1/8*cg1**3*tl_mat, B_ddB, B_dBd, B_Bdd, B_dEB, B_dBE, B_EBd, B_BEd, B_BdE, B_EdB, B_EEB, B_EBE, B_BEE

        else:
            return B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE, B_ddB, B_dBd, B_Bdd, B_dEB, B_dBE, B_EBd, B_BEd, B_BdE, B_EdB, B_EEB, B_EBE, B_BEE
            #return B_dEE
