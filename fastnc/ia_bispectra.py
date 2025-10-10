import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simpson
from scipy.optimize import bisect
from scipy.special import gamma, gammaincc

# Reusing window functions from halofit.py
def window_tophat(x):
    return 3.0/x**3 * (np.sin(x) - x*np.cos(x))

def window_gaussian(x):
    return np.exp(-0.5*x**2)

def window_gaussian_1deriv(x):
    return x*np.exp(-0.5*x**2)

def window_gaussian_2deriv(x):
    return x**2*np.exp(-0.5*x**2)

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

    def update(self):
        if self.has_changed:
            self._normalize_pklin()
            self._init_spline()
            self.has_changed = False

    def _normalize_pklin(self):
        if (self.k is not None) and (self.pklin is not None) and (self.cosmo is not None):
            k_interp = np.logspace(-3, 2, 1000)
            Delta = self.get_interpolated_pklin(k_interp)*k_interp**3 / 2/np.pi**2
            sigma8temp = self._sigmam(k_interp, Delta, 8.0, window_tophat)
            self.pklin *= (self.cosmo["sigma8"]/sigma8temp)**2

    def get_interpolated_pklin(self, k, z=None, ext=True):
        k = np.atleast_1d(k) # Ensure k is always at least 1D array
        pk = np.zeros_like(k, dtype=float)
        where = (self.k.min() <= k) & (k <= self.k.max())
        
        if np.any(where):
            logpk = ius(np.log(self.k), np.log(self.pklin))(np.log(k[where]))
            pk[where] = np.exp(logpk)

        if ext:
            where_low = k < self.k.min()
            if np.any(where_low):
                n_low = np.log(self.pklin[1]/self.pklin[0]) / np.log(self.k[1]/self.k[0])
                a_low = self.pklin[0]
                pk[where_low] = a_low*(k[where_low]/self.k[0])**n_low

            where_high = k > self.k.max()
            if np.any(where_high):
                n_high = np.log(self.pklin[-1]/self.pklin[-2]) / np.log(self.k[-1]/self.k[-2])
                a_high = self.pklin[-1]
                pk[where_high] = a_high*(k[where_high]/self.k[-1])**n_high
        
        if z is not None:
            pk *= ius(self.z, self.lgr, ext=2)(z)**2

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
        I2 = simpson(Delta * window(k*r)**2, x=np.log(k))
        if extrap:
            n = np.diff(np.log(Delta))[-1]/np.diff(np.log(k))[-1]
            A = Delta[-1] * k[-1]**(-n)
            tmin = k[-1] * r
            I2 += A * r**-n * 0.5*gamma(n/2) * gammaincc(n/2, tmin**2)
        return I2**0.5

    def F2_tree(self, k1, k2, k3):
        costheta12 = 0.5*(k3*k3-k1*k1-k2*k2)/(k1*k2)
        return (5./7.)+0.5*costheta12*(k1/k2+k2/k1)+(2./7.)*costheta12*costheta12

    def _construct_k_vectors(self, k1_mag, k2_mag, k3_mag):

        original_shape = k1_mag.shape
        k1_mag_flat = k1_mag.ravel()
        k2_mag_flat = k2_mag.ravel()
        k3_mag_flat = k3_mag.ravel()

        k3_vec_flat = np.array([k3_mag_flat, np.zeros_like(k3_mag_flat)])
        cos_alpha = (k3_mag_flat ** 2 + k1_mag_flat ** 2 - k2_mag_flat ** 2) / (2 * k3_mag_flat * k1_mag_flat)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        sin_alpha = np.sqrt(1 - cos_alpha ** 2)

        # We can choose k1y to be positive without loss of generality
        k1_vec_flat = np.array([k1_mag_flat * cos_alpha, k1_mag_flat * sin_alpha])
        k2_vec_flat = - k3_vec_flat - k1_vec_flat

        k1_vec = k1_vec_flat.reshape((2,) + original_shape)
        k2_vec = k2_vec_flat.reshape((2,) + original_shape)
        k3_vec = k3_vec_flat.reshape((2,) + original_shape)

        return k1_vec, k2_vec, k3_vec

    def get_F_00(self, k_i_vec, k_j_vec, k_i_mag, k_j_mag, k_l_mag, PL_i, PL_j, cg1, cg2_2, cg2_3, mode='ssg'):

        if mode == 'sgs' or mode == 'gss':
            kernel = cg1 * 2 * self.F2_tree(k_i_mag, k_j_mag, k_l_mag) * PL_i * PL_j

        elif mode == 'ggs':
            kernel = cg1**2 * 2 * self.F2_tree(k_i_mag, k_j_mag, k_l_mag) * PL_i * PL_j

        elif mode == 'sgg' or mode == 'gsg':
            k1x, k1y = k_i_vec[0] / k_i_mag, k_i_vec[1] / k_i_mag
            k2x, k2y = k_j_vec[0] / k_j_mag, k_j_vec[1] / k_j_mag
            k1_dot_k2 = (k1x * k2x + k1y * k2y)
            kernel = cg1 * 2 * PL_i * PL_j * (cg1 * self.F2_tree(k_i_mag, k_j_mag, k_l_mag) + (cg2_2+cg2_3) * k1_dot_k2**2 + cg2_3)

        elif mode == 'sss':
            kernel = 2 * self.F2_tree(k_i_mag, k_j_mag, k_l_mag) * PL_i * PL_j

        return kernel

    def get_F_20(self, k1_vec, k2_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_2, cg2_3, mode='ssg'):

        k1x, k1y = k1_vec[0]/k1_mag, k1_vec[1]/k1_mag
        k2x, k2y = k2_vec[0]/k2_mag, k2_vec[1]/k2_mag
        k1_dot_k2 = k1x*k2x + k1y*k2y

        term1 = cg1 * self.F2_tree(k1_mag, k2_mag, k3_mag)
        term2 = cg2_2 * (0.25 * (k1x**2 - k1y**2 + k2x**2 - k2y**2) + 0.5 * k1_dot_k2 * (k1x*k2x))
        term3 = cg2_3 * (0.25 * (2*k1x**2 - k1y**2 + 2*k2x**2 - k2y**2))

        if mode == 'sgg':
            result = cg1 * 2 * np.sqrt(2/3) * PL1 * PL2 * (term1 + term2 + term3)

        elif mode == 'ssg':
            result = 2 * np.sqrt(2/3) * PL1 * PL2 * (term1 + term2 + term3)

        elif mode == 'gsg':
            result = cg1 * 2 * np.sqrt(2/3) * PL1 * PL2 * (term1 + term2 + term3)

        elif mode == 'ggs':
            result = 2 * np.sqrt(2/3) * cg1 * PL1 * PL2 * term1

        elif mode == 'ggg':
            result = cg1**2 * 2 * np.sqrt(2 / 3) * PL1 * PL2 * (term1 + term2 + term3)

        return result

    #def get_F_21(self, k1_vec, k2_vec, k1_mag, k2_mag, PL1, PL2, cg2_2, cg2_3):
    #will not be used, must be rechecked (specifically the alpha,beta,gamma ordering), if uncommented
    #    k1x, k1y = k1_vec[0]/k1_mag, k1_vec[1]/k1_mag
    #    k2x, k2y = k2_vec[0]/k2_mag, k2_vec[1]/k2_mag
    #    return -1 * PL1 * PL2 * (cg2_2 + cg2_3) * (k1x*k1y + k2x*k2y)

    def get_F_22(self, k1_vec, k2_vec, k1_mag, k2_mag, PL1, PL2, cg1, cg2_2, cg2_3, mode='ssg'):

        if mode == 'ggs':
            return 0

        k1x, k1y = k1_vec[0]/k1_mag, k1_vec[1]/k1_mag
        k2x, k2y = k2_vec[0]/k2_mag, k2_vec[1]/k2_mag
        k1_dot_k2 = (k1x*k2x + k1y*k2y)
        partial = 2 * PL1 * PL2 * (cg2_2 * (0.5 * k1_dot_k2 * (k1y*k2y)) + cg2_3 * (0.25 * (k1y**2 + k2y**2)))

        if mode == 'ssg':
            result = partial

        elif mode == 'gsg':
            result = cg1 * partial

        elif mode == 'ggg':
            result = cg1**2 * partial

        return result

    def get_B000(self, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3):
        #This is not used, here for completeness
        F0_12 = self.get_F_00(k1_vec, k2_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, 0, 0, 0, mode='sss')
        F0_23 = self.get_F_00(k2_vec, k3_vec, k2_mag, k3_mag, k1_mag, PL2, PL3, 0, 0, 0, mode='sss')
        F0_31 = self.get_F_00(k3_vec, k1_vec, k3_mag, k1_mag, k2_mag, PL3, PL1, 0, 0, 0, mode='sss')
        return F0_12 + F0_23 + F0_31

    def get_B002(self, k_i_vec, k_j_vec, k_l_vec, k_i_mag, k_j_mag, k_l_mag, PL_i, PL_j, PL_l, cg1, cg2_2,
                cg2_3, m_val):
        #Here our convention is to always use alpha, beta, gamma = ssg.

        if m_val == 0:
            F_ij_m = self.get_F_20(k_i_vec, k_j_vec, k_i_mag, k_j_mag, k_l_mag, PL_i, PL_j, cg1, cg2_2, cg2_3, mode='ssg')
            delta_K_0m = np.sqrt(2/3)
            F0_jk = self.get_F_00(k_j_vec, k_l_vec, k_j_mag, k_l_mag, k_i_mag, PL_j, PL_l, cg1, cg2_2, cg2_3, mode='sgs')
            F0_li = self.get_F_00(k_l_vec, k_i_vec, k_l_mag, k_i_mag, k_j_mag, PL_l, PL_i, cg1, cg2_2, cg2_3, mode='gss')
            return F_ij_m + delta_K_0m * F0_jk + delta_K_0m * F0_li

        elif m_val == 2:
            F_ij_m = self.get_F_22(k_i_vec, k_j_vec, k_i_mag, k_j_mag, PL_i, PL_j, cg1, cg2_2, cg2_3, mode='ssg')
            return F_ij_m

        else:
            raise ValueError("m_val must be 0 or 2 for get_B002")

    def get_delta_K(self, m_val):
        # Kronecker delta definition: zero if m is not zero, and N0^(-1) if m is zero
        if m_val == 0.0:
            return np.sqrt(2/3)
        else:
            return 0.0

    def get_B022(self, k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, m2_val, m3_val):

        delta_K_0m2 = self.get_delta_K(m2_val)
        delta_K_0m3 = self.get_delta_K(m3_val)

        if m3_val == 0:
            F_12_m3 = self.get_F_20(k1_vec, k2_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_2, cg2_3, mode='sgg')
        elif m3_val == 2:
            F_12_m3 = self.get_F_22(k1_vec, k2_vec, k1_mag, k2_mag, PL1, PL2, cg1, cg2_2, cg2_3, mode='sgg')
        else:
            raise ValueError("m3_val must be 0 or 2 for get_B022")

        F0_23 = self.get_F_00(k2_vec, k3_vec, k2_mag, k3_mag, k1_mag, PL2, PL3, cg1, cg2_2, cg2_3, mode='ggs')

        if m2_val == 0:
            F_31_m2 = self.get_F_20(k3_vec, k1_vec, k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_2, cg2_3, mode='gsg')
        elif m2_val == 2:
            F_31_m2 = self.get_F_22(k3_vec, k1_vec, k3_mag, k1_mag, PL3, PL1, cg1, cg2_2, cg2_3, mode='gsg')
        else:
            raise ValueError("m2_val must be 0 or 2 for get_B022")

        term1 = delta_K_0m2 * F_12_m3
        term2 = delta_K_0m2 * delta_K_0m3 * F0_23
        term3 = delta_K_0m3 * F_31_m2

        return term1 + term2 + term3

    def get_B222(self, k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3,
                 m1_val, m2_val, m3_val):

        delta_K_0m1 = self.get_delta_K(m1_val)
        delta_K_0m2 = self.get_delta_K(m2_val)
        delta_K_0m3 = self.get_delta_K(m3_val)

        if m3_val == 0:
            F_12_m3 = self.get_F_20(k1_vec, k2_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, cg1, cg2_2, cg2_3, mode='ggg')
        elif m3_val == 2:
            F_12_m3 = self.get_F_22(k1_vec, k2_vec, k1_mag, k2_mag, PL1, PL2, cg1, cg2_2, cg2_3, mode='ggg')
        else:
            raise ValueError("m3_val must be 0 or 2 for get_B222")

        if m1_val == 0:
            F_23_m1 = self.get_F_20(k2_vec, k3_vec, k2_mag, k3_mag, k1_mag, PL2, PL3, cg1, cg2_2, cg2_3, mode='ggg')
        elif m1_val == 2:
            F_23_m1 = self.get_F_22(k2_vec, k3_vec, k2_mag, k3_mag, PL2, PL3, cg1, cg2_2, cg2_3, mode='ggg')
        else:
            raise ValueError("m1_val must be 0 or 2 for get_B222")

        if m2_val == 0:
            F_31_m2 = self.get_F_20(k3_vec, k1_vec, k3_mag, k1_mag, k2_mag, PL3, PL1, cg1, cg2_2, cg2_3, mode='ggg')
        elif m2_val == 2:
            F_31_m2 = self.get_F_22(k3_vec, k1_vec, k3_mag, k1_mag, PL3, PL1, cg1, cg2_2, cg2_3, mode='ggg')
        else:
            raise ValueError("m2_val must be 0 or 2 for get_B222")

        term1 = delta_K_0m1 * delta_K_0m2 * F_12_m3
        term2 = delta_K_0m2 * delta_K_0m3 * F_23_m1
        term3 = delta_K_0m3 * delta_K_0m1 * F_31_m2
        return term1 + term2 + term3

    def get_ia_bispectra(self, k1_mag, k2_mag, k3_mag, z, z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta, remove_alignment=False, do_non_linear=True):

        #Get C1, C1delta and C2 params
        c1rhocrit = 0.0134
        C1 = -A1 * ((1+z)/(1+z_piv))**alphaIA * c1rhocrit * self.cosmo['Om0'] / self.z2lgr(z)
        C1delta = bias_ta*C1
        C2 = A2 * ((1+z)/(1+z_piv))**alphaIA_2 * 5 * c1rhocrit * self.cosmo['Om0'] / self.z2lgr(z)**2

        # normalization to match the A1 of TATT, according to Eq. 66 of Bakx et al (1) (2025) and Eq. 30 of Bakx et. al. (2) (2025)
        C1 *= 2

        #Get cg1, cg2_2 and cg2_3 params
        cg1 = C1
        cg2_2 = C2
        cg2_3 = 0.5 * (3 * C1delta - C2)

        # Construct k-vectors
        k1_vec, k2_vec, k3_vec = self._construct_k_vectors(k1_mag, k2_mag, k3_mag)

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
        B_002_0 = self.get_B002(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 0)
        B_002_2 = self.get_B002(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 2)
        B_ddE = 0.5 * (np.sqrt(3/2) * B_002_0 - B_002_2)

        # B_delta_E_delta (permutation)
        B_002_0 = self.get_B002(k2_vec, k3_vec, k1_vec, k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_2, cg2_3, 0)
        B_002_2 = self.get_B002(k2_vec, k3_vec, k1_vec, k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_2, cg2_3, 2)
        B_dEd = 0.5 * (np.sqrt(3/2) * B_002_0 - B_002_2)

        # B_E_delta_delta (permutation)
        B_002_0 = self.get_B002(k3_vec, k1_vec, k2_vec, k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_2, cg2_3, 0)
        B_002_2 = self.get_B002(k3_vec, k1_vec, k2_vec, k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_2, cg2_3, 2)
        B_Edd = 0.5 * (np.sqrt(3 / 2) * B_002_0 - B_002_2)

        # B_delta_EE (Eq. 13)
        B_022_00 = self.get_B022(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 0, 0)
        B_022_02_perm1 = self.get_B022(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 0, 2)
        B_022_02_perm2 = self.get_B022(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 2, 0)
        B_dEE = np.sqrt(3/2)/4 * (np.sqrt(3/2) * B_022_00 + B_022_02_perm1 + B_022_02_perm2)

        # B_EE_delta (permutation)
        B_022_00 = self.get_B022(k2_vec, k3_vec, k1_vec, k2_mag, k3_mag, k1_mag, PL2, PL3, PL1, cg1, cg2_2, cg2_3, 0, 0)
        B_022_02_perm1 = self.get_B022(k2_vec, k3_vec, k1_vec, k2_mag, k3_mag, k1_mag, PL3, PL1, PL2, cg1, cg2_2, cg2_3, 0, 2)
        B_022_02_perm2 = self.get_B022(k2_vec, k3_vec, k1_vec, k2_mag, k3_mag, k1_mag, PL3, PL1, PL2, cg1, cg2_2, cg2_3, 2, 0)
        B_EEd = np.sqrt(3 / 2) / 4 * (np.sqrt(3 / 2) * B_022_00 + B_022_02_perm1 + B_022_02_perm2)

        # B_E_delta_E (permutation)
        B_022_00 = self.get_B022(k3_vec, k1_vec, k2_vec, k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_2, cg2_3, 0, 0)
        B_022_02_perm1 = self.get_B022(k3_vec, k1_vec, k2_vec, k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_2, cg2_3, 0, 2)
        B_022_02_perm2 = self.get_B022(k3_vec, k1_vec, k2_vec, k3_mag, k1_mag, k2_mag, PL3, PL1, PL2, cg1, cg2_2, cg2_3, 2, 0)
        B_EdE = np.sqrt(3 / 2) / 4 * (np.sqrt(3 / 2) * B_022_00 + B_022_02_perm1 + B_022_02_perm2)

        # B_EEE (Eq. 14)
        B_222_000 = self.get_B222(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 0,0,0)
        B_222_002_perm1 = self.get_B222(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 0,0,2)
        B_222_002_perm2 = self.get_B222(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 2,0,0)
        B_222_002_perm3 = self.get_B222(k1_vec, k2_vec, k3_vec, k1_mag, k2_mag, k3_mag, PL1, PL2, PL3, cg1, cg2_2, cg2_3, 0,2,0)
        B_EEE = 3/16 * (np.sqrt(3/2) * B_222_000 - B_222_002_perm1 - B_222_002_perm2 - B_222_002_perm3)

        tl_mat = 2 * (self.F2_tree(k1_mag, k2_mag, k3_mag) * PL1 * PL2 + self.F2_tree(k2_mag, k3_mag, k1_mag) * PL2 * PL3 +
                                 self.F2_tree(k3_mag, k1_mag, k2_mag) * PL3 * PL1)

        if remove_alignment:
            return B_ddE - 1/2*cg1*tl_mat, B_dEd - 1/2*cg1*tl_mat, B_Edd - 1/2*cg1*tl_mat, B_dEE - 1/4*cg1**2*tl_mat, \
                   B_EEd - 1/4*cg1**2*tl_mat, B_EdE - 1/4*cg1**2*tl_mat, B_EEE - 1/8*cg1**3*tl_mat
        else:
            return B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE