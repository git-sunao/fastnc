import numpy as np
from scipy.special import sici
from scipy.special import eval_legendre
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from . import twobessel

##############################################
# Utilities
def compute_multipoles(bispectrum, l, psi, Lmax, mu_safety_pad=1e-2):
    l  = np.asarray(l)
    psi= np.asarray(psi)
    mu = np.linspace(-1+mu_safety_pad, 1-mu_safety_pad, 100)
    dmu = mu[1]-mu[0]

    lg, psig, mug = np.meshgrid(l, psi, mu, indexing='ij')

    # Note that the bispectrum is specified by two sides and its inner angle,
    # while the multipole decomposition is defined with the outer angle.
    # Thus, we need a minus sign for mu.
    bl = bispectrum(lg, psig, -mug)
    
    coeffs = []
    for L in range(Lmax):
        p = eval_legendre(L, mug)
        c = np.trapz(bl*p, dx=dmu, axis=2) * (2*L+1)/2
        coeffs.append(c)
    coeffs = np.transpose(coeffs, (1, 2, 0))
    return coeffs

def sincos2angbar(psi, delta):
    cos2b = np.cos(delta) + np.sin(2*psi)
    sin2b = np.cos(2*psi) * np.sin(delta)
    norm   = np.sqrt(cos2b**2 + sin2b**2)
    return sin2b/norm, cos2b/norm

def log_extrap(x, n1, n2):
    low_x = high_x = []
    if(n1):
        dlnx_low = np.log(x[1]/x[0])
        low_x = x[0] * np.exp(dlnx_low * np.arange(-n1, 0) )
    if(n2):
        dlnx_high= np.log(x[-1]/x[-2])
        high_x = x[-1] * np.exp(dlnx_high * np.arange(1, n2+1) )
    x_extrap = np.hstack((low_x, x, high_x))
    return x_extrap

##############################################
# Class
class NaturalComponentsCalcurator:
    def __init__(self, bispectrum, loglmid=2, loglwid=5):
        self.bispectrum = bispectrum
        self.loglmid = loglmid
        self.loglwid = loglwid
        self.F_MN_dict = dict()

    def init_multipole(self, Lmax, psi_safety_pad=1e-3, mu_safety_pad=1e-5):
        l  = np.logspace(self.loglmid-self.loglwid/2, self.loglmid+self.loglwid/2, 300)
        psi= np.linspace(psi_safety_pad, np.pi/4, 60)
        bL = compute_multipoles(self.bispectrum, l, psi, Lmax, mu_safety_pad)
        self.bispectrum_multipoles = {'l':l, 'psi':psi, 'multipoles':bL, 'Lmax':Lmax}

    def get_bLlpsi(self, L, l, psi):
        # cast variables to numpy array
        l   = np.asarray(l)
        psi = np.asarray(psi)

        Lmax = self.bispectrum_multipoles['Lmax']
        if L> Lmax:
            raise ValueError('L={} is larger than Lmax={}'.format(L, Lmax))
        x = np.log(self.bispectrum_multipoles['l'])
        y = self.bispectrum_multipoles['psi']
        z = self.bispectrum_multipoles['multipoles'][:, :, L]
        bLfnc= rgi((x, y), z, bounds_error=False, fill_value=None)

        # convert psi to pi/2-psi if psi > pi/4
        sel = np.pi/4 < psi
        psi[sel] = np.pi/2 - psi[sel]

        return bLfnc((np.log(l), psi))

    # Multipole index manupilation
    def _mn2MN(self, i, m, n):
        if i == 0:
            M, N = m+3, n+3
        elif i == 1 or i == 2 or i == 3:
            M, N = m+1, n+1
        return M, N

    def _MN2mn(self, i, M, N):
        if i == 0:
            m, n = M-3, N-3
        elif i == 1 or i == 2 or i == 3:
            m, n = M-1, N-1
        return m, n

    # shape functions
    def S_K(self, K, x):
        if K == 0:
            return x - 2*np.pi
        else:
            return 2*np.sin(K*x/2)/K

    def _G_LJK(self, L, J, K, psi, Nx=100):
        shape = psi.shape

        # Integration variable
        x = np.linspace(0, 2*np.pi, Nx)

        # reshape array for the ease of calculation
        x, psi = np.meshgrid(x, psi)

        # Compute functions in the integrand
        sindb, cosdb = np.sin(J*x/2), np.cos(J*x/2)
        sinbb, cosbb = sincos2angbar(psi, x)
        pLx = eval_legendre(L, np.cos(x))
        s_K = self.S_K(K, x)

        # perform integration and multiply a prefactor (2)
        g_LJK = 2*np.sum(pLx*s_K*(cosdb*cosbb - sindb*sinbb), axis=1) * 2*np.pi/Nx
        
        # reshape back to the original shape
        g_LJK = np.reshape(g_LJK, shape)

        return g_LJK

    def G_LJK(self, L, J, K, psi, Nx=100, lazy=True):
        if lazy:
            x = np.linspace(psi.min(), psi.max(), 100)
            y = self._G_LJK(L, J, K, x, Nx)
            return ius(x, y)(psi)
        else:
            return self._G_LJK(L, J, K, psi, Nx)

    def sumGLJKbL(self, i, J, K, l, psi, Lmax=None):
        sumGLJKbL = np.zeros_like(psi)

        # depending on which natural component we are computing,
        # we change the J for G_LJK
        if i == 0:
            J4G = J
        elif i == 1:
            J4G = -4+J
        elif i == 2:
            J4G = 4+J
        elif i == 3:
            J4G = -J

        # Sum up G_LKJ*bL
        for L in range(Lmax or self.bispectrum_multipoles['Lmax']):
            # Compute G_LJK
            g_LJK = self.G_LJK(L, J4G, K, psi)

            # Compute bLl1l2
            bL = self.get_bLlpsi(L, l, psi)

            sumGLJKbL += g_LJK*bL
        return sumGLJKbL

    def F_MN(self, i, M, N, Lmax=None, dlnx=1e-8, nu=1.01, N_pad=50, return_x1x2=True):
        # Initialize l12bin bins
        l12bin = np.logspace(self.loglmid-self.loglwid/2, self.loglmid+self.loglwid/2, 200)

        # Make mesh grid
        l1, l2 = np.meshgrid(l12bin, l12bin)

        # Compute l, psi
        l   = np.sqrt(l1**2+l2**2)
        psi = np.arctan2(l2, l1)

        # Compute sumGLJKbL = \sum_L G_LJK * b_L(l1, l2)
        J, K = M-N, M+N
        sumGLJKbL = self.sumGLJKbL(i, J, K, l, psi, Lmax=Lmax)

        # Compute F_MN using 2DFFTLog
        m, n = self._MN2mn(i, M, N)
        tb  = twobessel.two_Bessel(l12bin, l12bin, sumGLJKbL*l1**2*l2**2, nu1=nu, nu2=nu, N_pad=N_pad)
        x1, x2, f_MN = tb.two_Bessel_binave(np.abs(m), np.abs(n), dlnx, dlnx)

        # Apply (-1)**m and (-1)**n 
        # These originate to J_m(x) = (-1)^m J_{-m}(x)
        if m < 0:
            f_MN *= (-1.)**m
        if n < 0:
            f_MN *= (-1.)**n

        # Transpose
        # Note that (l1, l2) is couple to (x2,x1), 
        # not to (x1, x2) which is convention of 2DFFTLog.
        f_MN = f_MN.T

        if return_x1x2:
            return x1, x2, f_MN
        else:
            return f_MN

    def init_F_MN(self, i, Kmax, Jmax, Lmax=None, dlnx=0.0001, nu=1.01, N_pad=50):
        K = np.arange(-Kmax, Kmax+1)
        J = np.arange(-Jmax, Jmax+1)

        KJMN_list = []
        for k in K:
            for j in J:
                if (k+j)%2 == 1:
                    continue
                m = (k+j)//2
                n = (k-j)//2
                if m > n and (i == 0 or i == 3):
                    # Skip this (M, N) configuration 
                    # because F_{MN}(x1, x2) = F_{NM}(x2, x1)
                    # for natural component i=0 and 3.
                    continue
                KJMN_list.append([k, j, m, n])
        KJMN_list = np.array(KJMN_list)
        print('The number of (M, N) grids to compute basis = {}'.format(KJMN_list.shape[0]))

        self.F_MN_dict[i] = dict()
        self.F_MN_setup= {'Kmax':Kmax,'Jmax':Jmax}
        for K, J, M, N in KJMN_list:
            x1, x2, f_MN = self.F_MN(i, M, N, Lmax=Lmax, nu=nu, N_pad=N_pad)
            self.F_MN_dict[i][(M, N)] = f_MN
            if M != N and (i == 0 or i == 3):
                self.F_MN_dict[i][(N, M)] = f_MN.T
        self.x1x2 = [x1, x2]

    def Gamma(self, i, dphi, Kmax=None, Jmax=None, return_x1x2=False):
        # Compute delta varphi
        # Note the notation: delta varphi = varphi1 - varphi2 = phi3 - pi, 
        # where dphi = phi3 is the inner angle between sides x1 and x2.
        # while dvphi is the outer angle.
        dvphi = dphi - np.pi

        # x1 and x2
        x1, x2 = self.x1x2

        # compute phibar
        xx1, xx2 = np.meshgrid(x1, x2)
        psi = np.arctan2(xx2, xx1)
        sin2pb, cos2pb = sincos2angbar(psi, dvphi)

        # compute Gamma^0(x1, x2, dphi)
        gamma = np.zeros((x1.size, x2.size), dtype=np.complex128)
        for (M, N), f_MN in self.F_MN_dict[i].items():
            K, J = M+N, M-N

            if Kmax is not None:
                if np.abs(K) > Kmax:
                    continue
            if Jmax is not None:
                if np.abs(J) > Jmax:
                    continue

            # Compute phase
            phase  = 1j**J * np.exp(0.5j*J*dvphi)

            # add
            gamma+= f_MN * phase

        # multiply prefactor
        if i == 0:
            prefactor = -1./(2*np.pi)**4 * (cos2pb - 1j*sin2pb)
        elif i == 1:
            prefactor = -1./(2*np.pi)**4 * (cos2pb - 1j*sin2pb) * (np.cos(2*dvphi) + 1j*np.sin(2*dvphi))
        elif i == 2:
            prefactor = -1./(2*np.pi)**4 * (cos2pb - 1j*sin2pb) * (np.cos(2*dvphi) - 1j*np.sin(2*dvphi))
        elif i == 3:
            prefactor = -1./(2*np.pi)**4 * (cos2pb + 1j*sin2pb)
        gamma *= prefactor

        # return
        if return_x1x2:
            return x1, x2, gamma
        else:
            return gamma

    def Gamma0(self, dphi, Kmax=None, Jmax=None, return_x1x2=False):
        return self.Gamma(0, dphi, Kmax=Kmax, Jmax=Jmax, return_x1x2=return_x1x2)

"""
TODO:
- multipole decomposition:
    accuracy is not so good at mu ~ +/-1, when we reconstruct 
    the bispectrum with legendre polynomials with coefficitnts.
- choice of Lmax
    this is related to the above item.
- choice of Jmax
- sign of Gamma^{(0)}
- ringing effect
    Can we use zero padding?
    Can we extrapolate bispectrum multipole in log(l1) and log(l2), but linear for b_L(l1, l2).
"""