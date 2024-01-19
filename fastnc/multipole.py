import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import RegularGridInterpolator as rgi
from . import utils
from tqdm import tqdm

class Multipole:
    def __init__(self, x, Lmax, method='linear', verbose=True):
        self.method = method
        self.x      = x
        self.Lmax   = Lmax
        self.verbose = verbose
        self.init_legendreP_table()

    def init_legendreP_table(self):
        if self.method == 'linear':
            Lmax = self.Lmax + 2
        elif self.method == 'riemann':
            Lmax = self.Lmax
        self.legendreP = dict()
        pbar = tqdm(np.arange(Lmax+1), desc='[legendreP]', disable=not self.verbose)
        for L in pbar:
            pbar.set_postfix({'L':L})
            p = eval_legendre(L, self.x)
            self.legendreP[L] = p

    def get_legendreP(self, L):
        return self.legendreP[L]

    def get_legendreP_int0(self, L):
        """
        p_L^0 = (2L+1)\int dx P_L(x)
        """
        if L == 0:
            out = self.x
        else:
            out = self.get_legendreP(L+1) - self.get_legendreP(L-1)
        return out

    def get_legendreP_int1(self, L):
        """
        p_L^1 = (2L+1)\int dx P_L(x) x
        """
        if L == 0:
            out = self.x**2/2
        elif L == 1:
            out = self.x**3
        else:
            pL   = self.get_legendreP(L)
            pLm1 = self.get_legendreP(L-1)
            pLm2 = self.get_legendreP(L-2)
            pLp1 = self.get_legendreP(L+1)
            pLp2 = self.get_legendreP(L+2)
            out = (pL - pLm2)/(2*L-1) \
                + self.x*(pLp1 - pLm1) \
                + (pL - pLp2)/(2*L+3)
        return out
    
    def get_linear_interp_coeffs(self, x, y, axis):
        """
        Get the coefficients for a linear interpolation of y(x).
        """
        a = np.diff(y, axis=axis)/np.diff(x,axis=axis)
        b = (y*np.roll(x,-1,axis=axis) - np.roll(y,-1,axis=axis)*x)
        b = np.delete(b, -1, axis=axis)
        b = b/np.diff(x,axis=axis)
        return a, b
    
    def _decompose_linear(self, y, L, axis=0):
        """
        Decompose a function y=f(x) into multipole moments.
        """
        a, b = self.get_linear_interp_coeffs(self.x, y, axis=axis)

        p_L_0 = np.diff(self.get_legendreP_int0(L), axis=axis)
        p_L_1 = np.diff(self.get_legendreP_int1(L), axis=axis)

        out = 0.5*np.sum(a*p_L_1 + b*p_L_0, axis=axis)

        return out

    def _decompose_riemann(self, y, L, axis=0):
        p = self.get_legendreP(L)
        out = np.sum(f*p,axis=axis) * (2*L+1)/2 * (x[1]-x[0])
        return out

    def _decompose(self, f, L, axis=0):
        if self.method == 'linear':
            return self._decompose_linear(f, L, axis=axis)
        elif self.method == 'riemann':
            return self._decompose_riemann(f, L, axis=axis)
    
    def decompose(self, f, L, axis=0):
        if np.isscalar(L):
            return self._decompose(f, L, axis=axis)
        else:
            out = []
            pbar = tqdm(L, desc='[multipole]', disable=not self.verbose)
            for l in pbar:
                pbar.set_postfix({'L':l})
                out.append(self._decompose(f, l, axis=axis))
            return np.array(out)

# class BispectrumMultipole:
#     """
#     Bispectrum multipole.

#     This class provides a way to compute the bispectrum multipoles of a given bispectrum
#     in a set of bins in l and psi, and for a range of values of mu. The computation is
#     performed using the Limber approximation.

#     Parameters
#     ----------
#     bispectrum : callable
#         A callable object that takes three arguments (l, psi, mu). Here, (l, psi, mu) 
#         are related to the (l1, l2, l3) parameter set as
#             l1 = l*cos(psi)
#             l2 = l*sin(psi)
#             l3 = l*(1 - sin(2psi)*mu)
#         Thus mu is the cosine of inner angle of triangle between l1 and l2 sides.
#     Lmax : int
#         The highest multipole to compute.
#     lmin : float
#         The minimum value of l for the first bin.
#     lmax : float
#         The maximum value of l for the last bin.
#     Nl : int
#         The number of bins in l.
#     Npsi : int, optional
#         The total number of bins in psi. The bins are distributed logarithmically 
#         and linearly between arctan(lmin/lmax) and pi/4. Default is 80.
#     Nmu : int, optional
#         The number of bins in mu. Default is 100.
#     mupad : float, optional
#         A small padding value added to the edges of the mu bins to avoid numerical
#         issues with the limiting bispectrum configuration. Default is 1e-4.

#     Attributes
#     ----------
#     Lmax : int
#         The highest multipole to compute.
#     l : ndarray
#         An array of shape (Nl,) containing the values of l for each bin.
#     psi : ndarray
#         An array of shape (Npsi,) containing the values of psi for each bin.
#     mu : ndarray
#         An array of shape (Nmu,) containing the values of mu for each bin.
#     bispectrum : callable
#         The bispectrum function used for the computation.
#     multipoles : ndarray
#         An array of shape (Nl, Lmax+1) containing the bispectrum multipoles.
#     """
#     def __init__(self, bispectrum, Lmax, ellmin, ellmax, psimin, nellbin=100, npsibin=80, nmubin=100, epmu=1e-7, method='linear', verbose=True, validate=True):
#         # Define highest multipole
#         self.Lmax = Lmax

#         # general settings
#         self.verbose = verbose

#         # Define bins
#         self.ell = np.logspace(np.log10(ellmin), np.log10(ellmax), nellbin)
#         self.psi = utils.loglinear(psimin, 1e-3, 1e-2, np.pi/4, npsibin//10, npsibin - npsibin//10)
#         self.mu  = np.linspace(-1+epmu, 1, nmubin)

#         # instantiate multipole decomposer
#         _, _, MU = np.meshgrid(self.ell, self.psi, self.mu, indexing='ij')
#         self.multipole = Multipole(MU, self.Lmax, method=method, verbose=self.verbose)

#         # Define bispectrum and compute multipoles
#         if bispectrum is not None:
#             self.set_bispectrum(bispectrum)

#     def set_bispectrum(self, bispectrum):
#         """
#         Set and compute the bispectrum multipoles.
#         """
#         # set bispectrum
#         self.bispectrum = bispectrum

#         # set bins
#         ELL, SPI, MU = np.meshgrid(self.ell, self.psi, self.mu, indexing='ij')

#         # Note that the bispectrum is specified by two sides and its inner angle,
#         # while the multipole decomposition is defined with the outer angle.
#         # Thus, we need a minus sign for mu.
#         b = self.bispectrum(ELL, SPI, -MU)

#         # Compute multipoles
#         L = np.arange(self.Lmax+1)
#         m = self.multipole.decompose(b, L, axis=2)
#         self.multipoles_data = m

#     def __single_L_call__(self, L, ell, psi, extrapolate=False, replace_close=True):
#         """
#         Compute the multipole decomposition of the bispectrum for a single value of L.

#         Parameters
#         ----------
#         L : int
#             The value of L.
#         l : float or ndarray
#             The value(s) of l.
#         psi : float or ndarray
#             The value(s) of psi.
#         extrapolate : bool, optional
#             If True, extrapolate the multipole decomposition outside the range of
#             l and psi. Default is False.
        
#         Returns
#         -------
#         multipole : float or ndarray
#             The multipole decomposition of the bispectrum.
#         """
#         # test L
#         if L > self.Lmax:
#             raise ValueError(f"L = {L} > Lmax = {self.Lmax}")

#         # cast
#         ell = np.asarray(ell).copy()
#         psi = np.asarray(psi).copy()

#         # settings for interpolation
#         if extrapolate:
#             bounds_error = False
#             fill_value = None
#         else:
#             bounds_error = True
#             fill_value = np.nan
        
#         # make interpolator
#         x = np.log(self.ell)
#         y = np.log(self.psi)
#         z = self.multipoles_data[L, :, :]
#         f = rgi((x, y), z, bounds_error=bounds_error, fill_value=fill_value)

#         # convert psi to pi/2-psi if psi > pi/4
#         sel = np.pi/4 < psi
#         psi[sel] = np.pi/2 - psi[sel]

#         # compute interpolated multipole
#         xin = np.log(ell)
#         yin = np.log(psi)

#         # check boundary
#         if replace_close:
#             # Sometimes the input l, psi value can slightly outside the support range
#             # due to the numerical error. This causes the interpolation to fail.
#             # Here we check if the input l, psi value is close to the boundary and
#             # set it to the boundary value if it is.
#             xin = utils.replace_close(xin, x.min(), x.max())
#             yin = utils.replace_close(yin, y.min(), y.max())

#         return f((xin, yin))

#     def __call__(self, L, ell, psi, extrapolate=False):
#         """
#         Compute the multipole decomposition of the bispectrum.

#         Parameters
#         ----------
#         L : int or ndarray
#             The value(s) of L.
#         l : float or ndarray
#             The value(s) of l.
#         psi : float or ndarray
#             The value(s) of psi.
#         extrapolate : bool, optional
#             If True, extrapolate the multipole decomposition outside the range of
#             l and psi. Default is False.
        
#         Returns
#         -------
#         multipole : float or ndarray
#             The multipole decomposition of the bispectrum.
#         """

#         if np.isscalar(L):
#             return self.__single_L_call__(L, ell, psi, extrapolate)
#         else:
#             out = []
#             for _L in L:
#                 o = self.__single_L_call__(_L, ell, psi, extrapolate)
#                 out.append(o)
#             return np.array(out)

#     def resum(self, ell, psi, mu, return_terms=False):
#         """
#         Compute the resummed bispectrum.

#         Parameters
#         ----------
#         l : float or ndarray
#             The value(s) of l.
#         psi : float or ndarray
#             The value(s) of psi.
#         mu : float or ndarray
#             The value(s) of mu.

#         Returns
#         -------
#         resummed : float or ndarray
#             The resummed bispectrum.
#         """
#         L = np.arange(self.Lmax)
#         out = []
#         for _L in L:
#             m = self.__call__(_L, ell, psi)
#             p = eval_legendre(_L, -mu)
#             out.append(m*p)

#         # return 
#         if return_terms:
#             return np.sum(out, axis=0), np.array(out)
#         else:
#             return np.sum(out, axis=0)