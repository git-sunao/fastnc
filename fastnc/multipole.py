import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import RegularGridInterpolator as rgi
from . import utils

def P_L_0_int(L, x):
    """
    p_L^0 = (2L+1)\int dx P_L(x)
    """
    if L == 0:
        out = x
    else:
        out = eval_legendre(L+1, x) - eval_legendre(L-1, x)
    return out

def P_L_1_int(L, x):
    """
    p_L^1 = (2L+1)\int dx P_L(x) x
    """
    if L == 0:
        out = x**2/2
    elif L == 1:
        out = x**3
    else:
        pL   = eval_legendre(L, x)
        pLm1 = eval_legendre(L-1, x)
        pLm2 = eval_legendre(L-2, x)
        pLp1 = eval_legendre(L+1, x)
        pLp2 = eval_legendre(L+2, x)
        out = (pL - pLm2)/(2*L-1) \
             + x*(pLp1 - pLm1) \
             + (pL - pLp2)/(2*L+3)
    return out

def get_linear_interp_coeffs(x, y, axis):
    """
    Get the coefficients for a linear interpolation of y(x).
    """
    a = np.diff(y, axis=axis)/np.diff(x,axis=axis)
    b = (y*np.roll(x,-1,axis=axis) - np.roll(y,-1,axis=axis)*x)
    b = np.delete(b, -1, axis=axis)
    b = b/np.diff(x,axis=axis)
    return a, b

def _decompose_multipole_linear(x, f, L, axis=0):
    """
    Decompose a function f(x) into multipole moments.
    """
    a, b = get_linear_interp_coeffs(x,f,axis=axis)

    p_L_0 = np.diff(P_L_0_int(L, x), axis=axis) # this part can be faster
    p_L_1 = np.diff(P_L_1_int(L, x), axis=axis)

    out = 0.5*np.sum(a*p_L_1 + b*p_L_0, axis=axis)

    return out

def _decompose_multipole_riemann(x, f, L, axis=0):
    p = eval_legendre(L, x)
    out = np.sum(f*p,axis=axis) * (2*L+1)/2 * (x[1]-x[0])
    return out

def decompose_multipole(x, f, L, axis=0, method='linear'):
    if method == 'linear':
        fnc = _decompose_multipole_linear
    elif method == 'riemann':
        fnc = _decompose_multipole_riemann

    if np.isscalar(L):
        return fnc(x, f, L, axis=axis)
    else:
        out = []
        for l in L:
            out.append(fnc(x, f, l, axis=axis))
        return np.array(out)

class BispectrumMultipoleCalculator:
    """
    Bispectrum multipole calculator.

    This class provides a way to compute the bispectrum multipoles of a given bispectrum
    in a set of bins in l and psi, and for a range of values of mu. The computation is
    performed using the Limber approximation.

    Parameters
    ----------
    bispectrum : callable
        A callable object that takes three arguments (l, psi, mu). Here, (l, psi, mu) 
        are related to the (l1, l2, l3) parameter set as
            l1 = l*cos(psi)
            l2 = l*sin(psi)
            l3 = l*(1 - sin(2psi)*mu)
        Thus mu is the cosine of inner angle of triangle between l1 and l2 sides.
    Lmax : int
        The highest multipole to compute.
    lmin : float
        The minimum value of l for the first bin.
    lmax : float
        The maximum value of l for the last bin.
    Nl : int
        The number of bins in l.
    Npsi : int, optional
        The total number of bins in psi. The bins are distributed logarithmically 
        and linearly between arctan(lmin/lmax) and pi/4. Default is 80.
    Nmu : int, optional
        The number of bins in mu. Default is 100.
    mupad : float, optional
        A small padding value added to the edges of the mu bins to avoid numerical
        issues with the limiting bispectrum configuration. Default is 1e-4.

    Attributes
    ----------
    Lmax : int
        The highest multipole to compute.
    l : ndarray
        An array of shape (Nl,) containing the values of l for each bin.
    psi : ndarray
        An array of shape (Npsi,) containing the values of psi for each bin.
    mu : ndarray
        An array of shape (Nmu,) containing the values of mu for each bin.
    bispectrum : callable
        The bispectrum function used for the computation.
    multipoles : ndarray
        An array of shape (Nl, Lmax+1) containing the bispectrum multipoles.
    """
    def __init__(self, bispectrum, Lmax, lmin, lmax, psimin, Nl=100, Npsi=80, Nmu=100, mupad=1e-4):
        # Define highest multipole
        self.Lmax = Lmax

        # Define bins
        self.l   = np.logspace(np.log10(lmin), np.log10(lmax), Nl)
        Npsi_low = Npsi//10
        Npsi_high= Npsi - Npsi_low
        self.psi = utils.loglinear(psimin, 1e-3, 1e-2, np.pi/4, Npsi_low, Npsi_high)
        self.mu  = np.linspace(-1+mupad, 1-mupad, Nmu)

        # Define bispectrum
        self.set_bispectrum(bispectrum)
        self.compute_mutlipoles()

    def set_bispectrum(self, bispectrum):
        self.bispectrum = bispectrum

    def compute_mutlipoles(self, method='linear'):
        """
        Compute the bispectrum multipoles.
        """
        l, psi, mu = np.meshgrid(self.l, self.psi, self.mu, indexing='ij')

        # Note that the bispectrum is specified by two sides and its inner angle,
        # while the multipole decomposition is defined with the outer angle.
        # Thus, we need a minus sign for mu.
        b = self.bispectrum(l, psi, -mu)

        # Compute multipoles
        L = np.arange(self.Lmax+1)
        m = decompose_multipole(mu, b, L, axis=2, method=method)
        self.multipoles_data = m

    def __single_L_call__(self, L, l, psi, extrapolate=False):
        """
        Compute the multipole decomposition of the bispectrum for a single value of L.

        Parameters
        ----------
        L : int
            The value of L.
        l : float or ndarray
            The value(s) of l.
        psi : float or ndarray
            The value(s) of psi.
        extrapolate : bool, optional
            If True, extrapolate the multipole decomposition outside the range of
            l and psi. Default is False.
        
        Returns
        -------
        multipole : float or ndarray
            The multipole decomposition of the bispectrum.
        """
        # test L
        if L > self.Lmax:
            raise ValueError(f"L = {L} > Lmax = {self.Lmax}")

        # cast
        l = np.asarray(l).copy()
        psi = np.asarray(psi).copy()

        # settings for interpolation
        if extrapolate:
            bounds_error = False
            fill_value = None
        else:
            bounds_error = True
            fill_value = np.nan
        
        # make interpolator
        x = np.log(self.l)
        y = np.log(self.psi)
        z = self.multipoles_data[L, :, :]
        f = rgi((x, y), z, bounds_error=bounds_error, fill_value=fill_value)

        # convert psi to pi/2-psi if psi > pi/4
        sel = np.pi/4 < psi
        psi[sel] = np.pi/2 - psi[sel]

        # compute interpolated multipole
        x = np.log(l)
        y = np.log(psi)
        return f((x, y))

    def __call__(self, L, l, psi, extrapolate=False):
        """
        Compute the multipole decomposition of the bispectrum.

        Parameters
        ----------
        L : int or ndarray
            The value(s) of L.
        l : float or ndarray
            The value(s) of l.
        psi : float or ndarray
            The value(s) of psi.
        extrapolate : bool, optional
            If True, extrapolate the multipole decomposition outside the range of
            l and psi. Default is False.
        
        Returns
        -------
        multipole : float or ndarray
            The multipole decomposition of the bispectrum.
        """

        if np.isscalar(L):
            return self.__single_L_call__(L, l, psi, extrapolate)
        else:
            out = []
            for _L in L:
                o = self.__single_L_call__(_L, l, psi, extrapolate)
                out.append(o)
            return np.array(out)

    def resum(self, l, psi, mu, return_terms=False):
        """
        Compute the resummed bispectrum.

        Parameters
        ----------
        l : float or ndarray
            The value(s) of l.
        psi : float or ndarray
            The value(s) of psi.
        mu : float or ndarray
            The value(s) of mu.

        Returns
        -------
        resummed : float or ndarray
            The resummed bispectrum.
        """
        L = np.arange(self.Lmax)
        out = []
        for _L in L:
            m = self.__call__(_L, l, psi)
            p = eval_legendre(_L, mu)
            out.append(m*p)

        # return 
        if return_terms:
            return np.sum(out, axis=0), np.array(out)
        else:
            return np.sum(out, axis=0)