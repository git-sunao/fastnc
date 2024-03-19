#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/03/19 15:16:05

Description:
multipole.py contains the Multipole class, 
which computes multipole moments
'''
import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import RegularGridInterpolator as rgi

class MultipoleBase:
    def __init__(self, x, Lmax, method='gauss-legendre', verbose=True):
        """
        Compute multipole of a function f(x).
        using some basis function indexed by L.
        The maximum multipole must be specified by Lmax.

        x (array)     : x values
        Lmax (int)    : maximum multipole moment
        method (str)  : method to compute the multipole moments. 
                        options are'gauss-legendre' or 'riemann'.
        verbose (bool): whether to print verbose output
        """
        self.x = x
        self.Lmax = Lmax
        self.method = method
        self.verbose = verbose
        self._init_basis_function()

    def _init_basis_function(self):
        """
        Initialize the basis function.
        used to compute the multipoles.

        This allows precomputing the basis function
        if necessary, to save computation time.
        """
        pass

    def _get_basis_function(self, L):
        """
        Return the basis function for multipole indices L

        L (array): multipole moment
        """
        NotImplementedError

    def __get_linear_interp_coeffs(self, fx, axis=0):
        """
        Get the coefficients for a linear interpolation of y(x).
        The function is interpolated as
        ..math::
            f(x) = a_i x + b_i 
        in each bin.

        fx (array): fx values evaluated at self.x
        """
        assert self.x.size == fx.shape[axis], "shape of x and fx must be the same"
        fx = np.moveaxis(fx, axis, -1)
        a = np.diff(fx, axis=-1)/np.diff(self.x)
        b = (fx*np.roll(self.x,-1) - np.roll(fx,-1,axis=-1)*self.x)
        b = np.delete(b, -1, axis=-1)
        b = b/np.diff(self.x)
        a, b = np.moveaxis(a, -1, axis), np.moveaxis(b, -1, axis)
        return a, b

    def decompose(self, fx, L, axis=0):
        if self.method == 'gauss-legendre':
            return self.__decompose_gauss_legendre(fx, L, axis=axis)
        elif self.method == 'riemann':
            return self.__decompose_riemann(fx, L, axis=axis)
        else:
            raise ValueError(f"method {self.method} is not supported")

    def __decompose_gauss_legendre(self, fx, L, axis=0):
        """
        Decompose a function f(x) into multipole moments
        using gauss-legendre method with coefficients of
        linear interpolation.

        fx (array): fx values evaluated at self.x
        L (array): multipole moment
        """
        a, b = self.__get_linear_interp_coeffs(fx, axis=axis)
        w0, w1 = self._get_basis_function(L)
        out = np.tensordot(w1, a, axes=([1], [axis]))
        out+= np.tensordot(w0, b, axes=([1], [axis]))
        return out

    def __decompose_riemann(self, fx, L, axis=0):
        """
        Decompose a function f(x) into multipole moments
        by approximating the integral with a Riemann sum.
        Note: This is **not** recommended, because this can create 
        artificial high multipole moments, even though the 
        true high multipoles are zeros.

        fx (array): fx values evaluated at self.x
        L (array): multipole moment
        """
        w = self._get_basis_function(L)
        out = np.tensordot(w, f, axes=([1], [axis])) * (self.x[1]-self.x[0])
        return out

class MultipoleLegendre(MultipoleBase):
    # MultipoleLegendre class inherits MultipoleBase class
    def _init_basis_function(self):
        if self.method == 'gauss-legendre':
            Lmax = self.Lmax + 2
            self.__init_legendreP_table(Lmax)
        elif self.method == 'riemann':
            Lmax = self.Lmax
            self.__init_legendreP_table(Lmax)
        else:
            raise ValueError(f"method {self.method} is not supported")

    def _get_basis_function(self, L):
        """
        Return the basis function for multipole indices L

        L (array): multipole moment
        """
        if self.method == 'gauss-legendre':
            # Note: Here axis is incremented by 1, because
            # L is added as the first axis on top of self.x.shape.
            w0 = 0.5*np.diff(self.__get_legendreP_int0(L), axis=1)
            w1 = 0.5*np.diff(self.__get_legendreP_int1(L), axis=1)
            basis = (w0, w1)
        elif self.method == 'rieamnn':
            basis = (2*L[:,None]+1)/2*self.__get_legendreP(L)
        else:
            raise ValueError(f"method {self.method} is not supported")
        return basis

    # Legendre polynomial P_L(x) related functions
    def __init_legendreP_table(self, Lmax):
        """
        Precompute the Legendre polynomials P_L(x) for all L, 
        to save computation time.

        Lmax (int): maximum multipole moment
        """
        self.legendreP_table = dict()
        for L in np.arange(Lmax+1):
            p = eval_legendre(L, self.x)
            self.legendreP_table[L] = p
    
    def __get_legendreP(self, L):
        """
        Return legendre polynomial P_L(x)

        L (array): multipole moment
        """
        if np.isscalar(L):
            L = np.array([L])
        
        out = np.zeros(L.shape+self.x.shape)
        for i, _L in enumerate(L):
            if _L<0:
                continue
            out[i,:] = self.legendreP_table[_L]
        return out

    def __get_legendreP_int0(self, L):
        """
        Return integral of legendre polynomial P_L(x)
        """
        out = self.__get_legendreP(L+1) - self.__get_legendreP(L-1)
        # assign L=0
        out[L==0,:] = self.x
        return out

    def __get_legendreP_int1(self, L):
        """
        Return integral of legendre polynomial P_L(x) with x:
        p_L^1 = (2L+1)\\int dx P_L(x) x

        L (array): multipole moment
        """
        pL   = self.__get_legendreP(L)
        pLm1 = self.__get_legendreP(L-1)
        pLm2 = self.__get_legendreP(L-2)
        pLp1 = self.__get_legendreP(L+1)
        pLp2 = self.__get_legendreP(L+2)
        out = (pL - pLm2)/(2*L[:,None]-1) \
            + self.x[None,:]*(pLp1 - pLm1) \
            + (pL - pLp2)/(2*L[:,None]+3)
        # assign L=0, 1
        out[L==0,:] = self.x**2/2
        out[L==1,:] = self.x**3
        return out
    
class MultipoleFourier(MultipoleBase):
    # MultipoleFourier class inherits MultipoleBase class
    def _init_basis_function(self):
        """
        Fourier basis is computationally low cost,
        so no need to precompute the basis function.
        """
        pass

    def _get_basis_function(self, L):
        """
        Return the basis function for multipole indices L

        L (array): multipole moment
        """
        if self.method == 'gauss-legendre':
            w0 = np.zeros(L.shape+self.x.shape, dtype=complex)
            w1 = np.zeros(L.shape+self.x.shape, dtype=complex)
            # L == 0
            w0[L==0,:] = self.x[None,:]
            w1[L==0,:] = self.x[None,:]**2/2
            # L != 0
            iL = 1j*L[L!=0,None]
            w0[L!=0,:] = np.exp(iL*self.x[None,:])/iL
            w1[L!=0,:] = (self.x[None,:]-1.0/iL)/iL * np.exp(iL*self.x[None,:])
            # diff
            w0 = np.diff(w0, axis=1)
            w1 = np.diff(w1, axis=1)
            basis = (w0, w1)
        elif self.method == 'rieamnn':
            basis = np.exp(1j*L[:,None]*self.x[None,:])
        else:
            raise ValueError(f"method {self.method} is not supported")
        return basis

class MultipoleSin(MultipoleBase):
    def _init_basis_function(self):
        """
        Sin basis is computationally low cost,
        so no need to precompute the basis function.
        """
        pass

    def _get_basis_function(self, L):
        """
        Return the basis function for multipole indices L

        L (array): multipole moment
        """
        if self.method == 'gauss-legendre':
            w0 = np.zeros(L.shape+self.x.shape)
            w1 = np.zeros(L.shape+self.x.shape)
            # L == 0
            w0[L==0,:] = 0
            w1[L==0,:] = 0
            # L != 0
            _L, _x = L[L!=0,None], self.x[None,:]
            w0[L!=0,:] = -np.cos(_L*_x)/_L
            w1[L!=0,:] = -_x*np.cos(_L*_x)/_L + np.sin(_L*_x)/_L**2
            # diff
            w0 = np.diff(w0, axis=1)
            w1 = np.diff(w1, axis=1)
            basis = (w0, w1)
        elif self.method == 'rieamnn':
            basis = np.sin(L[:,None]*self.x[None,:])
        else:
            raise ValueError(f"method {self.method} is not supported")
        return basis

class MultipoleCos(MultipoleBase):
    def _init_basis_function(self):
        """
        Cos basis is computationally low cost,
        so no need to precompute the basis function.
        """
        pass

    def _get_basis_function(self, L):
        """
        Return the basis function for multipole indices L

        L (array): multipole moment
        """
        if self.method == 'gauss-legendre':
            w0 = np.zeros(L.shape+self.x.shape)
            w1 = np.zeros(L.shape+self.x.shape)
            # L == 0
            w0[L==0,:] = self.x[None,:]
            w1[L==0,:] = self.x[None,:]**2/2
            # L != 0
            _L, _x = L[L!=0,None], self.x[None,:]
            w0[L!=0,:] = np.sin(_L*_x)/_L
            w1[L!=0,:] = _x*np.sin(_L*_x)/_L - np.cos(_L*_x)/_L**2
            # diff
            w0 = np.diff(w0, axis=1)
            w1 = np.diff(w1, axis=1)
            basis = (w0, w1)
        elif self.method == 'rieamnn':
            basis = np.cos(L[:,None]*self.x[None,:])
        else:
            raise ValueError(f"method {self.method} is not supported")
        return basis
