#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/02/12 21:24:48

Description:
multipole.py contains the Multipole class, 
which computes multipole moments
'''
import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import RegularGridInterpolator as rgi
from . import utils
from tqdm import tqdm

class Multipole:
    def __init__(self, x, Lmax, method='linear', verbose=True):
        """
        Compute multipole moments of a function f(x).

        x (array): x values
        Lmax (int): maximum multipole moment
        method (str): method to compute the multipole moments
        verbose (bool): whether to print verbose output
        """
        self.method = method
        self.x      = x
        self.Lmax   = Lmax
        self.verbose = verbose
        self.init_legendreP_table()

    def init_legendreP_table(self):
        """
        Precompute the Legendre polynomials P_L(x) for all L, 
        to save computation time.
        """

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
        """
        Return legendre polynomial P_L(x)

        L (int): multipole moment
        """
        return self.legendreP[L]

    def get_legendreP_int0(self, L):
        """
        Return integral of legendre polynomial P_L(x):
        p_L^0 = (2L+1)\\int dx P_L(x)

        L (int): multipole moment
        """
        if L == 0:
            out = self.x
        else:
            out = self.get_legendreP(L+1) - self.get_legendreP(L-1)
        return out

    def get_legendreP_int1(self, L):
        """
        Return integral of legendre polynomial P_L(x) with x:
        p_L^1 = (2L+1)\\int dx P_L(x) x

        L (int): multipole moment
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

        x (array): x values
        y (array): y values
        axis (int): axis along which to interpolate
        """
        a = np.diff(y, axis=axis)/np.diff(x,axis=axis)
        b = (y*np.roll(x,-1,axis=axis) - np.roll(y,-1,axis=axis)*x)
        b = np.delete(b, -1, axis=axis)
        b = b/np.diff(x,axis=axis)
        return a, b
    
    def _decompose_linear(self, y, L, axis=0):
        """
        Decompose a function y=f(x) into multipole moments
        by approximating the integral using a linear interpolation
        coefficients.

        y (array): y values
        L (int): multipole moment
        axis (int): axis along which to interpolate
        """
        a, b = self.get_linear_interp_coeffs(self.x, y, axis=axis)

        p_L_0 = np.diff(self.get_legendreP_int0(L), axis=axis)
        p_L_1 = np.diff(self.get_legendreP_int1(L), axis=axis)

        out = 0.5*np.sum(a*p_L_1 + b*p_L_0, axis=axis)

        return out

    def _decompose_riemann(self, f, L, axis=0):
        """
        Decompose a function y=f(x) into multipole moments
        by approximating the integral with a Riemann sum.
        Note: This is not recommended, because this can create 
        artificial high multipole moments, even though the 
        true high multipoles are zeros.

        y (array): y values
        L (int): multipole moment
        axis (int): axis along which to interpolate
        """
        p = self.get_legendreP(L)
        out = np.sum(f*p,axis=axis) * (2*L+1)/2 * (self.x[1]-self.x[0])
        return out

    def _decompose(self, f, L, axis=0):
        """
        Decompose a function y=f(x) into multipole moments.

        y (array): y values
        L (int): multipole moment
        axis (int): axis along which to interpolate
        """
        if self.method == 'linear':
            return self._decompose_linear(f, L, axis=axis)
        elif self.method == 'riemann':
            return self._decompose_riemann(f, L, axis=axis)
    
    def decompose(self, f, L, axis=0):
        """
        Decompose a function y=f(x) into multipole moments.

        y (array): y values
        L (int or array): multipole moment
        axis (int): axis along which to interpolate
        """
        if np.isscalar(L):
            return self._decompose(f, L, axis=axis)
        else:
            out = []
            pbar = tqdm(L, desc='[multipole]', disable=not self.verbose)
            for l in pbar:
                pbar.set_postfix({'L':l})
                out.append(self._decompose(f, l, axis=axis))
            return np.array(out)