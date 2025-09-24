#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/11/27 18:02:25

Description:
bispectrum.py contains classes for computing bispectrum 
and various methods of bispectrum: interpolation, 
multipole decomposition, etc.
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RegularGridInterpolator as rgi
# fastnc modules
from .halofit import get_Rb_bihalofit

# Base class
class BaryonModelBase:
    def __init__(self):
        pass

    def __call__(self, k1, k2, k3, z):
        """
        Return multiplicative factor to matter bispectrum 
        """
        return 1.0


# Models
class BaryonTakahashiTNGfit(BaryonModelBase):
    """
    Fitting formula built in Takahashi+
    using the bispectrum measured from TNG simulation
    With additional parameter (fb) to rescale 
    the suppresion factor.
    """
    def __init__(self, fb=1.0, suponly=False):
        self.suponly = suponly
        self.set_param(fb)
        
    def set_param(self, fb):
        self.fb = fb
        
    def __call__(self, k1, k2, k3, z):
        if self.fb == 0.0:
            return 1.0
        else:
            s = get_Rb_bihalofit(k1,k2,k3,z)
            # Force no enhancement
            if self.suponly:
                s[s>1.0] = 1.0
            return 1.0 + fb * (s-1.0)


class SeparableInterpSVD:
    def __init__(self, x, y, f, tol=1e-10, mmax=None, 
                 xl=None, xr=None, yl=None, yr=None, grid=True):
        U, S, V = np.linalg.svd(f, full_matrices=False)

        # Determin mmax
        if mmax is None:
            # compute error Frobenius
            sq = S**2
            ratio = np.cumsum(sq) / sq.sum()
            self.mmax = int(np.searchsorted(ratio, 1 - tol) + 1)
        else:
            self.mmax = mmax
        
        self.U   = U[:,:mmax]
        self.S   = S[:mmax]
        self.V   = V[:mmax,:]
        self.x   = x
        self.y   = y

        self.xl  = xl
        self.xr  = xr
        self.yl  = yl
        self.yr  = yr
        
        self.grid= grid
    
    def __call__(self, x, y):
        if self.grid:
            out = np.zeros((x.size, y.size))
        else:
            out = 0.0
            
        for m in range(self.mmax):
            u   = self.U[:,m]
            s   = self.S[m]
            v   = self.V[m,:]

            ui  = np.interp(x, self.x, u, left=self.xl, right=self.xr)
            vi  = np.interp(y, self.y, v, left=self.yl, right=self.yr)
            out+= s * ui * vi
        return out


class BaryonTransfer(BaryonModelBase):
    """
    Transfer-function-based modeling:
    The suppression factor is modeled by

    R(k1,k2,k3,z) = T(k1, z) * T(k2,z) * T(k3, z)
    """
    def __init__(self, k, z, tkz):
        self.make_tkz(k, z, tkz)

    def make_tkz(self, k, z, tkz):
        self.tkz = SeparableInterpSVD(np.log(k), z, tkz, grid=False)

    def __call__(self, k1, k2, k3, z):
        t1 = self.tkz(np.log(k1), z)
        t2 = self.tkz(np.log(k2), z)
        t3 = self.tkz(np.log(k3), z)
        return t1*t2*t3
