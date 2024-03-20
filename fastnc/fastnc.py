#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/03/20 10:36:13

Description:
This is the module of fastnc, which calculate the
natural components using 2d fftlog.
'''
import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import RegularGridInterpolator as rgi
from tqdm import tqdm
from glob import glob
import pandas as pd
import os
# fastnc modules
from . import twobessel
from . import trigutils
from . import utils
from .interpolation import SemiDiagonalInterpolator as sdi
from .coupling import MCF222LegendreFourier, MCF222FourierFourier

class FastNaturalComponents:
    """
    Calculate the natural components using 2dfftlog.

    Supported projection of shear:
    - x: x-projection
    - cent: centroid-projection
    - ortho: orthocenter-projection
    """
    def __init__(self, Lmax, Mmax, bispectrum=None, verbose=True, config_bin=None):
        """
        Parameters
        ----------
        Lmax (int): The maximum multipole moment.
        Mmax (int): The maximum angular Fourier mode.
        bispectrum (Bispectrum, optional): The bispectrum object. Defaults to None.
        verbose (bool, optional): Whether to print the progress. Defaults to True.
        config_bin (dict, optional): The configuration for binning. Defaults to None.

        Notes on config_bin
        - auto (bool): Whether to automatically set ell12min, ell12max, and nell12bin. Defaults to True.
        - ell12min (float): The minimum value of ell12. Defaults to None. If auto is False, this must be specified.
        - ell12max (float): The maximum value of ell12. Defaults to None. If auto is False, this must be specified.
        - nell12bin (int): The number of bins for ell12. Defaults to 200.
        """
        # general setup
        self.verbose = verbose

        # initialize Lmax, Mmax
        self.Lmax = Lmax
        self.Mmax = Mmax

        # instantiate GLM calculator
        self.GLM = MCF222LegendreFourier(Lmax, Mmax, verbose=self.verbose)

        # 2DFFTLog config
        self.config_fftlog = {'nu1':1.01, 'nu2':1.01, 'N_pad':0}

        # set bispectrum
        if bispectrum is not None:
            self.set_bispectrum(bispectrum)

        # set bin
        self.config_bin = {'auto':True, 'ell12min':None, 'ell12max':None, 'nell12bin':150}
        if config_bin is not None:
            self.config_bin.update(config_bin)

        # flag for recomputation
        self.has_changed = True

    def set_bispectrum(self, bispectrum, **args):
        """
        Set and compute the bispectrum multipoles.

        bispectrum (Bispectrum): The bispectrum object.
        args (dict): The arguments for bispectrum.decompose.
        """
        # update bispectrum multipole
        self.bispectrum = bispectrum
        self.bispectrum.decompose(**args)
        self.set_bin()
        self.has_changed = True

    def set_bin(self):
        """
        Set the bin.
        """
        if self.config_bin['auto']:
            ell12min = self.bispectrum.ell12min
            ell12max = self.bispectrum.ell12max
        else:
            ell12min = self.config_bin['ell12min']
            ell12max = self.config_bin['ell12max']
        nell12bin= self.config_bin['nell12bin']
        self.ell1 = self.ell2 = np.logspace(np.log10(ell12min), np.log10(ell12max), nell12bin)
        # FFT grid in Fourier space
        self.ELL1, self.ELL2 = np.meshgrid(self.ell1, self.ell2, indexing='ij')
        self.ELL  = np.sqrt(self.ELL1**2 + self.ELL2**2)
        self.PSI = np.arctan2(self.ELL2, self.ELL1)
        # FFT grid in real space
        self.t1, self.t2 = 1/self.ell1[::-1], 1/self.ell2[::-1]
        self.T1, self.T2 = np.meshgrid(self.t1, self.t2, indexing='ij')
    
    def HM(self, M, ell, psi, bL=None, Lmin=None, Lmax=None, **args):
        """
        Compute H_M(l1, l2 = \\sum_L (-1)^L * G_LM * b_L(l1, l2).

        M (int): The angular Fourier mode.
        ell (array): The ell values.
        psi (array): The psi values.
        bL (array): The bispectrum multipole. Defaults to None.
                    If None, it is computed using self.bispectrum.kappa_bispectrum_multipole.
                    By supplying bL, you can avoid recomputation of bL.
        Lmin (int, optional): The minimum value of L. Defaults to None.
        Lmax (int, optional): The maximum value of L. Defaults to None.
        """
        # Get bispectrum multipole indices, L array
        Lmin = Lmin or 0
        Lmax = Lmax or self.Lmax
        L = np.arange(Lmin, Lmax+1)

        # Get bispectrum multipole
        if bL is None:
            bL = self.bispectrum.kappa_bispectrum_multipole(L, self.ELL, self.PSI, **args)

        # Sum up GLM*bL over L
        GLM = self.GLM(L, M, self.PSI)
        HM = np.sum(((-1)**(L+1)*GLM.T*bL.T).T, axis=0)
        return HM

    def __init_kernel_table(self, Mmax=None, Lmin=None, Lmax=None, **args):
        """
        Initialize kernel table.

        Mmax (int, optional): The maximum value of M. Defaults to None.
        """
        # natural-component multipole indices
        Mmax = Mmax or self.Mmax
        M = np.arange(Mmax+1)

        # bispectrum multipole indices
        Lmin = Lmin or 0
        Lmax = Lmax or self.Lmax
        L = np.arange(Lmin, Lmax+1)

        # bispectrum multipole
        bL = self.bispectrum.kappa_bispectrum_multipole(L, self.ELL, self.PSI, **args)

        # initialize table
        self.tabHM = dict()
        for _ in tqdm(M, desc='[HM]', disable=not self.verbose):
            HM = self.HM(_, self.ELL, self.PSI, bL=bL, Lmin=Lmin, Lmax=Lmax)
            self.tabHM[_] = HM

        # update flag
        self.has_changed = False
        
    def GammaM_on_grid(self, mu, M, dlnt=None, **args):
        """
        Compute Gamma^(M).
        
        mu (array): The index of the natural component.
        M (array): The angular Fourier mode.
        dlnt (float, optional): The bin width for t1 and t2. Defaults to None. 
                                When None, dlnt=0, i.e. no bin averaging effect

        GammaM is computed on FFT grid
        The output shape will be (mu.size, M.size, self.t1.size, self.t2.size)
        """
        # get kernel
        if self.has_changed:
            self.__init_kernel_table(**args)

        # casting to array
        if np.isscalar(mu):
            mu = np.array([mu])
        if np.isscalar(M):
            M = np.array([M])

        # Some GammaM are degenerating 
        # and we want to avoid recomputation for GammaM. 
        # Here we prepare the request for which GammaM 
        # to compute. We later assign the results.
        request = dict()
        for _mu in mu:
            for _M in M:
                if _M<0:
                    continue
                if _M not in request:
                    request[_M] = [_mu]
                else:
                    request[_M].append(_mu)

        # Compute
        tabGM = dict()
        for _M, mu_list in request.items():
            # Initialize 2D-FFTLog, this is shared for all mu to speed up.
            HM = self.tabHM[_M]
            tb  = twobessel.two_Bessel(self.ell1, self.ell2, HM*self.ELL1**2*self.ELL2**2, **self.config_fftlog)
            # Loop over mu
            for _mu in mu_list:
                # Get (n,m) from M.
                m, n = [(_M-3,-_M-3), (-_M-1,_M-1), (_M+1,-_M-3), (_M-3,-_M+1)][_mu]
                if dlnt is None:
                    # compute GammaM on FFT grid
                    GM = tb.two_Bessel(np.abs(m), np.abs(n))[2]
                elif dlnt is not None:
                    # compute GammaM on FFT grid with bin-averaging effect
                    GM = tb.two_Bessel_binave(np.abs(m), np.abs(n), dlnt, dlnt)[2]
                # Apply (-1)**m and (-1)**n
                # These originate to J_m(x) = (-1)^m J_{-m}(x)
                GM *= (-1.)**m if m<0 else 1
                GM *= (-1.)**n if n<0 else 1
                # normalization
                GM /= (2*np.pi)**3

                # store
                if _mu == 1:
                    tabGM[(_mu, -_M)] = GM.T
                else:
                    tabGM[(_mu, _M)] = GM

        # Assign
        GM = []
        for _mu in mu:
            _ = []
            for _M in M:
                if _mu == 0 and _M>=0:
                    _.append(tabGM[(_mu, _M)])
                if _mu == 0 and _M<0:
                    _.append(tabGM[(_mu, -_M)].T)
                if _mu == 1 and _M>0:
                    _.append(tabGM[(_mu, -_M)].T)
                if _mu == 1 and _M<=0:
                    _.append(tabGM[(_mu, _M)])
                if _mu == 2 and _M>=0:
                    _.append(tabGM[(_mu, _M)])
                if _mu == 2 and _M<0:
                    _.append(tabGM[(3, -_M)].T)
                if _mu == 3 and _M>=0:
                    _.append(tabGM[(_mu, _M)])
                if _mu == 3 and _M<0:
                    _.append(tabGM[(2, -_M)].T)
            GM.append(_)
        GM = np.array(GM)

        # return
        return GM

    def GammaM_on_bin(self, mu, M, t1, t2, dlnt=None, **args):
        """
        Compute Gamma^(M).
        
        mu (int): The index of the natural component.
        M (int): The angular Fourier mode.
        dlnt (float, optional): The bin width for t1 and t2. Defaults to None.
        t1 (array, optional): The value of t1. Defaults to None.
        t2 (array, optional): The value of t2. Defaults to None.
        Lmax (int, optional): The maximum value of L. Defaults to None.

        GammaM is computed on user-defined bin
        The output shape is (mu.size, M.size) + t1.shape
        """
        # get kernel
        if self.has_changed:
            self.__init_kernel_table(**args)

        # casting to array
        if np.isscalar(mu):
            mu = np.array([mu])
        if np.isscalar(M):
            M = np.array([M])

        # Compute
        tabGM = dict()
        for _M in M:
            # Initialize 2D-FFTLog, this is shared for all mu to speed up.
            if _M >= 0:
                HM = self.tabHM[_M]
            else:
                HM = self.tabHM[-_M].T
            tb  = twobessel.two_Bessel(self.ell1, self.ell2, HM*self.ELL1**2*self.ELL2**2, **self.config_fftlog)
            # Loop over mu
            for _mu in mu:
                # Get (n,m) from M.
                m, n = [(_M-3,-_M-3), (-_M-1,_M-1), (_M+1,-_M-3), (_M-3,-_M+1)][_mu]
                if dlnt is None:
                    # compute GammaM on user-defined grid
                    GM = tb.two_Bessel_on_bin(np.abs(m), np.abs(n), t1, t2)[2]
                if dlnt is not None:
                    # compute GammaM on user-defined grid with bin-averaging effect
                    GM = tb.two_Bessel_binave_on_bin(np.abs(m), np.abs(n), t1, t2, dlnt, dlnt)[2]
                # Apply (-1)**m and (-1)**n
                # These originate to J_m(x) = (-1)^m J_{-m}(x)
                GM *= (-1.)**m if m<0 else 1
                GM *= (-1.)**n if n<0 else 1
                # normalization
                GM /= (2*np.pi)**3

                # store
                if _mu == 1:
                    tabGM[(_mu, -_M)] = GM.T # This is wrong, need to be fixed
                else:
                    tabGM[(_mu, _M)] = GM

        # Assign
        GM = []
        for _mu in mu:
            _ = []
            for _M in M:
                _.append(tabGM[(_mu, _M)])
            GM.append(_)
        GM = np.array(GM)

        # return
        return GM

    def GammaM(self, mu, M, t1=None, t2=None, dlnt=None, **args):
        if t1 is None and t2 is not None:
            raise ValueError('Error: t1 is None but t2 is not None')
        if t1 is not None and t2 is None:
            raise ValueError('Error: t1 is not None but t2 is None')
        if t1 is None or t2 is None:
            GM = self.GammaM_on_grid(mu, M, dlnt=dlnt, **args)
        else:
            GM = self.GammaM_on_bin(mu, M, t1, t2, dlnt=dlnt, **args)
        return GM

    def Gamma(self, mu, phi, t1=None, t2=None, Mmax=None, dlnt=None, projection='x', **args):
        """
        Compute Gamma_mu(t1, t2, dphi)

        mu (int): The index of the natural component.
        phi (float): The value of phi.
        t1 (array, optional): The value of t1. Defaults to None.
        t2 (array, optional): The value of t2. Defaults to None.
        Mmax (int, optional): The maximum value of M. Defaults to None.
        projection (str, optional): The projection shear. Defaults to 'x'.
        """
        Mmax = Mmax or self.Mmax

        # casting to array
        if np.isscalar(mu):
            mu = np.array([mu])
        if np.isscalar(phi):
            phi = np.array([phi])

        # compute multipoles
        M       = np.arange(-Mmax, Mmax+1)
        GM      = self.GammaM(mu, M, t1=t1, t2=t2, dlnt=dlnt, **args)

        # resummation
        if t1 is not None and t2 is not None:
            GM      = np.reshape(GM, (len(mu), M.size, -1))
            expMphi = np.exp(1j*M[:,None]*np.reshape(phi,-1))
            Gamma   = np.einsum('imk,mk->ik', GM, expMphi)/(2*np.pi)
            Gamma   = np.reshape(Gamma, (len(mu),)+t1.shape)
            Gamma  *= self.projection_factor(mu, t1, t2, phi, projection)
        else:
            expMphi = np.exp(1j*M[:,None]*phi)
            Gamma   = np.einsum('im...,mk->ik...', GM, expMphi)/(2*np.pi)
            Gamma  *= self.projection_factor(mu, phi[:,None,None], self.T1, self.T2, projection)

        return Gamma
    
    # multiplicative phase factor to convert between different projections
    def projection_factor(self, i, phi, t1=None, t2=None, projection='x'):
        """
        Compute the projection factor.

        i (int): The index of the natural component.
        t1 (array): The value of t1.
        t2 (array): The value of t2.
        phi (float): The value of phi.
        projection (str, optional): The projection shear. Defaults to 'x'.
        """
        if t1 is None:
            t1 = self.T1
        if t2 is None:
            t2 = self.T2

        # Compute projection factor
        if projection == 'x':
            factor = 1
        elif projection == 'cent':
            factor = x2cent(i, t1, t2, phi)
        elif projection == 'ortho':
            factor = x2ortho(i, t1, t2, phi)
        else:
            raise ValueError('Error: projection={} is not expected'.format(projection))

        # return
        return factor

# phase factors to convert between different projections
def x2ortho(i, t1, t2, phi):
    # Compute prefactor
    sin2pb, cos2pb = utils.sincos2angbar(np.arctan2(t2, t1), np.pi-phi)
    if i==0 or i==1 or i==2:
        out = cos2pb - 1j*sin2pb
    elif i==3:
        out = cos2pb + 1j*sin2pb
    return out

def ortho2cent(i, t1, t2, phi):
    t1, t2, t3 = trigutils.x1x2phi_to_x1x2x3(t1, t2, phi)

    def temp(t1, t2, t3):
        phi3 = np.arccos( (t1**2+t2**2-t3**2)/2/t1/t2 )
        cos2psi = ((t2**2-t1**2)**2 - 4*t1**2*t2**2*np.sin(phi3)**2)/4.0
        sin2psi = (t2**2-t1**2) * t1*t2 * np.sin(phi3)
        norm = np.sqrt(cos2psi**2 + sin2psi**2)
        exp2psi = cos2psi/norm + 1j*sin2psi/norm
        return exp2psi

    exp2psi3 = temp(t1, t2, t3)
    exp2psi1 = temp(t2, t3, t1)
    exp2psi2 = temp(t3, t1, t2)

    out = 1
    for j, phase in enumerate([1.0, exp2psi1, exp2psi2, exp2psi3]):
        if j == i:
            out *= phase
        else:
            out *= np.conj(phase)

    return out

def x2cent(mu, t1, t2, phi):
    # Equations between Eq. (15) and (16) 
    # of https://arxiv.org/abs/2309.08601
    v = t1+t2*np.exp(-1j*phi)
    q1 = v/np.conj(v)
    v = -2*t1+t2*np.exp(-1j*phi)
    q2 = v/np.conj(v)
    v = t1-2*t2*np.exp(-1j*phi)
    q3 = v/np.conj(v)

    if np.isscalar(mu):
        mu = [mu]

    out = []
    for i in mu:
        if i==0:
            o = q1*q2*q3 * np.exp(3j*phi)
        elif i==1:
            o = np.conj(q1)*q2*q3 * np.exp(1j*phi)
        elif i==2:
            o = q1*np.conj(q2)*q3 * np.exp(3j*phi)
        elif i==3:
            o = q1*q2*np.conj(q3) * np.exp(-1j*phi)
        else:
            raise ValueError('Error: i={} is not expected'.format(i))
        out.append(o)
    out = np.array(out)

    return out