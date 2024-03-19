#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/03/19 15:12:50

Description:
bispectrum.py contains classes for computing bispectrum 
and various methods of bispectrum: interpolation, 
multipole decomposition, etc.
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RegularGridInterpolator as rgi
from astropy.cosmology import wCDM, Planck18
from scipy.special import sici
from scipy.special import eval_legendre
from time import time

from . import trigutils
from .halofit import Halofit
from .multipole import Multipole, MultipoleLegendre
from .utils import loglinear, edge_correction


wPlanck18 = wCDM(H0=Planck18.H0, Om0=Planck18.Om0, Ode0=Planck18.Ode0, w0=-1.0, meta=Planck18.meta, name='wPlanck18')

class BispectrumBase:
    """
    Base class for bispectrum computation.

    Usage:
    >>> b = BispectrumBase()
    >>> b.set_cosmology(cosmo)
    >>> b.set_source_distribution(zs, pzs)
    >>> b.set_ell12mu_range(ell12min, ell12max, epmu)
    >>> b.interpolate(nrbin=35, nubin=25, nvbin=25, method='linear', nzbin=20)
    >>> b.decompose(Lmax, nellbin=100, npsibin=50, nmubin=50, method_decomp='linear', method_bispec='interp')
    >>> b.kappa_bispectrum_multipole(L, ell, psi)
    >>> b.kappa_bispectrum_resum(ell1, ell2, ell3)
    """
    # The predefined support range of ell1, ell2, mu
    # Can be set by the user from set_ell12mu_range method
    ell12min = None
    ell12max = None
    epmu     = 1e-7

    def __init__(self, cosmo=None, zs=None, pzs=None):
        """
        cosmo (astropy.cosmology): cosmology
        zs (array): redshift array
        pzs (array): probability distribution of source galaxies

        If the source redshift distribution is a single redshift,
        i.e. delta function distribution, zs and pzs can be scalar.
        """
        self.set_cosmology(cosmo or wPlanck18)
        if zs is None and pzs is None:
            zs, pzs = [1,2], [1,1]
            print(f'setting a default source distribution: zs={zs}, pzs={pzs}')
        self.set_source_distribution(zs, pzs)
        
        # defined the support range of ell1, ell2
        if self.ell12min is not None and self.ell12max is not None and self.epmu is not None:
            self.set_ell12mu_range(self.ell12min, self.ell12max, self.epmu)

    # Setter
    def set_cosmology(self, cosmo):
        """
        cosmo (astropy.cosmology): cosmology
        """
        self.cosmo = cosmo

        # compute array of chi and z
        z   = np.linspace(0, 5, 100)
        chi = self.cosmo.comoving_distance(z).value * self.cosmo.h # Mpc/h

        # spline chi <-> z
        self.z2chi = ius(z, chi)
        self.chi2z = ius(chi, z)
        self.has_changed = True

    def set_source_distribution(self, zs_list, pzs_list, sample_names=None):
        """
        Set source distribution.

        zs_list (list): redshift array
        pzs_list (list): probability distribution of source galaxies
        """
        # setting attributes
        self.n_sample = len(zs_list)

        # names of source samples
        if sample_names is None:
            sample_names = [str(i) for i in range(self.n_sample)]
        self.sample_names= sample_names

        # casting and shape check
        self.zs_dict = dict()
        self.pzs_dict = dict()
        for i, name in enumerate(self.sample_names):
            self.zs_dict[name] = np.asarray(zs_list[i])
            self.pzs_dict[name] = np.asarray(pzs_list[i])
            assert self.zs_dict[name].size == self.pzs_dict[name].size, "zs and pzs must have the same length"

        # rise flag
        self.has_changed = True

    def compute_lensing_kernel_per_sample(self, zs, pzs, nzlbin=101):
        """
        Set source distribution.

        zs (array): redshift array
        pzs (array): probability distribution of source galaxies
        """
        if zs.size == 1:
            zl = np.linspace(0.0, zs, nzlbin)
            chil = self.z2chi(zl)
            chis = self.z2chi(zs)
            g = 1.-chil/chis
            z2g = ius(zl, g, ext=1)
            chi2g = ius(chil, g, ext=1)
        else:
            zl = np.linspace(0, zs.max(), nzlbin)
            chil = self.z2chi(zl)
            chis = self.z2chi(zs)
            CHIL, CHIS = np.meshgrid(chil, chis, indexing='ij')

            # integrand
            I = np.ones_like(CHIL, dtype=float)
            I = np.divide(CHIL, CHIS, out=I, where=CHIS > CHIL)
            I = (pzs*(1-I))

            g = np.trapz(I, zs, axis=1)/np.trapz(pzs, zs)
            z2g = ius(zl, g, ext=1)
            chi2g = ius(chil, g, ext=1)
        return z2g, chi2g

    def compute_lensing_kernel(self, nzlbin=101):
        self.z2g_dict = dict()
        self.chi2g_dict = dict()
        for name in self.sample_names:
            z2g, chi2g = self.compute_lensing_kernel_per_sample(self.zs_dict[name], self.pzs_dict[name], nzlbin)
            self.z2g_dict[name] = z2g
            self.chi2g_dict[name] = chi2g
        self.zmax = max([self.zs_dict[name].max() for name in self.sample_names])

    def get_all_sample_combinations(self):
        combinations = []
        for i in range(self.n_sample):
            name_i = self.sample_names[i]
            for j in range(i, self.n_sample):
                name_j = self.sample_names[j]
                for k in range(j, self.n_sample):
                    name_k = self.sample_names[k]
                    combinations.append((name_i, name_j, name_k))
        return combinations

    def set_ell12mu_range(self, ell12min, ell12max, epmu):
        """
        Set support range of ell1, ell2, mu, where
        ell1 and ell2 are the two side lengths of the triangle,
        and mu is the cosine of the **outer** angle of the triangle 
        between ell1 and ell2. Thus the other side length ell3 can be
        computed from ell1, ell2, and mu:

            ell3 = (ell1**2 + ell2**2 + 2*ell1*ell2*mu)**0.5
        
        We avoid the squeezed limit of triangle by setting the minimum
        of mu to be 1-epmu, where epmu is a small number.
        """
        # fastnc args convention
        self.ell12min = ell12min
        self.ell12max = ell12max
        self.mumin    = -1.0
        self.mumax    = 1.0 - epmu

        # Multipole decomposition args convention.
        self.ellmin = 2**0.5 * ell12min
        self.ellmax = 2**0.5 * ell12max
        self.psimin = min(np.arctan2(ell12min, ell12max), np.pi/2 - np.arctan(ell12max/ell12min))
        self.psimax = np.pi/4

        # Interpolation args convention.
        # When ell, psi, mu run over the lectangular region
        # defined by the above ranges, the ranges of r, u, v
        # are also given by the ranges on ell, psi, mu.
        self.rmin = self.ellmin*min(5**-0.5, (1-(1-epmu)*2/5)**0.5)
        self.rmax = self.ellmax*max(2**-0.5, np.cos(self.psimin))
        self.umin = min(2**0.5*epmu**0.5, np.tan(self.psimin))
        self.umax = 1.0
        self.vmin = 0.0
        self.vmax = 1.0

    # Spectra methods
    # matter power spectrum (to be implemented in subclasses)
    def matter_bispectrum(self, k1, k2, k3, z):
        raise NotImplementedError

    # kappa bispectrum interface
    def kappa_bispectrum(self, ell1, ell2, ell3, sample_combination=(0,0,0), method='direct', **args):
        """
        Compute kappa bispectrum.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        method (str): method for computing kappa bispectrum (direct, interp, resum)
        """
        if method == 'direct':
            return self.kappa_bispectrum_direct(ell1, ell2, ell3, sample_combination, **args)
        elif method == 'interp':
            return self.kappa_bispectrum_interp(ell1, ell2, ell3, sample_combination)
        elif method == 'resum':
            return self.kappa_bispectrum_resum(ell1, ell2, ell3, sample_combination, **args)
        else:
            raise ValueError("method must be 'direct', 'interp', or 'resum'")
        
    # direct evaluation of kappa bispectrum from matter bispectrum
    def kappa_bispectrum_direct(self, ell1, ell2, ell3, sample_combination=(0,0,0), zmin=1e-4, nzbin=30, bm=None, return_bm=False, **args):
        """
        Compute kappa bispectrum by direct line-of-sight integration.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        zmin (float): minimum redshift
        nzbin (int): number of redshift bins
        args (dict): arguments for matter_bispectrum
        """
        # check scalar
        isscalar = np.isscalar(ell1)
        if isscalar:
            ell1 = np.array([ell1])
            ell2 = np.array([ell2])
            ell3 = np.array([ell3])

        # check shape
        if ell1.shape != ell2.shape or ell1.shape != ell3.shape:
            raise ValueError("l1, l2, l3 must have the same shape")

        # save input shape
        shape = ell1.shape

        # reshape to 1d
        ell1 = ell1.ravel()
        ell2 = ell2.ravel()
        ell3 = ell3.ravel()

        # compute lensing weight, encoding geometrical dependence.
        z = np.logspace(np.log10(zmin), np.log10(self.zmax), nzbin)
        chi = self.z2chi(z)
        weight = 1
        for name in sample_combination:
            weight *= self.chi2g_dict[str(name)](chi)
        weight *= 1.0/chi*(1+z)**3

        # create grids
        ELL1, Z = np.meshgrid(ell1, z, indexing='ij')
        ELL2, Z = np.meshgrid(ell2, z, indexing='ij')
        ELL3, Z = np.meshgrid(ell3, z, indexing='ij')
        CHI = self.z2chi(Z)
        K1, K2, K3 = ELL1/CHI, ELL2/CHI, ELL3/CHI

        # compute matter bispectrum
        if bm is None:
            bm = self.matter_bispectrum(K1, K2, K3, Z, **args)

        # integrand
        i = weight * bm

        # integrate
        bk = np.trapz(i, chi, axis=1)

        # Multiply prefactor
        bk *= (3/2 * (100/299792)**2 * self.cosmo.Om0)**3

        # reshape to the original shape
        bk = bk.reshape(shape)

        # convert to scalar if input is scalar
        if isscalar:
            bk = bk[0]

        if return_bm:
            return bk, bm
        else:
            return bk

    # interpolation
    def interpolate(self, nrbin=35, nubin=35, nvbin=25, method='linear', nzbin=30, data=None, sample_combinations=None, **args):
        """
        Interpolate kappa bispectrum. 
        The interpolation is done in (r,u,v)-space, which is defined in M. Jarvis+2003 
        (https://arxiv.org/abs/astro-ph/0307393). See also treecorr homepage
        (https://rmjarvis.github.io/TreeCorr/_build/html/correlation3.html).

        nrbin (int): number of r bins
        nubin (int): number of u bins
        nvbin (int): number of v bins
        method (str): method for interpolation
        nzbin (int): number of redshift bins
        args (dict): arguments for matter_bispectrum
        """
        r = np.logspace(np.log10(self.rmin), np.log10(self.rmax), nrbin)
        u = np.logspace(np.log10(self.umin), np.log10(self.umax), nubin)
        v = np.linspace(self.vmin, self.vmax, nvbin)

        R, U, V = np.meshgrid(r, u, v, indexing='ij')
        ell1, ell2, ell3 = trigutils.ruv_to_x1x2x3(R, U, V)

        if sample_combinations is None:
            sample_combinations = self.get_all_sample_combinations()

        self.ip = dict()
        bm = None
        for sample_combination in sample_combinations:
            bk, bm = self.kappa_bispectrum_direct(
                ell1, ell2, ell3, 
                nzbin=nzbin, 
                sample_combination=sample_combination,
                bm=bm, return_bm=True, **args)
            self.ip[sample_combination] = rgi((np.log(r), np.log(u), v), np.log(bk), method=method)

    def kappa_bispectrum_interp(self, ell1, ell2, ell3, sample_combination=(0,0,0)):
        """
        Compute kappa bispectrum by interpolation.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        """
        ip = self.ip[sample_combination]
        r, u, v = trigutils.x1x2x3_to_ruv(ell1, ell2, ell3, signed=False)
        x = edge_correction(np.log(r), ip.grid[0].min(), ip.grid[0].max())
        y = edge_correction(np.log(u), ip.grid[1].min(), ip.grid[1].max())
        z = edge_correction(v, ip.grid[2].min(), ip.grid[2].max())
        return np.exp(ip((x,y,z)))

    # multipole decomposition
    def init_multipole(self, Lmax, mu, method='gauss-legendre'):
        """
        Initialize multipole decomposition.

        Lmax (int): maximum multipole
        MU (array): mu array
        method (str): method for multipole decomposition
        """
        if (not hasattr(self, 'multipole')) or self.multipole.Lmax != Lmax or self.multipole.x.shape != mu.shape or np.any(self.multipole.x != mu) or self.multipole.method != method:
            self.multipole = MultipoleLegendre(mu, Lmax, method='gauss-legendre', verbose=True)

    def decompose(self, Lmax, nellbin=100, npsibin=80, nmubin=50, window=None, 
            sample_combinations=None,
            method_decomp='gauss-legendre', method_bispec='interp', **args):
        """
        Compute multipole decomposition of kappa bispectrum.

        Lmax (int): maximum multipole
        nellbin (int): number of ell bins
        npsibin (int): number of psi bins on linear part
        nmubin (int): number of mu bins on linear part
        window (array): window function in Fourier space, e.g. pixel window function.
        method_decomp (str): method for multipole decomposition
        method_bispec (str): method for computing bispectrum
        args (dict): arguments for kappa_bispectrum
        """
        ell = np.logspace(np.log10(self.ellmin), np.log10(self.ellmax), nellbin)
        psi = loglinear(self.psimin, 1e-3, self.psimax, 50, npsibin)
        mu = 1-loglinear(1-self.mumax, 5e-2, 1-self.mumin, 30, nmubin)[::-1] # capture the squeezed limit
        ELL, SPI, MU = np.meshgrid(ell, psi, mu, indexing='ij')
        self.init_multipole(Lmax, mu, method_decomp)

        ELL1, ELL2, ELL3 = trigutils.xpsimu_to_x1x2x3(ELL, SPI, MU)

        if sample_combinations is None:
            sample_combinations = self.get_all_sample_combinations()

        self.multipoles_data = {'ell':ell, 'psi':psi, 'bL':dict()}
        for sample_combination in sample_combinations:
            b = self.kappa_bispectrum(ELL1, ELL2, ELL3, sample_combination, method=method_bispec, **args)

            if window is not None:
                b*= window(ELL1, ELL2, ELL3)

            # Compute multipoles
            L = np.arange(Lmax+1)
            bL = self.multipole.decompose(b, L, axis=2)
            self.multipoles_data['bL'][sample_combination] = bL

    def _kappa_bispectrum_multipole(self, L, ell, psi, sample_combination=('0','0','0')):
        """
        Compute multipole of kappa bispectrum.

        L (int): multipole
        ell (array): ell array
        psi (array): psi array
        """
        Lmax = self.multipoles_data['bL'][sample_combination].shape[0]
        if L > Lmax:
            raise ValueError("L must be less than Lmax={}".format(Lmax))

        x = np.log(self.multipoles_data['ell'])
        y = np.log(self.multipoles_data['psi'])
        z = self.multipoles_data['bL'][sample_combination][L, :, :]
        f = rgi((x, y), z, bounds_error=True)

        # convert psi to pi/2-psi if psi > pi/4
        psi = psi.copy()
        sel = np.pi/4 < psi
        psi[sel] = np.pi/2 - psi[sel]

        x = edge_correction(np.log(ell), f.grid[0].min(), f.grid[0].max())
        y = edge_correction(np.log(psi), f.grid[1].min(), f.grid[1].max())
        out = f((x, y))

        return out

    def kappa_bispectrum_multipole(self, L, ell, psi, sample_combination=('0','0','0')):
        """
        Compute multipole of kappa bispectrum.

        L (int or array): multipole
        ell (array): ell array
        psi (array): psi array
        """
        isscalar = np.isscalar(L)
        if isscalar:
            L = np.array([L])
        out = np.zeros((L.size,) + ell.shape)
        for i, _L in enumerate(L):
            out[i] = self._kappa_bispectrum_multipole(_L, ell, psi, sample_combination=sample_combination)

        if isscalar:
            out = out[0]
            
        return out

    def kappa_bispectrum_resum(self, ell1, ell2, ell3, sample_combination=('0','0','0'), Lmax=None):
        """
        Compute kappa bispectrum by resummation of multipoles.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        """
        ell, psi, mu = trigutils.x1x2x3_to_xpsimu(ell1, ell2, ell3)
        if Lmax is None:
            Lmax = self.multipoles_data['bL'].shape[0]
        L = np.arange(Lmax)
        bL = self.kappa_bispectrum_multipole(L, ell, psi, sample_combination=sample_combination)
        pL = np.array([eval_legendre(_L, mu) for _L in L])
        out = np.sum(bL*pL, axis=0)
        return out


class BispectrumHalofit(BispectrumBase):
    """
    Bispectrum computed from halofit.
    """
    ell12min = 1e-1
    ell12max = 1e5
    def __init__(self, cosmo=None, zs=None, pzs=None):
        self.halofit = Halofit()
        super().__init__(cosmo, zs, pzs)

    def set_cosmology(self, cosmo, ns=None, sigma8=None):
        """
        Sets cosmology. 

        cosmo (astropy.cosmology): cosmology
        ns (float): spectral index of linear power spectrum
        sigma8 (float): sigma8 of linear power spectrum (at z=0.0)

        Note that the values of ns and sigma8 are set by two ways:
        1. Assigning ns and sigma8 as arguments of this method.
        2. Assigning ns and sigma8 to cosmo.meta.
        """
        super().set_cosmology(cosmo)
        # parameters for halofit
        dcosmo={'Om0': cosmo.Om0, 
                'Ode0': cosmo.Ode0,
                'ns': cosmo.meta.get('n'),
                'sigma8': cosmo.meta.get('sigma8'), 
                'w0': cosmo.w0, 
                'wa': 0.0,
                'fnu0': 0.0} 
        self.halofit.set_cosmology(dcosmo)

    def set_pklin(self, k, pklin):
        self.halofit.set_pklin(k, pklin)
        self.has_changed = True

    def set_lgr(self, z, lgr):
        self.halofit.set_lgr(z, lgr)
        self.has_changed = True

    def matter_bispectrum(self, k1, k2, k3, z, all_physical=True, which=['Bh1', 'Bh3']):
        return self.halofit.get_bihalofit(k1, k2, k3, z, all_physical=all_physical, which=which)

class BispectrumNFW1Halo(BispectrumBase):
    """
    Toy model
    """
    ell12min = 1e-2
    ell12max = 1e5
    def __init__(self, cosmo=None, zs=None, pzs=None, rs=10.0):
        super().__init__(cosmo, zs, pzs)
        self.set_rs(rs)

    def set_rs(self, rs):
        self.rs = rs

    @classmethod
    def rhok_NFW(cls, k, rs):
        y = k*rs
        si, ci = sici(y)
        return -np.cos(y)*ci + 0.5*np.sin(y)*(np.pi-2*si)

    def kappa_bispectrum_direct(self, ell1, ell2, ell3, **args):
        rs = np.deg2rad(self.rs/60.0) # in rad
        bl = 1
        for i, _ell in enumerate([ell1, ell2, ell3]):
            bl *= self.rhok_NFW(_ell, rs)
        return bl
