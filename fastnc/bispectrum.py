#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/03/21 18:00:50

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
from .multipole import MultipoleLegendre, MultipoleFourier
from .utils import loglinear, edge_correction


wPlanck18 = wCDM(H0=Planck18.H0, Om0=Planck18.Om0, Ode0=Planck18.Ode0, w0=-1.0, meta=Planck18.meta, name='wPlanck18')

class BispectrumBase:
    """
    Base class for bispectrum computation.

    Usage:
    >>> b = BispectrumBase()
    >>> b.set_cosmology(cosmo)
    >>> b.set_source_distribution(zs, pzs)
    >>> b.set_ell1mu_range(ell1min, ell1max, epmu)
    >>> b.interpolate(nrbin=35, nubin=25, nvbin=25, method='linear', nzbin=20)
    >>> b.decompose(Lmax, nellbin=100, npsibin=50, nmubin=50, method_decomp='linear', method_bispec='interp')
    >>> b.kappa_bispectrum_multipole(L, ell, psi)
    >>> b.kappa_bispectrum_resum(ell1, ell2, ell3)
    """
    # default configs
    config_scale     = dict(ell1min=None, ell1max=None, epmu=1e-7)
    config_losint    = dict(zmin=1e-4, nzbin=30)
    config_interp    = dict(nrbin=35, nubin=35, nvbin=25, method='linear', use=True)
    config_multipole = dict(nellbin=100, npsibin=80, nmubin=50, Lmax=1, \
        multipole_type='legendre', method='gauss-legendre', use=True)

    def __init__(self, config_scale=None, config_losint=None, config_interp=None, config_multipole=None):
        """
        Initialize BispectrumBase.

        config_scale (dict): config for scale (this should not be changed)
        config_losint (dict): config for line-of-sight integration
        config_interp (dict): config for interpolation
        config_multipole (dict): config for multipole decomposition
        """
        # set the support range of ell1, ell2
        self.set_scale_range(**(config_scale or {}))
        # init line-of-sight integration config
        self.set_losint(**(config_losint or {}))
        # init interpolation grid
        self.set_interpolation_grid(**(config_interp or {}))
        # init multipole decomposition grid
        self.set_multipole_grid(**(config_multipole or {}))
        
    # Binning
    def set_losint(self, **config):
        """
        Set line-of-sight integration config.

        Possible config keys are:
            zmin (float): minimum of redshift
            nzbin (int): number of bins for redshift
        """
        # update config
        self.config_losint.update(config)
        # source to class attributes
        self.zmin_losint  = self.config_losint['zmin']
        self.nzbin_losint = self.config_losint['nzbin']

    def set_scale_range(self, **config):
        """
        Set support range of ell1, ell2, mu, where
        ell1 and ell2 are the two side lengths of the triangle,
        and mu is the cosine of the **outer** angle of the triangle 
        between ell1 and ell2. Thus the other side length ell3 can be
        computed from ell1, ell2, and mu:

            ell3 = (ell1**2 + ell2**2 + 2*ell1*ell2*mu)**0.5
        
        We avoid the squeezed limit of triangle by setting the minimum
        of mu to be 1-epmu, where epmu is a small number.

        Possible config keys are:
            ell1min (float): minimum of ell1 and ell2
            ell1max (float): maximum of ell1 and ell2
            epmu (float): small number to avoid the squeezed limit
        """
        # update config
        self.config_scale.update(config)
        
        # fastnc args convention
        self.ell1min = self.config_scale['ell1min']
        self.ell1max = self.config_scale['ell1max']
        self.mumin    = -1.0
        self.mumax    = 1.0 - self.config_scale['epmu']

        # Multipole decomposition args convention.
        self.ellmin = 2**0.5 * self.ell1min
        self.ellmax = 2**0.5 * self.ell1max
        self.psimin = min(np.arctan2(self.ell1min, self.ell1max), np.pi/2 - np.arctan(self.ell1max/self.ell1min))
        self.psimax = np.pi/4

        # Interpolation args convention.
        # When ell, psi, mu run over the lectangular region
        # defined by the above ranges, the ranges of r, u, v
        # are also given by the ranges on ell, psi, mu.
        self.rmin = self.ellmin*min(5**-0.5, (1-self.mumax*2/5)**0.5)
        self.rmax = self.ellmax*max(2**-0.5, np.cos(self.psimin))
        self.umin = min(2**0.5*self.config_scale['epmu']**0.5, np.tan(self.psimin))
        self.umax = 1.0
        self.vmin = 0.0
        self.vmax = 1.0

    def set_interpolation_grid(self, **config):
        """
        Set interpolation grid.

        Possible config keys are:
            nrbin (int): number of bins for r
            nubin (int): number of bins for u
            nvbin (int): number of bins for v
            method (str): method for interpolation
            use (bool): whether to use interpolation
        """
        # update config 
        self.config_interp.update(config)
        # source to class attributes
        if not self.config_interp['use']: return 0
        r = np.logspace(np.log10(self.rmin), np.log10(self.rmax), \
            self.config_interp['nrbin'])
        u = np.logspace(np.log10(self.umin), np.log10(self.umax), \
            self.config_interp['nubin'])
        v = np.linspace(self.vmin, self.vmax, \
            self.config_interp['nvbin'])
        # create meshgrid
        R, U, V = np.meshgrid(r, u, v, indexing='ij')
        ELL1, ELL2, ELL3 = trigutils.ruv_to_x1x2x3(R, U, V)
        # save grid
        self.r_interp = r
        self.u_interp = u
        self.v_interp = v
        self.ELL1_interp = ELL1
        self.ELL2_interp = ELL2
        self.ELL3_interp = ELL3
        # method for interpolation
        self.method_interp = self.config_interp['method']
        # place holder for interpolation function
        self.bk_interp = dict()

    def set_multipole_grid(self, **config):
        """
        Set multipole decomposition grid.

        Possible config keys are:
            nellbin (int): number of bins for ell
            npsibin (int): number of bins for psi
            nmubin (int): number of bins for mu
            Lmax (int): maximum multipole
            multipole_type (str): type of multipole decomposition
            method (str): method for multipole evaluation
            use (bool): whether to use multipole decomposition
        """
        # update config
        self.config_multipole.update(config)
        # source to class attributes
        if not self.config_multipole['use']: return 0
        ell = np.logspace(np.log10(self.ellmin), np.log10(self.ellmax), \
            self.config_multipole['nellbin'])
        psi = loglinear(self.psimin, 1e-3, self.psimax, 50, \
            self.config_multipole['npsibin'])
        # capture the squeezed limit
        mu = 1-loglinear(1-self.mumax, 5e-2, 1-self.mumin, 30, \
            self.config_multipole['nmubin'])[::-1]
        # create meshgrid
        ELL, SPI, MU = np.meshgrid(ell, psi, mu, indexing='ij')
        ELL1, ELL2, ELL3 = trigutils.xpsimu_to_x1x2x3(ELL, SPI, MU)
        # save grid
        self.ell_multipole = ell
        self.pzs_multipole = psi
        self.ELL1_multipole = ELL1
        self.ELL2_multipole = ELL2
        self.ELL3_multipole = ELL3
        # place holder for multipole decomposition
        self.bL_multipole = dict()
        # maximum multipole
        self.Lmax_multipole = self.config_multipole['Lmax']
        self.multipole_type = self.config_multipole['multipole_type']
        # Multipole calculator
        if self.multipole_type == 'legendre':
            self.multipole_decomposer = MultipoleLegendre(mu, \
                self.Lmax_multipole, \
                method=self.config_multipole['method'])
        elif self.multipole_type == 'fourier':
            self.multipole_decomposer = MultipoleFourier(mu, \
                self.Lmax_multipole, \
                method=self.config_multipole['method'])
        elif self.multipole_type == 'cosine':
            self.multipole_decomposer = MultipoleCosine(mu, \
                self.Lmax_multipole, \
                method=self.config_multipole['method'])
        elif self.multipole_type == 'sine':
            self.multipole_decomposer = MultipoleSine(mu, \
                self.Lmax_multipole, \
                method=self.config_multipole['method'])
        else:
            raise ValueError(f"multipole_type {multipole_type} is not supported" \
                    "supported types are 'legendre', 'fourier', 'cosine', and 'sine'")

    # Setter
    def set_cosmology(self, cosmo):
        """
        Sets cosmology. 

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
            assert self.zs_dict[name].size == self.pzs_dict[name].size, \
                "zs and pzs must have the same length"

        # rise flag
        self.has_changed = True

    def set_window_function(self, window_function):
        """
        Set window function to be multiplied to the bispectrum.

        B^W(l1,l2,l3) = B(l1,l2,l3) * W(l1,l2,l3)
        """
        self.window_function = window_function

    # Redshift-bin related
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
        """
        Compute lensing kernel for all samples.

        nzlbin (int): number of bins for lensing kernel
        """
        self.z2g_dict = dict()
        self.chi2g_dict = dict()
        for name in self.sample_names:
            z2g, chi2g = self.compute_lensing_kernel_per_sample(self.zs_dict[name], self.pzs_dict[name], nzlbin)
            self.z2g_dict[name] = z2g
            self.chi2g_dict[name] = chi2g
        self.zmax_losint = max([self.zs_dict[name].max() for name in self.sample_names])

    def get_all_sample_combinations(self):
        """
        Get all possible combinations of sample names.
        """
        combinations = []
        for i in range(self.n_sample):
            name_i = self.sample_names[i]
            for j in range(i, self.n_sample):
                name_j = self.sample_names[j]
                for k in range(j, self.n_sample):
                    name_k = self.sample_names[k]
                    combinations.append((name_i, name_j, name_k))
        return combinations

    def parse_sample_combination(self, sample_combination):
        """
        Parse sample combination to a list of sample names.
        """
        if sample_combination is None:
            scs = self.get_all_sample_combinations()
            assert len(scs) == 1, "specify sample_combination!"
            return scs[0]
        else:
            return sample_combination
        
    # Spectra methods
    # matter power spectrum (to be implemented in subclasses)
    def matter_bispectrum(self, k1, k2, k3, z):
        raise NotImplementedError

    # kappa bispectrum interface
    def kappa_bispectrum(self, ell1, ell2, ell3, \
            sample_combination=None, \
            method='direct', **args):
        """
        Compute kappa bispectrum.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        method (str): method for computing kappa bispectrum (direct, interp, resum)
        """
        # parse sample_combination
        sample_combination = self.parse_sample_combination(sample_combination)

        if method == 'direct':
            return self.kappa_bispectrum_direct(ell1, ell2, ell3, sample_combination, **args)
        elif method == 'interp':
            return self.kappa_bispectrum_interp(ell1, ell2, ell3, sample_combination)
        elif method == 'resum':
            return self.kappa_bispectrum_resum(ell1, ell2, ell3, sample_combination, **args)
        else:
            raise ValueError("method must be 'direct', 'interp', or 'resum'")
        
    # direct evaluation of kappa bispectrum from matter bispectrum
    def kappa_bispectrum_direct(self, ell1, ell2, ell3, \
            sample_combination=None, window=True, \
            bm=None, return_bm=False, **args):
        """
        Compute kappa bispectrum by direct line-of-sight integration.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        sample_combination (tuple): sample combination
        bm (array): matter bispectrum, if None, it is computed
        return_bm (bool): return matter bispectrum if True
        args (dict): arguments for matter_bispectrum

        Note:
        The bm input can be used to save computation time when 
        computing kappa bispectrum for multiple sample_combinations.
        """
        # parse sample_combination
        sample_combination = self.parse_sample_combination(sample_combination)

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
        z = np.logspace(np.log10(self.zmin_losint), np.log10(self.zmax_losint), self.nzbin_losint)
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

        # multiply window 
        if hasattr(self, 'window_function') and window:
            bk *= self.window_function(ell1, ell2, ell3)

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
    def interpolate(self, sample_combinations=None, **args):
        """
        Interpolate kappa bispectrum. 
        The interpolation is done in (r,u,v)-space, which is defined in M. Jarvis+2003 
        (https://arxiv.org/abs/astro-ph/0307393). See also treecorr homepage
        (https://rmjarvis.github.io/TreeCorr/_build/html/correlation3.html).

        sample_combinations (list): list of sample combinations
        args (dict): arguments for matter_bispectrum
        """
        # If sample_combinations is not given, 
        # we get all possible combinations.
        if sample_combinations is None:
            sample_combinations = self.get_all_sample_combinations()
        # Prepare for the interpolation
        bm = None
        grid = (np.log(self.r_interp), np.log(self.u_interp), self.v_interp)
        for sc in sample_combinations:
            bk, bm = self.kappa_bispectrum_direct(
                self.ELL1_interp, 
                self.ELL2_interp, 
                self.ELL3_interp, 
                sample_combination=sc,
                window=False,
                bm=bm, 
                return_bm=True, 
                **args)
            self.bk_interp[sc] = rgi(grid, np.log(bk), method=self.method_interp)

    def kappa_bispectrum_interp(self, ell1, ell2, ell3, \
            sample_combination=None):
        """
        Compute kappa bispectrum by interpolation.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        """
        sample_combination = self.parse_sample_combination(sample_combination)
        ip = self.bk_interp[sample_combination]
        r, u, v = trigutils.x1x2x3_to_ruv(ell1, ell2, ell3, signed=False)
        x = edge_correction(np.log(r), ip.grid[0].min(), ip.grid[0].max())
        y = edge_correction(np.log(u), ip.grid[1].min(), ip.grid[1].max())
        z = edge_correction(v, ip.grid[2].min(), ip.grid[2].max())
        bk = np.exp(ip((x,y,z)))
        # multiply window 
        if hasattr(self, 'window_function'):
            bk *= self.window_function(ell1, ell2, ell3)
        return bk

    # multipole decomposition
    def decompose(self, sample_combinations=None, method_bispec='interp', **args):
        """
        Compute multipole decomposition of kappa bispectrum.

        args (dict): arguments for kappa_bispectrum
        """
        # If sample_combinations is not given, 
        # we get all possible combinations.
        if sample_combinations is None:
            sample_combinations = self.get_all_sample_combinations()
        # Compute multipole
        for sc in sample_combinations:
            b = self.kappa_bispectrum(
                    self.ELL1_multipole, 
                    self.ELL2_multipole, 
                    self.ELL3_multipole, 
                    sc, 
                    method=method_bispec, 
                    **args)
            # Compute multipoles
            L = np.arange(self.Lmax_multipole+1)
            bL = self.multipole_decomposer.decompose(b, L, axis=2)
            self.bL_multipole[sc] = bL

    def kappa_bispectrum_multipole(self, L, ell, psi, \
            sample_combination=None):
        """
        Compute multipole of kappa bispectrum.

        L (array): multipole
        ell (array): ell array
        psi (array): psi array
        """
        # parse sample_combination
        sample_combination = self.parse_sample_combination(sample_combination)
        # cast to array
        isscalar = np.isscalar(L)
        if isscalar:
            L = np.array([L])

        # compute multipole
        out = np.zeros((L.size,) + ell.shape)
        grid = (np.log(self.ell_multipole), np.log(self.pzs_multipole))
        for i, _L in enumerate(L):
            # interpolate
            z = self.bL_multipole[sample_combination][_L, :, :]
            ip= rgi(grid, z, bounds_error=True)
            # convert psi to pi/2-psi if psi > pi/4
            psi = psi.copy()
            sel = np.pi/4 < psi
            psi[sel] = np.pi/2 - psi[sel]

            x = edge_correction(np.log(ell), ip.grid[0].min(), ip.grid[0].max())
            y = edge_correction(np.log(psi), ip.grid[1].min(), ip.grid[1].max())
            out[i] = ip((x, y))

        if isscalar:
            out = out[0]
            
        return out

    def kappa_bispectrum_resum(self, ell1, ell2, ell3, \
            sample_combination=None):
        """
        Compute kappa bispectrum by resummation of multipoles.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        """
        sample_combination = self.parse_sample_combination(sample_combination)
        ell, psi, mu = trigutils.x1x2x3_to_xpsimu(ell1, ell2, ell3)
        L = np.arange(self.Lmax_multipole)
        bL = self.kappa_bispectrum_multipole(L, ell, psi, sample_combination=sample_combination)
        pL = np.array([eval_legendre(_L, mu) for _L in L])
        out = np.sum(bL*pL, axis=0)
        return out


class BispectrumHalofit(BispectrumBase):
    """
    Bispectrum computed from halofit.
    """
    # default configs
    config_scale     = dict(ell1min=1e-1, ell1max=1e5, epmu=1e-7)
    def __init__(self, config_scale=None, config_losint=None, config_interp=None, config_multipole=None):
        self.halofit = Halofit()
        super().__init__(config_scale, config_losint, config_interp, config_multipole)

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
    ell1min = 1e-2
    ell1max = 1e5
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
