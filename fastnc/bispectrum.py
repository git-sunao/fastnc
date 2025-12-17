#!/usr/bin/env python
"""
Author     : Sunao Sugiyama 
Last edit  : 2024/11/27 18:2:25

Description:
bispectrum.py contains classes for computing bispectrum 
and various methods of bispectrum: interpolation, 
multipole decomposition, etc.
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RegularGridInterpolator as rgi
from astropy.cosmology import wCDM, Planck18
from scipy.special import sici
from scipy.special import eval_legendre
# fastnc modules
from . import trigutils
from .halofit import Halofit
from .multipole import MultipoleLegendre, MultipoleFourier
from .utils import loglinear, edge_correction, update_config, get_config_key
from .baryon import BaryonModelBase
from .ia_bispectra import BispectraIA


wPlanck18 = wCDM(H0=Planck18.H0, Om0=Planck18.Om0, Ode0=Planck18.Ode0, w0=-1.0, meta=Planck18.meta, name="wPlanck18")

class BispectrumBase:
    r"""
    Base class for bispectrum computation.

    Parameters:
        config (dict)       : A configuration dict that can be used to pass in the below kwargs if
                              desired.  This dict is allowed to have addition entries in addition
                              to those listed below, which are ignored here. (default: None)
        verbose (bool)      : Whether to print the progress. (default: True)

    Keyword arguments:
        ell1min (float)     : minimum of ell1 and ell2
        ell1max (float)     : maximum of ell1 and ell2
        epmu (float)        : small number to avoid the squeezed limit
        zmin (float)        : minimum of redshift
        zmid (float)        : middle of redshift to switch from log to linear binning
        nzbin_log (int)     : number of bins for redshift in log scale
        nzbin_lin (int)     : number of bins for redshift in linear scale
        nrbin (int)         : number of bins for r
        nubin (int)         : number of bins for u
        nvbin (int)         : number of bins for v
        method (str)        : method for interpolation
        use_interp (bool)   : whether to use interpolation
        nellbin (int)       : number of bins for ell
        npsibin (int)       : number of bins for psi
        nmubin (int)        : number of bins for mu
        Lmax (int)          : maximum multipole (default: None)
        Lmax_diag (nint)    : maximum multipole for diagonal elements (default: None)
        multipole_type (str): type of multipole decomposition
        method (str)        : method for multipole evaluation
    
    Usage:
    >>> b = BispectrumBase()
    >>> b.set_cosmology(cosmo)
    >>> b.set_source_distribution(zs, pzs)
    >>> b.set_ell1mu_range(ell1min, ell1max, epmu)
    >>> b.interpolate(scombs=None, **args)
    >>> b.decompose(scombs=None, method_bispec='interp', **args)
    >>> b.kappa_bispectrum_multipole(L, ell, psi, scomb=scomb)
    """
    # default configs
    config_scale     = dict(ell1min=None, ell1max=None, epmu=1e-7)
    config_losint    = dict(zmin=1e-4, zmid=1e-1, nzbin_log=15, nzbin_lin=40, zbin=None)
    config_interp    = dict(nrbin=35, nubin=35, nvbin=25, method='linear', use_interp=True, ell_grid=False, puv_grid = False)
    config_multipole = dict(nellbin=100, npsibin=80, nmubin=50, nmubin_log=30, Lmax=None, Lmax_diag=None, \
        multipole_type='legendre', method='gauss-legendre')
    config_IA        = dict(NLA=False)

    def __init__(self, config=None, **kwargs):
        # set the support range of ell1, ell2
        self.set_scale_range(config, **kwargs)
        # init line-of-sight integration config
        self.set_losint(config, **kwargs)
        # init interpolation grid
        self.set_interpolation_grid(config, **kwargs)
        # init multipole decomposition grid
        self.set_multipole_grid(config, **kwargs)
        # init intrinsic alignment model
        update_config(self.config_IA, config, **kwargs)
        # set None for baryon model
        self.set_baryon_model(BaryonModelBase())
        
    # Binning
    def set_losint(self, config=None, **kwargs):
        """
        Set line-of-sight integration config.

        Parameters:
            config (dict)       : A configuration dict that can be used to pass in the below kwargs if
                                  desired.  This dict is allowed to have addition entries in addition
                                  to those listed below, which are ignored here. (default: None)

        Keyword arguments:
            zmin (float)        : minimum of redshift
            nzbin (int)         : number of bins for redshift
        """
        # update config
        update_config(self.config_losint, config, **kwargs)
        # source to class attributes
        self.zmin_losint  = self.config_losint['zmin']
        self.zmid_losint  = self.config_losint['zmid']
        self.nzbin_log_losint = self.config_losint['nzbin_log']
        self.nzbin_lin_losint = self.config_losint['nzbin_lin']
        self.zbin_losint = self.config_losint.get('zbin', None)

    def set_scale_range(self, config=None, **kwargs):
        """
        Set support range of ell1, ell2, mu.
        
        Parameters:
            config (dict)       : A configuration dict that can be used to pass in the below kwargs if
                                  desired.  This dict is allowed to have addition entries in addition
                                  to those listed below, which are ignored here. (default: None)
        
        Keyword arguments:
            ell1min (float)     : minimum of ell1 and ell2
            ell1max (float)     : maximum of ell1 and ell2
            epmu (float)        : small number to avoid the squeezed limit
        
        Description:
            Here ell1 and ell2 are the two side lengths of the triangle,
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
        update_config(self.config_scale, config, **kwargs)
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

    def set_interpolation_grid(self, config=None, **kwargs):
        """
        Set interpolation grid.

        Parameters:
            config (dict)       : A configuration dict that can be used to pass in the below kwargs if
                                  desired.  This dict is allowed to have addition entries in addition
                                  to those listed below, which are ignored here. (default: None)
        
        Keyword arguments:
            nrbin (int)         : number of bins for r
            nubin (int)         : number of bins for u
            nvbin (int)         : number of bins for v
            method (str)        : method for interpolation
            use_interp (bool)   : whether to use interpolation
        """
        # update config
        update_config(self.config_interp, config, **kwargs)
        # source to class attributes
        if not self.config_interp['use_interp']: return 0
        if not self.config_interp['ell_grid'] and not self.config_interp['puv_grid']:
            print('ruv')
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

        elif self.config_interp['puv_grid']:
            p = np.logspace(np.log10(self.rmin), np.log10(3*self.rmax), \
                self.config_interp['nrbin'])
            #pu = np.linspace(0.000001,0.5,20)
            #pv = np.linspace(0.000001,0.5,20)
            #pu = np.unique(np.concatenate((np.logspace(-4, -1, 8), np.linspace(0.1, 0.5, 33))))
            #pv = np.unique(np.concatenate((np.logspace(-4, -1, 8), np.linspace(0.1, 0.5, 33))))
            #pu = np.unique(np.concatenate((np.logspace(-4, -1, 8), np.linspace(0.1, 0.453, 22),
            #                               (0.5 * np.ones(8) - np.logspace(-4, np.log10(0.03), 8))[::-1],0.5* np.ones(1))))
            #pv = np.unique(np.concatenate((np.logspace(-4, -1, 8), np.linspace(0.1, 0.453, 22),
            #                               (0.5 * np.ones(8) - np.logspace(-4, np.log10(0.03), 8))[::-1],0.5* np.ones(1))))
            pu = np.unique(np.concatenate((1e-6 * np.ones(1), np.logspace(-4, np.log10(0.03), 8), np.linspace(0.047, 0.453, 22),
                                           (0.5 * np.ones(8) - np.logspace(-4, np.log10(0.03), 8))[::-1],
                                           0.5 * np.ones(1))))
            pv = np.unique(np.concatenate((1e-6 * np.ones(1), np.logspace(-4, np.log10(0.03), 8), np.linspace(0.047, 0.453, 22),
                                           (0.5 * np.ones(8) - np.logspace(-4, np.log10(0.03), 8))[::-1],
                                           0.5 * np.ones(1))))
            P, PU, PV = np.meshgrid(p, pu, pv, indexing='ij')
            ELL1 = PU*P
            ELL2 = PV*P
            ELL3 = P*(1-PU-PV)
            self.p_interp = p
            self.pu_interp = pu
            self.pv_interp = pv
            self.ELL1_interp = ELL1
            self.ELL2_interp = ELL2
            self.ELL3_interp = ELL3
            # method for interpolation
            self.method_interp = self.config_interp['method']
            # place holder for interpolation function
            self.bk_interp = dict()

        else:
            print('ellgrid')
            ell1 = np.logspace(0,3,70)
            ell2 = np.logspace(0,3,70)
            ell3 = np.logspace(0,3,70)
            # create meshgrid
            ELL1, ELL2, ELL3 = np.meshgrid(ell1, ell2, ell3, indexing='ij')
            # save grid
            self.ell1_interp = ell1
            self.ell2_interp = ell2
            self.ell3_interp = ell3
            self.ELL1_interp = ELL1
            self.ELL2_interp = ELL2
            self.ELL3_interp = ELL3
            print(np.shape(self.ELL1_interp))
            print(np.shape(self.ELL2_interp))
            print(np.shape(self.ELL3_interp))
            # method for interpolation
            self.method_interp = self.config_interp['method']
            # place holder for interpolation function
            self.bk_interp = dict()

    def set_multipole_grid(self, config=None, **kwargs):
        """
        Set multipole decomposition grid.

        Parameters:
            config (dict)       : A configuration dict that can be used to pass in the below kwargs if
                                  desired.  This dict is allowed to have addition entries in addition
                                  to those listed below, which are ignored here. (default: None)
        
        Keyword arguments:
            nellbin (int)       : number of bins for ell
            npsibin (int)       : number of bins for psi
            nmubin (int)        : number of bins for mu
            Lmax (int)          : maximum multipole
            multipole_type (str): type of multipole decomposition
            method (str)        : method for multipole evaluation
        """
        # update config
        update_config(self.config_multipole, config, **kwargs)
        # source to class attributes
        if self.config_multipole['Lmax'] is None: return 0
        ell = np.logspace(np.log10(self.ellmin), np.log10(self.ellmax), \
            self.config_multipole['nellbin'])
        psi = loglinear(self.psimin, 1e-3, self.psimax, 50, \
            self.config_multipole['npsibin'])
        # capture the squeezed limit
        mu = 1-loglinear(1-self.mumax, 5e-2, 1-self.mumin, \
            self.config_multipole['nmubin_log'], \
            self.config_multipole['nmubin'])[::-1] #why did it remove the indices?
        # create meshgrid
        ELL, PSI, MU = np.meshgrid(ell, psi, mu, indexing='ij')
        ELL1, ELL2, ELL3 = trigutils.xpsimu_to_x1x2x3(ELL, PSI, MU)
        # save grid
        self.ell_multipole = ell
        self.psi_multipole = psi
        self.ELL1_multipole = ELL1
        self.ELL2_multipole = ELL2
        self.ELL3_multipole = ELL3
        # place holder for multipole decomposition
        self.bL_multipole = dict()
        # maximum multipole
        self.Lmax_multipole = self.config_multipole['Lmax']
        self.multipole_type = self.config_multipole['multipole_type']
        # For additional multipole decomposition at diagonal
        self.Lmax_multipole_diag = self.config_multipole['Lmax_diag']
        if self.Lmax_multipole_diag is None:
            self.Lmax_multipole_diag = self.Lmax_multipole
        ELL, MU = np.meshgrid(ell, mu, indexing='ij')
        self.ELL1_multipole_diag = ELL/2**0.5
        self.ELL2_multipole_diag = ELL/2**0.5
        self.ELL3_multipole_diag = ELL*(1-MU)**0.5
        self.bL_multipole_diag = dict()
        # Multipole calculator
        if self.multipole_type == 'legendre':
            self.multipole_decomposer = MultipoleLegendre(mu, \
                self.Lmax_multipole_diag, \
                method=self.config_multipole['method'])
        elif self.multipole_type == 'fourier':
            self.multipole_decomposer = MultipoleFourier(mu, \
                self.Lmax_multipole_diag, \
                method=self.config_multipole['method'])
        elif self.multipole_type == 'cosine':
            self.multipole_decomposer = MultipoleCosine(mu, \
                self.Lmax_multipole_diag, \
                method=self.config_multipole['method'])
        elif self.multipole_type == 'sine':
            self.multipole_decomposer = MultipoleSine(mu, \
                self.Lmax_multipole_diag, \
                method=self.config_multipole['method'])
        else:
            raise ValueError(f"multipole_type {multipole_type} is not supported" \
                    "supported types are 'legendre', 'fourier', 'cosine', and 'sine'")

    # Setter
    def set_cosmology(self, cosmo):
        """
        Sets cosmology. 

        Parameters:
            cosmo (astropy.cosmology): cosmology
        """
        self.cosmo = cosmo

        # compute array of chi and z
        z   = np.linspace(0, 5, 100)
        chi = self.cosmo.comoving_distance(z).value * self.cosmo.h # Mpc/h
        dzdchi = np.diff(z)/np.diff(chi)

        # spline chi <-> z
        self.z2chi = ius(z, chi)
        self.chi2z = ius(chi, z)
        self.z2dzdchi = ius(0.5*(z[1:]+z[:-1]), dzdchi, ext=1)
        self.has_changed = True

    def set_source_distribution(self, zs_list, pzs_list, sample_names=None):
        """
        Set source distribution.

        Parameters:
            zs_list (list)  : redshift array
            pzs_list (list) : probability distribution of source galaxies
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

    def set_NLA_param(self, params):
        """
        Set parameters for nonlinear alignment effect.

        Parameters:
            params (dict) : parameters for nonlinear alignment effect
        """
        # check
        # Default assumes the power law redshift evolution.
        params['perbin'] = params.get('perbin', False)
        if params['perbin']:
            for name in self.sample_names:
                IA_name = f'AIA_{name}'
                assert IA_name in params, f'{IA_name} must be supplied.'
        else:
            assert 'AIA'     in params, 'AIA must be supplied.'
        assert 'alphaIA' in params, 'alphaIA must be supplied.'
        assert 'z0'      in params, 'z0 must be supplied.'
        self.NLA_params = params

    def set_window_function(self, window_function):
        """
        Set window function to be multiplied to the bispectrum.

        B^W(l1,l2,l3) = B(l1,l2,l3) * W(l1,l2,l3)
        """
        self.window_function = window_function

    # Redshift-bin related
    def _compute_lensing_kernel_per_sample(self, name, nzlbin=101):
        """
        Set source distribution.

        Parameters:
            name (str): sample name
        """
        # get zs, pzs array
        zs, pzs = self.zs_dict[name], self.pzs_dict[name]
        #for debugging purposes, force the combination to be:
        #zs = np.linspace(0, 2, 100)
        #mug = 0.5
        #sigmag = 0.05
        #pzs = np.exp(-0.5 * ((zs - mug) / sigmag) ** 2) / (sigmag * np.sqrt(2 * np.pi))
        
        prefactor = 3/2 * (100/299792)**2 * self.cosmo.Om0
        if zs.size == 1:
            zl = np.linspace(self.zmin_losint, zs, nzlbin)
            chil = self.z2chi(zl)
            chis = self.z2chi(zs)
            g = prefactor*(1.-chil/chis)
        else:
            zl = np.linspace(self.zmin_losint, zs.max(), nzlbin)
            chil = self.z2chi(zl)
            chis = self.z2chi(zs)
            CHIL, CHIS = np.meshgrid(chil, chis, indexing='ij')

            # integrand
            I = np.ones_like(CHIL, dtype=float)
            I = np.divide(CHIL, CHIS, out=I, where=CHIS > CHIL)
            I = (pzs*(1-I))

            g = prefactor*np.trapz(I, zs, axis=1)/np.trapz(pzs, zs)
        return zl, chil, g

    def _compute_NLA_kernel_per_sample(self, name, nzlbin=101):
        """
        Compute the kernel of nonlinear alignment effect.

        Parameters:
            name (str): sample name

        Note:
            In order for this to work, the following attributes must be set:
            - self.cosmo
            - self.z2chi
            - self.z2dzdchi
            - self.z2lgr
        """
        # get zs, pzs array
        zs, pzs = self.zs_dict[name], self.pzs_dict[name]
        
        # model param
        if self.NLA_params['perbin']:
            AIA = self.NLA_params[f'AIA_{name}']
        else:
            AIA = self.NLA_params['AIA']
        alphaIA = self.NLA_params['alphaIA']
        z0 = self.NLA_params['z0']
        # constant
        c1rhocrit = 0.0134
        # compute kernel
        zsmax = zs if zs.size==1 else np.max(zs)
        zl = np.linspace(self.zmin_losint, zsmax, nzlbin)
        chil = self.z2chi(zl)
        fIA = - AIA * ((1+zl)/(1+z0))**alphaIA * c1rhocrit * self.cosmo.Om0 / self.z2lgr(zl)
        pchis = np.interp(zl, zs, pzs, left=0, right=0) * self.z2dzdchi(zl)
        norm = np.trapz(pchis, chil)
        #g = fIA * pchis / norm / chil
        g = fIA * pchis / norm / chil / (zl+np.ones_like(zl))
        #verify if there should be a factor of a(chil) here -- I added the factor but left the old version commented on top.
        #the NLA kernel, which is fIA * pchis / norm, should be added to a lensing kernel which has the factor of chi/a.
        #Our definition of the lensing kernel doesnt include this factor and the factor is added in the end by multiplying
        #the bispectrum by *1/(chi*a**3) instead of *1/chi**4. However, to make the NLA kernel consistent, it should be
        #multiplied by a/chil.
        return zl, chil, g

    def compute_kernel(self, nzlbin=101):
        """
        Compute lensing kernel for all samples.

        Parameters:
            nzlbin (int): number of bins for lensing kernel
        """
        self.z2g_dict = dict()
        self.chi2g_dict = dict()
        self.z2W_dict = dict() # Window function for TATT
        self.chi2W_dict = dict() # Window function for TATT

        for name in self.sample_names:
            z, chi, g = self._compute_lensing_kernel_per_sample(name, nzlbin)

            #Source galaxy window function for TATT
            zs, pzs = self.zs_dict[name], self.pzs_dict[name]

            if zs.size == 1:
                def always_unit():
                    return 1
                self.z2W_dict[name] = always_unit
                self.chi2W_dict[name] = always_unit

            else:

                # for debugging purposes, force the combination to be:
                #zs = np.linspace(0, 2, 100)
                #mug = 0.5
                #sigmag = 0.05
                #pzs = np.exp(-0.5 * ((zs - mug) / sigmag) ** 2) / (sigmag * np.sqrt(2 * np.pi))

                #resume
                pchis = pzs*self.z2dzdchi(zs)
                chis = self.z2chi(zs)
                norm_pchis = np.trapz(pchis, chis)
                W_g = pchis / norm_pchis / chis / (zs + np.ones_like(zs))  # Normalized window function -- with a/chi term for compatible notation
                self.z2W_dict[name] = ius(zs, W_g, ext=1)
                self.chi2W_dict[name] = ius(chis, W_g, ext=1)

            if self.config_IA['NLA']:
                z, chi, gNLA = self._compute_NLA_kernel_per_sample(name, nzlbin)
                g += gNLA

            self.z2g_dict[name] = ius(z, g, ext=1)
            self.chi2g_dict[name] = ius(chi, g, ext=1)

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

    def parse_sample_combination(self, scomb):
        """
        Parse sample combination to a list of sample names.

        Parameters:
            scomb (tuple) : a tuple of sample names (str) showing 
                            the combination of samples.

        Hint:
            If you do not have idea about the sample names, run the following code
            to get possible sample combinations.
            >>> bispectrum.get_all_sample_combinations()
            Here `bispectrum` is your bispectrum instance.
        """
        if scomb is None:
            scs = self.get_all_sample_combinations()
            assert len(scs) == 1, "specify sample_combination!"
            return scs[0]
        else:
            if np.isscalar(scomb):
                return scomb
            else:
                return tuple(scomb)
        
    # Spectra methods
    # matter power spectrum (to be implemented in subclasses)
    def matter_bispectrum_no_baryon(self, k1, k2, k3, z):
        """
        Compute matter bispectrum.

        Parameters:
            k1 (array) : k1 array in h/Mpc unit
            k2 (array) : k2 array in h/Mpc unit
            k3 (array) : k3 array in h/Mpc unit
            z (array)  : redshift array
        """
        raise NotImplementedError

    def matter_bispectrum(self, k1, k2, k3, z):
        """
        Matter bispetrum including baryon
        
        Parameters:
            k1 (array) : k1 array in h/Mpc unit
            k2 (array) : k2 array in h/Mpc unit
            k3 (array) : k3 array in h/Mpc unit
            z (array)  : redshift array
        """
        b = self.matter_bispectrum_no_baryon(k1,k2,k3,z)
        r = self.baryon_model(k1,k2,k3,z)
        return b*r

    def ia_bispectrum(self, k1, k2, k3, z, z_piv, a1, alpha1, a2, alpha2, bias_ta):
        """
        Compute intrinsic alignment bispectrum components.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def get_los_kernel(self, scomb, kernel_type='lensing'):

        if np.isscalar(scomb) and isinstance(scomb, (int, float)):
            # special case where the scomb is given by the redshift
            # This is useful when computing the kappa bispectrum
            # before los integration, which will be used e.g. as the training
            # data for the los-integration independent emulator.
            z = scomb
            if np.isscalar(z):
                z = np.array([z])
            chi = self.z2chi(z)
            weight = np.ones(z.size)
        else:
            # compute lensing weight, encoding geometrical dependence.
            if hasattr(self, 'zbin_losint') and self.zbin_losint is not None:
                print('[fastnc] Using user defined zbin for los int')
                z = self.zbin_losint
            else:
                z = loglinear(self.zmin_losint, self.zmid_losint, self.zmax_losint, \
                    self.nzbin_log_losint, self.nzbin_lin_losint)
                #z = np.array([0.005,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5])
            
            chi = self.z2chi(z)
            weight = np.ones_like(chi) # Ensure weight is an array
            if kernel_type == 'lensing':
                for name in scomb:
                    weight *= self.chi2g_dict[name](chi)
            elif kernel_type == 'shape_window':
                for name in scomb:
                    weight *= self.chi2W_dict[name](chi)
            else:
                raise ValueError("kernel_type must be 'lensing' or 'shape_window'")

            #I commented this line out and added this factor directly on kappa_bispectrum_direct
            #Because if we are doing TATT we will call this function three times but only need one factor
            #weight *= 1.0/chi*(1+z)**3
        return z, chi, weight

    # kappa bispectrum interface
    def kappa_bispectrum(self, ell1, ell2, ell3, scomb=None, \
            method='direct', puv_grid = False, **args):
        """
        Compute kappa bispectrum.

        Parameters:
            ell1 (array)  : ell1 array
            ell2 (array)  : ell2 array
            ell3 (array)  : ell3 array
            scomb (tuple) : sample combination
            method (str)  : method for computing kappa bispectrum 
                            (direct, interp, resum)
        """
        # parse sample_combination
        scomb = self.parse_sample_combination(scomb)

        if method == 'direct':
            return self.kappa_bispectrum_direct(ell1, ell2, ell3, scomb, **args)
        elif method == 'interp':
            return self.kappa_bispectrum_interp(ell1, ell2, ell3, scomb, puv_grid = puv_grid)
        elif method == 'resum':
            return self.kappa_bispectrum_resum(ell1, ell2, ell3, scomb, **args)
        else:
            raise ValueError("method must be 'direct', 'interp', or 'resum'")
        
    # direct evaluation of kappa bispectrum from matter bispectrum
    def kappa_bispectrum_direct(self, ell1, ell2, ell3, scomb=None, \
            window=True, bm=None, return_bm=False, z=None, l_shift=0.0, **args):
        """
        Compute kappa bispectrum by direct line-of-sight integration.

        Parameters:
            ell1 (array)  : ell1 array
            ell2 (array)  : ell2 array
            ell3 (array)  : ell3 array
            scomb (tuple) : sample combination
            bm (array)                : matter bispectrum, if None, it is computed
            return_bm (bool)          : return matter bispectrum if True
            args (dict)               : arguments for matter_bispectrum

        Note:
        The bm input can be used to save computation time when 
        computing kappa bispectrum for multiple sample_combinations.
        """
        # parse sample_combination
        scomb = self.parse_sample_combination(scomb)

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

        # line-of-sight integration kernel
        z, chi, kernel = self.get_los_kernel(scomb)
        kernel *= 1.0 / chi * (1 + z) ** 3 #I am doing this product here instead of inside get_los_kernel so that it may be compatible with the TATT kernel
        
        # create grids
        ELL1, Z = np.meshgrid(ell1, z, indexing='ij')
        ELL2, Z = np.meshgrid(ell2, z, indexing='ij')
        ELL3, Z = np.meshgrid(ell3, z, indexing='ij')
        CHI = self.z2chi(Z)
        # K1, K2, K3 = ELL1/CHI, ELL2/CHI, ELL3/CHI
        K1, K2, K3 = (ELL1+l_shift)/CHI, (ELL2+l_shift)/CHI, (ELL3+l_shift)/CHI

        # compute matter bispectrum
        if (bm is None) or not isinstance(scomb, tuple):
            bm = self.matter_bispectrum(K1, K2, K3, Z, **args)

        # integrand
        i = kernel * bm

        # integrate
        if i.shape[1] > 1:
            bk = np.trapz(i, chi, axis=1)
        else:
            bk = i

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

    def kappa_bispectrum_IA_direct(self, ell1, ell2, ell3, scomb=None, \
            window=True, ia_bispec_comps=None, return_ia_bispec_comps=False, select_mode=None, Ct = 0, remove_alignment = False, z=None, l_shift=0.0):
        """
        Compute kappa bispectrum from intrinsic alignment bispectrum components by direct line-of-sight integration.

        Parameters:
            ell1 (array)  : ell1 array
            ell2 (array)  : ell2 array
            ell3 (array)  : ell3 array
            scomb (tuple) : sample combination
            ia_bispec_comps (tuple) : IA bispectrum components (B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE)
            return_ia_bispec_comps (bool) : return IA bispectrum components if True
            args (dict)               : arguments for ia_bispectrum
        """

        #set IA params
        z_piv = self.IA_params['z0']
        A1 = self.IA_params['a1']
        alphaIA = self.IA_params['alphaIA']
        A2 = self.IA_params['a2']
        alphaIA_2 = self.IA_params['alphaIA_2']
        bias_ta = self.IA_params['bias_ta']

        # parse sample_combination
        scomb = self.parse_sample_combination(scomb)

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

        # line-of-sight integration kernel
        if np.isscalar(scomb) and isinstance(scomb, (int, float)):
            z_los, chi_los, kernel_1 = self.get_los_kernel(scomb)
            kernel_2 = kernel_1
            kernel_3 = kernel_1
            W_1 = kernel_1
            W_2 = kernel_1
            W_3 = kernel_1

        else:
            z_los, chi_los, kernel_1 = self.get_los_kernel((scomb[0],), kernel_type='lensing')
            z_los, chi_los, kernel_2 = self.get_los_kernel((scomb[1],), kernel_type='lensing')
            z_los, chi_los, kernel_3 = self.get_los_kernel((scomb[2],), kernel_type='lensing')

            z_los, chi_los, W_1 = self.get_los_kernel((scomb[0],), kernel_type='shape_window')
            z_los, chi_los, W_2 = self.get_los_kernel((scomb[1],), kernel_type='shape_window')
            z_los, chi_los, W_3 = self.get_los_kernel((scomb[2],), kernel_type='shape_window')

        # create grids
        ELL1, Z = np.meshgrid(ell1, z_los, indexing='ij')
        ELL2, Z = np.meshgrid(ell2, z_los, indexing='ij')
        ELL3, Z = np.meshgrid(ell3, z_los, indexing='ij')
        CHI = self.z2chi(Z)
        K1, K2, K3 = (ELL1+l_shift)/CHI, (ELL2+l_shift)/CHI, (ELL3+l_shift)/CHI

        # compute IA bispectrum components
        if ia_bispec_comps is None:
            B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE = self.ia_bispectrum(K1, K2, K3, Z, z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta, Ct = Ct, remove_alignment=remove_alignment)
        else:
            B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE = ia_bispec_comps
            
        adjust = 1 / chi_los * (1 + z_los) ** 3
        integrand_ddE = kernel_1 * kernel_2 * W_3 * B_ddE * adjust
        integrand_dEd = kernel_1 * W_2 * kernel_3 * B_dEd * adjust
        integrand_Edd = W_1 * kernel_2 * kernel_3 * B_Edd * adjust

        integrand_dEE = kernel_1 * W_2 * W_3 * B_dEE * adjust
        integrand_EEd = W_1 * W_2 * kernel_3 * B_EEd * adjust
        integrand_EdE = W_1 * kernel_2 * W_3 * B_EdE * adjust

        integrand_EEE = W_1 * W_2 * W_3 * B_EEE * adjust

        # Integrate each component
        if integrand_ddE.shape[1] > 1:
            bk_ddE = np.trapz(integrand_ddE, chi_los, axis=1)
            bk_dEd = np.trapz(integrand_dEd, chi_los, axis=1)
            bk_Edd = np.trapz(integrand_Edd, chi_los, axis=1)
            bk_dEE = np.trapz(integrand_dEE, chi_los, axis=1)
            bk_EEd = np.trapz(integrand_EEd, chi_los, axis=1)
            bk_EdE = np.trapz(integrand_EdE, chi_los, axis=1)
            bk_EEE = np.trapz(integrand_EEE, chi_los, axis=1)

        else:
            bk_ddE = kernel_1 * kernel_2 * W_3 * B_ddE
            bk_dEd = kernel_1 * W_2 * kernel_3 * B_dEd
            bk_Edd = W_1 * kernel_2 * kernel_3 * B_Edd
            bk_dEE = kernel_1 * W_2 * W_3 * B_dEE
            bk_EEd = W_1 * W_2 * kernel_3 * B_EEd
            bk_EdE = W_1 * kernel_2 * W_3 * B_EdE
            bk_EEE = W_1 * W_2 * W_3 * B_EEE

        if select_mode == None:
            bk_total = bk_ddE + bk_dEd + bk_Edd + bk_dEE + bk_EEd + bk_EdE + bk_EEE

        elif select_mode == 'ddE':
            bk_total = bk_ddE
        elif select_mode == 'dEd':
            bk_total = bk_dEd
        elif select_mode == 'Edd':
            bk_total = bk_Edd
        elif select_mode == 'dEE':
            bk_total = bk_dEE
        elif select_mode == 'EEd':
            bk_total = bk_EEd
        elif select_mode == 'EdE':
            bk_total = bk_EdE
        elif select_mode == 'EEE':
            bk_total = bk_EEE

        # multiply window 
        if hasattr(self, 'window_function') and window:
            bk_total *= self.window_function(ell1, ell2, ell3)

        # reshape to the original shape
        bk_total = bk_total.reshape(shape)

        # convert to scalar if input is scalar
        if isscalar:
            bk_total = bk_total[0]

        if return_ia_bispec_comps:
            return bk_total, (B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE)
        else:
            return bk_total

    # interpolation
    def interpolate(self, scombs=None, select_tatt_component=None, Ct = 0, remove_alignment=False, ell_grid = False, puv_grid = False, **args):
        """
        Interpolate kappa bispectrum. 
        The interpolation is done in (r,u,v)-space, which is defined in M. Jarvis+2003 
        (https://arxiv.org/abs/astro-ph/0307393). See also treecorr homepage
        (https://rmjarvis.github.io/TreeCorr/_build/html/correlation3.html).

        Parameters:
            scombs (list): list of sample combinations
            args (dict): arguments for matter_bispectrum
        """
        # If sample_combinations is not given, 
        # we get all possible combinations.
        if scombs is None:
            scombs = self.get_all_sample_combinations()
        # Prepare for the interpolation
        bm = None
        if not ell_grid and not puv_grid:
            grid = (np.log(self.r_interp), np.log(self.u_interp), self.v_interp)
            #print('grid shape',np.shape(grid))
            #print('dimension 1 check',grid[1])
        elif puv_grid:
            grid = (np.log(self.p_interp), self.pu_interp, self.pv_interp)
        else:
            grid = (self.ell1_interp, self.ell2_interp, self.ell3_interp)
            #print('grid shape',np.shape(grid))
            #print('dimension 1 check',grid[1])

        if hasattr(self, 'ia_bispectra_calculator'):
            for sc in scombs:
                sc = self.parse_sample_combination(sc)
                bk = self.kappa_bispectrum_IA_direct(
                    self.ELL1_interp,
                    self.ELL2_interp,
                    self.ELL3_interp,
                    scomb=sc,
                    window=False,
                    select_mode=select_tatt_component,
                    Ct = Ct,
                    remove_alignment = remove_alignment,
                    **args)
                #not doing log here
                #print(np.shape(bk))
                #print(np.shape(grid))
                self.bk_interp[sc] = rgi(grid, bk, method=self.method_interp)

        else:
            for sc in scombs:
                sc = self.parse_sample_combination(sc)
                bk, bm = self.kappa_bispectrum_direct(
                    self.ELL1_interp,
                    self.ELL2_interp,
                    self.ELL3_interp,
                    scomb=sc,
                    window=False,
                    bm=bm,
                    return_bm=True,
                    **args)
                self.bk_interp[sc] = rgi(grid, np.log(bk), method=self.method_interp)

    def kappa_bispectrum_interp(self, ell1, ell2, ell3, scomb=None, ell_grid=False, puv_grid = False):
        """
        Compute kappa bispectrum by interpolation.

        Parameters:
            ell1 (array): ell1 array
            ell2 (array): ell2 array
            ell3 (array): ell3 array
        """
        scomb = self.parse_sample_combination(scomb)
        ip = self.bk_interp[scomb]
        r, u, v = trigutils.x1x2x3_to_ruv(ell1, ell2, ell3, signed=False)
        x = edge_correction(np.log(r), ip.grid[0].min(), ip.grid[0].max())
        y = edge_correction(np.log(u), ip.grid[1].min(), ip.grid[1].max())
        z = edge_correction(v, ip.grid[2].min(), ip.grid[2].max())

        p = ell1+ell2+ell3
        pu = ell1/p
        pv = ell2/p
        px = edge_correction(np.log(p), ip.grid[0].min(), ip.grid[0].max())
        py = edge_correction(pu, ip.grid[1].min(), ip.grid[1].max())
        pz = edge_correction(pv, ip.grid[2].min(), ip.grid[2].max())

        if hasattr(self, 'ia_bispectra_calculator'):
            if ell_grid:
                bk = ip((ell1,ell2,ell3))
            elif puv_grid:
                bk = ip((px,py,pz))
            else:
                bk = ip((x,y,z))
        else:
            bk = np.exp(ip((x,y,z)))

        # multiply window 
        if hasattr(self, 'window_function'):
            bk *= self.window_function(ell1, ell2, ell3)
        return bk

    # multipole decomposition
    def decompose(self, scombs=None, method_bispec='interp', puv_grid = False, **args):
        """
        Compute multipole decomposition of kappa bispectrum.

        Parameters:
            scombs (list)       : list of sample combinations
            method_bispec (str) : method for kappa_bispectrum
            args (dict)         : arguments for kappa_bispectrum
        """
        # If sample_combinations is not given, 
        # we get all possible combinations.
        if scombs is None:
            scombs = self.get_all_sample_combinations()
        # Compute multipole
        for sc in scombs:
            sc = self.parse_sample_combination(sc)
            b = self.kappa_bispectrum(
                    self.ELL1_multipole, 
                    self.ELL2_multipole, 
                    self.ELL3_multipole, 
                    sc, 
                    method=method_bispec,
                    puv_grid = puv_grid,
                    **args)
            # Compute multipoles
            L = np.arange(self.Lmax_multipole+1)
            bL = self.multipole_decomposer.decompose(b, L, axis=2)
            self.bL_multipole[sc] = bL

        # You may want to calculate higher multipole especially for 
        # diagonal elements, where ell1= ell2, corresponds to the 
        # squeezed limit bispectrum.
        if self.Lmax_multipole_diag <= self.Lmax_multipole:
            return 0
        print('Decomposing for diag.')
        for sc in scombs:
            sc = self.parse_sample_combination(sc)
            b = self.kappa_bispectrum(
                    self.ELL1_multipole_diag, 
                    self.ELL1_multipole_diag, 
                    self.ELL1_multipole_diag, 
                    sc, 
                    method=method_bispec,
                    puv_grid = puv_grid,
                    **args)
            # Compute multipoles
            L = np.arange(self.Lmax_multipole, self.Lmax_multipole_diag+1)
            bL = self.multipole_decomposer.decompose(b, L, axis=1)
            self.bL_multipole_diag[sc] = bL


    def kappa_bispectrum_multipole(self, L, ell, psi, scomb=None):
        """
        Compute multipole of kappa bispectrum.

        Parameters:
            L (array)     : multipole
            ell (array)   : ell array
            psi (array)   : psi array
            scomb (tuple) : sample combination
        """
        # parse sample_combination
        scomb = self.parse_sample_combination(scomb)
        # cast to array
        isscalar = np.isscalar(L)
        if isscalar:
            L = np.array([L])

        # compute multipole
        out = np.zeros((L.size,) + ell.shape)
        grid = (np.log(self.ell_multipole), np.log(self.psi_multipole))
        for i, _L in enumerate(L):
            # interpolate
            z = self.bL_multipole[scomb][_L, :, :]
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

    def kappa_bispectrum_multipole_diag(self, L, ell1, scomb=None):
        """
        Compute multipole of kappa bispectrum.

        Parameters:
            L (array)     : multipole
            ell1 (array)   : ell1 array
            scomb (tuple) : sample combination
        """
        # parse sample_combination
        scomb = self.parse_sample_combination(scomb)
        # cast to array
        isscalar = np.isscalar(L)
        if isscalar:
            L = np.array([L])

        # compute multipole
        out = np.zeros((L.size,) + ell1.shape)
        for i, _L in enumerate(L):
            # interpolate
            z = self.bL_multipole_diag[scomb][_L-self.Lmax_multipole, :]
            ip= ius(np.log(self.ell_multipole/2**0.5), z)
            out[i] = ip(np.log(ell1))

        if isscalar:
            out = out[0]
            
        return out

    def kappa_bispectrum_resum(self, ell1, ell2, ell3, scomb=None, Lmax=None):
        """
        Compute kappa bispectrum by resummation of multipoles.

        Parameters:
            L (array)     : multipole
            ell (array)   : ell array
            psi (array)   : psi array
            scomb (tuple) : sample combination
        """
        scomb = self.parse_sample_combination(scomb)
        ell, psi, mu = trigutils.x1x2x3_to_xpsimu(ell1, ell2, ell3)
        L = np.arange(Lmax or self.Lmax_multipole)
        bL = self.kappa_bispectrum_multipole(L, ell, psi, scomb=scomb)
        # pL = np.array([eval_legendre(_L, mu) for _L in L])
        _ = np.linspace(-1, 1, 100)
        pL = np.array([ius(_, eval_legendre(_L, _))(mu) for _L in L])
        out = np.sum(bL*pL, axis=0)
        return out

    def set_baryon_model(self, baryon_model):
        """
        Set a model of bispectrum suppression due to baryon
        as a function of (k1,k2,k3,z).
        Model parameter must be feeded already.
        """
        self.baryon_model = baryon_model


class BispectrumHalofit(BispectrumBase):
    """
    Bispectrum computed from halofit.
    """
    __doc__ += BispectrumBase.__doc__
    # default configs
    config_scale     = dict(ell1min=1e-1, ell1max=1e5, epmu=1e-7)
    
    def __init__(self, config=None, **kwargs):
        self.halofit = Halofit()
        super().__init__(config, **kwargs)

    def set_cosmology(self, cosmo, ns=None, sigma8=None):
        """
        Sets cosmology. 

        Parameters:
            cosmo (astropy.cosmology): cosmology
            ns (float)               : spectral index of linear power spectrum
            sigma8 (float)           : sigma8 of linear power spectrum (at z=0.0)

        Note:
            Note that the values of ns and sigma8 are set by two ways:
            1. Assigning ns and sigma8 as arguments of this method.
            2. Assigning ns and sigma8 to cosmo.meta.
        """
        super().set_cosmology(cosmo)
        # parameters for halofit
        dcosmo={'Om0': cosmo.Om0, 
                'Ode0': cosmo.Ode0,
                'ns': ns or cosmo.meta.get('n'),
                'sigma8': sigma8 or cosmo.meta.get('sigma8'), 
                'w0': cosmo.w0, 
                'wa': 0.0,
                'fnu0': 0.0} 
        self.halofit.set_cosmology(dcosmo)

    def set_pklin(self, k, pklin):
        """
        Set linear power spectrum.

        Parameters:
            k (array)    : wavenumber array
            pklin (array): linear power spectrum
        """
        self.halofit.set_pklin(k, pklin)
        self.has_changed = True

    def set_lgr(self, z, lgr):
        """
        Set linear growth rate.

        Parameters:
            z (float)  : redshift
            lgr (float): linear growth rate
        """
        self.z2lgr = ius(z, lgr, ext=1)
        self.halofit.set_lgr(z, lgr)
        self.has_changed = True

    def matter_bispectrum_no_baryon(self, k1, k2, k3, z, all_physical=True, which=['Bh1', 'Bh3']):
        b = self.halofit.get_bihalofit(k1, k2, k3, z, all_physical=all_physical, which=which)
        return b

class BispectrumTATT(BispectrumBase):
    """
    Bispectrum computed from intrinsic alignment (TATT model).
    """
    __doc__ += BispectrumBase.__doc__
    # default configs
    config_scale     = dict(ell1min=1e-1, ell1max=1e5, epmu=1e-7)
    
    def __init__(self, config=None, **kwargs):
        self.ia_bispectra_calculator = BispectraIA() #Initializes computation of TATT bispectra
        self.halofit = Halofit() #Initializes computation of Bihalofit for renormalization
        super().__init__(config, **kwargs)
        self.IA_params = {}
        self.baryon_params = {'fb': 0.0, 'suppress_only': False}

    def set_cosmology(self, cosmo, ns=None, sigma8=None):
        """
        Sets cosmology. 

        Parameters:
            cosmo (astropy.cosmology): cosmology
            ns (float)               : spectral index of linear power spectrum
            sigma8 (float)           : sigma8 of linear power spectrum (at z=0.0)
        """
        super().set_cosmology(cosmo)
        # parameters for IABispectra
        dcosmo={'Om0': cosmo.Om0, 
                'Ode0': cosmo.Ode0,
                'ns': ns or cosmo.meta.get('n'),
                'sigma8': sigma8 or cosmo.meta.get('sigma8'), 
                'w0': cosmo.w0, 
                'wa': 0.0,
                'fnu0': 0.0} 
        self.ia_bispectra_calculator.set_cosmology(dcosmo)
        self.halofit.set_cosmology(dcosmo)

    def set_pklin(self, k, pklin):

        """
        Set linear power spectrum.

        Parameters:
            k (array)    : wavenumber array
            pklin (array): linear power spectrum

        """
        self.ia_bispectra_calculator.set_pklin(k, pklin)
        self.halofit.set_pklin(k, pklin)
        self.has_changed = True
        
    def set_pknl(self, k, pknl):

        """
        Set non-linear power spectrum.

        Parameters:
            k (array)    : wavenumber array
            pknl (array): non-linear power spectrum

        """
        self.ia_bispectra_calculator.set_pknl(k, pknl)
        self.has_changed = True
        
    def set_lgr(self, z, lgr):
        """
        Set linear growth rate.#

        Parameters:
            z (float)  : redshift
            lgr (float): linear growth rate
        """
        self.z2lgr = ius(z, lgr, ext=1)
        self.ia_bispectra_calculator.set_lgr(z, lgr)
        self.halofit.set_lgr(z, lgr)
        self.ia_bispectra_calculator.z2lgr = self.z2lgr
        self.has_changed = True

    def set_IA_param(self, params):
        """
        Set parameters for Intrinsic Alignment.

        Parameters:
            params (dict) : parameters for intrinsic alignment effect (a1, alpha1, a2, alpha2, bias_ta)
        """
        if 'a1' not in params or 'alphaIA' not in params or 'a2' not in params or 'alphaIA_2' not in params or 'bias_ta' not in params:
            raise ValueError('a1, alphaIA, a2, alphaIA_2, and bias_ta must be given as parameters.')
        self.IA_params.update(params)

    def set_baryon_param(self, params):
        """
        Set parameter(s) of baryon for bispectrum renormalization

        keywords:
            fb: suppression factor relative to TNG-300
        """
        if 'fb' not in params:
            raise ValueError('fb must be given as a parameter (float)')
        self.baryon_params.update(params)

    def ia_bispectrum(self, k1, k2, k3, z, z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta, Ct = 0, remove_alignment=False):

        if remove_alignment:
            #B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE, B_ddB, B_dBd, B_Bdd, B_dEB, B_dBE, B_EBd, B_BEd, B_BdE, B_EdB, B_EEB, B_EBE, B_BEE = self.ia_bispectra_calculator.get_ia_bispectra(k1, k2, k3, z, z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta, remove_alignment=True)
            all_components = self.ia_bispectra_calculator.get_ia_bispectra(k1, k2, k3, z, z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta, Ct=Ct,
                                                          remove_alignment=True)
        else:
            #B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE, B_ddB, B_dBd, B_Bdd, B_dEB, B_dBE, B_EBd, B_BEd, B_BdE, B_EdB, B_EEB, B_EBE, B_BEE = self.ia_bispectra_calculator.get_ia_bispectra(k1, k2, k3, z, z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta, remove_alignment=False)
            all_components = self.ia_bispectra_calculator.get_ia_bispectra(k1, k2, k3, z, z_piv, A1, alphaIA, A2, alphaIA_2, bias_ta, Ct =Ct, remove_alignment=False)

        all_components = [np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in all_components]
        (B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE, B_ddB, B_dBd, B_Bdd, B_dEB, B_dBE, B_EBd, B_BEd, B_BdE, B_EdB, B_EEB, B_EBE, B_BEE) = all_components

        return B_ddE, B_dEd, B_Edd, B_dEE, B_EEd, B_EdE, B_EEE


class BispectrumGilMarin(BispectrumBase):
    """
    Bispectrum computed from Gil-Marin.

    reference: https://arxiv.org/pdf/1111.4477
    """
    __doc__ += BispectrumBase.__doc__
    # default configs
    config_scale     = dict(ell1min=1e-1, ell1max=1e5, epmu=1e-7)
    
    def __init__(self, config=None, **kwargs):
        self.halofit = Halofit()
        super().__init__(config, **kwargs)

        # gil marin's parameters
        self.a1 = 0.484
        self.a2 = 3.740
        self.a3 = -0.849
        self.a4 = 0.392
        self.a5 = 1.013
        self.a6 = -0.575
        self.a7 = 0.128
        self.a8 = -0.722
        self.a9 = -0.926

    def set_cosmology(self, cosmo, ns=None, sigma8=None):
        """
        Sets cosmology. 

        Parameters:
            cosmo (astropy.cosmology): cosmology
            ns (float)               : spectral index of linear power spectrum
            sigma8 (float)           : sigma8 of linear power spectrum (at z=0.0)

        Note:
            Note that the values of ns and sigma8 are set by two ways:
            1. Assigning ns and sigma8 as arguments of this method.
            2. Assigning ns and sigma8 to cosmo.meta.
        """
        super().set_cosmology(cosmo)
        # parameters for halofit
        dcosmo={'Om0': cosmo.Om0, 
                'Ode0': cosmo.Ode0,
                'ns': ns or cosmo.meta.get('n'),
                'sigma8': sigma8 or cosmo.meta.get('sigma8'), 
                'w0': cosmo.w0, 
                'wa': 0.0,
                'fnu0': 0.0} 
        self.halofit.set_cosmology(dcosmo)

    def set_pklin(self, k, pklin):
        """
        Set linear power spectrum.

        Parameters:
            k (array)    : wavenumber array
            pklin (array): linear power spectrum
        """
        self.halofit.set_pklin(k, pklin)
        self.has_changed = True

    def set_lgr(self, z, lgr):
        """
        Set linear growth rate.

        Parameters:
            z (float)  : redshift
            lgr (float): linear growth rate
        """
        self.z2lgr = ius(z, lgr, ext=1)
        self.halofit.set_lgr(z, lgr)
        self.has_changed = True

    def set_interp(self):
        if self.has_changed:
            self.set_k2n_gilmarin()
            self.set_z2knl_gilmarin()
            self.set_z2sigma8()
            self.has_changed = False

    def set_k2n_gilmarin(self):
        """
        Get effective spectral index from Gil-Marin.
        """
        k = self.halofit.k
        pklin = self.halofit.pklin
        neff = np.diff(np.log(pklin))/np.diff(np.log(k))
        self.k2n = ius((k[1:]*k[:-1])**0.5, neff, ext=0)

    def set_z2knl_gilmarin(self):
        """
        Get non-linear scale from Gil-Marin.
        """
        # Delta without linear growht factor
        k = self.halofit.k
        pklin = self.halofit.pklin
        Delta = k**3*pklin/(2*np.pi**2)
        # growth
        z = np.linspace(0, self.halofit.z.max(), 100)
        lgr = self.z2lgr(z)
        knl = np.full_like(z, k.max())
        for i in range(z.size):
            _ = k[Delta*lgr[i]**2 < 1.0]
            if _.size > 0:
                knl[i] = _.max()
        self.z2knl = ius(z, knl, ext=0)

    def set_z2sigma8(self):
        z = np.linspace(0, self.halofit.z.max(), 100)
        s8= np.array([self.halofit.get_sigma8z(_z) for _z in z])
        self.z2sigma8 = ius(z, s8, ext=0)

    def Q3(self, n):
        return (4 - 2 ** n) / (1 + 2 ** (n + 1))

    def agm(self, k, z, knl):
        n = self.k2n(k)
        q = k / knl
        factor = (q * self.a1) ** (n + self.a2)
        sigma8 = self.z2sigma8(z)
        return (1 + sigma8**self.a6 * (0.7 * self.Q3(n) ** (1 / 2)) * factor) / (1 + factor)

    def bgm(self, k, knl):
        q = k / knl
        n = self.k2n(k)
        return (1 + 0.2 * self.a3 * (n + 3) * (q * self.a7) ** (n + 3 + self.a8)) / (1 + (q * self.a7) ** (n + 3.5 + self.a8))

    def cgm(self, k, knl):
        q = k / knl
        n = self.k2n(k)
        return (1 + 4.5 * self.a4 / (1.5 + (n + 3) ** 4) * (q * self.a5) ** (n + 3 + self.a9)) / (1 + (q * self.a5) ** (n + 3.5 + self.a9))

    def F2_eff(self, z, k1, k2, k3, knl):
        dot = (-k3 ** 2 + k1 ** 2 + k2 ** 2) / 2
        # f2 = 5 / 7 * self.agm(k1, z, knl) * self.agm(k2, z, knl) + 2 * dot ** 2 / (7 * k1 ** 2 * k2 ** 2) * self.bgm(k1, knl) * self.bgm(k2, knl) - dot * (
        #             1 / k1 ** 2 + 1 / k2 ** 2) / 2 * self.cgm(k1, knl) * self.cgm(k2, knl)
        f2 = 5 / 7 + 2 * dot ** 2 / (7 * k1 ** 2 * k2 ** 2) - dot * (1 / k1 ** 2 + 1 / k2 ** 2) / 2
        return f2

    def matter_bispectrum_no_baryon(self, k1, k2, k3, z, **kwargs):
        print("Note: Gilmarin ignores kwargs.")

        PNL1 = self.halofit.get_pkhalofit(k1[:,0], z[0,:]).T
        PNL2 = self.halofit.get_pkhalofit(k2[:,0], z[0,:]).T
        PNL3 = self.halofit.get_pkhalofit(k3[:,0], z[0,:]).T

        # reshape
        shape = k1.shape
        k1 = k1.ravel()
        k2 = k2.ravel()
        k3 = k3.ravel()
        z = z.ravel()
        PNL1 = PNL1.ravel()
        PNL2 = PNL2.ravel()
        PNL3 = PNL3.ravel()

        self.set_interp()
        knl = self.z2knl(z)

        bk  =  2 * (self.F2_eff(z, k1, k2, k3, knl) * PNL1 * PNL2 + \
                    self.F2_eff(z, k2, k3, k1, knl) * PNL2 * PNL3 + \
                    self.F2_eff(z, k3, k1, k2, knl) * PNL3 * PNL1)

        bk  = bk.reshape(shape)
        return bk

class BispectrumNFW1Halo(BispectrumBase):
    """
    Toy model
    """
    __doc__ += BispectrumBase.__doc__
    # default configs
    config_scale     = dict(ell1min=1e-2, ell1max=1e5, epmu=1e-7)
    
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
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
