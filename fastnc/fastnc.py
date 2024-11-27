#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/11/27 14:12:40

Description:
This is the module of fastnc, which calculate the
natural components using 2d fftlog.
'''
import numpy as np
# fastnc modules
from .twobessel import two_Bessel
from . import trigutils
from .utils import sincos2angbar, update_config, get_config_key
from .coupling import MCF222LegendreFourier, MCF222FourierFourier

class FastNaturalComponents:
    r"""
    Calculates the natural components using 2dfftlog.

    Parameters:
        config (dict)         : A configuration dict that can be used to pass in the below kwargs if
                                desired.  This dict is allowed to have addition entries in addition
                                to those listed below, which are ignored here. (default: None)
        verbose (bool)        : Whether to print the progress. (default: True)
                                
    Keyword Arguments:
        Lmax (int)            : The maximum multipole moment.
        Mmax (int)            : The maximum angular Fourier mode.
        projection (str)      : The projection of the shear. (default: 'x')
        t1 (array)            : The bin on two sides of a triangle. The theoretical prediction is given on
                                this bin if an array is given. If dlnt is also givn, this is interpreted as
                                the lower edge of the theta bins. If None, the theory is calculated on
                                FFT grid. (default: None)
        phi (array)           : The opening angle of the triangle between t1 and t2. If None is given, the 
                                class calculates the multipole of natural computes up to Mmax, which is
                                stored to the class sttributed Gamma0M, Gamma1M, Gamma2M, and Gamma3M.
                                If an array is given, this class also computes natural component on the given
                                phi bins from multipoles, which is stored to Gamma0, Gamma1, Gamma2, Gamma3.
                                (default: None)
        mu (list)             : The index of the natural component. (default: [0,1,2,3])
        dlnt (float)          : The bin width of t, on which we want to evaluate the result. (default: None)
        nu1 (float)           : The hyper parameter for 2dFFTLog. (default: 1.01)
        nu2 (float)           : The hyper parameter for 2dFFTLog. (default: 1.01)
        N_pad (int)           : The padding size for 2dFFTLog. (default: 0)
        xy (float)            : The paramater to relate real and fourier bins of FFT.
        auto (bool)           : Whether to automatically set ell1min, ell1max, and nell1bin from bispectrum. 
                                (default: True).
        ell1min (float)       : The minimum value of ell1. (default: None)
        ell1max (float)       : The maximum value of ell1. (default: None)
        nfft (int)            : The number of ell1 (or ell2) FFT grid points per one side of triangle. 
                                (default: 150)
        multipole_type (str)  : The type of multipole. (default: 'legendre')
        cache (bool)          : Whether to use cache for mode coupling function. (default: True)
        verbose (bool)        : Whether to print the progress. (default: True)

    Usage:
        >>> # Initialize the class
        >>> fnc = FastNaturalComponents(Lmax=10, Mmax=10)
        >>> # Set the bispectrum
        >>> fnc.set_bispectrum(bispectrum)
        >>> # Compute the natural components
        >>> fnc.compute()
        >>> # access to the computed natural components
        >>> print('Natural component multipole')
        >>> print(fnc.Gamma0M)
        >>> print('corresponding bins', fnc.get_bins('multipole'))
        >>> print('Natural compoenent in SAS')
        >>> print(fnc.Gamma0)
        >>> print('corresponding bins', fnc.get_bins('SAS'))
    """
    # default configuration
    projection       = 'x'
    config_multipole = {'Lmax':None, 'Mmax':None, 'Lmax_diag':None, 'multipole_type':'legendre', \
                        'use_GLM_table':False, 'cache':True}
    config_bin       = {'t1':None, 'phi':None, 'mu':[0,1,2,3], 'dlnt':None}
    config_fftlog    = {'nu1':1.01, 'nu2':1.01, 'N_pad':0, 'xy':1}
    config_fftgrid   = {'auto':True, 'ell1min':None, 'ell1max':None, 'nfft':150}
        
    def __init__(self, config=None, verbose=True, **kwargs):
        # general setup
        self.verbose = verbose
        # setup shear projection
        self.projection = get_config_key(config, 'projection', self.projection, **kwargs)
        # setup multipole
        self.set_multipole(config, **kwargs)
        # setup bin
        self.set_bin(config, **kwargs)
        # setup fft grid config
        update_config(self.config_fftgrid, config, **kwargs)
        # setup fftlog config
        update_config(self.config_fftlog, config, **kwargs)
    
    def set_multipole(self, config=None, **kwargs):
        """
        Initialize mode coupling function.

        Parameters:
            config (dict)         : A configuration dict that can be used to pass in the below kwargs if
                                    desired.  This dict is allowed to have addition entries in addition
                                    to those listed below, which are ignored here. (default: None)
                                    
        Keyword Arguments:
            Lmax (int)            : The maximum multipole moment.
            Mmax (int)            : The maximum angular Fourier mode.
            multipole_type (str)  : The type of multipole. Either 'legendre' or 'fourier'.
        """
        # update config
        update_config(self.config_multipole, config, **kwargs)
        # initialize mode coupling function
        self.Lmax = self.config_multipole['Lmax']
        self.Lmax_diag = self.config_multipole['Lmax_diag']
        if self.Lmax_diag is None:
            self.Lmax_diag = self.Lmax
        self.Mmax = self.config_multipole['Mmax']
        self.multipole_type = self.config_multipole['multipole_type']
        if self.multipole_type == 'legendre':
            self.GLM = MCF222LegendreFourier(self.Lmax_diag, self.Mmax, verbose=self.verbose, cache=self.config_multipole['cache'])
        elif self.multipole_type == 'fourier':
            self.GLM = MCF222FourierFourier(self.Lmax_diag, self.Mmax, verbose=self.verbose, cache=self.config_multipole['cache'])
        else:
            raise ValueError('Error: multipole_type={} is not expected'.format(self.multipole_type))

    def set_bispectrum(self, bispectrum):
        """
        Set the bispectrum multipoles.

        bispectrum (Bispectrum): The bispectrum object.
        """
        # check the compatibility of multipole_type
        assert self.multipole_type == bispectrum.multipole_type, \
            "multipole_type of bispectrum and FastNaturalComponents must be the same"
        # set bispectrum multipole
        self.bispectrum = bispectrum
        self.set_fftgrid()
        
    def set_bin(self, config=None, **kwargs):
        """
        Set the bin on which we want to predict theory.

        Parameters:
            config (dict)         : A configuration dict that can be used to pass in the below kwargs if
                                    desired.  This dict is allowed to have addition entries in addition
                                    to those listed below, which are ignored here. (default: None)
        
        Keyword Arguments:
            t1 (array)            : The bin on two sides of a triangle. The theoretical prediction is given on
                                    this bin if an array is given. If dlnt is also givn, this is interpreted as
                                    the lower edge of the theta bins. If None, the theory is calculated on
                                    FFT grid. (default: None)
            phi (array)           : The opening angle of the triangle between t1 and t2. If None is given, the 
                                    class calculates the multipole of natural computes up to Mmax, which is
                                    stored to the class sttributed Gamma0M, Gamma1M, Gamma2M, and Gamma3M.
                                    If an array is given, this class also computes natural component on the given
                                    phi bins from multipoles, which is stored to Gamma0, Gamma1, Gamma2, Gamma3.
                                    (default: None)
            mu (list)             : The index of the natural component. (default: [0,1,2,3])
            dlnt (float)          : The bin width of t, on which we want to evaluate the result. (default: None)
        """
        # update config
        update_config(self.config_bin, config, **kwargs)
        # Set the bin on which we want to predict theory
        self.t1 = self.config_bin['t1']
        self.phi = self.config_bin['phi']
        self.mu  = self.config_bin['mu']
        self.dnlt = self.config_bin['dlnt']
        self.tune_fftgrid = self.t1 is not None

    def set_fftgrid(self, config=None, **kwargs):
        """
        Set the grid in real and fourier space.

        Parameters:
            config (dict)         : A configuration dict that can be used to pass in the below kwargs if
                                    desired.  This dict is allowed to have addition entries in addition
                                    to those listed below, which are ignored here. (default: None)
        
        Keyword Arguments:
            auto (bool)           : Whether to automatically set ell1min, ell1max, and nell1bin from bispectrum. 
                                    (default: True).
            ell1min (float)       : The minimum value of ell1. (default: None)
            ell1max (float)       : The maximum value of ell1. (default: None)
            nfft (int)            : The number of ell1 (or ell2) FFT grid points per one side of triangle. 
                                    (default: 150)
        """
        # update config
        update_config(self.config_fftgrid, config, **kwargs)
        # Set the range of Fourier modes:
        # This is set by config or automatically
        # sourced from bispectrum support range.
        if self.config_fftgrid['auto']:
            assert hasattr(self, 'bispectrum'), \
                "bispectrum must be set to use auto mode"
            ell1min = self.bispectrum.ell1min
            ell1max = self.bispectrum.ell1max
        else:
            ell1min = self.config_fftgrid['ell1min']
            ell1max = self.config_fftgrid['ell1max']
        # 
        if self.tune_fftgrid:
            print('Tuning FFT bins...') if self.verbose else None
            # theta bin on which we want predict
            # Tune FFT grid
            
            _ = get_tuned_fftgrid(np.log(ell1min), np.log(ell1max), \
                self.config_fftgrid['nfft'], np.log(self.t1))
            self.ell1_fft = np.exp(_[0])
            self.t1_fft   = np.exp(_[1])
            self.down_sampler = _[2]
            self.config_fftlog['xy'] = np.exp(_[3])
            assert np.allclose(self.t1, self.t1_fft[self.down_sampler]), \
                "Something went wrong in tuning FFT grid\n" \
                "t1={}\nt1_fft={}".format(self.t1, self.t1_fft[self.down_sampler])
        else:
            nfft= self.config_fftgrid['nfft']
            self.ell1_fft = np.logspace(np.log10(ell1min), np.log10(ell1max), nfft)
            self.t1 = self.t1_fft = self.config_fftlog['xy']/self.ell1_fft[::-1]
            self.down_sampler = np.arange(nfft) # no downsampling
        # Allocate the same bin to second side
        self.t2 = self.t1
        self.t2_fft = self.t1_fft
        self.ell2_fft = self.ell1_fft
        # 2dFFT grid in Fourier space
        self.ELL1_FFT, self.ELL2_FFT = np.meshgrid(self.ell1_fft, self.ell2_fft, indexing='ij')
        self.ELL_FFT  = np.sqrt(self.ELL1_FFT**2 + self.ELL2_FFT**2)
        self.PSI_FFT = np.arctan2(self.ELL2_FFT, self.ELL1_FFT)
        if self.config_multipole['use_GLM_table']:
            print('Preparing GLM table...') if self.verbose else None
            self.GLM.set_table(self.PSI_FFT)
    
    def HM(self, M, bL=None, L=None, bL_diag=None, L_diag=None, **args):
        """
        Compute H_M(l1, l2 = \\sum_L (-1)^L * G_LM * b_L(l1, l2) on FFT grid.

        Parameters:
            M (int)   : The angular Fourier mode.
            bL (array): The bispectrum multipole. Defaults to None.
                        If None, it is computed using self.bispectrum.kappa_bispectrum_multipole.
                        By supplying bL, you can avoid recomputation of bL.
        """
        # Get bispectrum multipole indices, L array
        if L is None:
            L = np.arange(self.Lmax+1)
        # Get bispectrum multipole
        if bL is None:
            bL = self.bispectrum.kappa_bispectrum_multipole(
                L, self.ELL_FFT, self.PSI_FFT, **args)
        if self.config_multipole['use_GLM_table']:
            GLM = self.GLM.from_table(L, M)
        else:
            GLM = self.GLM(L, M, self.PSI_FFT)
        # Sum up GLM*bL over L
        HM = np.sum(((-1)**L*GLM.T*bL.T).T, axis=0)

        if self.Lmax_diag <= self.Lmax:
            return HM

        # Do the same for diag
        if L_diag is None:
            L_diag = np.arange(self.Lmax, self.Lmax_diag+1)
        if bL_diag is None:
            bL_diag = self.bispectrum.kappa_bispectrum_multipole_diag(
                L_diag, self.ell1_fft, **args)
        GLM_diag = self.GLM(L_diag, M, np.array([np.pi/4]))[0,:]
        HM_diag = np.sum(((-1)**L_diag*GLM_diag.T*bL_diag.T).T, axis=0)
        HM += np.diag(HM_diag)

        return HM

    def get_bins(self, bin_type='multipole', t1_unit='radian', t1_rep='mid', mesh=True):
        """
        Get the bins on which the theory is computed.

        Parameters:
            bin_type (str): The type of bin. Either 'multipole' or 'SAS'.
            t1_unit (str) : The unit of t1. Either 'radian' or 'arcmin'.
            t1_rep  (str) : 'lower', 'mid', or 'upper'. The representative of the t1/t2 bin.
            mesh (bool)   : Whether to return meshgrid or not.
        
        Returns:
            bin1 (array): The first bin.
            bin2 (array): The second bin.
            bin3 (array): The third bin.
        """
        # Factor for unit conversion for theta1 and theta2
        if t1_unit == 'radian':
            ufactor = 1.0
        elif t1_unit in ['arcmin', 'minute']:
            ufactor = 180.0*60.0/np.pi
        else:
            raise ValueError('Error: t1_unit={} is not expected'.format(t1_unit))
        # Factor to convert the lower edge to middle or upper edge.
        if self.config_bin['dlnt'] is None:
            rfactor = 1.0
        else:
            if t1_rep == 'lower':
                rfactor = 1.0
            elif t1_rep == 'mid':
                rfactor = np.exp(self.config_bin['dlnt']/2)
            elif t1_rep == 'upper':
                rfactor = np.exp(self.config_bin['dlnt'])
            else:
                raise ValueError('Error: t1_rep={} is not expected'.format(t1_rep))
        # get bins
        if bin_type == 'multipole':
            bin1 = self.t1 * ufactor * rfactor
            bin2 = self.t2 * ufactor * rfactor
            bin3 = np.arange(-self.Mmax, self.Mmax+1)
        elif bin_type == 'SAS':
            bin1 = self.t1 * ufactor * rfactor
            bin2 = self.t2 * ufactor * rfactor
            bin3 = self.phi
        else:
            raise ValueError('Error: bin_type={} is not expected'.format(bin_type))
        if mesh:
            bin1, bin2, bin3 = np.meshgrid(bin1, bin2, bin3, indexing='ij')
        return bin1, bin2, bin3
    
    def compute(self, **args):
        """
        Compute GammaM and Gamma.

        args is a dictionary of arguments to be passed to
        self.bispectrum.kappa_bispectrum_multipole, e.g.
        sample_combination.
        """
        timer = args.pop('timer', lambda _: None)
        # 
        Lmin = args.pop('Lmin', 0)
        Lmax = args.pop('Lmax', self.Lmax)
        Mmax = args.pop('Mmax', self.Mmax)
        # natural-component multipole indices
        M = np.arange(Mmax+1)
        L = np.arange(Lmin, Lmax+1)
        # First we compute the kernel HM for all M
        bL = self.bispectrum.kappa_bispectrum_multipole(
            L, self.ELL_FFT, self.PSI_FFT, **args)
        timer('multipole')
        if self.Lmax_diag > self.Lmax:
            L_diag = np.arange(self.Lmax, self.Lmax_diag+1)
            bL_diag = self.bispectrum.kappa_bispectrum_multipole_diag(
                L_diag, self.ell1_fft, **args)
            timer('multipole diag')
        else:
            L_diag = bL_diag = None
        HM = [self.HM(_, bL=bL, L=L, L_diag=L_diag, bL_diag=bL_diag) for _ in M]
        timer('HM')

        # GammaM
        self.Gamma0M = np.zeros((self.t1.size, self.t2.size, 2*Mmax+1))
        self.Gamma1M = np.zeros((self.t1.size, self.t2.size, 2*Mmax+1))
        self.Gamma2M = np.zeros((self.t1.size, self.t2.size, 2*Mmax+1))
        self.Gamma3M = np.zeros((self.t1.size, self.t2.size, 2*Mmax+1))
        # For the sake of speed, we first loop over M
        # and then over mu. This is because the kernels of the 
        # natural-component multipoles only depends on M but not on mu.
        # This allows us to compute the kernels only once for each M.
        for _M, _HM in zip(M, HM):
            tb  = two_Bessel( \
                self.ell1_fft, self.ell2_fft, \
                _HM*self.ELL1_FFT**2*self.ELL2_FFT**2, \
                **self.config_fftlog)
            for _mu in self.mu:
                # Get (n,m) from M.
                m, n = [(_M-3,-_M-3), (-_M-1,_M-1), (_M+1,-_M-3), (_M-3,-_M+1)][_mu]
                if self.config_bin['dlnt'] is None:
                    # compute GammaM on FFT grid
                    GM = tb.two_Bessel(np.abs(m), np.abs(n))[2]
                elif self.config_bin['dlnt'] is not None:
                    # compute GammaM on FFT grid with bin-averaging effect
                    GM = tb.two_Bessel_binave(np.abs(m), np.abs(n), \
                        self.config_bin['dlnt'], self.config_bin['dlnt'])[2]
                # Apply (-1)**m and (-1)**n
                # These originate to J_m(x) = (-1)^m J_{-m}(x)
                GM *= (-1.)**m if m<0 else 1
                GM *= (-1.)**n if n<0 else 1
                # normalization
                GM /= (2*np.pi)**3
                # downsample
                GM = GM[np.ix_(self.down_sampler, self.down_sampler)]

                # Store
                if _mu == 0:
                    self.Gamma0M[:,:, _M+Mmax] = GM
                    self.Gamma0M[:,:,-_M+Mmax] = GM.T
                elif _mu == 1:
                    self.Gamma1M[:,:, _M+Mmax] = GM.T
                    self.Gamma1M[:,:,-_M+Mmax] = GM
                elif _mu == 2:
                    self.Gamma2M[:,:, _M+Mmax] = GM
                    self.Gamma3M[:,:,-_M+Mmax] = GM.T
                elif _mu == 3:
                    self.Gamma3M[:,:, _M+Mmax] = GM
                    self.Gamma2M[:,:,-_M+Mmax] = GM.T
        timer('GammaM')

        if self.phi is None:
            return
        M = np.arange(-Mmax, Mmax+1)
        expMphi = np.exp(1j*M[:,None]*self.phi[None,:])
        # resum multipoles
        self.Gamma0 = np.dot(self.Gamma0M, expMphi)/(2*np.pi)
        self.Gamma1 = np.dot(self.Gamma1M, expMphi)/(2*np.pi)
        self.Gamma2 = np.dot(self.Gamma2M, expMphi)/(2*np.pi)
        self.Gamma3 = np.dot(self.Gamma3M, expMphi)/(2*np.pi)
        timer('Gamma')
        # change shear projection
        self._change_shear_projection('x', self.projection)
        timer('projection')
    
    def _change_shear_projection(self, dept, dest):
        """
        Change the shear projection.

        Parameters:
            dept (str): The current shear projection.
            dest (str): The destination shear projection.
        """
        if self.verbose and dept != dest:
            print('changing shear projection from {} to {}'.format(dept, dest))
        # attributes are 1d arrays, so we cast them to 3d arrays
        T1  = self.t1[:,None,None]
        T2  = self.t2[None,:,None]
        PHI = self.phi[None,None,:]
        # Convert
        if dept == dest:
            return
        elif dept == 'x' and dest == 'cent':
            self.Gamma0 *= x2cent(0, T1, T2, PHI)
            self.Gamma1 *= x2cent(1, T1, T2, PHI)
            self.Gamma2 *= x2cent(2, T1, T2, PHI)
            self.Gamma3 *= x2cent(3, T1, T2, PHI)
        elif dept == 'cent' and dest == 'x':
            self.Gamma0 /= ortho2cent(0, T1, T2, PHI)
            self.Gamma1 /= ortho2cent(1, T1, T2, PHI)
            self.Gamma2 /= ortho2cent(2, T1, T2, PHI)
            self.Gamma3 /= ortho2cent(3, T1, T2, PHI)
        elif dept == 'x' and dest == 'ortho':
            self.Gamma0 *= x2ortho(0, T1, T2, PHI)
            self.Gamma1 *= x2ortho(1, T1, T2, PHI)
            self.Gamma2 *= x2ortho(2, T1, T2, PHI)
            self.Gamma3 *= x2ortho(3, T1, T2, PHI)
        elif dept == 'ortho' and dest == 'x':
            self.Gamma0 /= x2ortho(0, T1, T2, PHI)
            self.Gamma1 /= x2ortho(1, T1, T2, PHI)
            self.Gamma2 /= x2ortho(2, T1, T2, PHI)
            self.Gamma3 /= x2ortho(3, T1, T2, PHI)
        elif dept == 'ortho' and dest == 'cent':
            self.Gamma0 *= x2cent(0, T1, T2, PHI)
            self.Gamma1 *= x2cent(1, T1, T2, PHI)
            self.Gamma2 *= x2cent(2, T1, T2, PHI)
            self.Gamma3 *= x2cent(3, T1, T2, PHI)
        elif dept == 'cent' and dest == 'ortho':
            self.Gamma0 /= x2cent(0, T1, T2, PHI)
            self.Gamma1 /= x2cent(1, T1, T2, PHI)
            self.Gamma2 /= x2cent(2, T1, T2, PHI)
            self.Gamma3 /= x2cent(3, T1, T2, PHI)
        else:
            raise ValueError('Error: dept={} and dest={} is not expected'.format(dept, dest))

    @classmethod
    def _calculateT(cls, s, t, k1, k2, k3):
        # First calculate q values:
        q1 = (s+t)/3.
        q2 = q1-t
        q3 = q1-s

        # |qi|^2 shows up a lot, so save these.
        # The a stands for "absolute", and the ^2 part is implicit.
        a1 = np.abs(q1)**2
        a2 = np.abs(q2)**2
        a3 = np.abs(q3)**2
        a123 = a1*a2*a3

        # These combinations also appear multiple times.
        # The b doesn't stand for anything.  It's just the next letter after a.
        b1 = np.conjugate(q1)**2*q2*q3
        b2 = np.conjugate(q2)**2*q1*q3
        b3 = np.conjugate(q3)**2*q1*q2

        if k1==1 and k2==1 and k3==1:

            # Some factors we use multiple times
            expfactor = -np.exp(-(a1 + a2 + a3)/2)

            # JBJ Equation 51
            # Note that we actually accumulate the Gammas with a different choice for
            # alpha_i.  We accumulate the shears relative to the q vectors, not relative to s.
            # cf. JBJ Equation 41 and footnote 3.  The upshot is that we multiply JBJ's formulae
            # by (q1q2q3)^2 / |q1q2q3|^2 for T0 and (q1*q2q3)^2/|q1q2q3|^2 for T1.
            # Then T0 becomes
            # T0 = -(|q1 q2 q3|^2)/24 exp(-(|q1|^2+|q2|^2+|q3|^2)/2)
            T0 = expfactor * a123 / 24

            # JBJ Equation 52
            # After the phase adjustment, T1 becomes:
            # T1 = -[(|q1 q2 q3|^2)/24
            #        - (q1*^2 q2 q3)/9
            #        + (q1*^4 q2^2 q3^2 + 2 |q2 q3|^2 q1*^2 q2 q3)/(|q1 q2 q3|^2)/27
            #       ] exp(-(|q1|^2+|q2|^2+|q3|^2)/2)
            T1 = expfactor * (a123 / 24 - b1 / 9 + (b1**2 + 2*a2*a3*b1) / (a123 * 27))
            T2 = expfactor * (a123 / 24 - b2 / 9 + (b2**2 + 2*a1*a3*b2) / (a123 * 27))
            T3 = expfactor * (a123 / 24 - b3 / 9 + (b3**2 + 2*a1*a2*b3) / (a123 * 27))

        else:
            # SKL Equation 63:
            k1sq = k1*k1
            k2sq = k2*k2
            k3sq = k3*k3
            Theta2 = ((k1sq*k2sq + k1sq*k3sq + k2sq*k3sq)/3.)**0.5
            k1sq /= Theta2   # These are now what SKL calls theta_i^2 / Theta^2
            k2sq /= Theta2
            k3sq /= Theta2
            Theta4 = Theta2*Theta2
            Theta6 = Theta4*Theta2
            S = k1sq * k2sq * k3sq

            # SKL Equation 64:
            Z = ((2*k2sq + 2*k3sq - k1sq) * a1 +
                 (2*k3sq + 2*k1sq - k2sq) * a2 +
                 (2*k1sq + 2*k2sq - k3sq) * a3) / (6*Theta2)
            expfactor = -S * np.exp(-Z) / Theta4

            # SKL Equation 65:
            f1 = (k2sq+k3sq)/2 + (k2sq-k3sq)*(q2-q3)/(6*q1)
            f2 = (k3sq+k1sq)/2 + (k3sq-k1sq)*(q3-q1)/(6*q2)
            f3 = (k1sq+k2sq)/2 + (k1sq-k2sq)*(q1-q2)/(6*q3)
            f1c = np.conjugate(f1)
            f2c = np.conjugate(f2)
            f3c = np.conjugate(f3)

            # SKL Equation 69:
            g1 = k2sq*k3sq + (k3sq-k2sq)*k1sq*(q2-q3)/(3*q1)
            g2 = k3sq*k1sq + (k1sq-k3sq)*k2sq*(q3-q1)/(3*q2)
            g3 = k1sq*k2sq + (k2sq-k1sq)*k3sq*(q1-q2)/(3*q3)
            g1c = np.conjugate(g1)
            g2c = np.conjugate(g2)
            g3c = np.conjugate(g3)

            # SKL Equation 62:
            T0 = expfactor * a123 * f1c**2 * f2c**2 * f3c**2 / (24.*Theta6)

            # SKL Equation 68:
            T1 = expfactor * (
                a123 * f1**2 * f2c**2 * f3c**2 / (24*Theta6) -
                b1 * f1*f2c*f3c*g1c / (9*Theta4) +
                (b1**2 * g1c**2 + 2*k2sq*k3sq*a2*a3*b1 * f2c * f3c) / (a123 * 27*Theta2))
            T2 = expfactor * (
                a123 * f1c**2 * f2**2 * f3c**2 / (24*Theta6) -
                b2 * f1c*f2*f3c*g2c / (9*Theta4) +
                (b2**2 * g2c**2 + 2*k1sq*k3sq*a1*a3*b2 * f1c * f3c) / (a123 * 27*Theta2))
            T3 = expfactor * (
                a123 * f1c**2 * f2c**2 * f3**2 / (24*Theta6) -
                b3 * f1c*f2c*f3*g3c / (9*Theta4) +
                (b3**2 * g3c**2 + 2*k1sq*k2sq*a1*a2*b3 * f1c * f2c) / (a123 * 27*Theta2))

        return T0, T1, T2, T3

    def calculateMap3(self, R, k2=1, k3=1, unit='arcmin'):
        r"""Calculate the skewness of the aperture mass from the correlation function.

        The equations for this come from Jarvis, Bernstein & Jain (2004, MNRAS, 352).
        See their section 3, especially equations 51 and 52 for the :math:`T_i` functions,
        equations 60 and 61 for the calculation of :math:`\langle \cal M^3 \rangle` and
        :math:`\langle \cal M^2 M^* \rangle`, and equations 55-58 for how to convert
        these to the return values.

        If k2 or k3 != 1, then this routine calculates the generalization of the skewness
        proposed by Schneider, Kilbinger & Lombardi (2005, A&A, 431):
        :math:`\langle M_{ap}^3(R, k_2 R, k_3 R)\rangle` and related values.

        If k2 = k3 = 1 (the default), then there are only 4 combinations of Map and Mx
        that are relevant:

        - map3 = :math:`\langle M_{ap}^3(R)\rangle`
        - map2mx = :math:`\langle M_{ap}^2(R) M_\times(R)\rangle`,
        - mapmx2 = :math:`\langle M_{ap}(R) M_\times(R)\rangle`
        - mx3 = :math:`\langle M_{\rm \times}^3(R)\rangle`

        However, if k2 or k3 != 1, then there are 8 combinations:

        - map3 = :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
        - mapmapmx = :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
        - mapmxmap = :math:`\langle M_{ap}(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmapmap = :math:`\langle M_\times(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmxmap = :math:`\langle M_\times(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmapmx = :math:`\langle M_\times(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
        - mapmxmx = :math:`\langle M_{ap}(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
        - mx3 = :math:`\langle M_\times(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`

        To accommodate this full generality, we always return all 8 values, along with the
        estimated variance (which is equal for each), even when k2 = k3 = 1.

        .. note::

            The formulae for the ``m2_uform`` = 'Schneider' definition of the aperture mass,
            described in the documentation of `calculateMapSq`, are not known, so that is not an
            option here.  The calculations here use the definition that corresponds to
            ``m2_uform`` = 'Crittenden'.

        Parameters:
            R (array):      The R values at which to calculate the aperture mass statistics.
                            (default: None, which means use self.rnom1d)
            k2 (float):     If given, the ratio R2/R1 in the SKL formulae. (default: 1)
            k3 (float):     If given, the ratio R3/R1 in the SKL formulae. (default: 1)

        Returns:
            Tuple containing:

            - map3 = array of :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
            - mapmapmx = array of :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
            - mapmxmap = array of :math:`\langle M_{ap}(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmapmap = array of :math:`\langle M_\times(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmxmap = array of :math:`\langle M_\times(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmapmx = array of :math:`\langle M_\times(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
            - mapmxmx = array of :math:`\langle M_{ap}(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
            - mx3 = array of :math:`\langle M_\times(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
            - varmap3 = array of variance estimates of the above values
        """
        # As in the calculateMapSq function, we Make s and t matrices, so we can eventually do the
        # integral by doing a matrix product.
        R = np.asarray(R)

        # Get Meshed Bins.
        t1m, t2m, phim = self.get_bins(bin_type='SAS', t1_unit=unit, t1_rep='mid', mesh=True)

        # Pick s = d2, so dlogs is bin_size
        s = d2 = np.outer(1./R, t2m.ravel())

        # d3
        d3 = np.outer(1./R, t1m.ravel())
        t = d3 * np.exp(1j * phim.ravel())

        # Next we need to construct the T values.
        T0, T1, T2, T3 = self._calculateT(s,t,1.,k2,k3)

        # integration measure
        bin_size = np.log(self.t1[1]/self.t1[0])
        phi_bin_size = self.phi[1] - self.phi[0]
        d2t = d3**2 * bin_size * phi_bin_size / (2*np.pi)
        
        # integration measure
        sds = s * s * bin_size

        # Note: these are really d2t/2piR^2 and sds/R^2, which are what actually show up
        # in JBJ equations 45 and 50.

        T0 *= sds * d2t
        T1 *= sds * d2t
        T2 *= sds * d2t
        T3 *= sds * d2t

        # Now do the integral by taking the matrix products.
        gam0 =-self.Gamma0.ravel()
        gam1 =-self.Gamma1.ravel()
        gam2 =-self.Gamma3.ravel()
        gam3 =-self.Gamma2.ravel()
        mmm = T0.dot(gam0)
        mcmm = T1.dot(gam1)
        mmcm = T2.dot(gam2)
        mmmc = T3.dot(gam3)

        # SAS binning counts each triangle with each vertex in the c1 position.
        # Just need to account for the cases where 1-2-3 are clockwise, rather than CCW.
        if k2 == 1 and k3 == 1:
            mmm *= 2
            mcmm *= 2
            mmcm += mmmc
            mmmc = mmcm
        else:
            # Repeat the above with 2,3 swapped.
            T0, T1, T2, T3 = self._calculateT(s,t,1,k3,k2)
            T0 *= sds * d2t
            T1 *= sds * d2t
            T2 *= sds * d2t
            T3 *= sds * d2t
            mmm += T0.dot(gam0)
            mcmm += T1.dot(gam1)
            mmmc += T2.dot(gam2)
            mmcm += T3.dot(gam3)

        map3 = 0.25 * np.real(mcmm + mmcm + mmmc + mmm)
        mapmapmx = 0.25 * np.imag(mcmm + mmcm - mmmc + mmm)
        mapmxmap = 0.25 * np.imag(mcmm - mmcm + mmmc + mmm)
        mxmapmap = 0.25 * np.imag(-mcmm + mmcm + mmmc + mmm)
        mxmxmap = 0.25 * np.real(mcmm + mmcm - mmmc - mmm)
        mxmapmx = 0.25 * np.real(mcmm - mmcm + mmmc - mmm)
        mapmxmx = 0.25 * np.real(-mcmm + mmcm + mmmc - mmm)
        mx3 = 0.25 * np.imag(mcmm + mmcm + mmmc - mmm)

        return map3, mapmapmx, mapmxmap, mxmapmap, mxmxmap, mxmapmx, mapmxmx, mx3

# phase factors to convert between different projections
def x2ortho(mu, t1, t2, phi):
    """
    Convert x-projection to orthocenter-projection.

    Parameters:
        mu (int)    : The index of the natural component.
        t1 (array)  : The value of t1.
        t2 (array)  : The value of t2.
        phi (array) : The value of phi.
    """
    # Compute prefactor
    sin2pb, cos2pb = sincos2angbar(np.arctan2(t2, t1), np.pi-phi)
    if mu==0 or mu==1 or mu==2:
        out = cos2pb - 1j*sin2pb
    elif mu==3:
        out = cos2pb + 1j*sin2pb
    return out

def ortho2cent(mu, t1, t2, phi):
    """
    Convert orthocenter-projection to centroid-projection.

    Parameters:
        mu (int)    : The index of the natural component.
        t1 (array)  : The value of t1.
        t2 (array)  : The value of t2.
        phi (array) : The value of phi.
    """
    NotImplementedError('This function is not validated yet')
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
        if j == mu:
            out *= phase
        else:
            out *= np.conj(phase)

    return out

def x2cent(mu, t1, t2, phi):
    """
    Convert x-projection to centroid-projection.

    Parameters:
        mu (int)    : The index of the natural component.
        t1 (array)  : The value of t1.
        t2 (array)  : The value of t2.
        phi (array) : The value of phi.

    References:
        Equations between Eq. (15) and (16) 
        of https://arxiv.org/abs/2309.08601
    """
    v = t1+t2*np.exp(-1j*phi)
    q1 = v/np.conj(v)
    v = -2*t1+t2*np.exp(-1j*phi)
    q2 = v/np.conj(v)
    v = t1-2*t2*np.exp(-1j*phi)
    q3 = v/np.conj(v)

    if mu==0:
        out = q1*q2*q3 * np.exp(3j*phi)
    elif mu==1:
        out = np.conj(q1)*q2*q3 * np.exp(1j*phi)
    elif mu==2:
        out = q1*np.conj(q2)*q3 * np.exp(3j*phi)
    elif mu==3:
        out = q1*q2*np.conj(q3) * np.exp(-1j*phi)
    else:
        raise ValueError('Error: mu={} is not expected'.format(mu))

    return out

def tune_fft_grid_size(lmin, lmax, nfft_min, dt, log=False):
    """
    This function tunes the size of FFT grid so that
    the bin size to be an integer multiple of dt, 
    and also number of bins to be greater than nfft_min.

    Parameters:
        lmin    (float): The minimum value of t.
        lmax    (float): The maximum value of t.
        nfft_min(int)  : The minimum number of FFT grid.
        dt      (float): The bin width of t, on which 
                        we want to evaluate the result

    Derivation:
        We parametrize the FFT bin size as
            dl = dt/nskip
        where nskip is an integer to be tuned
        Using this, the number of FFT bins is given by
            nfft = (lmax-lmin)/dl + 1
        The requirement on nfft is
            nfft >= nfft_min
        Also, nfft must be an integer, so by defining
            (lmax-lmin)/dl+1 = nfft + ep
        where nfft is the integer part of the lhs and 
        ep is the fractional part, we require
            0 <= ep < 1
        After tuning nskip, we also need to tune the range as
            lmax = lmax - dl*ep/2
            lmin = lmin + dl*ep/2

        The nskip can be found by
            nskip = ceil( (nfft_min-1)*dt/(lmax-lmin) )
        Once nskip is chosen, the other parameters can be derived
        as follows
            dl = dt/nskip
            nfft = floor( (lmax-lmin)/dl+1 )
            ep = (lmax-lmin)/dl - nfft
            lmax = lmax - dl*ep/2
            lmin = lmin + dl*ep/2

        With this output you can make FFT grid by
            np.linspace(lmin, lmax, nfft)
    """
    nskip = np.ceil( (nfft_min-1)*dt/(lmax-lmin) ).astype(int)
    dl = dt/nskip
    nfft = np.floor( (lmax-lmin)/dl+1 ).astype(int)
    ep = (lmax-lmin)/dl+1 - nfft
    lmax = lmax - dl*ep/2
    lmin = lmin + dl*ep/2
    if nfft%2 != 0:
        nfft -= 1
        lmax -= dl/2
        lmin += dl/2
    return lmin, lmax, nfft, nskip

def tune_fft_real_bin(l, t_pivot):
    """
    This tunes the position of fft grid in real
    space so that we can easily get the desired 
    bin `t` just by downsampling the t_fft.

    Parameters:
        l       (array): The fft grid in fourier space.
        t_pivot (float): The pivot value of t.
    
    Derivation:
        The fft grid in fourier and real space is 
        expressed by
            l_i = l_0 + i dl
            t_i = t_0 + i dt
        where dl = dt and i=0,1,...,N-1. 
        We can relate them as
            t_0(p)  = p - l_{N-1}
                    = p - l_max
        This is equivalent with
            t_i(p)  = p - l_{N-1} + idt
                    = p - l_{N-1-i}
        
        Consider we want one of the real-space
        fft grid to fall on t_pivot by nicely choosing p.
        First, we consider p=0, and where in fft grid
        t_pivot falls on. Because p is not tuned, there
        is no garantee that t_pivot falls on the grid.
        We denote this residual in grid index by ep.
            t_pivot = t_(i+ep)(0)
                    = -l_max + (i+ep)dt
        Then this ep is given as the fractional part of
            ep = frac[ (t_pivot+l_max)/dt ]
        Using this ep value, we can then chose p value
            t_pivot = ep*dt - l_max + i*dt
            p = ep*dt
    """
    dl = l[1]-l[0]
    ep = np.modf( (t_pivot+l[-1])/dl )[0]
    p = ep*dl
    return p

def get_tuned_fftgrid(lmin, lmax, nfft_min, t):
    """
    This function returns the tuned FFT grid in
    fourier space and the corresponding real-space
    grid index.

    Parameters:
        lmin    (float): The minimum value of t.
        lmax    (float): The maximum value of t.
        nfft_min(int)  : The minimum number of FFT grid.
        t       (float): The pivot value of t.
    """
    dt = np.diff(t)[0]
    lmin, lmax, nfft, nskip = tune_fft_grid_size(lmin, lmax, nfft_min, dt)
    l_fft = np.linspace(lmin, lmax, nfft)
    p = tune_fft_real_bin(l_fft, t[0])
    t_fft = p-l_fft[::-1]
    start = np.argmin(np.abs(t_fft-t[0]))
    sel = start + nskip*np.arange(t.size)
    return l_fft, t_fft, sel, p
