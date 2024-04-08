#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/04/08 14:02:10

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
    config_multipole = {'Lmax':None, 'Mmax':None, 'multipole_type':'legendre'}
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
        self.Mmax = self.config_multipole['Mmax']
        self.multipole_type = self.config_multipole['multipole_type']
        if self.multipole_type == 'legendre':
            self.GLM = MCF222LegendreFourier(self.Lmax, self.Mmax, verbose=self.verbose, use_cache=use_cache)
        elif self.multipole_type == 'fourier':
            self.GLM = MCF222FourierFourier(self.Lmax, self.Mmax, verbose=self.verbose, use_cache=use_cache)
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
            self.t1 = self.t1_fft = self.config_fftlog['xy']/self.ell1_fft
            self.down_sampler = np.arange(nfft) # no downsampling
        # Allocate the same bin to second side
        self.t2 = self.t1
        self.t2_fft = self.t1_fft
        self.ell2_fft = self.ell1_fft
        # 2dFFT grid in Fourier space
        self.ELL1_FFT, self.ELL2_FFT = np.meshgrid(self.ell1_fft, self.ell2_fft, indexing='ij')
        self.ELL_FFT  = np.sqrt(self.ELL1_FFT**2 + self.ELL2_FFT**2)
        self.PSI_FFT = np.arctan2(self.ELL2_FFT, self.ELL1_FFT)
    
    def HM(self, M, bL=None, **args):
        """
        Compute H_M(l1, l2 = \\sum_L (-1)^L * G_LM * b_L(l1, l2) on FFT grid.

        Parameters:
            M (int)   : The angular Fourier mode.
            bL (array): The bispectrum multipole. Defaults to None.
                        If None, it is computed using self.bispectrum.kappa_bispectrum_multipole.
                        By supplying bL, you can avoid recomputation of bL.
        """
        # Get bispectrum multipole indices, L array
        L = np.arange(self.Lmax+1)
        # Get bispectrum multipole
        if bL is None:
            bL = self.bispectrum.kappa_bispectrum_multipole(
                L, self.ELL_FFT, self.PSI_FFT, **args)
        # Sum up GLM*bL over L
        GLM = self.GLM(L, M, self.PSI_FFT)
        HM = np.sum(((-1)**(L+1)*GLM.T*bL.T).T, axis=0)
        return HM

    def get_bins(self, bin_type='multipole', t1_unit='radian', mesh=True):
        """
        Get the bins on which the theory is computed.

        Parameters:
            bin_type (str): The type of bin. Either 'multipole' or 'SAS'.
            t1_unit (str) : The unit of t1. Either 'radian' or 'arcmin'.
            mesh (bool)   : Whether to return meshgrid or not.
        
        Returns:
            bin1 (array): The first bin.
            bin2 (array): The second bin.
            bin3 (array): The third bin.
        """
        ufactor = 1.0 if t1_unit == 'radian' else 180.0*60.0/np.pi
        if bin_type == 'multipole':
            bin1 = self.t1 * ufactor
            bin2 = self.t2 * ufactor
            bin3 = np.arange(-self.Mmax, self.Mmax+1)
        elif bin_type == 'SAS':
            bin1 = self.t1 * ufactor
            bin2 = self.t2 * ufactor
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
        # natural-component multipole indices
        M = np.arange(self.Mmax+1)
        L = np.arange(self.Lmax+1)
        # First we compute the kernel HM for all M
        bL = self.bispectrum.kappa_bispectrum_multipole(
            L, self.ELL_FFT, self.PSI_FFT, **args)
        HM = [self.HM(_, bL=bL) for _ in M]

        # GammaM
        self.Gamma0M = np.zeros((2*self.Mmax+1, self.t1.size, self.t2.size))
        self.Gamma1M = np.zeros((2*self.Mmax+1, self.t1.size, self.t2.size))
        self.Gamma2M = np.zeros((2*self.Mmax+1, self.t1.size, self.t2.size))
        self.Gamma3M = np.zeros((2*self.Mmax+1, self.t1.size, self.t2.size))
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
                    self.Gamma0M[ _M+self.Mmax] = GM
                    self.Gamma0M[-_M+self.Mmax] = GM.T
                elif _mu == 1:
                    self.Gamma1M[ _M+self.Mmax] = GM.T
                    self.Gamma1M[-_M+self.Mmax] = GM
                elif _mu == 2:
                    self.Gamma2M[ _M+self.Mmax] = GM
                    self.Gamma3M[-_M+self.Mmax] = GM.T
                elif _mu == 3:
                    self.Gamma3M[ _M+self.Mmax] = GM
                    self.Gamma2M[-_M+self.Mmax] = GM.T

        if self.phi is None:
            return
        M = np.arange(-self.Mmax, self.Mmax+1)
        expMphi = np.exp(1j*M[:,None]*self.phi[None,:])
        # resum multipoles
        self.Gamma0 = np.tensordot(expMphi, self.Gamma0M, axes=([0],[0]))/(2*np.pi)
        self.Gamma1 = np.tensordot(expMphi, self.Gamma1M, axes=([0],[0]))/(2*np.pi)
        self.Gamma2 = np.tensordot(expMphi, self.Gamma2M, axes=([0],[0]))/(2*np.pi)
        self.Gamma3 = np.tensordot(expMphi, self.Gamma3M, axes=([0],[0]))/(2*np.pi)
        # change shear projection
        self._change_shear_projection('x', self.projection)
    
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
        PHI = self.phi[:,None,None]
        T1  = self.t1[None,:,None]
        T2  = self.t2[None,None,:]
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
