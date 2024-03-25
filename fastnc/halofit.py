#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/03/25 17:44:04

Description:
halofit.py contains the Halofit class. 
See the references below:
https://arxiv.org/abs/1208.2701
https://arxiv.org/abs/1911.07886
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps
from scipy.optimize import bisect
from scipy.special import gamma, gammaincc

class Halofit:
    """
    Halofit class including power and bi spectrum.
    Fitting formula of the nonlinear matter power spectrum 
    based on the halo model. 

    Parameters:
        k     (np.ndarray): array of comoving Fourier modes
        pklin (np.ndarray): array of linear power spectrum at the wave numbers
        z     (np.ndarray): array of redshifts
        lgr   (np.ndarray): linear growth rate at the redshifts

    References:
        See https://arxiv.org/abs/1208.2701 for halofit model.
        See https://arxiv.org/abs/1911.07886 for bihalofit model.
    """
    def __init__(self, k=None, pklin=None, z=None, lgr=None, cosmo=None):
        self.set_lgr(z, lgr)
        self.set_pklin(k, pklin)
        self.set_cosmology(cosmo)

    def set_lgr(self, z, lgr):
        """
        Set the linear growth rate at the redshifts.

        Parameters:
            z     (np.ndarray): array of redshifts
            lgr   (np.ndarray): linear growth rate at the redshifts
        """
        self.z     = z
        self.lgr   = lgr
        self.has_changed = True
    
    def set_pklin(self, k, pklin):
        """
        Set the linear power spectrum at the wave numbers.

        Parameters:
            k     (np.ndarray): array of comoving Fourier modes
            pklin (np.ndarray): array of linear power spectrum at the wave numbers
        """
        self.k     = k
        self.pklin = pklin
        self._normalize_pklin()
        self.has_changed = True

    def set_cosmology(self, cosmo):
        """
        Set the cosmological parameters.

        Parameters:
            cosmo (dict): dictionary of cosmological parameters

        cosmo needs to have the following keys:
            Om0   : matter density parameter at z=0
            Ode0  : dark energy density parameter at z=0
            ns    : spectral index of linear matter power spectrum
            w0    : w0 parameter of dark energy EoS
            sigma8: sigma8 at z=0
            (below is irrelevant to bihalofit)
            wa    : wa parameter of dark eneryg EoS
            fnu0  : fraction of neutrino energy density relative to matter at z=0
        """
        self.cosmo = cosmo
        self.has_changed = True

    def update(self):
        """
        Update the internal variables.
        """
        if self.has_changed:
            self._normalize_pklin()
            self._init_spline()
            self.has_changed = False
    
    def _normalize_pklin(self):
        """
        Normalize the linear power spectrum with sigma8.
        """
        if (self.k is not None) and (self.pklin is not None) and (self.cosmo is not None):
            # compute sigma8 with current pklin
            k     = np.logspace(-3, 2, 1000)
            Delta = self.get_interpolated_pklin(k)*k**3 / 2/np.pi**2
            sigma8temp = self._sigmam(k, Delta, 8.0, window_tophat)
            self.pklin *= (self.cosmo['sigma8']/sigma8temp)**2

    def get_interpolated_pklin(self, k, z=None, ext=True):
        """
        Interpolate the power spectrum for a given redshift.

        Parameters:
            k   (np.ndarray): array of comoving Fourier modes
            z   (float)     : redshift
            ext (bool)      : if True, extrapolate the power spectrum for higher k
        """
        # array to store log(pk)
        pk = np.zeros_like(k)

        # do interpolation
        where = (self.k.min() <= k) & (k <= self.k.max())
        logpk = ius(np.log(self.k), np.log(self.pklin))(np.log(k[where]))
        pk[where] = np.exp(logpk)

        if ext:
            # extrapolate by power law for lower k
            where = k < self.k.min()
            n = np.log(self.pklin[1]/self.pklin[0]) / np.log(self.k[1]/self.k[0])
            a = self.pklin[0]
            pk[where] = a*(k[where]/self.k[0])**n

            # extrapolate by power law for higher k
            where = k > self.k.max()
            n = np.log(self.pklin[-1]/self.pklin[-2]) / np.log(self.k[-1]/self.k[-2])
            a = self.pklin[-1]
            pk[where] = a*(k[where]/self.k[-1])**n
        
        # multiply the linear growth rate
        # to the linear power at z = 0.0
        if z is not None:
            pk *= ius(self.z, self.lgr, ext=2)(z)**2

        return pk

    def _sigmam(self, k, Delta, r, window, extrap=False):
        """
        Calculate sigmaM.

        Parameters:
            k (np.ndarray)    : array of comoving Fourier modes
            Delta (np.ndarray): array of Delta^2(k)
            r (float)         : smoothing scale
            window (function) : window function
            extrap (bool)     : if True, extrapolate the power spectrum for higher k

        Returns:
            sigmam (float): sigmaM

        Note:
            The integral is calculated by the Simpson's rule.
            If extrap is True, the power spectrum is extrapolated by a power law: A k^n.

            And the integral is calculated as follows:

            ..math::
                sigma_m(R)  = \\int_-inf^inf dlnk Delta(k) W^2(kR)
                            = \\int_-inf^kmax dlnk Delta(k) W^2(kR) + \\int_kmax^inf dlnk Delta(k) W^2(kR)
                            ~ sum_i Delta_i W^2(k_i R) dlnk + \\int_kmax^inf dlnk Delta(k) W^2(kR)

            Assuming the Delta at high k can be approximated by a power law, ~ A k^n, and Gaussian window,

            ..math::
                sigma_m(R)  = I2 + \\int_kmax^inf dlnk Ak^n e^(-k^2R^2)
                            = I2 + A R^(-n) Gamma(n/2) Gamma^reg(n/2, tmin)

            where tmin = kmax^2R^2, and Gamma^reg is the regularized gamma function.
        """
        I2    = simps(Delta * window(k*r)**2, np.log(k))

        if extrap:
            # spectral index and amplitude of Delta at high k
            n = np.diff(np.log(Delta))[-1]/np.diff(np.log(k))[-1]
            A = Delta[-1] * k[-1]**(-n)
            tmin = k[-1] * r
            I2 += A * r**-n * 0.5*gamma(n/2) * gammaincc(n/2, tmin**2)
            
        return I2**0.5
        
    def get_r_sigma(self, z, rtol=1e-5, maxiter=10):
        """
        Returns nonlinear scale, r_sigma.

        Parameters:
            z       (float): redshift
            rtol    (float): relative tolerance
            maxiter (int)  : maximum number of iterations
        """
        k     = np.logspace(-3, 2, 1000)
        Delta = self.get_interpolated_pklin(k, z)*k**3 / 2/np.pi**2
        def eq(R): return self._sigmam(k, Delta, R, window_gaussian, extrap=True) - 1.0
        # initial guess
        rmin, rmax = 1.0/k.max(), 1.0/k.min()
        niter = 0
        while eq(rmin)*eq(rmax) > 0 and niter <= maxiter:
            rmin /= 2.0
            niter += 1
        if niter > maxiter:
            r_sigma = rmin
        else:
            r_sigma = abs(bisect(eq, rmin, rmax, rtol=rtol))
        return r_sigma
    
    # ingredients
    def get_neff(self, z, r_sigma=None):
        """
        Returns effective spectral index.

        Parameters:
            z       (float): redshift
            r_sigma (float): smoothing scale
        """
        if r_sigma is None:
            r_sigma = self.get_r_sigma(z)
        # Get effecitive spectrum index: neff
        k       = np.logspace(-4, 2, 2000)
        Delta   = self.get_interpolated_pklin(k, z)*k**3 / 2/np.pi**2
        sigmaG1 = self._sigmam(k, Delta, r_sigma, window_gaussian_1deriv)
        neff    = -3.0 + 2.0*sigmaG1**2
        return neff
    
    def get_C(self, z, r_sigma=None):
        """
        Returns spectral curvature.

        Parameters:
            z       (float): redshift
            r_sigma (float): smoothing scale
        """
        if r_sigma is None:
            r_sigma = self.get_r_sigma(z)
        # Get effecitive spectrum index: neff
        k       = np.logspace(-4, 2, 2000)
        Delta   = self.get_interpolated_pklin(k, z)*k**3 / 2/np.pi**2
        sigmaG1 = self._sigmam(k, Delta, r_sigma, window_gaussian_1deriv)
        sigmaG2 = self._sigmam(k, Delta, r_sigma, window_gaussian_2deriv)
        C       = 4*sigmaG1**4 + 4*sigmaG1**2 - 4*sigmaG2**2
        return C
    
    def get_sigma8z(self, z):
        """
        Returns sigma8 at a given redshift z.

        Parameters:
            z (float): redshift
        """
        k       = np.logspace(-4, 2, 1000)
        Delta   = self.get_interpolated_pklin(k, z)*k**3 / 2/np.pi**2
        sigma8z = self._sigmam(k, Delta, 8.0, window_tophat)
        return sigma8z
    
    def get_Omz(self, z):
        """
        Returns Omega_m(z).

        Parameters:
            z (float): redshift
        """
        Om0, Ode0, w0, wa = self.cosmo['Om0'], self.cosmo['Ode0'], self.cosmo['w0'], self.cosmo['wa']
        a = 1.0/(1+z)
        Qa2 = a**(-1.0-3.0*(w0+wa))*np.exp(-3.0*(1-a)*wa)
        Omt =1.0+(Om0+Ode0-1.0)/(1-Om0-Ode0+Ode0*Qa2+Om0/a)
        Omz =Omt*Om0/(Om0+Ode0*a*Qa2)
        return Omz
    
    def get_Odez(self, z):
        """
        Returns Omega_de(z).

        Parameters:
            z (float): redshift
        """
        Om0, Ode0, w0, wa = self.cosmo['Om0'], self.cosmo['Ode0'], self.cosmo['w0'], self.cosmo['wa']
        a = 1.0/(1+z)
        Qa2 = a**(-1.0-3.0*(w0+wa))*np.exp(-3.0*(1-a)*wa)
        Ot =1.0+(Om0+Ode0-1.0)/(1-Om0-Ode0+Ode0*Qa2+Om0/a)
        Ode=Ot*Ode0*Qa2/(Ode0*Qa2+Om0/a)
        return Ode
    
    def _init_spline(self, zmid=0.5, dz_low=0.15, dz_high=0.3):
        """
        Initialize the spline for halofit.

        Parameters:
            zmid    (float): the middle redshift for the spline
            dz_low  (float): the lower redshift interval for the spline
            dz_high (float): the higher redshift interval for the spline
        """
        # For acculate calculation, we divide z array into two sections
        # Defining zmid, the first section is [0, zmid) and the second is [zmid, max),
        # where we use finer bin for the first than the second.
        # The default zmid, dz_low, dz_high is optimized to have 0.05% accuracy 
        # only with ~100msec CPU time.
        if zmid <= self.z.min():
            z = np.arange(self.z.min(), self.z.max(), dz_high)
        elif self.z.max() <= zmid:
            z = np.arange(self.z.min(), self.z.max(), dz_low)
        else:
            z = np.hstack([np.arange(self.z.min(), zmid, dz_low), 
                           np.arange(zmid, self.z.max(), dz_high)])
        if not self.z.max() in z:
            z = np.append(z, self.z.max())
        self.lazy_arrays = {'z':z}
        
        names = ['r_sigma', 'neff', 'C', 'sigma8z', 'Omz', 'Odez']
        for name in names:
            self.lazy_arrays[name] = np.empty(z.shape)
        for i, _z in enumerate(z):
            self.lazy_arrays['r_sigma'][i] = self.get_r_sigma(_z)
            self.lazy_arrays['neff'][i]    = self.get_neff(_z, self.lazy_arrays['r_sigma'][i])
            self.lazy_arrays['C'][i]       = self.get_C(_z, self.lazy_arrays['r_sigma'][i])
            self.lazy_arrays['sigma8z'][i] = self.get_sigma8z(_z)
        self.lazy_arrays['Omz']     = self.get_Omz(z)
        self.lazy_arrays['Odez']    = self.get_Odez(z)
    
    def get_halofit_coeffs(self, z):
        """
        Returns the coefficients of halofit.

        Parameters:
            z (np.ndarray): array of redshifts
        """
        r_sigma = ius(self.lazy_arrays['z'], self.lazy_arrays['r_sigma'])(z)
        neff    = ius(self.lazy_arrays['z'], self.lazy_arrays['neff'])(z)
        C       = ius(self.lazy_arrays['z'], self.lazy_arrays['C'])(z)
        Odez    = ius(self.lazy_arrays['z'], self.lazy_arrays['Odez'])(z)
        Omz     = ius(self.lazy_arrays['z'], self.lazy_arrays['Omz'])(z)
        w       = self.cosmo['w0'] + self.cosmo['wa']*z/(z+1)
        fnu0    = self.cosmo['fnu0']
        
        an = 10.**(1.5222 + 2.8553*neff + 2.3706*neff**2 + 0.9903*neff**3 + 0.2250*neff**4 \
                    - 0.6038*C + 0.1749*Odez*(1+w))
        bn = 10.**(-0.5642 + 0.5864*neff + 0.5716*neff**2 - 1.5474*C + 0.2279*Odez*(1+w))
        cn = 10.**(0.3698 + 2.0404*neff + 0.8161*neff**2 + 0.5869*C)

        gan = 0.1971 - 0.0843*neff + 0.8460*C
        aln = np.abs( 6.0835 + 1.3373*neff - 0.1959*neff**2 - 5.5274*C )
        ben = 2.0379 - 0.7354*neff + 0.3157*neff**2 + 1.2490*neff**3 + 0.3980*neff**4 \
                    - 0.1682*C + fnu0*(1.081 + 0.395*neff**2)
        mun = np.zeros(z.shape)
        nun = 10**(5.2105 + 3.6902*neff)
        
        f1b, f2b, f3b = Omz**-0.0307, Omz**-0.0585, Omz**0.0743
        f1a, f2a, f3a = Omz**-0.0732, Omz**-0.1423, Omz**0.0725
        frac = Odez/(1-Omz)
        f1, f2, f3 = frac*f1b+(1-frac)*f1a, frac*f2b+(1-frac)*f2a, frac*f3b+(1-frac)*f3a
        
        coeffs = dict()
        coeffs['r_sigma'] = r_sigma
        coeffs['an']      = an
        coeffs['bn']      = bn
        coeffs['cn']      = cn
        coeffs['gan']     = gan
        coeffs['aln']     = aln
        coeffs['ben']     = ben
        coeffs['mun']     = mun
        coeffs['nun']     = nun
        coeffs['f1']      = f1
        coeffs['f2']      = f2
        coeffs['f3']      = f3
        
        return coeffs
        
    def get_pkhalofit(self, k, z):
        """
        Returns the halofit prediction of nonlinear matter power spectrum.

        Parameters:
            k (np.ndarray): array of comoving Fourier modes
            z (np.ndarray): array of redshifts
        """
        # update the internal variables
        self.update()

        z = np.asarray(z)
        c = self.get_halofit_coeffs(z)
        fnu0 = self.cosmo['fnu0']
        Om0  = self.cosmo['Om0']
        
        pkhalo = []
        for i, _z in enumerate(z):
            pklin = self.get_interpolated_pklin(k, _z)
            DeltaL= pklin * k**3/2/np.pi**2
            
            y     = k * c['r_sigma'][i]
            f     = y/4. + y**2/8.
            
            DeltaLaa = DeltaL*(1.+fnu0*47.48*k**2/(1.+1.5*k**2))
            DeltaQ   = DeltaL*((1.+DeltaLaa)**c['ben'][i])/(1.+c['aln'][i]*DeltaLaa) * np.exp(-f)
            DeltaH   = c['an'][i]*y**(3.*c['f1'][i]) \
                        / (1.+c['bn'][i]*y**c['f2'][i] + (c['cn'][i]*y*c['f3'][i])**(3.-c['gan'][i]))
            DeltaH   = DeltaH / (1. + c['mun'][i]/y + c['nun'][i]/y**2) \
                        * (1+fnu0*(0.977-18.015*(Om0-0.3)))
            pkhalo.append( (DeltaQ + DeltaH) * (2.*np.pi**2) / k**3 )
        pkhalo = np.array(pkhalo)
    
        return pkhalo
    
    def get_bihalofit_coeffs(self, z):
        """
        Returns the coefficients of bihalofit, 
        or parts of coefficients which are independent of triangle.

        Parameters:
            z (np.ndarray): array of redshifts
        """
        r_sigma = ius(self.lazy_arrays['z'], self.lazy_arrays['r_sigma'])(z)
        neff    = ius(self.lazy_arrays['z'], self.lazy_arrays['neff'])(z)
        Omz     = ius(self.lazy_arrays['z'], self.lazy_arrays['Omz'])(z)
        sigma8z = ius(self.lazy_arrays['z'], self.lazy_arrays['sigma8z'])(z)
        log10sigma8z = np.log10(sigma8z)
        
        # initialize the array
        coeffs = np.zeros(z.shape,dtype=[('r_sigma', 'f8'), 
                                         ('log10an1', 'f8'),
                                         ('log10an2', 'f8'),
                                         ('bn', 'f8'),
                                         ('cn', 'f8'),
                                         ('log10aln1', 'f8'),
                                         ('log10aln2', 'f8'),
                                         ('log10ben1', 'f8'),
                                         ('log10ben2', 'f8'),
                                         ('gan', 'f8'),
                                         ('fn', 'f8'),
                                         ('gn', 'f8'),
                                         ('hn', 'f8'),
                                         ('mn', 'f8'),
                                         ('nn', 'f8'),
                                         ('mun', 'f8'),
                                         ('nun', 'f8'),
                                         ('pn', 'f8'),
                                         ('en', 'f8'),
                                         ('dn', 'f8')])
        coeffs['r_sigma']   = r_sigma
        
        # 1-halo term coefficients
        ## a part of an, add -0.310*r1^gan later
        coeffs['log10an1'] = -2.167-2.944*log10sigma8z-1.106*log10sigma8z**2-2.865*log10sigma8z**3
        coeffs['log10an2'] = -0.310*np.ones(z.shape)
        coeffs['bn'] = 10.**(-3.428-2.681*log10sigma8z+1.624*log10sigma8z**2-0.095*log10sigma8z**3)
        coeffs['cn'] = 10.**(0.159-1.107*neff)
        ## parts of alphan to combined with r2 later
        coeffs['log10aln1'] = -4.348-3.006*neff-0.5745*neff**2
        coeffs['log10aln2'] = 10**(-0.9+0.2*neff) 
        ## parts of betan to be combined with r2 later
        coeffs['log10ben1'] = -1.731-2.845*neff-1.4995*neff**2-0.2811*neff**3
        coeffs['log10ben2'] = 0.007*np.ones(z.shape)
        # gan used in an
        coeffs['gan']  = 10**(0.182+0.57*neff)
        
        # 3-halo term coefficients
        coeffs['fn'] = 10**(-10.533-16.838*neff-9.3048*neff**2-1.8263*neff**3)
        coeffs['gn'] = 10**(2.787+2.405*neff+0.4577*neff**2)
        coeffs['hn'] = 10**(-1.118-0.394*neff)
        coeffs['mn'] = 10**(-2.605-2.434*log10sigma8z+5.71*log10sigma8z**2)
        coeffs['nn'] = 10**(-4.468-3.08*log10sigma8z+1.035*log10sigma8z**2)
        coeffs['mun']= 10**(15.312+22.977*neff+10.9579*neff**2+1.6586*neff**3)
        coeffs['nun']= 10**(1.347+1.246*neff+0.4525*neff**2)
        coeffs['pn'] = 10**(0.071-0.433*neff)
        coeffs['en'] = 10**(-0.632+0.646*neff)
        coeffs['dn'] = 10**(-0.483+0.892*log10sigma8z-0.086*Omz)
        
        return coeffs
        
    def F2_tree(self, k1, k2, k3):
        """
        Returns the tree level bispectrum.

        Parameters:
            k1 (np.ndarray): array of comoving Fourier modes in h/Mpc unit
            k2 (np.ndarray): array of comoving Fourier modes in h/Mpc unit
            k3 (np.ndarray): array of comoving Fourier modes in h/Mpc unit
        """
        costheta12=0.5*(k3*k3-k1*k1-k2*k2)/(k1*k2)
        return (5./7.)+0.5*costheta12*(k1/k2+k2/k1)+(2./7.)*costheta12*costheta12
        
    def get_bihalofit(self, k1, k2, k3, z, verbose=False, all_physical=False, which=['Bh3', 'Bh1']):
        """
        Returns the bihalofit prediction of matter bispectrum.

        Parameters:
            k1           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
            k2           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
            k3           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
            z            (np.ndarray): array of redshifts
            verbose      (bool)      : if True, print the progress
            all_physical (bool)      : if True, assumes all the triangle satisfy the 
                                       triangle condition
            which        (str or list): which part of bispectrum to calculate
        """
        if isinstance(which, str):
            which = [which]

        for term in which:
            if term not in ['BT', 'Bh1', 'Bh3']:
                raise ValueError('which arg should be one of BT, Bh1, Bh3 or a list of them.')

        # update the internal variables
        self.update()
        
        # N
        N = z.shape
        
        # ratio of modes
        kmin, kmid, kmax = np.sort([k1,k2,k3], axis=0)
        r1, r2 = kmin/kmax, (kmid+kmin-kmax)/kmax
        
        # coefficients
        c  = self.get_bihalofit_coeffs(z)
        ns = self.cosmo['ns']
        
        # select physical modes
        sel = kmax <= kmin+kmid
        if all_physical:
            sel[:] = True
        z = z[sel]
        k1, k2, k3 = k1[sel], k2[sel], k3[sel]
        r1, r2 = r1[sel], r2[sel]
        c = c[sel]
        
        # dimensionless Fourier modes
        q1, q2, q3 = k1*c['r_sigma'], k2*c['r_sigma'], k3*c['r_sigma']
        
        # linear power spectrum
        if 'BT' in which or 'Bh3' in which:
            PL1 = self.get_interpolated_pklin(k1, z)
            PL2 = self.get_interpolated_pklin(k2, z)
            PL3 = self.get_interpolated_pklin(k3, z)

        # total bispectrum
        Btot  = np.zeros(N)

        # tree level
        if 'BT' in which:
            BT  = np.zeros(N)
            BT[sel] = 2*self.F2_tree(k1,k2,k3)*PL1*PL2 \
                            + 2*self.F2_tree(k2,k3,k1)*PL2*PL3 \
                            + 2*self.F2_tree(k3,k1,k2)*PL3*PL1
            BT[np.logical_not(sel)] = np.nan
            Btot += BT
        
        # 1-halo term, Eq. (B4)
        if 'Bh1' in which:
            Bh1 = np.ones(N)
            for q in [q1, q2, q3]:
                # Combine parts of coefficients
                an = 10**(c['log10an1' ]+c['log10an2' ]*r1**c['gan'])
                aln= 10**(c['log10aln1']+c['log10aln2']*r2**2)
                aln[aln > 1.-(2./3.)*ns] = 1.-(2./3.)*ns
                bn = c['bn']
                ben= 10**(c['log10ben1']+c['log10ben2']*r2)
                cn = c['cn']
                # Multiply the result to form Bh1
                Bh1[sel] *= 1/(an*q**aln + bn*q**ben) / (1+1/cn/q)
            Bh1[np.logical_not(sel)] = np.nan
            Btot += Bh1
        
        # 3-halo term
        if 'Bh3' in which:
            PE = []
            for PL, q in [(PL1,q1), (PL2,q2), (PL3,q3)]:
                PE.append( (1+c['fn']*q**2)/(1+c['gn']*q+c['hn']*q**2)*PL \
                        + 1/(c['mn']*q**c['mun']+c['nn']*q**c['nun']) \
                        /(1+(c['pn']*q)**-3) )
            PE1, PE2, PE3 = PE
            Bh3 = np.zeros(N)
            Bh3[sel] = 2*(self.F2_tree(k1,k2,k3)+c['dn']*q3)*PE1*PE2 \
                            + 2*(self.F2_tree(k2,k3,k1)+c['dn']*q1)*PE2*PE3 \
                            + 2*(self.F2_tree(k3,k1,k2)+c['dn']*q2)*PE3*PE1
            Bh3[sel]*= 1./(1.+c['en']*q1)/(1.+c['en']*q2)/(1.+c['en']*q3)
            Bh3[np.logical_not(sel)] = np.nan
            Btot += Bh3
        
        return Btot
    
    def get_Rb_bihalofit(self, args):
        """
        Returns the baryon ratio on bispectrum.

        args: a structured array with key: k1, k2, k3 z
        """
        # ratio of modes
        kmin, kmid, kmax = np.sort([k1,k2,k3], axis=0)
        
        # Baryon effective redshift
        sel = kmax <= kmin + kmid
        
        # coefficients
        a = 1/(1+z[sel])
        A0  = np.zeros(a.size)
        A0[a>0.5] = 0.068*(a[a>0.5]-0.5)**0.47
        mu0 = 0.018*a + 0.837*a**2
        si0 = 0.881*mu0
        al0 = 2.346
        A1  = np.zeros(a.size)
        A1[a>0.2] = 1.052*(a[a>0.2]-0.2)**1.41
        mu1 = np.abs(0.172+3.048*a-0.675*a**2)
        si1 = (0.494-0.039*a)*mu1
        kst = 29.90 - 38.73*a+24.30*a**2
        al2 = 2.25
        be2 = 0.563/((a/0.06)**0.02+1) / al2
        
        # Assign baryon ratios
        Rb = np.ones(z.size)
        for k in [k1[sel], k2[sel], k3[sel]]:
            x = np.log10(k)
            Rb[sel]*= A0*np.exp(-np.abs((x-mu0)/si0)**al0) \
                        - A1*np.exp(-((x-mu1)/si1)**2) \
                        + ((k/kst)**al2+1)**be2
        
        Rb[z[sel]>5] = 1.0
        Rb[np.logical_not(sel)] = np.nan
        
        return rb
    
def window_tophat(x):
    """
    Top-hat window function.
    """
    return 3.0/x**3 * (np.sin(x) - x*np.cos(x))

def window_gaussian(x):
    """
    Gaussian window function.
    """
    return np.exp(-0.5*x**2)

def window_gaussian_1deriv(x):
    """
    First derivative of Gaussian window function.
    """
    return x*np.exp(-0.5*x**2)

def window_gaussian_2deriv(x):
    """
    Second derivative of Gaussian window function.
    """
    return x**2*np.exp(-0.5*x**2)