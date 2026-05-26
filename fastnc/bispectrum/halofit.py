#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2026/05/26 00:16:32

Description:
halofit.py contains the Halofit class. 
See the references below:
https://arxiv.org/abs/1208.2701
https://arxiv.org/abs/1911.07886
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simpson
from scipy.optimize import bisect
from scipy.special import gamma, gammaincc

try:
    from fastnc.hankel.wrapper import PowerLawFFTLogConfig, power_law_fftlog_coefficients
except Exception:  # pragma: no cover - fallback for standalone copies
    PowerLawFFTLogConfig = None
    power_law_fftlog_coefficients = None

try:
    from fastnc.hankel.radial import RadialFFTLogExpansion
except Exception:  # pragma: no cover - fallback for standalone copies
    RadialFFTLogExpansion = None

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
        self.set_cosmology(cosmo)
        self.set_pklin(k, pklin)

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
        I2    = simpson(Delta * window(k*r)**2, x=np.log(k))

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
        k     = np.logspace(-4, 2, 1000)
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
        
    def dln_pklin_dlnk(self, k, z=None, eps=1.0e-3):
        """Return ``d ln P_L(k,z) / d ln k`` using a symmetric log step."""
        self.update()

        k = np.asarray(k, dtype=float)
        if np.any(k <= 0.0):
            raise ValueError("k must be positive when evaluating dln_pklin_dlnk")

        kp = k * np.exp(eps)
        km = k * np.exp(-eps)
        pp = self.get_interpolated_pklin(kp, z)
        pm = self.get_interpolated_pklin(km, z)
        return (np.log(pp) - np.log(pm)) / (2.0 * eps)

    def cyclic_F2_sum_safe(
        self,
        k1, k2, k3,
        z,
        X1, X2, X3,
        dlnX_dlnk_func=None,
        eps_sq=1.0e-4,
    ):
        """Safely evaluate the cyclic ``F2`` weighted sum.

        This returns

            2 F2(k1,k2) X1 X2
          + 2 F2(k2,k3) X2 X3
          + 2 F2(k3,k1) X3 X1

        but switches to the squeezed-limit expression when
        ``q/k < eps_sq`` with ``q=min(k1,k2,k3)``.  The switch avoids
        direct evaluation of the individually divergent long-short ``F2``
        terms.  It is not a regulator; it is a numerically stable evaluation
        of the same leading squeezed limit.
        """
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k3 = np.asarray(k3, dtype=float)
        z = np.asarray(z, dtype=float)
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        X3 = np.asarray(X3, dtype=float)

        k1, k2, k3, z, X1, X2, X3 = np.broadcast_arrays(k1, k2, k3, z, X1, X2, X3)
        out = np.zeros_like(k1, dtype=float)

        ks = np.stack([k1, k2, k3], axis=0)
        Xs = np.stack([X1, X2, X3], axis=0)
        order = np.argsort(ks, axis=0)
        ks_sorted = np.take_along_axis(ks, order, axis=0)
        Xs_sorted = np.take_along_axis(Xs, order, axis=0)

        q = ks_sorted[0]
        ka = ks_sorted[1]
        kb = ks_sorted[2]
        Xq = Xs_sorted[0]
        Xa = Xs_sorted[1]
        Xb = Xs_sorted[2]

        k = 0.5 * (ka + kb)
        Xk = 0.5 * (Xa + Xb)

        # Floating-point tolerant triangle test and boundary clipping.
        physical, kb_safe = self._clip_triangle_boundary(q, ka, kb)
        
        # Use the clipped long side in squeezed diagnostics.
        k = 0.5 * (ka + kb_safe)
        
        positive_short = k > 0.0
        squeezed = physical & positive_short & (q / k < eps_sq)
        normal = physical & (~squeezed)
        
        out[~physical] = np.nan

        if np.any(normal):
            out[normal] = (
                2.0 * self.F2_tree(k1[normal], k2[normal], k3[normal]) * X1[normal] * X2[normal]
                + 2.0 * self.F2_tree(k2[normal], k3[normal], k1[normal]) * X2[normal] * X3[normal]
                + 2.0 * self.F2_tree(k3[normal], k1[normal], k2[normal]) * X3[normal] * X1[normal]
            )

        if np.any(squeezed):
            qs = q[squeezed]
            kas = ka[squeezed]
            kbs = kb[squeezed]
            kk = k[squeezed]
            zs = z[squeezed]

            mu2 = np.zeros_like(qs)
            nonzero = qs > 0.0
            mu2[nonzero] = ((kas[nonzero] - kbs[nonzero]) / qs[nonzero]) ** 2
            mu2 = np.clip(mu2, 0.0, 1.0)

            if dlnX_dlnk_func is None:
                n = self.dln_pklin_dlnk(kk, zs)
            else:
                n = dlnX_dlnk_func(kk, zs)

            Xq_safe = np.where(qs > 0.0, Xq[squeezed], 0.0)
            out[squeezed] = Xk[squeezed] * Xq_safe * (
                13.0 / 7.0
                + (8.0 / 7.0 - n) * mu2
            )

        return out

    def _triangle_physical_mask(self, kmin, kmid, kmax, rtol=None):
        if rtol is None:
            rtol = 64.0 * np.finfo(float).eps    
        scale = np.maximum(kmax, kmin + kmid)
        return kmax <= kmin + kmid + rtol * scale

    def _clip_triangle_boundary(self, kmin, kmid, kmax, rtol=None):
        physical = self._triangle_physical_mask(kmin, kmid, kmax, rtol=rtol)
        kmax_safe = np.where(physical, np.minimum(kmax, kmin + kmid), kmax)
        return physical, kmax_safe

    def get_bihalofit(self, k1, k2, k3, z, verbose=False, which=['Bh3', 'Bh1'], squeezed_safe=True, eps_sq=1.0e-4):
        """
        Returns the bihalofit prediction of matter bispectrum.

        Parameters:
            k1           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
            k2           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
            k3           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
            z            (np.ndarray): array of redshifts
            verbose      (bool)      : if True, print the progress
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
        z = np.broadcast_to(z, k1.shape)
        N = z.shape

        # physical triangle test, ratio of modes
        kmin, kmid, kmax = np.sort([k1, k2, k3], axis=0)
        physical, kmax_safe = self._clip_triangle_boundary(kmin, kmid, kmax)
        sel = physical
        
        den = np.maximum(kmax_safe, np.finfo(float).tiny)
        r1 = kmin / den
        r2 = (kmid + kmin - kmax_safe) / den
        r2 = np.maximum(r2, 0.0)
        
        # coefficients
        c  = self.get_bihalofit_coeffs(z)
        ns = self.cosmo['ns']
        
        z = z[sel]
        k1, k2, k3 = k1[sel], k2[sel], k3[sel]
        r1, r2 = r1[sel], r2[sel]
        c = c[sel]
        
        # dimensionless Fourier modes.  A tiny positive floor avoids 0*inf
        # indeterminacies in bihalofit fitting factors at exactly degenerate
        # triangle endpoints.  This is only an evaluation guard; it is far below
        # any practical model scale.
        q_floor = 1.0e-100
        q1 = np.maximum(k1*c['r_sigma'], q_floor)
        q2 = np.maximum(k2*c['r_sigma'], q_floor)
        q3 = np.maximum(k3*c['r_sigma'], q_floor)
        
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
            if squeezed_safe:
                BT[sel] = self.cyclic_F2_sum_safe(
                    k1, k2, k3, z,
                    PL1, PL2, PL3,
                    dlnX_dlnk_func=self.dln_pklin_dlnk,
                    eps_sq=eps_sq,
                )
            else:
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
            if squeezed_safe:
                Bh3_F2 = self.cyclic_F2_sum_safe(
                    k1, k2, k3, z,
                    PE1, PE2, PE3,
                    dlnX_dlnk_func=None,
                    eps_sq=eps_sq,
                )
            else:
                Bh3_F2 = 2*self.F2_tree(k1,k2,k3)*PE1*PE2 \
                                + 2*self.F2_tree(k2,k3,k1)*PE2*PE3 \
                                + 2*self.F2_tree(k3,k1,k2)*PE3*PE1
            Bh3_extra = 2*c['dn']*q3*PE1*PE2 \
                            + 2*c['dn']*q1*PE2*PE3 \
                            + 2*c['dn']*q2*PE3*PE1
            Bh3[sel] = Bh3_F2 + Bh3_extra
            Bh3[sel]*= 1./(1.+c['en']*q1)/(1.+c['en']*q2)/(1.+c['en']*q3)
            Bh3[np.logical_not(sel)] = np.nan
            Btot += Bh3
        
        return Btot



    def to_multipole(self, **kwargs):
        """Return a semi-analytic BiHalofit multipole evaluator."""
        return HalofitMultipole.from_halofit(self, **kwargs)


class HalofitMultipole(Halofit):
    """Semi-analytic BiHalofit multipole evaluator."""

    def __init__(
        self,
        k=None,
        pklin=None,
        z=None,
        lgr=None,
        cosmo=None,
        *,
        n_fftlog=128,
        k_fft_min=None,
        k_fft_max=None,
        fftlog_pad=4.0,
        bias_D=0.0,
        bias_H=0.0,
        n_kernel_phi=128,
        r_decimals=14,
        c_window_width=0.0,
    ):
        super().__init__(k=k, pklin=pklin, z=z, lgr=lgr, cosmo=cosmo)
        self._init_multipole_state(
            n_fftlog=n_fftlog,
            k_fft_min=k_fft_min,
            k_fft_max=k_fft_max,
            fftlog_pad=fftlog_pad,
            bias_D=bias_D,
            bias_H=bias_H,
            n_kernel_phi=n_kernel_phi,
            r_decimals=r_decimals,
            c_window_width=c_window_width,
        )

    @classmethod
    def from_halofit(cls, halofit, **kwargs):
        """Create a multipole evaluator sharing an existing Halofit state."""
        obj = cls.__new__(cls)
        obj.__dict__ = halofit.__dict__.copy()
        obj._init_multipole_state(**kwargs)
        return obj

    def set_lgr(self, z, lgr):
        super().set_lgr(z, lgr)
        self._clear_radial_fftlog_cache()

    def set_pklin(self, k, pklin):
        super().set_pklin(k, pklin)
        self._clear_radial_fftlog_cache()

    def set_cosmology(self, cosmo):
        super().set_cosmology(cosmo)
        self._clear_radial_fftlog_cache()

    def _clear_radial_fftlog_cache(self):
        cache = getattr(self, "_radial_fftlog_cache", None)
        if cache is not None:
            cache.clear()

    def _init_multipole_state(
        self,
        *,
        n_fftlog=128,
        k_fft_min=None,
        k_fft_max=None,
        fftlog_pad=4.0,
        bias_D=0.0,
        bias_H=0.0,
        n_kernel_phi=128,
        r_decimals=14,
        c_window_width=0.0,
    ):
        self.n_fftlog = int(n_fftlog)
        self.k_fft_min = k_fft_min
        self.k_fft_max = k_fft_max
        self.fftlog_pad = float(fftlog_pad)
        self.bias_D = float(bias_D)
        self.bias_H = float(bias_H)
        self.n_kernel_phi = int(n_kernel_phi)
        self.r_decimals = int(r_decimals)
        self.c_window_width = float(c_window_width)
        self._radial_fftlog_cache = {}

    def _radial_cache_key(self, z, k_fft, bias_D, bias_H, n_kernel_phi):
        return (
            float(z),
            int(k_fft.size),
            float(k_fft[0]),
            float(k_fft[-1]),
            float(bias_D),
            float(bias_H),
            int(n_kernel_phi),
            self.r_decimals,
            self.c_window_width,
        )

    def _get_radial_fftlog_pair(self, z, k_fft, bias_D, bias_H, n_kernel_phi):
        key = self._radial_cache_key(z, k_fft, bias_D, bias_H, n_kernel_phi)
        cached = self._radial_fftlog_cache.get(key)
        if cached is not None:
            return cached

        c = self.get_bihalofit_coeffs(np.asarray([z], dtype=float))[0]
        D_fft, _, H_fft = self._bihalofit_3h_radial_functions(k_fft, z, c=c)
        radial_D = self._make_radial_fftlog_expansion(
            k_fft, D_fft, bias=bias_D, n_kernel_phi=n_kernel_phi,
            r_decimals=self.r_decimals, c_window_width=self.c_window_width,
        )
        radial_H = self._make_radial_fftlog_expansion(
            k_fft, H_fft, bias=bias_H, n_kernel_phi=n_kernel_phi,
            r_decimals=self.r_decimals, c_window_width=self.c_window_width,
        )

        if radial_D is None or radial_H is None:
            coeff_D, nu_D = self._fftlog_power_law_coefficients(k_fft, D_fft, bias=bias_D)
            coeff_H, nu_H = self._fftlog_power_law_coefficients(k_fft, H_fft, bias=bias_H)
        else:
            coeff_D, nu_D = radial_D.c_m, radial_D.z_m
            coeff_H, nu_H = radial_H.c_m, radial_H.z_m

        cached = (radial_D, radial_H, coeff_D, nu_D, coeff_H, nu_H)
        self._radial_fftlog_cache[key] = cached
        return cached

    def _bihalofit_3h_radial_functions(self, k, z, c=None):
        """Return D(k), P_E(k), and H(k)=D(k)P_E(k) for BiHalofit 3h."""
        k = np.asarray(k, dtype=float)
        if c is None:
            c = self.get_bihalofit_coeffs(np.asarray([z], dtype=float))[0]

        q = np.maximum(k * c['r_sigma'], 1.0e-100)
        PL = self.get_interpolated_pklin(k, z)
        PE = ((1.0 + c['fn'] * q**2) / (1.0 + c['gn'] * q + c['hn'] * q**2)) * PL
        PE = PE + 1.0 / (c['mn'] * q**c['mun'] + c['nn'] * q**c['nun']) / (1.0 + (c['pn'] * q)**(-3.0))
        D = 1.0 / (1.0 + c['en'] * q)
        return D, PE, D * PE

    def _fftlog_power_law_coefficients(self, k, f, bias=0.0):
        """Represent ``f(k)`` as ``sum_m coeff_m k**nu_m``.

        This delegates to ``fastnc.hankel.wrapper.power_law_fftlog_coefficients``
        when available, so the 1D FFTLog convention is centralized in the
        Hankel/FFTLog interface.  A local implementation is kept as a
        standalone fallback.
        """
        if power_law_fftlog_coefficients is not None and PowerLawFFTLogConfig is not None:
            return power_law_fftlog_coefficients(
                k,
                f,
                PowerLawFFTLogConfig(bias=float(bias), c_window_width=0.0),
            )

        k = np.asarray(k, dtype=float)
        f = np.asarray(f, dtype=float)
        if k.ndim != 1 or f.ndim != 1 or k.size != f.size:
            raise ValueError("k and f must be one-dimensional arrays with the same length")
        if np.any(k <= 0.0):
            raise ValueError("FFTLog grid k must be positive")

        x = np.log(k)
        dx = x[1] - x[0]
        if not np.allclose(np.diff(x), dx, rtol=1.0e-6, atol=1.0e-12):
            raise ValueError("FFTLog grid must be uniformly spaced in ln k")

        g = f * np.exp(-bias * x)
        eta = 2.0 * np.pi * np.fft.fftfreq(k.size, d=dx)
        coeff = np.fft.fft(g) / k.size * np.exp(-1j * eta * x[0])
        nu = bias + 1j * eta
        return coeff, nu

    def _make_fftlog_grid(self, kmin, kmax, n_fftlog):
        """Return an endpoint-excluded logarithmic FFTLog grid."""
        if kmin <= 0.0 or kmax <= kmin:
            raise ValueError("Require 0 < kmin < kmax for the FFTLog grid")
        x0 = np.log(kmin)
        x1 = np.log(kmax)
        return np.exp(x0 + (x1 - x0) * np.arange(n_fftlog) / n_fftlog)


    def _make_radial_fftlog_expansion(
        self,
        k_fft,
        f_fft,
        bias=0.0,
        n_kernel_phi=128,
        r_decimals=14,
        c_window_width=0.0,
    ):
        """Return reusable radial FFTLog expansion on a fixed k-grid."""
        if RadialFFTLogExpansion is None:
            return None

        return RadialFFTLogExpansion(
            x_fft=k_fft,
            fx=f_fft,
            nu=float(bias),
            N_extrap_low=0,
            N_extrap_high=0,
            c_window_width=float(c_window_width),
            N_pad=0,
            n_r=max(512, int(k_fft.size) * 4),
            n_phi=int(n_kernel_phi),
            mode_block=16,
            r_decimals=r_decimals,
            bounds_error=False,
        )

    def _powerlaw_cosine_kernel_quad(self, L, nu, k1, k2, n_phi=128):
        """Compute K_L^nu(k1,k2) by Gauss-Legendre quadrature.

        This is the stable fallback definition of the analytic kernel

            K_L^nu = (2-delta_L0)/pi int_0^pi dphi k3(phi)^nu cos(L phi).
        """
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k1, k2 = np.broadcast_arrays(k1, k2)

        x, w = np.polynomial.legendre.leggauss(n_phi)
        phi = 0.5 * np.pi * (x + 1.0)
        weight = 0.5 * np.pi * w

        k3sq = k1[..., None]**2 + k2[..., None]**2 + 2.0 * k1[..., None] * k2[..., None] * np.cos(phi)
        k3sq = np.maximum(k3sq, 0.0)
        k3 = np.sqrt(k3sq)
        pref = (2.0 - (1.0 if int(L) == 0 else 0.0)) / np.pi
        vals = np.exp(nu * np.log(np.maximum(k3, np.finfo(float).tiny)))
        return pref * np.sum(weight * vals * np.cos(int(L) * phi), axis=-1)

    def _powerlaw_cosine_kernel_series(self, L, nu, k1, k2, max_terms=4096, tol=1.0e-12):
        """Compute K_L^nu using the convergent binomial series.

        This is efficient for k_</k_> sufficiently below unity.  For r close to
        unity the series converges slowly; use the quadrature kernel there.
        """
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k1, k2 = np.broadcast_arrays(k1, k2)
        kmax = np.maximum(k1, k2)
        kmin = np.minimum(k1, k2)
        r = kmin / kmax
        alpha = 0.5 * nu
        L = int(L)

        # Recursively build binomial coefficients up to max_terms+L.
        nmax = max_terms + L + 1
        b = np.empty(nmax + 1, dtype=complex)
        b[0] = 1.0 + 0.0j
        for n in range(nmax):
            b[n + 1] = b[n] * (alpha - n) / (n + 1.0)

        out = np.zeros_like(r, dtype=complex)
        active = np.ones(r.shape, dtype=bool)
        rpow = r**L
        r2 = r * r
        for q in range(max_terms):
            term = b[q + L] * b[q] * rpow
            out += term
            if q > 16:
                active = np.abs(term) > tol * np.maximum(1.0, np.abs(out))
                if not np.any(active):
                    break
            rpow = rpow * r2

        if L > 0:
            out *= 2.0
        return kmax**nu * out

    def _powerlaw_cosine_kernel(self, L, nu, k1, k2, n_phi=128, method="auto", r_series_max=0.85):
        """Compute K_L^nu(k1,k2) with series/quadrature switching."""
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k1, k2 = np.broadcast_arrays(k1, k2)
        r = np.minimum(k1, k2) / np.maximum(k1, k2)

        if method == "quad":
            return self._powerlaw_cosine_kernel_quad(L, nu, k1, k2, n_phi=n_phi)
        if method == "series":
            return self._powerlaw_cosine_kernel_series(L, nu, k1, k2)
        if method != "auto":
            raise ValueError("method must be 'auto', 'series', or 'quad'")

        out = np.empty(k1.shape, dtype=complex)
        use_series = r < r_series_max
        if np.any(use_series):
            out[use_series] = self._powerlaw_cosine_kernel_series(
                L, nu, k1[use_series], k2[use_series]
            )
        if np.any(~use_series):
            out[~use_series] = self._powerlaw_cosine_kernel_quad(
                L, nu, k1[~use_series], k2[~use_series], n_phi=n_phi
            )
        return out

    def _kernel_sum(self, L, coeff, nu, shift, k1, k2, n_phi=128, method="auto"):
        """Return sum_m coeff_m K_L^{nu_m+shift}(k1,k2)."""
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        out = np.zeros(np.broadcast_shapes(k1.shape, k2.shape), dtype=complex)
        for cm, num in zip(coeff, nu):
            if np.abs(cm) == 0.0:
                continue
            out += cm * self._powerlaw_cosine_kernel(
                L, num + shift, k1, k2, n_phi=n_phi, method=method
            )
        return out

    def _bihalofit_T12_coefficients(self, k1, k2):
        """Return C0,C2,C4 such that F2(k1,k2)=C0+C2*k3^2+C4*k3^4."""
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        D = 2.0 * k1 * k2
        S = k1**2 + k2**2
        R = 0.5 * (k1 / k2 + k2 / k1)
        C0 = 5.0 / 7.0 - R * S / D + (2.0 / 7.0) * S**2 / D**2
        C2 = R / D - (4.0 / 7.0) * S / D**2
        C4 = (2.0 / 7.0) / D**2
        return C0, C2, C4

    def _bihalofit_F2_variable_side_coefficients(self, kfixed, kopp):
        """Return Dm2,D0,D2 for F2(kfixed,k3) with opposite side kopp.

        The result satisfies

            F2(kfixed,k3,kopp) = Dm2*k3**(-2) + D0 + D2*k3**2.
        """
        kfixed = np.asarray(kfixed, dtype=float)
        kopp = np.asarray(kopp, dtype=float)
        U = (kopp**2 - kfixed**2) / (2.0 * kfixed)
        V = -1.0 / (2.0 * kfixed)
        Dm2 = 0.5 * kfixed * U + (2.0 / 7.0) * U**2
        D0 = 5.0 / 7.0 + 0.5 * (kfixed * V + U / kfixed) + (4.0 / 7.0) * U * V
        D2 = 0.5 * V / kfixed + (2.0 / 7.0) * V**2
        return Dm2, D0, D2

    def _bihalofit_Tcyc_coefficients(self, k1, k2, PE1, PE2, dn_over_kNL):
        """Return E_{-2}, E_0, E_2 for the pair-combined cyclic term.

        The cyclic pair is written as

            T_cyc(phi) = 2 D1 D2 H3 [E_-2 k3^-2 + E_0 + E_2 k3^2],

        where D(k)=1/(1+e_n q), H(k)=D(k)P_E(k), and k3 is the
        variable side associated with the opening angle between k1 and k2.
        """
        Dm2_23, D0_23, D2_23 = self._bihalofit_F2_variable_side_coefficients(k2, k1)
        Dm2_31, D0_31, D2_31 = self._bihalofit_F2_variable_side_coefficients(k1, k2)
        Em2 = PE2 * Dm2_23 + PE1 * Dm2_31
        E0 = PE2 * D0_23 + PE1 * D0_31 + dn_over_kNL * (k1 * PE2 + k2 * PE1)
        E2 = PE2 * D2_23 + PE1 * D2_31
        return Em2, E0, E2

    def _bihalofit_3h_cyclic_multipole_quad(self, k1, k2, L, z, c, n_phi=256):
        """Reference quadrature for the pair-combined cyclic 3h term."""
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k1, k2 = np.broadcast_arrays(k1, k2)
        D1, PE1, _ = self._bihalofit_3h_radial_functions(k1, z, c=c)
        D2, PE2, _ = self._bihalofit_3h_radial_functions(k2, z, c=c)
        Em2, E0, E2 = self._bihalofit_Tcyc_coefficients(
            k1, k2, PE1, PE2, c['dn'] * c['r_sigma']
        )

        x, w = np.polynomial.legendre.leggauss(n_phi)
        phi = 0.5 * np.pi * (x + 1.0)
        weight = 0.5 * np.pi * w
        cosphi = np.cos(phi)
        k3 = np.sqrt(np.maximum(k1[..., None]**2 + k2[..., None]**2 + 2.0 * k1[..., None] * k2[..., None] * cosphi, 0.0))
        _, _, H3 = self._bihalofit_3h_radial_functions(k3, z, c=c)
        tiny = np.finfo(float).tiny
        poly = Em2[..., None] / np.maximum(k3, tiny)**2 + E0[..., None] + E2[..., None] * k3**2
        Tcyc = 2.0 * D1[..., None] * D2[..., None] * H3 * poly
        pref = (2.0 - (1.0 if int(L) == 0 else 0.0)) / np.pi
        return pref * np.sum(weight * Tcyc * np.cos(int(L) * phi), axis=-1)

    def get_bihalofit_3h_multipole_quad(self, k1, k2, L, z, n_phi=256):
        """Direct Gauss-Legendre reference multipole of the BiHalofit 3h term."""
        self.update()
        z = float(z)
        c = self.get_bihalofit_coeffs(np.asarray([z], dtype=float))[0]
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k1, k2 = np.broadcast_arrays(k1, k2)

        x, w = np.polynomial.legendre.leggauss(n_phi)
        phi = 0.5 * np.pi * (x + 1.0)
        weight = 0.5 * np.pi * w
        k3 = np.sqrt(np.maximum(k1[..., None]**2 + k2[..., None]**2 + 2.0 * k1[..., None] * k2[..., None] * np.cos(phi), 0.0))
        B3 = self.get_bihalofit(
            np.broadcast_to(k1[..., None], k3.shape).ravel(),
            np.broadcast_to(k2[..., None], k3.shape).ravel(),
            k3.ravel(),
            np.full(k3.size, z),
            which=['Bh3'],
            squeezed_safe=False,
        ).reshape(k3.shape)
        pref = (2.0 - (1.0 if int(L) == 0 else 0.0)) / np.pi
        return pref * np.sum(weight * B3 * np.cos(int(L) * phi), axis=-1)

    def get_bihalofit_3h_multipole_semianalytic(
        self,
        k1,
        k2,
        L,
        z,
        n_fftlog=None,
        k_fft_min=None,
        k_fft_max=None,
        fftlog_pad=None,
        bias_D=None,
        bias_H=None,
        kernel_method="auto",
        n_kernel_phi=None,
        cyclic_r_quad=0.97,
        cyclic_quad_n_phi=256,
        return_parts=False,
    ):
        """Semi-analytic cosine multipole of the BiHalofit 3h term.

        This computes

            B_{3h,L}(k1,k2) = (2-delta_L0)/pi int_0^pi dphi B_3h(k1,k2,k3(phi)) cos(L phi)

        using FFTLog power-law decompositions for the radial functions
        D(k)=1/(1+e_n q) and H(k)=D(k)P_E(k).  The term in which
        F2(k1,k2) depends only on the opening angle is evaluated by shifted
        kernels K_L^{nu+s}.  The cyclic pair is evaluated in pair-combined
        form.  For k_</k_> > cyclic_r_quad, the cyclic pair falls back to a
        direct pair-combined quadrature to avoid the squeezed-limit
        cancellation problem associated with isolated K_L^{nu-2} kernels.
        """
        self.update()
        z = float(z)
        L = int(L)
        if L < 0:
            raise ValueError("L must be non-negative")

        if n_fftlog is None:
            n_fftlog = self.n_fftlog
        if k_fft_min is None:
            k_fft_min = self.k_fft_min
        if k_fft_max is None:
            k_fft_max = self.k_fft_max
        if fftlog_pad is None:
            fftlog_pad = self.fftlog_pad
        if bias_D is None:
            bias_D = self.bias_D
        if bias_H is None:
            bias_H = self.bias_H
        if n_kernel_phi is None:
            n_kernel_phi = self.n_kernel_phi

        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k1, k2 = np.broadcast_arrays(k1, k2)
        if np.any(k1 <= 0.0) or np.any(k2 <= 0.0):
            raise ValueError("k1 and k2 must be positive")

        c = self.get_bihalofit_coeffs(np.asarray([z], dtype=float))[0]
        D1, PE1, _ = self._bihalofit_3h_radial_functions(k1, z, c=c)
        D2, PE2, _ = self._bihalofit_3h_radial_functions(k2, z, c=c)

        k3_min = np.abs(k1 - k2)
        k3_max = k1 + k2
        positive = k3_min[k3_min > 0.0]
        if k_fft_min is None:
            if positive.size:
                k_fft_min = positive.min() / fftlog_pad
            else:
                k_fft_min = np.minimum(k1, k2).min() * 1.0e-5
            k_fft_min = max(k_fft_min, min(self.k.min(), np.minimum(k1, k2).min()) * 1.0e-3)
        if k_fft_max is None:
            k_fft_max = k3_max.max() * fftlog_pad
        k_fft = self._make_fftlog_grid(k_fft_min, k_fft_max, int(n_fftlog))

        radial_D, radial_H, coeff_D, nu_D, coeff_H, nu_H = self._get_radial_fftlog_pair(
            z=z,
            k_fft=k_fft,
            bias_D=bias_D,
            bias_H=bias_H,
            n_kernel_phi=n_kernel_phi,
        )


        # T12: variable dependence comes from D(k3) and k3 D(k3).
        C0, C2, C4 = self._bihalofit_T12_coefficients(k1, k2)
        if radial_D is None:
            KD0 = self._kernel_sum(L, coeff_D, nu_D, 0.0, k1, k2, n_phi=n_kernel_phi, method=kernel_method)
            KD1 = self._kernel_sum(L, coeff_D, nu_D, 1.0, k1, k2, n_phi=n_kernel_phi, method=kernel_method)
            KD2 = self._kernel_sum(L, coeff_D, nu_D, 2.0, k1, k2, n_phi=n_kernel_phi, method=kernel_method)
            KD4 = self._kernel_sum(L, coeff_D, nu_D, 4.0, k1, k2, n_phi=n_kernel_phi, method=kernel_method)
        else:
            KD = radial_D.kernel_sum_many_shifts(
                L=L,
                shifts=(0, 1, 2, 4),
                k1=k1,
                k2=k2,
                use_unique_r=True,
                r_decimals=14,
            )
            KD0, KD1, KD2, KD4 = KD[..., 0], KD[..., 1], KD[..., 2], KD[..., 3]
        T12 = 2.0 * D1 * D2 * PE1 * PE2 * (
            C0 * KD0 + C2 * KD2 + C4 * KD4 + c['dn'] * c['r_sigma'] * KD1
        )

        # Tcyc: pair-combined cyclic contribution.
        Em2, E0, E2 = self._bihalofit_Tcyc_coefficients(k1, k2, PE1, PE2, c['dn'] * c['r_sigma'])
        if radial_H is None:
            KHm2 = self._kernel_sum(L, coeff_H, nu_H, -2.0, k1, k2, n_phi=n_kernel_phi, method=kernel_method)
            KH0 = self._kernel_sum(L, coeff_H, nu_H, 0.0, k1, k2, n_phi=n_kernel_phi, method=kernel_method)
            KH2 = self._kernel_sum(L, coeff_H, nu_H, 2.0, k1, k2, n_phi=n_kernel_phi, method=kernel_method)
        else:
            KH = radial_H.kernel_sum_many_shifts(
                L=L,
                shifts=(-2, 0, 2),
                k1=k1,
                k2=k2,
                use_unique_r=True,
                r_decimals=7,
            )
            KHm2, KH0, KH2 = KH[..., 0], KH[..., 1], KH[..., 2]
        Tcyc = 2.0 * D1 * D2 * (Em2 * KHm2 + E0 * KH0 + E2 * KH2)

        r = np.minimum(k1, k2) / np.maximum(k1, k2)
        fallback = r > cyclic_r_quad
        if np.any(fallback):
            Tcyc = np.array(Tcyc, dtype=complex, copy=True)
            Tcyc[fallback] = self._bihalofit_3h_cyclic_multipole_quad(
                k1[fallback], k2[fallback], L, z, c, n_phi=cyclic_quad_n_phi
            )

        total = T12 + Tcyc
        total = np.real_if_close(total, tol=1000)
        if return_parts:
            return {
                'total': total,
                'T12': np.real_if_close(T12, tol=1000),
                'Tcyc': np.real_if_close(Tcyc, tol=1000),
                'k_fft': k_fft,
                'coeff_D': coeff_D,
                'nu_D': nu_D,
                'coeff_H': coeff_H,
                'nu_H': nu_H,
            }
        return total


def get_Rb_bihalofit(k1, k2, k3, z):
    """
    Returns the baryon ratio on bispectrum.

    Parameters:
        k1           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
        k2           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
        k3           (np.ndarray): array of comoving Fourier modes in h/Mpc unit
        z            (np.ndarray): array of redshifts
    """
    # ratio of modes
    kmin, kmid, kmax = np.sort([k1,k2,k3], axis=0)
    
    # Baryon effective redshift
    sel = z<5
    
    # coefficients
    a = 1/(1+z[sel])
    A0  = np.zeros(a.shape)
    A0[a>0.5] = 0.068*(a[a>0.5]-0.5)**0.47
    mu0 = 0.018*a + 0.837*a**2
    si0 = 0.881*mu0
    al0 = 2.346
    A1  = np.zeros(a.shape)
    A1[a>0.2] = 1.052*(a[a>0.2]-0.2)**1.41
    mu1 = np.abs(0.172+3.048*a-0.675*a**2)
    si1 = (0.494-0.039*a)*mu1
    kst = 29.90 - 38.73*a+24.30*a**2
    al2 = 2.25
    be2 = 0.563/((a/0.06)**0.02+1) / al2

    # Assign baryon ratios
    Rb = np.ones(z.shape)
    for k in [k1[sel], k2[sel], k3[sel]]:
        x = np.log10(k)
        Rb[sel]*= A0*np.exp(-np.abs((x-mu0)/si0)**al0) \
                    - A1*np.exp(-((x-mu1)/si1)**2) \
                    + ((k/kst)**al2+1)**be2
            
    return Rb
    
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
