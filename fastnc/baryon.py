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
from scipy.interpolate import NearestNDInterpolator

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
            return 1.0 + self.fb * (s-1.0)


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


class BaryonBACCOemu(BaryonModelBase):
    """
    Baryonic boost factor from the BACCOemu matter bispectrum emulator.

    The boost R(k1,k2,k3,z) returned by baccoemu.Matter_bispectrum()
    .get_baryonic_boost() is applied as a multiplicative correction to
    the gravity-only bispectrum:

        B_total(k1,k2,k3,z) = B_gravity(k1,k2,k3,z) * R(k1,k2,k3,z)

    Parameters
    ----------
    bispec : BispectrumHalofit
        The bispectrum instance with the desired cosmological parameters
    M_c : float, optional
        Log10 of the characteristic halo mass for gas expulsion [Msun/h].
        (default: 14.0)
    eta : float, optional
        Extent of gas ejection beyond the halo boundary. (default: -0.3)
    beta : float, optional
        Slope of the AGN feedback efficiency. (default: -0.22)
    M1_z0_cen : float, optional
        Log10 of the characteristic stellar mass of central galaxies at
        z=0 [Msun/h]. (default: 10.5)
    theta_inn : float, optional
        Inner slope of the gas density profile. (default: -0.86)

    Example
    -------
    >>> from fastnc.baryons import BaryonBACCOemu
    >>> bispec = BispectrumHalofit()
    >>> bispec.set_cosmology(cosmo)
    >>> baryon_model = BaryonBACCOemu(bispec, M_c=14.0, eta=-0.3)
    >>> bispec.set_baryon_model(baryon_model)
    """

    _default_baryon_params = dict(
        M_c=14.0,
        eta=-0.3,
        beta=-0.22,
        M1_z0_cen=10.5,
        theta_inn=-0.86,
    )

    def __init__(self, bispec,
                 M_c=None, eta=None, beta=None, M1_z0_cen=None, theta_inn=None):
        try:
            import baccoemu
        except ImportError:
            raise ImportError(
                "baccoemu is required for BaryonBACCOemu. "
                "Please install it before using this class."
            )
        self._emulator = baccoemu.Matter_bispectrum()
        self._bispec = bispec

        # Baryonic feedback parameters, falling back to defaults
        self._baryon_params = dict(self._default_baryon_params)
        if M_c is not None: self._baryon_params['M_c'] = M_c
        if eta is not None: self._baryon_params['eta'] = eta
        if beta is not None: self._baryon_params['beta'] = beta
        if M1_z0_cen is not None: self._baryon_params['M1_z0_cen'] = M1_z0_cen
        if theta_inn is not None: self._baryon_params['theta_inn'] = theta_inn

    def set_baryon_params(self, **kwargs):
        """
        Update baryonic feedback parameters.

        Accepted keys: M_c, eta, beta, M1_z0_cen, theta_inn.
        """
        for key in kwargs:
            if key not in self._default_baryon_params:
                raise ValueError(
                    f"Unknown baryonic parameter '{key}'. "
                    f"Accepted: {list(self._default_baryon_params.keys())}"
                )
        self._baryon_params.update(kwargs)

    def _get_cosmo_params(self):
        """
        Extract cosmological parameters from the bispectrum's astropy cosmo object.
        """
        cosmo = self._bispec.cosmo
        if cosmo is None:
            raise RuntimeError(
                "No cosmology found. Call BispectrumHalofit.set_cosmology() first."
            )
        if cosmo.Ob0 is None:
            raise ValueError(
                "The astropy cosmo object has Ob0=None. "
                "Pass Ob0 (baryon density) when constructing the wCDM object."
            )
        return dict(
            omega_cold   = cosmo.Om0,
            sigma8_cold  = cosmo.meta['sigma8'],
            omega_baryon = cosmo.Ob0,
        )

    def __call__(self, k1, k2, k3, z):
        """
        Return the baryonic boost R(k1, k2, k3, z).

        Parameters
        ----------
        k1, k2, k3 : array_like
            Must have the same shape.
        z : array_like
            Redshift, same shape as k1/k2/k3.

        Returns
        -------
        R : ndarray
            Baryonic boost factor, same shape as the inputs.

        Notes
        -----
        baccoemu expects 1-D arrays and a scalar expfactor per call.
        Since fastnc passes 2-D (n_ell, n_z) arrays during the
        line-of-sight integration, we loop over unique redshift slices
        so that each baccoemu call receives a flat 1-D array and a
        single scalar expfactor.
        """
        k1 = np.atleast_2d(k1)
        k2 = np.atleast_2d(k2)
        k3 = np.atleast_2d(k3)
        z = np.atleast_2d(z)

        num_triangles, num_z = k1.shape
        boost_out = np.ones((num_triangles, num_z))

        cosmo_params = {
            'omega_cold': 0.3153,
            'omega_baryon': 0.0493,
            'hubble': 0.6736,
            'ns': 0.9649,
            'sigma8_cold': 0.8111,
            'w0': -1.0,
            'wa': 0.0,
            'neutrino_mass': 0.0,
        }

        for i in range(num_z):

            k1_slice = k1[:, i]  # shape: (num_triangles,)
            k2_slice = k2[:, i]
            k3_slice = k3[:, i]

            sides = np.stack([k1_slice, k2_slice, k3_slice], axis=1)  # (N, 3)

            sides_lowres = sides.astype(np.float32)
            sides_sorted = np.sort(sides_lowres, axis=1)  # (N, 3), ascending

            ks_small = sides_sorted[:, 0]  # smallest k
            ks_mid = sides_sorted[:, 1]  # middle k
            ks_large = sides_sorted[:, 2]  # largest k

            invalid_mask = ks_large >= (ks_small + ks_mid)  # True where INVALID
            valid_mask = ~invalid_mask  # True where VALID

            k1_valid = ks_small[valid_mask]
            k2_valid = ks_mid[valid_mask]
            k3_valid = ks_large[valid_mask]

            current_z = z[0, i]
            expfactor = 1.0 / (1.0 + current_z)

            params = {**cosmo_params, **self._baryon_params, 'expfactor': expfactor}

            k_emu, b_emu, extrap_flag = self._emulator.get_baryonic_boost(
                k1=k1_valid,
                k2=k2_valid,
                k3=k3_valid,
                **params
            )

            #good_indices = np.where(~extrap_flag)[0]
            #print(b_emu[good_indices])

            #if b_emu is not None and len(b_emu) > 0:
            #    rows = np.where(valid_mask)[0][good_indices]
            #    boost_out[rows, i] = b_emu[good_indices]

            #lim = 1.0
            lim = 2.0
            spurious = b_emu >= lim * np.ones_like(b_emu)
            good_indices = np.where(~spurious)[0]

            if b_emu is not None and len(b_emu) > 0:
                rows = np.where(valid_mask)[0][good_indices]
                boost_out[rows, i] = b_emu[good_indices]

        return boost_out